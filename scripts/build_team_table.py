#!/usr/bin/env python3
"""
build_team_table.py

Pre-compute a unified team lookup table merging team identity, Stan model
indices, bullpen quality, and wRC+ offense data into one CSV.

Output: data/processed/team_table.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────
ATT_STD_EST = 0.109   # from posterior att_run_1 team mean std
BULLPEN_SCALE = 0.1   # bullpen_adj = -bullpen_depth_score * BULLPEN_SCALE


def _parse_season_from_path(path: Path) -> int | None:
    parent = path.parent.name
    if parent.isdigit() and len(parent) == 4:
        return int(parent)
    return None


def _collect_season_files(root: Path, basename: str, explicit_path: Path | None = None) -> list[tuple[int | None, Path]]:
    seen: set[Path] = set()
    out: list[tuple[int | None, Path]] = []
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    candidates.append(root / basename)
    for p in sorted(root.glob(f"*/{basename}")):
        candidates.append(p)
    for p in candidates:
        if not p.exists():
            continue
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append((_parse_season_from_path(p), p))
    return out


def load_canonical_teams(path: Path) -> pd.DataFrame:
    """Load canonical team registry as the base table."""
    df = pd.read_csv(path, dtype=str)
    # Keep only the columns we need from the registry
    keep = ["canonical_id", "team_name", "conference", "academic_year"]
    available = [c for c in keep if c in df.columns]
    df = df[available].copy()
    df = df.rename(columns={"academic_year": "season"})
    return df


def load_team_index(path: Path) -> pd.DataFrame:
    """Load Stan model team index (canonical_id → team_idx)."""
    df = pd.read_csv(path, dtype=str)
    df["team_idx"] = pd.to_numeric(df["team_idx"], errors="coerce").fillna(0).astype(int)
    return df[["canonical_id", "team_idx"]]


def load_bullpen_quality(path: Path) -> pd.DataFrame:
    """Load bullpen quality scores and compute bullpen_adj."""
    df = pd.read_csv(path, dtype=str)
    # Rename join key to canonical_id if needed
    if "team_canonical_id" in df.columns and "canonical_id" not in df.columns:
        df = df.rename(columns={"team_canonical_id": "canonical_id"})

    df["bullpen_depth_score"] = pd.to_numeric(df["bullpen_depth_score"], errors="coerce")
    df["bullpen_quality_z"] = df["bullpen_depth_score"]
    df["bullpen_adj"] = -df["bullpen_depth_score"] * BULLPEN_SCALE

    cols = ["canonical_id", "bullpen_quality_z", "bullpen_adj"]
    if "n_relievers" in df.columns:
        cols.append("n_relievers")
    result = df[cols].copy()
    # Deduplicate: keep the row with most recent/best data per team
    # (multi-season bullpen data creates duplicates)
    result = result.sort_values("bullpen_quality_z", ascending=False, na_position="last")
    result = result.drop_duplicates("canonical_id", keep="first")
    return result


def build_d1b_crosswalk(xwalk_path: Path) -> dict[str, str]:
    """Build d1baseball_name → canonical_id mapping with apostrophe normalization."""
    xw_df = pd.read_csv(xwalk_path, dtype=str)
    d1b_to_cid: dict[str, str] = {}
    for _, r in xw_df.iterrows():
        d1b_name = str(r.get("d1baseball_name", "")).strip().lower()
        cid = str(r.get("canonical_id", "")).strip()
        if not d1b_name or not cid:
            continue
        d1b_to_cid[d1b_name] = cid
        # Normalize apostrophes: Unicode curly → ASCII and vice versa
        norm_ascii = d1b_name.replace("\u2019", "'")
        if norm_ascii != d1b_name:
            d1b_to_cid[norm_ascii] = cid
        norm_unicode = d1b_name.replace("'", "\u2019")
        if norm_unicode != d1b_name:
            d1b_to_cid[norm_unicode] = cid
    return d1b_to_cid


def load_wrc_plus(bat_path: Path, xwalk_path: Path, d1b_root: Path, as_of_season: int) -> pd.DataFrame:
    """
    Load D1B batting_advanced.tsv and compute team mean wRC+.

    Returns a DataFrame with columns:
        canonical_id, wrc_plus, wrc_offense_adj
    """
    d1b_to_cid = build_d1b_crosswalk(xwalk_path)

    team_wrc: dict[str, list[float]] = {}
    season_rows: list[dict[str, object]] = []
    bat_files = _collect_season_files(d1b_root, "batting_advanced.tsv", bat_path)
    for season_hint, fpath in bat_files:
        season_val = int(season_hint) if season_hint is not None else int(as_of_season)
        bat_df = pd.read_csv(fpath, sep="\t", dtype=str)
        for _, r in bat_df.iterrows():
            team = str(r.get("Team", "")).strip()
            team_key = team.lower().replace("\u2019", "'")
            cid = d1b_to_cid.get(team_key) or d1b_to_cid.get(team.lower(), "")
            if not cid:
                continue
            try:
                wrc_val = float(r["wRC+"])
                season_rows.append({"canonical_id": cid, "season": season_val, "wrc_plus": wrc_val})
            except (ValueError, TypeError, KeyError):
                pass

    if season_rows:
        s_df = pd.DataFrame(season_rows)
        # Recency-weighted aggregation across seasons up to as_of_season.
        for cid, grp in s_df.groupby("canonical_id"):
            grp = grp[grp["season"] <= as_of_season]
            if grp.empty:
                continue
            weights = (0.65 ** (as_of_season - grp["season"].astype(int))).clip(lower=0.15)
            val = float(np.average(grp["wrc_plus"].astype(float), weights=weights))
            team_wrc.setdefault(cid, []).append(val)

    if not team_wrc:
        print("  WARNING: No wRC+ data loaded — check batting_advanced.tsv and crosswalk",
              file=sys.stderr)
        return pd.DataFrame(columns=["canonical_id", "wrc_plus", "wrc_offense_adj"])

    team_means = {cid: float(np.mean(vals)) for cid, vals in team_wrc.items()}
    all_means = np.array(list(team_means.values()))
    wrc_std = float(np.std(all_means)) if len(all_means) > 1 else 23.8

    print(f"  wRC+ loaded: {len(team_means)} teams, "
          f"μ={np.mean(all_means):.1f} σ={wrc_std:.1f}", file=sys.stderr)

    rows = []
    for cid, wrc_mean in team_means.items():
        wrc_offense_adj = (wrc_mean - 100.0) / max(wrc_std, 1.0) * ATT_STD_EST
        rows.append({
            "canonical_id": cid,
            "wrc_plus": round(wrc_mean, 2),
            "wrc_offense_adj": round(wrc_offense_adj, 6),
        })

    return pd.DataFrame(rows)


def load_batting_fb_factor(bb_path: Path, xwalk_path: Path, d1b_root: Path, as_of_season: int) -> pd.DataFrame:
    """
    Load D1B batting_batted_ball.tsv and compute team fly-ball factor.

    batting_fb_factor = team_mean_FB% / league_mean_FB%
    Values > 1.0 = team hits more fly balls than average (more wind-sensitive).
    Values < 1.0 = ground-ball-heavy team (less wind-sensitive).

    Returns a DataFrame with columns:
        canonical_id, batting_fb_pct, batting_fb_factor
    """
    d1b_to_cid = build_d1b_crosswalk(xwalk_path)

    team_fb: dict[str, list[float]] = {}
    bb_files = _collect_season_files(d1b_root, "batting_batted_ball.tsv", bb_path)
    for season_hint, fpath in bb_files:
        season_val = int(season_hint) if season_hint is not None else int(as_of_season)
        if season_val > as_of_season:
            continue
        bb_df = pd.read_csv(fpath, sep="\t", dtype=str)
        for _, r in bb_df.iterrows():
            team = str(r.get("Team", "")).strip()
            team_key = team.lower().replace("\u2019", "'")
            cid = d1b_to_cid.get(team_key) or d1b_to_cid.get(team.lower(), "")
            if not cid:
                continue
            try:
                fb_str = str(r.get("FB%", "")).strip().rstrip("%")
                fb_val = float(fb_str)
                team_fb.setdefault(cid, []).append(fb_val)
            except (ValueError, TypeError):
                pass

    if not team_fb:
        print("  WARNING: No batting FB% data loaded", file=sys.stderr)
        return pd.DataFrame(columns=["canonical_id", "batting_fb_pct", "batting_fb_factor"])

    team_means = {cid: float(np.mean(vals)) for cid, vals in team_fb.items()}
    all_means = np.array(list(team_means.values()))
    league_avg = float(np.mean(all_means))

    print(f"  Batting FB% loaded: {len(team_means)} teams, "
          f"league avg={league_avg:.1f}%", file=sys.stderr)

    rows = []
    for cid, fb_mean in team_means.items():
        factor = fb_mean / league_avg if league_avg > 0 else 1.0
        rows.append({
            "canonical_id": cid,
            "batting_fb_pct": round(fb_mean, 1),
            "batting_fb_factor": round(factor, 4),
        })

    return pd.DataFrame(rows)


def build_team_table(
    registry_path: Path,
    team_index_path: Path,
    bullpen_path: Path,
    bat_path: Path,
    xwalk_path: Path,
    d1b_root: Path,
    as_of_season: int | None = None,
) -> pd.DataFrame:
    """Merge all sources into a single team table."""

    # 1. Base: canonical team registry
    print("Loading canonical team registry...", file=sys.stderr)
    base = load_canonical_teams(registry_path)
    print(f"  {len(base)} teams in registry", file=sys.stderr)
    if as_of_season is None:
        s = pd.to_numeric(base.get("season"), errors="coerce").dropna()
        as_of_season = int(s.max()) if not s.empty else 2026

    # 2. Stan model team indices
    print("Loading Stan model team indices...", file=sys.stderr)
    if team_index_path.exists():
        idx_df = load_team_index(team_index_path)
        print(f"  {len(idx_df)} teams with Stan indices", file=sys.stderr)
        base = base.merge(idx_df, on="canonical_id", how="left")
        base["team_idx"] = base["team_idx"].fillna(0).astype(int)
    else:
        print(f"  WARNING: {team_index_path} not found — team_idx set to 0", file=sys.stderr)
        base["team_idx"] = 0

    # 3. Bullpen quality
    print("Loading bullpen quality...", file=sys.stderr)
    if bullpen_path.exists():
        bq_df = load_bullpen_quality(bullpen_path)
        n_with_bullpen = bq_df["bullpen_quality_z"].notna().sum()
        print(f"  {n_with_bullpen} teams with bullpen data", file=sys.stderr)
        base = base.merge(bq_df, on="canonical_id", how="left")
    else:
        print(f"  WARNING: {bullpen_path} not found — bullpen fields will be empty",
              file=sys.stderr)
        base["bullpen_quality_z"] = np.nan
        base["bullpen_adj"] = np.nan

    # 4. D1B wRC+ offense adjustment
    print("Loading D1B wRC+ batting stats...", file=sys.stderr)
    if bat_path.exists() and xwalk_path.exists():
        wrc_df = load_wrc_plus(bat_path, xwalk_path, d1b_root=d1b_root, as_of_season=as_of_season)
        print(f"  {len(wrc_df)} teams with wRC+ data", file=sys.stderr)
        base = base.merge(wrc_df, on="canonical_id", how="left")
    else:
        missing = []
        if not bat_path.exists():
            missing.append(str(bat_path))
        if not xwalk_path.exists():
            missing.append(str(xwalk_path))
        print(f"  WARNING: Missing files {missing} — wRC+ fields will be empty", file=sys.stderr)
        base["wrc_plus"] = np.nan
        base["wrc_offense_adj"] = np.nan

    # Fill missing wrc_offense_adj with 0.0 (no adjustment for unknown teams)
    base["wrc_offense_adj"] = base["wrc_offense_adj"].fillna(0.0)

    # 5. Batting FB% factor (fly ball tendency for wind model)
    print("Loading D1B batting batted-ball stats (FB%)...", file=sys.stderr)
    bb_path = d1b_root / "batting_batted_ball.tsv"
    if bb_path.exists() and xwalk_path.exists():
        fb_df = load_batting_fb_factor(bb_path, xwalk_path, d1b_root=d1b_root,
                                        as_of_season=as_of_season)
        print(f"  {len(fb_df)} teams with batting FB% data", file=sys.stderr)
        base = base.merge(fb_df, on="canonical_id", how="left")
    else:
        print(f"  WARNING: Missing batted ball data — batting_fb_factor set to 1.0",
              file=sys.stderr)
    base["batting_fb_pct"] = base.get("batting_fb_pct", pd.Series(dtype=float)).fillna(0.0)
    base["batting_fb_factor"] = base.get("batting_fb_factor", pd.Series(dtype=float)).fillna(1.0)

    # 6. Conference strength from cross-conference win rates
    print("Computing conference strength from cross-conf records...", file=sys.stderr)
    games_path = Path("data/processed/games.csv")
    if games_path.exists() and "conference" in base.columns:
        team_conf_map = dict(zip(base["canonical_id"], base["conference"]))
        try:
            games = pd.read_csv(games_path, dtype=str)
            games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
            games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")
            games = games.dropna(subset=["home_score", "away_score"])
            games["home_won"] = games["home_score"] > games["away_score"]
            games["home_conf"] = games["home_canonical_id"].map(team_conf_map)
            games["away_conf"] = games["away_canonical_id"].map(team_conf_map)

            # Cross-conference only
            cross = games[games["home_conf"] != games["away_conf"]]
            conf_wins: dict[str, list] = {}
            for _, g in cross.iterrows():
                hc, ac = g["home_conf"], g["away_conf"]
                if pd.notna(hc):
                    conf_wins.setdefault(hc, []).append(1 if g["home_won"] else 0)
                if pd.notna(ac):
                    conf_wins.setdefault(ac, []).append(0 if g["home_won"] else 1)

            # Convert to log-rate adjustment
            conf_strength = {}
            for conf, outcomes in conf_wins.items():
                if len(outcomes) >= 20:
                    win_rate = sum(outcomes) / len(outcomes)
                    z = (win_rate - 0.5) / 0.15  # rough std of conference win rates
                    conf_strength[conf] = float(np.clip(z * ATT_STD_EST * 0.5, -0.15, 0.15))

            base["conf_strength_adj"] = base["conference"].map(conf_strength).fillna(0.0)
            n_with_conf = (base["conf_strength_adj"] != 0.0).sum()
            print(f"  {n_with_conf} teams have conference strength adjustment", file=sys.stderr)
        except Exception as e:
            print(f"  WARNING: Conference strength computation failed: {e}", file=sys.stderr)
            base["conf_strength_adj"] = 0.0
    else:
        base["conf_strength_adj"] = 0.0

    # 7. Batter handedness composition (for bilateral platoon model)
    hand_csv = Path("data/processed/team_batter_handedness.csv")
    if hand_csv.exists():
        hand_df = pd.read_csv(hand_csv, dtype=str)
        hand_df["pct_rhb"] = pd.to_numeric(hand_df["pct_rhb"], errors="coerce")
        hand_df["pct_lhb"] = pd.to_numeric(hand_df["pct_lhb"], errors="coerce")
        hand_df["effective_rhb_frac"] = pd.to_numeric(hand_df["effective_rhb_frac"], errors="coerce")
        hand_keep = hand_df[["canonical_id", "pct_rhb", "pct_lhb", "effective_rhb_frac"]].copy()
        base = base.merge(hand_keep, on="canonical_id", how="left")
        # Default to league average for teams not in sidearm data
        base["pct_rhb"] = base["pct_rhb"].fillna(0.682)
        base["pct_lhb"] = base["pct_lhb"].fillna(0.290)
        base["effective_rhb_frac"] = base["effective_rhb_frac"].fillna(0.696)
        n_hand = (base["pct_rhb"] != 0.682).sum()
        print(f"  Batter handedness: {n_hand} teams with team-specific data", file=sys.stderr)
    else:
        print(f"  WARNING: {hand_csv} not found — using league avg RHB for all teams", file=sys.stderr)
        base["pct_rhb"] = 0.682
        base["pct_lhb"] = 0.290
        base["effective_rhb_frac"] = 0.696

    # 8. n_games — not available yet, set to None
    base["n_games"] = np.nan

    # 9. Final column ordering per spec
    output_cols = [
        "canonical_id",
        "team_idx",
        "team_name",
        "conference",
        "season",
        "bullpen_quality_z",
        "bullpen_adj",
        "wrc_plus",
        "wrc_offense_adj",
        "conf_strength_adj",
        "batting_fb_pct",
        "batting_fb_factor",
        "pct_rhb",
        "pct_lhb",
        "effective_rhb_frac",
        "n_games",
    ]
    # Only include columns that exist
    output_cols = [c for c in output_cols if c in base.columns]
    base = base[output_cols]

    return base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build unified team lookup table (team_table.csv)"
    )
    parser.add_argument(
        "--out",
        default="data/processed/team_table.csv",
        help="Output path (default: data/processed/team_table.csv)",
    )
    parser.add_argument(
        "--registry",
        default="data/registries/canonical_teams_2026.csv",
        help="Canonical team registry CSV",
    )
    parser.add_argument(
        "--team-index",
        default="data/processed/run_event_team_index.csv",
        help="Stan model team index CSV",
    )
    parser.add_argument(
        "--bullpen",
        default="data/processed/bullpen_quality.csv",
        help="Bullpen quality CSV",
    )
    parser.add_argument(
        "--batting",
        default="data/raw/d1baseball/batting_advanced.tsv",
        help="D1B batting advanced TSV",
    )
    parser.add_argument(
        "--crosswalk",
        default="data/registries/d1baseball_crosswalk.csv",
        help="D1B team name → canonical_id crosswalk CSV",
    )
    parser.add_argument(
        "--d1b-root",
        default="data/raw/d1baseball",
        help="Root directory for seasonized D1B files (supports YYYY subdirs).",
    )
    parser.add_argument(
        "--as-of-season",
        type=int,
        default=None,
        help="As-of season for recency weighting (default inferred from registry).",
    )
    args = parser.parse_args()

    # Resolve paths relative to repo root (script lives in scripts/)
    repo_root = Path(__file__).parent.parent
    out_path = repo_root / args.out
    registry_path = repo_root / args.registry
    team_index_path = repo_root / args.team_index
    bullpen_path = repo_root / args.bullpen
    bat_path = repo_root / args.batting
    xwalk_path = repo_root / args.crosswalk
    d1b_root = repo_root / args.d1b_root

    print(f"Building team table...", file=sys.stderr)

    tt = build_team_table(
        registry_path=registry_path,
        team_index_path=team_index_path,
        bullpen_path=bullpen_path,
        bat_path=bat_path,
        xwalk_path=xwalk_path,
        d1b_root=d1b_root,
        as_of_season=args.as_of_season,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tt.to_csv(out_path, index=False)

    print(f"\nWrote {len(tt)} rows to {out_path}", file=sys.stderr)
    print(f"  With team_idx > 0:    {(tt['team_idx'] > 0).sum()}", file=sys.stderr)
    print(f"  With bullpen_quality_z: {tt['bullpen_quality_z'].notna().sum()}", file=sys.stderr)
    print(f"  With wRC+:            {tt['wrc_plus'].notna().sum()}", file=sys.stderr)
    print(f"  With wrc_offense_adj != 0: {(tt['wrc_offense_adj'] != 0).sum()}", file=sys.stderr)
    print(f"  With batting_fb_factor:   {(tt['batting_fb_factor'] != 1.0).sum()}", file=sys.stderr)


if __name__ == "__main__":
    main()
