"""
Build bullpen depth/quality and pitcher workload/fatigue tables from ESPN JSONL.

Reads scraped ESPN game JSONL files and produces two CSVs:

  1. bullpen_quality.csv   -- per-team per-season bullpen aggregate metrics
  2. pitcher_workload.csv  -- per-pitcher per-appearance rolling workload & fatigue

Usage:
  python3 scripts/build_bullpen_fatigue.py
  python3 scripts/build_bullpen_fatigue.py --espn-dir data/raw/espn \
      --canonical data/registries/canonical_teams_2026.csv \
      --out-quality data/processed/bullpen_quality.csv \
      --out-workload data/processed/pitcher_workload.csv \
      --seasons 2024,2025,2026
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401  -- puts src/ on sys.path
from ncaa_baseball.phase1 import (
    build_odds_name_to_canonical,
    load_canonical_teams,
    resolve_odds_teams,
)

# ---------------------------------------------------------------------------
# IP parsing: ESPN encodes partial innings as .1 = 1/3, .2 = 2/3
# ---------------------------------------------------------------------------

def parse_ip(raw: str | float | None) -> float | None:
    """Convert ESPN IP string to decimal innings.

    "5.0" -> 5.0, "5.1" -> 5.333, "5.2" -> 5.667.
    """
    if raw is None or raw == "":
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    whole = int(val)
    frac = round(val - whole, 1)
    if abs(frac - 0.1) < 0.01:
        return whole + 1.0 / 3.0
    if abs(frac - 0.2) < 0.01:
        return whole + 2.0 / 3.0
    return val


def _safe_float(x, default: float | None = None) -> float | None:
    if x is None or x == "":
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _parse_pc(stats: dict) -> float | None:
    """Extract pitch count from stats dict.  Prefer 'PC'; fall back to 'PC-ST'."""
    pc = _safe_float(stats.get("PC"))
    if pc is not None:
        return pc
    pcst = (stats.get("PC-ST") or "").strip()
    if pcst:
        try:
            return float(pcst.split("-")[0])
        except (ValueError, IndexError):
            pass
    return None


# ---------------------------------------------------------------------------
# JSONL ingestion
# ---------------------------------------------------------------------------

def _extract_appearances(
    game: dict,
    resolve_team,
) -> list[dict]:
    """Extract per-pitcher appearance rows from one game JSON."""
    box = game.get("boxscore") or {}
    if not box:
        return []
    game_date = (game.get("date") or "")[:10]
    event_id = str(game.get("event_id") or game.get("id") or "")
    try:
        season = int(game.get("season") or 0)
    except (TypeError, ValueError):
        season = 0

    home = game.get("home_team") or {}
    away = game.get("away_team") or {}

    rows: list[dict] = []
    for team_info, home_away in ((home, "home"), (away, "away")):
        team_id = team_info.get("id")
        team_name = (team_info.get("name") or "").strip()
        abbr = (team_info.get("abbreviation") or "").strip()

        section = box.get(abbr) or box.get(str(team_id or "")) or {}
        pitching = section.get("pitching")
        if not pitching:
            continue

        canonical_id = resolve_team(team_name) or ""

        for athlete in pitching:
            stats = athlete.get("stats") or {}
            ip = parse_ip(stats.get("IP"))
            if ip is None:
                continue  # skip lines with no IP

            espn_id = athlete.get("espn_id")
            if espn_id is not None:
                try:
                    espn_id = str(int(espn_id))
                except (TypeError, ValueError):
                    espn_id = str(espn_id)

            rows.append({
                "game_date": game_date,
                "event_id": event_id,
                "season": season,
                "team_abbr": abbr,
                "team_name": team_name,
                "team_canonical_id": canonical_id,
                "home_away": home_away,
                "pitcher_espn_id": espn_id or "",
                "pitcher_name": (athlete.get("name") or "").strip(),
                "starter": bool(athlete.get("starter")),
                "IP": ip,
                "H": _safe_float(stats.get("H"), 0.0),
                "ER": _safe_float(stats.get("ER"), 0.0),
                "R": _safe_float(stats.get("R"), 0.0),
                "BB": _safe_float(stats.get("BB"), 0.0),
                "K": _safe_float(stats.get("K"), 0.0),
                "HR": _safe_float(stats.get("HR"), 0.0),
                "PC": _parse_pc(stats),
            })
    return rows


def load_all_appearances(
    espn_dir: Path,
    seasons: list[str],
    resolve_team,
) -> pd.DataFrame:
    """Read JSONL files and return a DataFrame of all pitcher appearances."""
    all_rows: list[dict] = []
    files_read = 0
    games_total = 0
    games_with_pitching = 0

    for season in seasons:
        path = espn_dir / f"games_{season}.jsonl"
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        files_read += 1
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    g = json.loads(line)
                except json.JSONDecodeError:
                    continue
                games_total += 1
                rows = _extract_appearances(g, resolve_team)
                if rows:
                    games_with_pitching += 1
                    all_rows.extend(rows)

    pct = 100.0 * games_with_pitching / max(games_total, 1)
    print(f"Read {files_read} JSONL file(s): {games_total} games, "
          f"{games_with_pitching} with pitching ({pct:.1f}%), "
          f"{len(all_rows)} pitcher appearances")

    if not all_rows:
        return pd.DataFrame(columns=[
            "game_date", "event_id", "season", "team_abbr", "team_name",
            "team_canonical_id", "home_away", "pitcher_espn_id",
            "pitcher_name", "starter", "IP", "H", "ER", "R", "BB", "K",
            "HR", "PC",
        ])
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Table A: Bullpen Quality (per team, per season)
# ---------------------------------------------------------------------------

def build_bullpen_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team per-season bullpen aggregate metrics for relievers only."""
    rp = df[~df["starter"]].copy()
    if rp.empty:
        return pd.DataFrame(columns=[
            "team_canonical_id", "season", "bullpen_ip", "bullpen_era",
            "bullpen_k_per_9", "bullpen_bb_per_9", "bullpen_whip",
            "n_relievers", "bullpen_depth_score",
        ])

    grp = rp.groupby(["team_canonical_id", "season"], dropna=False)
    agg = grp.agg(
        bullpen_ip=("IP", "sum"),
        total_er=("ER", "sum"),
        total_h=("H", "sum"),
        total_bb=("BB", "sum"),
        total_k=("K", "sum"),
        total_hr=("HR", "sum"),
        n_relievers=("pitcher_espn_id", "nunique"),
    ).reset_index()

    ip_safe = agg["bullpen_ip"].clip(lower=0.01)
    agg["bullpen_era"] = 9.0 * agg["total_er"] / ip_safe
    agg["bullpen_k_per_9"] = 9.0 * agg["total_k"] / ip_safe
    agg["bullpen_bb_per_9"] = 9.0 * agg["total_bb"] / ip_safe
    agg["bullpen_whip"] = (agg["total_h"] + agg["total_bb"]) / ip_safe

    # Composite depth score: z-score of ERA (invert), K/9, BB/9 (invert)
    # Higher score = better bullpen
    def _zscore(s: pd.Series) -> pd.Series:
        m, sd = s.mean(), s.std()
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=s.index)
        return (s - m) / sd

    z_era = -_zscore(agg["bullpen_era"])        # lower ERA is better
    z_k9 = _zscore(agg["bullpen_k_per_9"])       # higher K/9 is better
    z_bb9 = -_zscore(agg["bullpen_bb_per_9"])     # lower BB/9 is better
    agg["bullpen_depth_score"] = (z_era + z_k9 + z_bb9) / 3.0

    out_cols = [
        "team_canonical_id", "season", "bullpen_ip", "bullpen_era",
        "bullpen_k_per_9", "bullpen_bb_per_9", "bullpen_whip",
        "n_relievers", "bullpen_depth_score",
    ]
    return agg[out_cols].sort_values(
        ["season", "bullpen_depth_score"], ascending=[True, False],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table B: Pitcher Recent Workload (per pitcher, per appearance)
# ---------------------------------------------------------------------------

def build_pitcher_workload(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling workload & fatigue for every pitcher appearance."""
    if df.empty:
        return pd.DataFrame(columns=[
            "pitcher_espn_id", "pitcher_name", "team_canonical_id",
            "game_date", "season", "starter", "ip_today", "pc_today",
            "ip_last_3d", "ip_last_5d", "ip_last_7d",
            "pc_last_3d", "pc_last_5d", "pc_last_7d",
            "days_rest", "appearances_last_7d", "fatigue_score",
        ])

    work = df[["pitcher_espn_id", "pitcher_name", "team_canonical_id",
               "game_date", "season", "starter", "event_id",
               "IP", "PC"]].copy()
    work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce")
    work = work.dropna(subset=["game_date"]).copy()
    work = work.sort_values(["pitcher_espn_id", "game_date", "event_id"]).reset_index(drop=True)
    work["IP"] = work["IP"].fillna(0.0)
    work["PC"] = work["PC"].fillna(0.0)

    # For each appearance, compute rolling lookback windows.
    # Group by pitcher and iterate chronologically.
    results: list[dict] = []

    for pid, grp in work.groupby("pitcher_espn_id", sort=False):
        grp = grp.sort_values("game_date").reset_index(drop=True)
        dates = grp["game_date"].values
        ips = grp["IP"].values
        pcs = grp["PC"].values

        for i in range(len(grp)):
            row = grp.iloc[i]
            current_date = dates[i]

            # Lookback windows (prior appearances only, not including current)
            ip_3d = 0.0
            ip_5d = 0.0
            ip_7d = 0.0
            pc_3d = 0.0
            pc_5d = 0.0
            pc_7d = 0.0
            appearances_7d = 0
            last_appearance_date = None

            for j in range(i - 1, -1, -1):
                prev_date = dates[j]
                delta_days = (current_date - prev_date) / np.timedelta64(1, "D")
                if delta_days > 7:
                    break
                if last_appearance_date is None:
                    last_appearance_date = prev_date
                appearances_7d += 1
                if delta_days <= 7:
                    ip_7d += ips[j]
                    pc_7d += pcs[j]
                if delta_days <= 5:
                    ip_5d += ips[j]
                    pc_5d += pcs[j]
                if delta_days <= 3:
                    ip_3d += ips[j]
                    pc_3d += pcs[j]

            if last_appearance_date is not None:
                days_rest = float(
                    (current_date - last_appearance_date) / np.timedelta64(1, "D")
                )
            else:
                days_rest = np.nan  # first appearance -- no prior data

            # Fatigue composite:
            # 0.4 * (pc_last_3d / 100) + 0.3 * (ip_last_5d / 10) + 0.3 * (1 / (1 + days_rest))
            if np.isnan(days_rest):
                fatigue = 0.4 * (pc_3d / 100.0) + 0.3 * (ip_5d / 10.0) + 0.3 * 0.0
            else:
                fatigue = (
                    0.4 * (pc_3d / 100.0)
                    + 0.3 * (ip_5d / 10.0)
                    + 0.3 * (1.0 / (1.0 + days_rest))
                )

            results.append({
                "pitcher_espn_id": pid,
                "pitcher_name": row["pitcher_name"],
                "team_canonical_id": row["team_canonical_id"],
                "game_date": pd.Timestamp(current_date).strftime("%Y-%m-%d"),
                "season": row["season"],
                "starter": row["starter"],
                "event_id": row["event_id"],
                "ip_today": row["IP"],
                "pc_today": row["PC"],
                "ip_last_3d": round(ip_3d, 3),
                "ip_last_5d": round(ip_5d, 3),
                "ip_last_7d": round(ip_7d, 3),
                "pc_last_3d": round(pc_3d, 1),
                "pc_last_5d": round(pc_5d, 1),
                "pc_last_7d": round(pc_7d, 1),
                "days_rest": days_rest if not np.isnan(days_rest) else None,
                "appearances_last_7d": appearances_7d,
                "fatigue_score": round(fatigue, 4),
            })

    out = pd.DataFrame(results)
    return out.sort_values(
        ["pitcher_espn_id", "game_date"],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------

def print_summary(quality: pd.DataFrame, workload: pd.DataFrame) -> None:
    """Print summary stats to stdout."""
    sep = "-" * 72

    # --- Bullpen Quality ---
    if not quality.empty:
        print(f"\n{sep}")
        print("BULLPEN QUALITY SUMMARY")
        print(sep)
        print(f"Teams: {quality['team_canonical_id'].nunique()}, "
              f"Seasons: {sorted(quality['season'].unique())}")
        print(f"Median bullpen IP: {quality['bullpen_ip'].median():.1f}, "
              f"Median ERA: {quality['bullpen_era'].median():.2f}, "
              f"Median K/9: {quality['bullpen_k_per_9'].median():.2f}")

        latest = quality["season"].max()
        q_latest = quality[quality["season"] == latest].copy()

        if len(q_latest) >= 10:
            print(f"\nTop 10 bullpens ({latest} by depth_score):")
            top = q_latest.nlargest(10, "bullpen_depth_score")
            for _, r in top.iterrows():
                print(f"  {r['team_canonical_id']:30s}  ERA={r['bullpen_era']:5.2f}  "
                      f"K/9={r['bullpen_k_per_9']:5.2f}  BB/9={r['bullpen_bb_per_9']:5.2f}  "
                      f"WHIP={r['bullpen_whip']:5.2f}  depth={r['bullpen_depth_score']:+.3f}  "
                      f"(n={int(r['n_relievers'])})")

            print(f"\nBottom 10 bullpens ({latest} by depth_score):")
            bot = q_latest.nsmallest(10, "bullpen_depth_score")
            for _, r in bot.iterrows():
                print(f"  {r['team_canonical_id']:30s}  ERA={r['bullpen_era']:5.2f}  "
                      f"K/9={r['bullpen_k_per_9']:5.2f}  BB/9={r['bullpen_bb_per_9']:5.2f}  "
                      f"WHIP={r['bullpen_whip']:5.2f}  depth={r['bullpen_depth_score']:+.3f}  "
                      f"(n={int(r['n_relievers'])})")
    else:
        print("\n[No bullpen quality data]")

    # --- Pitcher Workload / Fatigue ---
    if not workload.empty:
        print(f"\n{sep}")
        print("PITCHER WORKLOAD / FATIGUE SUMMARY")
        print(sep)
        print(f"Total appearances: {len(workload)}, "
              f"Unique pitchers: {workload['pitcher_espn_id'].nunique()}")

        starters = workload[workload["starter"]].copy()
        relievers = workload[~workload["starter"]].copy()
        print(f"Starter appearances: {len(starters)}, "
              f"Reliever appearances: {len(relievers)}")

        if not workload.empty:
            print(f"\nFatigue score distribution (all pitchers):")
            desc = workload["fatigue_score"].describe()
            for stat in ("mean", "std", "min", "25%", "50%", "75%", "max"):
                print(f"  {stat:>5s}: {desc[stat]:.4f}")

        # Most fatigued appearances (starters)
        if len(starters) >= 10:
            print(f"\nMost fatigued starters (top 10 by fatigue_score):")
            top_fat = starters.nlargest(10, "fatigue_score")
            for _, r in top_fat.iterrows():
                rest_str = f"{r['days_rest']:.0f}d" if r["days_rest"] is not None and not pd.isna(r["days_rest"]) else "N/A"
                print(f"  {r['pitcher_name']:25s}  {r['game_date']}  "
                      f"fatigue={r['fatigue_score']:.4f}  rest={rest_str}  "
                      f"pc_3d={r['pc_last_3d']:.0f}  ip_5d={r['ip_last_5d']:.1f}  "
                      f"team={r['team_canonical_id']}")

        # Least fatigued starters (most rested)
        if len(starters) >= 10:
            # Filter to starters with at least one prior appearance
            with_rest = starters[starters["days_rest"].notna()]
            if len(with_rest) >= 10:
                print(f"\nLeast fatigued starters (bottom 10 by fatigue_score, with prior appearance):")
                bot_fat = with_rest.nsmallest(10, "fatigue_score")
                for _, r in bot_fat.iterrows():
                    rest_str = f"{r['days_rest']:.0f}d"
                    print(f"  {r['pitcher_name']:25s}  {r['game_date']}  "
                          f"fatigue={r['fatigue_score']:.4f}  rest={rest_str}  "
                          f"pc_3d={r['pc_last_3d']:.0f}  ip_5d={r['ip_last_5d']:.1f}  "
                          f"team={r['team_canonical_id']}")
    else:
        print("\n[No workload data]")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build bullpen quality and pitcher workload/fatigue tables from ESPN JSONL.",
    )
    parser.add_argument(
        "--espn-dir", type=Path,
        default=Path("data/raw/espn"),
        help="Directory containing games_YYYY.jsonl files",
    )
    parser.add_argument(
        "--canonical", type=Path,
        default=Path("data/registries/canonical_teams_2026.csv"),
        help="Path to canonical teams registry CSV",
    )
    parser.add_argument(
        "--out-quality", type=Path,
        default=Path("data/processed/bullpen_quality.csv"),
        help="Output path for bullpen quality CSV",
    )
    parser.add_argument(
        "--out-workload", type=Path,
        default=Path("data/processed/pitcher_workload.csv"),
        help="Output path for pitcher workload CSV",
    )
    parser.add_argument(
        "--seasons", type=str, default="2024,2025,2026",
        help="Comma-separated seasons to process",
    )
    args = parser.parse_args()

    # ---- Load canonical team registry for name resolution ----
    if not args.canonical.exists():
        print(f"ERROR: canonical teams file not found: {args.canonical}", file=sys.stderr)
        return 1

    canonical = load_canonical_teams(args.canonical)
    name_to_canonical = build_odds_name_to_canonical(canonical)

    def resolve_team(team_name: str) -> str | None:
        t = resolve_odds_teams(team_name, team_name, canonical, name_to_canonical)[0]
        return t[0] if t else None

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]

    # ---- Extract all pitcher appearances ----
    print("Loading pitcher appearances from ESPN JSONL ...")
    appearances = load_all_appearances(args.espn_dir, seasons, resolve_team)
    if appearances.empty:
        print("No pitcher appearances found.  Exiting.")
        return 0

    # Ensure numeric types
    for col in ("IP", "H", "ER", "R", "BB", "K", "HR", "PC"):
        appearances[col] = pd.to_numeric(appearances[col], errors="coerce")
    appearances["starter"] = appearances["starter"].fillna(False).astype(bool)

    # ---- Table A: Bullpen Quality ----
    print("Building bullpen quality table ...")
    quality = build_bullpen_quality(appearances)
    args.out_quality.parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(args.out_quality, index=False)
    print(f"  -> {args.out_quality}  ({len(quality)} rows, "
          f"{quality['team_canonical_id'].nunique()} teams)")

    # ---- Table B: Pitcher Workload ----
    print("Building pitcher workload table ...")
    workload = build_pitcher_workload(appearances)
    args.out_workload.parent.mkdir(parents=True, exist_ok=True)
    workload.to_csv(args.out_workload, index=False)
    print(f"  -> {args.out_workload}  ({len(workload)} rows, "
          f"{workload['pitcher_espn_id'].nunique()} unique pitchers)")

    # ---- Summary ----
    print_summary(quality, workload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
