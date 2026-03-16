"""
Resolve starting pitchers for a day's schedule and enrich with pitcher_table data.

Extracts starter resolution + D1B enrichment from predict_day.py into a standalone module.
With the pre-computed pitcher_table.csv, this is a simple:
  1. StarterLookup.get_starter() → projected starter name + Stan index
  2. Join enrichment from pitcher_table (throws, ability_adj, fb_sensitivity)
  3. Aggregate bullpen FB sensitivity per team from pitcher_table
  4. Look up wRC+ offense adjustment from team_table (only for team_idx=0)

Usage:
    python3 scripts/resolve_starters.py \\
        --schedule data/daily/2026-03-14/schedule.csv \\
        --date 2026-03-14 \\
        --out data/daily/2026-03-14/starters.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

import _bootstrap  # noqa: F401 — adds scripts/ to sys.path so local imports work
from lookup_starters import StarterLookup


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Lowercase + normalise apostrophes."""
    return s.lower().replace("\u2019", "'").replace("\u2018", "'")


def _espn_numeric_id(pitcher_id: str) -> str:
    """Extract numeric ESPN pitcher id from known pitcher_id formats."""
    if not pitcher_id:
        return ""
    pid = str(pitcher_id).strip()
    if pid.startswith("ESPN_"):
        return pid.replace("ESPN_", "", 1)
    if pid.isdigit():
        return pid
    return ""


def _ip_to_float(ip_val) -> float:
    """Convert baseball IP notation (e.g. 5.2 = 5 and 2/3) to decimal innings."""
    if ip_val is None:
        return 0.0
    s = str(ip_val).strip()
    if not s:
        return 0.0
    try:
        if "." in s:
            whole_s, frac_s = s.split(".", 1)
            whole = int(whole_s) if whole_s else 0
            if frac_s in ("0", "1", "2"):
                outs = int(frac_s)
                return float(whole + outs / 3.0)
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _expected_starter_ip(appearances: pd.DataFrame, cid: str, pitcher_id: str) -> float:
    """Estimate starter expected IP from recent starts for this pitcher/team."""
    if appearances.empty or not cid:
        return 5.5

    team_apps = appearances[appearances["team_canonical_id"] == cid]
    if team_apps.empty:
        return 5.5

    pid = str(pitcher_id or "").strip()
    rows = team_apps[team_apps["role"] == "starter"].copy()
    if rows.empty:
        return 5.5

    if pid:
        rows_pid = rows[rows["pitcher_id"] == pid]
        if not rows_pid.empty:
            rows = rows_pid

    rows = rows.sort_values("game_date", ascending=False).head(5)
    ip_vals = rows["ip_float"].tolist()
    if not ip_vals:
        return 5.5

    # Recency weights: most recent has highest influence.
    w = [1.0, 0.75, 0.55, 0.4, 0.3][: len(ip_vals)]
    num = sum(float(x) * float(wi) for x, wi in zip(ip_vals, w))
    den = sum(w)
    exp_ip = num / den if den > 0 else 5.5
    return max(3.5, min(7.5, float(exp_ip)))


def _match_pitcher_by_id(
    pitcher_tbl: pd.DataFrame,
    cid: str,
    pitcher_id: str,
) -> Optional[pd.Series]:
    """Find pitcher_table row by canonical_id + pitcher_espn_id."""
    if not cid:
        return None
    espn_id = _espn_numeric_id(pitcher_id)
    if not espn_id:
        return None
    sub = pitcher_tbl[
        (pitcher_tbl["team_canonical_id"] == cid)
        & (pitcher_tbl["pitcher_espn_id"] == espn_id)
    ]
    if len(sub) == 1:
        return sub.iloc[0]
    return None


def _match_pitcher(
    pitcher_tbl: pd.DataFrame,
    cid: str,
    name: str,
) -> Optional[pd.Series]:
    """Find a row in pitcher_table for (team_canonical_id, pitcher_name).

    Matching strategy (mirrors predict_day.py _d1b_lookup):
      1. Exact (cid, name) after normalisation
      2. Last-name match if unique
      3. First-initial + last-name if unique among multiple last-name hits
      4. Collapsed-name match (strips spaces/hyphens/dots)
    """
    if not cid or not name or name.lower() == "unknown":
        return None

    sub = pitcher_tbl[pitcher_tbl["team_canonical_id"] == cid]
    if sub.empty:
        return None

    name_norm = _norm(name)

    # 1. Exact match
    mask = sub["_name_norm"] == name_norm
    if mask.sum() == 1:
        return sub[mask].iloc[0]

    # 2. Last-name match
    parts = name_norm.split()
    if not parts:
        return None
    lastname = parts[-1]
    mask_ln = sub["_name_norm"].str.split().str[-1] == lastname
    hits = sub[mask_ln]
    if len(hits) == 1:
        return hits.iloc[0]

    # 3. First-initial + last-name
    if len(hits) > 1 and len(parts) >= 2:
        initial = parts[0].rstrip(".")
        if len(initial) == 1:
            filtered = hits[hits["_name_norm"].str.startswith(initial)]
            if len(filtered) == 1:
                return filtered.iloc[0]

    # 4. Collapsed name
    collapsed = name_norm.replace(" ", "").replace("-", "").replace(".", "")
    mask_col = sub["_name_norm"].str.replace(" ", "").str.replace("-", "").str.replace(".", "") == collapsed
    hits_col = sub[mask_col]
    if len(hits_col) == 1:
        return hits_col.iloc[0]

    return None


def _match_pitcher_with_method(
    pitcher_tbl: pd.DataFrame,
    cid: str,
    pitcher_id: str,
    name: str,
) -> tuple[Optional[pd.Series], str]:
    """Return matched pitcher row plus resolution method."""
    by_id = _match_pitcher_by_id(pitcher_tbl, cid, pitcher_id)
    if by_id is not None:
        return by_id, "id_match"
    by_name = _match_pitcher(pitcher_tbl, cid, name)
    if by_name is not None:
        return by_name, "name_match"
    return None, "lookup_only"


# ── Core function ─────────────────────────────────────────────────────────────

def resolve_starters(
    schedule_csv: Path,
    pitcher_table_csv: Path = Path("data/processed/pitcher_table.csv"),
    team_table_csv: Path = Path("data/processed/team_table.csv"),
    appearances_csv: Path = Path("data/processed/pitcher_appearances.csv"),
    pitcher_registry_csv: Path = Path("data/processed/pitcher_registry.csv"),
    pitcher_index_csv: Path = Path("data/processed/run_event_pitcher_index.csv"),
    weekend_rotations_csv: Path = Path("data/processed/weekend_rotations.csv"),
    d1b_rotations_csv: Path = Path("data/processed/d1baseball_rotations.csv"),
    canonical_csv: Path = Path("data/registries/canonical_teams_2026.csv"),
    date: str = "",
    out_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Resolve starting pitchers for each game in schedule_csv.

    Parameters
    ----------
    schedule_csv:
        Path to a schedule CSV with columns: game_num, home_canonical_id,
        away_canonical_id (and optionally home_team_idx, away_team_idx).
    pitcher_table_csv:
        Pre-computed pitcher lookup table (build_pitcher_table.py output).
    team_table_csv:
        Pre-computed team table with wrc_offense_adj (build_team_table.py output).
    appearances_csv, pitcher_registry_csv, pitcher_index_csv:
        Inputs for StarterLookup.
    weekend_rotations_csv, d1b_rotations_csv:
        Rotation data passed to StarterLookup.
    canonical_csv:
        Canonical team registry — only needed by StarterLookup internally.
    date:
        Game date string (YYYY-MM-DD).  Required for starter projection.
    out_csv:
        If provided, write output CSV here.

    Returns
    -------
    pd.DataFrame with one row per game and columns defined in the module docstring.
    """
    # ── Load schedule ─────────────────────────────────────────────────────────
    schedule = pd.read_csv(schedule_csv, dtype=str)

    # Accept both column name conventions:
    #   home_canonical_id / away_canonical_id  (canonical)
    #   home_cid / away_cid                    (build_schedule output)
    if "home_canonical_id" not in schedule.columns and "home_cid" in schedule.columns:
        schedule = schedule.rename(columns={"home_cid": "home_canonical_id",
                                            "away_cid": "away_canonical_id"})

    required = {"game_num", "home_canonical_id", "away_canonical_id"}
    missing = required - set(schedule.columns)
    if missing:
        raise ValueError(f"schedule_csv missing columns: {missing}")

    # ── Load pitcher_table ────────────────────────────────────────────────────
    pt = pd.read_csv(pitcher_table_csv, dtype=str)
    if "pitcher_espn_id" in pt.columns:
        pt["pitcher_espn_id"] = pt["pitcher_espn_id"].fillna("").astype(str).str.strip()
    # Pre-normalise names for matching
    pt["_name_norm"] = pt["pitcher_name"].fillna("").apply(_norm)
    # Numeric coercions
    for col in ("pitcher_idx", "fb_sensitivity", "d1b_ability_adj"):
        pt[col] = pd.to_numeric(pt[col], errors="coerce")
    pt["pitcher_idx"] = pt["pitcher_idx"].fillna(0).astype(int)
    pt["fb_sensitivity"] = pt["fb_sensitivity"].fillna(1.0)
    pt["d1b_ability_adj"] = pt["d1b_ability_adj"].fillna(0.0)

    # ── Load appearances for IP expectation + QA context ─────────────────────
    app = pd.read_csv(appearances_csv, dtype=str)
    for col in ("pitcher_id", "team_canonical_id", "role"):
        if col not in app.columns:
            app[col] = ""
        app[col] = app[col].fillna("").astype(str).str.strip()
    app["game_date"] = pd.to_datetime(app.get("game_date"), errors="coerce")
    app["ip_float"] = app.get("ip", "").apply(_ip_to_float)

    # ── Pre-compute bullpen FB sensitivity per team ───────────────────────────
    # Use all pitchers in the team (not just non-SP), averaged.
    # If a team has no data, default 1.0.
    bp_fb_by_team: dict[str, float] = {}
    for cid, grp in pt.groupby("team_canonical_id"):
        vals = grp["fb_sensitivity"].dropna()
        bp_fb_by_team[str(cid)] = float(vals.mean()) if len(vals) > 0 else 1.0

    # ── Load team_table for wRC+ offense adjustment + batting FB factor ──────
    wrc_adj_by_team: dict[str, float] = {}  # canonical_id → adj (only for team_idx=0)
    batting_fb_by_team: dict[str, float] = {}  # canonical_id → FB factor (for wind model)
    team_idx_by_cid: dict[str, int] = {}
    if team_table_csv.exists():
        tt = pd.read_csv(team_table_csv, dtype=str)
        tt["team_idx"] = pd.to_numeric(tt["team_idx"], errors="coerce").fillna(0).astype(int)
        tt["wrc_offense_adj"] = pd.to_numeric(tt["wrc_offense_adj"], errors="coerce").fillna(0.0)
        tt["batting_fb_factor"] = pd.to_numeric(tt.get("batting_fb_factor"), errors="coerce").fillna(1.0)
        for _, row in tt.iterrows():
            cid = str(row.get("canonical_id", "")).strip()
            if not cid:
                continue
            tidx = int(row["team_idx"])
            team_idx_by_cid[cid] = tidx
            # Batting FB factor for wind scaling (all teams)
            batting_fb_by_team[cid] = float(row["batting_fb_factor"])
            # Only apply wRC+ adj to teams NOT in the Stan model
            if tidx == 0:
                adj = float(row["wrc_offense_adj"])
                if adj != 0.0:
                    wrc_adj_by_team[cid] = adj

    # ── Instantiate StarterLookup ─────────────────────────────────────────────
    print("Loading StarterLookup...", file=sys.stderr)

    # Build kwargs — only pass files that exist so StarterLookup doesn't error
    sl_kwargs: dict = {
        "appearances_csv": appearances_csv,
        "registry_csv": pitcher_registry_csv,
        "pitcher_index_csv": pitcher_index_csv,
        "canonical_csv": canonical_csv,
    }
    if weekend_rotations_csv.exists():
        sl_kwargs["weekend_rotations_csv"] = weekend_rotations_csv
    if d1b_rotations_csv.exists():
        sl_kwargs["d1baseball_rotations_csv"] = d1b_rotations_csv

    starter_lookup = StarterLookup(**sl_kwargs)

    # ── Resolve each game ─────────────────────────────────────────────────────
    rows = []
    for _, game in schedule.iterrows():
        game_num = str(game["game_num"])
        h_cid = str(game.get("home_canonical_id", "")).strip()
        a_cid = str(game.get("away_canonical_id", "")).strip()

        # Get projected starters via StarterLookup
        hp_name, hp_id, hp_idx_raw = starter_lookup.get_starter(h_cid, date)
        ap_name, ap_id, ap_idx_raw = starter_lookup.get_starter(a_cid, date)

        hp_idx = int(hp_idx_raw) if hp_idx_raw else 0
        ap_idx = int(ap_idx_raw) if ap_idx_raw else 0

        # ── Enrich from pitcher_table ─────────────────────────────────────────
        hp_row, hp_resolution = _match_pitcher_with_method(pt, h_cid, hp_id, hp_name)
        ap_row, ap_resolution = _match_pitcher_with_method(pt, a_cid, ap_id, ap_name)

        def _extract(row: Optional[pd.Series], starter_idx: int) -> dict:
            """Extract enrichment fields from a pitcher_table row."""
            if row is None:
                return {
                    "idx": starter_idx,
                    "throws": "",
                    "ability_adj": 0.0,
                    "ability_src": "",
                    "fb_sens": 1.0,
                }
            # pitcher_table's pitcher_idx supersedes StarterLookup's idx when
            # available and non-zero (both should agree, but table is canonical).
            idx = int(row["pitcher_idx"]) if int(row["pitcher_idx"]) > 0 else starter_idx
            throws = str(row.get("throws", "")).strip()
            throws = throws if throws not in ("", "nan") else ""
            # Only use ability_adj when pitcher has no posterior (idx == 0)
            ability_adj = 0.0
            ability_src = ""
            if idx == 0:
                ability_adj = float(row["d1b_ability_adj"]) if not pd.isna(row["d1b_ability_adj"]) else 0.0
                src = str(row.get("d1b_ability_source", "")).strip()
                ability_src = src if src not in ("", "nan") else ""
            fb_sens = float(row["fb_sensitivity"]) if not pd.isna(row["fb_sensitivity"]) else 1.0
            return {
                "idx": idx,
                "throws": throws,
                "ability_adj": ability_adj,
                "ability_src": ability_src,
                "fb_sens": fb_sens,
            }

        hp_info = _extract(hp_row, hp_idx)
        ap_info = _extract(ap_row, ap_idx)

        # Resolved starter indices (from table if available, else StarterLookup)
        hp_idx_final = hp_info["idx"]
        ap_idx_final = ap_info["idx"]

        # Bullpen FB sensitivity (team average from pitcher_table)
        hp_bp_fb_sens = bp_fb_by_team.get(h_cid, 1.0)
        ap_bp_fb_sens = bp_fb_by_team.get(a_cid, 1.0)

        # Dynamic expected starter innings (used by simulate.py IP split).
        hp_expected_ip = _expected_starter_ip(app, h_cid, hp_id)
        ap_expected_ip = _expected_starter_ip(app, a_cid, ap_id)

        # wRC+ offense adjustment (team_table, only for team_idx=0)
        home_wrc_adj = wrc_adj_by_team.get(h_cid, 0.0)
        away_wrc_adj = wrc_adj_by_team.get(a_cid, 0.0)

        # Batting fly ball factor for wind model scaling
        home_batting_fb = batting_fb_by_team.get(h_cid, 1.0)
        away_batting_fb = batting_fb_by_team.get(a_cid, 1.0)

        print(
            f"  Game {game_num}: {a_cid} @ {h_cid} | "
            f"HP={hp_name} (idx={hp_idx_final}, throws={hp_info['throws'] or '?'}, "
            f"adj={hp_info['ability_adj']:+.3f} [{hp_info['ability_src'] or 'none'}]) | "
            f"AP={ap_name} (idx={ap_idx_final}, throws={ap_info['throws'] or '?'}, "
            f"adj={ap_info['ability_adj']:+.3f} [{ap_info['ability_src'] or 'none'}])",
            file=sys.stderr,
        )

        rows.append({
            "game_num": game_num,
            "home_canonical_id": h_cid,
            "away_canonical_id": a_cid,
            "home_cid": h_cid,
            "away_cid": a_cid,
            "home_starter": hp_name,
            "away_starter": ap_name,
            "home_starter_idx": hp_idx_final,
            "away_starter_idx": ap_idx_final,
            "hp_throws": hp_info["throws"],
            "ap_throws": ap_info["throws"],
            "hp_ability_adj": hp_info["ability_adj"],
            "ap_ability_adj": ap_info["ability_adj"],
            "hp_ability_src": hp_info["ability_src"],
            "ap_ability_src": ap_info["ability_src"],
            "hp_fb_sens": hp_info["fb_sens"],
            "ap_fb_sens": ap_info["fb_sens"],
            "hp_bp_fb_sens": hp_bp_fb_sens,
            "ap_bp_fb_sens": ap_bp_fb_sens,
            "hp_expected_ip": hp_expected_ip,
            "ap_expected_ip": ap_expected_ip,
            "home_resolution_method": hp_resolution,
            "away_resolution_method": ap_resolution,
            "home_d1b_fallback": int(str(hp_id).startswith("d1b_")),
            "away_d1b_fallback": int(str(ap_id).startswith("d1b_")),
            "home_wrc_adj": home_wrc_adj,
            "away_wrc_adj": away_wrc_adj,
            "home_batting_fb": home_batting_fb,
            "away_batting_fb": away_batting_fb,
            "platoon_adj_home": 0.0,
            "platoon_adj_away": 0.0,
        })

    result = pd.DataFrame(rows)

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_csv, index=False)
        print(f"\nWrote {len(result)} rows → {out_csv}", file=sys.stderr)

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve starting pitchers for a day's schedule."
    )
    parser.add_argument(
        "--schedule", type=Path, required=True,
        help="Path to schedule CSV (game_num, home_canonical_id, away_canonical_id)"
    )
    parser.add_argument("--date", type=str, required=True, help="Game date YYYY-MM-DD")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output CSV path (default: data/daily/{date}/starters.csv)"
    )
    parser.add_argument(
        "--pitcher-table", type=Path,
        default=Path("data/processed/pitcher_table.csv")
    )
    parser.add_argument(
        "--team-table", type=Path,
        default=Path("data/processed/team_table.csv")
    )
    parser.add_argument(
        "--appearances", type=Path,
        default=Path("data/processed/pitcher_appearances.csv")
    )
    parser.add_argument(
        "--pitcher-registry", type=Path,
        default=Path("data/processed/pitcher_registry.csv")
    )
    parser.add_argument(
        "--pitcher-index", type=Path,
        default=Path("data/processed/run_event_pitcher_index.csv")
    )
    parser.add_argument(
        "--weekend-rotations", type=Path,
        default=Path("data/processed/weekend_rotations.csv")
    )
    parser.add_argument(
        "--d1b-rotations", type=Path,
        default=Path("data/processed/d1baseball_rotations.csv")
    )
    parser.add_argument(
        "--canonical", type=Path,
        default=Path("data/registries/canonical_teams_2026.csv")
    )
    args = parser.parse_args()

    out = args.out or Path(f"data/daily/{args.date}/starters.csv")

    resolve_starters(
        schedule_csv=args.schedule,
        pitcher_table_csv=args.pitcher_table,
        team_table_csv=args.team_table,
        appearances_csv=args.appearances,
        pitcher_registry_csv=args.pitcher_registry,
        pitcher_index_csv=args.pitcher_index,
        weekend_rotations_csv=args.weekend_rotations,
        d1b_rotations_csv=args.d1b_rotations,
        canonical_csv=args.canonical,
        date=args.date,
        out_csv=out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
