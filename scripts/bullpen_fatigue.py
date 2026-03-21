#!/usr/bin/env python3
"""
bullpen_fatigue.py — Rolling bullpen workload tracker.

Computes rolling 3-day reliever IP usage per team from pitcher_appearances.csv.
Outputs a fatigue score (z-score of recent bullpen IP) that can be used as a
simulation adjustment: gassed bullpen → opponent scores more.

Usage:
  python3 scripts/bullpen_fatigue.py --date 2026-03-16
  python3 scripts/bullpen_fatigue.py --date 2026-03-16 --window 3 --out data/daily/2026-03-16/fatigue.csv

Output columns:
  canonical_id, team_name, bp_ip_3d, bp_appearances_3d, games_3d,
  bp_ip_per_game_3d, fatigue_z, fatigue_flag
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────

# Threshold for "fatigued" flag (z-score above mean)
FATIGUE_Z_THRESHOLD = 1.5

# Log-rate penalty per fatigue z-score unit
# Rationale: gassed bullpen → ~2-3% more runs allowed per σ of extra usage
FATIGUE_COEFF = 0.015


def compute_bullpen_fatigue(
    appearances_csv: Path,
    game_date: str,
    window_days: int = 3,
    required_team_ids: set[str] | list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute rolling bullpen workload for each team.

    Args:
        appearances_csv: Path to pitcher_appearances.csv
        game_date: Target date (YYYY-MM-DD) — computes workload in the
                   window_days BEFORE this date
        window_days: Number of days to look back (default 3)

    Returns:
        DataFrame with one row per team:
            canonical_id, bp_ip_3d, bp_appearances_3d, games_3d,
            bp_ip_per_game_3d, fatigue_z, fatigue_flag, fatigue_adj
    """
    required = {
        str(t).strip()
        for t in (required_team_ids or [])
        if str(t).strip()
    }
    app = pd.read_csv(appearances_csv, dtype=str)
    app["game_date"] = pd.to_datetime(app["game_date"], errors="coerce")

    # Parse IP (innings pitched) — handle "5.1" = 5⅓, "5.2" = 5⅔ format
    def parse_ip(val: str) -> float:
        try:
            val = str(val).strip()
            if not val or val == "nan":
                return 0.0
            f = float(val)
            whole = int(f)
            frac = f - whole
            # Baseball IP: .1 = ⅓, .2 = ⅔
            if abs(frac - 0.1) < 0.05:
                return whole + 1 / 3
            elif abs(frac - 0.2) < 0.05:
                return whole + 2 / 3
            return f
        except (ValueError, TypeError):
            return 0.0

    ip_col = "ip" if "ip" in app.columns else "ip_raw"
    app["ip_float"] = app[ip_col].apply(parse_ip)

    # Filter to relievers only
    relievers = app[app["role"] == "reliever"].copy()

    # Date window: [game_date - window_days, game_date)
    target = pd.Timestamp(game_date)
    window_start = target - timedelta(days=window_days)
    mask = (relievers["game_date"] >= window_start) & (relievers["game_date"] < target)
    recent = relievers[mask].copy()

    # Also count team games in window (starters + relievers)
    all_recent = app[
        (app["game_date"] >= window_start) & (app["game_date"] < target)
    ]
    games_by_team = (
        all_recent.groupby("team_canonical_id")["game_date"]
        .nunique()
        .to_dict()
    )

    # Aggregate by team
    if recent.empty:
        print(f"  No reliever appearances in {window_days}-day window before {game_date}",
              file=sys.stderr)
        empty = pd.DataFrame(columns=[
            "canonical_id", "bp_ip_3d", "bp_appearances_3d", "games_3d",
            "bp_ip_per_game_3d", "fatigue_z", "fatigue_flag", "fatigue_adj", "fatigue_data_status",
        ])
        if required:
            # Explicitly emit neutral rows for required teams to preserve downstream
            # contract coverage and avoid silent behavior changes.
            base = pd.DataFrame({"canonical_id": sorted(required)})
            base["bp_ip_3d"] = 0.0
            base["bp_appearances_3d"] = 0
            base["games_3d"] = 0
            base["bp_ip_per_game_3d"] = 0.0
            base["fatigue_z"] = 0.0
            base["fatigue_flag"] = 0
            base["fatigue_adj"] = 0.0
            base["fatigue_data_status"] = "imputed_no_window_data"
            return base
        return empty

    team_stats = (
        recent.groupby("team_canonical_id")
        .agg(
            bp_ip_3d=("ip_float", "sum"),
            bp_appearances_3d=("ip_float", "count"),
        )
        .reset_index()
        .rename(columns={"team_canonical_id": "canonical_id"})
    )

    team_stats["games_3d"] = team_stats["canonical_id"].map(games_by_team).fillna(0).astype(int)
    team_stats["bp_ip_per_game_3d"] = np.where(
        team_stats["games_3d"] > 0,
        team_stats["bp_ip_3d"] / team_stats["games_3d"],
        team_stats["bp_ip_3d"],
    )

    # Z-score of bullpen IP usage (relative to all teams in window)
    mean_ip = team_stats["bp_ip_3d"].mean()
    std_ip = team_stats["bp_ip_3d"].std()
    if std_ip > 0:
        team_stats["fatigue_z"] = (team_stats["bp_ip_3d"] - mean_ip) / std_ip
    else:
        team_stats["fatigue_z"] = 0.0

    team_stats["fatigue_flag"] = (team_stats["fatigue_z"] > FATIGUE_Z_THRESHOLD).astype(int)

    # Adjustment: positive = opponent scores more (gassed bullpen hurts)
    # Only apply when fatigue_z > 0 (above-average usage)
    team_stats["fatigue_adj"] = np.where(
        team_stats["fatigue_z"] > 0,
        team_stats["fatigue_z"] * FATIGUE_COEFF,
        0.0,
    )
    team_stats["fatigue_adj"] = team_stats["fatigue_adj"].round(4)
    team_stats["fatigue_data_status"] = "observed_window_data"

    # Fill missing required teams with neutral defaults so downstream simulation
    # has explicit coverage metadata rather than sparse team rows.
    if required:
        observed = set(team_stats["canonical_id"].astype(str))
        missing = sorted(required - observed)
        if missing:
            fill = pd.DataFrame({"canonical_id": missing})
            fill["bp_ip_3d"] = 0.0
            fill["bp_appearances_3d"] = 0
            fill["games_3d"] = 0
            fill["bp_ip_per_game_3d"] = 0.0
            fill["fatigue_z"] = 0.0
            fill["fatigue_flag"] = 0
            fill["fatigue_adj"] = 0.0
            fill["fatigue_data_status"] = "imputed_missing_window_data"
            team_stats = pd.concat([team_stats, fill], ignore_index=True)

    # Round for display
    team_stats["bp_ip_3d"] = team_stats["bp_ip_3d"].round(1)
    team_stats["bp_ip_per_game_3d"] = team_stats["bp_ip_per_game_3d"].round(1)
    team_stats["fatigue_z"] = team_stats["fatigue_z"].round(2)

    return team_stats


def print_fatigue_report(df: pd.DataFrame, date: str, top_n: int = 20) -> None:
    """Print a human-readable fatigue report."""
    if df.empty:
        print("No fatigue data available.")
        return

    print(f"\n{'='*70}")
    print(f"  BULLPEN FATIGUE REPORT — {date}")
    print(f"  {len(df)} teams with reliever appearances in 3-day window")
    print(f"{'='*70}")

    # Most fatigued
    fatigued = df[df["fatigue_flag"] == 1].sort_values("fatigue_z", ascending=False)
    if not fatigued.empty:
        print(f"\n  ⚠️  FATIGUED BULLPENS (z > {FATIGUE_Z_THRESHOLD}):")
        for _, r in fatigued.iterrows():
            print(f"    {r['canonical_id']:<25} {r['bp_ip_3d']:>5.1f} IP in {int(r['games_3d'])} games "
                  f"(z={r['fatigue_z']:+.2f}, adj={r['fatigue_adj']:+.4f})")

    # Top N most worked
    top = df.nlargest(top_n, "bp_ip_3d")
    print(f"\n  {'Team':<25} {'BP IP':>6} {'Apps':>5} {'Games':>6} {'IP/G':>5} {'z':>6} {'Adj':>7} {'Flag':>5}")
    print(f"  {'-'*25} {'-'*6} {'-'*5} {'-'*6} {'-'*5} {'-'*6} {'-'*7} {'-'*5}")
    for _, r in top.iterrows():
        flag = "⚠️" if r["fatigue_flag"] else ""
        print(f"  {r['canonical_id']:<25} {r['bp_ip_3d']:>6.1f} {int(r['bp_appearances_3d']):>5} "
              f"{int(r['games_3d']):>6} {r['bp_ip_per_game_3d']:>5.1f} {r['fatigue_z']:>+6.2f} "
              f"{r['fatigue_adj']:>+7.4f} {flag:>5}")

    # League stats
    print(f"\n  League avg: {df['bp_ip_3d'].mean():.1f} IP, "
          f"median: {df['bp_ip_3d'].median():.1f} IP, "
          f"std: {df['bp_ip_3d'].std():.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute rolling bullpen fatigue from pitcher appearances."
    )
    parser.add_argument("--date", type=str, required=True, help="Game date YYYY-MM-DD")
    parser.add_argument("--window", type=int, default=3, help="Days to look back (default 3)")
    parser.add_argument(
        "--appearances", type=Path,
        default=Path("data/processed/pitcher_appearances.csv"),
    )
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--quiet", action="store_true", help="Suppress report output")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    app_path = repo_root / args.appearances

    fatigue = compute_bullpen_fatigue(app_path, args.date, window_days=args.window)

    if not args.quiet:
        print_fatigue_report(fatigue, args.date)

    if args.out:
        out_path = repo_root / args.out if not args.out.is_absolute() else args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fatigue.to_csv(out_path, index=False)
        print(f"\nWrote {len(fatigue)} rows → {out_path}", file=sys.stderr)
    elif fatigue.empty:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
