#!/usr/bin/env python3
"""Merge ESPN run_events with linescore-derived run_events.

Reads the ESPN-only run_events.csv (from extract_espn.py) and the
linescore-derived run_events_from_linescores.csv, deduplicates on event_id
(preferring ESPN where both exist), and writes a unified run_events.csv.

Ensures consistent columns — linescore rows won't have pitcher IDs and that's OK.
"""
import sys
from pathlib import Path

import pandas as pd

CORE_COLS = [
    "event_id", "game_date", "season",
    "home_canonical_id", "away_canonical_id",
    "home_pitcher_espn_id", "away_pitcher_espn_id",
    "home_run_1", "home_run_2", "home_run_3", "home_run_4",
    "away_run_1", "away_run_2", "away_run_3", "away_run_4",
    "home_score", "away_score",
]


def main():
    espn_csv = Path("data/processed/run_events.csv")
    ls_csv = Path("data/processed/run_events_from_linescores.csv")
    out_csv = Path("data/processed/run_events.csv")

    espn = pd.read_csv(espn_csv, dtype=str)
    n_before = len(espn)

    if not ls_csv.exists():
        print(f"No linescores file at {ls_csv}, nothing to merge.", file=sys.stderr)
        return

    ls = pd.read_csv(ls_csv, dtype=str)

    # Rename linescore pitcher columns to match ESPN convention
    if "home_pitcher_id" in ls.columns and "home_pitcher_espn_id" not in ls.columns:
        ls = ls.rename(columns={
            "home_pitcher_id": "home_pitcher_espn_id",
            "away_pitcher_id": "away_pitcher_espn_id",
        })

    # Ensure all core columns exist in both
    for c in CORE_COLS:
        if c not in espn.columns:
            espn[c] = ""
        if c not in ls.columns:
            ls[c] = ""

    # Keep only core columns to avoid schema drift
    espn = espn[CORE_COLS]
    ls = ls[CORE_COLS]

    # Concat — ESPN first so dedup keeps ESPN version
    combined = pd.concat([espn, ls], ignore_index=True)
    combined = combined.drop_duplicates(subset=["event_id"], keep="first")
    combined.to_csv(out_csv, index=False)

    n_new = len(combined) - n_before
    print(
        f"Run events: {n_before} ESPN + {len(ls)} linescores "
        f"→ {len(combined)} combined ({n_new} new)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
