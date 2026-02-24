"""
Concatenate all roster_2026_*.csv files in data/raw/rosters into one CSV.

Use after scrape_rosters_2026.py has written per-team roster files (or after
importing rosters from another source into that directory).

Usage:
  python3 scripts/export_rosters_2026.py --roster-dir data/raw/rosters --out data/processed/rosters/rosters_2026.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

def main() -> int:
    parser = argparse.ArgumentParser(description="Export single rosters_2026.csv from per-team roster CSVs.")
    parser.add_argument("--roster-dir", type=Path, default=Path("data/raw/rosters"), help="Directory with roster_2026_*.csv")
    parser.add_argument("--out", type=Path, default=Path("data/processed/rosters/rosters_2026.csv"), help="Output CSV")
    args = parser.parse_args()

    files = sorted(args.roster_dir.glob("roster_2026_*.csv"))
    if not files:
        print(f"No roster_2026_*.csv files in {args.roster_dir}. Run scrape_rosters_2026.py first (or copy rosters there).")
        return 1

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Skip {f.name}: {e}")
    if not dfs:
        return 1
    out = pd.concat(dfs, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows from {len(dfs)} files -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
