"""
Build team and pitcher index CSVs for the Stan run-event model (Path A step 6.2).

Reads run_events.csv, collects unique canonical_id (teams) and pitcher_espn_id (pitchers),
assigns stable integer indices, and writes run_event_team_index.csv and
run_event_pitcher_index.csv. Stan (or fit_run_event_model.py) uses these to map
canonical_id / pitcher_espn_id to 1..N_teams and 0..M_pitchers (0 = unknown starter).

Usage:
  python3 scripts/build_run_event_indices.py --run-events data/processed/run_events.csv
  python3 scripts/build_run_event_indices.py  # uses defaults
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build team and pitcher index CSVs from run_events for Stan.",
    )
    parser.add_argument(
        "--run-events",
        type=Path,
        default=Path("data/processed/run_events.csv"),
        help="Path to run_events.csv from step 6.1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for output index CSVs",
    )
    args = parser.parse_args()

    if not args.run_events.exists():
        print(f"Missing run_events file: {args.run_events}")
        return 1

    df = pd.read_csv(args.run_events, dtype=str)
    for c in ("home_canonical_id", "away_canonical_id", "home_pitcher_espn_id", "away_pitcher_espn_id"):
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()

    # Unique teams (non-empty canonical_id only)
    home_teams = df["home_canonical_id"].loc[df["home_canonical_id"] != ""]
    away_teams = df["away_canonical_id"].loc[df["away_canonical_id"] != ""]
    teams = pd.Series(pd.unique(pd.concat([home_teams, away_teams], ignore_index=True)))
    teams = teams.loc[teams != ""].dropna().sort_values().reset_index(drop=True)
    team_index = pd.DataFrame({"canonical_id": teams, "team_idx": range(1, len(teams) + 1)})

    # Unique pitchers; use 0 for unknown/missing
    home_p = df["home_pitcher_espn_id"].loc[df["home_pitcher_espn_id"].astype(str).str.strip() != ""]
    away_p = df["away_pitcher_espn_id"].loc[df["away_pitcher_espn_id"].astype(str).str.strip() != ""]
    pitchers = pd.Series(pd.unique(pd.concat([home_p, away_p], ignore_index=True)))
    pitchers = pitchers.loc[pitchers.astype(str).str.strip() != ""].dropna()
    pitchers = pitchers.astype(str).str.strip()
    # Sort by numeric value when possible for stable ordering
    num = pd.to_numeric(pitchers, errors="coerce")
    pitchers = pitchers.iloc[num.fillna(num.max() or 0).argsort()].reset_index(drop=True)
    pitcher_index = pd.DataFrame({"pitcher_espn_id": pitchers, "pitcher_idx": range(1, len(pitchers) + 1)})
    # Prepend unknown so Stan can use 0 for missing starter
    unknown = pd.DataFrame([{"pitcher_espn_id": "unknown", "pitcher_idx": 0}])
    pitcher_index = pd.concat([unknown, pitcher_index], ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    team_path = args.out_dir / "run_event_team_index.csv"
    pitcher_path = args.out_dir / "run_event_pitcher_index.csv"
    team_index.to_csv(team_path, index=False)
    pitcher_index.to_csv(pitcher_path, index=False)

    print(f"Wrote {len(team_index)} teams -> {team_path}")
    print(f"Wrote {len(pitcher_index)} pitchers (incl. unknown=0) -> {pitcher_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
