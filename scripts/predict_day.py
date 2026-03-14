"""
Daily prediction pipeline orchestrator.

Calls four focused modules in sequence:
  1. resolve_schedule  — fetch game schedule from NCAA/ESPN APIs
  2. resolve_starters  — project starters + enrich with ability estimates
  3. resolve_weather   — fetch weather + park factors
  4. simulate          — Monte Carlo simulation (pure math, no API calls)

Usage:
  python3 scripts/predict_day.py --date 2026-03-14
  python3 scripts/predict_day.py --date 2026-03-14 --N 5000 --no-weather
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from resolve_schedule import resolve_schedule
from resolve_starters import resolve_starters
from resolve_weather import resolve_weather
from simulate import simulate_games, format_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily NCAA baseball predictions.")
    parser.add_argument("--date", required=True, help="Game date YYYY-MM-DD")
    parser.add_argument("--N", type=int, default=5000, help="Simulations per game")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, help="Output CSV path")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument("--no-weather", action="store_true", help="Skip weather API")
    # Data file paths
    parser.add_argument("--posterior", type=Path,
                        default=Path("data/processed/run_event_posterior_2k.csv"))
    parser.add_argument("--meta", type=Path,
                        default=Path("data/processed/run_event_fit_meta.json"))
    parser.add_argument("--pitcher-table", type=Path,
                        default=Path("data/processed/pitcher_table.csv"))
    parser.add_argument("--team-table", type=Path,
                        default=Path("data/processed/team_table.csv"))
    parser.add_argument("--canonical", type=Path,
                        default=Path("data/registries/canonical_teams_2026.csv"))
    parser.add_argument("--stadium-csv", type=Path,
                        default=Path("data/registries/stadium_orientations.csv"))
    parser.add_argument("--park-factors", type=Path,
                        default=Path("data/processed/park_factors.csv"))
    # Legacy args (for backward compat — now read from pitcher/team tables)
    parser.add_argument("--team-index", type=Path, default=None)
    parser.add_argument("--pitcher-index", type=Path, default=None)
    parser.add_argument("--appearances", type=Path,
                        default=Path("data/processed/pitcher_appearances.csv"))
    parser.add_argument("--pitcher-registry", type=Path,
                        default=Path("data/processed/pitcher_registry.csv"))
    args = parser.parse_args()

    daily_dir = Path(f"data/daily/{args.date}")
    daily_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Schedule ──
    print(f"Step 1/4: Resolving schedule for {args.date}...", file=sys.stderr)
    schedule_csv = daily_dir / "schedule.csv"
    schedule = resolve_schedule(
        date=args.date,
        team_table_csv=args.team_table,
        canonical_csv=args.canonical,
        out_csv=schedule_csv,
    )
    n_games = len(schedule)
    print(f"  {n_games} games found", file=sys.stderr)
    if n_games == 0:
        print("No games found.", file=sys.stderr)
        return 0

    # ── Step 2: Starters ──
    print("Step 2/4: Resolving starters...", file=sys.stderr)
    starters_csv = daily_dir / "starters.csv"
    resolve_starters(
        schedule_csv=schedule_csv,
        pitcher_table_csv=args.pitcher_table,
        team_table_csv=args.team_table,
        appearances_csv=args.appearances,
        pitcher_registry_csv=args.pitcher_registry,
        canonical_csv=args.canonical,
        date=args.date,
        out_csv=starters_csv,
    )

    # ── Step 3: Weather ──
    print("Step 3/4: Fetching weather...", file=sys.stderr)
    weather_csv = daily_dir / "weather.csv"
    if args.no_weather:
        weather = schedule[["game_num"]].copy()
        for col in ["park_factor", "wind_adj_raw", "non_wind_adj",
                    "wind_out_mph", "wind_out_lf", "wind_out_cf", "wind_out_rf",
                    "temp_f", "wind_mph", "wind_dir_deg", "elevation_ft"]:
            weather[col] = 0.0
        weather["home_cid"] = schedule["home_cid"]
        weather["weather_mode"] = "none"
        weather.to_csv(weather_csv, index=False)
        print("  Skipped (--no-weather)", file=sys.stderr)
    else:
        resolve_weather(
            schedule_csv=schedule_csv,
            stadium_csv=args.stadium_csv,
            park_factors_csv=args.park_factors,
            date=args.date,
            out_csv=weather_csv,
        )

    # ── Step 4: Simulate ──
    print(f"Step 4/4: Simulating ({args.N} draws per game)...", file=sys.stderr)
    predictions = simulate_games(
        schedule_csv=schedule_csv,
        starters_csv=starters_csv,
        weather_csv=weather_csv,
        posterior_csv=args.posterior,
        meta_json=args.meta,
        team_table_csv=args.team_table,
        n_sims=args.N,
        seed=args.seed,
    )

    # ── Output ──
    out_csv = args.out or Path(f"data/processed/predictions_{args.date}.csv")
    predictions.to_csv(out_csv, index=False)
    print(f"\nWrote {len(predictions)} predictions -> {out_csv}", file=sys.stderr)

    if args.json:
        print(json.dumps(predictions.to_dict("records"), indent=2))
    else:
        format_predictions(predictions, args.date)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
