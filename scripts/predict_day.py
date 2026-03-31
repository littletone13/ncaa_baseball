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
from bullpen_fatigue import compute_bullpen_fatigue
from simulate import simulate_games, format_predictions
from build_calibration_report import build_calibration_report
from build_starter_qa_report import build_starter_qa_report
from scrape_wrrundown import build_url, scrape_page, parse_wrrundown, write_csv as write_wrrundown_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily NCAA baseball predictions.")
    parser.add_argument("--date", required=True, help="Game date YYYY-MM-DD")
    parser.add_argument("--N", type=int, default=5000, help="Simulations per game")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--phase",
        type=str,
        default="standard",
        choices=["early", "refresh", "standard"],
        help="Execution phase for early deploy and scheduled refresh runs.",
    )
    parser.add_argument("--out", type=Path, help="Output CSV path")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument("--no-weather", action="store_true", help="Skip weather API")
    parser.add_argument(
        "--include-started",
        action="store_true",
        help="Include games that have already started (default drops started/in-progress).",
    )
    parser.add_argument(
        "--start-buffer-min",
        type=int,
        default=15,
        help="Minutes of buffer when filtering started games (default: 15).",
    )
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
    parser.add_argument(
        "--calibration-out",
        type=Path,
        default=None,
        help="Calibration CSV output path (default: data/processed/calibration_{date}_{phase}.csv)",
    )
    parser.add_argument(
        "--calibration-md-out",
        type=Path,
        default=None,
        help="Calibration markdown output path (default: data/processed/calibration_{date}_{phase}.md)",
    )
    parser.add_argument("--ha-target", type=float, default=0.0,
                        help="Target home_advantage mean (post-hoc correction). "
                             "0 = use learned posterior (recommended, NCAA HA is ~0.115). "
                             "Set >0 to override.")
    args = parser.parse_args()
    user_set_n = "--N" in sys.argv
    if not user_set_n:
        if args.phase == "early":
            args.N = 2000
        elif args.phase == "refresh":
            args.N = 4000

    daily_dir = Path(f"data/daily/{args.date}")
    daily_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 0: Pull fresh odds (so market anchor fires) ──
    import os, subprocess
    odds_key = os.environ.get("ODDS_API_KEY") or os.environ.get("THE_ODDS_API_KEY", "")
    if not odds_key:
        # Try .env file
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("ODDS_API_KEY="):
                    odds_key = line.split("=", 1)[1].strip()
    if odds_key:
        print("Step 0: Pulling fresh odds for market anchor...", file=sys.stderr)
        try:
            r = subprocess.run(
                [sys.executable, str(Path(__file__).parent / "pull_odds.py"),
                 "--mode", "current", "--regions", "us,us2,eu",
                 "--markets", "h2h,totals,spreads"],
                capture_output=True, text=True, timeout=30,
                env={**os.environ, "ODDS_API_KEY": odds_key},
            )
            for line in r.stderr.strip().split("\n")[-2:]:
                print(f"  {line}", file=sys.stderr)
        except Exception as e:
            print(f"  Odds pull failed (non-fatal): {e}", file=sys.stderr)
    else:
        print("Step 0: No ODDS_API_KEY found, skipping odds pull", file=sys.stderr)

    # ── Step 1: Schedule ──
    print(f"Step 1/4: Resolving schedule for {args.date}...", file=sys.stderr)
    schedule_csv = daily_dir / "schedule.csv"
    schedule = resolve_schedule(
        date=args.date,
        team_table_csv=args.team_table,
        canonical_csv=args.canonical,
        drop_started=not args.include_started,
        start_buffer_min=args.start_buffer_min,
        out_csv=schedule_csv,
    )
    n_games = len(schedule)
    print(f"  {n_games} games found", file=sys.stderr)
    if n_games == 0:
        print("No games found.", file=sys.stderr)
        return 0

    # ── Step 2: Starters ──
    print("Step 2/5: Resolving starters...", file=sys.stderr)
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

    # ── Step 2b: Starter QA report ──
    starter_qa_csv = Path(f"data/processed/starter_qa_{args.date}_{args.phase}.csv")
    starter_qa_md = Path(f"data/processed/starter_qa_{args.date}_{args.phase}.md")
    build_starter_qa_report(
        starters_csv=starters_csv,
        out_csv=starter_qa_csv,
        out_md=starter_qa_md,
    )
    print(f"Starter QA report -> {starter_qa_csv}", file=sys.stderr)

    # ── Step 2c: WR Rundown intel (sharp handicapper picks + analysis) ──
    wrrundown_csv = daily_dir / "wrrundown_intel.csv"
    try:
        url = build_url(args.date)
        cache_path = Path(".firecrawl") / f"wrrundown-{args.date}.md"
        md_text = scrape_page(url, cache_path)
        if md_text:
            wr_picks = parse_wrrundown(md_text)
            if wr_picks:
                write_wrrundown_csv(wr_picks, wrrundown_csv)
                n_with_analysis = sum(1 for p in wr_picks if p.get("analysis"))
                print(f"  WR Rundown: {len(wr_picks)} picks ({n_with_analysis} with write-ups) -> {wrrundown_csv}",
                      file=sys.stderr)
            else:
                print("  WR Rundown: no picks found (page may not have content yet)", file=sys.stderr)
        else:
            print("  WR Rundown: page not available yet", file=sys.stderr)
    except Exception as e:
        print(f"  WR Rundown scrape failed (non-fatal): {e}", file=sys.stderr)

    # ── Step 3: Weather ──
    print("Step 3/5: Fetching weather...", file=sys.stderr)
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

    # ── Step 3b: Bullpen fatigue ──
    fatigue_csv = daily_dir / "fatigue.csv"
    print("Step 3b/5: Computing bullpen fatigue...", file=sys.stderr)
    try:
        fatigue = compute_bullpen_fatigue(
            appearances_csv=args.appearances,
            game_date=args.date,
            window_days=3,
            required_team_ids=set(schedule["home_cid"].astype(str)) | set(schedule["away_cid"].astype(str)),
        )
        fatigue.to_csv(fatigue_csv, index=False)
        n_fatigued = int((fatigue["fatigue_flag"] == 1).sum()) if not fatigue.empty else 0
        print(f"  {len(fatigue)} teams, {n_fatigued} flagged as fatigued", file=sys.stderr)
    except Exception as e:
        print(f"  Fatigue computation failed: {e}", file=sys.stderr)
        fatigue_csv = None

    # ── Step 4: Simulate ──
    print(f"Step 4/5: Simulating ({args.N} draws per game)...", file=sys.stderr)
    ha_target = args.ha_target if args.ha_target > 0 else None
    predictions = simulate_games(
        schedule_csv=schedule_csv,
        starters_csv=starters_csv,
        weather_csv=weather_csv,
        posterior_csv=args.posterior,
        meta_json=args.meta,
        team_table_csv=args.team_table,
        n_sims=args.N,
        seed=args.seed,
        ha_target=ha_target,
        fatigue_csv=fatigue_csv,
    )

    # ── Output ──
    default_pred = Path(f"data/processed/predictions_{args.date}_{args.phase}.csv")
    out_csv = args.out or default_pred
    predictions.to_csv(out_csv, index=False)
    print(f"\nWrote {len(predictions)} predictions -> {out_csv}", file=sys.stderr)

    # ── Step 5: Calibration report (market-coherent checks) ──
    calib_csv = args.calibration_out or Path(f"data/processed/calibration_{args.date}_{args.phase}.csv")
    calib_md = args.calibration_md_out or Path(f"data/processed/calibration_{args.date}_{args.phase}.md")
    build_calibration_report(
        predictions_csv=out_csv,
        out_csv=calib_csv,
        out_md=calib_md,
    )
    print(f"Calibration report -> {calib_csv}", file=sys.stderr)

    if args.json:
        print(json.dumps(predictions.to_dict("records"), indent=2))
    else:
        format_predictions(predictions, args.date)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
