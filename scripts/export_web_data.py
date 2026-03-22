#!/usr/bin/env python3
"""
export_web_data.py — Export prediction + odds data as JSON for the web dashboard.

Reads predictions CSV + odds JSONL + canonical teams, joins them, and writes:
  - web/public/data/predictions-YYYY-MM-DD.json
  - web/public/data/manifest.json
  - web/public/data/teams.json (canonical_id → odds_api_name)

Usage:
  python3 scripts/export_web_data.py --date 2026-03-22
  python3 scripts/export_web_data.py --date 2026-03-22 --out web/public/data/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


KEEP_COLS = [
    "game_num", "away", "home", "home_cid", "away_cid",
    "home_win_prob", "away_win_prob", "ml_home", "ml_away",
    "exp_home", "exp_away", "exp_total",
    "home_starter", "away_starter", "hp_throws", "ap_throws",
    "temp_f", "wind_mph", "wind_out_mph", "weather_adj", "park_factor",
    "rain_chance_pct",
    "home_bullpen_adj", "away_bullpen_adj",
    "home_wrc_adj", "away_wrc_adj",
    "over_prob",
    "home_win_by_2plus", "away_win_by_2plus",
    "exp_total_p10", "exp_total_p50", "exp_total_p90",
]


def load_odds(odds_jsonl: Path) -> list[dict]:
    """Load odds JSONL and return list of game dicts."""
    if not odds_jsonl.exists():
        return []
    games = []
    with open(odds_jsonl) as f:
        for line in f:
            games.append(json.loads(line))
    return games


def build_team_lookup(canonical_csv: Path) -> dict[str, str]:
    """Build canonical_id → odds_api_name lookup."""
    canon = pd.read_csv(canonical_csv, dtype=str)
    lookup = {}
    for _, r in canon.iterrows():
        cid = str(r.get("canonical_id", "")).strip()
        oname = str(r.get("odds_api_name", "")).strip()
        if cid and oname and oname != "nan":
            lookup[cid] = oname
    return lookup


def load_opening_lines(odds_log: Path, game_date: str) -> dict[tuple[str, str], dict]:
    """Load earliest odds for each game from the append-only log for opening lines."""
    openers: dict[tuple[str, str], dict] = {}
    if not odds_log.exists():
        return openers

    with open(odds_log) as f:
        for line in f:
            rec = json.loads(line)
            # Only consider records from the same day or day before
            ct = rec.get("commence_time", "")
            if not ct:
                continue
            ct_date = ct[:10]
            if ct_date != game_date:
                continue

            key = (rec.get("home_team", ""), rec.get("away_team", ""))
            if key in openers:
                continue  # Keep the first (earliest) snapshot

            # Extract opening ML and total
            best_h_ml = None
            best_a_ml = None
            total_line = None
            for bm in rec.get("bookmaker_lines", []):
                for mkt in bm.get("markets", []):
                    if mkt["key"] == "h2h":
                        for o in mkt["outcomes"]:
                            if o["name"] == key[0]:  # home
                                if best_h_ml is None or o["price"] > best_h_ml:
                                    best_h_ml = o["price"]
                            elif o["name"] == key[1]:  # away
                                if best_a_ml is None or o["price"] > best_a_ml:
                                    best_a_ml = o["price"]
                    if mkt["key"] == "totals":
                        for o in mkt["outcomes"]:
                            if o["name"] == "Over" and total_line is None:
                                total_line = o.get("point")

            openers[key] = {
                "open_home_ml": best_h_ml,
                "open_away_ml": best_a_ml,
                "open_total_line": total_line,
            }

    return openers


def merge_odds_into_predictions(
    games: list[dict], odds: list[dict], team_lookup: dict[str, str],
    opening_lines: dict[tuple[str, str], dict] | None = None,
) -> list[dict]:
    """Merge odds data into prediction game dicts by matching team names."""
    # Build reverse lookup: odds_api_name → odds game
    odds_by_teams: dict[tuple[str, str], dict] = {}
    for og in odds:
        key = (og.get("home_team", ""), og.get("away_team", ""))
        odds_by_teams[key] = og

    for game in games:
        home_odds_name = team_lookup.get(game.get("home_cid", ""), "")
        away_odds_name = team_lookup.get(game.get("away_cid", ""), "")
        if not home_odds_name or not away_odds_name:
            continue

        odds_game = odds_by_teams.get((home_odds_name, away_odds_name))
        if not odds_game:
            continue

        # Find best ML
        best_h_ml = None
        best_a_ml = None
        total_line = None
        h_spread = None
        a_spread = None

        for bm in odds_game.get("bookmaker_lines", []):
            for mkt in bm.get("markets", []):
                if mkt["key"] == "h2h":
                    for o in mkt["outcomes"]:
                        if o["name"] == home_odds_name:
                            if best_h_ml is None or o["price"] > best_h_ml:
                                best_h_ml = o["price"]
                        if o["name"] == away_odds_name:
                            if best_a_ml is None or o["price"] > best_a_ml:
                                best_a_ml = o["price"]
                if mkt["key"] == "totals":
                    for o in mkt["outcomes"]:
                        if o["name"] == "Over" and total_line is None:
                            total_line = o.get("point")
                if mkt["key"] == "spreads":
                    for o in mkt["outcomes"]:
                        if o["name"] == home_odds_name and h_spread is None:
                            h_spread = o.get("point")
                        if o["name"] == away_odds_name and a_spread is None:
                            a_spread = o.get("point")

        game["mkt_home_ml"] = best_h_ml
        game["mkt_away_ml"] = best_a_ml
        game["mkt_total_line"] = total_line
        game["mkt_home_spread"] = h_spread
        game["mkt_away_spread"] = a_spread
        game["commence_time"] = odds_game.get("commence_time")
        game["odds"] = odds_game.get("bookmaker_lines", [])

        # Opening lines
        if opening_lines:
            odds_key = (home_odds_name, away_odds_name)
            openers = opening_lines.get(odds_key, {})
            game["open_home_ml"] = openers.get("open_home_ml")
            game["open_away_ml"] = openers.get("open_away_ml")
            game["open_total_line"] = openers.get("open_total_line")

    return games


def main() -> int:
    parser = argparse.ArgumentParser(description="Export web data JSON")
    parser.add_argument("--date", required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Predictions CSV (default: data/processed/predictions_DATE.csv)",
    )
    parser.add_argument(
        "--odds",
        type=Path,
        default=Path("data/raw/odds/odds_latest.jsonl"),
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=Path("data/registries/canonical_teams_2026.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "hoopsbracketanalysis" / "public" / "data",
    )
    args = parser.parse_args()

    if args.predictions is None:
        args.predictions = Path(f"data/processed/predictions_{args.date}.csv")

    if not args.predictions.exists():
        print(f"Predictions not found: {args.predictions}", file=sys.stderr)
        return 1

    # Load predictions
    pred = pd.read_csv(args.predictions)
    cols = [c for c in KEEP_COLS if c in pred.columns]
    pred = pred[cols].copy()

    # Convert to list of dicts with proper types
    games = []
    for _, row in pred.iterrows():
        g = {}
        for col in cols:
            val = row[col]
            if pd.isna(val):
                g[col] = None
            elif isinstance(val, (int, float)):
                g[col] = round(float(val), 4) if isinstance(val, float) else int(val)
            else:
                g[col] = str(val)
        games.append(g)

    # Load and merge odds
    odds = load_odds(args.odds)
    team_lookup = build_team_lookup(args.canonical)

    # Load opening lines from odds log
    odds_log = Path("data/raw/odds/odds_pull_log.jsonl")
    opening_lines = load_opening_lines(odds_log, args.date)
    print(f"Opening lines found for {len(opening_lines)} games", file=sys.stderr)

    games = merge_odds_into_predictions(games, odds, team_lookup, opening_lines)

    # Write predictions JSON
    args.out.mkdir(parents=True, exist_ok=True)
    pred_path = args.out / f"predictions-{args.date}.json"
    with open(pred_path, "w") as f:
        json.dump(games, f, separators=(",", ":"))
    print(f"Wrote {len(games)} games → {pred_path}", file=sys.stderr)

    # Write/update manifest
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"latest": args.date, "dates": []}

    if args.date not in manifest["dates"]:
        manifest["dates"].append(args.date)
        manifest["dates"].sort()
    manifest["latest"] = args.date

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Updated manifest → {manifest_path}", file=sys.stderr)

    # Write teams lookup
    teams_path = args.out / "teams.json"
    with open(teams_path, "w") as f:
        json.dump(team_lookup, f, separators=(",", ":"))
    print(f"Wrote {len(team_lookup)} teams → {teams_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
