"""
Phase 1: Load odds JSONL, resolve teams to canonical, apply model (Elo if ratings exist else prior), compare to market.

Usage:
  python3 scripts/run_phase1_odds_compare.py --odds-jsonl data/raw/odds/odds_baseball_ncaa_20260221.jsonl
  # With fitted Elo ratings (after build_games_from_espn.py + fit_phase1_elo.py):
  python3 scripts/run_phase1_odds_compare.py --ratings data/processed/phase1_team_ratings.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from ncaa_baseball.phase1 import (
    build_odds_name_to_canonical,
    compare_to_market,
    load_canonical_teams,
    load_ratings,
    prior_win_prob,
    resolve_odds_teams,
    win_prob_from_elo,
)


def load_odds_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1: prior-only model vs devigged odds.",
    )
    parser.add_argument(
        "--odds-jsonl",
        type=Path,
        default=Path("data/raw/odds/odds_baseball_ncaa_20260221.jsonl"),
        help="Odds JSONL (one JSON object per game)",
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=Path("data/registries/canonical_teams_2026.csv"),
        help="Canonical teams CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/phase1_compare.csv"),
        help="Output comparison CSV",
    )
    parser.add_argument(
        "--home-advantage",
        type=float,
        default=0.08,
        help="Home advantage in log-odds for prior (default 0.08 ~ 52%% home)",
    )
    parser.add_argument(
        "--ratings",
        type=Path,
        default=Path("data/processed/phase1_team_ratings.csv"),
        help="Elo ratings CSV from fit_phase1_elo.py (optional; use prior if missing)",
    )
    parser.add_argument(
        "--home-advantage-elo",
        type=float,
        default=30.0,
        help="Home advantage in Elo points when using ratings (default 30)",
    )
    args = parser.parse_args()

    canonical = load_canonical_teams(args.canonical)
    name_to_canonical = build_odds_name_to_canonical(canonical)
    ratings = load_ratings(args.ratings)
    use_elo = len(ratings) > 0
    if use_elo:
        print(f"Using Elo ratings from {args.ratings} ({len(ratings)} teams)")
    else:
        print("No ratings file; using prior-only win prob")

    if not args.odds_jsonl.exists():
        print(f"Odds file not found: {args.odds_jsonl}")
        return 1

    games = load_odds_jsonl(args.odds_jsonl)
    p_prior_home, p_prior_away = prior_win_prob(home_advantage_logit=args.home_advantage)

    rows = []
    unresolved = 0
    for g in games:
        event_id = g.get("id") or ""
        home_odds = (g.get("home_team") or "").strip()
        away_odds = (g.get("away_team") or "").strip()
        market_home = g.get("consensus_fair_home")
        market_away = g.get("consensus_fair_away")
        commence = g.get("commence_time") or ""

        home_t, away_t = resolve_odds_teams(
            home_odds, away_odds, canonical, name_to_canonical,
        )
        if home_t is None or away_t is None:
            unresolved += 1
            rows.append({
                "event_id": event_id,
                "commence_time": commence,
                "home_odds_name": home_odds,
                "away_odds_name": away_odds,
                "canonical_home_id": None,
                "canonical_away_id": None,
                "model_win_prob_home": p_prior_home,
                "model_win_prob_away": p_prior_away,
                "market_fair_home": market_home,
                "market_fair_away": market_away,
                "edge_home": None,
                "edge_away": None,
                "resolved": False,
            })
            continue

        home_id = str(home_t[0])
        away_id = str(away_t[0])
        if use_elo and home_id in ratings and away_id in ratings:
            p_home, p_away = win_prob_from_elo(
                ratings[home_id], ratings[away_id], home_advantage_elo=args.home_advantage_elo
            )
        else:
            p_home, p_away = p_prior_home, p_prior_away

        cmp = compare_to_market(p_home, p_away, market_home, market_away)
        rows.append({
            "event_id": event_id,
            "commence_time": commence,
            "home_odds_name": home_odds,
            "away_odds_name": away_odds,
            "canonical_home_id": home_t[0],
            "canonical_away_id": away_t[0],
            "model_win_prob_home": cmp["model_win_prob_home"],
            "model_win_prob_away": cmp["model_win_prob_away"],
            "market_fair_home": cmp["market_fair_home"],
            "market_fair_away": cmp["market_fair_away"],
            "edge_home": cmp["edge_home"],
            "edge_away": cmp["edge_away"],
            "resolved": True,
        })

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(rows)} games -> {args.out}")
    print(f"  Resolved: {len(rows) - unresolved}, Unresolved: {unresolved}")
    if unresolved > 0:
        print("  Unresolved odds names (fill name_crosswalk_manual_2026.csv odds_api_team_name for these):")
        for r in rows:
            if not r["resolved"]:
                print(f"    - {r['home_odds_name']} | {r['away_odds_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
