"""
Fit Elo ratings from games_espn.csv (only games with resolved canonical ids).

Updates ratings sequentially by date. Writes data/processed/phase1_team_ratings.csv
(canonical_id, elo_rating, n_games) for use in Phase 1 odds comparison.

Usage:
  python3 scripts/fit_phase1_elo.py --games data/processed/games_espn.csv --out data/processed/phase1_team_ratings.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Elo parameters
DEFAULT_K = 32
DEFAULT_INITIAL = 1500.0
# Home advantage: add this many points to home rating for expected score (e.g. 30 ~ 54% home when equal)
DEFAULT_HOME_ADVANTAGE = 30.0


def elo_expected(home_rating: float, away_rating: float, home_adv: float) -> float:
    """Expected score for home team (0-1)."""
    return 1.0 / (1.0 + 10.0 ** ((away_rating - (home_rating + home_adv)) / 400.0))


def fit_elo(
    games: pd.DataFrame,
    k: float = DEFAULT_K,
    initial: float = DEFAULT_INITIAL,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
) -> pd.DataFrame:
    """
    Fit Elo sequentially. games must have date, canonical_home_id, canonical_away_id, winner_home.
    Returns DataFrame with canonical_id, elo_rating, n_games.
    """
    games = games.dropna(subset=["canonical_home_id", "canonical_away_id"])
    games = games.sort_values("date").reset_index(drop=True)
    ratings: dict[str, float] = {}
    n_games: dict[str, int] = {}

    def get_rating(cid: str) -> float:
        if cid not in ratings:
            ratings[cid] = initial
            n_games[cid] = 0
        return ratings[cid]

    for _, row in games.iterrows():
        home_id = str(row["canonical_home_id"])
        away_id = str(row["canonical_away_id"])
        winner_home = int(row["winner_home"])
        r_h = get_rating(home_id)
        r_a = get_rating(away_id)
        exp_h = elo_expected(r_h, r_a, home_advantage)
        actual_h = float(winner_home)
        ratings[home_id] = r_h + k * (actual_h - exp_h)
        ratings[away_id] = r_a + k * ((1.0 - actual_h) - (1.0 - exp_h))
        n_games[home_id] = n_games.get(home_id, 0) + 1
        n_games[away_id] = n_games.get(away_id, 0) + 1

    out = pd.DataFrame([
        {"canonical_id": cid, "elo_rating": ratings[cid], "n_games": n_games[cid]}
        for cid in sorted(ratings)
    ])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit Elo from games_espn.csv, write phase1_team_ratings.csv",
    )
    parser.add_argument(
        "--games",
        type=Path,
        default=Path("data/processed/games_espn.csv"),
        help="Games CSV from build_games_from_espn.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/phase1_team_ratings.csv"),
        help="Output ratings CSV",
    )
    parser.add_argument("--k", type=float, default=DEFAULT_K, help="Elo K factor")
    parser.add_argument("--initial", type=float, default=DEFAULT_INITIAL, help="Initial rating")
    parser.add_argument("--home-advantage", type=float, default=DEFAULT_HOME_ADVANTAGE, help="Home Elo bonus")
    args = parser.parse_args()

    if not args.games.exists():
        print(f"Games file not found: {args.games}. Run build_games_from_espn.py first.")
        return 1

    df = pd.read_csv(args.games)
    for col in ("date", "canonical_home_id", "canonical_away_id", "winner_home"):
        if col not in df.columns:
            print(f"Missing column: {col}")
            return 1

    ratings = fit_elo(df, k=args.k, initial=args.initial, home_advantage=args.home_advantage)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(args.out, index=False)
    print(f"Wrote {len(ratings)} team ratings -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
