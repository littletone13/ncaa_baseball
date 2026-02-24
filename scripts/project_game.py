"""
Project a single game: Elo + optional SP/bullpen (Mack) + optional market blend (Peabody/NoVig).

Neutral site: set --neutral so home advantage = 0.
With --use-pitchers: applies SP ratings, expected innings, bullpen workload (run build_pitcher_ratings.py first).
With --market-fair-home: blends model with devigged market; use --n-games to weight market more early season.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from ncaa_baseball.phase1 import (
    build_odds_name_to_canonical,
    load_canonical_teams,
    load_ratings,
    resolve_odds_teams,
    win_prob_from_elo,
)
from ncaa_baseball.pitcher_model import (
    get_bullpen_workload,
    get_sp_rating,
    load_bullpen_workload,
    load_pitcher_ratings,
    load_team_pitcher_strength,
    blend_with_market,
    win_prob_with_pitchers,
)

DEFAULT_ELO_IF_MISSING = 1500.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Project one game (Elo + optional SP/bullpen + market blend).")
    parser.add_argument("--team-a", required=True, help="First team (home for formula)")
    parser.add_argument("--team-b", required=True, help="Second team (away)")
    parser.add_argument("--canonical", type=Path, default=Path("data/registries/canonical_teams_2026.csv"))
    parser.add_argument("--ratings", type=Path, default=Path("data/processed/phase1_team_ratings.csv"))
    parser.add_argument("--neutral", action="store_true", help="Home advantage = 0")
    parser.add_argument("--home-advantage-elo", type=float, default=30.0)
    parser.add_argument("--use-pitchers", action="store_true", help="Use SP ratings + bullpen workload")
    parser.add_argument("--game-date", type=str, default="", help="YYYY-MM-DD for bullpen workload")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--home-sp-id", type=str, default="", help="ESPN pitcher id for team A starter")
    parser.add_argument("--away-sp-id", type=str, default="", help="ESPN pitcher id for team B starter")
    parser.add_argument("--pitcher-ratings", type=Path, default=Path("data/processed/pitcher_ratings.csv"))
    parser.add_argument("--team-pitcher-strength", type=Path, default=Path("data/processed/team_pitcher_strength.csv"))
    parser.add_argument("--bullpen-workload", type=Path, default=Path("data/processed/bullpen_workload.csv"))
    parser.add_argument("--market-fair-home", type=float, default=None, help="Devigged home win prob (NoVig) for blend")
    parser.add_argument("--n-games", type=int, default=None, help="Team games played this season (for market blend)")
    args = parser.parse_args()

    canonical = load_canonical_teams(args.canonical)
    name_to_canonical = build_odds_name_to_canonical(canonical)
    home_t, away_t = resolve_odds_teams(
        args.team_a.strip(), args.team_b.strip(), canonical, name_to_canonical,
    )
    if home_t is None or away_t is None:
        print(f"Could not resolve teams: {args.team_a!r} / {args.team_b!r}")
        return 1

    home_id, away_id = home_t[0], away_t[0]
    ratings = load_ratings(args.ratings)
    home_adv = 0.0 if args.neutral else args.home_advantage_elo
    home_rating = ratings.get(str(home_id), DEFAULT_ELO_IF_MISSING)
    away_rating = ratings.get(str(away_id), DEFAULT_ELO_IF_MISSING)

    if str(home_id) not in ratings:
        print(f"  (Team A {args.team_a} has no Elo; using {DEFAULT_ELO_IF_MISSING})")
    if str(away_id) not in ratings:
        print(f"  (Team B {args.team_b} has no Elo; using {DEFAULT_ELO_IF_MISSING})")

    p_a, p_b = win_prob_from_elo(home_rating, away_rating, home_advantage_elo=home_adv)

    if args.use_pitchers and args.pitcher_ratings.exists() and args.team_pitcher_strength.exists():
        pr = load_pitcher_ratings(args.pitcher_ratings)
        ts = load_team_pitcher_strength(args.team_pitcher_strength)
        sp_ra9_home, exp_ip_home = get_sp_rating(
            args.home_sp_id or None, home_id, args.season, pr, ts,
        )
        sp_ra9_away, exp_ip_away = get_sp_rating(
            args.away_sp_id or None, away_id, args.season, pr, ts,
        )
        ip_1d_home, ip_3d_home = 0.0, 0.0
        ip_1d_away, ip_3d_away = 0.0, 0.0
        if args.game_date and args.bullpen_workload.exists():
            wl = load_bullpen_workload(args.bullpen_workload)
            ip_1d_home, ip_3d_home = get_bullpen_workload(home_id, args.game_date, wl)
            ip_1d_away, ip_3d_away = get_bullpen_workload(away_id, args.game_date, wl)
        p_a, p_b = win_prob_with_pitchers(
            home_rating, away_rating,
            sp_ra9_home, sp_ra9_away,
            exp_ip_home, exp_ip_away,
            ip_1d_home, ip_1d_away,
            home_advantage_elo=home_adv,
        )
        print(f"  SP (shrunk RA9): {args.team_a} {sp_ra9_home:.2f} ({exp_ip_home:.1f} IP)  |  {args.team_b} {sp_ra9_away:.2f} ({exp_ip_away:.1f} IP)")
        if ip_1d_home or ip_1d_away:
            print(f"  Bullpen IP last 1d: {args.team_a} {ip_1d_home:.1f}  |  {args.team_b} {ip_1d_away:.1f}")

    if args.market_fair_home is not None and 0 < args.market_fair_home < 1:
        n = args.n_games if args.n_games is not None else 0
        p_a, p_b = blend_with_market(p_a, args.market_fair_home, n)
        print(f"  Blended with market (n_games={n}): {args.team_a} {p_a:.1%}  |  {args.team_b} {p_b:.1%}")

    print(f"  {args.team_a} (Elo {home_rating:.0f})  vs  {args.team_b} (Elo {away_rating:.0f})")
    if args.neutral:
        print("  Neutral site (home advantage = 0)")
    print(f"  Win prob  {args.team_a}: {p_a:.1%}  |  {args.team_b}: {p_b:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
