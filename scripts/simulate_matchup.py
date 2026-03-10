"""
Simulate a matchup by team names (not indices).

High-level wrapper around simulate_run_event_game.py that resolves team names
to indices automatically, optionally looks up park factors, bullpen quality,
and starting pitchers.

Usage:
  python3 scripts/simulate_matchup.py "UCLA" "Ohio State"
  python3 scripts/simulate_matchup.py "LSU" "Alabama" --N 20000
  python3 scripts/simulate_matchup.py "Texas" "Oklahoma" --home-pitcher "ESPN_67525"
  python3 scripts/simulate_matchup.py "Florida" "Georgia" --verbose

  # Batch mode: simulate all games from ESPN scoreboard for a date
  python3 scripts/simulate_matchup.py --date 2026-03-09
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from ncaa_baseball.phase1 import (
    build_odds_name_to_canonical,
    load_canonical_teams,
    resolve_odds_teams,
)

# Import core simulation functions
sys.path.insert(0, str(Path(__file__).parent))
from simulate_run_event_game import (
    simulate_full_game,
    expected_runs,
    prob_to_american,
)


def load_lookups(
    team_index_path: Path,
    pitcher_index_path: Path,
    canonical_path: Path,
    park_factors_path: Path,
    bullpen_quality_path: Path,
) -> dict:
    """Load all lookup tables needed for simulation."""
    # Team index: canonical_id -> team_idx
    team_df = pd.read_csv(team_index_path, dtype=str)
    team_idx_map = {}
    for _, r in team_df.iterrows():
        cid = str(r.get("canonical_id", "")).strip()
        idx = int(r.get("team_idx", 0))
        if cid:
            team_idx_map[cid] = idx

    # Pitcher index: pitcher_id -> pitcher_idx
    pitcher_df = pd.read_csv(pitcher_index_path, dtype=str)
    pitcher_idx_map: dict[str, int] = {}
    for _, r in pitcher_df.iterrows():
        pid = str(r.get("pitcher_espn_id", "")).strip()
        idx = int(r.get("pitcher_idx", 0))
        if pid and pid.lower() != "unknown":
            pitcher_idx_map[pid] = idx

    # Canonical teams for name resolution
    canonical = load_canonical_teams(canonical_path)
    name_to_canonical = build_odds_name_to_canonical(canonical)

    # Build friendly name -> canonical_id map
    name_to_cid: dict[str, str] = {}
    for _, row in canonical.iterrows():
        tname = str(row.get("team_name", "")).strip()
        cid = str(row.get("canonical_id", "")).strip()
        if tname and cid:
            name_to_cid[tname.lower()] = cid
            # Also map short forms
            for word in tname.split():
                w = word.lower().strip()
                if len(w) > 3 and w not in name_to_cid:
                    name_to_cid[w] = cid

    # Park factors: home_team_canonical_id -> log(adjusted_pf)
    pf_map: dict[str, float] = {}
    if park_factors_path.exists():
        pf_df = pd.read_csv(park_factors_path)
        for _, r in pf_df.iterrows():
            htid = str(r.get("home_team_id", "")).strip()
            adj = r.get("adjusted_pf")
            if htid and adj is not None and not (isinstance(adj, float) and math.isnan(adj)):
                pf_map[htid] = math.log(float(adj))

    # Bullpen quality: (canonical_id, season) -> bullpen_adj
    bp_map: dict[tuple[str, int], float] = {}
    if bullpen_quality_path.exists():
        bq_df = pd.read_csv(bullpen_quality_path)
        for _, r in bq_df.iterrows():
            cid = str(r.get("team_canonical_id", "")).strip()
            season = int(r.get("season", 0))
            score = r.get("bullpen_depth_score")
            if cid and season and score is not None and not (isinstance(score, float) and math.isnan(score)):
                bp_map[(cid, season)] = -float(score) * 0.1

    return {
        "team_idx_map": team_idx_map,
        "pitcher_idx_map": pitcher_idx_map,
        "canonical": canonical,
        "name_to_canonical": name_to_canonical,
        "name_to_cid": name_to_cid,
        "pf_map": pf_map,
        "bp_map": bp_map,
    }


def resolve_team_name(name: str, lookups: dict) -> tuple[str, int]:
    """Resolve a team name to (canonical_id, team_idx). Returns ("", 0) if not found."""
    name = name.strip()
    if not name:
        return "", 0

    # Direct lookup by canonical_id
    if name in lookups["team_idx_map"]:
        return name, lookups["team_idx_map"][name]

    # Lookup by name
    name_lower = name.lower()
    cid = lookups["name_to_cid"].get(name_lower)
    if cid and cid in lookups["team_idx_map"]:
        return cid, lookups["team_idx_map"][cid]

    # Try resolve_odds_teams for fuzzy matching
    h_t, a_t = resolve_odds_teams(
        name, "", lookups["canonical"], lookups["name_to_canonical"],
    )
    if h_t:
        cid = h_t[0]
        if cid in lookups["team_idx_map"]:
            return cid, lookups["team_idx_map"][cid]

    # Try partial match on canonical IDs
    for cid, idx in lookups["team_idx_map"].items():
        if name_lower in cid.lower():
            return cid, idx

    return "", 0


def resolve_pitcher(pitcher_str: str, lookups: dict) -> int:
    """Resolve a pitcher string to pitcher_idx. Returns 0 if not found."""
    if not pitcher_str or pitcher_str.strip() == "0":
        return 0
    pitcher_str = pitcher_str.strip()

    # Direct lookup
    if pitcher_str in lookups["pitcher_idx_map"]:
        return lookups["pitcher_idx_map"][pitcher_str]

    # Try with ESPN_ prefix
    if not pitcher_str.startswith("ESPN_") and pitcher_str.isdigit():
        prefixed = f"ESPN_{pitcher_str}"
        if prefixed in lookups["pitcher_idx_map"]:
            return lookups["pitcher_idx_map"][prefixed]

    return 0


def simulate_one_matchup(
    home_name: str,
    away_name: str,
    draws_df: pd.DataFrame,
    meta: dict,
    lookups: dict,
    home_pitcher_str: str = "",
    away_pitcher_str: str = "",
    N: int = 10_000,
    runline: float = -1.5,
    total_line: float = 11.5,
    season: int = 2026,
    seed: int | None = None,
    verbose: bool = False,
) -> dict:
    """Simulate one matchup, return results dict."""
    home_cid, home_idx = resolve_team_name(home_name, lookups)
    away_cid, away_idx = resolve_team_name(away_name, lookups)

    if home_idx == 0:
        if verbose:
            print(f"Warning: could not resolve home team '{home_name}', using league average")
    if away_idx == 0:
        if verbose:
            print(f"Warning: could not resolve away team '{away_name}', using league average")

    home_pitcher_idx = resolve_pitcher(home_pitcher_str, lookups)
    away_pitcher_idx = resolve_pitcher(away_pitcher_str, lookups)

    # Park factor from home team
    pf = lookups["pf_map"].get(home_cid, 0.0)

    # Bullpen quality
    h_bp = lookups["bp_map"].get((home_cid, season), 0.0)
    a_bp = lookups["bp_map"].get((away_cid, season), 0.0)

    N_teams = meta["N_teams"]
    N_pitchers = meta["N_pitchers"]
    n_draws = len(draws_df)
    rng = np.random.default_rng(seed)

    wins_home = 0
    home_rl_cover = 0
    away_rl_cover = 0
    overs = 0
    exp_home_sum, exp_away_sum = 0.0, 0.0
    margin_hist: list[int] = []
    total_hist: list[int] = []

    for _ in range(N):
        row_idx = rng.integers(0, n_draws)
        draw = draws_df.iloc[row_idx]

        eh, ea = expected_runs(
            draw, home_idx, away_idx,
            home_pitcher_idx, away_pitcher_idx,
            park_factor=pf, home_bp_adj=h_bp, away_bp_adj=a_bp,
        )
        exp_home_sum += eh
        exp_away_sum += ea

        home_runs, away_runs = simulate_full_game(
            draw, home_idx, away_idx,
            home_pitcher_idx, away_pitcher_idx,
            N_teams, N_pitchers, rng,
            park_factor=pf, home_bp_adj=h_bp, away_bp_adj=a_bp,
        )

        margin = home_runs - away_runs
        margin_hist.append(margin)
        total_hist.append(home_runs + away_runs)
        if margin > 0:
            wins_home += 1
        if margin + runline > 0:
            home_rl_cover += 1
        if margin - abs(runline) < 0:
            away_rl_cover += 1
        if (home_runs + away_runs) > total_line:
            overs += 1

    win_prob = wins_home / N
    exp_h = exp_home_sum / N
    exp_a = exp_away_sum / N

    return {
        "home_team": home_name,
        "away_team": away_name,
        "home_cid": home_cid,
        "away_cid": away_cid,
        "home_idx": home_idx,
        "away_idx": away_idx,
        "home_pitcher_idx": home_pitcher_idx,
        "away_pitcher_idx": away_pitcher_idx,
        "park_factor": pf,
        "home_bp_adj": h_bp,
        "away_bp_adj": a_bp,
        "N": N,
        "win_prob_home": win_prob,
        "win_prob_away": 1 - win_prob,
        "ml_home_american": prob_to_american(win_prob),
        "ml_away_american": prob_to_american(1 - win_prob),
        "exp_home_runs": exp_h,
        "exp_away_runs": exp_a,
        "exp_total": exp_h + exp_a,
        "runline": runline,
        "home_rl_cover": home_rl_cover / N,
        "away_rl_cover": away_rl_cover / N,
        "total_line": total_line,
        "over_prob": overs / N,
        "under_prob": 1 - overs / N,
    }


def print_matchup_result(r: dict) -> None:
    """Pretty-print simulation results."""
    print(f"\n{'='*60}")
    print(f"  {r['away_team']} @ {r['home_team']}")
    print(f"{'='*60}")
    if r["home_idx"] == 0 or r["away_idx"] == 0:
        print(f"  ⚠️  Unresolved team(s) — using league average")
    print(f"  Park factor: {r['park_factor']:+.4f}" if r['park_factor'] != 0 else "  Park factor: neutral")
    if r['home_pitcher_idx'] > 0 or r['away_pitcher_idx'] > 0:
        print(f"  Pitchers: home_idx={r['home_pitcher_idx']}, away_idx={r['away_pitcher_idx']}")

    print(f"\n  Moneyline:")
    print(f"    {r['home_team']:>25s}  {r['win_prob_home']:.1%}  ({r['ml_home_american']:+d})")
    print(f"    {r['away_team']:>25s}  {r['win_prob_away']:.1%}  ({r['ml_away_american']:+d})")

    print(f"\n  Expected runs:")
    print(f"    {r['home_team']:>25s}  {r['exp_home_runs']:.2f}")
    print(f"    {r['away_team']:>25s}  {r['exp_away_runs']:.2f}")
    print(f"    {'Total':>25s}  {r['exp_total']:.2f}")

    rl = r["runline"]
    print(f"\n  Run line ({rl:+.1f}):")
    print(f"    Home {rl:+.1f}  {r['home_rl_cover']:.1%}  ({prob_to_american(r['home_rl_cover']):+d})")
    print(f"    Away {-rl:+.1f}  {r['away_rl_cover']:.1%}  ({prob_to_american(r['away_rl_cover']):+d})")

    tl = r["total_line"]
    print(f"\n  Total ({tl:.1f}):")
    print(f"    Over   {r['over_prob']:.1%}  ({prob_to_american(r['over_prob']):+d})")
    print(f"    Under  {r['under_prob']:.1%}  ({prob_to_american(r['under_prob']):+d})")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate matchup by team names with auto-resolved indices.",
    )
    parser.add_argument("home_team", nargs="?", default="", help="Home team name")
    parser.add_argument("away_team", nargs="?", default="", help="Away team name")
    parser.add_argument("--home-pitcher", type=str, default="", help="Home pitcher ID")
    parser.add_argument("--away-pitcher", type=str, default="", help="Away pitcher ID")
    parser.add_argument("--N", type=int, default=10_000, help="Simulations per matchup")
    parser.add_argument("--runline", type=float, default=-1.5)
    parser.add_argument("--total", type=float, default=11.5)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")

    # Data paths
    parser.add_argument("--posterior", type=Path, default=Path("data/processed/run_event_posterior.csv"))
    parser.add_argument("--meta", type=Path, default=Path("data/processed/run_event_fit_meta.json"))
    parser.add_argument("--team-index", type=Path, default=Path("data/processed/run_event_team_index.csv"))
    parser.add_argument("--pitcher-index", type=Path, default=Path("data/processed/run_event_pitcher_index.csv"))
    parser.add_argument("--canonical", type=Path, default=Path("data/registries/canonical_teams_2026.csv"))
    parser.add_argument("--park-factors", type=Path, default=Path("data/processed/park_factors.csv"))
    parser.add_argument("--bullpen-quality", type=Path, default=Path("data/processed/bullpen_quality.csv"))

    # Batch mode
    parser.add_argument("--date", type=str, default="",
                        help="Simulate all games from ESPN scoreboard for a date (YYYY-MM-DD)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    # Validate
    if not args.posterior.exists() or not args.meta.exists():
        print("Run fit_run_event_model.py first to create posterior and meta.")
        return 1

    # Load lookups
    print("Loading model and lookups...", file=sys.stderr)
    draws_df = pd.read_csv(args.posterior)
    with open(args.meta) as f:
        meta = json.load(f)
    lookups = load_lookups(
        args.team_index, args.pitcher_index, args.canonical,
        args.park_factors, args.bullpen_quality,
    )
    print(f"  {len(draws_df)} posterior draws, "
          f"{len(lookups['team_idx_map'])} teams, "
          f"{len(lookups['pitcher_idx_map'])} pitchers", file=sys.stderr)

    if args.date:
        # Batch mode: fetch scoreboard and simulate all games
        from urllib.request import Request, urlopen
        dt = args.date.replace("-", "")
        url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/scoreboard?dates={dt}&limit=200"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh)"})
        data = json.loads(urlopen(req, timeout=15).read())
        events = data.get("events", [])
        print(f"\n{len(events)} games on {args.date}", file=sys.stderr)

        results = []
        for e in events:
            comps = e.get("competitions", [{}])
            if not comps:
                continue
            competitors = comps[0].get("competitors", [])
            if len(competitors) != 2:
                continue
            home = next((x for x in competitors if x.get("homeAway") == "home"), None)
            away = next((x for x in competitors if x.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            h_name = home.get("team", {}).get("displayName", "?")
            a_name = away.get("team", {}).get("displayName", "?")

            r = simulate_one_matchup(
                h_name, a_name, draws_df, meta, lookups,
                N=args.N, runline=args.runline, total_line=args.total,
                season=args.season, verbose=args.verbose,
            )
            results.append(r)
            if not args.json:
                print_matchup_result(r)

        if args.json:
            print(json.dumps(results, indent=2))

        return 0

    # Single matchup mode
    if not args.home_team or not args.away_team:
        parser.error("Provide home_team and away_team, or use --date for batch mode")

    r = simulate_one_matchup(
        args.home_team, args.away_team, draws_df, meta, lookups,
        home_pitcher_str=args.home_pitcher,
        away_pitcher_str=args.away_pitcher,
        N=args.N, runline=args.runline, total_line=args.total,
        season=args.season, seed=args.seed, verbose=args.verbose,
    )

    if args.json:
        print(json.dumps(r, indent=2))
    else:
        print_matchup_result(r)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
