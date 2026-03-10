"""
Simulate one matchup from the run-event posterior (Path A step 6.4).

Implements Mack Ch 18 simulation: draw posterior params, sample run-event counts
via NegBin(run_1/run_2) and Poisson(run_3/run_4), compute total runs as
1*run_1 + 2*run_2 + 3*run_3 + 4*run_4, resolve ties with extra innings
(scaling_factor = 1/9 per inning), output win/runline/total probabilities.

Includes park factors, bullpen quality adjustments, and live weather/wind adjustment.

Reads CmdStanPy-formatted posterior CSV (columns: int_run_k, att_run_k.i,
def_run_k.i, pitcher_ability.i, home_advantage, theta_run_1, theta_run_2,
beta_park, beta_bullpen).

Usage:
  python3 scripts/simulate_run_event_game.py --home-team 13 --away-team 3 --N 10000
  python3 scripts/simulate_run_event_game.py --home-team 13 --away-team 3 --park-factor 0.05
  python3 scripts/simulate_run_event_game.py --home-team 168 --away-team 265 --weather  # live weather
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Optional weather module (for --weather flag)
try:
    from weather_park_adjustment import get_weather_park_adj
except ImportError:
    get_weather_park_adj = None


# ── Sampling helpers ──────────────────────────────────────────────────────────

def _sample_negbin(mu: float, theta: float, rng: np.random.Generator) -> int:
    """Sample from NegBin(mu, theta) parameterization (mean=mu, dispersion=theta)."""
    theta = max(1e-6, float(theta))
    mu = max(1e-8, float(mu))
    p = theta / (theta + mu)
    return int(rng.negative_binomial(n=theta, p=p))


def _sample_poisson(mu: float, rng: np.random.Generator) -> int:
    """Sample from Poisson(mu)."""
    mu = max(1e-8, float(mu))
    return int(rng.poisson(lam=mu))


# ── Parameter extraction from CmdStanPy draws row ────────────────────────────

def _get_vector(draw: pd.Series, prefix: str, n: int) -> np.ndarray:
    """Read vector[n] param: prefix[1], prefix[2], ..., prefix[n] (CmdStan bracket notation)."""
    out = np.zeros(n)
    for i in range(1, n + 1):
        # CmdStan uses bracket notation: prefix[i]
        key = f"{prefix}[{i}]"
        if key in draw.index:
            out[i - 1] = draw[key]
        else:
            # Fallback: dot notation (PyMC compatibility)
            key2 = f"{prefix}.{i}"
            if key2 in draw.index:
                out[i - 1] = draw[key2]
    return out


# ── Core simulation ──────────────────────────────────────────────────────────

def simulate_9_innings(
    draw: pd.Series,
    home_team_idx: int,
    away_team_idx: int,
    home_pitcher_idx: int,
    away_pitcher_idx: int,
    N_teams: int,
    N_pitchers: int,
    rng: np.random.Generator,
    scale: float = 1.0,
    park_factor: float = 0.0,
    home_bp_adj: float = 0.0,
    away_bp_adj: float = 0.0,
) -> tuple[int, int]:
    """
    Simulate run-event counts for one game from one posterior draw.

    Returns (home_total_runs, away_total_runs).
    scale < 1 is used for extra innings (1/9 per inning).
    park_factor: log(adjusted_pf) for game venue (0 = neutral/unknown).
    home_bp_adj: home team's bullpen quality adj (positive = worse bullpen).
    away_bp_adj: away team's bullpen quality adj.
    """
    home_runs = 0
    away_runs = 0

    # Park and bullpen effects from posterior
    beta_park = float(draw.get("beta_park", 1.0))
    beta_bullpen = float(draw.get("beta_bullpen", 0.0))
    park = beta_park * park_factor
    # Bullpen direction: away bullpen quality affects home scoring, vice versa
    bp_h = beta_bullpen * away_bp_adj
    bp_a = beta_bullpen * home_bp_adj

    for k in range(1, 5):
        int_k = float(draw[f"int_run_{k}"])
        # CmdStan bracket notation; fallback to dot for PyMC
        att_h = float(draw.get(f"att_run_{k}[{home_team_idx}]", draw.get(f"att_run_{k}.{home_team_idx}", 0)))
        att_a = float(draw.get(f"att_run_{k}[{away_team_idx}]", draw.get(f"att_run_{k}.{away_team_idx}", 0)))
        def_h = float(draw.get(f"def_run_{k}[{home_team_idx}]", draw.get(f"def_run_{k}.{home_team_idx}", 0)))
        def_a = float(draw.get(f"def_run_{k}[{away_team_idx}]", draw.get(f"def_run_{k}.{away_team_idx}", 0)))
        home_adv = float(draw["home_advantage"])

        # Pitcher effects (0 = unknown -> 0)
        p_away = float(draw.get(f"pitcher_ability[{away_pitcher_idx}]", draw.get(f"pitcher_ability.{away_pitcher_idx}", 0))) if away_pitcher_idx >= 1 else 0.0
        p_home = float(draw.get(f"pitcher_ability[{home_pitcher_idx}]", draw.get(f"pitcher_ability.{home_pitcher_idx}", 0))) if home_pitcher_idx >= 1 else 0.0

        # Log-linear rate (matching Stan model exactly)
        log_lam_h = int_k + att_h + def_a + home_adv + p_away + park + bp_h
        log_lam_a = int_k + att_a + def_h + p_home + park + bp_a

        mu_h = np.exp(log_lam_h) * scale
        mu_a = np.exp(log_lam_a) * scale

        # Sample counts
        if k <= 2:
            theta = float(draw[f"theta_run_{k}"])
            count_h = _sample_negbin(mu_h, theta, rng)
            count_a = _sample_negbin(mu_a, theta, rng)
        else:
            count_h = _sample_poisson(mu_h, rng)
            count_a = _sample_poisson(mu_a, rng)

        # Total runs: count of k-run events * k runs per event
        home_runs += k * count_h
        away_runs += k * count_a

    return home_runs, away_runs


def simulate_full_game(
    draw: pd.Series,
    home_team_idx: int,
    away_team_idx: int,
    home_pitcher_idx: int,
    away_pitcher_idx: int,
    N_teams: int,
    N_pitchers: int,
    rng: np.random.Generator,
    max_extra_innings: int = 20,
    park_factor: float = 0.0,
    home_bp_adj: float = 0.0,
    away_bp_adj: float = 0.0,
) -> tuple[int, int]:
    """Simulate 9 innings + extra innings until tie is broken (Mack Ch 18)."""
    home_runs, away_runs = simulate_9_innings(
        draw, home_team_idx, away_team_idx,
        home_pitcher_idx, away_pitcher_idx,
        N_teams, N_pitchers, rng,
        park_factor=park_factor,
        home_bp_adj=home_bp_adj,
        away_bp_adj=away_bp_adj,
    )

    # Extra innings: scale = 1/9 per inning (Mack Ch 18 section 7)
    extra = 0
    while home_runs == away_runs and extra < max_extra_innings:
        h_extra, a_extra = simulate_9_innings(
            draw, home_team_idx, away_team_idx,
            home_pitcher_idx, away_pitcher_idx,
            N_teams, N_pitchers, rng,
            scale=1.0 / 9.0,
            park_factor=park_factor,
            home_bp_adj=home_bp_adj,
            away_bp_adj=away_bp_adj,
        )
        home_runs += h_extra
        away_runs += a_extra
        extra += 1

    # If still tied after max extra innings, coin flip
    if home_runs == away_runs:
        if rng.random() < 0.5:
            home_runs += 1
        else:
            away_runs += 1

    return home_runs, away_runs


def expected_runs(
    draw: pd.Series,
    home_team_idx: int,
    away_team_idx: int,
    home_pitcher_idx: int,
    away_pitcher_idx: int,
    park_factor: float = 0.0,
    home_bp_adj: float = 0.0,
    away_bp_adj: float = 0.0,
) -> tuple[float, float]:
    """Expected home and away runs (E[mu]) for one posterior draw."""
    beta_park = float(draw.get("beta_park", 1.0))
    beta_bullpen = float(draw.get("beta_bullpen", 0.0))
    park = beta_park * park_factor
    bp_h = beta_bullpen * away_bp_adj
    bp_a = beta_bullpen * home_bp_adj

    exp_home, exp_away = 0.0, 0.0
    for k in range(1, 5):
        int_k = float(draw[f"int_run_{k}"])
        att_h = float(draw.get(f"att_run_{k}[{home_team_idx}]", draw.get(f"att_run_{k}.{home_team_idx}", 0)))
        att_a = float(draw.get(f"att_run_{k}[{away_team_idx}]", draw.get(f"att_run_{k}.{away_team_idx}", 0)))
        def_h = float(draw.get(f"def_run_{k}[{home_team_idx}]", draw.get(f"def_run_{k}.{home_team_idx}", 0)))
        def_a = float(draw.get(f"def_run_{k}[{away_team_idx}]", draw.get(f"def_run_{k}.{away_team_idx}", 0)))
        home_adv = float(draw["home_advantage"])
        p_away = float(draw.get(f"pitcher_ability[{away_pitcher_idx}]", draw.get(f"pitcher_ability.{away_pitcher_idx}", 0))) if away_pitcher_idx >= 1 else 0.0
        p_home = float(draw.get(f"pitcher_ability[{home_pitcher_idx}]", draw.get(f"pitcher_ability.{home_pitcher_idx}", 0))) if home_pitcher_idx >= 1 else 0.0

        mu_h = np.exp(int_k + att_h + def_a + home_adv + p_away + park + bp_h)
        mu_a = np.exp(int_k + att_a + def_h + p_home + park + bp_a)
        exp_home += k * mu_h
        exp_away += k * mu_a
    return exp_home, exp_away


def prob_to_american(p: float) -> int:
    if p <= 0.0:
        return 9999
    if p >= 1.0:
        return -9999
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate one game from run-event posterior (moneyline, runline, total).",
    )
    parser.add_argument("--posterior", type=Path, default=Path("data/processed/run_event_posterior.csv"))
    parser.add_argument("--meta", type=Path, default=Path("data/processed/run_event_fit_meta.json"))
    parser.add_argument("--home-team", type=int, required=True, help="home_team_idx (1..N_teams, 0=league avg)")
    parser.add_argument("--away-team", type=int, required=True, help="away_team_idx (1..N_teams, 0=league avg)")
    parser.add_argument("--home-pitcher", type=int, default=0, help="home_pitcher_idx (0=unknown)")
    parser.add_argument("--away-pitcher", type=int, default=0, help="away_pitcher_idx (0=unknown)")
    parser.add_argument("--home-name", type=str, default="")
    parser.add_argument("--away-name", type=str, default="")
    parser.add_argument("--home-pitcher-name", type=str, default="")
    parser.add_argument("--away-pitcher-name", type=str, default="")
    parser.add_argument("--pitcher-index", type=Path,
                        default=Path("data/processed/run_event_pitcher_index.csv"),
                        help="Pitcher index CSV for name->idx lookup")
    parser.add_argument("--park-factor", type=float, default=0.0,
                        help="log(adjusted_pf) for game venue (0 = neutral/unknown)")
    parser.add_argument("--home-bp-adj", type=float, default=0.0,
                        help="Home team bullpen quality adj (0 = neutral/unknown)")
    parser.add_argument("--away-bp-adj", type=float, default=0.0,
                        help="Away team bullpen quality adj (0 = neutral/unknown)")
    parser.add_argument("--weather", action="store_true",
                        help="Fetch live weather and add wind/temp adjustment to park factor. "
                             "Requires --home-canonical-id to look up stadium location.")
    parser.add_argument("--home-canonical-id", type=str, default="",
                        help="Home team canonical_id for weather lookup (e.g. NCAA_614704)")
    parser.add_argument("--stadium-csv", type=Path,
                        default=Path("data/registries/stadium_orientations.csv"))
    parser.add_argument("--N", type=int, default=10_000, help="Number of simulations (Mack uses 10K)")
    parser.add_argument("--runline", type=float, default=-1.5)
    parser.add_argument("--total", type=float, default=11.5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not args.posterior.exists() or not args.meta.exists():
        print("Run fit_run_event_model.py first to create posterior and meta.")
        return 1

    draws_df = pd.read_csv(args.posterior)
    with open(args.meta) as f:
        meta = json.load(f)
    N_teams = meta["N_teams"]
    N_pitchers = meta["N_pitchers"]

    # ── Pitcher name -> idx lookup (if pitcher index available) ───────────
    # Allows looking up pitchers by their unified ID (e.g. "NCAA_john_smith__BSB_UCLA")
    # when --home-pitcher / --away-pitcher are left at 0
    if args.pitcher_index.exists() and (args.home_pitcher == 0 or args.away_pitcher == 0):
        pi_df = pd.read_csv(args.pitcher_index, dtype=str)
        pid_to_idx: dict[str, int] = {}
        for _, r in pi_df.iterrows():
            pid = str(r.get("pitcher_espn_id", "")).strip()
            idx_val = int(r.get("pitcher_idx", 0))
            if pid and pid.lower() != "unknown":
                pid_to_idx[pid] = idx_val
                # Also allow lookup by lowercase pitcher name portion
                # e.g., "NCAA_john_smith__BSB_UCLA" can match partial "john_smith"
                if pid.startswith("NCAA_") and "__" in pid:
                    name_part = pid.split("__")[0].replace("NCAA_", "")
                    pid_to_idx[name_part] = idx_val

        # If pitcher name provided but no index, try to resolve
        hp_name = args.home_pitcher_name.strip()
        ap_name = args.away_pitcher_name.strip()
        if args.home_pitcher == 0 and hp_name:
            match = pid_to_idx.get(hp_name, 0)
            if match:
                args.home_pitcher = match
                print(f"Resolved home pitcher '{hp_name}' -> idx {match}")
        if args.away_pitcher == 0 and ap_name:
            match = pid_to_idx.get(ap_name, 0)
            if match:
                args.away_pitcher = match
                print(f"Resolved away pitcher '{ap_name}' -> idx {match}")

    for idx, name in [(args.home_team, "home_team"), (args.away_team, "away_team")]:
        if idx < 0 or idx > N_teams:
            print(f"{name} must be in 0..{N_teams} (0 = league average)")
            return 1

    # ── Live weather adjustment ─────────────────────────────────────────────
    weather_adj = 0.0
    if args.weather:
        if get_weather_park_adj is None:
            print("Error: weather_park_adjustment.py not found on import path.")
            print("Run from scripts/ directory or add it to PYTHONPATH.")
            return 1
        cid = args.home_canonical_id.strip()
        if not cid:
            print("Error: --weather requires --home-canonical-id (e.g. NCAA_614704)")
            return 1
        wx = get_weather_park_adj(canonical_id=cid, stadium_csv=args.stadium_csv)
        if "error" in wx:
            print(f"Weather warning: {wx['error']} — using 0 weather adjustment")
        else:
            weather_adj = wx["total_adj"]
            print(f"Weather: {wx.get('venue_name', '')} — "
                  f"{wx['temp_f']:.0f}°F, wind {wx['wind_speed_mph']:.0f} mph "
                  f"from {wx['wind_dir_deg']:.0f}° "
                  f"(out component: {wx['wind_out_mph']:.1f} mph)")
            print(f"Weather adj: {weather_adj:+.4f} (log-rate) → "
                  f"{np.exp(weather_adj):.3f}x run multiplier")

    pf = args.park_factor + weather_adj

    rng = np.random.default_rng(args.seed)
    n_draws = len(draws_df)
    wins_home = 0
    runline_covers = 0
    overs = 0
    home_runline_fav = 0
    away_runline_fav = 0
    exp_home_sum, exp_away_sum = 0.0, 0.0

    h_bp = args.home_bp_adj
    a_bp = args.away_bp_adj

    for _ in range(args.N):
        row_idx = rng.integers(0, n_draws)
        draw = draws_df.iloc[row_idx]

        eh, ea = expected_runs(
            draw, args.home_team, args.away_team,
            args.home_pitcher, args.away_pitcher,
            park_factor=pf, home_bp_adj=h_bp, away_bp_adj=a_bp,
        )
        exp_home_sum += eh
        exp_away_sum += ea

        home_runs, away_runs = simulate_full_game(
            draw, args.home_team, args.away_team,
            args.home_pitcher, args.away_pitcher,
            N_teams, N_pitchers, rng,
            park_factor=pf, home_bp_adj=h_bp, away_bp_adj=a_bp,
        )

        margin = home_runs - away_runs
        if margin > 0:
            wins_home += 1
        if margin > 1.5:
            home_runline_fav += 1
        if margin > -1.5:
            runline_covers += 1
        if margin < -1.5:
            away_runline_fav += 1
        if (home_runs + away_runs) > args.total:
            overs += 1

    N = args.N
    win_prob = wins_home / N
    home_name = args.home_name.strip() or f"team_{args.home_team}"
    away_name = args.away_name.strip() or f"team_{args.away_team}"
    hp_name = args.home_pitcher_name.strip() or ("(unknown)" if args.home_pitcher == 0 else f"pitcher_{args.home_pitcher}")
    ap_name = args.away_pitcher_name.strip() or ("(unknown)" if args.away_pitcher == 0 else f"pitcher_{args.away_pitcher}")

    exp_h = exp_home_sum / N
    exp_a = exp_away_sum / N

    print(f"Matchup: {home_name} vs {away_name}")
    print(f"Starters: {hp_name} (home) | {ap_name} (away)")
    if pf != 0.0:
        print(f"Park factor: {pf:+.4f} (log-scale)")
    if h_bp != 0.0 or a_bp != 0.0:
        print(f"Bullpen adj: home={h_bp:+.4f}, away={a_bp:+.4f}")
    print(f"N_sims = {N}")
    print(f"Win prob (home)      = {win_prob:.4f}  (American: {prob_to_american(win_prob):+d})")
    print(f"Expected runs (home) = {exp_h:.2f}  (away) = {exp_a:.2f}  (total = {exp_h + exp_a:.2f})")
    print(f"Home -1.5 cover      = {home_runline_fav / N:.4f}  (American: {prob_to_american(home_runline_fav / N):+d})")
    print(f"Away -1.5 cover      = {away_runline_fav / N:.4f}  (American: {prob_to_american(away_runline_fav / N):+d})")
    print(f"Over {args.total}          = {overs / N:.4f}  (American: {prob_to_american(overs / N):+d})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
