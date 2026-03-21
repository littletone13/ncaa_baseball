#!/usr/bin/env python3
"""
backtest_posterior.py — Backtest a posterior against known game outcomes.

Takes games from games.csv, looks up team/pitcher indices, and simulates
using the full Stan posterior (with NegBin for run_1/2, Poisson for run_3/4,
and SCORING_CALIBRATION). Reports Brier, LogLoss, MAE, bias, correlation.

Usage:
  python3 scripts/backtest_posterior.py
  python3 scripts/backtest_posterior.py --date-range 2026-03-01:2026-03-15
  python3 scripts/backtest_posterior.py --N 1000  # fewer draws (faster)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCORING_CALIBRATION = 0.325  # Must match simulate.py


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest posterior on known games")
    parser.add_argument("--posterior", type=Path, default=Path("data/processed/run_event_posterior_2k.csv"))
    parser.add_argument("--meta", type=Path, default=Path("data/processed/run_event_fit_meta.json"))
    parser.add_argument("--games", type=Path, default=Path("data/processed/games.csv"))
    parser.add_argument("--team-index", type=Path, default=Path("data/processed/run_event_team_index.csv"))
    parser.add_argument("--pitcher-index", type=Path, default=Path("data/processed/run_event_pitcher_index.csv"))
    parser.add_argument("--date-range", default="2026-02-14:2026-03-15")
    parser.add_argument("--N", type=int, default=1000, help="Simulations per game")
    parser.add_argument("--tune-calibration", action="store_true")
    args = parser.parse_args()

    # Parse date range
    start_str, end_str = args.date_range.split(":")
    start_date = pd.Timestamp(start_str)
    end_date = pd.Timestamp(end_str)

    # Load posterior
    print("Loading posterior...", file=sys.stderr)
    meta = json.load(open(args.meta))
    N_teams = meta["N_teams"]
    N_pitchers = meta["N_pitchers"]
    draws_df = pd.read_csv(args.posterior)
    n_draws = len(draws_df)
    print(f"  {n_draws} draws, {N_teams} teams, {N_pitchers} pitchers", file=sys.stderr)

    # Extract arrays
    int_run = np.zeros((n_draws, 4))
    theta_run = np.zeros((n_draws, 2))
    for k in range(4):
        int_run[:, k] = draws_df[f"int_run_{k+1}"].values + SCORING_CALIBRATION
    for k in range(2):
        theta_run[:, k] = draws_df[f"theta_run_{k+1}"].values
    home_adv = draws_df["home_advantage"].values

    att = np.zeros((n_draws, N_teams + 1, 4))
    def_ = np.zeros((n_draws, N_teams + 1, 4))
    for k in range(4):
        for t in range(1, N_teams + 1):
            col_a = f"att_run_{k+1}[{t}]"
            col_d = f"def_run_{k+1}[{t}]"
            if col_a in draws_df.columns:
                att[:, t, k] = draws_df[col_a].values
            if col_d in draws_df.columns:
                def_[:, t, k] = draws_df[col_d].values

    pitcher_ab = np.zeros((n_draws, N_pitchers + 1))
    for p in range(1, N_pitchers + 1):
        col = f"pitcher_ability[{p}]"
        if col in draws_df.columns:
            pitcher_ab[:, p] = draws_df[col].values
    del draws_df

    # Load indices
    team_idx_df = pd.read_csv(args.team_index)
    team_map = dict(zip(team_idx_df["canonical_id"], team_idx_df["team_idx"]))

    pitcher_idx_df = pd.read_csv(args.pitcher_index, dtype=str)
    pitcher_map: dict[str, int] = {}
    for _, r in pitcher_idx_df.iterrows():
        pid = str(r["pitcher_espn_id"]).strip()
        if pid and pid != "unknown":
            pitcher_map[pid] = int(r["pitcher_idx"])

    # Load games
    games = pd.read_csv(args.games, dtype=str)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
    games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")
    mask = (
        (games["game_date"] >= start_date)
        & (games["game_date"] <= end_date)
        & games["home_score"].notna()
    )
    subset = games[mask].copy()
    subset["actual_total"] = subset["home_score"] + subset["away_score"]
    subset["home_win"] = (subset["home_score"] > subset["away_score"]).astype(int)
    print(f"Games in range: {len(subset)}", file=sys.stderr)

    # Simulate
    rng = np.random.default_rng(42)
    results = []

    for _, g in subset.iterrows():
        h_tidx = team_map.get(g["home_canonical_id"], 0)
        a_tidx = team_map.get(g["away_canonical_id"], 0)
        if h_tidx == 0 or a_tidx == 0:
            continue

        hp_id = str(g.get("home_pitcher_espn_id", "")).strip().replace(".0", "")
        ap_id = str(g.get("away_pitcher_espn_id", "")).strip().replace(".0", "")
        h_pidx = min(pitcher_map.get(hp_id, 0), N_pitchers)
        a_pidx = min(pitcher_map.get(ap_id, 0), N_pitchers)

        draw_indices = rng.choice(n_draws, size=args.N, replace=True)
        home_wins = 0
        totals = []

        for d in draw_indices:
            h_score = 0
            a_score = 0

            p_away = pitcher_ab[d, a_pidx] if a_pidx > 0 else 0.0
            p_home = pitcher_ab[d, h_pidx] if h_pidx > 0 else 0.0
            ha = home_adv[d]

            for k in range(4):
                # Home scoring
                log_rate_h = (
                    int_run[d, k]
                    + att[d, h_tidx, k]
                    + def_[d, a_tidx, k]
                    + ha
                    + p_away
                )
                # Away scoring
                log_rate_a = (
                    int_run[d, k]
                    + att[d, a_tidx, k]
                    + def_[d, h_tidx, k]
                    + p_home
                )

                rate_h = np.exp(np.clip(log_rate_h, -5, 5))
                rate_a = np.exp(np.clip(log_rate_a, -5, 5))

                if k < 2:
                    # NegBin for run_1, run_2
                    theta = theta_run[d, k]
                    if theta > 0 and rate_h > 0:
                        p_nb = theta / (theta + rate_h)
                        h_score += rng.negative_binomial(theta, p_nb)
                    else:
                        h_score += rng.poisson(rate_h)
                    if theta > 0 and rate_a > 0:
                        p_nb = theta / (theta + rate_a)
                        a_score += rng.negative_binomial(theta, p_nb)
                    else:
                        a_score += rng.poisson(rate_a)
                else:
                    # Poisson for run_3, run_4
                    h_score += rng.poisson(rate_h)
                    a_score += rng.poisson(rate_a)

            if h_score > a_score:
                home_wins += 1
            totals.append(h_score + a_score)

        results.append({
            "home_cid": g["home_canonical_id"],
            "away_cid": g["away_canonical_id"],
            "home_win_prob": home_wins / args.N,
            "exp_total": np.mean(totals),
            "actual_total": float(g["actual_total"]),
            "home_win": int(g["home_win"]),
            "h_tidx": h_tidx,
            "a_tidx": a_tidx,
            "h_pidx": h_pidx,
            "a_pidx": a_pidx,
        })

    df = pd.DataFrame(results)
    print(f"Simulated: {len(df)} games", file=sys.stderr)

    if df.empty:
        print("No games simulated.", file=sys.stderr)
        return 1

    # Metrics
    hw_prob = df["home_win_prob"].values
    hw_actual = df["home_win"].values.astype(float)
    exp_total = df["exp_total"].values
    act_total = df["actual_total"].values

    brier = float(np.mean((hw_prob - hw_actual) ** 2))
    eps = 1e-8
    logloss = -float(
        np.mean(
            hw_actual * np.log(np.clip(hw_prob, eps, 1 - eps))
            + (1 - hw_actual) * np.log(np.clip(1 - hw_prob, eps, 1 - eps))
        )
    )
    mae = float(np.mean(np.abs(exp_total - act_total)))
    bias = float(np.mean(exp_total - act_total))
    corr = float(np.corrcoef(exp_total, act_total)[0, 1]) if len(df) > 2 else 0
    rmse = float(np.sqrt(np.mean((exp_total - act_total) ** 2)))

    print(f"\n{'='*60}")
    print(f"  BACKTEST REPORT — {len(df)} games ({start_str} to {end_str})")
    print(f"{'='*60}")
    print(f"\n  WIN PROBABILITY:")
    print(f"    Brier Score: {brier:.4f}  (coin flip = 0.25)")
    print(f"    Log Loss:    {logloss:.4f}  (coin flip = 0.693)")
    print(f"    Home win: actual={hw_actual.mean():.1%}  predicted={hw_prob.mean():.1%}")

    # Calibration bins
    bins = [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]
    print(f"\n    {'Bin':>10s} {'N':>5s} {'Pred':>7s} {'Actual':>7s} {'Gap':>7s}")
    print(f"    {'-'*40}")
    for lo, hi in bins:
        mask = (hw_prob >= lo) & (hw_prob < hi)
        n = mask.sum()
        if n == 0:
            continue
        pred_mean = hw_prob[mask].mean()
        act_mean = hw_actual[mask].mean()
        gap = act_mean - pred_mean
        label = f"{lo*100:.0f}-{hi*100:.0f}%"
        print(f"    {label:>10s} {n:>5d} {pred_mean:>6.1%} {act_mean:>6.1%} {gap:>+6.1%}")

    print(f"\n  TOTAL SCORING:")
    print(f"    MAE:  {mae:.2f} runs")
    print(f"    RMSE: {rmse:.2f} runs")
    print(f"    Bias: {bias:+.2f} runs")
    print(f"    Corr: {corr:.3f}")
    print(f"    Avg actual: {act_total.mean():.2f}  Avg predicted: {exp_total.mean():.2f}")

    if args.tune_calibration and abs(bias) > 0.3:
        adj = np.log(act_total.mean() / exp_total.mean())
        new_cal = SCORING_CALIBRATION + adj
        print(f"\n  ⚠ CALIBRATION ADJUSTMENT:")
        print(f"    Current: {SCORING_CALIBRATION:.4f}")
        print(f"    Shift:   {adj:+.4f}")
        print(f"    New:     {new_cal:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
