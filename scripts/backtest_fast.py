"""
Fast backtest: pre-extract posterior parameters into numpy arrays for speed.

Same logic as backtest_model.py but 10-50x faster by avoiding pandas Series
lookups during simulation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast backtest of run-event model.")
    parser.add_argument("--run-events", type=Path, default=Path("data/processed/run_events_expanded.csv"))
    parser.add_argument("--posterior", type=Path, default=Path("data/processed/run_event_posterior_2k.csv"))
    parser.add_argument("--meta", type=Path, default=Path("data/processed/run_event_fit_meta.json"))
    parser.add_argument("--team-index", type=Path, default=Path("data/processed/run_event_team_index.csv"))
    parser.add_argument("--pitcher-index", type=Path, default=Path("data/processed/run_event_pitcher_index.csv"))
    parser.add_argument("--park-factors", type=Path, default=Path("data/processed/park_factors.csv"))
    parser.add_argument("--bullpen-quality", type=Path, default=Path("data/processed/bullpen_quality.csv"))
    parser.add_argument("--holdout-days", type=int, default=7)
    parser.add_argument("--N", type=int, default=2000, help="Simulations per game")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    for p in (args.run_events, args.posterior, args.meta, args.team_index, args.pitcher_index):
        if not p.exists():
            print(f"Missing: {p}")
            return 1

    with open(args.meta) as f:
        meta = json.load(f)
    N_teams = meta["N_teams"]
    N_pitchers = meta["N_pitchers"]

    # Load indices
    team_df = pd.read_csv(args.team_index)
    pitcher_df = pd.read_csv(args.pitcher_index, dtype=str)
    team_map = dict(zip(team_df["canonical_id"], team_df["team_idx"]))
    pitcher_map: dict[str, int] = {"unknown": 0, "": 0}
    for _, r in pitcher_df.iterrows():
        pid = str(r.get("pitcher_espn_id", "")).strip()
        if pid and pid.lower() != "unknown":
            pitcher_map[pid] = int(r.get("pitcher_idx", 0))

    # Load park factors
    pf_map: dict[str, float] = {}
    if args.park_factors.exists():
        pf_df = pd.read_csv(args.park_factors)
        for _, r in pf_df.iterrows():
            htid = str(r.get("home_team_id", "")).strip()
            adj = r.get("adjusted_pf")
            if htid and adj is not None and not (isinstance(adj, float) and math.isnan(adj)):
                pf_map[htid] = math.log(float(adj))

    # Load bullpen quality
    bp_map: dict[tuple[str, int], float] = {}
    if args.bullpen_quality.exists():
        bq_df = pd.read_csv(args.bullpen_quality)
        for _, r in bq_df.iterrows():
            cid = str(r.get("team_canonical_id", "")).strip()
            season = int(r.get("season", 0))
            score = r.get("bullpen_depth_score")
            if cid and season and score is not None and not (isinstance(score, float) and math.isnan(score)):
                bp_map[(cid, season)] = -float(score) * 0.1

    # ── Pre-extract posterior parameters into numpy arrays ─────────────────
    print("Loading posterior and extracting parameters into numpy arrays...")
    draws_df = pd.read_csv(args.posterior)
    n_draws = len(draws_df)
    print(f"  {n_draws} draws, {len(draws_df.columns)} columns")

    # Global parameters: shape (n_draws,)
    int_run = np.zeros((n_draws, 4))
    theta_run = np.zeros((n_draws, 2))
    home_adv = np.zeros(n_draws)
    beta_park = np.ones(n_draws)
    beta_bullpen = np.zeros(n_draws)

    for k in range(4):
        col = f"int_run_{k+1}"
        int_run[:, k] = draws_df[col].values
    for k in range(2):
        col = f"theta_run_{k+1}"
        theta_run[:, k] = draws_df[col].values
    home_adv[:] = draws_df["home_advantage"].values
    if "beta_park" in draws_df.columns:
        beta_park[:] = draws_df["beta_park"].values
    if "beta_bullpen" in draws_df.columns:
        beta_bullpen[:] = draws_df["beta_bullpen"].values

    # Team attack/defense: shape (n_draws, N_teams+1, 4)  [0-indexed, team 0 = padding]
    att = np.zeros((n_draws, N_teams + 1, 4))
    def_ = np.zeros((n_draws, N_teams + 1, 4))
    for k in range(4):
        for t in range(1, N_teams + 1):
            col_att = f"att_run_{k+1}[{t}]"
            col_def = f"def_run_{k+1}[{t}]"
            if col_att in draws_df.columns:
                att[:, t, k] = draws_df[col_att].values
            if col_def in draws_df.columns:
                def_[:, t, k] = draws_df[col_def].values
    print(f"  Extracted team attack/defense: {N_teams} teams × 4 run types")

    # Pitcher ability: shape (n_draws, N_pitchers+1)  [0-indexed, pitcher 0 = unknown = 0]
    pitcher_ab = np.zeros((n_draws, N_pitchers + 1))
    for p in range(1, N_pitchers + 1):
        col = f"pitcher_ability[{p}]"
        if col in draws_df.columns:
            pitcher_ab[:, p] = draws_df[col].values
    print(f"  Extracted pitcher abilities: {N_pitchers} pitchers")

    # Free the DataFrame
    del draws_df

    # ── Load holdout games ─────────────────────────────────────────────────
    re_df = pd.read_csv(args.run_events, dtype=str)
    date_col = "game_date" if "game_date" in re_df.columns else "date"
    re_df["date"] = pd.to_datetime(re_df[date_col], errors="coerce")
    max_date = re_df["date"].max()
    cutoff = max_date - pd.Timedelta(days=args.holdout_days)
    holdout = re_df[re_df["date"] > cutoff].copy()

    holdout["home_idx"] = holdout["home_canonical_id"].map(lambda x: team_map.get(str(x).strip()))
    holdout["away_idx"] = holdout["away_canonical_id"].map(lambda x: team_map.get(str(x).strip()))
    holdout = holdout.dropna(subset=["home_idx", "away_idx"])
    holdout["home_idx"] = holdout["home_idx"].astype(int)
    holdout["away_idx"] = holdout["away_idx"].astype(int)

    if "home_score" in holdout.columns and "home_total_runs" not in holdout.columns:
        holdout["home_total_runs"] = pd.to_numeric(holdout["home_score"], errors="coerce").fillna(0).astype(int)
        holdout["away_total_runs"] = pd.to_numeric(holdout["away_score"], errors="coerce").fillna(0).astype(int)
    for c in ["home_total_runs", "away_total_runs"]:
        if c in holdout.columns:
            holdout[c] = pd.to_numeric(holdout[c], errors="coerce").fillna(0).astype(int)

    hp_col = "home_pitcher_id" if "home_pitcher_id" in holdout.columns else "home_pitcher_espn_id"
    ap_col = "away_pitcher_id" if "away_pitcher_id" in holdout.columns else "away_pitcher_espn_id"
    holdout["hp_idx"] = holdout[hp_col].map(
        lambda x: pitcher_map.get(str(x).strip() if pd.notna(x) else "", 0)
    ).fillna(0).astype(int)
    holdout["ap_idx"] = holdout[ap_col].map(
        lambda x: pitcher_map.get(str(x).strip() if pd.notna(x) else "", 0)
    ).fillna(0).astype(int)

    print(f"\nHoldout: {len(holdout)} games from {cutoff.date()} to {max_date.date()}")
    print(f"Using {n_draws} posterior draws, {args.N} sims per game")
    print(f"Simulating...\n")

    # ── Fast simulation ────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    results = []

    for i, (_, row) in enumerate(holdout.iterrows()):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(holdout)}]", file=sys.stderr)

        h_idx = int(row["home_idx"])
        a_idx = int(row["away_idx"])
        hp = int(row["hp_idx"])
        ap = int(row["ap_idx"])
        h_cid = str(row.get("home_canonical_id", "")).strip()
        a_cid = str(row.get("away_canonical_id", "")).strip()
        season = int(row.get("season", 2026)) if "season" in row.index else 2026
        pf = pf_map.get(h_cid, 0.0)
        h_bp = bp_map.get((h_cid, season), 0.0)
        a_bp = bp_map.get((a_cid, season), 0.0)

        actual_home = int(row.get("home_total_runs", 0))
        actual_away = int(row.get("away_total_runs", 0))
        actual_winner = "home" if actual_home > actual_away else "away"

        wins_home = 0
        exp_h_sum, exp_a_sum = 0.0, 0.0

        for _ in range(args.N):
            d = rng.integers(0, n_draws)

            # Expected runs (vectorized per draw)
            eh, ea = 0.0, 0.0
            park_eff = beta_park[d] * pf
            bp_h_eff = beta_bullpen[d] * a_bp  # away bullpen affects home scoring
            bp_a_eff = beta_bullpen[d] * h_bp  # home bullpen affects away scoring

            home_runs_sim = 0
            away_runs_sim = 0

            for k in range(4):
                log_lam_h = (int_run[d, k] + att[d, h_idx, k] + def_[d, a_idx, k]
                             + home_adv[d] + pitcher_ab[d, ap] + park_eff + bp_h_eff)
                log_lam_a = (int_run[d, k] + att[d, a_idx, k] + def_[d, h_idx, k]
                             + pitcher_ab[d, hp] + park_eff + bp_a_eff)

                mu_h = np.exp(log_lam_h)
                mu_a = np.exp(log_lam_a)
                eh += (k + 1) * mu_h
                ea += (k + 1) * mu_a

                # Sample counts
                if k <= 1:  # run_1 and run_2: NegBin
                    theta = max(1e-6, theta_run[d, k])
                    p_h = theta / (theta + max(1e-8, mu_h))
                    p_a = theta / (theta + max(1e-8, mu_a))
                    count_h = rng.negative_binomial(n=theta, p=p_h)
                    count_a = rng.negative_binomial(n=theta, p=p_a)
                else:  # run_3, run_4: Poisson
                    count_h = rng.poisson(lam=max(1e-8, mu_h))
                    count_a = rng.poisson(lam=max(1e-8, mu_a))

                home_runs_sim += (k + 1) * count_h
                away_runs_sim += (k + 1) * count_a

            exp_h_sum += eh
            exp_a_sum += ea

            # Extra innings
            extra = 0
            while home_runs_sim == away_runs_sim and extra < 20:
                for k in range(4):
                    log_lam_h = (int_run[d, k] + att[d, h_idx, k] + def_[d, a_idx, k]
                                 + home_adv[d] + pitcher_ab[d, ap] + park_eff + bp_h_eff)
                    log_lam_a = (int_run[d, k] + att[d, a_idx, k] + def_[d, h_idx, k]
                                 + pitcher_ab[d, hp] + park_eff + bp_a_eff)
                    mu_h = np.exp(log_lam_h) / 9.0
                    mu_a = np.exp(log_lam_a) / 9.0
                    if k <= 1:
                        theta = max(1e-6, theta_run[d, k])
                        p_h = theta / (theta + max(1e-8, mu_h))
                        p_a = theta / (theta + max(1e-8, mu_a))
                        home_runs_sim += (k + 1) * rng.negative_binomial(n=theta, p=p_h)
                        away_runs_sim += (k + 1) * rng.negative_binomial(n=theta, p=p_a)
                    else:
                        home_runs_sim += (k + 1) * rng.poisson(lam=max(1e-8, mu_h))
                        away_runs_sim += (k + 1) * rng.poisson(lam=max(1e-8, mu_a))
                extra += 1

            if home_runs_sim == away_runs_sim:
                if rng.random() < 0.5:
                    home_runs_sim += 1
                else:
                    away_runs_sim += 1

            if home_runs_sim > away_runs_sim:
                wins_home += 1

        win_prob = wins_home / args.N
        exp_h = exp_h_sum / args.N
        exp_a = exp_a_sum / args.N

        results.append({
            "date": str(row.get("date", ""))[:10],
            "home_team": h_cid,
            "away_team": a_cid,
            "actual_home_runs": actual_home,
            "actual_away_runs": actual_away,
            "actual_total": actual_home + actual_away,
            "actual_winner": actual_winner,
            "model_home_win_prob": win_prob,
            "model_exp_home": exp_h,
            "model_exp_away": exp_a,
            "model_exp_total": exp_h + exp_a,
            "model_correct": (win_prob > 0.5 and actual_winner == "home") or
                             (win_prob < 0.5 and actual_winner == "away"),
        })

    res_df = pd.DataFrame(results)

    # ── Metrics ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS ({len(res_df)} games)")
    print(f"{'='*60}")

    n_correct = res_df["model_correct"].sum()
    accuracy = n_correct / len(res_df)
    print(f"\n  Win prediction accuracy: {accuracy:.1%} ({n_correct}/{len(res_df)})")

    actual_home_win = (res_df["actual_winner"] == "home").astype(float)
    brier = ((res_df["model_home_win_prob"] - actual_home_win) ** 2).mean()
    brier_baseline = ((actual_home_win.mean() - actual_home_win) ** 2).mean()
    brier_skill = 1 - brier / brier_baseline if brier_baseline > 0 else 0
    print(f"  Brier score: {brier:.4f} (baseline: {brier_baseline:.4f}, skill: {brier_skill:.3f})")

    print(f"\n  Calibration (home win prob bins):")
    print(f"  {'Bin':>12s}  {'N':>5s}  {'Model':>8s}  {'Actual':>8s}  {'Gap':>8s}")
    for lo, hi in [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]:
        mask = (res_df["model_home_win_prob"] >= lo) & (res_df["model_home_win_prob"] < hi)
        n = mask.sum()
        if n > 0:
            model_avg = res_df.loc[mask, "model_home_win_prob"].mean()
            actual_avg = actual_home_win[mask].mean()
            gap = actual_avg - model_avg
            print(f"  [{lo:.1f}-{hi:.1f})  {n:5d}  {model_avg:8.3f}  {actual_avg:8.3f}  {gap:+8.3f}")

    total_mae = (res_df["model_exp_total"] - res_df["actual_total"]).abs().mean()
    total_corr = res_df["model_exp_total"].corr(res_df["actual_total"].astype(float))
    print(f"\n  Expected total runs:")
    print(f"    MAE: {total_mae:.2f}")
    print(f"    Correlation: {total_corr:.3f}")
    print(f"    Model mean total: {res_df['model_exp_total'].mean():.2f}")
    print(f"    Actual mean total: {res_df['actual_total'].mean():.2f}")

    for threshold in [0.55, 0.60, 0.65]:
        confident = res_df[(res_df["model_home_win_prob"] > threshold) |
                           (res_df["model_home_win_prob"] < (1 - threshold))]
        if len(confident) > 0:
            conf_acc = confident["model_correct"].mean()
            print(f"\n  Confident picks (>{threshold:.0%}): {conf_acc:.1%} ({len(confident)} games)")

    if args.out:
        res_df.to_csv(args.out, index=False)
        print(f"\n  Results saved to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
