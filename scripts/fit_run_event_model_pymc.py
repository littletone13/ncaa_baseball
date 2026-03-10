"""
Fit run-event model with PyMC (Path A step 6.3, no CmdStan).

Same model as stan/ncaa_baseball_run_events.stan: Mack Ch 18 architecture —
sum-to-zero centering, NegBin for run_1/run_2, Poisson for run_3/run_4,
single pitcher_ability scalar, single home_advantage, park factors,
and bullpen quality adjustments.

Writes run_event_posterior.csv and run_event_fit_meta.json in the format
simulate_run_event_game.py expects.

Usage:
  pip install pymc
  python3 scripts/fit_run_event_model_pymc.py
  python3 scripts/fit_run_event_model_pymc.py --subsample 300 --draws 200  # quicker
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pymc as pm
    import pytensor.tensor as pt
except ImportError:
    pm = None
    pt = None


def _load_fit_data(
    run_events_path: Path,
    team_index_path: Path,
    pitcher_index_path: Path,
    subsample: int | None,
) -> tuple[pd.DataFrame, int, int]:
    re_df = pd.read_csv(run_events_path, dtype=str)
    team_df = pd.read_csv(team_index_path)
    pitcher_df = pd.read_csv(pitcher_index_path)

    run_cols = [f"home_run_{k}" for k in range(1, 5)] + [f"away_run_{k}" for k in range(1, 5)]
    for c in run_cols:
        if c in re_df.columns:
            re_df[c] = pd.to_numeric(re_df[c], errors="coerce").fillna(0).astype(int)

    team_map = dict(zip(team_df["canonical_id"], team_df["team_idx"]))
    pitcher_map: dict[str, int] = {"unknown": 0, "": 0}
    for _, r in pitcher_df.iterrows():
        pid = str(r["pitcher_espn_id"]).strip()
        if pid and pid.lower() != "unknown":
            pitcher_map[pid] = int(r["pitcher_idx"])

    def team_idx(cid: str) -> int | None:
        cid = str(cid).strip() if pd.notna(cid) else ""
        return team_map.get(cid) if cid else None

    def pitcher_idx(espn_id: str) -> int:
        pid = str(espn_id).strip() if pd.notna(espn_id) else ""
        if not pid or pid == "nan":
            return 0
        return pitcher_map.get(pid, 0)

    re_df["home_team_idx"] = re_df["home_canonical_id"].map(team_idx)
    re_df["away_team_idx"] = re_df["away_canonical_id"].map(team_idx)
    re_df["home_pitcher_idx"] = re_df["home_pitcher_espn_id"].map(pitcher_idx)
    re_df["away_pitcher_idx"] = re_df["away_pitcher_espn_id"].map(pitcher_idx)

    mask = re_df["home_team_idx"].notna() & re_df["away_team_idx"].notna()
    fit_df = re_df.loc[mask].copy()
    fit_df["home_team_idx"] = fit_df["home_team_idx"].astype(int)
    fit_df["away_team_idx"] = fit_df["away_team_idx"].astype(int)
    if fit_df.empty:
        raise SystemExit("No games with both teams in index.")
    if subsample is not None and len(fit_df) > subsample:
        fit_df = fit_df.sample(n=subsample, random_state=42).reset_index(drop=True)
    N_teams = int(team_df["team_idx"].max())
    N_pitchers = int(pitcher_df.loc[pitcher_df["pitcher_espn_id"] != "unknown", "pitcher_idx"].max())
    return fit_df, N_teams, N_pitchers


def _posterior_to_draws_csv(idata, N_teams: int, N_pitchers: int) -> pd.DataFrame:
    """Turn PyMC idata into DataFrame with CmdStanPy-compatible column names."""
    post = idata.posterior
    n_draws = post.sizes["draw"] * post.sizes["chain"]
    rows = []
    for d in range(n_draws):
        chain = d // post.sizes["draw"]
        draw = d % post.sizes["draw"]
        row = {}
        # Scalar parameters
        row["home_advantage"] = float(post["home_advantage"].values[chain, draw])
        row["theta_run_1"] = float(post["theta_run_1"].values[chain, draw])
        row["theta_run_2"] = float(post["theta_run_2"].values[chain, draw])
        row["beta_park"] = float(post["beta_park"].values[chain, draw])
        row["beta_bullpen"] = float(post["beta_bullpen"].values[chain, draw])
        # Per-run-type intercepts
        for k in range(1, 5):
            row[f"int_run_{k}"] = float(post[f"int_run_{k}"].values[chain, draw])
        # Per-run-type team abilities (sum-to-zero centered)
        for k in range(1, 5):
            for i in range(1, N_teams + 1):
                row[f"att_run_{k}.{i}"] = float(post[f"att_run_{k}"].values[chain, draw, i - 1])
                row[f"def_run_{k}.{i}"] = float(post[f"def_run_{k}"].values[chain, draw, i - 1])
        # Pitcher ability (single scalar, sum-to-zero centered)
        for i in range(1, N_pitchers + 1):
            row[f"pitcher_ability.{i}"] = float(post["pitcher_ability"].values[chain, draw, i - 1])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit run-event model with PyMC; save posterior.")
    parser.add_argument("--run-events", type=Path, default=Path("data/processed/run_events.csv"))
    parser.add_argument("--team-index", type=Path, default=Path("data/processed/run_event_team_index.csv"))
    parser.add_argument("--pitcher-index", type=Path, default=Path("data/processed/run_event_pitcher_index.csv"))
    parser.add_argument("--park-factors", type=Path, default=Path("data/processed/park_factors.csv"))
    parser.add_argument("--bullpen-quality", type=Path, default=Path("data/processed/bullpen_quality.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--subsample", type=int, default=None, help="Use at most this many games")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--draws", type=int, default=2000, help="Post-warmup draws per chain")
    parser.add_argument("--tune", type=int, default=500, help="Tuning/warmup steps per chain")
    parser.add_argument("--cores", type=int, default=4)
    args = parser.parse_args()

    if pm is None or pt is None:
        print("Install PyMC: pip install pymc")
        return 1

    for p in (args.run_events, args.team_index, args.pitcher_index):
        if not p.exists():
            print(f"Missing: {p}")
            return 1

    fit_df, N_teams, N_pitchers = _load_fit_data(
        args.run_events, args.team_index, args.pitcher_index, args.subsample
    )
    N_games = len(fit_df)
    print(f"Fitting: {N_games} games, {N_teams} teams, {N_pitchers} pitchers")

    home_team = fit_df["home_team_idx"].values.astype(int) - 1  # 0-based for PyMC indexing
    away_team = fit_df["away_team_idx"].values.astype(int) - 1
    home_pitcher = fit_df["home_pitcher_idx"].values.astype(int)  # 0 = unknown, 1..N_pitchers
    away_pitcher = fit_df["away_pitcher_idx"].values.astype(int)

    obs = {}
    for k in range(1, 5):
        obs[f"home_run_{k}"] = fit_df[f"home_run_{k}"].values
        obs[f"away_run_{k}"] = fit_df[f"away_run_{k}"].values

    # ── Park factors: log(adjusted_pf), default 0 (neutral) ──────────────────
    park_factor_data = np.zeros(N_games)
    n_park = 0
    if args.park_factors.exists():
        pf_df = pd.read_csv(args.park_factors)
        pf_map: dict[str, float] = {}
        for _, r in pf_df.iterrows():
            htid = str(r.get("home_team_id", "")).strip()
            adj = r.get("adjusted_pf")
            if htid and adj is not None and not (isinstance(adj, float) and math.isnan(adj)):
                pf_map[htid] = math.log(float(adj))
        for i, row in fit_df.iterrows():
            home_cid = str(row.get("home_canonical_id", "")).strip()
            pf = pf_map.get(home_cid)
            if pf is not None:
                idx = fit_df.index.get_loc(i)
                park_factor_data[idx] = pf
                n_park += 1
        print(f"Park factors: {n_park}/{N_games} games matched")
    else:
        print(f"Park factors: not found, using 0 (neutral)")

    # ── Bullpen quality: map team+season → adj ────────────────────────────────
    home_bp_data = np.zeros(N_games)
    away_bp_data = np.zeros(N_games)
    n_bp = 0
    if args.bullpen_quality.exists():
        bq_df = pd.read_csv(args.bullpen_quality)
        bp_map: dict[tuple[str, int], float] = {}
        for _, r in bq_df.iterrows():
            cid = str(r.get("team_canonical_id", "")).strip()
            season = int(r.get("season", 0))
            score = r.get("bullpen_depth_score")
            if cid and season and score is not None and not (isinstance(score, float) and math.isnan(score)):
                bp_map[(cid, season)] = -float(score) * 0.1
        for i, row in fit_df.iterrows():
            home_cid = str(row.get("home_canonical_id", "")).strip()
            away_cid = str(row.get("away_canonical_id", "")).strip()
            season = int(row.get("season", 0)) if "season" in row.index else 0
            idx = fit_df.index.get_loc(i)
            h_bp = bp_map.get((home_cid, season))
            a_bp = bp_map.get((away_cid, season))
            if h_bp is not None:
                home_bp_data[idx] = h_bp
            if a_bp is not None:
                away_bp_data[idx] = a_bp
            if h_bp is not None or a_bp is not None:
                n_bp += 1
        print(f"Bullpen quality: {n_bp}/{N_games} games matched")
    else:
        print(f"Bullpen quality: not found, using 0")

    with pm.Model() as model:
        # Intercepts — positive (log scale), college-calibrated priors
        int_run_1 = pm.TruncatedNormal("int_run_1", mu=1.2, sigma=0.3, lower=0)
        int_run_2 = pm.TruncatedNormal("int_run_2", mu=0.7, sigma=0.3, lower=0)
        int_run_3 = pm.TruncatedNormal("int_run_3", mu=0.1, sigma=0.3, lower=0)
        int_run_4 = pm.TruncatedNormal("int_run_4", mu=-0.2, sigma=0.3, lower=0)
        intercepts = [int_run_1, int_run_2, int_run_3, int_run_4]

        # Home advantage — single scalar
        home_advantage = pm.TruncatedNormal("home_advantage", mu=0, sigma=0.1, lower=0)

        # Dispersion — only for run_1 and run_2 (NegBin)
        theta_run_1 = pm.Gamma("theta_run_1", alpha=30, beta=1)
        theta_run_2 = pm.Gamma("theta_run_2", alpha=30, beta=1)
        thetas = [theta_run_1, theta_run_2]

        # Park and bullpen coefficients
        beta_park = pm.Normal("beta_park", mu=1, sigma=0.3)
        beta_bullpen = pm.Normal("beta_bullpen", mu=0, sigma=0.2)

        # Park and bullpen data as shared tensors
        park_shared = pt.as_tensor_variable(park_factor_data)
        home_bp_shared = pt.as_tensor_variable(home_bp_data)
        away_bp_shared = pt.as_tensor_variable(away_bp_data)

        # Team abilities — raw (then center)
        att_raw = {}
        def_raw = {}
        att_centered = {}
        def_centered = {}
        for k in range(1, 5):
            att_raw[k] = pm.Normal(f"att_run_{k}_raw", mu=0, sigma=0.2, shape=N_teams)
            def_raw[k] = pm.Normal(f"def_run_{k}_raw", mu=0, sigma=0.2, shape=N_teams)
            att_centered[k] = pm.Deterministic(f"att_run_{k}", att_raw[k] - att_raw[k].mean())
            def_centered[k] = pm.Deterministic(f"def_run_{k}", def_raw[k] - def_raw[k].mean())

        # Pitcher ability — single scalar per pitcher, centered
        pitcher_raw = pm.Normal("pitcher_ability_raw", mu=0, sigma=0.15, shape=N_pitchers)
        pitcher_ability = pm.Deterministic("pitcher_ability", pitcher_raw - pitcher_raw.mean())

        # Park effect (shared across all run types)
        park_effect = beta_park * park_shared
        # Bullpen: away bullpen affects home scoring and vice versa
        bp_h = beta_bullpen * away_bp_shared  # away bullpen quality -> home scoring
        bp_a = beta_bullpen * home_bp_shared  # home bullpen quality -> away scoring

        for k in range(1, 5):
            # Pitcher effect (0 = unknown -> 0)
            p_away = pt.switch(away_pitcher >= 1, pitcher_ability[away_pitcher - 1], 0.0)
            p_home = pt.switch(home_pitcher >= 1, pitcher_ability[home_pitcher - 1], 0.0)

            log_lam_h = intercepts[k - 1] + att_centered[k][home_team] + def_centered[k][away_team] + home_advantage + p_away + park_effect + bp_h
            log_lam_a = intercepts[k - 1] + att_centered[k][away_team] + def_centered[k][home_team] + p_home + park_effect + bp_a

            mu_h = pt.exp(log_lam_h)
            mu_a = pt.exp(log_lam_a)

            if k <= 2:
                # NegBin for run_1, run_2
                theta = thetas[k - 1]
                pm.NegativeBinomial(f"home_run_{k}", mu=mu_h, alpha=theta, observed=obs[f"home_run_{k}"])
                pm.NegativeBinomial(f"away_run_{k}", mu=mu_a, alpha=theta, observed=obs[f"away_run_{k}"])
            else:
                # Poisson for run_3, run_4
                pm.Poisson(f"home_run_{k}", mu=mu_h, observed=obs[f"home_run_{k}"])
                pm.Poisson(f"away_run_{k}", mu=mu_a, observed=obs[f"away_run_{k}"])

        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            progressbar=True,
            return_inferencedata=True,
            target_accept=0.9,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    draws_df = _posterior_to_draws_csv(idata, N_teams, N_pitchers)
    posterior_csv = args.out_dir / "run_event_posterior.csv"
    draws_df.to_csv(posterior_csv, index=False)
    print(f"Posterior: {len(draws_df)} draws -> {posterior_csv}")

    meta = {"N_teams": N_teams, "N_pitchers": N_pitchers, "n_draws": len(draws_df)}
    meta_json = args.out_dir / "run_event_fit_meta.json"
    with open(meta_json, "w") as f:
        json.dump(meta, f)
    print(f"Meta -> {meta_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
