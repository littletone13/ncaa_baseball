"""
Fit Stan run-event model (Path A step 6.3).

Loads run_events.csv and team/pitcher index CSVs, filters to games with both teams
resolved, builds Stan data, runs CmdStanPy sample(), saves posterior draws and
meta so simulate_run_event_game.py can use them.

Stan model: Mack Ch 18 architecture — sum-to-zero centering, NegBin for run_1/run_2,
Poisson for run_3/run_4, single pitcher_ability scalar, single home_advantage,
park factors, and bullpen quality adjustments.

Usage:
  pip install cmdstanpy  # and install CmdStan: python -m cmdstanpy.install_cmdstan
  python3 scripts/fit_run_event_model.py --run-events data/processed/run_events.csv
  python3 scripts/fit_run_event_model.py --chains 2 --iter 500  # smaller for testing
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from cmdstanpy import CmdStanModel
except ImportError:
    CmdStanModel = None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit Stan run-event model; save posterior and meta.",
    )
    parser.add_argument("--run-events", type=Path, default=Path("data/processed/run_events.csv"))
    parser.add_argument("--team-index", type=Path, default=Path("data/processed/run_event_team_index.csv"))
    parser.add_argument("--pitcher-index", type=Path, default=Path("data/processed/run_event_pitcher_index.csv"))
    parser.add_argument("--stan-file", type=Path, default=Path("stan/ncaa_baseball_run_events.stan"))
    parser.add_argument("--park-factors", type=Path, default=Path("data/processed/park_factors.csv"))
    parser.add_argument("--bullpen-quality", type=Path, default=Path("data/processed/bullpen_quality.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--iter", type=int, default=10000, help="Total iterations per chain (including warmup)")
    parser.add_argument("--warmup", type=int, default=None, help="Warmup iterations per chain (default: iter//5)")
    parser.add_argument("--subsample", type=int, default=None, help="Use at most this many games (for quick test runs)")
    args = parser.parse_args()

    if CmdStanModel is None:
        print("Install cmdstanpy and CmdStan: pip install cmdstanpy && python -m cmdstanpy.install_cmdstan")
        return 1

    for p in (args.run_events, args.team_index, args.pitcher_index, args.stan_file):
        if not p.exists():
            print(f"Missing: {p}")
            return 1

    # Load tables
    re_df = pd.read_csv(args.run_events, dtype=str)
    team_df = pd.read_csv(args.team_index)
    pitcher_df = pd.read_csv(args.pitcher_index)

    # Coerce run counts to int
    run_cols = [f"home_run_{k}" for k in range(1, 5)] + [f"away_run_{k}" for k in range(1, 5)]
    for c in run_cols:
        if c in re_df.columns:
            re_df[c] = pd.to_numeric(re_df[c], errors="coerce").fillna(0).astype(int)

    # Map canonical_id -> team_idx, pitcher_id -> pitcher_idx
    team_map = dict(zip(team_df["canonical_id"], team_df["team_idx"]))
    pitcher_map: dict[str, int] = {"unknown": 0, "": 0}
    for _, r in pitcher_df.iterrows():
        pid = str(r["pitcher_espn_id"]).strip()
        if pid and pid.lower() != "unknown":
            pitcher_map[pid] = int(r["pitcher_idx"])

    def team_idx(cid) -> int | None:
        if cid is None or (isinstance(cid, float) and pd.isna(cid)):
            return None
        cid = str(cid).strip()
        return team_map.get(cid) if cid else None

    def pitcher_idx(pid_val) -> int:
        if pid_val is None or (isinstance(pid_val, float) and pd.isna(pid_val)):
            return 0
        pid = str(pid_val).strip() or "unknown"
        return pitcher_map.get(pid, 0)

    re_df["home_team_idx"] = re_df["home_canonical_id"].map(team_idx)
    re_df["away_team_idx"] = re_df["away_canonical_id"].map(team_idx)
    # Support both old column name (home_pitcher_espn_id) and new (home_pitcher_id)
    hp_col = "home_pitcher_id" if "home_pitcher_id" in re_df.columns else "home_pitcher_espn_id"
    ap_col = "away_pitcher_id" if "away_pitcher_id" in re_df.columns else "away_pitcher_espn_id"
    re_df["home_pitcher_idx"] = re_df[hp_col].map(pitcher_idx)
    re_df["away_pitcher_idx"] = re_df[ap_col].map(pitcher_idx)

    # Keep only games with both teams in index
    mask = re_df["home_team_idx"].notna() & re_df["away_team_idx"].notna()
    fit_df = re_df.loc[mask].copy()
    fit_df["home_team_idx"] = fit_df["home_team_idx"].astype(int)
    fit_df["away_team_idx"] = fit_df["away_team_idx"].astype(int)
    if fit_df.empty:
        print("No games with both teams in index; check run_events and team index.")
        return 1

    if args.subsample is not None and len(fit_df) > args.subsample:
        fit_df = fit_df.sample(n=args.subsample, random_state=42).reset_index(drop=True)
        print(f"Subsampled to {len(fit_df)} games for quick run.")
    N_games = len(fit_df)
    N_teams = int(team_df["team_idx"].max())
    N_pitchers = int(pitcher_df.loc[pitcher_df["pitcher_espn_id"] != "unknown", "pitcher_idx"].max())

    # ── Park factors: map venue → log(adjusted_pf), default 0 (neutral) ──────
    park_factor_vec = [0.0] * N_games
    n_park_matched = 0
    if args.park_factors.exists():
        pf_df = pd.read_csv(args.park_factors)
        # Build lookup by home_team_id -> adjusted_pf (log scale)
        pf_map: dict[str, float] = {}
        for _, r in pf_df.iterrows():
            htid = str(r.get("home_team_id", "")).strip()
            adj = r.get("adjusted_pf")
            if htid and adj is not None and not (isinstance(adj, float) and math.isnan(adj)):
                pf_map[htid] = math.log(float(adj))
        # Also build venue-name lookup as fallback
        pf_venue_map: dict[str, float] = {}
        for _, r in pf_df.iterrows():
            vname = str(r.get("venue_name", "")).strip()
            adj = r.get("adjusted_pf")
            if vname and adj is not None and not (isinstance(adj, float) and math.isnan(adj)):
                pf_venue_map[vname] = math.log(float(adj))

        # Map each game: use home_canonical_id to look up park factor
        for i, row in fit_df.iterrows():
            home_cid = str(row.get("home_canonical_id", "")).strip()
            venue = str(row.get("venue_name", "")).strip()
            pf = pf_map.get(home_cid)
            if pf is None and venue:
                pf = pf_venue_map.get(venue)
            if pf is not None:
                idx = fit_df.index.get_loc(i)
                park_factor_vec[idx] = pf
                n_park_matched += 1
        print(f"Park factors: {n_park_matched}/{N_games} games matched ({n_park_matched/N_games*100:.1f}%)")
    else:
        print(f"Park factors: {args.park_factors} not found, using 0 (neutral) for all games.")

    # ── Bullpen quality: map team+season → bullpen_depth_score ────────────────
    home_bullpen_vec = [0.0] * N_games
    away_bullpen_vec = [0.0] * N_games
    n_bp_matched = 0
    if args.bullpen_quality.exists():
        bq_df = pd.read_csv(args.bullpen_quality)
        # Build (canonical_id, season) -> bullpen_depth_score
        # Then convert to z-score-based log-scale adjustment
        # Positive score = better bullpen; we want: better bullpen -> negative adj
        # (opponent scores less), so bullpen_adj = -depth_score * scale
        bp_map: dict[tuple[str, int], float] = {}
        for _, r in bq_df.iterrows():
            cid = str(r.get("team_canonical_id", "")).strip()
            season = int(r.get("season", 0))
            score = r.get("bullpen_depth_score")
            if cid and season and score is not None and not (isinstance(score, float) and math.isnan(score)):
                # depth_score is already z-score composite; negate and scale
                # so positive adj = worse bullpen = opponent scores more
                bp_map[(cid, season)] = -float(score) * 0.1  # scale to log-rate magnitude
        for i, row in fit_df.iterrows():
            home_cid = str(row.get("home_canonical_id", "")).strip()
            away_cid = str(row.get("away_canonical_id", "")).strip()
            season = int(row.get("season", 0)) if "season" in row.index else 0
            idx = fit_df.index.get_loc(i)
            h_bp = bp_map.get((home_cid, season))
            a_bp = bp_map.get((away_cid, season))
            if h_bp is not None:
                home_bullpen_vec[idx] = h_bp
            if a_bp is not None:
                away_bullpen_vec[idx] = a_bp
            if h_bp is not None or a_bp is not None:
                n_bp_matched += 1
        print(f"Bullpen quality: {n_bp_matched}/{N_games} games with at least one team matched")
    else:
        print(f"Bullpen quality: {args.bullpen_quality} not found, using 0 for all games.")

    # ── Conference index: team_idx → conf_idx ──────────────────────────────────
    if "conf_idx" in team_df.columns:
        team_conf_map = dict(zip(team_df["team_idx"].astype(int), team_df["conf_idx"].astype(int)))
        N_conf = int(team_df["conf_idx"].max())
    else:
        # Fallback: all teams in one conference
        team_conf_map = {}
        N_conf = 1
    team_conf_arr = [team_conf_map.get(t, 1) for t in range(1, N_teams + 1)]
    print(f"Conference hierarchy: {N_conf} conferences")

    # ── FIP informative priors for pitcher ability ────────────────────────────
    pt_path = Path("data/processed/pitcher_table.csv")
    fip_prior = [0.0] * N_pitchers  # length N_pitchers (1-indexed in Stan)
    n_fip_priors = 0
    if pt_path.exists():
        pt = pd.read_csv(pt_path)
        pt["pitcher_idx"] = pd.to_numeric(pt["pitcher_idx"], errors="coerce").fillna(0).astype(int)
        pt["fip"] = pd.to_numeric(pt.get("fip"), errors="coerce")

        valid_fip = pt.loc[pt["fip"].notna(), "fip"].values
        if len(valid_fip) > 10:
            fip_mean = float(np.mean(valid_fip))
            fip_std = float(np.std(valid_fip))
            if fip_std > 0.1:
                for _, row in pt.iterrows():
                    pidx = int(row["pitcher_idx"])
                    fip_val = row["fip"]
                    if 1 <= pidx <= N_pitchers and pd.notna(fip_val):
                        # Positive z = high FIP = bad pitcher = positive ability (allows runs)
                        z = (float(fip_val) - fip_mean) / fip_std
                        z = float(np.clip(z, -2.0, 2.0))
                        fip_prior[pidx - 1] = z * 0.08  # scale by estimated ability std
                        n_fip_priors += 1
        print(f"FIP priors: {n_fip_priors}/{N_pitchers} pitchers have informative priors "
              f"(FIP μ={fip_mean:.2f} σ={fip_std:.2f})")
    else:
        print("FIP priors: pitcher_table.csv not found, using uninformative priors for all pitchers")

    stan_data = {
        "N_games": N_games,
        "N_teams": N_teams,
        "N_pitchers": N_pitchers,
        "N_conf": N_conf,
        "team_conf": team_conf_arr,
        "home_team_idx": fit_df["home_team_idx"].tolist(),
        "away_team_idx": fit_df["away_team_idx"].tolist(),
        "home_pitcher_idx": fit_df["home_pitcher_idx"].tolist(),
        "away_pitcher_idx": fit_df["away_pitcher_idx"].tolist(),
        "home_run_1": fit_df["home_run_1"].tolist(),
        "home_run_2": fit_df["home_run_2"].tolist(),
        "home_run_3": fit_df["home_run_3"].tolist(),
        "home_run_4": fit_df["home_run_4"].tolist(),
        "away_run_1": fit_df["away_run_1"].tolist(),
        "away_run_2": fit_df["away_run_2"].tolist(),
        "away_run_3": fit_df["away_run_3"].tolist(),
        "away_run_4": fit_df["away_run_4"].tolist(),
        "park_factor": park_factor_vec,
        "home_bullpen_adj": home_bullpen_vec,
        "away_bullpen_adj": away_bullpen_vec,
        "fip_prior": fip_prior,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_json = args.out_dir / "run_event_stan_data.json"
    with open(data_json, "w") as f:
        json.dump(stan_data, f, indent=0)
    print(f"Stan data: {N_games} games, {N_teams} teams, {N_pitchers} pitchers -> {data_json}")

    # Compile and fit
    stan_path = args.stan_file if args.stan_file.is_absolute() else Path.cwd() / args.stan_file
    model = CmdStanModel(stan_file=str(stan_path))
    warmup = args.warmup if args.warmup is not None else args.iter // 5
    iter_sampling = max(1, args.iter - warmup)
    fit = model.sample(
        data=stan_data,
        chains=args.chains,
        iter_warmup=warmup,
        iter_sampling=iter_sampling,
        show_progress=True,
        output_dir=str(args.out_dir),
    )

    # Save posterior draws as single CSV (one row per draw) for simulate script
    draws = fit.draws_pd()
    posterior_csv = args.out_dir / "run_event_posterior.csv"
    draws.to_csv(posterior_csv, index=False)
    print(f"Posterior: {len(draws)} draws -> {posterior_csv}")

    meta = {
        "N_teams": N_teams,
        "N_pitchers": N_pitchers,
        "N_conf": N_conf,
        "n_draws": len(draws),
    }
    meta_json = args.out_dir / "run_event_fit_meta.json"
    with open(meta_json, "w") as f:
        json.dump(meta, f)
    print(f"Meta -> {meta_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
