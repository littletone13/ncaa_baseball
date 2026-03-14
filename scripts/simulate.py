"""
Pure Monte Carlo simulation engine for NCAA baseball predictions.

Takes pre-resolved inputs (schedule, starters, weather CSVs), loads the
Stan posterior, runs Monte Carlo simulation, and returns predictions.

Makes NO API calls. Deterministic (same seed -> same output).

Usage:
  python3 scripts/simulate.py \\
      --schedule data/daily/2026-03-14/schedule.csv \\
      --starters data/daily/2026-03-14/starters.csv \\
      --weather data/daily/2026-03-14/weather.csv \\
      --N 5000 --seed 42 \\
      --out data/processed/predictions_2026-03-14.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Utility ──────────────────────────────────────────────────────────────────

def prob_to_american(p: float) -> int:
    """Convert win probability to American moneyline odds."""
    if p <= 0.0:
        return 9999
    if p >= 1.0:
        return -9999
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


# ── Posterior Loading ────────────────────────────────────────────────────────

def load_posterior(
    posterior_csv: Path,
    meta_json: Path,
) -> dict:
    """Load Stan posterior draws into NumPy arrays.

    Returns dict with keys:
        int_run, theta_run, home_adv, beta_park, beta_bullpen,
        att, def_, pitcher_ab, n_draws, N_teams, N_pitchers
    """
    with open(meta_json) as f:
        meta = json.load(f)
    N_teams = meta["N_teams"]
    N_pitchers = meta["N_pitchers"]

    draws_df = pd.read_csv(posterior_csv)
    n_draws = len(draws_df)

    int_run = np.zeros((n_draws, 4))
    theta_run = np.zeros((n_draws, 2))
    home_adv = np.zeros(n_draws)
    beta_park = np.ones(n_draws)
    beta_bullpen = np.zeros(n_draws)

    for k in range(4):
        int_run[:, k] = draws_df[f"int_run_{k+1}"].values
    for k in range(2):
        theta_run[:, k] = draws_df[f"theta_run_{k+1}"].values
    home_adv[:] = draws_df["home_advantage"].values
    if "beta_park" in draws_df.columns:
        beta_park[:] = draws_df["beta_park"].values
    if "beta_bullpen" in draws_df.columns:
        beta_bullpen[:] = draws_df["beta_bullpen"].values

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

    return {
        "int_run": int_run,
        "theta_run": theta_run,
        "home_adv": home_adv,
        "beta_park": beta_park,
        "beta_bullpen": beta_bullpen,
        "att": att,
        "def_": def_,
        "pitcher_ab": pitcher_ab,
        "n_draws": n_draws,
        "N_teams": N_teams,
        "N_pitchers": N_pitchers,
    }


# ── Simulation ───────────────────────────────────────────────────────────────

def simulate_games(
    schedule_csv: Path,
    starters_csv: Path,
    weather_csv: Path,
    posterior_csv: Path,
    meta_json: Path,
    team_table_csv: Path,
    n_sims: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Pure Monte Carlo simulation. No API calls. Deterministic.

    Reads pre-resolved schedule, starters, and weather CSVs plus the
    Stan posterior and team table. Returns DataFrame with one row per game.
    """
    # ── Load posterior ────────────────────────────────────────────────────
    print("Loading posterior...", file=sys.stderr)
    post = load_posterior(posterior_csv, meta_json)
    int_run = post["int_run"]
    theta_run = post["theta_run"]
    home_adv = post["home_adv"]
    beta_park = post["beta_park"]
    beta_bullpen = post["beta_bullpen"]
    att = post["att"]
    def_ = post["def_"]
    pitcher_ab = post["pitcher_ab"]
    n_draws = post["n_draws"]
    N_teams = post["N_teams"]
    N_pitchers = post["N_pitchers"]
    print(f"  {n_draws} draws, {N_teams} teams, {N_pitchers} pitchers", file=sys.stderr)

    # ── Load team table (bullpen quality + team index) ────────────────────
    team_table = pd.read_csv(team_table_csv, dtype=str)
    # Build canonical_id -> team_idx map
    team_idx_map: dict[str, int] = {}
    for _, r in team_table.iterrows():
        cid = str(r.get("canonical_id", "")).strip()
        tidx = r.get("team_idx", "0")
        if cid:
            team_idx_map[cid] = int(tidx)

    # Bullpen quality: bullpen_adj column (already sign-corrected in team_table)
    bp_map: dict[str, float] = {}
    for _, r in team_table.iterrows():
        cid = str(r.get("canonical_id", "")).strip()
        bp_val = r.get("bullpen_adj", "0")
        if cid:
            try:
                bp_map[cid] = float(bp_val) if bp_val and str(bp_val).strip() else 0.0
            except (ValueError, TypeError):
                bp_map[cid] = 0.0

    # ── Load input CSVs ──────────────────────────────────────────────────
    schedule = pd.read_csv(schedule_csv, dtype=str)
    starters = pd.read_csv(starters_csv, dtype=str)
    weather = pd.read_csv(weather_csv, dtype=str)

    # Index by game_num for fast lookup
    starters_by_game = {int(r["game_num"]): r for _, r in starters.iterrows()}
    weather_by_game = {int(r["game_num"]): r for _, r in weather.iterrows()}

    # ── Constants ─────────────────────────────────────────────────────────
    STARTER_IP_FRAC = 0.61  # ~5.5 / 9
    BULLPEN_IP_FRAC = 1.0 - STARTER_IP_FRAC

    # ── Simulate each game ───────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    all_results = []

    for _, sched_row in schedule.iterrows():
        game_num = int(sched_row["game_num"])
        h_cid = str(sched_row["home_cid"]).strip()
        a_cid = str(sched_row["away_cid"]).strip()
        h_name = str(sched_row["home_name"]).strip()
        a_name = str(sched_row["away_name"]).strip()

        # Team indices (clamp to posterior size)
        h_idx = team_idx_map.get(h_cid, 0)
        a_idx = team_idx_map.get(a_cid, 0)
        if h_idx > N_teams:
            h_idx = 0
        if a_idx > N_teams:
            a_idx = 0

        # ── Starters ─────────────────────────────────────────────────────
        st = starters_by_game.get(game_num, {})
        hp_name = _safe_str(st, "home_starter", "unknown")
        ap_name = _safe_str(st, "away_starter", "unknown")
        hp_idx = _safe_int(st, "home_starter_idx", 0)
        ap_idx = _safe_int(st, "away_starter_idx", 0)
        hp_hand = _safe_str(st, "hp_throws", "")
        ap_hand = _safe_str(st, "ap_throws", "")

        # D1B ability adjustments (for pitchers without posterior data)
        hp_era_adj = _safe_float(st, "hp_ability_adj", 0.0)
        ap_era_adj = _safe_float(st, "ap_ability_adj", 0.0)
        hp_adj_src = _safe_str(st, "hp_ability_src", "")
        ap_adj_src = _safe_str(st, "ap_ability_src", "")

        # FB sensitivity
        hp_fb_sens = _safe_float(st, "hp_fb_sens", 1.0)
        ap_fb_sens = _safe_float(st, "ap_fb_sens", 1.0)
        hp_bp_fb_sens = _safe_float(st, "hp_bp_fb_sens", 1.0)
        ap_bp_fb_sens = _safe_float(st, "ap_bp_fb_sens", 1.0)

        # Offense adjustments (wRC+ for non-model teams)
        h_att_adj = _safe_float(st, "home_wrc_adj", 0.0)
        a_att_adj = _safe_float(st, "away_wrc_adj", 0.0)

        # Platoon
        platoon_h = _safe_float(st, "platoon_adj_home", 0.0)
        platoon_a = _safe_float(st, "platoon_adj_away", 0.0)

        # Clamp pitcher indices
        if hp_idx >= N_pitchers + 1:
            hp_idx = 0
        if ap_idx >= N_pitchers + 1:
            ap_idx = 0

        # ── Weather / park ────────────────────────────────────────────────
        wx = weather_by_game.get(game_num, {})
        pf = _safe_float(wx, "park_factor", 0.0)
        wind_adj_raw = _safe_float(wx, "wind_adj_raw", 0.0)
        non_wind_adj = _safe_float(wx, "non_wind_adj", 0.0)
        temp_f = _safe_float_or_none(wx, "temp_f")
        wind_mph = _safe_float_or_none(wx, "wind_mph")
        wind_out_mph = _safe_float_or_none(wx, "wind_out_mph")
        wind_out_lf = _safe_float_or_none(wx, "wind_out_lf")
        wind_out_cf = _safe_float_or_none(wx, "wind_out_cf")
        wind_out_rf = _safe_float_or_none(wx, "wind_out_rf")
        weather_mode = _safe_str(wx, "weather_mode", "")

        # Display-level weather adj (average of both starters for summary)
        weather_adj = wind_adj_raw * (hp_fb_sens + ap_fb_sens) / 2.0 + non_wind_adj

        # Park + weather decomposition
        base_pf = pf + non_wind_adj

        # Home scoring: away pitcher on mound
        ap_blended_sens = STARTER_IP_FRAC * ap_fb_sens + BULLPEN_IP_FRAC * ap_bp_fb_sens
        wind_adj_home = wind_adj_raw * ap_blended_sens

        # Away scoring: home pitcher on mound
        hp_blended_sens = STARTER_IP_FRAC * hp_fb_sens + BULLPEN_IP_FRAC * hp_bp_fb_sens
        wind_adj_away = wind_adj_raw * hp_blended_sens

        # Pure bullpen wind (for extra innings)
        wind_adj_home_bp = wind_adj_raw * ap_bp_fb_sens
        wind_adj_away_bp = wind_adj_raw * hp_bp_fb_sens

        # Bullpen quality
        h_bp = bp_map.get(h_cid, 0.0)
        a_bp = bp_map.get(a_cid, 0.0)

        print(f"  Game {game_num+1}: {a_name} @ {h_name}  "
              f"[h_idx={h_idx}, a_idx={a_idx}, hp={hp_idx}, ap={ap_idx}]",
              file=sys.stderr)

        # ── Monte Carlo loop ──────────────────────────────────────────────
        wins_home = 0
        exp_h_sum, exp_a_sum = 0.0, 0.0
        home_rl_cover = 0
        away_rl_cover = 0
        overs = 0
        total_line = 11.5

        for _ in range(n_sims):
            d = rng.integers(0, n_draws)
            base_park_eff = beta_park[d] * base_pf
            park_eff_h = base_park_eff + wind_adj_home
            park_eff_a = base_park_eff + wind_adj_away
            bp_h_eff = beta_bullpen[d] * a_bp   # home batting: away bullpen
            bp_a_eff = beta_bullpen[d] * h_bp   # away batting: home bullpen

            home_runs_sim, away_runs_sim = 0, 0
            eh, ea = 0.0, 0.0

            for k in range(4):
                log_lam_h = (int_run[d, k] + att[d, h_idx, k] + def_[d, a_idx, k]
                             + home_adv[d] + pitcher_ab[d, ap_idx] + ap_era_adj
                             + park_eff_h + bp_h_eff + platoon_h + h_att_adj)
                log_lam_a = (int_run[d, k] + att[d, a_idx, k] + def_[d, h_idx, k]
                             + pitcher_ab[d, hp_idx] + hp_era_adj
                             + park_eff_a + bp_a_eff + platoon_a + a_att_adj)
                mu_h = np.exp(log_lam_h)
                mu_a = np.exp(log_lam_a)
                eh += (k + 1) * mu_h
                ea += (k + 1) * mu_a

                if k <= 1:
                    theta = max(1e-6, theta_run[d, k])
                    p_h = theta / (theta + max(1e-8, mu_h))
                    p_a = theta / (theta + max(1e-8, mu_a))
                    home_runs_sim += (k + 1) * rng.negative_binomial(n=theta, p=p_h)
                    away_runs_sim += (k + 1) * rng.negative_binomial(n=theta, p=p_a)
                else:
                    home_runs_sim += (k + 1) * rng.poisson(lam=max(1e-8, mu_h))
                    away_runs_sim += (k + 1) * rng.poisson(lam=max(1e-8, mu_a))

            exp_h_sum += eh
            exp_a_sum += ea

            # Extra innings (bullpen pitching -> use bullpen FB sensitivity for wind)
            park_eff_h_bp = base_park_eff + wind_adj_home_bp
            park_eff_a_bp = base_park_eff + wind_adj_away_bp
            extra = 0
            while home_runs_sim == away_runs_sim and extra < 20:
                for k in range(4):
                    log_lam_h = (int_run[d, k] + att[d, h_idx, k] + def_[d, a_idx, k]
                                 + home_adv[d] + pitcher_ab[d, ap_idx]
                                 + park_eff_h_bp + bp_h_eff)
                    log_lam_a = (int_run[d, k] + att[d, a_idx, k] + def_[d, h_idx, k]
                                 + pitcher_ab[d, hp_idx]
                                 + park_eff_a_bp + bp_a_eff)
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

            margin = home_runs_sim - away_runs_sim
            if margin > 0:
                wins_home += 1
            if margin > 1.5:
                home_rl_cover += 1
            if margin < -1.5:
                away_rl_cover += 1
            if (home_runs_sim + away_runs_sim) > total_line:
                overs += 1

        # ── Aggregate results ─────────────────────────────────────────────
        N = n_sims
        win_prob = wins_home / N
        exp_h = exp_h_sum / N
        exp_a = exp_a_sum / N

        result = {
            "game_num": game_num + 1,
            "away": a_name,
            "home": h_name,
            "home_cid": h_cid,
            "away_cid": a_cid,
            "home_starter": hp_name,
            "away_starter": ap_name,
            "home_starter_idx": hp_idx,
            "away_starter_idx": ap_idx,
            "hp_throws": hp_hand,
            "ap_throws": ap_hand,
            "home_win_prob": win_prob,
            "away_win_prob": 1 - win_prob,
            "ml_home": prob_to_american(win_prob),
            "ml_away": prob_to_american(1 - win_prob),
            "exp_home": exp_h,
            "exp_away": exp_a,
            "exp_total": exp_h + exp_a,
            "home_rl_cover": home_rl_cover / N,
            "away_rl_cover": away_rl_cover / N,
            "over_prob": overs / N,
            "park_factor": pf,
            "wind_adj_raw": round(wind_adj_raw, 4),
            "non_wind_adj": round(non_wind_adj, 4),
            "weather_adj": round(weather_adj, 4),
            "hp_fb_sens": round(hp_fb_sens, 3),
            "ap_fb_sens": round(ap_fb_sens, 3),
            "hp_bp_fb_sens": round(hp_bp_fb_sens, 3),
            "ap_bp_fb_sens": round(ap_bp_fb_sens, 3),
            "temp_f": temp_f,
            "wind_mph": wind_mph,
            "wind_out_mph": wind_out_mph,
            "wind_out_lf": wind_out_lf,
            "wind_out_cf": wind_out_cf,
            "wind_out_rf": wind_out_rf,
            "weather_mode": weather_mode if weather_mode else None,
            "hp_d1b_adj": round(hp_era_adj, 4) if hp_era_adj != 0 else None,
            "ap_d1b_adj": round(ap_era_adj, 4) if ap_era_adj != 0 else None,
            "hp_d1b_src": hp_adj_src if hp_era_adj != 0 else None,
            "ap_d1b_src": ap_adj_src if ap_era_adj != 0 else None,
            "home_wrc_adj": round(h_att_adj, 4) if h_att_adj != 0 else None,
            "away_wrc_adj": round(a_att_adj, 4) if a_att_adj != 0 else None,
            "platoon_adj_home": round(platoon_h, 4),
            "platoon_adj_away": round(platoon_a, 4),
        }
        all_results.append(result)

    return pd.DataFrame(all_results)


# ── Field access helpers (dict or pd.Series, handle NaN/empty) ───────────

def _safe_str(row, key: str, default: str = "") -> str:
    if isinstance(row, dict):
        val = row.get(key, default)
    else:
        val = row.get(key, default) if key in row.index else default
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    s = str(val).strip()
    return s if s else default


def _safe_int(row, key: str, default: int = 0) -> int:
    s = _safe_str(row, key, "")
    if not s:
        return default
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return default


def _safe_float(row, key: str, default: float = 0.0) -> float:
    s = _safe_str(row, key, "")
    if not s:
        return default
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def _safe_float_or_none(row, key: str):
    s = _safe_str(row, key, "")
    if not s:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# ── Output Formatting ────────────────────────────────────────────────────────

def format_predictions(predictions: pd.DataFrame, date: str) -> None:
    """Print formatted prediction cards to stdout."""
    results = predictions.to_dict("records")

    # Sort by confidence (biggest edge first)
    results.sort(key=lambda r: abs(r["home_win_prob"] - 0.5), reverse=True)

    n_draws_label = "posterior"

    print(f"\n{'='*90}")
    print(f"  NCAA BASEBALL PREDICTIONS -- {date}")
    print(f"  {len(results)} games")
    print(f"{'='*90}\n")

    # ── Matchup cards with starters ──────────────────────────────────────
    for r in results:
        conf = max(r["home_win_prob"], r["away_win_prob"])
        if conf >= 0.75:
            tier = "***"
        elif conf >= 0.65:
            tier = " **"
        elif abs(r["home_win_prob"] - 0.5) >= 0.05:
            tier = "  +"
        else:
            tier = "  o"

        fav_name = r["home"] if r["home_win_prob"] > 0.5 else r["away"]
        fav_prob = max(r["home_win_prob"], r["away_win_prob"])

        print(f"  {tier}  {r['away']:>22s}  @  {r['home']:<22s}   {fav_name} {fav_prob:.0%}")

        # Starters line (with handedness)
        as_lbl = r["away_starter"] if r["away_starter"] != "unknown" else "??"
        hs_lbl = r["home_starter"] if r["home_starter"] != "unknown" else "??"
        as_model = "Y" if r["away_starter_idx"] > 0 else "N"
        hs_model = "Y" if r["home_starter_idx"] > 0 else "N"
        as_hand = f" ({r['ap_throws']}HP)" if r.get("ap_throws") else ""
        hs_hand = f" ({r['hp_throws']}HP)" if r.get("hp_throws") else ""
        plat_tag = ""
        if r.get("platoon_adj_home", 0) != 0 or r.get("platoon_adj_away", 0) != 0:
            plat_parts = []
            if r.get("platoon_adj_home", 0) != 0:
                plat_parts.append(f"home={r['platoon_adj_home']:+.3f}")
            if r.get("platoon_adj_away", 0) != 0:
                plat_parts.append(f"away={r['platoon_adj_away']:+.3f}")
            plat_tag = f"  platoon[{', '.join(plat_parts)}]"
        print(f"       SP: {as_lbl}{as_hand} [{as_model}]  vs  {hs_lbl}{hs_hand} [{hs_model}]{plat_tag}")

        # Weather line
        if r.get("temp_f") is not None:
            wind_str = f"wind out {r['wind_out_mph']:+.0f}mph" if r.get("wind_out_mph") is not None else ""
            fb_str = f"  fb[hp={r['hp_fb_sens']:.2f},ap={r['ap_fb_sens']:.2f}]" if r.get("hp_fb_sens") else ""
            print(f"       Wx: {r['temp_f']:.0f}F  {wind_str}  (adj={r['weather_adj']:+.4f}){fb_str}")
        else:
            print(f"       Wx: unavailable")

        # Line
        print(f"       ML: Home {r['ml_home']:+d}  Away {r['ml_away']:+d}  |  "
              f"Total {r['exp_total']:.1f}  O/U {r['over_prob']:.0%}")
        print()

    # ── Summary table ────────────────────────────────────────────────────
    print(f"{'_'*90}")
    print(f"  SUMMARY TABLE")
    print(f"{'_'*90}")
    print(f"  {'#':>2s}  {'Away':>20s}  {'A.SP':>12s}  {'@':>1s}  {'Home':>20s}  {'H.SP':>12s}  "
          f"{'P(H)':>6s}  {'ML.H':>6s}  {'ML.A':>6s}  {'Tot':>5s}  {'Wx':>5s}")
    print(f"  {'--':>2s}  {'----':>20s}  {'----':>12s}  {'-':>1s}  {'----':>20s}  {'----':>12s}  "
          f"{'----':>6s}  {'----':>6s}  {'----':>6s}  {'---':>5s}  {'--':>5s}")
    for r in sorted(results, key=lambda x: x["game_num"]):
        wp = r["home_win_prob"]
        as_short = r["away_starter"][:12] if r["away_starter"] != "unknown" else "??"
        hs_short = r["home_starter"][:12] if r["home_starter"] != "unknown" else "??"
        temp = f"{r['temp_f']:.0f}" if r.get("temp_f") is not None else "  - "
        print(f"  {r['game_num']:2d}  {r['away']:>20s}  {as_short:>12s}  @  {r['home']:>20s}  {hs_short:>12s}  "
              f"{wp:>5.0%}  {r['ml_home']:>+6d}  {r['ml_away']:>+6d}  {r['exp_total']:>5.1f}  {temp:>5s}")

    # ── Starter coverage report ──────────────────────────────────────────
    n_sp_posterior = sum(1 for r in results if r["home_starter_idx"] > 0) + \
                     sum(1 for r in results if r["away_starter_idx"] > 0)
    n_sp_d1b = sum(1 for r in results if r.get("hp_d1b_adj")) + \
               sum(1 for r in results if r.get("ap_d1b_adj"))
    n_sp_total = 2 * len(results)
    n_wx = sum(1 for r in results if r.get("temp_f") is not None)
    print(f"\n  Coverage: starters {n_sp_posterior}/{n_sp_total} posterior | "
          f"{n_sp_d1b}/{n_sp_total} D1B-fallback | "
          f"weather {n_wx}/{len(results)} games")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pure Monte Carlo simulation engine for NCAA baseball predictions."
    )
    parser.add_argument("--schedule", type=Path, required=True,
                        help="Schedule CSV (game_num, home_cid, away_cid, ...)")
    parser.add_argument("--starters", type=Path, required=True,
                        help="Starters CSV (game_num, home_starter, ...)")
    parser.add_argument("--weather", type=Path, required=True,
                        help="Weather CSV (game_num, park_factor, wind_adj_raw, ...)")
    parser.add_argument("--posterior", type=Path,
                        default=Path("data/processed/run_event_posterior_2k.csv"),
                        help="Posterior draws CSV")
    parser.add_argument("--meta", type=Path,
                        default=Path("data/processed/run_event_fit_meta.json"),
                        help="Model metadata JSON")
    parser.add_argument("--team-table", type=Path,
                        default=Path("data/processed/team_table.csv"),
                        help="Team table CSV with bullpen_adj and team_idx")
    parser.add_argument("--N", type=int, default=5000,
                        help="Number of simulations per game")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output CSV path")
    parser.add_argument("--date", type=str, default=None,
                        help="Date label for formatted output (YYYY-MM-DD)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress formatted output, only write CSV")
    args = parser.parse_args()

    # Validate inputs
    for label, p in [("schedule", args.schedule), ("starters", args.starters),
                     ("weather", args.weather), ("posterior", args.posterior),
                     ("meta", args.meta), ("team-table", args.team_table)]:
        if not p.exists():
            print(f"Missing {label}: {p}", file=sys.stderr)
            return 1

    # Infer date from schedule path if not specified
    date_label = args.date
    if not date_label:
        # Try to extract from path like data/daily/2026-03-14/schedule.csv
        parts = args.schedule.parts
        for part in parts:
            if len(part) == 10 and part[4] == "-" and part[7] == "-":
                date_label = part
                break
        if not date_label:
            date_label = "unknown"

    # Run simulation
    predictions = simulate_games(
        schedule_csv=args.schedule,
        starters_csv=args.starters,
        weather_csv=args.weather,
        posterior_csv=args.posterior,
        meta_json=args.meta,
        team_table_csv=args.team_table,
        n_sims=args.N,
        seed=args.seed,
    )

    # Save CSV
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.out, index=False)
        print(f"\n  Saved {len(predictions)} games to {args.out}", file=sys.stderr)

    # Print formatted output
    if not args.quiet:
        format_predictions(predictions, date_label)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
