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

import _bootstrap  # noqa: F401
from ncaa_baseball.model_runtime import (
    FATIGUE_POLICY_CHOICES,
    SCORING_CALIBRATION,
    assert_scoring_calibration_parity,
    enforce_fatigue_coverage_policy,
)

# ── Scoring constants ────────────────────────────────────────────────────────

# Run-event multipliers: run_1=1, run_2=2, run_3=3, run_4=5.4
# run_4 represents "4+ runs in an inning". Actual avg is ~5.4 runs
# when 4+ score (includes 5, 6, 7+ run innings capped at run_4 count).
RUN_MULT = [1, 2, 3, 5.4]

# Script-level alias kept for backward compatibility and explicit parity checks.
SIMULATE_SCORING_CALIBRATION = SCORING_CALIBRATION
assert_scoring_calibration_parity("simulate.py", SIMULATE_SCORING_CALIBRATION)

# LHP platoon adjustment (log-rate): positive = teams score more runs facing LHP.
# LHP platoon adjustment (log-rate): positive = teams score more runs facing LHP.
# This is the AVERAGE effect across all teams. Individual teams are scaled by
# their actual RHB fraction relative to the league average.
from platoon_adjustment import DEFAULT_LHP_ADJ as _PLATOON_CHECK
PLATOON_LHP_ADJ = 0.03
assert abs(PLATOON_LHP_ADJ - _PLATOON_CHECK) < 1e-12, (
    f"simulate.py PLATOON_LHP_ADJ={PLATOON_LHP_ADJ} != "
    f"platoon_adjustment.DEFAULT_LHP_ADJ={_PLATOON_CHECK}"
)

# Default bullpen LHP fraction when team data unavailable (~NCAA average).
PLATOON_NCAA_BP_LHP_FRAC = 0.30

# League average effective RHB fraction (from sidearm rosters).
# Used to scale team-specific platoon: team_adj = LHP_ADJ * (team_rhb / LEAGUE_RHB)
LEAGUE_AVG_EFFECTIVE_RHB = 0.696

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


def _logit(p: float) -> float:
    x = min(1.0 - 1e-6, max(1e-6, float(p)))
    return float(np.log(x / (1.0 - x)))


def _inv_logit(x: float) -> float:
    z = float(np.exp(np.clip(x, -20.0, 20.0)))
    return z / (1.0 + z)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


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
    # Apply global scoring calibration to intercepts (corrects for Stan shrinkage)
    int_run += SCORING_CALIBRATION
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
    ha_target: float | None = None,
    fatigue_csv: Path | None = None,
    fatigue_policy: str = "de-risk",
    fatigue_min_coverage: float = 0.8,
    context_csv: Path | None = None,
) -> pd.DataFrame:
    """
    Pure Monte Carlo simulation. No API calls. Deterministic.

    Reads pre-resolved schedule, starters, and weather CSVs plus the
    Stan posterior and team table. Returns DataFrame with one row per game.

    ha_target: if set, shift home_advantage posterior mean to this value.
               Use 0.05 for ~53-54% NCAA home win rate. None = no correction.
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

    # ── Fix 1: Post-hoc home advantage correction ────────────────────────
    if ha_target is not None:
        ha_mean = float(home_adv.mean())
        if abs(ha_mean - ha_target) > 0.005:
            ha_shift = ha_mean - ha_target
            home_adv -= ha_shift
            print(f"  HA correction: {ha_mean:.4f} → {home_adv.mean():.4f}", file=sys.stderr)

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

    # ── Load bullpen fatigue adjustments (optional) ─────────────────────
    fatigue_map: dict[str, float] = {}  # canonical_id -> fatigue_adj (log-rate)
    if fatigue_csv is not None and Path(fatigue_csv).exists():
        fat_df = pd.read_csv(fatigue_csv, dtype=str)
        for _, r in fat_df.iterrows():
            cid = str(r.get("canonical_id", "")).strip()
            adj = float(r.get("fatigue_adj", 0))
            if cid:
                fatigue_map[cid] = adj
        n_fatigued = sum(1 for v in fatigue_map.values() if v > 0)
        print(f"  Bullpen fatigue: {len(fatigue_map)} teams loaded, {n_fatigued} with positive adj",
              file=sys.stderr)

    # ── Load input CSVs ──────────────────────────────────────────────────
    schedule = pd.read_csv(schedule_csv, dtype=str)
    starters = pd.read_csv(starters_csv, dtype=str)
    weather = pd.read_csv(weather_csv, dtype=str)

    # Index by game_num for fast lookup
    starters_by_game = {int(r["game_num"]): r for _, r in starters.iterrows()}
    weather_by_game = {int(r["game_num"]): r for _, r in weather.iterrows()}

    # ── Fatigue coverage contract ────────────────────────────────────────
    required_teams = set(schedule["home_cid"].dropna().astype(str).str.strip()) | set(
        schedule["away_cid"].dropna().astype(str).str.strip()
    )
    fatigue_decision = enforce_fatigue_coverage_policy(
        required_team_ids=required_teams,
        fatigue_team_ids=set(fatigue_map.keys()),
        policy=fatigue_policy,
        min_coverage=fatigue_min_coverage,
        context_label="simulate",
    )
    print(f"  {fatigue_decision.message}", file=sys.stderr)
    if fatigue_decision.action == "de-risk":
        fatigue_map = {}

    # ── Load game context (rest, day/night, surface, travel, form) ─────
    context_by_game: dict[str, dict] = {}
    if context_csv is not None and Path(context_csv).exists():
        ctx_df = pd.read_csv(context_csv, dtype=str)
        for _, r in ctx_df.iterrows():
            gn = str(r.get("game_num", "")).strip()
            if gn:
                context_by_game[gn] = r.to_dict()
        print(f"  Game context: {len(context_by_game)} games loaded "
              f"(rest, day/night, surface, travel, form)", file=sys.stderr)

    # ── Constants ─────────────────────────────────────────────────────────
    DEFAULT_STARTER_IP = 5.5

    pass  # constants moved to module level

    # ── Simulate each game ───────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    all_results = []

    for _, sched_row in schedule.iterrows():
        game_num = int(sched_row["game_num"])
        h_cid = str(sched_row["home_cid"]).strip()
        a_cid = str(sched_row["away_cid"]).strip()
        h_name = str(sched_row["home_name"]).strip()
        a_name = str(sched_row["away_name"]).strip()
        mkt_anchor_weight = _safe_float(sched_row, "mkt_anchor_weight", 0.0)
        mkt_home_win_prob = _safe_float_or_none(sched_row, "mkt_home_win_prob")
        mkt_total_line = _safe_float_or_none(sched_row, "mkt_total_line")
        time_to_start_min = _safe_float_or_none(sched_row, "time_to_start_min")
        start_utc = _safe_str(sched_row, "start_utc", "")

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
        hp_expected_ip = _safe_float(st, "hp_expected_ip", DEFAULT_STARTER_IP)
        ap_expected_ip = _safe_float(st, "ap_expected_ip", DEFAULT_STARTER_IP)
        hp_expected_ip = float(np.clip(hp_expected_ip, 3.5, 7.5))
        ap_expected_ip = float(np.clip(ap_expected_ip, 3.5, 7.5))
        hp_starter_ip_frac = hp_expected_ip / 9.0
        ap_starter_ip_frac = ap_expected_ip / 9.0
        hp_bullpen_ip_frac = 1.0 - hp_starter_ip_frac
        ap_bullpen_ip_frac = 1.0 - ap_starter_ip_frac
        home_res_method = _safe_str(st, "home_resolution_method", "")
        away_res_method = _safe_str(st, "away_resolution_method", "")
        home_d1b_fallback = _safe_int(st, "home_d1b_fallback", 0)
        away_d1b_fallback = _safe_int(st, "away_d1b_fallback", 0)
        hp_confirmed = _safe_int(st, "hp_confirmed", 0)
        ap_confirmed = _safe_int(st, "ap_confirmed", 0)

        # Offense adjustments (wRC+ for non-model teams)
        h_att_adj = _safe_float(st, "home_wrc_adj", 0.0)
        a_att_adj = _safe_float(st, "away_wrc_adj", 0.0)

        # Batting fly ball factor (team batting FB% / league avg FB%)
        # Scales wind effect for the batting team: high-FB teams benefit more from tailwind
        h_bat_fb = _safe_float(st, "home_batting_fb", 1.0)
        a_bat_fb = _safe_float(st, "away_batting_fb", 1.0)

        # ── Platoon (IP-weighted: starter + bullpen) ──────────────────────
        # Home batters face: away starter (ap_starter_ip_frac) + away bullpen (ap_bullpen_ip_frac)
        # Away batters face: home starter (hp_starter_ip_frac) + home bullpen (hp_bullpen_ip_frac)
        # hp_hand/ap_hand already read above from starters.csv hp_throws/ap_throws.
        away_bp_lhp = _safe_float(st, "away_bp_lhp_frac", PLATOON_NCAA_BP_LHP_FRAC)
        home_bp_lhp = _safe_float(st, "home_bp_lhp_frac", PLATOON_NCAA_BP_LHP_FRAC)
        # Bilateral platoon: scale LHP effect by batting team's actual RHB composition.
        # Teams with more RHB get a bigger platoon boost vs LHP (and vice versa).
        # team_rhb_scale = team_effective_rhb / league_avg_rhb
        h_pct_rhb = _safe_float(st, "home_pct_rhb", LEAGUE_AVG_EFFECTIVE_RHB)
        a_pct_rhb = _safe_float(st, "away_pct_rhb", LEAGUE_AVG_EFFECTIVE_RHB)
        h_rhb_scale = h_pct_rhb / LEAGUE_AVG_EFFECTIVE_RHB  # >1 if more RHB than avg
        a_rhb_scale = a_pct_rhb / LEAGUE_AVG_EFFECTIVE_RHB
        # Starter platoon: base LHP_ADJ scaled by batting team's RHB composition
        # platoon_h = effect on HOME runs when facing AWAY pitcher
        # Home batters face away pitcher → scale by HOME team's RHB%
        ap_starter_plat = (PLATOON_LHP_ADJ * h_rhb_scale) if ap_hand == "L" else 0.0
        hp_starter_plat = (PLATOON_LHP_ADJ * a_rhb_scale) if hp_hand == "L" else 0.0
        # Bullpen platoon: also scaled by batting team's RHB composition
        ap_bp_plat = PLATOON_LHP_ADJ * away_bp_lhp * h_rhb_scale
        hp_bp_plat = PLATOON_LHP_ADJ * home_bp_lhp * a_rhb_scale
        platoon_h = ap_starter_plat * ap_starter_ip_frac + ap_bp_plat * ap_bullpen_ip_frac
        platoon_a = hp_starter_plat * hp_starter_ip_frac + hp_bp_plat * hp_bullpen_ip_frac
        # Bullpen-only platoon for extra innings (starter is out, only BP LHP frac matters)
        platoon_h_bp = ap_bp_plat
        platoon_a_bp = hp_bp_plat

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
        weather_status = _safe_str(wx, "weather_status", "")
        weather_error = _safe_str(wx, "weather_error", "")
        rain_chance_pct = _safe_float_or_none(wx, "rain_chance_pct")

        # Display-level weather adj (average of both starters for summary)
        weather_adj = wind_adj_raw * (hp_fb_sens + ap_fb_sens) / 2.0 + non_wind_adj

        # Park + weather decomposition
        base_pf = pf

        # Home scoring: away pitcher on mound, home team batting
        # Wind effect = wind_raw × pitcher_FB_sens × batting_team_FB_factor
        ap_blended_sens = ap_starter_ip_frac * ap_fb_sens + ap_bullpen_ip_frac * ap_bp_fb_sens
        wind_adj_home = wind_adj_raw * ap_blended_sens * h_bat_fb

        # Away scoring: home pitcher on mound, away team batting
        hp_blended_sens = hp_starter_ip_frac * hp_fb_sens + hp_bullpen_ip_frac * hp_bp_fb_sens
        wind_adj_away = wind_adj_raw * hp_blended_sens * a_bat_fb

        # Pure bullpen wind (for extra innings)
        wind_adj_home_bp = wind_adj_raw * ap_bp_fb_sens * h_bat_fb
        wind_adj_away_bp = wind_adj_raw * hp_bp_fb_sens * a_bat_fb

        # Bullpen quality
        h_bp = bp_map.get(h_cid, 0.0)
        a_bp = bp_map.get(a_cid, 0.0)

        # Bullpen fatigue: fatigued bullpen → opponent scores more
        # h_fatigue_adj applied when home bullpen pitches (away team batting)
        # a_fatigue_adj applied when away bullpen pitches (home team batting)
        h_fatigue_adj = fatigue_map.get(h_cid, 0.0)
        a_fatigue_adj = fatigue_map.get(a_cid, 0.0)

        # Dynamic bullpen availability: penalty when top arms pitched in last 2 days.
        # Blends with rolling fatigue to capture WHICH arms are unavailable, not just volume.
        # Positive = opponent scores more (key relievers are tired/used up).
        h_bp_avail_adj = _safe_float(st, "home_bp_avail_adj", 0.0)
        a_bp_avail_adj = _safe_float(st, "away_bp_avail_adj", 0.0)
        h_fatigue_adj = h_fatigue_adj + h_bp_avail_adj
        a_fatigue_adj = a_fatigue_adj + a_bp_avail_adj

        # ── Game context adjustments (rest, day/night, surface, travel, form) ──
        ctx = context_by_game.get(str(game_num), {})
        home_context_adj = _safe_float(ctx, "home_context_adj", 0.0)
        away_context_adj = _safe_float(ctx, "away_context_adj", 0.0)

        print(f"  Game {game_num}: {a_name} @ {h_name}  "
              f"[h_idx={h_idx}, a_idx={a_idx}, hp={hp_idx}, ap={ap_idx}]",
              file=sys.stderr)

        # ── Market anchor adjustment (time-aware, pilot calibrated) ────────
        anchor_home_shift = 0.0
        anchor_away_shift = 0.0
        pilot_home_prob = None
        pilot_total = None
        if mkt_anchor_weight > 0 and (mkt_home_win_prob is not None or mkt_total_line is not None):
            n_pilot = int(max(250, min(800, n_sims // 8)))
            pilot_wins = 0
            pilot_h_sum = 0.0
            pilot_a_sum = 0.0
            for _ in range(n_pilot):
                d = rng.integers(0, n_draws)
                base_park_eff = beta_park[d] * base_pf
                park_eff_h = base_park_eff + non_wind_adj + wind_adj_home
                park_eff_a = base_park_eff + non_wind_adj + wind_adj_away
                bp_h_eff = beta_bullpen[d] * a_bp + a_fatigue_adj  # away BP fatigued → home scores more
                bp_a_eff = beta_bullpen[d] * h_bp + h_fatigue_adj  # home BP fatigued → away scores more
                home_runs_sim, away_runs_sim = 0, 0
                eh, ea = 0.0, 0.0
                for k in range(4):
                    log_lam_h = (int_run[d, k] + att[d, h_idx, k] + def_[d, a_idx, k]
                                 + home_adv[d] + pitcher_ab[d, ap_idx] + ap_era_adj
                                 + park_eff_h + bp_h_eff + platoon_h + h_att_adj
                                 + home_context_adj)
                    log_lam_a = (int_run[d, k] + att[d, a_idx, k] + def_[d, h_idx, k]
                                 + pitcher_ab[d, hp_idx] + hp_era_adj
                                 + park_eff_a + bp_a_eff + platoon_a + a_att_adj
                                 + away_context_adj)
                    mu_h = np.exp(log_lam_h)
                    mu_a = np.exp(log_lam_a)
                    eh += RUN_MULT[k] * mu_h
                    ea += RUN_MULT[k] * mu_a
                    if k <= 1:
                        theta = max(1e-6, theta_run[d, k])
                        p_h = theta / (theta + max(1e-8, mu_h))
                        p_a = theta / (theta + max(1e-8, mu_a))
                        home_runs_sim += RUN_MULT[k] * rng.negative_binomial(n=theta, p=p_h)
                        away_runs_sim += RUN_MULT[k] * rng.negative_binomial(n=theta, p=p_a)
                    else:
                        home_runs_sim += RUN_MULT[k] * rng.poisson(lam=max(1e-8, mu_h))
                        away_runs_sim += RUN_MULT[k] * rng.poisson(lam=max(1e-8, mu_a))
                if home_runs_sim > away_runs_sim:
                    pilot_wins += 1
                pilot_h_sum += eh
                pilot_a_sum += ea
            pilot_home_prob = pilot_wins / max(1, n_pilot)
            pilot_total = (pilot_h_sum + pilot_a_sum) / max(1, n_pilot)

            total_shift = 0.0
            side_shift = 0.0
            if mkt_total_line is not None and pilot_total and pilot_total > 0:
                # Anchor toward market total — stronger caps to let market pull harder
                total_shift = _clamp(
                    mkt_anchor_weight * np.log(max(0.01, float(mkt_total_line)) / pilot_total) / 2.0,
                    -0.40,
                    0.40,
                )
            if mkt_home_win_prob is not None:
                # Anchor toward market side — stronger caps for more market influence
                side_shift = _clamp(
                    mkt_anchor_weight * (_logit(float(mkt_home_win_prob)) - _logit(pilot_home_prob or 0.5)) / 2.0,
                    -0.50,
                    0.50,
                )
            anchor_home_shift = total_shift + side_shift
            anchor_away_shift = total_shift - side_shift

        # ── Monte Carlo loop ──────────────────────────────────────────────
        wins_home = 0
        exp_h_sum, exp_a_sum = 0.0, 0.0
        home_rl_cover = 0
        away_rl_cover = 0
        rl_steps = [2, 3, 4, 5, 6]
        home_win_by: dict[int, int] = {k: 0 for k in rl_steps}
        away_win_by: dict[int, int] = {k: 0 for k in rl_steps}
        overs = 0
        total_line = 11.5
        home_runs_mc = np.zeros(n_sims, dtype=np.int16)
        away_runs_mc = np.zeros(n_sims, dtype=np.int16)
        total_runs_mc = np.zeros(n_sims, dtype=np.int16)

        for i in range(n_sims):
            d = rng.integers(0, n_draws)
            base_park_eff = beta_park[d] * base_pf
            park_eff_h = base_park_eff + non_wind_adj + wind_adj_home
            park_eff_a = base_park_eff + non_wind_adj + wind_adj_away
            bp_h_eff = beta_bullpen[d] * a_bp + a_fatigue_adj  # home batting: away bullpen
            bp_a_eff = beta_bullpen[d] * h_bp + h_fatigue_adj  # away batting: home bullpen

            home_runs_sim, away_runs_sim = 0, 0
            eh, ea = 0.0, 0.0

            for k in range(4):
                log_lam_h = (int_run[d, k] + att[d, h_idx, k] + def_[d, a_idx, k]
                             + home_adv[d] + pitcher_ab[d, ap_idx] + ap_era_adj
                             + park_eff_h + bp_h_eff + platoon_h + h_att_adj
                             + home_context_adj + anchor_home_shift)
                log_lam_a = (int_run[d, k] + att[d, a_idx, k] + def_[d, h_idx, k]
                             + pitcher_ab[d, hp_idx] + hp_era_adj
                             + park_eff_a + bp_a_eff + platoon_a + a_att_adj
                             + away_context_adj + anchor_away_shift)
                mu_h = np.exp(log_lam_h)
                mu_a = np.exp(log_lam_a)
                eh += RUN_MULT[k] * mu_h
                ea += RUN_MULT[k] * mu_a

                if k <= 1:
                    theta = max(1e-6, theta_run[d, k])
                    p_h = theta / (theta + max(1e-8, mu_h))
                    p_a = theta / (theta + max(1e-8, mu_a))
                    home_runs_sim += RUN_MULT[k] * rng.negative_binomial(n=theta, p=p_h)
                    away_runs_sim += RUN_MULT[k] * rng.negative_binomial(n=theta, p=p_a)
                else:
                    home_runs_sim += RUN_MULT[k] * rng.poisson(lam=max(1e-8, mu_h))
                    away_runs_sim += RUN_MULT[k] * rng.poisson(lam=max(1e-8, mu_a))

            exp_h_sum += eh
            exp_a_sum += ea

            # Extra innings (bullpen pitching -> use bullpen FB sensitivity for wind)
            park_eff_h_bp = base_park_eff + non_wind_adj + wind_adj_home_bp
            park_eff_a_bp = base_park_eff + non_wind_adj + wind_adj_away_bp
            extra = 0
            while home_runs_sim == away_runs_sim and extra < 20:
                for k in range(4):
                    # Extra innings are bullpen-only: no starter ability/platoon,
                    # but bullpen platoon (LHP frac), wRC+ offense, and context still apply.
                    log_lam_h = (int_run[d, k] + att[d, h_idx, k] + def_[d, a_idx, k]
                                 + home_adv[d]
                                 + park_eff_h_bp + bp_h_eff + platoon_h_bp + h_att_adj
                                 + home_context_adj + anchor_home_shift)
                    log_lam_a = (int_run[d, k] + att[d, a_idx, k] + def_[d, h_idx, k]
                                 + park_eff_a_bp + bp_a_eff + platoon_a_bp + a_att_adj
                                 + away_context_adj + anchor_away_shift)
                    mu_h = np.exp(log_lam_h) / 9.0
                    mu_a = np.exp(log_lam_a) / 9.0
                    if k <= 1:
                        theta = max(1e-6, theta_run[d, k])
                        p_h = theta / (theta + max(1e-8, mu_h))
                        p_a = theta / (theta + max(1e-8, mu_a))
                        home_runs_sim += RUN_MULT[k] * rng.negative_binomial(n=theta, p=p_h)
                        away_runs_sim += RUN_MULT[k] * rng.negative_binomial(n=theta, p=p_a)
                    else:
                        home_runs_sim += RUN_MULT[k] * rng.poisson(lam=max(1e-8, mu_h))
                        away_runs_sim += RUN_MULT[k] * rng.poisson(lam=max(1e-8, mu_a))
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
            for k in rl_steps:
                if margin >= k:
                    home_win_by[k] += 1
                if margin <= -k:
                    away_win_by[k] += 1
            if (home_runs_sim + away_runs_sim) > total_line:
                overs += 1
            home_runs_mc[i] = int(home_runs_sim)
            away_runs_mc[i] = int(away_runs_sim)
            total_runs_mc[i] = int(home_runs_sim + away_runs_sim)

        # ── Aggregate results ─────────────────────────────────────────────
        N = n_sims
        win_prob = wins_home / N
        exp_h = exp_h_sum / N
        exp_a = exp_a_sum / N
        exp_total = exp_h + exp_a
        total_p10 = float(np.quantile(total_runs_mc, 0.10))
        total_p50 = float(np.quantile(total_runs_mc, 0.50))
        total_p90 = float(np.quantile(total_runs_mc, 0.90))
        margin_mc = home_runs_mc.astype(np.int32) - away_runs_mc.astype(np.int32)
        margin_p10 = float(np.quantile(margin_mc, 0.10))
        margin_p50 = float(np.quantile(margin_mc, 0.50))
        margin_p90 = float(np.quantile(margin_mc, 0.90))
        win_se = float(np.sqrt(max(1e-8, win_prob * (1.0 - win_prob) / N)))
        home_win_ci_lo = max(0.0, win_prob - 1.96 * win_se)
        home_win_ci_hi = min(1.0, win_prob + 1.96 * win_se)

        starter_missing = int(hp_idx == 0) + int(ap_idx == 0)
        any_team_fallback = int(h_idx == 0 or a_idx == 0)
        weather_bad = 0 if (weather_status.startswith("ok") or weather_status == "") else 1
        any_d1b = int(abs(hp_era_adj) > 1e-12 or abs(ap_era_adj) > 1e-12)
        fragility = 0.0
        fragility += 0.20 * float(starter_missing)
        fragility += 0.20 * float(any_team_fallback)
        fragility += 0.20 * float(weather_bad)
        fragility += 0.10 * float(any_d1b)
        fragility = _clamp(fragility, 0.0, 1.0)
        if fragility >= 0.60:
            fragility_flag = "high"
        elif fragility >= 0.30:
            fragility_flag = "medium"
        else:
            fragility_flag = "low"

        result = {
            "game_num": game_num,
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
            "exp_total": exp_total,
            "home_win_ci_lo": home_win_ci_lo,
            "home_win_ci_hi": home_win_ci_hi,
            "exp_total_p10": total_p10,
            "exp_total_p50": total_p50,
            "exp_total_p90": total_p90,
            "margin_p10": margin_p10,
            "margin_p50": margin_p50,
            "margin_p90": margin_p90,
            "home_rl_cover": home_rl_cover / N,
            "away_rl_cover": away_rl_cover / N,
            "home_win_by_2plus": home_win_by[2] / N,
            "away_win_by_2plus": away_win_by[2] / N,
            "home_win_by_3plus": home_win_by[3] / N,
            "away_win_by_3plus": away_win_by[3] / N,
            "home_win_by_4plus": home_win_by[4] / N,
            "away_win_by_4plus": away_win_by[4] / N,
            "home_win_by_5plus": home_win_by[5] / N,
            "away_win_by_5plus": away_win_by[5] / N,
            "home_win_by_6plus": home_win_by[6] / N,
            "away_win_by_6plus": away_win_by[6] / N,
            "over_prob": overs / N,
            "park_factor": pf,
            "wind_adj_raw": round(wind_adj_raw, 4),
            "non_wind_adj": round(non_wind_adj, 4),
            "weather_adj": round(weather_adj, 4),
            "hp_fb_sens": round(hp_fb_sens, 3),
            "ap_fb_sens": round(ap_fb_sens, 3),
            "hp_bp_fb_sens": round(hp_bp_fb_sens, 3),
            "ap_bp_fb_sens": round(ap_bp_fb_sens, 3),
            "home_batting_fb": round(h_bat_fb, 3),
            "away_batting_fb": round(a_bat_fb, 3),
            "hp_expected_ip": round(hp_expected_ip, 2),
            "ap_expected_ip": round(ap_expected_ip, 2),
            "hp_starter_ip_frac": round(hp_starter_ip_frac, 3),
            "ap_starter_ip_frac": round(ap_starter_ip_frac, 3),
            "home_resolution_method": home_res_method if home_res_method else None,
            "away_resolution_method": away_res_method if away_res_method else None,
            "home_d1b_fallback": int(home_d1b_fallback),
            "away_d1b_fallback": int(away_d1b_fallback),
            "hp_confirmed": int(hp_confirmed),
            "ap_confirmed": int(ap_confirmed),
            "home_bullpen_adj": round(h_bp, 4),
            "away_bullpen_adj": round(a_bp, 4),
            "home_fatigue_adj": round(h_fatigue_adj, 4),
            "away_fatigue_adj": round(a_fatigue_adj, 4),
            "temp_f": temp_f,
            "wind_mph": wind_mph,
            "wind_out_mph": wind_out_mph,
            "wind_out_lf": wind_out_lf,
            "wind_out_cf": wind_out_cf,
            "wind_out_rf": wind_out_rf,
            "weather_mode": weather_mode if weather_mode else None,
            "weather_status": weather_status if weather_status else None,
            "weather_error": weather_error if weather_error else None,
            "rain_chance_pct": rain_chance_pct,
            "hp_d1b_adj": round(hp_era_adj, 4) if hp_era_adj != 0 else None,
            "ap_d1b_adj": round(ap_era_adj, 4) if ap_era_adj != 0 else None,
            "hp_d1b_src": hp_adj_src if hp_era_adj != 0 else None,
            "ap_d1b_src": ap_adj_src if ap_era_adj != 0 else None,
            "home_wrc_adj": round(h_att_adj, 4) if h_att_adj != 0 else None,
            "away_wrc_adj": round(a_att_adj, 4) if a_att_adj != 0 else None,
            "platoon_adj_home": round(platoon_h, 4),
            "platoon_adj_away": round(platoon_a, 4),
            "fragility_score": round(fragility, 3),
            "fragility_flag": fragility_flag,
            "mkt_anchor_weight": round(float(mkt_anchor_weight), 3),
            "mkt_home_win_prob": mkt_home_win_prob,
            "mkt_total_line": mkt_total_line,
            "time_to_start_min": time_to_start_min,
            "start_utc": start_utc if start_utc else None,
            "pilot_home_win_prob": pilot_home_prob,
            "pilot_exp_total": pilot_total,
            "anchor_home_shift": round(anchor_home_shift, 4),
            "anchor_away_shift": round(anchor_away_shift, 4),
            "fatigue_policy": fatigue_decision.policy,
            "fatigue_coverage": round(fatigue_decision.coverage, 4),
            "fatigue_action": fatigue_decision.action,
            # Game context layers
            "home_context_adj": round(home_context_adj, 4),
            "away_context_adj": round(away_context_adj, 4),
            "home_rest_adj": _safe_float(ctx, "home_rest_adj", 0.0),
            "away_rest_adj": _safe_float(ctx, "away_rest_adj", 0.0),
            "day_night": str(ctx.get("day_night", "unknown")),
            "surface": str(ctx.get("surface", "grass")),
            "travel_miles": ctx.get("travel_miles"),
            "home_form_adj": _safe_float(ctx, "home_form_adj", 0.0),
            "away_form_adj": _safe_float(ctx, "away_form_adj", 0.0),
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
    parser.add_argument("--ha-target", type=float, default=0.09,
                        help="Target home_advantage mean (post-hoc correction). "
                             "0.09 ≈ 52%% equal-team home win rate. None=no correction.")
    parser.add_argument("--fatigue", type=Path, default=None,
                        help="Optional bullpen fatigue CSV (canonical_id, fatigue_adj).")
    parser.add_argument(
        "--fatigue-policy",
        type=str,
        default="de-risk",
        choices=FATIGUE_POLICY_CHOICES,
        help="How to handle low fatigue coverage: abort, de-risk, or ignore.",
    )
    parser.add_argument(
        "--fatigue-min-coverage",
        type=float,
        default=0.8,
        help="Minimum fatigue team coverage required for apply behavior [0-1].",
    )
    parser.add_argument("--context", type=Path, default=None,
                        help="Game context CSV (rest, day/night, surface, travel, form)")
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
        ha_target=args.ha_target,
        fatigue_csv=args.fatigue,
        fatigue_policy=args.fatigue_policy,
        fatigue_min_coverage=args.fatigue_min_coverage,
        context_csv=args.context,
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
