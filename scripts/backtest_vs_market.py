#!/usr/bin/env python3
"""
Comprehensive backtest: model predictions vs. market odds vs. baselines.

Joins historical odds (JSONL) with actual game results (run_events) and the
Stan posterior to produce:

  1. Model vs. Market moneyline accuracy & calibration
  2. ROI / P&L simulation at various edge thresholds
  3. Totals bias quantification (model vs. market vs. actual)
  4. Baseline comparisons (home-field-only, market consensus, model)
  5. Post-hoc calibration diagnostics (Platt scaling, market blend)
  6. Per-game detail CSV for further analysis

Fixes integrated:
  Fix 1: Post-hoc home_advantage correction (--ha-target)
  Fix 2: Spread scaling for att/def arrays (--spread-scale)
  Fix 3: Platt scaling calibration (5-fold CV)
  Fix 4: Market blend analysis (logit-space, grid search)
  Fix 5: D1B pitcher adjustment boost (--d1b-boost)

Usage:
  python3 scripts/backtest_vs_market.py
  python3 scripts/backtest_vs_market.py --ha-target 0.05 --d1b-boost 1.6
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date as date_cls, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# ── Utilities ─────────────────────────────────────────────────────────────────

def american_to_prob(ml: int | float) -> float:
    """American odds → implied probability (no vig)."""
    ml = float(ml)
    if ml < 0:
        return abs(ml) / (abs(ml) + 100.0)
    return 100.0 / (ml + 100.0)


def prob_to_american(p: float) -> int:
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


def profit_on_bet(ml: int | float, won: bool) -> float:
    """Profit per $1 wagered at American odds. Returns negative if lost."""
    if not won:
        return -1.0
    ml = float(ml)
    if ml > 0:
        return ml / 100.0
    return 100.0 / abs(ml)


def kelly_fraction(edge: float, odds_prob: float) -> float:
    """Half-Kelly fraction. edge = model_prob - implied_prob."""
    if edge <= 0 or odds_prob <= 0 or odds_prob >= 1:
        return 0.0
    b = (1.0 / odds_prob) - 1.0  # decimal payout - 1
    f = edge / (1.0 - odds_prob) if b > 0 else 0.0
    return max(0.0, min(f * 0.5, 0.10))  # half-Kelly, cap at 10%


# ── Platt Scaling (manual, no sklearn) ────────────────────────────────────────

def platt_fit(logits: np.ndarray, outcomes: np.ndarray, max_iter: int = 50) -> tuple[float, float]:
    """Fit Platt scaling parameters (a, b) via Newton-Raphson.

    Model: P(y=1 | x) = sigmoid(a * logit(p) + b)
    Returns (a, b).
    """
    a, b = 1.0, 0.0
    x = logits.copy()
    y = outcomes.astype(float)
    n = len(x)
    if n == 0:
        return 1.0, 0.0

    for _ in range(max_iter):
        s = 1.0 / (1.0 + np.exp(-(a * x + b)))
        s = np.clip(s, 1e-8, 1 - 1e-8)

        # Gradient
        diff = s - y
        grad_a = np.dot(diff, x) / n
        grad_b = diff.mean()

        # Hessian diagonal approx
        w = s * (1 - s)
        H_aa = np.dot(w, x * x) / n + 1e-8
        H_bb = w.mean() + 1e-8

        a -= grad_a / H_aa
        b -= grad_b / H_bb

    return float(a), float(b)


def platt_predict(logits: np.ndarray, a: float, b: float) -> np.ndarray:
    """Apply Platt scaling: sigmoid(a * logit + b)."""
    z = a * logits + b
    return 1.0 / (1.0 + np.exp(-z))


def platt_cv(probs: np.ndarray, outcomes: np.ndarray, n_folds: int = 5) -> np.ndarray:
    """5-fold cross-validated Platt scaling. Returns calibrated probabilities."""
    n = len(probs)
    logits = np.log(np.clip(probs, 1e-6, 1 - 1e-6) / np.clip(1 - probs, 1e-6, 1 - 1e-6))
    calibrated = np.copy(probs)

    indices = np.arange(n)
    np.random.RandomState(42).shuffle(indices)
    fold_size = n // n_folds

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        a, b = platt_fit(logits[train_idx], outcomes[train_idx])
        calibrated[test_idx] = platt_predict(logits[test_idx], a, b)

    return np.clip(calibrated, 0.01, 0.99)


# ── Murphy Brier Decomposition ────────────────────────────────────────────────

def murphy_decomposition(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> dict:
    """Murphy decomposition of Brier score: BS = Reliability - Resolution + Uncertainty."""
    y = outcomes.astype(float)
    n = len(y)
    base_rate = y.mean()
    uncertainty = base_rate * (1 - base_rate)

    # Bin predictions
    bins = np.linspace(0, 1, n_bins + 1)
    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if i == n_bins - 1:
            mask = mask | (probs == bins[i + 1])
        nk = mask.sum()
        if nk == 0:
            continue
        fk = probs[mask].mean()
        ok = y[mask].mean()
        reliability += nk * (fk - ok) ** 2
        resolution += nk * (ok - base_rate) ** 2

    reliability /= n
    resolution /= n
    brier = ((probs - y) ** 2).mean()

    return {
        "brier": float(brier),
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
    }


# ── Name Resolution (exact match only — no fuzzy) ────────────────────────────

def build_name_map(teams_csv: Path) -> dict[str, str]:
    """Build odds team name → canonical_id map from explicit columns only."""
    df = pd.read_csv(teams_csv)
    exact: dict[str, str] = {}

    for _, r in df.iterrows():
        cid = str(r["canonical_id"])
        for col in ["odds_api_name", "espn_name"]:
            val = str(r.get(col, "")).strip()
            if val and val != "nan":
                exact[val] = cid

    return exact


def resolve_odds_name(name: str, exact: dict) -> str | None:
    """Resolve an odds API team name to canonical_id via exact match."""
    return exact.get(name)


def _normalize_pitcher_name(name: str) -> str:
    """Normalize pitcher name for crosswalk matching."""
    return name.strip().lower().replace(".", "").replace("-", "").replace("'", "")


def parse_ncaa_pid(pid: str) -> tuple[str, str] | None:
    """Parse NCAA_ format pitcher ID → (name_parts, team_canonical_id).

    Format: NCAA_firstname_lastname__team_canonical_id
    or      NCAA_lastname__team_canonical_id

    Returns (normalized_name, team_canonical_id) or None if not parseable.
    """
    if not pid.startswith("NCAA_"):
        return None
    rest = pid[5:]  # strip "NCAA_"
    if "__" not in rest:
        return None
    name_part, team_part = rest.rsplit("__", 1)
    # name_part is like "connor_schlect" or "t_holman"
    name_norm = _normalize_pitcher_name(name_part.replace("_", " "))
    return name_norm, team_part


def build_pitcher_crosswalk(pitcher_table_path: Path) -> dict:
    """Build (normalized_name, team) → {pitcher_idx, d1b_adj} from pitcher_table.

    Returns dict mapping (norm_name, team_canonical_id) → dict with keys:
      - pitcher_idx: int (0 if no posterior)
      - d1b_adj: float (0.0 if no D1B data)
    Also returns a secondary dict keyed by (last_name_only, team) for fallback.
    """
    pt = pd.read_csv(pitcher_table_path, dtype=str)
    by_full_name: dict[tuple[str, str], dict] = {}
    by_last_name: dict[tuple[str, str], dict] = {}

    for _, r in pt.iterrows():
        name = str(r.get("pitcher_name", "")).strip()
        team = str(r.get("team_canonical_id", "")).strip()
        idx = int(r.get("pitcher_idx", 0))
        adj_str = r.get("d1b_ability_adj", "0")
        try:
            adj = float(adj_str)
        except (ValueError, TypeError):
            adj = 0.0

        if not name or name == "nan" or not team:
            continue

        info = {"pitcher_idx": idx, "d1b_adj": adj}
        norm = _normalize_pitcher_name(name)

        # Full name key
        by_full_name[(norm, team)] = info

        # Last name only key (for NCAA_ IDs that may only have last name)
        parts = norm.split()
        if parts:
            last = parts[-1]
            by_last_name[(last, team)] = info

    return {"full": by_full_name, "last": by_last_name}


def resolve_ncaa_pitcher(
    pid: str,
    crosswalk: dict,
) -> tuple[int, float]:
    """Resolve an NCAA_ format pitcher ID to (pitcher_idx, d1b_adj).

    Tries full name match first, then last-name-only fallback.
    Returns (0, 0.0) if no match found.
    """
    parsed = parse_ncaa_pid(pid)
    if parsed is None:
        return 0, 0.0

    norm_name, team = parsed

    # Try full name match
    info = crosswalk["full"].get((norm_name, team))
    if info:
        return info["pitcher_idx"], info["d1b_adj"]

    # Try last-name-only (NCAA_ IDs sometimes use single name like "NCAA_oughton__team")
    parts = norm_name.split()
    if parts:
        last = parts[-1]
        info = crosswalk["last"].get((last, team))
        if info:
            return info["pitcher_idx"], info["d1b_adj"]

    return 0, 0.0


# ── Simulation Engine (vectorized, from backtest_fast.py) ────────────────────

def load_posterior_arrays(
    posterior_path: Path, meta_path: Path
) -> dict:
    """Load posterior into numpy arrays for fast simulation."""
    with open(meta_path) as f:
        meta = json.load(f)
    N_teams = meta["N_teams"]
    N_pitchers = meta["N_pitchers"]

    draws_df = pd.read_csv(posterior_path)
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
            col_att = f"att_run_{k+1}[{t}]"
            col_def = f"def_run_{k+1}[{t}]"
            if col_att in draws_df.columns:
                att[:, t, k] = draws_df[col_att].values
            if col_def in draws_df.columns:
                def_[:, t, k] = draws_df[col_def].values

    pitcher_ab = np.zeros((n_draws, N_pitchers + 1))
    for p in range(1, N_pitchers + 1):
        col = f"pitcher_ability[{p}]"
        if col in draws_df.columns:
            pitcher_ab[:, p] = draws_df[col].values

    return {
        "N_teams": N_teams, "N_pitchers": N_pitchers, "n_draws": n_draws,
        "int_run": int_run, "theta_run": theta_run, "home_adv": home_adv,
        "beta_park": beta_park, "beta_bullpen": beta_bullpen,
        "att": att, "def_": def_, "pitcher_ab": pitcher_ab,
    }


def simulate_game(
    post: dict,
    h_idx: int, a_idx: int, hp_idx: int, ap_idx: int,
    park_factor: float, h_bp: float, a_bp: float,
    N: int, rng: np.random.Generator,
    hp_d1b_adj: float = 0.0, ap_d1b_adj: float = 0.0,
) -> tuple[float, float, float]:
    """Simulate a game N times. Returns (home_win_prob, exp_home, exp_away).
    hp_d1b_adj/ap_d1b_adj: log-rate adjustments for pitchers with no posterior."""
    wins_home = 0
    exp_h_sum = 0.0
    exp_a_sum = 0.0

    n_draws = post["n_draws"]
    int_run = post["int_run"]
    theta_run = post["theta_run"]
    home_adv = post["home_adv"]
    beta_park = post["beta_park"]
    beta_bullpen = post["beta_bullpen"]
    att = post["att"]
    def_ = post["def_"]
    pitcher_ab = post["pitcher_ab"]

    for _ in range(N):
        d = rng.integers(0, n_draws)
        park_eff = beta_park[d] * park_factor
        bp_h_eff = beta_bullpen[d] * a_bp
        bp_a_eff = beta_bullpen[d] * h_bp

        eh, ea = 0.0, 0.0
        home_runs_sim, away_runs_sim = 0, 0

        # Pitcher effects: posterior ability + D1B fallback for idx=0
        ap_eff = pitcher_ab[d, ap_idx] + ap_d1b_adj
        hp_eff = pitcher_ab[d, hp_idx] + hp_d1b_adj

        for k in range(4):
            log_lam_h = (int_run[d, k] + att[d, h_idx, k] + def_[d, a_idx, k]
                         + home_adv[d] + ap_eff + park_eff + bp_h_eff)
            log_lam_a = (int_run[d, k] + att[d, a_idx, k] + def_[d, h_idx, k]
                         + hp_eff + park_eff + bp_a_eff)

            mu_h = np.exp(log_lam_h)
            mu_a = np.exp(log_lam_a)
            eh += (k + 1) * mu_h
            ea += (k + 1) * mu_a

            if k <= 1:
                theta = max(1e-6, theta_run[d, k])
                p_h = theta / (theta + max(1e-8, mu_h))
                p_a = theta / (theta + max(1e-8, mu_a))
                count_h = rng.negative_binomial(n=theta, p=p_h)
                count_a = rng.negative_binomial(n=theta, p=p_a)
            else:
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
                             + home_adv[d] + ap_eff + park_eff + bp_h_eff)
                log_lam_a = (int_run[d, k] + att[d, a_idx, k] + def_[d, h_idx, k]
                             + hp_eff + park_eff + bp_a_eff)
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

    return wins_home / N, exp_h_sum / N, exp_a_sum / N


# ── Reporting ────────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def report_calibration(df: pd.DataFrame, prob_col: str, label: str) -> None:
    """Print calibration table for a probability column."""
    actual = df["home_won"].astype(float)
    print(f"\n  {label} calibration (home win prob bins):")
    print(f"  {'Bin':>12s}  {'N':>5s}  {'Pred':>8s}  {'Actual':>8s}  {'Gap':>8s}")
    for lo, hi in [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]:
        mask = (df[prob_col] >= lo) & (df[prob_col] < hi)
        n = mask.sum()
        if n > 0:
            pred_avg = df.loc[mask, prob_col].mean()
            actual_avg = actual[mask].mean()
            gap = actual_avg - pred_avg
            print(f"  [{lo:.1f}-{hi:.1f})  {n:5d}  {pred_avg:8.3f}  {actual_avg:8.3f}  {gap:+8.3f}")


def report_roi(df: pd.DataFrame, prob_col: str, label: str) -> None:
    """Report ROI at various edge thresholds."""
    print(f"\n  {label} — ROI by edge threshold (flat $1 bets):")
    print(f"  {'Threshold':>10s}  {'Bets':>5s}  {'Won':>5s}  {'Win%':>6s}  {'P&L':>8s}  {'ROI':>8s}  {'Kelly ROI':>10s}")

    for thresh in [0.03, 0.05, 0.08, 0.10, 0.15]:
        # Find games where model disagrees with market
        edge_col = df[prob_col] - df["market_home_prob"]
        mask = edge_col.abs() >= thresh

        sub = df[mask].copy()
        if len(sub) == 0:
            print(f"  {thresh:>9.0%}  {'--':>5s}")
            continue

        # Bet on whichever side the model favors
        sub_edge = edge_col[mask]
        sub["bet_home"] = sub_edge > 0
        sub["bet_won"] = (
            (sub["bet_home"] & sub["home_won"]) |
            (~sub["bet_home"] & ~sub["home_won"])
        )

        # Flat bet P&L: use best available market odds
        flat_pnl = 0.0
        kelly_pnl = 0.0
        kelly_bankroll = 1.0

        for _, row in sub.iterrows():
            if row["bet_home"]:
                ml = row["best_home_ml"]
                model_p = row[prob_col]
                implied_p = row["market_home_prob"]
            else:
                ml = row["best_away_ml"]
                model_p = 1 - row[prob_col]
                implied_p = 1 - row["market_home_prob"]

            won = bool(row["bet_won"])
            flat_pnl += profit_on_bet(ml, won)

            # Kelly sizing
            edge = model_p - implied_p
            kf = kelly_fraction(edge, implied_p)
            wager = kelly_bankroll * kf
            kelly_bankroll += wager * profit_on_bet(ml, won)

        n_bets = len(sub)
        n_won = sub["bet_won"].sum()
        win_pct = n_won / n_bets
        flat_roi = flat_pnl / n_bets
        kelly_roi = (kelly_bankroll - 1.0)

        print(f"  {thresh:>9.0%}  {n_bets:5d}  {n_won:5d}  {win_pct:5.1%}  {flat_pnl:+8.2f}  {flat_roi:+7.1%}  {kelly_roi:+9.1%}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest model vs. market odds.")
    parser.add_argument("--odds", type=Path, default=Path("data/raw/odds/odds_historical_2026.jsonl"))
    parser.add_argument("--run-events", type=Path, default=Path("data/processed/run_events_expanded.csv"))
    parser.add_argument("--posterior", type=Path, default=Path("data/processed/run_event_posterior_2k.csv"))
    parser.add_argument("--meta", type=Path, default=Path("data/processed/run_event_fit_meta.json"))
    parser.add_argument("--team-index", type=Path, default=Path("data/processed/run_event_team_index.csv"))
    parser.add_argument("--pitcher-index", type=Path, default=Path("data/processed/run_event_pitcher_index.csv"))
    parser.add_argument("--teams-csv", type=Path, default=Path("data/registries/canonical_teams_2026.csv"))
    parser.add_argument("--team-table", type=Path, default=Path("data/processed/team_table.csv"))
    parser.add_argument("--N", type=int, default=2000, help="Simulations per game")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/processed/backtest_vs_market.csv"))

    # ── Fix parameters ────────────────────────────────────────────────────
    parser.add_argument("--ha-target", type=float, default=0.05,
                        help="Target home_advantage mean (shift posterior). 0=no correction. Default 0.05 (~53-54%%)")
    parser.add_argument("--spread-scale", type=float, default=1.0,
                        help="Scale att/def arrays by this factor (>1 widens spread)")
    parser.add_argument("--d1b-boost", type=float, default=1.0,
                        help="Multiply D1B ability adjustments by this factor")
    args = parser.parse_args()

    # ── Load odds ────────────────────────────────────────────────────────────
    with open(args.odds) as f:
        odds_records = [json.loads(line) for line in f]
    print(f"Loaded {len(odds_records)} odds records")

    # ── Load actuals ─────────────────────────────────────────────────────────
    re_df = pd.read_csv(args.run_events, dtype=str)
    re_df["game_date"] = pd.to_datetime(re_df["game_date"])
    re_df["home_score"] = pd.to_numeric(re_df["home_score"], errors="coerce")
    re_df["away_score"] = pd.to_numeric(re_df["away_score"], errors="coerce")

    # Build lookup: (date_str, home_canonical, away_canonical) →
    #   (home_score, away_score, home_pitcher_id, away_pitcher_id)
    actuals: dict[tuple[str, str, str], tuple[int, int, str, str]] = {}
    for _, r in re_df.iterrows():
        key = (
            r["game_date"].strftime("%Y-%m-%d"),
            str(r["home_canonical_id"]),
            str(r["away_canonical_id"]),
        )
        hs = r["home_score"]
        as_ = r["away_score"]
        hp_id = str(r.get("home_pitcher_id", "")).strip()
        ap_id = str(r.get("away_pitcher_id", "")).strip()
        if hp_id == "nan":
            hp_id = ""
        if ap_id == "nan":
            ap_id = ""
        if pd.notna(hs) and pd.notna(as_):
            actuals[key] = (int(hs), int(as_), hp_id, ap_id)

    print(f"Loaded {len(actuals)} games with actuals")

    # ── Load model ───────────────────────────────────────────────────────────
    print("Loading posterior...")
    post = load_posterior_arrays(args.posterior, args.meta)
    print(f"  {post['n_draws']} draws, {post['N_teams']} teams, {post['N_pitchers']} pitchers")
    print(f"  Original home_adv: mean={post['home_adv'].mean():.4f}, std={post['home_adv'].std():.4f}")

    # ── Fix 1: Post-hoc home advantage correction ────────────────────────────
    if args.ha_target > 0:
        ha_current = post["home_adv"].mean()
        ha_shift = ha_current - args.ha_target
        post["home_adv"] = post["home_adv"] - ha_shift
        print(f"  HA correction: {ha_current:.4f} → {post['home_adv'].mean():.4f} (shifted by {-ha_shift:+.4f})")

    # ── Fix 2: Spread scaling ────────────────────────────────────────────────
    if args.spread_scale != 1.0:
        for k in range(4):
            post["att"][:, :, k] *= args.spread_scale
            post["def_"][:, :, k] *= args.spread_scale
        post["pitcher_ab"] *= args.spread_scale
        print(f"  Spread scale: {args.spread_scale}x applied to att/def/pitcher_ab")

    # Team/pitcher index maps
    team_idx_df = pd.read_csv(args.team_index)
    team_map = dict(zip(team_idx_df["canonical_id"], team_idx_df["team_idx"]))

    pitcher_idx_df = pd.read_csv(args.pitcher_index, dtype=str)
    pitcher_map: dict[str, int] = {"unknown": 0, "": 0}
    for _, r in pitcher_idx_df.iterrows():
        pid = str(r.get("pitcher_espn_id", "")).strip()
        if pid and pid.lower() != "unknown":
            pitcher_map[pid] = int(r.get("pitcher_idx", 0))

    # Team table for bullpen_z
    bp_map: dict[str, float] = {}
    if args.team_table.exists():
        tt = pd.read_csv(args.team_table)
        for _, r in tt.iterrows():
            cid = str(r.get("canonical_id", ""))
            bz = r.get("bullpen_z", 0.0)
            if pd.notna(bz):
                bp_map[cid] = float(bz) * -0.1

    # D1B ability adjustments for pitchers with idx=0 (no posterior)
    # Also build ESPN ID → (pitcher_idx, d1b_adj) map for ESPN_ format IDs
    d1b_adj_map: dict[str, float] = {}
    espn_pitcher_info: dict[str, dict] = {}  # bare ESPN ID → {pitcher_idx, d1b_adj}
    pitcher_table_path = Path("data/processed/pitcher_table.csv")
    if pitcher_table_path.exists():
        pt = pd.read_csv(pitcher_table_path, dtype=str)
        for _, r in pt.iterrows():
            pid = str(r.get("pitcher_espn_id", "")).strip()
            idx = int(r.get("pitcher_idx", 0))
            adj = r.get("d1b_ability_adj", "0")
            try:
                adj_f = float(adj)
            except (ValueError, TypeError):
                adj_f = 0.0
            if pid and pid != "nan":
                espn_pitcher_info[pid] = {"pitcher_idx": idx, "d1b_adj": adj_f}
                if adj_f != 0.0:
                    d1b_adj_map[pid] = adj_f
        print(f"  D1B ability adjustments: {len(d1b_adj_map)} pitchers")
        print(f"  ESPN pitcher table entries: {len(espn_pitcher_info)}")

    # Build NCAA_ pitcher crosswalk (name+team → pitcher info)
    ncaa_crosswalk = build_pitcher_crosswalk(pitcher_table_path) if pitcher_table_path.exists() else {"full": {}, "last": {}}
    print(f"  NCAA crosswalk: {len(ncaa_crosswalk['full'])} full-name, {len(ncaa_crosswalk['last'])} last-name entries")

    # ── Name resolution (exact match only) ──────────────────────────────────
    exact_map = build_name_map(args.teams_csv)

    # ── Join odds ↔ actuals ↔ model ─────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    results = []
    n_no_canonical = 0
    n_no_actuals = 0
    n_no_team_idx = 0

    d1b_boost = args.d1b_boost

    for i, orec in enumerate(odds_records):
        home_name = orec["home_team"]
        away_name = orec["away_team"]
        h_cid = resolve_odds_name(home_name, exact_map)
        a_cid = resolve_odds_name(away_name, exact_map)

        if not h_cid or not a_cid:
            n_no_canonical += 1
            continue

        # Odds commence_time is UTC; run_events game_date is local.
        # Try exact date, then ±1 day to handle timezone offset.
        game_date_str = orec["commence_time"][:10]
        base = date_cls.fromisoformat(game_date_str)
        found_actual = None
        for offset in [0, -1, 1]:
            d = (base + timedelta(days=offset)).isoformat()
            key = (d, h_cid, a_cid)
            if key in actuals:
                found_actual = actuals[key]
                game_date_str = d  # use the matched date
                break

        if found_actual is None:
            n_no_actuals += 1
            continue

        actual_home, actual_away, hp_espn_id, ap_espn_id = found_actual
        home_won = actual_home > actual_away

        # Extract market odds
        consensus_home = orec.get("consensus_fair_home")
        consensus_away = orec.get("consensus_fair_away")

        # Find best available moneyline from bookmakers
        best_home_ml = None
        best_away_ml = None
        market_total_line = None
        n_books = 0

        for bl in orec.get("bookmaker_lines", []):
            for m in bl.get("markets", []):
                if m.get("key") == "h2h":
                    n_books += 1
                    for o in m.get("outcomes", []):
                        if o["name"] == home_name:
                            ml = o["price"]
                            if best_home_ml is None or ml > best_home_ml:
                                best_home_ml = ml
                        elif o["name"] == away_name:
                            ml = o["price"]
                            if best_away_ml is None or ml > best_away_ml:
                                best_away_ml = ml
                elif m.get("key") == "totals":
                    for o in m.get("outcomes", []):
                        if o.get("name") == "Over" and "point" in o:
                            market_total_line = float(o["point"])

        if consensus_home is None or best_home_ml is None:
            continue

        # Model team indices
        h_idx = team_map.get(h_cid, 0)
        a_idx = team_map.get(a_cid, 0)

        # Clamp to posterior bounds
        h_idx = h_idx if h_idx <= post["N_teams"] else 0
        a_idx = a_idx if a_idx <= post["N_teams"] else 0

        # Resolve starting pitchers from run_events
        # Two formats: "ESPN_12345" (44%) and "NCAA_name__team" (56%)
        hp_idx, hp_d1b = 0, 0.0
        ap_idx, ap_d1b = 0, 0.0

        if hp_espn_id:
            if hp_espn_id.startswith("ESPN_"):
                hp_bare = hp_espn_id[5:]  # strip "ESPN_"
                info = espn_pitcher_info.get(hp_bare)
                if info:
                    hp_idx = info["pitcher_idx"]
                    hp_d1b = info["d1b_adj"] if hp_idx == 0 else 0.0
                else:
                    hp_idx = pitcher_map.get(hp_bare, 0)
            elif hp_espn_id.startswith("NCAA_"):
                hp_idx, hp_d1b = resolve_ncaa_pitcher(hp_espn_id, ncaa_crosswalk)
                if hp_idx > 0:
                    hp_d1b = 0.0  # has posterior, don't use D1B fallback

        if ap_espn_id:
            if ap_espn_id.startswith("ESPN_"):
                ap_bare = ap_espn_id[5:]
                info = espn_pitcher_info.get(ap_bare)
                if info:
                    ap_idx = info["pitcher_idx"]
                    ap_d1b = info["d1b_adj"] if ap_idx == 0 else 0.0
                else:
                    ap_idx = pitcher_map.get(ap_bare, 0)
            elif ap_espn_id.startswith("NCAA_"):
                ap_idx, ap_d1b = resolve_ncaa_pitcher(ap_espn_id, ncaa_crosswalk)
                if ap_idx > 0:
                    ap_d1b = 0.0  # has posterior

        # Clamp to posterior bounds
        hp_idx = hp_idx if hp_idx <= post["N_pitchers"] else 0
        ap_idx = ap_idx if ap_idx <= post["N_pitchers"] else 0

        # Fix 5: D1B boost
        hp_d1b *= d1b_boost
        ap_d1b *= d1b_boost

        pf = 0.0  # No park factor without stadium lookup (TODO)
        h_bp = bp_map.get(h_cid, 0.0)
        a_bp = bp_map.get(a_cid, 0.0)

        model_home_prob, exp_home, exp_away = simulate_game(
            post, h_idx, a_idx, hp_idx, ap_idx,
            pf, h_bp, a_bp, args.N, rng,
            hp_d1b_adj=hp_d1b, ap_d1b_adj=ap_d1b,
        )

        model_edge = model_home_prob - consensus_home

        results.append({
            "date": game_date_str,
            "home": home_name,
            "away": away_name,
            "home_cid": h_cid,
            "away_cid": a_cid,
            "h_idx": h_idx,
            "a_idx": a_idx,
            "hp_idx": hp_idx,
            "ap_idx": ap_idx,
            "hp_d1b_adj": hp_d1b,
            "ap_d1b_adj": ap_d1b,
            "actual_home": actual_home,
            "actual_away": actual_away,
            "actual_total": actual_home + actual_away,
            "home_won": home_won,
            "model_home_prob": model_home_prob,
            "model_exp_total": exp_home + exp_away,
            "market_home_prob": consensus_home,
            "market_away_prob": consensus_away,
            "market_total_line": market_total_line,
            "best_home_ml": best_home_ml,
            "best_away_ml": best_away_ml,
            "n_books": n_books,
            "model_edge_home": model_edge,
            "model_ml_home": prob_to_american(model_home_prob),
            "model_ml_away": prob_to_american(1 - model_home_prob),
        })

        if (len(results)) % 50 == 0:
            print(f"  Simulated {len(results)} games...", file=sys.stderr)

    df = pd.DataFrame(results)
    print(f"\nJoined {len(df)} games (skipped: {n_no_canonical} no canonical, "
          f"{n_no_actuals} no actuals, odds with no market data filtered)")

    if df.empty:
        print("ERROR: No games to analyze.")
        return 1

    # ── SECTION 1: Model vs. Market Accuracy ─────────────────────────────────
    print_section("1. MODEL vs. MARKET — MONEYLINE ACCURACY")

    actual_hw = df["home_won"].astype(float)

    # Model metrics
    model_correct = ((df["model_home_prob"] > 0.5) & df["home_won"]) | \
                    ((df["model_home_prob"] < 0.5) & ~df["home_won"])
    model_acc = model_correct.mean()
    model_brier = ((df["model_home_prob"] - actual_hw) ** 2).mean()
    model_logloss = -(
        actual_hw * np.log(np.clip(df["model_home_prob"], 1e-6, 1 - 1e-6)) +
        (1 - actual_hw) * np.log(np.clip(1 - df["model_home_prob"], 1e-6, 1 - 1e-6))
    ).mean()

    # Market metrics
    market_correct = ((df["market_home_prob"] > 0.5) & df["home_won"]) | \
                     ((df["market_home_prob"] < 0.5) & ~df["home_won"])
    market_acc = market_correct.mean()
    market_brier = ((df["market_home_prob"] - actual_hw) ** 2).mean()
    market_logloss = -(
        actual_hw * np.log(np.clip(df["market_home_prob"], 1e-6, 1 - 1e-6)) +
        (1 - actual_hw) * np.log(np.clip(1 - df["market_home_prob"], 1e-6, 1 - 1e-6))
    ).mean()

    # Baseline: always predict home wins at empirical rate
    home_rate = actual_hw.mean()
    baseline_brier = ((home_rate - actual_hw) ** 2).mean()
    baseline_logloss = -(
        actual_hw * np.log(max(1e-6, home_rate)) +
        (1 - actual_hw) * np.log(max(1e-6, 1 - home_rate))
    ).mean()

    # Murphy decomposition
    model_murphy = murphy_decomposition(df["model_home_prob"].values, actual_hw.values)
    market_murphy = murphy_decomposition(df["market_home_prob"].values, actual_hw.values)

    print(f"\n  {'Metric':<22s}  {'Model':>10s}  {'Market':>10s}  {'Home-Only':>10s}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}")
    print(f"  {'Accuracy':<22s}  {model_acc:>9.1%}  {market_acc:>9.1%}  {home_rate:>9.1%}")
    print(f"  {'Brier Score':<22s}  {model_brier:>10.4f}  {market_brier:>10.4f}  {baseline_brier:>10.4f}")
    print(f"  {'Log Loss':<22s}  {model_logloss:>10.4f}  {market_logloss:>10.4f}  {baseline_logloss:>10.4f}")
    print(f"  {'Brier Skill (vs base)':<22s}  {1 - model_brier/baseline_brier:>10.3f}  {1 - market_brier/baseline_brier:>10.3f}  {'0.000':>10s}")

    print(f"\n  Murphy Brier Decomposition (BS = Reliability - Resolution + Uncertainty):")
    print(f"  {'Component':<22s}  {'Model':>10s}  {'Market':>10s}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}")
    print(f"  {'Reliability (↓better)':<22s}  {model_murphy['reliability']:>10.4f}  {market_murphy['reliability']:>10.4f}")
    print(f"  {'Resolution (↑better)':<22s}  {model_murphy['resolution']:>10.4f}  {market_murphy['resolution']:>10.4f}")
    print(f"  {'Uncertainty':<22s}  {model_murphy['uncertainty']:>10.4f}  {market_murphy['uncertainty']:>10.4f}")

    print(f"\n  Games: {len(df)} | Home win rate: {home_rate:.1%}")
    print(f"  Model prob range: [{df['model_home_prob'].min():.3f}, {df['model_home_prob'].max():.3f}], std={df['model_home_prob'].std():.3f}")
    print(f"  Market prob range: [{df['market_home_prob'].min():.3f}, {df['market_home_prob'].max():.3f}], std={df['market_home_prob'].std():.3f}")
    model_home_fav_pct = (df["model_home_prob"] > 0.5).mean()
    market_home_fav_pct = (df["market_home_prob"] > 0.5).mean()
    print(f"  Home favored: Model {model_home_fav_pct:.1%} | Market {market_home_fav_pct:.1%} | Actual {home_rate:.1%}")

    # Pitcher coverage
    has_hp = (df["hp_idx"] > 0).sum()
    has_ap = (df["ap_idx"] > 0).sum()
    has_hp_d1b = (df["hp_d1b_adj"].abs() > 0.001).sum()
    has_ap_d1b = (df["ap_d1b_adj"].abs() > 0.001).sum()
    both_posterior = ((df["hp_idx"] > 0) & (df["ap_idx"] > 0)).sum()
    any_info = ((df["hp_idx"] > 0) | (df["ap_idx"] > 0) | (df["hp_d1b_adj"].abs() > 0.001) | (df["ap_d1b_adj"].abs() > 0.001)).sum()
    print(f"\n  Pitcher coverage:")
    print(f"    Home SP with posterior: {has_hp}/{len(df)} ({has_hp/len(df):.0%})")
    print(f"    Away SP with posterior: {has_ap}/{len(df)} ({has_ap/len(df):.0%})")
    print(f"    Both SPs with posterior: {both_posterior}/{len(df)} ({both_posterior/len(df):.0%})")
    print(f"    Home SP with D1B fallback: {has_hp_d1b}")
    print(f"    Away SP with D1B fallback: {has_ap_d1b}")
    print(f"    Games with any pitcher info: {any_info}/{len(df)} ({any_info/len(df):.0%})")

    # ── SECTION 2: Calibration ───────────────────────────────────────────────
    print_section("2. CALIBRATION")
    report_calibration(df, "model_home_prob", "Model (raw)")
    report_calibration(df, "market_home_prob", "Market")

    # ── SECTION 3: Fix 3 — Platt Scaling Calibration (5-fold CV) ────────────
    print_section("3. PLATT SCALING CALIBRATION (5-fold CV)")

    model_probs = df["model_home_prob"].values
    outcomes = actual_hw.values

    calibrated_probs = platt_cv(model_probs, outcomes, n_folds=5)
    df["model_calibrated"] = calibrated_probs

    cal_brier = ((calibrated_probs - outcomes) ** 2).mean()
    cal_murphy = murphy_decomposition(calibrated_probs, outcomes)
    cal_correct = ((calibrated_probs > 0.5) & (outcomes > 0.5)) | \
                  ((calibrated_probs < 0.5) & (outcomes < 0.5))
    cal_acc = cal_correct.mean()

    # Fit on full data for reporting coefficients
    logits_full = np.log(np.clip(model_probs, 1e-6, 1 - 1e-6) / np.clip(1 - model_probs, 1e-6, 1 - 1e-6))
    a_full, b_full = platt_fit(logits_full, outcomes)

    print(f"\n  Platt parameters (full-data fit): a={a_full:.3f}, b={b_full:.3f}")
    print(f"  Interpretation: a>1 means model is under-confident, b>0 means model under-predicts home wins")
    print(f"\n  {'Metric':<22s}  {'Raw Model':>10s}  {'Calibrated':>10s}  {'Market':>10s}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}")
    print(f"  {'Accuracy':<22s}  {model_acc:>9.1%}  {cal_acc:>9.1%}  {market_acc:>9.1%}")
    print(f"  {'Brier Score':<22s}  {model_brier:>10.4f}  {cal_brier:>10.4f}  {market_brier:>10.4f}")
    print(f"  {'Reliability':<22s}  {model_murphy['reliability']:>10.4f}  {cal_murphy['reliability']:>10.4f}  {market_murphy['reliability']:>10.4f}")
    print(f"  {'Resolution':<22s}  {model_murphy['resolution']:>10.4f}  {cal_murphy['resolution']:>10.4f}  {market_murphy['resolution']:>10.4f}")

    report_calibration(df, "model_calibrated", "Model (Platt-calibrated)")

    # ── SECTION 4: Fix 4 — Market Blend Analysis ─────────────────────────────
    print_section("4. MARKET BLEND ANALYSIS (logit-space)")

    print(f"\n  Grid search: model_weight from 0.05 to 0.95")
    print(f"  {'Weight':>8s}  {'Brier':>8s}  {'Accuracy':>8s}  {'Reliability':>10s}  {'Resolution':>10s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")

    best_blend_brier = 1.0
    best_blend_weight = 0.0
    best_blend_probs = None

    model_logits = np.log(np.clip(model_probs, 1e-6, 1 - 1e-6) / np.clip(1 - model_probs, 1e-6, 1 - 1e-6))
    market_probs = df["market_home_prob"].values
    market_logits = np.log(np.clip(market_probs, 1e-6, 1 - 1e-6) / np.clip(1 - market_probs, 1e-6, 1 - 1e-6))

    for w in np.arange(0.05, 1.00, 0.05):
        blended_logits = w * model_logits + (1 - w) * market_logits
        blended_probs = 1.0 / (1.0 + np.exp(-blended_logits))
        bl_brier = ((blended_probs - outcomes) ** 2).mean()
        bl_murphy = murphy_decomposition(blended_probs, outcomes)
        bl_correct = ((blended_probs > 0.5) & (outcomes > 0.5)) | \
                     ((blended_probs < 0.5) & (outcomes < 0.5))
        bl_acc = bl_correct.mean()

        marker = "  ← best" if bl_brier < best_blend_brier else ""
        print(f"  {w:>7.2f}  {bl_brier:>8.4f}  {bl_acc:>7.1%}  {bl_murphy['reliability']:>10.4f}  {bl_murphy['resolution']:>10.4f}{marker}")

        if bl_brier < best_blend_brier:
            best_blend_brier = bl_brier
            best_blend_weight = w
            best_blend_probs = blended_probs.copy()

    print(f"\n  Optimal blend: {best_blend_weight:.0%} model + {1-best_blend_weight:.0%} market → Brier={best_blend_brier:.4f}")
    print(f"  vs. Model-only Brier={model_brier:.4f}, Market-only Brier={market_brier:.4f}")

    # Store best blend
    df["model_blended"] = best_blend_probs

    # Also test: calibrated model blended with market
    print(f"\n  Calibrated model + market blend:")
    cal_logits = np.log(np.clip(calibrated_probs, 1e-6, 1 - 1e-6) / np.clip(1 - calibrated_probs, 1e-6, 1 - 1e-6))
    best_calblend_brier = 1.0
    best_calblend_w = 0.0
    for w in np.arange(0.10, 0.90, 0.05):
        bl_logits = w * cal_logits + (1 - w) * market_logits
        bl_probs = 1.0 / (1.0 + np.exp(-bl_logits))
        bl_brier = ((bl_probs - outcomes) ** 2).mean()
        if bl_brier < best_calblend_brier:
            best_calblend_brier = bl_brier
            best_calblend_w = w
    print(f"  Optimal: {best_calblend_w:.0%} calibrated + {1-best_calblend_w:.0%} market → Brier={best_calblend_brier:.4f}")

    # ── SECTION 5: ROI / P&L ────────────────────────────────────────────────
    print_section("5. ROI SIMULATION")
    report_roi(df, "model_home_prob", "Raw model vs. Market")
    if best_blend_probs is not None:
        report_roi(df, "model_blended", "Blended model vs. Market")
    report_roi(df, "model_calibrated", "Calibrated model vs. Market")

    # ── SECTION 6: Totals Bias ───────────────────────────────────────────────
    print_section("6. TOTALS BIAS")

    has_total = df["market_total_line"].notna()
    t = df[has_total].copy()

    if len(t) > 0:
        model_total_mean = t["model_exp_total"].mean()
        market_total_mean = t["market_total_line"].mean()
        actual_total_mean = t["actual_total"].mean()

        model_total_mae = (t["model_exp_total"] - t["actual_total"]).abs().mean()
        market_total_mae = (t["market_total_line"] - t["actual_total"]).abs().mean()

        model_total_bias = (t["model_exp_total"] - t["actual_total"]).mean()
        market_total_bias = (t["market_total_line"] - t["actual_total"]).mean()

        model_total_corr = t["model_exp_total"].corr(t["actual_total"].astype(float))
        market_total_corr = t["market_total_line"].corr(t["actual_total"].astype(float))

        print(f"\n  {len(t)} games with totals lines")
        print(f"\n  {'Metric':<20s}  {'Model':>10s}  {'Market':>10s}  {'Actual':>10s}")
        print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}")
        print(f"  {'Mean Total':<20s}  {model_total_mean:>10.2f}  {market_total_mean:>10.2f}  {actual_total_mean:>10.2f}")
        print(f"  {'Bias (pred-actual)':<20s}  {model_total_bias:>+10.2f}  {market_total_bias:>+10.2f}  {'':>10s}")
        print(f"  {'MAE':<20s}  {model_total_mae:>10.2f}  {market_total_mae:>10.2f}  {'':>10s}")
        print(f"  {'Correlation':<20s}  {model_total_corr:>10.3f}  {market_total_corr:>10.3f}  {'':>10s}")
    else:
        print("  No games with totals lines found.")

    # ── SECTION 7: Baseline Comparisons ──────────────────────────────────────
    print_section("7. FINAL COMPARISON — ALL VARIANTS")

    bl_murphy = murphy_decomposition(best_blend_probs, outcomes) if best_blend_probs is not None else {}

    print(f"\n  {'Variant':<30s}  {'Brier':>8s}  {'Acc':>8s}  {'Reliab':>8s}  {'Resol':>8s}  {'BSS':>8s}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'Home-field baseline':<30s}  {baseline_brier:>8.4f}  {home_rate:>7.1%}  {'—':>8s}  {'—':>8s}  {'0.000':>8s}")
    print(f"  {'Market consensus':<30s}  {market_brier:>8.4f}  {market_acc:>7.1%}  {market_murphy['reliability']:>8.4f}  {market_murphy['resolution']:>8.4f}  {1 - market_brier/baseline_brier:>8.3f}")
    print(f"  {'Model (raw)':<30s}  {model_brier:>8.4f}  {model_acc:>7.1%}  {model_murphy['reliability']:>8.4f}  {model_murphy['resolution']:>8.4f}  {1 - model_brier/baseline_brier:>8.3f}")
    print(f"  {'Model (Platt-calibrated)':<30s}  {cal_brier:>8.4f}  {cal_acc:>7.1%}  {cal_murphy['reliability']:>8.4f}  {cal_murphy['resolution']:>8.4f}  {1 - cal_brier/baseline_brier:>8.3f}")
    if best_blend_probs is not None:
        bl_acc = ((best_blend_probs > 0.5) == (outcomes > 0.5)).mean()
        blend_label = f"Blend ({best_blend_weight:.0%}M+{1-best_blend_weight:.0%}Mkt)"
        print(f"  {blend_label:<30s}  {best_blend_brier:>8.4f}  {bl_acc:>7.1%}  {bl_murphy.get('reliability',0):>8.4f}  {bl_murphy.get('resolution',0):>8.4f}  {1 - best_blend_brier/baseline_brier:>8.3f}")
    calblend_label = f"Cal+Blend ({best_calblend_w:.0%}C+{1-best_calblend_w:.0%}Mkt)"
    print(f"  {calblend_label:<30s}  {best_calblend_brier:>8.4f}  {'—':>8s}  {'—':>8s}  {'—':>8s}  {1 - best_calblend_brier/baseline_brier:>8.3f}")

    # ── SECTION 8: Does model add value? ──────────────────────────────────────
    print_section("8. INDEPENDENT INFORMATION TEST")

    print(f"\n  Does model provide independent information beyond market?")
    df["model_minus_market"] = df["model_home_prob"] - df["market_home_prob"]
    agree_mask = df["model_minus_market"].abs() < 0.05
    disagree_mask = df["model_minus_market"].abs() >= 0.05
    if disagree_mask.sum() >= 10:
        agree_acc = model_correct[agree_mask].mean() if agree_mask.sum() > 0 else float("nan")
        disagree_model_acc = model_correct[disagree_mask].mean()
        disagree_market_acc = market_correct[disagree_mask].mean()
        print(f"    When model ≈ market (<5% diff): {agree_mask.sum()} games, model acc={agree_acc:.1%}")
        print(f"    When model ≠ market (≥5% diff): {disagree_mask.sum()} games")
        print(f"      Model acc: {disagree_model_acc:.1%}  |  Market acc: {disagree_market_acc:.1%}")

    # Key insight: is model's residual orthogonal to market?
    if len(df) > 20:
        from numpy.linalg import lstsq
        X = np.column_stack([
            market_logits,
            model_logits,
            np.ones(len(df)),
        ])
        y = outcomes
        beta, _, _, _ = lstsq(X, y, rcond=None)
        print(f"\n  OLS: outcome ~ β₁·market_logit + β₂·model_logit + β₀")
        print(f"    β₁ (market): {beta[0]:.4f}")
        print(f"    β₂ (model):  {beta[1]:.4f}")
        print(f"    β₀ (bias):   {beta[2]:.4f}")
        print(f"    → Model adds {'positive' if beta[1] > 0.01 else 'negligible'} signal beyond market (β₂={'>' if beta[1] > 0.01 else '≈'}0)")

    # ── Save detail CSV ──────────────────────────────────────────────────────
    df.to_csv(args.out, index=False)
    print(f"\n  Detail CSV: {args.out} ({len(df)} rows)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print_section("SUMMARY")
    print(f"\n  Fix parameters used:")
    print(f"    --ha-target {args.ha_target} (home advantage shifted to {post['home_adv'].mean():.4f})")
    print(f"    --spread-scale {args.spread_scale}")
    print(f"    --d1b-boost {args.d1b_boost}")
    print(f"\n  Key results:")
    print(f"    Model Brier:      {model_brier:.4f} → Calibrated: {cal_brier:.4f} → Best blend: {best_blend_brier:.4f}")
    print(f"    Market Brier:     {market_brier:.4f}")
    print(f"    Model beats market? {'YES' if best_blend_brier < market_brier else 'NO'} (blend)")
    gap = best_blend_brier - market_brier
    print(f"    Gap to market:    {gap:+.4f} ({'model better' if gap < 0 else 'market better'})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
