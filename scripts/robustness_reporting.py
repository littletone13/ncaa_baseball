from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def profit_on_bet(ml: int | float, won: bool) -> float:
    """Profit per $1 stake at American odds."""
    if not won:
        return -1.0
    x = float(ml)
    if x > 0:
        return x / 100.0
    return 100.0 / abs(x)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def apply_uncertainty_columns(
    df: pd.DataFrame,
    min_bet_confidence: float = 0.55,
    default_weather_confidence: float = 0.50,
    default_fatigue_confidence: float = 0.50,
) -> pd.DataFrame:
    """
    Add uncertainty-aware confidence columns:
      - starter_confidence
      - weather_confidence
      - fatigue_confidence
      - bet_confidence
      - bet_gate_pass
    """
    out = df.copy()

    hp_idx = pd.to_numeric(out.get("hp_idx", 0), errors="coerce").fillna(0)
    ap_idx = pd.to_numeric(out.get("ap_idx", 0), errors="coerce").fillna(0)
    hp_d1b = pd.to_numeric(out.get("hp_d1b_adj", 0), errors="coerce").fillna(0.0).abs() > 1e-12
    ap_d1b = pd.to_numeric(out.get("ap_d1b_adj", 0), errors="coerce").fillna(0.0).abs() > 1e-12

    both_post = (hp_idx > 0) & (ap_idx > 0)
    one_post = (hp_idx > 0) ^ (ap_idx > 0)
    both_fallback = ((hp_idx <= 0) & hp_d1b) & ((ap_idx <= 0) & ap_d1b)
    starter_conf = np.where(
        both_post,
        1.00,
        np.where(one_post, 0.72, np.where(both_fallback, 0.62, 0.42)),
    )
    out["starter_confidence"] = starter_conf.astype(float)

    if "weather_status" in out.columns:
        ws = out["weather_status"].astype(str).str.lower().fillna("")
        weather_conf = np.where(
            ws.str.startswith("ok"),
            1.00,
            np.where(
                ws.str.contains("fallback") | ws.str.contains("hourly"),
                0.72,
                np.where(ws.eq("") | ws.eq("nan"), default_weather_confidence, 0.35),
            ),
        )
    else:
        weather_conf = np.full(len(out), default_weather_confidence, dtype=float)
    out["weather_confidence"] = pd.Series(weather_conf, index=out.index).astype(float)

    if "home_fatigue_adj" in out.columns or "away_fatigue_adj" in out.columns:
        h_known = pd.to_numeric(out.get("home_fatigue_adj", np.nan), errors="coerce").notna()
        a_known = pd.to_numeric(out.get("away_fatigue_adj", np.nan), errors="coerce").notna()
        fatigue_conf = np.where(h_known & a_known, 1.00, np.where(h_known | a_known, 0.70, default_fatigue_confidence))
    else:
        fatigue_conf = np.full(len(out), default_fatigue_confidence, dtype=float)
    out["fatigue_confidence"] = pd.Series(fatigue_conf, index=out.index).astype(float)

    out["bet_confidence"] = (
        0.50 * out["starter_confidence"]
        + 0.30 * out["weather_confidence"]
        + 0.20 * out["fatigue_confidence"]
    ).map(lambda x: _clamp(float(x), 0.0, 1.0))
    out["bet_gate_pass"] = out["bet_confidence"] >= float(min_bet_confidence)
    return out


def add_regime_columns(df: pd.DataFrame, teams_csv: Path) -> pd.DataFrame:
    """Add conference and totals-band slices for robustness reporting."""
    out = df.copy()
    if teams_csv.exists():
        teams = pd.read_csv(teams_csv, dtype=str).fillna("")
        conf_map = dict(zip(teams["canonical_id"], teams.get("conference", "")))
    else:
        conf_map = {}

    out["home_conf"] = out.get("home_cid", "").astype(str).map(conf_map).fillna("Unknown")
    out["away_conf"] = out.get("away_cid", "").astype(str).map(conf_map).fillna("Unknown")
    same_conf = out["home_conf"].eq(out["away_conf"]) & out["home_conf"].ne("") & out["home_conf"].ne("Unknown")
    out["conference_slice"] = np.where(same_conf, out["home_conf"], "cross_conf")

    mkt_total = pd.to_numeric(out.get("market_total_line", np.nan), errors="coerce")
    out["totals_band"] = pd.cut(
        mkt_total,
        bins=[-np.inf, 9.5, 10.5, 11.5, 12.5, np.inf],
        labels=["lt9_5", "9_5_10_5", "10_5_11_5", "11_5_12_5", "gte12_5"],
    ).astype(str)
    out.loc[mkt_total.isna(), "totals_band"] = "unknown"
    return out


def _max_drawdown_units(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    curve = pnl.cumsum()
    peaks = curve.cummax()
    dd = (peaks - curve).max()
    return float(dd if pd.notna(dd) else 0.0)


@dataclass
class EvalResult:
    n: int
    won: int
    win_rate: float
    pnl: float
    roi: float
    max_dd: float
    objective: float
    bets: pd.DataFrame


def evaluate_threshold_strategy(
    df: pd.DataFrame,
    prob_col: str,
    threshold: float,
    dd_penalty: float,
    min_bet_confidence: float = 0.55,
) -> EvalResult:
    x = df.copy()
    x["model_prob"] = pd.to_numeric(x.get(prob_col), errors="coerce")
    x["market_prob"] = pd.to_numeric(x.get("market_home_prob"), errors="coerce")
    x["best_home_ml"] = pd.to_numeric(x.get("best_home_ml"), errors="coerce")
    x["best_away_ml"] = pd.to_numeric(x.get("best_away_ml"), errors="coerce")
    x["home_won"] = x.get("home_won", False).astype(bool)
    x["edge"] = x["model_prob"] - x["market_prob"]
    x["bet_confidence"] = pd.to_numeric(x.get("bet_confidence"), errors="coerce").fillna(1.0)
    x["date"] = pd.to_datetime(x.get("date"), errors="coerce")

    mask = (
        x["edge"].abs() >= float(threshold)
    ) & (
        x["bet_confidence"] >= float(min_bet_confidence)
    ) & (
        x["model_prob"].notna() & x["market_prob"].notna() & x["best_home_ml"].notna() & x["best_away_ml"].notna()
    )
    bets = x.loc[mask].copy()
    if bets.empty:
        return EvalResult(0, 0, 0.0, 0.0, 0.0, 0.0, -1e9, bets)

    bets["bet_home"] = bets["edge"] > 0
    bets["bet_won"] = (bets["bet_home"] & bets["home_won"]) | (~bets["bet_home"] & ~bets["home_won"])
    bets["bet_ml"] = np.where(bets["bet_home"], bets["best_home_ml"], bets["best_away_ml"])
    bets["bet_profit"] = [
        profit_on_bet(ml, bool(won))
        for ml, won in zip(bets["bet_ml"].tolist(), bets["bet_won"].tolist())
    ]
    bets = bets.sort_values(["date", "home", "away"], kind="stable")

    n = int(len(bets))
    won = int(bets["bet_won"].sum())
    pnl = float(pd.to_numeric(bets["bet_profit"], errors="coerce").fillna(0.0).sum())
    roi = pnl / max(1, n)
    max_dd = _max_drawdown_units(pd.to_numeric(bets["bet_profit"], errors="coerce").fillna(0.0))
    objective = float(roi - float(dd_penalty) * (max_dd / max(1, n)))
    return EvalResult(
        n=n,
        won=won,
        win_rate=won / max(1, n),
        pnl=pnl,
        roi=roi,
        max_dd=max_dd,
        objective=objective,
        bets=bets,
    )


def build_regime_robustness_table(bets: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ROI/drawdown by robustness slices."""
    if bets.empty:
        return pd.DataFrame(columns=["slice", "n", "won", "win_rate", "pnl", "roi", "max_dd"])

    rows: list[dict[str, object]] = []
    slice_specs = {
        "conference": bets.get("conference_slice", pd.Series(["unknown"] * len(bets), index=bets.index)).astype(str),
        "totals_band": bets.get("totals_band", pd.Series(["unknown"] * len(bets), index=bets.index)).astype(str),
        "date_block": bets.get("date_block", pd.Series(["unknown"] * len(bets), index=bets.index)).astype(str),
    }

    for dim, values in slice_specs.items():
        tmp = bets.copy()
        tmp["_slice_val"] = values
        for slice_val, grp in tmp.groupby("_slice_val", dropna=False):
            pnl_series = pd.to_numeric(grp["bet_profit"], errors="coerce").fillna(0.0)
            n = int(len(grp))
            won = int(pd.to_numeric(grp["bet_won"], errors="coerce").fillna(0).sum())
            pnl = float(pnl_series.sum())
            roi = pnl / max(1, n)
            rows.append(
                {
                    "slice": f"{dim}_{slice_val}",
                    "n": n,
                    "won": won,
                    "win_rate": won / max(1, n),
                    "pnl": pnl,
                    "roi": roi,
                    "max_dd": _max_drawdown_units(pnl_series),
                }
            )

    out = pd.DataFrame(rows).sort_values(["slice"], kind="stable").reset_index(drop=True)
    return out
