#!/usr/bin/env python3
"""
Automated audit scorecard for NCAA baseball betting readiness.

Outputs a strict BET_READY: YES/NO based on fail-fast and robustness gates.
"""
from __future__ import annotations

import argparse
import json
import re
import runpy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class GateResult:
    gate: str
    passed: bool
    blocking: bool
    detail: str
    evidence: dict[str, Any]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _read_scoring_calibration(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m = re.search(r"SCORING_CALIBRATION\s*=\s*([-+]?[0-9]*\.?[0-9]+)", text)
    return float(m.group(1)) if m else None


def _latest_date_folder(daily_root: Path) -> Path | None:
    if not daily_root.exists():
        return None
    cands = [p for p in daily_root.iterdir() if p.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}$", p.name)]
    return sorted(cands)[-1] if cands else None


def _american_to_prob(ml: float) -> float:
    if ml < 0:
        return abs(ml) / (abs(ml) + 100.0)
    return 100.0 / (ml + 100.0)


def gate_calibration_parity(repo_root: Path) -> GateResult:
    shared = _read_scoring_calibration(repo_root / "src/ncaa_baseball/model_runtime.py")
    sim = None
    bt = None
    err = None
    try:
        sim_globals = runpy.run_path(str(repo_root / "scripts/simulate.py"), run_name="__audit__")
        bt_globals = runpy.run_path(str(repo_root / "scripts/backtest_posterior.py"), run_name="__audit__")
        sim = float(sim_globals.get("SIMULATE_SCORING_CALIBRATION"))
        bt = float(bt_globals.get("BACKTEST_SCORING_CALIBRATION"))
    except Exception as ex:
        err = str(ex)

    ok = (
        err is None
        and shared is not None
        and sim is not None
        and bt is not None
        and abs(sim - shared) < 1e-12
        and abs(bt - shared) < 1e-12
    )
    detail = (
        "simulate/backtest calibrations match shared runtime config"
        if ok
        else f"calibration mismatch: shared={shared}, simulate={sim}, backtest_posterior={bt}"
    )
    return GateResult(
        gate="calibration_parity",
        passed=ok,
        blocking=True,
        detail=detail,
        evidence={"shared": shared, "simulate": sim, "backtest_posterior": bt, "error": err},
    )


def gate_prediction_invariants(repo_root: Path, date_label: str) -> GateResult:
    pred = repo_root / f"data/processed/predictions_{date_label}.csv"
    if not pred.exists():
        return GateResult(
            gate="prediction_invariants",
            passed=False,
            blocking=True,
            detail=f"missing predictions file: {pred}",
            evidence={"path": str(pred)},
        )
    df = pd.read_csv(pred)
    required = {"home_win_prob", "away_win_prob", "exp_home", "exp_away", "exp_total"}
    missing = sorted(required - set(df.columns))
    if missing:
        return GateResult(
            gate="prediction_invariants",
            passed=False,
            blocking=True,
            detail=f"missing required columns: {missing}",
            evidence={"missing_columns": missing},
        )
    p = pd.to_numeric(df["home_win_prob"], errors="coerce")
    q = pd.to_numeric(df["away_win_prob"], errors="coerce")
    exp_total = pd.to_numeric(df["exp_total"], errors="coerce")
    exp_home = pd.to_numeric(df["exp_home"], errors="coerce")
    exp_away = pd.to_numeric(df["exp_away"], errors="coerce")

    bounds_ok = bool(((p >= 0.0) & (p <= 1.0) & (q >= 0.0) & (q <= 1.0)).all())
    comp_err = float((p + q - 1.0).abs().max())
    nonneg_ok = bool(((exp_home >= 0.0) & (exp_away >= 0.0) & (exp_total >= 0.0)).all())

    # Odds monotonicity check against simulate.py behavior.
    probs = np.linspace(0.01, 0.99, 99)
    mls = [int(round(-100 * x / (1 - x))) if x >= 0.5 else int(round(100 * (1 - x) / x)) for x in probs]
    mono_ok = all(mls[i] > mls[i + 1] for i in range(len(mls) - 1))
    rt_err = 0.0
    for pr in probs:
        ml = int(round(-100 * pr / (1 - pr))) if pr >= 0.5 else int(round(100 * (1 - pr) / pr))
        rt_err = max(rt_err, abs(_american_to_prob(float(ml)) - pr))

    ok = bounds_ok and comp_err <= 1e-8 and nonneg_ok and mono_ok and rt_err <= 0.01
    detail = (
        "probability, complement, non-negativity, and odds monotonicity checks passed"
        if ok
        else "one or more invariant checks failed"
    )
    return GateResult(
        gate="prediction_invariants",
        passed=ok,
        blocking=True,
        detail=detail,
        evidence={
            "rows": int(len(df)),
            "bounds_ok": bounds_ok,
            "max_complement_error": comp_err,
            "nonnegative_expectations": nonneg_ok,
            "odds_monotone": mono_ok,
            "odds_roundtrip_max_error": rt_err,
            "path": str(pred),
        },
    )


def gate_pipeline_contracts(repo_root: Path, date_label: str, fatigue_min_coverage: float) -> GateResult:
    daily = repo_root / "data/daily" / date_label
    schedule = daily / "schedule.csv"
    starters = daily / "starters.csv"
    weather = daily / "weather.csv"
    fatigue = daily / "fatigue.csv"
    required_files = [schedule, starters, weather, fatigue]
    missing_files = [str(p) for p in required_files if not p.exists()]
    if missing_files:
        return GateResult(
            gate="pipeline_contracts",
            passed=False,
            blocking=True,
            detail=f"missing daily files: {missing_files}",
            evidence={"missing_files": missing_files},
        )

    sch = pd.read_csv(schedule, dtype=str)
    st = pd.read_csv(starters, dtype=str)
    wx = pd.read_csv(weather, dtype=str)
    fat = pd.read_csv(fatigue, dtype=str)

    sch_cols = {"game_num", "home_cid", "away_cid"}
    st_cols = {"game_num", "home_cid", "away_cid"}
    # weather.csv historically may omit away_cid in this repo; game_num/home_cid are
    # the strict contract used for joins and consistency checks.
    wx_cols = {"game_num", "home_cid"}
    fat_cols = {"canonical_id", "fatigue_adj"}
    missing_cols = {
        "schedule": sorted(sch_cols - set(sch.columns)),
        "starters": sorted(st_cols - set(st.columns)),
        "weather": sorted(wx_cols - set(wx.columns)),
        "fatigue": sorted(fat_cols - set(fat.columns)),
    }
    if any(missing_cols[k] for k in missing_cols):
        return GateResult(
            gate="pipeline_contracts",
            passed=False,
            blocking=True,
            detail=f"missing required columns: {missing_cols}",
            evidence=missing_cols,
        )

    sch_key_unique = int(sch["game_num"].nunique()) == int(len(sch))
    st_key_unique = int(st["game_num"].nunique()) == int(len(st))
    wx_key_unique = int(wx["game_num"].nunique()) == int(len(wx))

    st_join = sch[["game_num"]].merge(st[["game_num"]], on="game_num", how="left", indicator=True)
    wx_join = sch[["game_num"]].merge(wx[["game_num"]], on="game_num", how="left", indicator=True)
    st_missing = int((st_join["_merge"] != "both").sum())
    wx_missing = int((wx_join["_merge"] != "both").sum())

    # Fatigue coverage measured against unique teams scheduled that day.
    scheduled_teams = set(sch["home_cid"].astype(str)) | set(sch["away_cid"].astype(str))
    fatigue_teams = set(fat["canonical_id"].astype(str))
    covered = len(scheduled_teams & fatigue_teams)
    total = max(1, len(scheduled_teams))
    fatigue_coverage = covered / total
    fatigue_ok = fatigue_coverage >= fatigue_min_coverage

    ok = (
        sch_key_unique
        and st_key_unique
        and wx_key_unique
        and st_missing == 0
        and wx_missing == 0
        and fatigue_ok
    )
    detail = (
        "daily schemas, join integrity, and fatigue coverage checks passed"
        if ok
        else "daily contracts failed (keys/joins/fatigue coverage)"
    )
    return GateResult(
        gate="pipeline_contracts",
        passed=ok,
        blocking=True,
        detail=detail,
        evidence={
            "date": date_label,
            "schedule_rows": int(len(sch)),
            "starters_rows": int(len(st)),
            "weather_rows": int(len(wx)),
            "fatigue_rows": int(len(fat)),
            "schedule_game_num_unique": sch_key_unique,
            "starters_game_num_unique": st_key_unique,
            "weather_game_num_unique": wx_key_unique,
            "missing_starters_join_rows": st_missing,
            "missing_weather_join_rows": wx_missing,
            "fatigue_coverage": fatigue_coverage,
            "fatigue_min_coverage_required": fatigue_min_coverage,
            "paths": {
                "schedule": str(schedule),
                "starters": str(starters),
                "weather": str(weather),
                "fatigue": str(fatigue),
            },
        },
    )


def gate_robustness(repo_root: Path, edge_threshold: float, min_roi: float, max_drawdown: float, min_positive_regimes: float) -> GateResult:
    sweeps = repo_root / "data/processed/audit_sweeps"
    threshold_csv = sweeps / "edge_threshold_sweep.csv"
    regime_csv = sweeps / "regime_robustness.csv"
    missing = [str(p) for p in [threshold_csv, regime_csv] if not p.exists()]
    if missing:
        return GateResult(
            gate="robustness_gates",
            passed=False,
            blocking=True,
            detail=f"missing robustness artifacts: {missing}",
            evidence={"missing_files": missing},
        )

    edge_df = pd.read_csv(threshold_csv)
    reg_df = pd.read_csv(regime_csv)
    req_edge = {"threshold"}
    req_reg = {"slice", "roi"}
    if not req_edge.issubset(edge_df.columns) or not req_reg.issubset(reg_df.columns):
        return GateResult(
            gate="robustness_gates",
            passed=False,
            blocking=True,
            detail="robustness artifact schema mismatch",
            evidence={"edge_cols": list(edge_df.columns), "regime_cols": list(reg_df.columns)},
        )

    roi_col = "roi_tradable" if "roi_tradable" in edge_df.columns else ("roi" if "roi" in edge_df.columns else None)
    dd_col = "max_dd_tradable" if "max_dd_tradable" in edge_df.columns else ("max_dd" if "max_dd" in edge_df.columns else None)
    if roi_col is None or dd_col is None:
        return GateResult(
            gate="robustness_gates",
            passed=False,
            blocking=True,
            detail="robustness artifact missing ROI/drawdown columns",
            evidence={"edge_cols": list(edge_df.columns)},
        )
    edge_df["threshold"] = pd.to_numeric(edge_df["threshold"], errors="coerce")
    edge_df[roi_col] = pd.to_numeric(edge_df[roi_col], errors="coerce")
    edge_df[dd_col] = pd.to_numeric(edge_df[dd_col], errors="coerce")
    edge_df = edge_df.dropna(subset=["threshold", roi_col, dd_col])

    idx = (edge_df["threshold"] - edge_threshold).abs().idxmin()
    row = edge_df.loc[idx]
    roi_ok = float(row[roi_col]) >= min_roi
    dd_ok = float(row[dd_col]) <= max_drawdown

    date_blocks = reg_df[reg_df["slice"].astype(str).str.startswith("date_block_")].copy()
    date_blocks["roi"] = pd.to_numeric(date_blocks["roi"], errors="coerce")
    date_blocks = date_blocks.dropna(subset=["roi"])
    if len(date_blocks) == 0:
        return GateResult(
            gate="robustness_gates",
            passed=False,
            blocking=True,
            detail="no date-block rows found in regime robustness file",
            evidence={"path": str(regime_csv)},
        )
    pos_ratio = float((date_blocks["roi"] > 0.0).mean())
    regime_ok = pos_ratio >= min_positive_regimes

    ok = roi_ok and dd_ok and regime_ok
    detail = (
        "robustness gates passed"
        if ok
        else "robustness gates failed (ROI/drawdown/regime consistency)"
    )
    return GateResult(
        gate="robustness_gates",
        passed=ok,
        blocking=True,
        detail=detail,
        evidence={
            "edge_threshold_used": edge_threshold,
            "edge_row": {
                "threshold": float(row["threshold"]),
                "roi": float(row[roi_col]),
                "max_dd": float(row[dd_col]),
                "n_bets": int(row["n"]) if "n" in row.index else None,
            },
            "roi_column_used": roi_col,
            "max_dd_column_used": dd_col,
            "min_roi_required": min_roi,
            "max_drawdown_allowed": max_drawdown,
            "date_blocks_positive_ratio": pos_ratio,
            "min_positive_ratio_required": min_positive_regimes,
            "paths": {"edge_threshold_sweep": str(threshold_csv), "regime_robustness": str(regime_csv)},
        },
    )


def gate_promotion_checks(repo_root: Path) -> GateResult:
    scorecard_csv = repo_root / "data/processed/audit_sweeps/promotion_scorecard.csv"
    anchor_oos_csv = repo_root / "data/processed/audit_sweeps/anchor_policy_oos.csv"
    missing = [str(p) for p in [scorecard_csv, anchor_oos_csv] if not p.exists()]
    if missing:
        return GateResult(
            gate="promotion_checks",
            passed=False,
            blocking=True,
            detail=f"missing promotion artifacts: {missing}",
            evidence={"missing_files": missing},
        )

    sc = pd.read_csv(scorecard_csv)
    req = {"criterion", "passed", "blocking"}
    if not req.issubset(sc.columns):
        return GateResult(
            gate="promotion_checks",
            passed=False,
            blocking=True,
            detail="promotion scorecard schema mismatch",
            evidence={"columns": list(sc.columns)},
        )
    sc["passed"] = sc["passed"].astype(str).str.lower().isin({"true", "1", "yes"})
    sc["blocking"] = sc["blocking"].astype(str).str.lower().isin({"true", "1", "yes"})
    blocking_fails = sc[(sc["blocking"]) & (~sc["passed"])]
    canary_row = sc[sc["criterion"] == "canary_dynamic_vs_fixed_sharpe_oos"]
    fail_fast_row = sc[sc["criterion"] == "fail_fast_quote_coverage"]
    canary_pass = bool(canary_row["passed"].iloc[0]) if len(canary_row) else False
    fail_fast_pass = bool(fail_fast_row["passed"].iloc[0]) if len(fail_fast_row) else False
    ok = len(blocking_fails) == 0 and canary_pass and fail_fast_pass
    return GateResult(
        gate="promotion_checks",
        passed=ok,
        blocking=True,
        detail="promotion scorecard + canary/fail-fast checks passed" if ok else "promotion checks failed",
        evidence={
            "blocking_failures": blocking_fails[["criterion", "passed"]].to_dict("records"),
            "canary_pass": canary_pass,
            "fail_fast_pass": fail_fast_pass,
            "scorecard_path": str(scorecard_csv),
            "anchor_oos_path": str(anchor_oos_csv),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Automated betting readiness scorecard.")
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--date", type=str, default=None, help="Daily date folder (YYYY-MM-DD). Default: latest in data/daily.")
    p.add_argument("--fatigue-min-coverage", type=float, default=0.80)
    p.add_argument("--edge-threshold", type=float, default=0.02)
    p.add_argument("--min-roi", type=float, default=0.0, help="Minimum ROI required at edge threshold.")
    p.add_argument("--max-drawdown", type=float, default=25.0, help="Max drawdown units allowed at edge threshold.")
    p.add_argument("--min-positive-date-blocks", type=float, default=0.67, help="Min fraction of positive ROI date blocks.")
    p.add_argument("--out-json", type=Path, default=None, help="Optional scorecard output JSON path.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    daily_root = repo_root / "data/daily"
    chosen_date = args.date
    if not chosen_date:
        latest = _latest_date_folder(daily_root)
        if latest is None:
            print("BET_READY: NO")
            print("Reason: no daily folders found under data/daily")
            return 1
        chosen_date = latest.name

    gates: list[GateResult] = []
    gates.append(gate_calibration_parity(repo_root))
    gates.append(gate_prediction_invariants(repo_root, chosen_date))
    gates.append(gate_pipeline_contracts(repo_root, chosen_date, args.fatigue_min_coverage))
    gates.append(
        gate_robustness(
            repo_root=repo_root,
            edge_threshold=args.edge_threshold,
            min_roi=args.min_roi,
            max_drawdown=args.max_drawdown,
            min_positive_regimes=args.min_positive_date_blocks,
        )
    )
    gates.append(gate_promotion_checks(repo_root))

    blocking_failed = [g for g in gates if g.blocking and not g.passed]
    bet_ready = len(blocking_failed) == 0

    print("=" * 84)
    print("NCAA BASEBALL AUDIT SCORECARD")
    print("=" * 84)
    print(f"Date evaluated: {chosen_date}")
    print(f"BET_READY: {'YES' if bet_ready else 'NO'}")
    print()
    print("Gate Results:")
    for g in gates:
        status = "PASS" if g.passed else "FAIL"
        mode = "BLOCK" if g.blocking else "INFO"
        print(f"- [{status}] [{mode}] {g.gate}: {g.detail}")

    if blocking_failed:
        print()
        print("Blocking reasons:")
        for g in blocking_failed:
            print(f"- {g.gate}: {g.detail}")

    payload = {
        "bet_ready": bet_ready,
        "date_evaluated": chosen_date,
        "gates": [asdict(g) for g in gates],
        "blocking_failures": [asdict(g) for g in blocking_failed],
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        print()
        print(f"Saved scorecard JSON: {args.out_json}")

    # Non-zero exit when not bet-ready for easy CI integration.
    return 0 if bet_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
