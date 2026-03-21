#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from robustness_reporting import (
    add_regime_columns,
    apply_uncertainty_columns,
    build_regime_robustness_table,
    evaluate_threshold_strategy,
)


def _parse_float_list(text: str) -> list[float]:
    vals: list[float] = []
    for chunk in text.split(","):
        s = chunk.strip()
        if not s:
            continue
        vals.append(float(s))
    return vals


def _run_backtest_for_scale(args, scale: float, out_csv: Path) -> None:
    cmd = [
        sys.executable,
        str(args.backtest_script),
        "--odds", str(args.odds),
        "--run-events", str(args.run_events),
        "--posterior", str(args.posterior),
        "--meta", str(args.meta),
        "--team-index", str(args.team_index),
        "--pitcher-index", str(args.pitcher_index),
        "--teams-csv", str(args.teams_csv),
        "--team-table", str(args.team_table),
        "--N", str(args.N),
        "--seed", str(args.seed),
        "--ha-target", str(args.ha_target),
        "--spread-scale", str(scale),
        "--d1b-boost", str(args.d1b_boost),
        "--out", str(out_csv),
        "--min-bet-confidence", str(args.min_bet_confidence),
        "--default-weather-confidence", str(args.default_weather_confidence),
        "--default-fatigue-confidence", str(args.default_fatigue_confidence),
    ]
    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Walk-forward sweep for spread-scale + edge threshold with OOS acceptance checks."
    )
    p.add_argument("--backtest-script", type=Path, default=Path("scripts/backtest_vs_market.py"))
    p.add_argument("--odds", type=Path, default=Path("data/raw/odds/odds_historical_2026.jsonl"))
    p.add_argument("--run-events", type=Path, default=Path("data/processed/run_events_expanded.csv"))
    p.add_argument("--posterior", type=Path, default=Path("data/processed/run_event_posterior_2k.csv"))
    p.add_argument("--meta", type=Path, default=Path("data/processed/run_event_fit_meta.json"))
    p.add_argument("--team-index", type=Path, default=Path("data/processed/run_event_team_index.csv"))
    p.add_argument("--pitcher-index", type=Path, default=Path("data/processed/run_event_pitcher_index.csv"))
    p.add_argument("--teams-csv", type=Path, default=Path("data/registries/canonical_teams_2026.csv"))
    p.add_argument("--team-table", type=Path, default=Path("data/processed/team_table.csv"))
    p.add_argument("--N", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ha-target", type=float, default=0.05)
    p.add_argument("--d1b-boost", type=float, default=1.0)
    p.add_argument("--spread-scales", type=str, default="0.85,1.0,1.15")
    p.add_argument("--thresholds", type=str, default="0.02,0.03,0.05,0.08,0.10")
    p.add_argument("--train-days", type=int, default=21)
    p.add_argument("--test-days", type=int, default=7)
    p.add_argument("--step-days", type=int, default=7)
    p.add_argument("--dd-penalty", type=float, default=1.0, help="Objective = ROI - dd_penalty*(max_dd/n)")
    p.add_argument("--prob-col", type=str, default="model_calibrated")
    p.add_argument("--baseline-spread-scale", type=float, default=1.0)
    p.add_argument("--baseline-threshold", type=float, default=0.05)
    p.add_argument("--min-bet-confidence", type=float, default=0.55)
    p.add_argument("--default-weather-confidence", type=float, default=0.50)
    p.add_argument("--default-fatigue-confidence", type=float, default=0.50)
    p.add_argument("--min-folds", type=int, default=2)
    p.add_argument("--min-oos-roi", type=float, default=0.0)
    p.add_argument("--max-oos-drawdown", type=float, default=999.0)
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/audit_sweeps"))
    p.add_argument("--details-dir", type=Path, default=Path("data/processed/audit_sweeps/details"))
    p.add_argument("--reuse-details", action="store_true", help="Use existing detail CSVs when present")
    return p


def main() -> int:
    args = build_parser().parse_args()
    scales = _parse_float_list(args.spread_scales)
    thresholds = _parse_float_list(args.thresholds)
    if not scales or not thresholds:
        raise SystemExit("spread-scales and thresholds must be non-empty")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.details_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build/load per-scale detailed backtest files.
    by_scale: dict[float, pd.DataFrame] = {}
    for scale in scales:
        detail_path = args.details_dir / f"backtest_vs_market_spread_{scale:.3f}.csv"
        if not (args.reuse_details and detail_path.exists()):
            print(f"[walk-forward] running backtest_vs_market for spread_scale={scale:.3f}", file=sys.stderr)
            _run_backtest_for_scale(args, scale, detail_path)
        df = pd.read_csv(detail_path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = apply_uncertainty_columns(
            df=df,
            min_bet_confidence=args.min_bet_confidence,
            default_weather_confidence=args.default_weather_confidence,
            default_fatigue_confidence=args.default_fatigue_confidence,
        )
        df = add_regime_columns(df, args.teams_csv)
        by_scale[scale] = df

    # 2) Walk-forward fold construction from baseline scale timeline.
    base_df = by_scale.get(args.baseline_spread_scale)
    if base_df is None:
        base_df = by_scale[scales[0]]
    all_dates = sorted(pd.to_datetime(base_df["date"].dropna().unique()))
    if not all_dates:
        raise SystemExit("No dates found in backtest detail data.")

    first_date = all_dates[0]
    last_date = all_dates[-1]
    fold_rows: list[dict[str, object]] = []
    train_grid_rows: list[dict[str, object]] = []
    selected_bets_all: list[pd.DataFrame] = []
    baseline_bets_all: list[pd.DataFrame] = []

    fold_id = 0
    cursor = first_date + pd.Timedelta(days=max(1, args.train_days))
    while cursor <= last_date:
        fold_id += 1
        test_start = cursor
        test_end = min(last_date, test_start + pd.Timedelta(days=max(1, args.test_days) - 1))
        train_end = test_start - pd.Timedelta(days=1)
        train_start = train_end - pd.Timedelta(days=max(1, args.train_days) - 1)

        best = None
        best_eval = None
        for scale in scales:
            df_scale = by_scale[scale]
            train_df = df_scale[(df_scale["date"] >= train_start) & (df_scale["date"] <= train_end)]
            for th in thresholds:
                ev = evaluate_threshold_strategy(
                    df=train_df,
                    prob_col=args.prob_col,
                    threshold=th,
                    dd_penalty=args.dd_penalty,
                    min_bet_confidence=args.min_bet_confidence,
                )
                train_grid_rows.append(
                    {
                        "fold": fold_id,
                        "train_start": train_start.date().isoformat(),
                        "train_end": train_end.date().isoformat(),
                        "spread_scale": scale,
                        "threshold": th,
                        "n": ev.n,
                        "won": ev.won,
                        "win_rate": ev.win_rate,
                        "pnl": ev.pnl,
                        "roi": ev.roi,
                        "max_dd": ev.max_dd,
                        "objective": ev.objective,
                    }
                )
                if best is None or ev.objective > best_eval.objective:
                    best = (scale, th)
                    best_eval = ev

        if best is None or best_eval is None:
            break

        sel_scale, sel_th = best
        oos_df_sel = by_scale[sel_scale][(by_scale[sel_scale]["date"] >= test_start) & (by_scale[sel_scale]["date"] <= test_end)]
        sel_oos = evaluate_threshold_strategy(
            df=oos_df_sel,
            prob_col=args.prob_col,
            threshold=sel_th,
            dd_penalty=args.dd_penalty,
            min_bet_confidence=args.min_bet_confidence,
        )
        if not sel_oos.bets.empty:
            b = sel_oos.bets.copy()
            b["fold"] = fold_id
            b["date_block"] = f"{test_start.date().isoformat()}_{test_end.date().isoformat()}"
            b["spread_scale"] = sel_scale
            b["threshold"] = sel_th
            selected_bets_all.append(b)

        oos_df_base = by_scale.get(args.baseline_spread_scale, by_scale[scales[0]])
        oos_df_base = oos_df_base[(oos_df_base["date"] >= test_start) & (oos_df_base["date"] <= test_end)]
        base_oos = evaluate_threshold_strategy(
            df=oos_df_base,
            prob_col=args.prob_col,
            threshold=args.baseline_threshold,
            dd_penalty=args.dd_penalty,
            min_bet_confidence=args.min_bet_confidence,
        )
        if not base_oos.bets.empty:
            bb = base_oos.bets.copy()
            bb["fold"] = fold_id
            bb["date_block"] = f"{test_start.date().isoformat()}_{test_end.date().isoformat()}"
            bb["spread_scale"] = args.baseline_spread_scale
            bb["threshold"] = args.baseline_threshold
            baseline_bets_all.append(bb)

        fold_rows.append(
            {
                "fold": fold_id,
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "selected_spread_scale": sel_scale,
                "selected_threshold": sel_th,
                "train_objective": best_eval.objective,
                "train_roi": best_eval.roi,
                "train_max_dd": best_eval.max_dd,
                "oos_n": sel_oos.n,
                "oos_roi": sel_oos.roi,
                "oos_max_dd": sel_oos.max_dd,
                "oos_objective": sel_oos.objective,
                "baseline_n": base_oos.n,
                "baseline_roi": base_oos.roi,
                "baseline_max_dd": base_oos.max_dd,
                "baseline_objective": base_oos.objective,
            }
        )

        cursor = cursor + pd.Timedelta(days=max(1, args.step_days))

    folds_df = pd.DataFrame(fold_rows)
    grid_df = pd.DataFrame(train_grid_rows)
    folds_df.to_csv(args.out_dir / "walk_forward_folds.csv", index=False)
    grid_df.to_csv(args.out_dir / "walk_forward_train_grid.csv", index=False)

    sel_bets = pd.concat(selected_bets_all, ignore_index=True) if selected_bets_all else pd.DataFrame()
    base_bets = pd.concat(baseline_bets_all, ignore_index=True) if baseline_bets_all else pd.DataFrame()
    if not sel_bets.empty:
        sel_bets.to_csv(args.out_dir / "walk_forward_oos_selected_bets.csv", index=False)
    if not base_bets.empty:
        base_bets.to_csv(args.out_dir / "walk_forward_oos_baseline_bets.csv", index=False)

    # 3) Summary and acceptance check.
    def _summary(bets: pd.DataFrame) -> dict[str, float]:
        if bets.empty:
            return {"n": 0, "won": 0, "win_rate": 0.0, "pnl": 0.0, "roi": 0.0, "max_dd": 0.0, "objective": -1e9}
        pnl = pd.to_numeric(bets["bet_profit"], errors="coerce").fillna(0.0)
        n = int(len(bets))
        won = int(pd.to_numeric(bets["bet_won"], errors="coerce").fillna(0).sum())
        curve = pnl.cumsum()
        peaks = curve.cummax()
        max_dd = float((peaks - curve).max())
        roi = float(pnl.sum() / max(1, n))
        objective = float(roi - args.dd_penalty * (max_dd / max(1, n)))
        return {
            "n": n,
            "won": won,
            "win_rate": won / max(1, n),
            "pnl": float(pnl.sum()),
            "roi": roi,
            "max_dd": max_dd,
            "objective": objective,
        }

    selected_summary = _summary(sel_bets)
    baseline_summary = _summary(base_bets)
    n_folds = int(len(folds_df))

    pass_folds = n_folds >= args.min_folds
    pass_vs_baseline = selected_summary["objective"] > baseline_summary["objective"]
    pass_roi = selected_summary["roi"] >= args.min_oos_roi
    pass_dd = selected_summary["max_dd"] <= args.max_oos_drawdown
    accepted = bool(pass_folds and pass_vs_baseline and pass_roi and pass_dd)

    acceptance = {
        "accepted": accepted,
        "criteria": {
            "min_folds": args.min_folds,
            "min_oos_roi": args.min_oos_roi,
            "max_oos_drawdown": args.max_oos_drawdown,
            "must_beat_baseline_objective": True,
        },
        "checks": {
            "folds_check_pass": pass_folds,
            "vs_baseline_check_pass": pass_vs_baseline,
            "roi_check_pass": pass_roi,
            "drawdown_check_pass": pass_dd,
        },
        "selected_oos": selected_summary,
        "baseline_oos": baseline_summary,
        "n_folds": n_folds,
    }
    (args.out_dir / "walk_forward_acceptance.json").write_text(json.dumps(acceptance, indent=2), encoding="utf-8")

    # 4) Required sweep artifacts used by audit scorecard.
    chosen_scale = (
        float(folds_df["selected_spread_scale"].mode().iloc[0])
        if not folds_df.empty
        else args.baseline_spread_scale
    )
    oos_dates = (
        pd.to_datetime(sel_bets["date"], errors="coerce").dropna().unique().tolist()
        if not sel_bets.empty
        else []
    )
    oos_ref = by_scale.get(chosen_scale, by_scale[scales[0]])
    if oos_dates:
        oos_ref = oos_ref[oos_ref["date"].isin(pd.to_datetime(oos_dates))]

    th_rows: list[dict[str, object]] = []
    for th in sorted(set(thresholds + [args.baseline_threshold])):
        ev = evaluate_threshold_strategy(
            df=oos_ref,
            prob_col=args.prob_col,
            threshold=th,
            dd_penalty=args.dd_penalty,
            min_bet_confidence=args.min_bet_confidence,
        )
        th_rows.append(
            {
                "threshold": th,
                "spread_scale": chosen_scale,
                "n": ev.n,
                "won": ev.won,
                "win_rate": ev.win_rate,
                "pnl": ev.pnl,
                "roi": ev.roi,
                "max_dd": ev.max_dd,
                "objective": ev.objective,
            }
        )
    pd.DataFrame(th_rows).sort_values("threshold").to_csv(args.out_dir / "edge_threshold_sweep.csv", index=False)

    regime_df = build_regime_robustness_table(sel_bets if not sel_bets.empty else pd.DataFrame())
    regime_df.to_csv(args.out_dir / "regime_robustness.csv", index=False)

    print(f"Wrote walk-forward outputs to {args.out_dir}")
    print(f"Acceptance: {'PASS' if accepted else 'FAIL'}")
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
