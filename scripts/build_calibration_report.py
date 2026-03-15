"""Build daily market-coherent calibration report.

Checks consistency across ML / totals / runline tails in one place, per game.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _to_float(v, default=None):
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _starter_certainty_bucket(home_idx: float, away_idx: float) -> str:
    h_known = home_idx > 0
    a_known = away_idx > 0
    if h_known and a_known:
        return "high"
    if h_known or a_known:
        return "mixed"
    return "low"


def _bullpen_edge_bucket(home_bp: float | None, away_bp: float | None) -> str:
    if home_bp is None or away_bp is None:
        return "unknown"
    gap = abs(home_bp - away_bp)
    if gap >= 0.08:
        return "high_edge"
    if gap >= 0.04:
        return "medium_edge"
    return "low_edge"


def build_calibration_report(predictions_csv: Path, out_csv: Path, out_md: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(predictions_csv, dtype=str).fillna("")

    rows: list[dict[str, object]] = []
    for _, r in df.iterrows():
        home = str(r.get("home", "")).strip()
        away = str(r.get("away", "")).strip()
        wp_h = _to_float(r.get("home_win_prob"), 0.5)
        wp_a = _to_float(r.get("away_win_prob"), 0.5)
        rl_h = _to_float(r.get("home_rl_cover"), 0.0)
        rl_a = _to_float(r.get("away_rl_cover"), 0.0)
        h2 = _to_float(r.get("home_win_by_2plus"), rl_h)
        h3 = _to_float(r.get("home_win_by_3plus"), None)
        h4 = _to_float(r.get("home_win_by_4plus"), None)
        h5 = _to_float(r.get("home_win_by_5plus"), None)
        h6 = _to_float(r.get("home_win_by_6plus"), None)
        a2 = _to_float(r.get("away_win_by_2plus"), rl_a)
        a3 = _to_float(r.get("away_win_by_3plus"), None)
        a4 = _to_float(r.get("away_win_by_4plus"), None)
        a5 = _to_float(r.get("away_win_by_5plus"), None)
        a6 = _to_float(r.get("away_win_by_6plus"), None)
        exp_total = _to_float(r.get("exp_total"), 0.0)
        mkt_home = _to_float(r.get("mkt_home_win_prob"), None)
        mkt_total = _to_float(r.get("mkt_total_line"), None)
        home_starter_idx = _to_float(r.get("home_starter_idx"), 0.0) or 0.0
        away_starter_idx = _to_float(r.get("away_starter_idx"), 0.0) or 0.0
        home_bp_adj = _to_float(r.get("home_bullpen_adj"), None)
        away_bp_adj = _to_float(r.get("away_bullpen_adj"), None)
        starter_bucket = _starter_certainty_bucket(home_starter_idx, away_starter_idx)
        bullpen_bucket = _bullpen_edge_bucket(home_bp_adj, away_bp_adj)

        # Cross-market consistency checks
        ml_sum_err = abs((wp_h + wp_a) - 1.0)
        rl_leq_win_err = max(0.0, h2 - wp_h) + max(0.0, a2 - wp_a)
        mono_err = 0.0
        for chain in ([h2, h3, h4, h5, h6], [a2, a3, a4, a5, a6]):
            for i in range(len(chain) - 1):
                x = chain[i]
                y = chain[i + 1]
                if x is None or y is None:
                    continue
                mono_err += max(0.0, y - x)

        market_ml_delta = None if mkt_home is None else (wp_h - mkt_home)
        market_total_delta = None if mkt_total is None else (exp_total - mkt_total)

        calib_risk = 0.0
        calib_risk += min(1.0, ml_sum_err * 20.0)
        calib_risk += min(1.0, rl_leq_win_err * 12.0)
        calib_risk += min(1.0, mono_err * 20.0)
        if market_ml_delta is not None:
            calib_risk += min(1.0, abs(market_ml_delta) * 4.0)
        if market_total_delta is not None:
            calib_risk += min(1.0, abs(market_total_delta) / 2.0)
        calib_risk = min(1.0, calib_risk / 5.0)

        if calib_risk >= 0.6:
            calib_flag = "high"
        elif calib_risk >= 0.3:
            calib_flag = "medium"
        else:
            calib_flag = "low"

        rows.append(
            {
                "game_num": int(float(r.get("game_num", "0") or 0)),
                "game": f"{away} @ {home}",
                "home_win_prob": wp_h,
                "away_win_prob": wp_a,
                "exp_total": exp_total,
                "mkt_home_win_prob": mkt_home,
                "mkt_total_line": mkt_total,
                "market_ml_delta": market_ml_delta,
                "market_total_delta": market_total_delta,
                "starter_certainty_bucket": starter_bucket,
                "bullpen_edge_bucket": bullpen_bucket,
                "ml_sum_err": ml_sum_err,
                "rl_leq_win_err": rl_leq_win_err,
                "tail_monotonicity_err": mono_err,
                "calibration_risk": calib_risk,
                "calibration_flag": calib_flag,
            }
        )

    out = pd.DataFrame(rows).sort_values(["calibration_risk", "game_num"], ascending=[False, True])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        n = len(out)
        hi = int((out["calibration_flag"] == "high").sum()) if n else 0
        med = int((out["calibration_flag"] == "medium").sum()) if n else 0
        lo = int((out["calibration_flag"] == "low").sum()) if n else 0
        top = out.head(12)
        starter_seg = (
            out.groupby("starter_certainty_bucket", dropna=False)
            .agg(
                games=("game", "count"),
                mean_risk=("calibration_risk", "mean"),
                mean_abs_ml_delta=("market_ml_delta", lambda s: pd.to_numeric(s, errors="coerce").abs().mean()),
                mean_abs_total_delta=("market_total_delta", lambda s: pd.to_numeric(s, errors="coerce").abs().mean()),
            )
            .reset_index()
            .sort_values("games", ascending=False)
        )
        bullpen_seg = (
            out.groupby("bullpen_edge_bucket", dropna=False)
            .agg(
                games=("game", "count"),
                mean_risk=("calibration_risk", "mean"),
                mean_abs_ml_delta=("market_ml_delta", lambda s: pd.to_numeric(s, errors="coerce").abs().mean()),
                mean_abs_total_delta=("market_total_delta", lambda s: pd.to_numeric(s, errors="coerce").abs().mean()),
            )
            .reset_index()
            .sort_values("games", ascending=False)
        )
        lines = [
            "# Daily Calibration Report",
            "",
            f"- Games: {n}",
            f"- High risk: {hi}",
            f"- Medium risk: {med}",
            f"- Low risk: {lo}",
            "",
            "## Highest-Risk Games",
            "",
            "| Game | Risk | Flag | ML delta | Total delta | Tail err |",
            "|---|---:|---|---:|---:|---:|",
        ]
        for _, r in top.iterrows():
            ml_delta = ""
            if not pd.isna(r["market_ml_delta"]):
                ml_delta = f"{float(r['market_ml_delta']):+.3f}"
            total_delta = ""
            if not pd.isna(r["market_total_delta"]):
                total_delta = f"{float(r['market_total_delta']):+.2f}"
            lines.append(
                f"| {r['game']} | {float(r['calibration_risk']):.3f} | {r['calibration_flag']} | "
                f"{ml_delta} | "
                f"{total_delta} | "
                f"{float(r['tail_monotonicity_err']):.3f} |"
            )
        lines.extend(
            [
                "",
                "## Segmented Calibration: Starter Certainty",
                "",
                "| Bucket | Games | Mean risk | Mean abs ML delta | Mean abs Total delta |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for _, r in starter_seg.iterrows():
            ml_abs = "" if pd.isna(r["mean_abs_ml_delta"]) else f"{float(r['mean_abs_ml_delta']):.3f}"
            tot_abs = "" if pd.isna(r["mean_abs_total_delta"]) else f"{float(r['mean_abs_total_delta']):.3f}"
            lines.append(
                f"| {r['starter_certainty_bucket']} | {int(r['games'])} | {float(r['mean_risk']):.3f} | {ml_abs} | {tot_abs} |"
            )
        lines.extend(
            [
                "",
                "## Segmented Calibration: Bullpen Edge",
                "",
                "| Bucket | Games | Mean risk | Mean abs ML delta | Mean abs Total delta |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for _, r in bullpen_seg.iterrows():
            ml_abs = "" if pd.isna(r["mean_abs_ml_delta"]) else f"{float(r['mean_abs_ml_delta']):.3f}"
            tot_abs = "" if pd.isna(r["mean_abs_total_delta"]) else f"{float(r['mean_abs_total_delta']):.3f}"
            lines.append(
                f"| {r['bullpen_edge_bucket']} | {int(r['games'])} | {float(r['mean_risk']):.3f} | {ml_abs} | {tot_abs} |"
            )
        out_md.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build market-coherent calibration report.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()
    build_calibration_report(args.predictions, args.out_csv, args.out_md)
    print(f"Wrote calibration report: {args.out_csv}")
    if args.out_md:
        print(f"Wrote calibration markdown: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

