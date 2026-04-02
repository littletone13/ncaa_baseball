#!/usr/bin/env python3
"""
backtest_vs_market.py — Grade model predictions against market odds and actual results.

Fixes from audit:
  #5: Uses RAW market American ML for payout (not fair/devigged odds)
  #7: Uses latest available snapshot per game as closing proxy
  #9: Handles doubleheaders via first-match (documented limitation)

Stores raw vig prices (ML home/away, over/under juice) for accurate P&L.

Usage:
  python3 scripts/backtest_vs_market.py --out data/processed/backtest_market.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401


# ── Odds math ────────────────────────────────────────────────────────────────

def american_to_implied(ml: float) -> float:
    """American ML → raw implied probability (includes vig)."""
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    elif ml > 0:
        return 100 / (ml + 100)
    return 0.5


def american_to_decimal(ml: float) -> float:
    """American ML → decimal odds (what the book actually pays)."""
    if ml < 0:
        return 1 + 100 / abs(ml)
    elif ml > 0:
        return 1 + ml / 100
    return 2.0


def devig_proportional(home_impl: float, away_impl: float) -> float:
    """Devig home probability using proportional/multiplicative method."""
    total = home_impl + away_impl
    if total > 0:
        return home_impl / total
    return 0.5


def prob_to_american(p: float) -> int:
    if p >= 0.5:
        return -round(p / (1 - p) * 100)
    else:
        return round((1 - p) / p * 100)


# ── Parse odds from pull log ─────────────────────────────────────────────────

def parse_odds_log(
    odds_log: Path,
    canonical_csv: Path,
) -> dict[tuple[str, str, str], dict]:
    """
    Parse odds pull log → per-game closing odds with RAW market MLs and juice.

    Returns dict keyed by (game_date, home_cid, away_cid) with:
      - mkt_home_prob: devigged fair probability (for edge calc only)
      - raw_home_ml: consensus raw American ML for home (for payout)
      - raw_away_ml: consensus raw American ML for away (for payout)
      - mkt_total: median total line
      - raw_over_price: raw American odds on over (for payout)
      - raw_under_price: raw American odds on under (for payout)
      - n_books_ml, n_books_total: book counts
    """
    # Build odds_name → canonical_id
    canon = pd.read_csv(canonical_csv, dtype=str)
    odds_to_cid: dict[str, str] = {}
    for _, r in canon.iterrows():
        cid = str(r.get("canonical_id", "")).strip()
        for col in ("odds_api_name", "espn_name", "team_name"):
            n = str(r.get(col, "")).strip()
            if n and n != "nan":
                odds_to_cid[n] = cid

    game_odds: dict[tuple[str, str, str], dict] = {}

    with open(odds_log) as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue

            commence = row.get("commence_time", "")
            if not commence or not commence.startswith("2026"):
                continue

            game_date = commence[:10]
            home_name = row.get("home_team", "")
            away_name = row.get("away_team", "")
            if not home_name or not away_name:
                continue

            h_cid = odds_to_cid.get(home_name, "")
            a_cid = odds_to_cid.get(away_name, "")
            if not h_cid or not a_cid:
                continue

            game_key = (game_date, h_cid, a_cid)

            # Extract per-book data
            home_mls: list[float] = []
            away_mls: list[float] = []
            total_lines: list[float] = []
            over_prices: list[float] = []
            under_prices: list[float] = []

            for bk in row.get("bookmaker_lines", []):
                for mkt in bk.get("markets", []):
                    outcomes = mkt.get("outcomes", [])
                    if mkt.get("key") == "h2h" and len(outcomes) >= 2:
                        hml = aml = None
                        for o in outcomes:
                            if o.get("name") == home_name:
                                hml = o.get("price")
                            elif o.get("name") == away_name:
                                aml = o.get("price")
                        if hml is not None and aml is not None:
                            home_mls.append(float(hml))
                            away_mls.append(float(aml))
                    elif mkt.get("key") == "totals":
                        for o in outcomes:
                            if o.get("point"):
                                if o.get("name") == "Over":
                                    total_lines.append(float(o["point"]))
                                    if o.get("price"):
                                        over_prices.append(float(o["price"]))
                                elif o.get("name") == "Under":
                                    if o.get("price"):
                                        under_prices.append(float(o["price"]))

            if not home_mls and not total_lines:
                continue

            rec: dict = {}

            if home_mls:
                # Consensus raw ML (median — robust to outlier books)
                rec["raw_home_ml"] = float(np.median(home_mls))
                rec["raw_away_ml"] = float(np.median(away_mls))
                rec["n_books_ml"] = len(home_mls)

                # Devig for edge calc only
                home_implieds = [american_to_implied(ml) for ml in home_mls]
                away_implieds = [american_to_implied(ml) for ml in away_mls]
                rec["mkt_home_prob"] = devig_proportional(
                    np.mean(home_implieds), np.mean(away_implieds)
                )

            if total_lines:
                rec["mkt_total"] = float(np.median(total_lines))
                rec["n_books_total"] = len(total_lines)
                # Store actual juice on over AND under
                rec["raw_over_price"] = float(np.median(over_prices)) if over_prices else -110.0
                rec["raw_under_price"] = float(np.median(under_prices)) if under_prices else -110.0

            # Always keep latest snapshot (overwrite earlier)
            game_odds[game_key] = rec

    return game_odds


# ── Load model predictions ───────────────────────────────────────────────────

def load_all_predictions(pred_dir: Path) -> dict[tuple[str, str, str], dict]:
    """Load all prediction CSVs, keyed by (date, home_cid, away_cid)."""
    preds: dict[tuple[str, str, str], dict] = {}
    for f in sorted(pred_dir.glob("predictions_2026-*_standard.csv")):
        date = f.stem.replace("predictions_", "").replace("_standard", "")
        if len(date) != 10:
            continue
        try:
            df = pd.read_csv(f, dtype=str)
        except Exception:
            continue
        if "home_win_prob" not in df.columns:
            continue
        for _, row in df.iterrows():
            h_cid = str(row.get("home_cid", "")).strip()
            a_cid = str(row.get("away_cid", "")).strip()
            if not h_cid or not a_cid:
                continue
            key = (date, h_cid, a_cid)
            try:
                preds[key] = {
                    "model_home_prob": float(row["home_win_prob"]),
                    "model_total": float(row["exp_total"]) if pd.notna(row.get("exp_total")) else None,
                    "home_name": str(row.get("home", "")),
                    "away_name": str(row.get("away", "")),
                }
            except (ValueError, TypeError):
                continue
    return preds


# ── Load game results ────────────────────────────────────────────────────────

def load_results(games_csv: Path) -> dict[tuple[str, str, str], dict]:
    games = pd.read_csv(games_csv, dtype=str)
    games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
    games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")
    games = games[games["home_score"].notna() & games["away_score"].notna()].copy()
    results: dict[tuple[str, str, str], dict] = {}
    for _, r in games.iterrows():
        date = str(r.get("game_date", "")).strip()
        if not date.startswith("2026"):
            continue
        h_cid = str(r.get("home_canonical_id", "")).strip()
        a_cid = str(r.get("away_canonical_id", "")).strip()
        if not h_cid or not a_cid:
            continue
        key = (date, h_cid, a_cid)
        if key not in results:
            results[key] = {
                "home_score": float(r["home_score"]),
                "away_score": float(r["away_score"]),
                "actual_total": float(r["home_score"]) + float(r["away_score"]),
                "home_win": int(float(r["home_score"]) > float(r["away_score"])),
            }
    return results


# ── Run backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    odds_log: Path = Path("data/raw/odds/odds_pull_log.jsonl"),
    canonical_csv: Path = Path("data/registries/canonical_teams_2026.csv"),
    games_csv: Path = Path("data/processed/games.csv"),
    pred_dir: Path = Path("data/processed"),
    out_csv: Path | None = None,
) -> pd.DataFrame:

    print("Loading odds...", file=sys.stderr)
    odds = parse_odds_log(odds_log, canonical_csv)
    print(f"  {len(odds)} games with odds", file=sys.stderr)

    print("Loading predictions...", file=sys.stderr)
    preds = load_all_predictions(pred_dir)
    print(f"  {len(preds)} model predictions", file=sys.stderr)

    print("Loading results...", file=sys.stderr)
    results = load_results(games_csv)
    print(f"  {len(results)} completed games", file=sys.stderr)

    rows = []
    for key in preds:
        if key not in odds or key not in results:
            continue
        p = preds[key]
        o = odds[key]
        r = results[key]

        row = {
            "date": key[0], "home_cid": key[1], "away_cid": key[2],
            "home_name": p.get("home_name", ""),
            "away_name": p.get("away_name", ""),
            "model_home_prob": p["model_home_prob"],
            "model_total": p.get("model_total"),
            "home_win": r["home_win"],
            "actual_total": r["actual_total"],
            "home_score": r["home_score"],
            "away_score": r["away_score"],
        }
        # Raw odds for payout
        for field in ("mkt_home_prob", "raw_home_ml", "raw_away_ml", "n_books_ml",
                       "mkt_total", "raw_over_price", "raw_under_price", "n_books_total"):
            if field in o:
                row[field] = o[field]

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n  MATCHED: {len(df)} games (model + odds + result)", file=sys.stderr)

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"  Wrote → {out_csv}", file=sys.stderr)

    return df


def print_report(df: pd.DataFrame) -> None:
    both_ml = df[df["mkt_home_prob"].notna() & df["raw_home_ml"].notna()].copy()
    both_tot = df[df["model_total"].notna() & df["mkt_total"].notna()].copy()

    print(f"\n{'='*70}")
    print(f"  BACKTEST (CORRECTED) — raw market ML/juice for payout")
    print(f"  {len(df)} total | {len(both_ml)} ML | {len(both_tot)} totals")
    print(f"  {df['date'].min()} to {df['date'].max()}")
    print(f"{'='*70}")

    if len(both_ml) > 0:
        both_ml["model_brier"] = (both_ml["model_home_prob"] - both_ml["home_win"]) ** 2
        both_ml["mkt_brier"] = (both_ml["mkt_home_prob"] - both_ml["home_win"]) ** 2
        bd = both_ml["mkt_brier"].mean() - both_ml["model_brier"].mean()
        print(f"\n  MODEL vs MARKET ({len(both_ml)} games):")
        print(f"  Model Brier: {both_ml['model_brier'].mean():.4f}  Market: {both_ml['mkt_brier'].mean():.4f}  Edge: {bd:+.4f}")

        both_ml["home_edge"] = both_ml["model_home_prob"] - both_ml["mkt_home_prob"]

        print(f"\n  MONEYLINE (payout at RAW market odds):")
        print(f"  {'Edge':<10} {'Bets':>5} {'W-L':>10} {'Win%':>6} {'P&L':>9} {'ROI':>7}")
        print(f"  {'-'*10} {'-'*5} {'-'*10} {'-'*6} {'-'*9} {'-'*7}")

        for thresh in [0.01, 0.03, 0.05, 0.08, 0.10, 0.15]:
            bets = []
            for _, r in both_ml.iterrows():
                he = r["home_edge"]
                if he >= thresh:
                    dec = american_to_decimal(r["raw_home_ml"])
                    won = r["home_win"] == 1
                    bets.append({"won": won, "pnl": (dec - 1) if won else -1, "side": "home"})
                elif -he >= thresh:
                    dec = american_to_decimal(r["raw_away_ml"])
                    won = r["home_win"] == 0
                    bets.append({"won": won, "pnl": (dec - 1) if won else -1, "side": "away"})
            if not bets:
                continue
            bdf = pd.DataFrame(bets)
            n = len(bdf); w = bdf["won"].sum(); pnl = bdf["pnl"].sum()
            print(f"  {thresh*100:>5.0f}%+    {n:>5} {f'{w}-{n-w}':>10} {w/n*100:>5.1f}% {pnl:>+9.2f} {pnl/n*100:>+6.1f}%")

        print(f"\n  SIDE SPLIT (3%+ edge):")
        for side in ["home", "away"]:
            bets = []
            for _, r in both_ml.iterrows():
                e = r["home_edge"] if side == "home" else -r["home_edge"]
                if e < 0.03: continue
                raw_ml = r["raw_home_ml"] if side == "home" else r["raw_away_ml"]
                dec = american_to_decimal(raw_ml)
                won = (r["home_win"] == 1) if side == "home" else (r["home_win"] == 0)
                bets.append({"won": won, "pnl": (dec - 1) if won else -1})
            if bets:
                bdf = pd.DataFrame(bets)
                n = len(bdf); w = bdf["won"].sum(); pnl = bdf["pnl"].sum()
                print(f"    {side.upper()}: {w}-{n-w} ({w/n*100:.1f}%), P&L {pnl:+.2f}, ROI {pnl/n*100:+.1f}%")

    if len(both_tot) > 0:
        both_tot["model_total"] = pd.to_numeric(both_tot["model_total"])
        both_tot["mkt_total"] = pd.to_numeric(both_tot["mkt_total"])
        both_tot["raw_over_price"] = pd.to_numeric(both_tot.get("raw_over_price", -110)).fillna(-110)
        both_tot["raw_under_price"] = pd.to_numeric(both_tot.get("raw_under_price", -110)).fillna(-110)

        model_mae = (both_tot["model_total"] - both_tot["actual_total"]).abs().mean()
        mkt_mae = (both_tot["mkt_total"] - both_tot["actual_total"]).abs().mean()
        print(f"\n  TOTALS ({len(both_tot)} games):")
        print(f"  Model MAE: {model_mae:.2f}  Market MAE: {mkt_mae:.2f}")
        print(f"  Avg over juice: {both_tot['raw_over_price'].mean():.0f}  Avg under juice: {both_tot['raw_under_price'].mean():.0f}")

        print(f"\n  TOTALS BETTING (payout at RAW juice):")
        print(f"  {'Edge':<12} {'Bets':>5} {'W-L':>10} {'Win%':>6} {'P&L':>9} {'ROI':>7}")
        print(f"  {'-'*12} {'-'*5} {'-'*10} {'-'*6} {'-'*9} {'-'*7}")

        for thresh in [0.5, 1.0, 1.5, 2.0, 3.0]:
            bets = []
            for _, r in both_tot.iterrows():
                diff = r["model_total"] - r["mkt_total"]
                if abs(diff) < thresh: continue
                side = "over" if diff > 0 else "under"
                if r["actual_total"] == r["mkt_total"]: continue
                won = (r["actual_total"] > r["mkt_total"]) if side == "over" else (r["actual_total"] < r["mkt_total"])
                raw_price = r["raw_over_price"] if side == "over" else r["raw_under_price"]
                dec = american_to_decimal(raw_price)
                bets.append({"won": won, "pnl": (dec - 1) if won else -1, "side": side})
            if not bets: continue
            bdf = pd.DataFrame(bets)
            n = len(bdf); w = bdf["won"].sum(); pnl = bdf["pnl"].sum()
            print(f"  {thresh:>4.1f}+ runs  {n:>5} {f'{w}-{n-w}':>10} {w/n*100:>5.1f}% {pnl:>+9.2f} {pnl/n*100:>+6.1f}%")

        print(f"\n  OVER/UNDER SPLIT (1.0+ run edge):")
        for side in ["over", "under"]:
            bets = []
            for _, r in both_tot.iterrows():
                diff = r["model_total"] - r["mkt_total"]
                if abs(diff) < 1.0: continue
                if (side == "over" and diff <= 0) or (side == "under" and diff >= 0): continue
                if r["actual_total"] == r["mkt_total"]: continue
                won = (r["actual_total"] > r["mkt_total"]) if side == "over" else (r["actual_total"] < r["mkt_total"])
                raw_price = r["raw_over_price"] if side == "over" else r["raw_under_price"]
                dec = american_to_decimal(raw_price)
                bets.append({"won": won, "pnl": (dec - 1) if won else -1})
            if bets:
                bdf = pd.DataFrame(bets)
                n = len(bdf); w = bdf["won"].sum(); pnl = bdf["pnl"].sum()
                print(f"    {side.upper()}: {w}-{n-w} ({w/n*100:.1f}%), P&L {pnl:+.2f}, ROI {pnl/n*100:+.1f}%")

    if len(both_ml) > 0:
        print(f"\n  STATISTICAL NOTES:")
        print(f"  ML sample: {len(both_ml)} games (need ~1000 for 95% CI < ±3%)")
        se = np.sqrt(0.5 * 0.5 / len(both_ml))
        print(f"  Win rate SE: ±{se*100:.1f}% (95% CI = ±{1.96*se*100:.1f}%)")

    print(f"\n{'='*70}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest model vs market with raw odds payout.")
    parser.add_argument("--out", type=Path, default=Path("data/processed/backtest_market.csv"))
    args = parser.parse_args()
    df = run_backtest(out_csv=args.out)
    if len(df) > 0:
        print_report(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
