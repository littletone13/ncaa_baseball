#!/usr/bin/env python3
"""
backtest_vs_market.py — Grade model predictions against market odds and results.

Design principles:
  - Edge calc: use pre-computed consensus_fair_home/away from pull_odds.py (already devigged)
  - Payout: use best available US book price (DK, FD, MGM, Caesars — where you'd actually bet)
  - Tests ALL markets: ML home, ML away, overs, unders, spreads
  - Prices validated: American odds must be <= -100 or >= 100
  - Default juice: -115 when no valid price available

Usage:
  python3 scripts/backtest_vs_market.py
  python3 scripts/backtest_vs_market.py --out data/processed/backtest_market.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

# US books we'd actually bet at (in preference order for best price)
US_BOOKS = {
    "draftkings", "fanduel", "betmgm", "caesars", "betrivers",
    "hard_rock_bet", "bet365", "bovada", "barstool", "pointsbet",
    "thescore_bet", "wynnbet", "betparx", "bally_bet",
}

DEFAULT_JUICE = -115  # standard NCAA baseball totals juice


def _is_valid_american(price: float) -> bool:
    """Check if a price is a valid American odds value."""
    return price <= -100 or price >= 100


def american_to_decimal(ml: float) -> float:
    if ml < 0:
        return 1 + 100 / abs(ml)
    elif ml > 0:
        return 1 + ml / 100
    return 2.0


def american_to_implied(ml: float) -> float:
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    elif ml > 0:
        return 100 / (ml + 100)
    return 0.5


def prob_to_american(p: float) -> int:
    if p >= 0.5:
        return -round(p / (1 - p) * 100)
    return round((1 - p) / p * 100)


def parse_odds_log(odds_log: Path, canonical_csv: Path) -> dict:
    """Parse odds pull log into clean per-game records.

    For each game, stores:
      - fair_home_prob: consensus devigged probability (from pull_odds.py)
      - best_home_ml: best available US book home ML (for payout)
      - best_away_ml: best available US book away ML (for payout)
      - mkt_total: consensus total line
      - best_over_price: best US book over price (for payout)
      - best_under_price: best US book under price (for payout)
      - best_home_spread: spread line
      - best_home_spread_price: spread price for home
    """
    canon = pd.read_csv(canonical_csv, dtype=str)
    odds_to_cid = {}
    for _, r in canon.iterrows():
        cid = str(r.get("canonical_id", "")).strip()
        for col in ("odds_api_name", "espn_name", "team_name"):
            n = str(r.get(col, "")).strip()
            if n and n != "nan":
                odds_to_cid[n] = cid

    game_odds = {}

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
            h_cid = odds_to_cid.get(home_name, "")
            a_cid = odds_to_cid.get(away_name, "")
            if not h_cid or not a_cid:
                continue

            game_key = (game_date, h_cid, a_cid)

            # Use pre-computed fair probabilities from pull_odds.py
            fair_home = row.get("consensus_fair_home")
            fair_away = row.get("consensus_fair_away")

            # Collect best prices from US books
            home_mls = []
            away_mls = []
            over_prices = []  # (line, price) tuples
            under_prices = []
            spread_prices = []  # (point, price) tuples
            total_lines = []

            for bk in row.get("bookmaker_lines", []):
                bk_key = bk.get("bookmaker_key", "").lower()
                is_us = bk_key in US_BOOKS

                for mkt in bk.get("markets", []):
                    outcomes = mkt.get("outcomes", [])

                    if mkt.get("key") == "h2h":
                        hml = aml = None
                        for o in outcomes:
                            p = o.get("price")
                            if p is None:
                                continue
                            p = float(p)
                            if not _is_valid_american(p):
                                continue
                            if o.get("name") == home_name:
                                hml = p
                            elif o.get("name") == away_name:
                                aml = p
                        if hml is not None and aml is not None:
                            home_mls.append((hml, is_us))
                            away_mls.append((aml, is_us))

                    elif mkt.get("key") == "totals":
                        for o in outcomes:
                            pt = o.get("point")
                            p = o.get("price")
                            if pt is None or p is None:
                                continue
                            p = float(p)
                            if not _is_valid_american(p):
                                continue
                            total_lines.append(float(pt))
                            if o.get("name") == "Over":
                                over_prices.append((float(pt), p, is_us))
                            elif o.get("name") == "Under":
                                under_prices.append((float(pt), p, is_us))

                    elif mkt.get("key") == "spreads":
                        for o in outcomes:
                            pt = o.get("point")
                            p = o.get("price")
                            if pt is None or p is None:
                                continue
                            p = float(p)
                            if not _is_valid_american(p):
                                continue
                            if o.get("name") == home_name:
                                spread_prices.append((float(pt), p, is_us))

            rec = {}

            # Fair probability for edge calc
            if fair_home is not None and fair_away is not None:
                rec["fair_home_prob"] = float(fair_home)
            elif home_mls:
                # Fallback: compute from raw MLs
                h_impl = [american_to_implied(ml) for ml, _ in home_mls]
                a_impl = [american_to_implied(ml) for ml, _ in away_mls]
                total = np.mean(h_impl) + np.mean(a_impl)
                rec["fair_home_prob"] = np.mean(h_impl) / total if total > 0 else 0.5

            # Best ML prices (prefer US books, then best price overall)
            if home_mls:
                us_home = [ml for ml, us in home_mls if us]
                us_away = [ml for ml, us in away_mls if us]
                all_home = [ml for ml, _ in home_mls]
                all_away = [ml for ml, _ in away_mls]
                # For home favorite (negative ML): least negative = best price
                # For home dog (positive ML): most positive = best price
                # Use median of US books if available, else median of all
                rec["best_home_ml"] = float(np.median(us_home)) if us_home else float(np.median(all_home))
                rec["best_away_ml"] = float(np.median(us_away)) if us_away else float(np.median(all_away))
                rec["n_books_ml"] = len(home_mls)

            # Totals
            if total_lines:
                rec["mkt_total"] = float(np.median(total_lines))
                rec["n_books_total"] = len(set(total_lines))

            # Best over/under prices at consensus line
            if over_prices:
                consensus_line = rec.get("mkt_total")
                if consensus_line:
                    # Filter to prices at the consensus line
                    at_line = [(p, us) for pt, p, us in over_prices if pt == consensus_line]
                    if not at_line:
                        # Use all prices
                        at_line = [(p, us) for _, p, us in over_prices]
                    us_over = [p for p, us in at_line if us]
                    all_over = [p for p, _ in at_line]
                    rec["best_over_price"] = float(np.median(us_over)) if us_over else float(np.median(all_over))
                else:
                    all_p = [p for _, p, _ in over_prices]
                    rec["best_over_price"] = float(np.median(all_p))

            if under_prices:
                consensus_line = rec.get("mkt_total")
                if consensus_line:
                    at_line = [(p, us) for pt, p, us in under_prices if pt == consensus_line]
                    if not at_line:
                        at_line = [(p, us) for _, p, us in under_prices]
                    us_under = [p for p, us in at_line if us]
                    all_under = [p for p, _ in at_line]
                    rec["best_under_price"] = float(np.median(us_under)) if us_under else float(np.median(all_under))
                else:
                    all_p = [p for _, p, _ in under_prices]
                    rec["best_under_price"] = float(np.median(all_p))

            # Spreads
            if spread_prices:
                us_spreads = [(pt, p) for pt, p, us in spread_prices if us]
                all_spreads = [(pt, p) for pt, p, _ in spread_prices]
                src = us_spreads if us_spreads else all_spreads
                # Use the most common spread line
                from collections import Counter
                line_counts = Counter(pt for pt, _ in src)
                consensus_spread = line_counts.most_common(1)[0][0]
                at_line = [p for pt, p in src if pt == consensus_spread]
                rec["mkt_spread"] = consensus_spread
                rec["best_spread_price"] = float(np.median(at_line))

            if rec:
                # Merge: update only fields present in this snapshot (don't drop earlier valid data)
                existing = game_odds.get(game_key, {})
                existing.update(rec)
                game_odds[game_key] = existing

    # Post-process: validate all stored prices, replace invalid with default
    for key, rec in game_odds.items():
        for field in ("best_over_price", "best_under_price", "best_spread_price"):
            val = rec.get(field)
            if val is not None and not _is_valid_american(val):
                rec[field] = DEFAULT_JUICE

    return game_odds


def load_predictions(pred_dir: Path) -> dict:
    preds = {}
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
            h = str(row.get("home_cid", "")).strip()
            a = str(row.get("away_cid", "")).strip()
            if not h or not a:
                continue
            try:
                preds[(date, h, a)] = {
                    "model_home_prob": float(row["home_win_prob"]),
                    "model_total": float(row["exp_total"]) if pd.notna(row.get("exp_total")) else None,
                    "home": str(row.get("home", "")),
                    "away": str(row.get("away", "")),
                }
            except (ValueError, TypeError):
                continue
    return preds


def load_results(games_csv: Path) -> dict:
    games = pd.read_csv(games_csv, dtype=str)
    games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
    games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")
    games = games.dropna(subset=["home_score", "away_score"])
    results = {}
    for _, r in games.iterrows():
        d = str(r.get("game_date", ""))
        if not d.startswith("2026"):
            continue
        h = str(r.get("home_canonical_id", "")).strip()
        a = str(r.get("away_canonical_id", "")).strip()
        key = (d, h, a)
        if key not in results:
            hs, as_ = float(r["home_score"]), float(r["away_score"])
            results[key] = {
                "home_score": hs, "away_score": as_,
                "actual_total": hs + as_,
                "home_win": int(hs > as_),
                "margin": hs - as_,
            }
    return results


def grade_bet(won: bool, price: float) -> float:
    """P&L for a $1 flat bet at the given American odds price."""
    dec = american_to_decimal(price)
    return (dec - 1) if won else -1.0


def run_backtest(
    odds_log: Path = Path("data/raw/odds/odds_pull_log.jsonl"),
    canonical_csv: Path = Path("data/registries/canonical_teams_2026.csv"),
    games_csv: Path = Path("data/processed/games.csv"),
    pred_dir: Path = Path("data/processed"),
    out_csv: Path | None = None,
):
    print("Loading odds...", file=sys.stderr)
    odds = parse_odds_log(odds_log, canonical_csv)
    print(f"  {len(odds)} games with odds", file=sys.stderr)

    print("Loading predictions...", file=sys.stderr)
    preds = load_predictions(pred_dir)
    print(f"  {len(preds)} predictions", file=sys.stderr)

    print("Loading results...", file=sys.stderr)
    results = load_results(games_csv)
    print(f"  {len(results)} completed games", file=sys.stderr)

    # Match all three
    rows = []
    for key in preds:
        if key not in odds or key not in results:
            continue
        p, o, r = preds[key], odds[key], results[key]
        row = {
            "date": key[0], "home_cid": key[1], "away_cid": key[2],
            "home": p.get("home", ""), "away": p.get("away", ""),
            "model_home_prob": p["model_home_prob"],
            "model_total": p.get("model_total"),
            "home_win": r["home_win"], "actual_total": r["actual_total"],
            "margin": r["margin"],
        }
        row.update(o)
        rows.append(row)

    df = pd.DataFrame(rows)
    n = len(df)
    print(f"\n  MATCHED: {n} games", file=sys.stderr)

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    if n == 0:
        return df

    # ── Report ──
    print(f"\n{'='*70}")
    print(f"  BACKTEST — {n} games | {df['date'].min()} to {df['date'].max()}")
    print(f"{'='*70}")

    # ML section
    ml = df[df["fair_home_prob"].notna() & df["best_home_ml"].notna()].copy()
    if len(ml) > 0:
        ml["edge_home"] = ml["model_home_prob"] - ml["fair_home_prob"]

        print(f"\n  MONEYLINES ({len(ml)} games, payout at best US book price)")
        print(f"  {'Edge':<8} {'Bets':>5} {'W-L':>10} {'Win%':>6} {'P&L':>9} {'ROI':>7}")
        print(f"  {'-'*8} {'-'*5} {'-'*10} {'-'*6} {'-'*9} {'-'*7}")

        for thresh in [0.01, 0.03, 0.05, 0.08, 0.10, 0.15]:
            bets = []
            for _, r in ml.iterrows():
                if r["edge_home"] >= thresh:
                    bets.append(grade_bet(r["home_win"] == 1, r["best_home_ml"]))
                elif -r["edge_home"] >= thresh:
                    bets.append(grade_bet(r["home_win"] == 0, r["best_away_ml"]))
            if not bets:
                continue
            w = sum(1 for b in bets if b > 0)
            pnl = sum(bets)
            print(f"  {thresh*100:>5.0f}%+  {len(bets):>5} {f'{w}-{len(bets)-w}':>10} "
                  f"{w/len(bets)*100:>5.1f}% {pnl:>+9.2f} {pnl/len(bets)*100:>+6.1f}%")

        # Side split
        print(f"\n  SIDE SPLIT (3%+ edge):")
        for side, label in [("home", "HOME"), ("away", "AWAY")]:
            bets = []
            for _, r in ml.iterrows():
                e = r["edge_home"] if side == "home" else -r["edge_home"]
                if e < 0.03:
                    continue
                price = r["best_home_ml"] if side == "home" else r["best_away_ml"]
                won = (r["home_win"] == 1) if side == "home" else (r["home_win"] == 0)
                bets.append(grade_bet(won, price))
            if bets:
                w = sum(1 for b in bets if b > 0)
                pnl = sum(bets)
                print(f"    {label}: {w}-{len(bets)-w} ({w/len(bets)*100:.1f}%), "
                      f"P&L {pnl:+.2f}, ROI {pnl/len(bets)*100:+.1f}%")

    # Totals section — test BOTH overs and unders
    tot = df[df["model_total"].notna() & df["mkt_total"].notna()].copy()
    if len(tot) > 0:
        tot["model_total"] = pd.to_numeric(tot["model_total"])
        tot["mkt_total"] = pd.to_numeric(tot["mkt_total"])
        tot["total_edge"] = tot["model_total"] - tot["mkt_total"]

        # Fill missing prices with default
        tot["best_over_price"] = pd.to_numeric(tot.get("best_over_price", DEFAULT_JUICE)).fillna(DEFAULT_JUICE)
        tot["best_under_price"] = pd.to_numeric(tot.get("best_under_price", DEFAULT_JUICE)).fillna(DEFAULT_JUICE)

        print(f"\n  TOTALS ({len(tot)} games)")
        print(f"  Avg over price: {tot['best_over_price'].median():.0f}  "
              f"Avg under price: {tot['best_under_price'].median():.0f}")

        print(f"\n  {'Edge':<12} {'Bets':>5} {'W-L':>10} {'Win%':>6} {'P&L':>9} {'ROI':>7}")
        print(f"  {'-'*12} {'-'*5} {'-'*10} {'-'*6} {'-'*9} {'-'*7}")

        for thresh in [0.5, 1.0, 1.5, 2.0, 3.0]:
            bets = []
            for _, r in tot.iterrows():
                diff = r["total_edge"]
                if abs(diff) < thresh:
                    continue
                if r["actual_total"] == r["mkt_total"]:
                    continue  # push

                if diff > 0:  # over
                    won = r["actual_total"] > r["mkt_total"]
                    price = r["best_over_price"]
                else:  # under
                    won = r["actual_total"] < r["mkt_total"]
                    price = r["best_under_price"]

                bets.append(grade_bet(won, price))

            if not bets:
                continue
            w = sum(1 for b in bets if b > 0)
            pnl = sum(bets)
            print(f"  {thresh:>4.1f}+ runs {len(bets):>5} {f'{w}-{len(bets)-w}':>10} "
                  f"{w/len(bets)*100:>5.1f}% {pnl:>+9.2f} {pnl/len(bets)*100:>+6.1f}%")

        # Over/under split
        print(f"\n  OVER/UNDER SPLIT (1.0+ run edge):")
        for side, label in [("over", "OVER"), ("under", "UNDER")]:
            bets = []
            for _, r in tot.iterrows():
                diff = r["total_edge"]
                if side == "over" and diff < 1.0:
                    continue
                if side == "under" and diff > -1.0:
                    continue
                if r["actual_total"] == r["mkt_total"]:
                    continue
                if side == "over":
                    won = r["actual_total"] > r["mkt_total"]
                    price = r["best_over_price"]
                else:
                    won = r["actual_total"] < r["mkt_total"]
                    price = r["best_under_price"]
                bets.append(grade_bet(won, price))
            if bets:
                w = sum(1 for b in bets if b > 0)
                pnl = sum(bets)
                print(f"    {label}: {w}-{len(bets)-w} ({w/len(bets)*100:.1f}%), "
                      f"P&L {pnl:+.2f}, ROI {pnl/len(bets)*100:+.1f}%")

    # Spreads section
    sprd = df[df["mkt_spread"].notna() & df["best_spread_price"].notna()].copy()
    if len(sprd) > 0:
        sprd["mkt_spread"] = pd.to_numeric(sprd["mkt_spread"])
        sprd["best_spread_price"] = pd.to_numeric(sprd["best_spread_price"])
        sprd["model_margin"] = sprd["model_home_prob"].apply(
            lambda p: 0  # placeholder — need model margin, not just win prob
        )

        print(f"\n  SPREADS ({len(sprd)} games with spread lines)")
        # Grade: if model expects home to cover, bet home spread
        bets_cover = []
        for _, r in sprd.iterrows():
            spread = r["mkt_spread"]  # e.g., -1.5 means home favored by 1.5
            covered = r["margin"] > -spread if spread < 0 else r["margin"] > spread
            # Only bet if model disagrees with spread direction
            model_margin = (r["model_home_prob"] - 0.5) * 10  # rough margin estimate
            if model_margin > -spread + 1:  # model says home covers by 1+ run
                bets_cover.append(grade_bet(covered, r["best_spread_price"]))
        if bets_cover:
            w = sum(1 for b in bets_cover if b > 0)
            pnl = sum(bets_cover)
            print(f"    Home cover (model favors): {w}-{len(bets_cover)-w}, "
                  f"P&L {pnl:+.2f}, ROI {pnl/len(bets_cover)*100:+.1f}%")

    # Stats
    print(f"\n  STATISTICAL NOTES:")
    if len(ml) > 0:
        se = np.sqrt(0.25 / len(ml))
        print(f"  ML: {len(ml)} games, SE ±{se*100:.1f}%, 95% CI ±{1.96*se*100:.1f}%")
    if len(tot) > 0:
        se = np.sqrt(0.25 / len(tot))
        print(f"  Totals: {len(tot)} games, SE ±{se*100:.1f}%, 95% CI ±{1.96*se*100:.1f}%")
    print(f"  Need ~1000+ bets for meaningful significance at ±3% CI")

    print(f"\n{'='*70}")
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/processed/backtest_market.csv"))
    args = parser.parse_args()
    run_backtest(out_csv=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
