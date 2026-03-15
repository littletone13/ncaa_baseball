#!/usr/bin/env python3
"""Export a multi-page NCAA baseball betting sheet PDF."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def prob_to_american(p: float) -> int:
    p = max(1e-6, min(1 - 1e-6, float(p)))
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


def american_to_prob(price: int | float) -> float:
    x = float(price)
    if x > 0:
        return 100.0 / (x + 100.0)
    return abs(x) / (abs(x) + 100.0)


def devig_two_way_prob(price_a: int | float, price_b: int | float) -> tuple[float, float]:
    """Multiplicative devig for a two-way market."""
    pa = american_to_prob(price_a)
    pb = american_to_prob(price_b)
    tot = pa + pb
    if tot <= 0:
        return 0.5, 0.5
    return pa / tot, pb / tot


def fmt_american(val: Any) -> str:
    try:
        n = int(float(val))
    except (TypeError, ValueError):
        return ""
    return f"{n:+d}"


def poisson_over_prob(mean_runs: float, total_line: float) -> float:
    """Approximate P(total runs > total_line) using Poisson(mean_runs)."""
    lam = max(0.01, float(mean_runs))
    k = int(math.floor(float(total_line)))
    # CDF up to k via stable recursion.
    p0 = math.exp(-lam)
    cdf = p0
    term = p0
    for i in range(1, k + 1):
        term = term * lam / i
        cdf += term
    return max(0.0, min(1.0, 1.0 - cdf))


def wind_label(wind_mph: Any, wind_out_mph: Any) -> str:
    """Human-readable wind speed + direction class (in/out/cross)."""
    try:
        ws = float(wind_mph)
        wo = float(wind_out_mph)
    except (TypeError, ValueError):
        return ""
    if ws < 1.0:
        cls = "calm"
    elif abs(wo) <= max(1.5, 0.20 * ws):
        cls = "crosswind"
    elif wo > 0:
        cls = "out"
    else:
        cls = "in"
    return f"{ws:.0f} mph {cls}"


def model_runline_prob(row: dict[str, Any], side: str, point: float) -> float | None:
    """
    Return model cover probability for a runline side/point.

    Supports half-point runlines using simulated margin tails:
      home_win_by_Kplus / away_win_by_Kplus where K = abs(point) + 0.5
    Falls back to legacy home_rl_cover / away_rl_cover for 1.5 when needed.
    """
    try:
        p = float(point)
    except (TypeError, ValueError):
        return None
    # Standard baseball spread points are half-runs (e.g., 1.5, 2.5, 3.5).
    if abs((abs(p) % 1.0) - 0.5) > 1e-9:
        return None
    step = int(round(abs(p) + 0.5))

    try:
        p_home_m15 = float(row.get("home_rl_cover", 0.0))
        p_away_m15 = float(row.get("away_rl_cover", 0.0))
    except (TypeError, ValueError):
        return None

    def _tail_prob(which: str, k: int) -> float | None:
        key = f"{which}_win_by_{k}plus"
        val = row.get(key)
        try:
            if val is not None and str(val) != "":
                return float(val)
        except (TypeError, ValueError):
            pass
        if k == 2:
            return p_home_m15 if which == "home" else p_away_m15
        return None

    side = str(side).lower()
    if side == "home":
        p_home_k = _tail_prob("home", step)
        p_away_k = _tail_prob("away", step)
        if p < 0:
            return p_home_k
        if p_away_k is None:
            return None
        return 1.0 - p_away_k
    if side == "away":
        p_home_k = _tail_prob("home", step)
        p_away_k = _tail_prob("away", step)
        if p < 0:
            return p_away_k
        if p_home_k is None:
            return None
        return 1.0 - p_home_k
    return None


def _book_abbrev(book_key: str) -> str:
    m = {
        "draftkings": "DK",
        "bovada": "BO",
        "fanduel": "FD",
        "betmgm": "MGM",
        "espnbet": "ESPN",
        "betonlineag": "BOL",
        "mybookieag": "MYB",
        "lowvig": "LV",
        "hardrockbet": "HR",
        "fliff": "FL",
    }
    return m.get(book_key, book_key[:4].upper())


def _best_price(candidates: list[tuple[int, str]]) -> tuple[int | None, str]:
    if not candidates:
        return None, ""
    price, book = max(candidates, key=lambda x: x[0])
    return int(price), book


def load_odds_by_cid_pair(odds_jsonl: Path, canonical_csv: Path) -> dict[tuple[str, str], dict[str, Any]]:
    canon = pd.read_csv(canonical_csv, dtype=str).fillna("")
    name_to_cid: dict[str, str] = {}
    for _, r in canon.iterrows():
        cid = str(r.get("canonical_id", "")).strip()
        for col in ("odds_api_name", "espn_name", "team_name"):
            nm = str(r.get(col, "")).strip().lower()
            if cid and nm:
                name_to_cid[nm] = cid

    out: dict[tuple[str, str], dict[str, Any]] = {}
    if not odds_jsonl.exists():
        return out

    with open(odds_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            h_nm = str(rec.get("home_team", "")).strip().lower()
            a_nm = str(rec.get("away_team", "")).strip().lower()
            h_cid = name_to_cid.get(h_nm, "")
            a_cid = name_to_cid.get(a_nm, "")
            if not h_cid or not a_cid:
                continue
            out[(h_cid, a_cid)] = rec
    return out


def build_team_idx_map(team_table_csv: Path) -> dict[str, int]:
    if not team_table_csv.exists():
        return {}
    df = pd.read_csv(team_table_csv, dtype=str).fillna("")
    out: dict[str, int] = {}
    for _, r in df.iterrows():
        cid = str(r.get("canonical_id", "")).strip()
        if not cid:
            continue
        try:
            out[cid] = int(float(r.get("team_idx", 0)))
        except (TypeError, ValueError):
            out[cid] = 0
    return out


def build_team_posterior_pitcher_count(pitcher_table_csv: Path) -> dict[str, int]:
    if not pitcher_table_csv.exists():
        return {}
    df = pd.read_csv(pitcher_table_csv, dtype=str).fillna("")
    df["pitcher_idx_num"] = pd.to_numeric(df.get("pitcher_idx", 0), errors="coerce").fillna(0).astype(int)
    grp = (
        df.groupby("team_canonical_id")["pitcher_idx_num"]
        .apply(lambda s: int((s > 0).sum()))
        .to_dict()
    )
    return {str(k): int(v) for k, v in grp.items() if str(k).strip()}


def compute_tier(
    home_cid: str,
    away_cid: str,
    home_starter_idx: Any,
    away_starter_idx: Any,
    team_idx_map: dict[str, int],
    team_post_pitchers: dict[str, int],
) -> str:
    try:
        hs = int(float(home_starter_idx))
    except (TypeError, ValueError):
        hs = 0
    try:
        aas = int(float(away_starter_idx))
    except (TypeError, ValueError):
        aas = 0

    h_in = team_idx_map.get(str(home_cid).strip(), 0) > 0
    a_in = team_idx_map.get(str(away_cid).strip(), 0) > 0
    h_depth = team_post_pitchers.get(str(home_cid).strip(), 0)
    a_depth = team_post_pitchers.get(str(away_cid).strip(), 0)

    if h_in and a_in and hs > 0 and aas > 0 and h_depth >= 3 and a_depth >= 3:
        return "A"
    if (h_in and a_in) or (hs > 0 or aas > 0):
        return "B"
    return "C"


def _extract_best_market(rec: dict[str, Any], home_name: str, away_name: str) -> dict[str, Any]:
    best_away_ml: list[tuple[int, str]] = []
    best_home_ml: list[tuple[int, str]] = []
    totals_candidates: dict[float, dict[str, list[tuple[int, str]]]] = {}
    spread_candidates: dict[float, dict[str, list[tuple[int, str, float]]]] = {}
    spread_best_by_side_point: dict[tuple[str, float], tuple[int, str]] = {}
    spread_fair_by_side_point: dict[tuple[str, float], list[float]] = {}

    def _best_price_with_point(candidates: list[tuple[int, str, float]]) -> tuple[int | None, str, float | None]:
        if not candidates:
            return None, "", None
        price, book, point = max(candidates, key=lambda x: x[0])
        return int(price), book, float(point)

    for bm in rec.get("bookmaker_lines", []):
        bkey = _book_abbrev(str(bm.get("bookmaker_key", "")))
        for m in bm.get("markets", []):
            key = m.get("key")
            outcomes = m.get("outcomes") or []
            if key == "h2h":
                for o in outcomes:
                    name = str(o.get("name", ""))
                    price = o.get("price")
                    if price is None:
                        continue
                    if name == away_name:
                        best_away_ml.append((int(price), bkey))
                    elif name == home_name:
                        best_home_ml.append((int(price), bkey))
            elif key == "totals":
                for o in outcomes:
                    point = o.get("point")
                    price = o.get("price")
                    name = str(o.get("name", ""))
                    if point is None or price is None or name not in ("Over", "Under"):
                        continue
                    p = float(point)
                    totals_candidates.setdefault(p, {"Over": [], "Under": []})[name].append((int(price), bkey))
            elif key == "spreads":
                away_book_points: dict[float, int] = {}
                home_book_points: dict[float, int] = {}
                for o in outcomes:
                    point = o.get("point")
                    price = o.get("price")
                    name = str(o.get("name", ""))
                    if point is None or price is None:
                        continue
                    p = float(point)
                    spread_candidates.setdefault(abs(p), {"away": [], "home": []})
                    if name == away_name:
                        spread_candidates[abs(p)]["away"].append((int(price), bkey, p))
                        prev = away_book_points.get(p)
                        if prev is None or int(price) > prev:
                            away_book_points[p] = int(price)
                        key_sp = ("away", float(p))
                        prev_best = spread_best_by_side_point.get(key_sp)
                        if prev_best is None or int(price) > prev_best[0]:
                            spread_best_by_side_point[key_sp] = (int(price), bkey)
                    elif name == home_name:
                        spread_candidates[abs(p)]["home"].append((int(price), bkey, p))
                        prev = home_book_points.get(p)
                        if prev is None or int(price) > prev:
                            home_book_points[p] = int(price)
                        key_sp = ("home", float(p))
                        prev_best = spread_best_by_side_point.get(key_sp)
                        if prev_best is None or int(price) > prev_best[0]:
                            spread_best_by_side_point[key_sp] = (int(price), bkey)
                # Devig spreads from same-book opposite points.
                for ap, apx in away_book_points.items():
                    hp = -float(ap)
                    hpx = home_book_points.get(hp)
                    if hpx is None:
                        continue
                    fair_away, fair_home = devig_two_way_prob(apx, hpx)
                    spread_fair_by_side_point.setdefault(("away", float(ap)), []).append(fair_away)
                    spread_fair_by_side_point.setdefault(("home", float(hp)), []).append(fair_home)

    away_ml, away_ml_bk = _best_price(best_away_ml)
    home_ml, home_ml_bk = _best_price(best_home_ml)

    total_line = None
    over_px = over_bk = under_px = under_bk = ""
    if totals_candidates:
        best_point = max(
            totals_candidates.items(),
            key=lambda kv: len(kv[1]["Over"]) + len(kv[1]["Under"]),
        )[0]
        total_line = best_point
        opx, obk = _best_price(totals_candidates[best_point]["Over"])
        upx, ubk = _best_price(totals_candidates[best_point]["Under"])
        over_px, over_bk = fmt_american(opx), obk
        under_px, under_bk = fmt_american(upx), ubk

    spread_line = None
    away_sp_px = away_sp_bk = home_sp_px = home_sp_bk = ""
    away_sp_point = None
    home_sp_point = None
    if spread_candidates:
        best_abs = min(spread_candidates.keys())
        spread_line = best_abs
        apx, abk, apt = _best_price_with_point(spread_candidates[best_abs]["away"])
        hpx, hbk, hpt = _best_price_with_point(spread_candidates[best_abs]["home"])
        away_sp_px, away_sp_bk = fmt_american(apx), abk
        home_sp_px, home_sp_bk = fmt_american(hpx), hbk
        away_sp_point = apt
        home_sp_point = hpt
        # Guard against malformed feed rows; points should be opposite.
        if away_sp_point is not None and home_sp_point is not None:
            if abs(away_sp_point + home_sp_point) > 1e-6:
                away_sp_px = away_sp_bk = ""
                home_sp_px = home_sp_bk = ""
                away_sp_point = None
                home_sp_point = None

    spread_ladder: list[dict[str, Any]] = []
    for (side, point), (price, book) in spread_best_by_side_point.items():
        fairs = spread_fair_by_side_point.get((side, float(point)), [])
        fair_prob = float(sum(fairs) / len(fairs)) if fairs else None
        spread_ladder.append(
            {
                "side": side,
                "point": float(point),
                "price": int(price),
                "book": book,
                "fair_prob": fair_prob,
            }
        )
    spread_ladder = sorted(
        spread_ladder,
        key=lambda x: (abs(float(x["point"])), 0 if x["side"] == "away" else 1),
    )

    return {
        "away_ml": fmt_american(away_ml),
        "away_ml_bk": away_ml_bk,
        "home_ml": fmt_american(home_ml),
        "home_ml_bk": home_ml_bk,
        "total_line": total_line,
        "over_px": over_px,
        "over_bk": over_bk,
        "under_px": under_px,
        "under_bk": under_bk,
        "spread_line": spread_line,
        "away_sp_point": away_sp_point,
        "home_sp_point": home_sp_point,
        "away_sp_px": away_sp_px,
        "away_sp_bk": away_sp_bk,
        "home_sp_px": home_sp_px,
        "home_sp_bk": home_sp_bk,
        "spread_ladder": spread_ladder,
    }


def _table(rows: list[list[str]], col_widths: list[float]) -> Table:
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8.0),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 7.3),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("ALIGN", (0, 1), (2, -1), "LEFT"),
                ("ALIGN", (3, 1), (-1, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def build_pdf(
    predictions_csv: Path,
    odds_jsonl: Path,
    canonical_csv: Path,
    team_table_csv: Path,
    pitcher_table_csv: Path,
    out_pdf: Path,
    date_label: str,
    sims_label: str = "5,000",
) -> None:
    df = pd.read_csv(predictions_csv).sort_values("game_num")
    df["fair_rl_home_m15"] = df["home_rl_cover"].apply(prob_to_american)
    df["fair_rl_away_m15"] = df["away_rl_cover"].apply(prob_to_american)
    team_idx_map = build_team_idx_map(team_table_csv)
    team_post_pitchers = build_team_posterior_pitcher_count(pitcher_table_csv)
    df["data_tier"] = df.apply(
        lambda r: compute_tier(
            home_cid=r.get("home_cid", ""),
            away_cid=r.get("away_cid", ""),
            home_starter_idx=r.get("home_starter_idx", 0),
            away_starter_idx=r.get("away_starter_idx", 0),
            team_idx_map=team_idx_map,
            team_post_pitchers=team_post_pitchers,
        ),
        axis=1,
    )
    tier_rank = {"A": 0, "B": 1, "C": 2}
    df["tier_rank"] = df["data_tier"].map(tier_rank).fillna(2).astype(int)
    df = df.sort_values(["tier_rank", "game_num"]).reset_index(drop=True)

    odds_by_pair = load_odds_by_cid_pair(odds_jsonl, canonical_csv)

    market_rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        key = (str(r.get("home_cid", "")).strip(), str(r.get("away_cid", "")).strip())
        rec = odds_by_pair.get(key)
        if not rec:
            continue
        # Use odds event team names for market outcome matching.
        m = _extract_best_market(
            rec,
            str(rec.get("home_team", "")),
            str(rec.get("away_team", "")),
        )
        market_rows.append(
            {
                "game": f"{r['away']} @ {r['home']}",
                "away": str(r["away"]),
                "home": str(r["home"]),
                "data_tier": str(r.get("data_tier", "C")),
                "exp_total_model": float(r.get("exp_total", 0.0)),
                "wind_mph": r.get("wind_mph"),
                "wind_out_mph": r.get("wind_out_mph"),
                "away_ml_model": int(r["ml_away"]),
                "home_ml_model": int(r["ml_home"]),
                "away_prob_model": float(r["away_win_prob"]),
                "home_prob_model": float(r["home_win_prob"]),
                "fragility_score": float(r.get("fragility_score", 0.0) or 0.0),
                "fragility_flag": str(r.get("fragility_flag", "low")),
                "total_band": float(r.get("exp_total_p90", 0.0) or 0.0) - float(r.get("exp_total_p10", 0.0) or 0.0),
                **m,
            }
        )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=landscape(letter),
        leftMargin=0.35 * inch,
        rightMargin=0.35 * inch,
        topMargin=0.35 * inch,
        bottomMargin=0.35 * inch,
    )
    styles = getSampleStyleSheet()
    story = []

    dt = datetime.strptime(date_label, "%Y-%m-%d")
    date_human = dt.strftime("%A, %B %d, %Y")
    story.append(Paragraph("<b>NCAA Baseball Betting Sheet</b>", styles["Title"]))
    story.append(Paragraph(f"{date_human} | {len(df)} Games | {sims_label} Sim Monte Carlo", styles["Normal"]))
    story.append(Spacer(1, 0.08 * inch))

    # Page 1: simulation fair values
    story.append(Paragraph("<b>Simulation Fair Values</b>", styles["Heading3"]))
    story.append(Paragraph("Model-generated fair odds from simulation output (no vig).", styles["Normal"]))
    story.append(Spacer(1, 0.12 * inch))
    fair = [[
        "Tier",
        "Game",
        "Away SP",
        "Home SP",
        "Away ML",
        "Home ML",
        "Total",
        "Rain%",
        "Wind",
        "Away -1.5",
        "Home -1.5",
    ]]
    for _, r in df.iterrows():
        rain_val = r.get("rain_chance_pct")
        rain_txt = ""
        if pd.notna(rain_val):
            rain_txt = f"{float(rain_val):.0f}"
        wind_txt = wind_label(r.get("wind_mph"), r.get("wind_out_mph"))
        fair.append(
            [
                str(r.get("data_tier", "C")),
                f"{r['away']} @ {r['home']}",
                str(r.get("away_starter", "")),
                str(r.get("home_starter", "")),
                fmt_american(r.get("ml_away")),
                fmt_american(r.get("ml_home")),
                f"{float(r.get('exp_total', 0.0)):.1f}",
                rain_txt,
                wind_txt,
                fmt_american(r.get("fair_rl_away_m15")),
                fmt_american(r.get("fair_rl_home_m15")),
            ]
        )
    story.append(
        _table(
            fair,
            [0.35 * inch, 2.2 * inch, 1.4 * inch, 1.4 * inch, 0.76 * inch, 0.76 * inch, 0.58 * inch, 0.46 * inch, 0.58 * inch, 0.78 * inch, 0.78 * inch],
        )
    )

    # Page 2: market prices
    story.append(PageBreak())
    story.append(Paragraph("<b>Best Market Odds - Moneyline and Totals</b>", styles["Heading3"]))
    story.append(
        Paragraph(
            "Best available price across books from latest odds pull.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.12 * inch))
    market_tbl = [["Tier", "Game", "Away ML", "Bk", "Home ML", "Bk", "Line", "Over", "Bk", "Under", "Bk"]]
    for r in market_rows:
        line = "" if r["total_line"] is None else f"{float(r['total_line']):.1f}"
        market_tbl.append(
            [
                r.get("data_tier", "C"),
                r["game"],
                r["away_ml"],
                r["away_ml_bk"],
                r["home_ml"],
                r["home_ml_bk"],
                line,
                r["over_px"],
                r["over_bk"],
                r["under_px"],
                r["under_bk"],
            ]
        )
    if len(market_tbl) == 1:
        market_tbl.append(["(no matched odds rows)", "", "", "", "", "", "", "", "", ""])
    story.append(
        _table(
            market_tbl,
            [0.35 * inch, 2.6 * inch, 0.75 * inch, 0.45 * inch, 0.75 * inch, 0.45 * inch, 0.5 * inch, 0.75 * inch, 0.45 * inch, 0.75 * inch, 0.45 * inch],
        )
    )

    # Page 3: spread/runline + ML edges
    story.append(PageBreak())
    story.append(Paragraph("<b>Best Market Runlines</b>", styles["Heading3"]))
    run_tbl = [["Tier", "Game", "Away RL", "Odds", "Bk", "Home RL", "Odds", "Bk"]]
    for r in market_rows:
        if r["spread_line"] is None:
            continue
        away_rl = ""
        home_rl = ""
        if r.get("away_sp_point") is not None:
            away_rl = f"{float(r['away_sp_point']):+.1f}"
        if r.get("home_sp_point") is not None:
            home_rl = f"{float(r['home_sp_point']):+.1f}"
        run_tbl.append(
            [
                r.get("data_tier", "C"),
                r["game"],
                away_rl,
                r["away_sp_px"],
                r["away_sp_bk"],
                home_rl,
                r["home_sp_px"],
                r["home_sp_bk"],
            ]
        )
    if len(run_tbl) == 1:
        run_tbl.append(["", "(no matched spread lines)", "", "", "", "", "", ""])
    story.append(_table(run_tbl, [0.35 * inch, 2.75 * inch, 0.6 * inch, 0.75 * inch, 0.45 * inch, 0.6 * inch, 0.75 * inch, 0.45 * inch]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>Top Edges - Sim vs Market (Moneyline)</b>", styles["Heading3"]))
    edge_rows = [["Tier", "Game", "Bet", "Market", "Bk", "Sim%", "Fair", "Edge%", "Adj%"]]
    edges: list[tuple[float, list[str]]] = []
    for r in market_rows:
        if r["away_ml"]:
            away_px = int(r["away_ml"])
            away_edge = (r["away_prob_model"] - american_to_prob(away_px)) * 100
            edge_adj = away_edge * max(0.0, 1.0 - float(r.get("fragility_score", 0.0)))
            edges.append(
                (
                    edge_adj,
                    [
                        r.get("data_tier", "C"),
                        r["game"],
                        f"{r['away']} ML",
                        fmt_american(away_px),
                        r["away_ml_bk"],
                        f"{r['away_prob_model']*100:.1f}%",
                        fmt_american(r["away_ml_model"]),
                        f"{away_edge:+.1f}%",
                        f"{edge_adj:+.1f}%",
                    ],
                )
            )
        if r["home_ml"]:
            home_px = int(r["home_ml"])
            home_edge = (r["home_prob_model"] - american_to_prob(home_px)) * 100
            edge_adj = home_edge * max(0.0, 1.0 - float(r.get("fragility_score", 0.0)))
            edges.append(
                (
                    edge_adj,
                    [
                        r.get("data_tier", "C"),
                        r["game"],
                        f"{r['home']} ML",
                        fmt_american(home_px),
                        r["home_ml_bk"],
                        f"{r['home_prob_model']*100:.1f}%",
                        fmt_american(r["home_ml_model"]),
                        f"{home_edge:+.1f}%",
                        f"{edge_adj:+.1f}%",
                    ],
                )
            )
    for _, row in sorted(edges, key=lambda x: x[0], reverse=True)[:30]:
        edge_rows.append(row)
    if len(edge_rows) == 1:
        edge_rows.append(["", "(no matched ML rows)", "", "", "", "", "", ""])
    story.append(_table(edge_rows, [0.35 * inch, 2.5 * inch, 0.95 * inch, 0.7 * inch, 0.4 * inch, 0.62 * inch, 0.62 * inch, 0.62 * inch, 0.62 * inch]))
    story.append(Spacer(1, 0.16 * inch))

    story.append(Paragraph("<b>Top Edges - Sim vs Market (Totals)</b>", styles["Heading3"]))
    tot_rows = [["Tier", "Game", "Wind", "Bet", "Market", "Bk", "Sim%", "Fair", "Edge%", "Adj%"]]
    tot_edges: list[tuple[float, list[str]]] = []
    for r in market_rows:
        line = r.get("total_line")
        if line is None:
            continue
        model_over = poisson_over_prob(r.get("exp_total_model", 0.0), float(line))
        model_under = 1.0 - model_over
        over_px = r.get("over_px", "")
        under_px = r.get("under_px", "")
        if over_px:
            ov = int(over_px)
            edge = (model_over - american_to_prob(ov)) * 100.0
            edge_adj = edge * max(0.0, 1.0 - float(r.get("fragility_score", 0.0)))
            tot_edges.append(
                (
                    edge_adj,
                    [
                        r.get("data_tier", "C"),
                        r["game"],
                        wind_label(r.get("wind_mph"), r.get("wind_out_mph")),
                        f"Over {float(line):.1f}",
                        fmt_american(ov),
                        r.get("over_bk", ""),
                        f"{model_over*100:.1f}%",
                        fmt_american(prob_to_american(model_over)),
                        f"{edge:+.1f}%",
                        f"{edge_adj:+.1f}%",
                    ],
                )
            )
        if under_px:
            un = int(under_px)
            edge = (model_under - american_to_prob(un)) * 100.0
            edge_adj = edge * max(0.0, 1.0 - float(r.get("fragility_score", 0.0)))
            tot_edges.append(
                (
                    edge_adj,
                    [
                        r.get("data_tier", "C"),
                        r["game"],
                        wind_label(r.get("wind_mph"), r.get("wind_out_mph")),
                        f"Under {float(line):.1f}",
                        fmt_american(un),
                        r.get("under_bk", ""),
                        f"{model_under*100:.1f}%",
                        fmt_american(prob_to_american(model_under)),
                        f"{edge:+.1f}%",
                        f"{edge_adj:+.1f}%",
                    ],
                )
            )
    for _, row in sorted(tot_edges, key=lambda x: x[0], reverse=True)[:30]:
        tot_rows.append(row)
    if len(tot_rows) == 1:
        tot_rows.append(["", "(no matched totals rows)", "", "", "", "", "", "", ""])
    story.append(_table(tot_rows, [0.32 * inch, 2.1 * inch, 0.78 * inch, 0.78 * inch, 0.62 * inch, 0.4 * inch, 0.56 * inch, 0.56 * inch, 0.56 * inch, 0.56 * inch]))
    story.append(Spacer(1, 0.16 * inch))

    story.append(Paragraph("<b>Top Edges - Sim vs Market (Runlines)</b>", styles["Heading3"]))
    rl_rows = [["Tier", "Game", "Bet", "Market", "Bk", "Sim%", "Fair", "Edge%", "Adj%"]]
    rl_edges: list[tuple[float, list[str]]] = []
    for r in market_rows:
        for sp in r.get("spread_ladder", []):
            side = str(sp.get("side", "")).lower()
            point = sp.get("point")
            price = sp.get("price")
            if point is None or price is None or side not in ("away", "home"):
                continue
            p_cov = model_runline_prob(r, side, float(point))
            if p_cov is None:
                continue
            fair_prob = sp.get("fair_prob")
            if fair_prob is None:
                fair_prob = american_to_prob(int(price))
            edge = (p_cov - float(fair_prob)) * 100.0
            team = r["away"] if side == "away" else r["home"]
            edge_adj = edge * max(0.0, 1.0 - float(r.get("fragility_score", 0.0)))
            rl_edges.append(
                (
                    edge_adj,
                    [
                        r.get("data_tier", "C"),
                        r["game"],
                        f"{team} {float(point):+.1f}",
                        fmt_american(int(price)),
                        str(sp.get("book", "")),
                        f"{p_cov*100:.1f}%",
                        fmt_american(prob_to_american(p_cov)),
                        f"{edge:+.1f}%",
                        f"{edge_adj:+.1f}%",
                    ],
                )
            )

    for _, row in sorted(rl_edges, key=lambda x: x[0], reverse=True)[:30]:
        rl_rows.append(row)
    if len(rl_rows) == 1:
        rl_rows.append(["", "(no priced runlines with model support)", "", "", "", "", "", ""])
    story.append(_table(rl_rows, [0.32 * inch, 2.3 * inch, 0.95 * inch, 0.68 * inch, 0.4 * inch, 0.58 * inch, 0.58 * inch, 0.58 * inch, 0.58 * inch]))

    doc.build(story)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export NCAA baseball betting sheet PDF.")
    parser.add_argument("--date", required=True, help="Date label (YYYY-MM-DD)")
    parser.add_argument("--predictions", type=Path, default=None, help="Predictions CSV path")
    parser.add_argument("--odds-jsonl", type=Path, default=Path("data/raw/odds/odds_latest.jsonl"))
    parser.add_argument("--canonical", type=Path, default=Path("data/registries/canonical_teams_2026.csv"))
    parser.add_argument("--team-table", type=Path, default=Path("data/processed/team_table.csv"))
    parser.add_argument("--pitcher-table", type=Path, default=Path("data/processed/pitcher_table.csv"))
    parser.add_argument("--out", type=Path, default=None, help="Output PDF path")
    parser.add_argument("--sims-label", type=str, default="5,000", help="Simulation count label for header")
    args = parser.parse_args()

    predictions_csv = args.predictions or Path(f"data/processed/predictions_{args.date}.csv")
    out_pdf = args.out or Path(f"output/pdf/ncaa_baseball_betting_sheet_{args.date}.pdf")
    if not predictions_csv.exists():
        raise SystemExit(f"Predictions file not found: {predictions_csv}")

    build_pdf(
        predictions_csv=predictions_csv,
        odds_jsonl=args.odds_jsonl,
        canonical_csv=args.canonical,
        team_table_csv=args.team_table,
        pitcher_table_csv=args.pitcher_table,
        out_pdf=out_pdf,
        date_label=args.date,
        sims_label=args.sims_label,
    )
    print(f"Wrote PDF: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
