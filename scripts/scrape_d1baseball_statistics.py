#!/usr/bin/env python3
"""Scrape D1Baseball leaderboard tables for one or more seasons.

Pulls batting/pitching Standard, Advanced, and Batted-Ball tables from:
  https://d1baseball.com/statistics/?season=YYYY&split=overall

Writes TSV files under data/raw/d1baseball/{season}/:
  batting_standard.tsv
  batting_advanced.tsv
  batting_batted_ball.tsv
  pitching_standard.tsv
  pitching_advanced.tsv
  pitching_batted_ball.tsv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://d1baseball.com/statistics/"
TABLE_MAP = {
    "batting-stats": "batting_standard.tsv",
    "advanced-batting-stats": "batting_advanced.tsv",
    "batted-ball-batting-stats": "batting_batted_ball.tsv",
    "pitching-stats": "pitching_standard.tsv",
    "advanced-pitching-stats": "pitching_advanced.tsv",
    "batted-ball-pitching-stats": "pitching_batted_ball.tsv",
}


MASKED_TOKENS = {".123", "1.23", "12.3", "12"}


def _parse_table(table) -> pd.DataFrame:
    thead = table.find("thead")
    tbody = table.find("tbody")
    if thead is None or tbody is None:
        return pd.DataFrame()

    headers = [th.get_text(" ", strip=True) for th in thead.find_all("th")]
    rows = []
    for tr in tbody.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if not cells:
            continue
        # align to header count
        if len(cells) < len(headers):
            cells = cells + ([""] * (len(headers) - len(cells)))
        elif len(cells) > len(headers):
            cells = cells[: len(headers)]
        rows.append(cells)

    if not rows:
        return pd.DataFrame(columns=headers)
    return pd.DataFrame(rows, columns=headers)


def _masked_token_ratio(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    total = 0
    masked = 0
    for col in df.columns:
        if col in {"Player", "Team", "Class", "Qual."}:
            continue
        vals = df[col].astype(str).str.strip()
        total += len(vals)
        masked += vals.isin(MASKED_TOKENS).sum()
    if total == 0:
        return 0.0
    return float(masked) / float(total)


def scrape_season(season: int, out_root: Path, session: requests.Session) -> dict[str, int]:
    url = f"{BASE_URL}?season={season}&split=overall"
    resp = session.get(url, timeout=45)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    season_dir = out_root / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for table_id, fname in TABLE_MAP.items():
        table = soup.find("table", id=table_id)
        if table is None:
            counts[fname] = -1
            continue
        df = _parse_table(table)
        out_path = season_dir / fname
        df.to_csv(out_path, sep="\t", index=False)
        counts[fname] = len(df)
        counts[f"{fname}__masked_ratio"] = int(round(100 * _masked_token_ratio(df)))
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape D1Baseball stats tables by season.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2026],
        help="Season years to scrape (e.g., 2024 2025 2026)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/raw/d1baseball"),
        help="Root output directory",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=2.0,
        help="Delay between season requests",
    )
    parser.add_argument(
        "--allow-masked",
        action="store_true",
        help="Allow paywall-masked placeholder values (not recommended).",
    )
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )

    args.out_root.mkdir(parents=True, exist_ok=True)
    any_masked = False
    for i, season in enumerate(args.seasons):
        counts = scrape_season(season, args.out_root, session)
        print(f"[{season}]")
        for fname in TABLE_MAP.values():
            n = counts.get(fname, -1)
            mr = counts.get(f"{fname}__masked_ratio", 0)
            if n >= 0:
                print(f"  {fname}: {n} rows (masked≈{mr}%)")
                if mr >= 30:
                    any_masked = True
            else:
                print(f"  {fname}: MISSING")
        if i < len(args.seasons) - 1 and args.sleep_sec > 0:
            time.sleep(args.sleep_sec)
    if any_masked and not args.allow_masked:
        print(
            "ERROR: Detected likely paywall-masked stats values. "
            "Re-run after authenticated export, or pass --allow-masked to keep placeholders."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
