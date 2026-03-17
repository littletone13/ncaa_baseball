#!/usr/bin/env python3
"""Scrape D1Baseball leaderboard tables for one or more seasons.

Uses Playwright (headless Chromium) to authenticate and render JS-loaded stats.

Pulls batting/pitching Standard, Advanced, and Batted-Ball tables from:
  https://d1baseball.com/statistics/?season=YYYY&split=overall

Writes TSV files under data/raw/d1baseball/{season}/:
  batting_standard.tsv
  batting_advanced.tsv
  batting_batted_ball.tsv
  pitching_standard.tsv
  pitching_advanced.tsv
  pitching_batted_ball.tsv

Credentials: set D1B_USER and D1B_PASS in .env (or environment).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, Page


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


def _login(page: Page, user: str, password: str) -> None:
    """Authenticate with D1Baseball via WordPress login form."""
    page.goto("https://d1baseball.com/wp-login.php", wait_until="networkidle")
    page.fill("#user_login", user)
    page.fill("#user_pass", password)
    page.click("#wp-submit")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(3000)
    # Verify login via cookies (URL may stay at wp-login.php after redirect)
    cookie_names = {c["name"] for c in page.context.cookies()}
    logged_in = any(n.startswith("wordpress_logged_in") for n in cookie_names)
    if not logged_in:
        raise RuntimeError("D1Baseball login failed — check D1B_USER / D1B_PASS in .env")
    print("Authenticated with D1Baseball.")


def _extract_table_all_rows(page: Page, table_id: str) -> str:
    """Expand a DataTable to show all rows and return its outer HTML."""
    # Show all rows for this specific table
    page.evaluate(f"""
        () => {{
            var table = jQuery('#{table_id}');
            if (table.length && jQuery.fn.DataTable.isDataTable(table)) {{
                table.DataTable().page.len(-1).draw();
            }}
        }}
    """)
    page.wait_for_timeout(2000)
    el = page.query_selector(f"#{table_id}")
    if el is None:
        return ""
    return el.evaluate("el => el.outerHTML")


# Tab CSS selectors → which tables they reveal
TAB_GROUPS = [
    # (tab_selector, table_ids)
    (".stat-toggle[data-target='#batting-standard']", ["batting-stats"]),
    (".stat-toggle[data-target='#batting-advanced']", ["advanced-batting-stats"]),
    (".stat-toggle[data-target='#batting-batted-ball']", ["batted-ball-batting-stats"]),
    (".stat-toggle[data-target='#pitching-standard']", ["pitching-stats"]),
    (".stat-toggle[data-target='#pitching-advanced']", ["advanced-pitching-stats"]),
    (".stat-toggle[data-target='#pitching-batted-ball']", ["batted-ball-pitching-stats"]),
]


def scrape_season(season: int, out_root: Path, page: Page) -> dict[str, int]:
    url = f"{BASE_URL}?season={season}&split=overall"
    page.goto(url, wait_until="domcontentloaded", timeout=60000)

    # Wait for DataTables to initialize (tables are JS-rendered)
    page.wait_for_selector("table.dataTable", timeout=30000)
    page.wait_for_timeout(5000)

    season_dir = out_root / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}

    for tab_selector, table_ids in TAB_GROUPS:
        # Click the tab to make its tables visible
        tab = page.query_selector(tab_selector)
        if tab:
            tab.click()
            page.wait_for_timeout(1500)

        for table_id in table_ids:
            fname = TABLE_MAP.get(table_id)
            if fname is None:
                continue
            html = _extract_table_all_rows(page, table_id)
            if not html:
                counts[fname] = -1
                continue
            soup = BeautifulSoup(html, "html.parser")
            table = soup.find("table")
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
        default=3.0,
        help="Delay between season requests",
    )
    parser.add_argument(
        "--allow-masked",
        action="store_true",
        help="Allow paywall-masked placeholder values (not recommended).",
    )
    args = parser.parse_args()

    load_dotenv()
    d1b_user = os.environ.get("D1B_USER", "")
    d1b_pass = os.environ.get("D1B_PASS", "")

    args.out_root.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        if d1b_user and d1b_pass:
            _login(page, d1b_user, d1b_pass)
        else:
            print("WARNING: D1B_USER / D1B_PASS not set — scraping without auth (stats may be masked).")

        any_masked = False
        for i, season in enumerate(args.seasons):
            counts = scrape_season(season, args.out_root, page)
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

        browser.close()

    if any_masked and not args.allow_masked:
        print(
            "ERROR: Detected likely paywall-masked stats values. "
            "Re-run after authenticated export, or pass --allow-masked to keep placeholders."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
