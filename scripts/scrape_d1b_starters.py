#!/usr/bin/env python3
"""
scrape_d1b_starters.py — Scrape probable starters from D1Baseball rotation pages.

Uses Playwright with a persistent browser profile (your logged-in Chrome session)
to bypass Cloudflare and access subscriber content.

D1Baseball rotation pages show projected weekly starters:
  https://d1baseball.com/team/{slug}/rotation/

Usage:
  # First time: login to D1Baseball in the browser that opens, then close it
  python3 scripts/scrape_d1b_starters.py --login

  # Scrape starters for today's games
  python3 scripts/scrape_d1b_starters.py --date 2026-03-17

  # Scrape all teams
  python3 scripts/scrape_d1b_starters.py --date 2026-03-17 --all-teams

Output:
  data/daily/{date}/d1b_starters.csv
  Columns: canonical_id, slug, game_day, pitcher_name, source
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd


BROWSER_DATA_DIR = Path.home() / ".cache" / "ncaa_baseball_browser"


def get_d1b_slug(canonical_id: str, crosswalk_csv: Path) -> str | None:
    """Look up D1Baseball slug from canonical_id via crosswalk."""
    xw = pd.read_csv(crosswalk_csv, dtype=str)
    match = xw[xw["canonical_id"] == canonical_id]
    if match.empty:
        return None
    d1b_name = str(match.iloc[0].get("d1baseball_name", "")).strip()
    if not d1b_name:
        return None
    # Convert "Texas Tech" → "texas-tech"
    slug = re.sub(r"[^a-z0-9]+", "-", d1b_name.lower()).strip("-")
    return slug


def launch_login_browser():
    """Open browser for user to login to D1Baseball."""
    from playwright.sync_api import sync_playwright

    BROWSER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(BROWSER_DATA_DIR),
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = context.new_page()
        page.goto("https://d1baseball.com/login/")
        print("Browser opened. Please log in to D1Baseball, then close the browser.")
        print("Your session will be saved for future scraping.")
        try:
            page.wait_for_event("close", timeout=300000)  # 5 min
        except:
            pass
        context.close()


def scrape_rotation_page(slug: str, page) -> list[dict]:
    """
    Scrape a team's rotation page for projected starters.

    Returns list of dicts with:
        {game_day, pitcher_name, role}
    """
    url = f"https://d1baseball.com/team/{slug}/rotation/"
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=20000)
        time.sleep(2)

        # Wait for content to render
        page.wait_for_selector("table", timeout=10000)
    except Exception as e:
        print(f"    Failed to load {url}: {e}", file=sys.stderr)
        return []

    starters = []
    body = page.inner_text("body")

    # D1Baseball rotation pages typically show:
    # "Weekend Rotation" or "Projected Starters"
    # with Friday/Saturday/Sunday columns
    # and pitcher names with ERA/IP stats

    # Parse the table content
    tables = page.query_selector_all("table")
    for table in tables:
        text = table.inner_text()
        if any(kw in text.lower() for kw in ["rotation", "starter", "friday", "weekend"]):
            rows = table.query_selector_all("tr")
            for row in rows:
                cells = row.query_selector_all("td, th")
                cell_texts = [c.inner_text().strip() for c in cells]
                if len(cell_texts) >= 2:
                    # Look for pitcher names (typically first column)
                    name = cell_texts[0]
                    if name and not name.startswith("#") and len(name) > 2:
                        # Try to detect day (Friday, Saturday, Sunday, midweek)
                        day = ""
                        for ct in cell_texts:
                            if any(d in ct.lower() for d in ["fri", "sat", "sun", "tue", "wed", "thu", "mon"]):
                                day = ct
                                break
                        starters.append({
                            "pitcher_name": name,
                            "game_day": day,
                            "role": "starter",
                        })

    # Fallback: look for pitcher names in any structured format
    if not starters:
        # Try parsing text content for common rotation patterns
        lines = body.split("\n")
        for line in lines:
            line = line.strip()
            # Match patterns like "Friday: John Smith (3-1, 2.45)" or "Game 1: John Smith"
            match = re.match(
                r"(?:Game\s*\d|Friday|Saturday|Sunday|Midweek|Tuesday|Wednesday|Monday|Thursday)"
                r"[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)",
                line,
            )
            if match:
                starters.append({
                    "pitcher_name": match.group(1),
                    "game_day": line.split(":")[0].strip() if ":" in line else "",
                    "role": "starter",
                })

    return starters


def scrape_starters_for_date(
    date: str,
    schedule_csv: Path,
    crosswalk_csv: Path,
    out_csv: Path,
) -> pd.DataFrame:
    """Scrape D1B rotation pages for all teams playing on given date."""
    from playwright.sync_api import sync_playwright

    schedule = pd.read_csv(schedule_csv, dtype=str)

    # Collect all team canonical_ids playing today
    team_cids = set()
    for _, r in schedule.iterrows():
        team_cids.add(str(r.get("home_cid", "")).strip())
        team_cids.add(str(r.get("away_cid", "")).strip())
    team_cids.discard("")

    # Map to D1B slugs
    xw = pd.read_csv(crosswalk_csv, dtype=str)
    cid_to_slug = {}
    for _, r in xw.iterrows():
        cid = str(r.get("canonical_id", "")).strip()
        d1b_name = str(r.get("d1baseball_name", "")).strip()
        if cid and d1b_name:
            slug = re.sub(r"[^a-z0-9]+", "-", d1b_name.lower()).strip("-")
            # Handle unicode apostrophes
            slug = slug.replace("\u2019", "").replace("'", "")
            cid_to_slug[cid] = slug

    teams_to_scrape = [(cid, cid_to_slug[cid]) for cid in team_cids if cid in cid_to_slug]
    print(f"  {len(teams_to_scrape)} teams to scrape rotation pages", file=sys.stderr)

    if not teams_to_scrape:
        return pd.DataFrame()

    BROWSER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_starters = []
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(BROWSER_DATA_DIR),
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = context.new_page()

        for i, (cid, slug) in enumerate(teams_to_scrape):
            print(f"  [{i+1}/{len(teams_to_scrape)}] {cid} → {slug}", file=sys.stderr)
            starters = scrape_rotation_page(slug, page)
            for s in starters:
                s["canonical_id"] = cid
                s["slug"] = slug
                s["source"] = "d1baseball_rotation"
            all_starters.extend(starters)

            # Rate limit to avoid blocking
            time.sleep(2)

        context.close()

    df = pd.DataFrame(all_starters)
    if not df.empty:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\n  Wrote {len(df)} starters → {out_csv}", file=sys.stderr)

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape D1Baseball rotation pages.")
    parser.add_argument("--date", type=str, help="Game date YYYY-MM-DD")
    parser.add_argument("--login", action="store_true",
                        help="Open browser for D1Baseball login (first time setup)")
    parser.add_argument("--crosswalk", type=Path,
                        default=Path("data/registries/d1baseball_crosswalk.csv"))
    parser.add_argument("--all-teams", action="store_true",
                        help="Scrape all 308 teams (not just today's games)")
    args = parser.parse_args()

    if args.login:
        launch_login_browser()
        return 0

    if not args.date:
        print("--date required (or use --login for first-time setup)", file=sys.stderr)
        return 1

    daily_dir = Path(f"data/daily/{args.date}")
    schedule_csv = daily_dir / "schedule.csv"
    if not schedule_csv.exists():
        print(f"  No schedule at {schedule_csv} — run predict_day.py first", file=sys.stderr)
        return 1

    out_csv = daily_dir / "d1b_starters.csv"
    scrape_starters_for_date(
        date=args.date,
        schedule_csv=schedule_csv,
        crosswalk_csv=args.crosswalk,
        out_csv=out_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
