#!/usr/bin/env python3
"""Scrape Sidearm live stats pages to confirm starting pitchers.

Sidearm live stats pages are JS-rendered and only populated ~30 min before
game time. This script uses Playwright to load the page and extract the
starting pitcher from the lineup/boxscore.

Usage:
    # Scrape all games for a date (checks which games are ~30 min from start)
    .venv/bin/python3 scripts/scrape_sidearm_starters.py --date 2026-03-18

    # Scrape a specific team
    .venv/bin/python3 scripts/scrape_sidearm_starters.py --date 2026-03-18 --team BSB_TEXAS

    # Force scrape all games regardless of time window
    .venv/bin/python3 scripts/scrape_sidearm_starters.py --date 2026-03-18 --force

    # Run as daemon, checking every 10 minutes
    .venv/bin/python3 scripts/scrape_sidearm_starters.py --date 2026-03-18 --daemon --interval 600
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from sidearm_urls import SIDEARM_URLS
except ImportError:
    SIDEARM_URLS = {}


def load_schedule(date: str) -> pd.DataFrame:
    """Load daily schedule with game times."""
    sched_path = Path(f"data/daily/{date}/schedule.csv")
    if not sched_path.exists():
        print(f"No schedule at {sched_path}", file=sys.stderr)
        return pd.DataFrame()
    return pd.read_csv(sched_path, dtype=str)


def load_starters(date: str) -> pd.DataFrame:
    """Load current starters CSV."""
    path = Path(f"data/daily/{date}/starters.csv")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str)


def get_sidearm_live_url(canonical_id: str) -> str | None:
    """Get the Sidearm live stats base URL for a team."""
    domain = SIDEARM_URLS.get(canonical_id)
    if not domain:
        return None
    return f"https://{domain}/sports/baseball/schedule"


def games_in_window(schedule: pd.DataFrame, window_min: int = 45) -> list[dict]:
    """Find games starting within the next N minutes."""
    now = datetime.now(timezone.utc)
    games = []
    for _, row in schedule.iterrows():
        game_time_str = ""
        for col in ["start_time", "commence_time", "game_time"]:
            if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
                game_time_str = str(row[col]).strip()
                break
        if not game_time_str:
            continue
        try:
            gt = pd.to_datetime(game_time_str, utc=True)
        except Exception:
            continue

        # Include games from 45 min before to 30 min after start
        if now - timedelta(minutes=30) <= gt <= now + timedelta(minutes=window_min):
            games.append({
                "home_cid": str(row.get("home_cid", "")).strip(),
                "away_cid": str(row.get("away_cid", "")).strip(),
                "home_team": str(row.get("home_team", "")).strip(),
                "away_team": str(row.get("away_team", "")).strip(),
                "game_time": gt,
            })
    return games


async def scrape_starter_from_sidearm(page, url: str, team_name: str) -> dict | None:
    """Navigate to a Sidearm schedule page and try to find today's starter.

    Returns dict with {pitcher_name, throws} or None.
    """
    try:
        await page.goto(url, timeout=15000, wait_until="networkidle")
        await page.wait_for_timeout(3000)  # Extra wait for JS rendering

        # Strategy 1: Look for "live stats" or "box score" link for today's game
        # Sidearm schedule pages show game cards with links
        content = await page.content()

        # Strategy 2: Try the team's live stats page directly
        # Many Sidearm sites have /sports/baseball/stats/game/{id} pages
        # But we need the game ID first

        # Strategy 3: Look for pitcher info in the page content
        # Some schedule pages show "Probable Starter" or "Starting Pitcher"

        # Try to find any mention of "starter" or "probable"
        starter_patterns = [
            r'(?:probable|starting)\s+(?:pitcher|p)\s*[:\-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'(?:SP|RHP|LHP)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        ]

        for pat in starter_patterns:
            match = re.search(pat, content, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                throws = ""
                if "LHP" in content[max(0, match.start()-10):match.start()]:
                    throws = "L"
                elif "RHP" in content[max(0, match.start()-10):match.start()]:
                    throws = "R"
                return {"pitcher_name": name, "throws": throws}

        # Strategy 4: Find live stats links and follow them
        live_links = await page.query_selector_all('a[href*="livestats"], a[href*="sidearmstats"]')
        for link in live_links[:3]:
            href = await link.get_attribute("href")
            if href:
                try:
                    await page.goto(href if href.startswith("http") else url.rsplit("/", 3)[0] + href,
                                   timeout=15000, wait_until="networkidle")
                    await page.wait_for_timeout(2000)

                    # On live stats pages, look for pitching box
                    pitching_els = await page.query_selector_all('[class*="pitcher"], [class*="pitching"], [data-stat="pitching"]')
                    if pitching_els:
                        text = await pitching_els[0].inner_text()
                        # First pitcher listed is usually the starter
                        lines = text.strip().split("\n")
                        if lines:
                            name = lines[0].strip()
                            if name and len(name) > 2:
                                return {"pitcher_name": name, "throws": ""}
                except Exception:
                    continue

        return None
    except Exception as e:
        print(f"  Error scraping {url}: {e}", file=sys.stderr)
        return None


async def scrape_statbroadcast_boxscore(page, url: str) -> dict | None:
    """Try to scrape a StatBroadcast/Sidearm live stats boxscore page."""
    try:
        await page.goto(url, timeout=20000, wait_until="networkidle")
        await page.wait_for_timeout(3000)

        # StatBroadcast pages have structured pitcher data
        # Look for the pitching table
        content = await page.content()

        # Pattern: first pitcher in pitching stats table is usually starter
        # <td class="pitcher-name">LastName, FirstName</td>
        pitcher_pattern = r'class="[^"]*pitcher[^"]*"[^>]*>([^<]+)</'
        matches = re.findall(pitcher_pattern, content, re.IGNORECASE)
        if matches:
            name = matches[0].strip()
            # Convert "LastName, FirstName" to "FirstName LastName"
            if "," in name:
                parts = name.split(",", 1)
                name = f"{parts[1].strip()} {parts[0].strip()}"
            return {"pitcher_name": name, "throws": ""}

        return None
    except Exception as e:
        print(f"  Error on boxscore: {e}", file=sys.stderr)
        return None


def update_starters_csv(date: str, updates: dict[tuple[str, str], dict]) -> int:
    """Apply confirmed starter updates to starters.csv.

    updates: {(canonical_id, 'home'|'away'): {pitcher_name, throws}}
    Returns number of updates applied.
    """
    path = Path(f"data/daily/{date}/starters.csv")
    if not path.exists():
        return 0

    df = pd.read_csv(path, dtype=str)
    n_updated = 0

    for _, row in df.iterrows():
        idx = row.name
        for side, cid_col, sp_col, throws_col in [
            ("home", "home_cid", "home_starter", "hp_throws"),
            ("away", "away_cid", "away_starter", "ap_throws"),
        ]:
            cid = str(row.get(cid_col, "")).strip()
            key = (cid, side)
            if key in updates:
                upd = updates[key]
                old_sp = str(row.get(sp_col, "")).strip()
                new_sp = upd["pitcher_name"]
                if old_sp.lower() != new_sp.lower():
                    print(f"  UPDATE: {cid} {side} starter: {old_sp} → {new_sp}", file=sys.stderr)
                    df.at[idx, sp_col] = new_sp
                    if upd.get("throws"):
                        df.at[idx, throws_col] = upd["throws"]
                    n_updated += 1

    if n_updated > 0:
        # Backup original
        backup = path.with_suffix(".csv.bak")
        if not backup.exists():
            import shutil
            shutil.copy2(path, backup)
        df.to_csv(path, index=False)
        print(f"  Saved {n_updated} updates to {path}", file=sys.stderr)

    return n_updated


async def scrape_games(date: str, games: list[dict], headless: bool = True) -> dict:
    """Scrape starting pitchers for a list of games."""
    from playwright.async_api import async_playwright

    updates = {}

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        )
        page = await context.new_page()

        for game in games:
            h_cid = game["home_cid"]
            a_cid = game["away_cid"]

            # Try home team's sidearm page
            h_url = get_sidearm_live_url(h_cid)
            if h_url:
                print(f"  Checking {game['home_team']} schedule page...", file=sys.stderr)
                result = await scrape_starter_from_sidearm(page, h_url, game["home_team"])
                if result:
                    updates[(h_cid, "home")] = result
                    print(f"    Found: {result['pitcher_name']}", file=sys.stderr)

            # Brief delay to avoid hammering
            await page.wait_for_timeout(1000)

        await browser.close()

    return updates


def main() -> int:
    import asyncio

    parser = argparse.ArgumentParser(description="Scrape Sidearm for starting pitchers")
    parser.add_argument("--date", required=True, help="Game date YYYY-MM-DD")
    parser.add_argument("--team", help="Only scrape specific team (canonical_id)")
    parser.add_argument("--force", action="store_true", help="Scrape all games regardless of time window")
    parser.add_argument("--window", type=int, default=45, help="Minutes before game to start checking (default: 45)")
    parser.add_argument("--daemon", action="store_true", help="Run continuously, checking every --interval seconds")
    parser.add_argument("--interval", type=int, default=600, help="Daemon check interval in seconds (default: 600)")
    parser.add_argument("--headless", action="store_true", default=True, help="Run browser headless")
    parser.add_argument("--no-headless", dest="headless", action="store_false", help="Show browser window")
    args = parser.parse_args()

    schedule = load_schedule(args.date)
    if schedule.empty:
        print(f"No schedule for {args.date}", file=sys.stderr)
        return 1

    print(f"Schedule: {len(schedule)} games on {args.date}", file=sys.stderr)

    if args.daemon:
        print(f"Running as daemon, checking every {args.interval}s...", file=sys.stderr)
        while True:
            games = games_in_window(schedule, args.window)
            if games:
                print(f"\n[{datetime.now().strftime('%H:%M')}] {len(games)} games in window", file=sys.stderr)
                updates = asyncio.run(scrape_games(args.date, games, args.headless))
                if updates:
                    update_starters_csv(args.date, updates)
            else:
                print(f"[{datetime.now().strftime('%H:%M')}] No games in window", file=sys.stderr)
            time.sleep(args.interval)
    else:
        if args.force:
            games = []
            for _, row in schedule.iterrows():
                games.append({
                    "home_cid": str(row.get("home_cid", "")).strip(),
                    "away_cid": str(row.get("away_cid", "")).strip(),
                    "home_team": str(row.get("home_team", "")).strip(),
                    "away_team": str(row.get("away_team", "")).strip(),
                    "game_time": None,
                })
        else:
            games = games_in_window(schedule, args.window)

        if args.team:
            games = [g for g in games if g["home_cid"] == args.team or g["away_cid"] == args.team]

        if not games:
            print("No games in scrape window. Use --force to scrape all.", file=sys.stderr)
            return 0

        print(f"Scraping {len(games)} games...", file=sys.stderr)
        updates = asyncio.run(scrape_games(args.date, games, args.headless))

        if updates:
            n = update_starters_csv(args.date, updates)
            print(f"Applied {n} starter updates", file=sys.stderr)
        else:
            print("No starter updates found", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
