#!/usr/bin/env python3
"""
pull_direct_odds.py — Pull odds directly from DK, FanDuel, BetMGM, BetRivers.

Runs each book's scraper in dry-run mode, parses the output, and merges
into our odds pipeline format (odds_latest.jsonl / odds_pull_log.jsonl).

This supplements the-odds-api with direct book prices, especially for
games/totals that the API doesn't cover.

Usage:
  python3 scripts/pull_direct_odds.py                    # pull all books
  python3 scripts/pull_direct_odds.py --books dk,mgm     # specific books
  python3 scripts/pull_direct_odds.py --merge             # merge into odds_latest.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401


SCRAPERS = {
    "dk": "scripts/draftkings_scraper.py",
    "mgm": "scripts/betmgm_scraper.py",
    "fd": "scripts/fanduel_scraper.py",
    "br": "scripts/betrivers_scraper.py",
}

BOOK_NAMES = {
    "dk": "DraftKings",
    "mgm": "BetMGM",
    "fd": "FanDuel",
    "br": "BetRivers",
}


def run_scraper(book_key: str) -> list[dict]:
    """Run a book scraper and capture its parsed output."""
    script = SCRAPERS.get(book_key)
    if not script or not Path(script).exists():
        print(f"  {book_key}: scraper not found at {script}", file=sys.stderr)
        return []

    try:
        result = subprocess.run(
            [sys.executable, script, "--sport", "baseball"],
            capture_output=True, text=True, timeout=30,
        )
        # Parse the stdout for odds data
        # The scrapers print structured output we can parse
        lines = result.stdout.split("\n") + result.stderr.split("\n")

        # Look for odds rows in the output
        odds = []
        for line in lines:
            # BetMGM format: "  UCF at West Virginia (id=...): 6 odds from grid"
            # The actual odds are printed in a table format
            # We need to capture the structured data differently
            pass

        return odds
    except Exception as e:
        print(f"  {book_key}: error — {e}", file=sys.stderr)
        return []


def pull_all_direct(books: list[str] | None = None) -> dict:
    """Pull from all direct book APIs and return combined game data.

    Returns dict keyed by (home_team, away_team) with prices from each book.
    """
    if books is None:
        books = list(SCRAPERS.keys())

    # Import the scrapers directly instead of subprocess
    all_odds = {}

    for book in books:
        book_name = BOOK_NAMES.get(book, book)
        print(f"Pulling {book_name}...", file=sys.stderr)

        try:
            if book == "mgm":
                from betmgm_scraper import scrape_betmgm_baseball
                rows = scrape_betmgm_baseball()
            elif book == "dk":
                from draftkings_scraper import scrape_dk_baseball
                rows = scrape_dk_baseball()
            elif book == "br":
                from betrivers_scraper import scrape_betrivers_baseball
                rows = scrape_betrivers_baseball()
            elif book == "fd":
                from fanduel_scraper import scrape_fanduel_baseball
                rows = scrape_fanduel_baseball()
            else:
                rows = []

            print(f"  {book_name}: {len(rows)} odds rows", file=sys.stderr)

            for row in rows:
                game_key = (row.get("home_team", ""), row.get("away_team", ""))
                if game_key not in all_odds:
                    all_odds[game_key] = {
                        "home_team": row.get("home_team", ""),
                        "away_team": row.get("away_team", ""),
                        "commence_time": row.get("commence_time", ""),
                        "books": {},
                    }

                book_data = all_odds[game_key]["books"].setdefault(book_name, {})
                market = row.get("market", "")
                if market == "moneyline" or market == "h2h":
                    book_data["home_ml"] = row.get("home_price")
                    book_data["away_ml"] = row.get("away_price")
                elif market == "total" or market == "totals":
                    book_data["total_line"] = row.get("line")
                    book_data["over_price"] = row.get("home_price")  # over
                    book_data["under_price"] = row.get("away_price")  # under
                elif market == "spread" or market == "spreads":
                    book_data["spread_line"] = row.get("line")
                    book_data["spread_home_price"] = row.get("home_price")
                    book_data["spread_away_price"] = row.get("away_price")

        except Exception as e:
            print(f"  {book_name}: import/scrape error — {e}", file=sys.stderr)

    return all_odds


def merge_to_odds_log(all_odds: dict, odds_log: Path) -> int:
    """Merge direct book odds into the odds pull log JSONL."""
    n_written = 0
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(odds_log, "a") as f:
        for (home, away), data in all_odds.items():
            record = {
                "home_team": home,
                "away_team": away,
                "commence_time": data.get("commence_time", ""),
                "bookmaker_lines": [],
                "source": "direct_scraper",
                "fetched_at": timestamp,
            }

            for book_name, prices in data.get("books", {}).items():
                markets = []
                if "home_ml" in prices:
                    markets.append({
                        "key": "h2h",
                        "outcomes": [
                            {"name": home, "price": prices["home_ml"]},
                            {"name": away, "price": prices["away_ml"]},
                        ],
                    })
                if "total_line" in prices:
                    markets.append({
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": prices["total_line"], "price": prices.get("over_price", -115)},
                            {"name": "Under", "point": prices["total_line"], "price": prices.get("under_price", -115)},
                        ],
                    })
                if "spread_line" in prices:
                    markets.append({
                        "key": "spreads",
                        "outcomes": [
                            {"name": home, "point": prices["spread_line"], "price": prices.get("spread_home_price", -115)},
                            {"name": away, "point": -prices["spread_line"], "price": prices.get("spread_away_price", -115)},
                        ],
                    })

                if markets:
                    record["bookmaker_lines"].append({
                        "bookmaker_key": book_name.lower().replace(" ", ""),
                        "bookmaker_title": book_name,
                        "markets": markets,
                    })

            if record["bookmaker_lines"]:
                f.write(json.dumps(record) + "\n")
                n_written += 1

    return n_written


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull direct book odds (DK, FD, MGM, BR)")
    parser.add_argument("--books", help="Comma-separated book keys: dk,mgm,fd,br")
    parser.add_argument("--merge", action="store_true", help="Merge into odds_pull_log.jsonl")
    args = parser.parse_args()

    books = args.books.split(",") if args.books else None
    all_odds = pull_all_direct(books)

    n_games = len(all_odds)
    n_books = sum(len(d.get("books", {})) for d in all_odds.values())
    print(f"\nTotal: {n_games} games, {n_books} book entries", file=sys.stderr)

    if args.merge and all_odds:
        odds_log = Path("data/raw/odds/odds_pull_log.jsonl")
        n = merge_to_odds_log(all_odds, odds_log)
        print(f"Merged {n} records → {odds_log}", file=sys.stderr)

    # Print summary
    for (home, away), data in sorted(all_odds.items()):
        books_str = ", ".join(data.get("books", {}).keys())
        print(f"  {away} @ {home}: {books_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
