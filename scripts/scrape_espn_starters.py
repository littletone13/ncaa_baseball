#!/usr/bin/env python3
"""
scrape_espn_starters.py — Backfill starting pitcher IDs for ESPN games.

ESPN score-only games (no PBP data) don't include starter info in our JSONL.
This script fetches the ESPN summary API for each game to get starter pitcher
IDs and names, then outputs a CSV mapping event_id → pitcher IDs.

Usage:
  python3 scripts/scrape_espn_starters.py
  python3 scripts/scrape_espn_starters.py --seasons 2024,2025 --resume
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
BASE = "https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball"


def fetch_json(url: str, retries: int = 3, delay: float = 0.3) -> dict | None:
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": UA})
            with urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
            elif attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    return None


def extract_starters(summary: dict) -> dict:
    """Extract starting pitchers from ESPN summary response."""
    result = {
        "home_pitcher_id": "",
        "home_pitcher_name": "",
        "away_pitcher_id": "",
        "away_pitcher_name": "",
    }

    boxscore = summary.get("boxscore", {})
    players = boxscore.get("players", [])

    for p in players:
        home_away = p.get("homeAway", "")
        prefix = "home" if home_away == "home" else "away"

        for stat_group in p.get("statistics", []):
            if stat_group.get("type") == "pitching":
                for athlete in stat_group.get("athletes", []):
                    if athlete.get("starter"):
                        ath = athlete.get("athlete", {})
                        espn_id = str(ath.get("id", ""))
                        name = ath.get("displayName", "")
                        if espn_id:
                            result[f"{prefix}_pitcher_id"] = f"ESPN_{espn_id}"
                            result[f"{prefix}_pitcher_name"] = name

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape starting pitcher IDs from ESPN summary API."
    )
    parser.add_argument("--espn-dir", type=Path, default=Path("data/raw/espn"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/processed/espn_starter_backfill.csv"))
    parser.add_argument("--seasons", type=str, default="2024,2025,2026")
    parser.add_argument("--resume", action="store_true",
                        help="Skip games already in output file")
    parser.add_argument("--delay", type=float, default=0.25,
                        help="Delay between API calls")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max games to scrape (0=unlimited)")
    args = parser.parse_args()

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]

    # Collect ESPN games without starters
    games_to_scrape: list[dict] = []
    for season in seasons:
        path = args.espn_dir / f"games_{season}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    g = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                # Skip PBP games (already have starters)
                re = g.get("run_events")
                if re and isinstance(re, dict) and re.get("home") and re.get("away"):
                    continue
                # Skip games that already have starters
                starters = g.get("starters") or {}
                hp = starters.get("home_pitcher") or {}
                if hp.get("espn_id") or hp.get("id"):
                    continue

                event_id = g.get("event_id") or g.get("id") or ""
                if event_id:
                    games_to_scrape.append({
                        "event_id": str(event_id),
                        "season": season,
                        "date": str(g.get("date", ""))[:10],
                    })

    print(f"Found {len(games_to_scrape)} ESPN games needing starter backfill")

    # Load already-scraped for --resume
    scraped_ids: set[str] = set()
    if args.resume and args.out.exists():
        with args.out.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scraped_ids.add(str(row.get("event_id", "")))
        print(f"Resume mode: {len(scraped_ids)} already scraped")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_scraped = 0
    n_found = 0
    n_failed = 0

    mode = "a" if args.resume and args.out.exists() else "w"
    with args.out.open(mode, newline="", encoding="utf-8") as outf:
        writer = csv.DictWriter(outf, fieldnames=[
            "event_id", "season", "date",
            "home_pitcher_id", "home_pitcher_name",
            "away_pitcher_id", "away_pitcher_name",
        ])
        if mode == "w":
            writer.writeheader()

        for i, game in enumerate(games_to_scrape):
            eid = game["event_id"]
            if eid in scraped_ids:
                continue
            if args.limit and n_scraped >= args.limit:
                break

            if n_scraped > 0:
                time.sleep(args.delay)

            url = f"{BASE}/summary?event={eid}"
            data = fetch_json(url)

            if data is None:
                n_failed += 1
                if n_scraped % 500 == 0 and n_scraped > 0:
                    print(f"  [{n_scraped}] {eid}: API failed", file=sys.stderr)
                n_scraped += 1
                continue

            starters = extract_starters(data)
            if starters["home_pitcher_id"] or starters["away_pitcher_id"]:
                n_found += 1

            writer.writerow({
                "event_id": eid,
                "season": game["season"],
                "date": game["date"],
                **starters,
            })
            n_scraped += 1

            if n_scraped % 200 == 0:
                print(f"  [{n_scraped}/{len(games_to_scrape)}] found={n_found}, "
                      f"failed={n_failed}", file=sys.stderr)

    print(f"\n=== ESPN Starter Backfill ===")
    print(f"Scraped: {n_scraped}")
    print(f"With starters: {n_found}")
    print(f"Failed: {n_failed}")
    print(f"Output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
