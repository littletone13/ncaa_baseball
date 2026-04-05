#!/usr/bin/env python3
"""
BetRivers/Kambi Direct API Scraper — NCAA Baseball, Hockey, Basketball, Lacrosse

Uses the public Kambi offering API (CDN-hosted, no auth required).
BetRivers is powered by Kambi — same odds feed as Unibet, 888sport, etc.

Usage:
    python adapters/betrivers_scraper.py                       # dry run
    python adapters/betrivers_scraper.py --push                 # push to Supabase
    python adapters/betrivers_scraper.py --sport hockey         # single sport

Key endpoints:
    /offering/v2018/rsiusil/listView/{sport}/{league}/all/all.json  → event list
    /offering/v2018/rsiusil/betoffer/event/{eventId}.json           → all odds for event

Kambi odds format: odds field is decimal × 1000 (e.g., 2500 = 2.50)
Also provides: oddsAmerican ("+150"), oddsFractional ("6/4")
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone, date
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────
# Kambi API is public — no auth needed
BASE_URL = "https://eu1.offering-api.kambicdn.com/offering/v2018/rsiusil"

COMMON_PARAMS = {
    "lang": "en_US",
    "market": "US-IL",
    "client_id": "200",
    "channel_id": "3",
    "includeParticipants": "true",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X)",
    "Accept": "application/json",
}

# ── Sport / League paths ──────────────────────────────────────────
SPORTS = {
    "baseball": {
        "sport_path": "baseball",
        "league_path": "ncaa",
        "sport_key": "ncaa_baseball",
        "kambi_sport": "BASEBALL",
    },
    "hockey": {
        "sport_path": "ice_hockey",
        "league_path": "ncaa",
        "sport_key": "ncaa_hockey",
        "kambi_sport": "ICE_HOCKEY",
    },
    "basketball": {
        "sport_path": "basketball",
        "league_path": "ncaab",
        "sport_key": "ncaa_basketball",
        "kambi_sport": "BASKETBALL",
    },
    "lacrosse": {
        "sport_path": "lacrosse",
        "league_path": "ncaa",
        "sport_key": "ncaa_lacrosse",
        "kambi_sport": "LACROSSE",
    },
}


def _fetch(path: str, params: dict = None) -> dict | None:
    """Make Kambi API request."""
    url = f"{BASE_URL}{path}"
    p = {**COMMON_PARAMS, **(params or {})}
    try:
        r = requests.get(url, params=p, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"  WARN: {r.status_code} for {path}")
            return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def list_events(sport_path: str, league_path: str) -> list:
    """Get all events for a sport/league."""
    data = _fetch(
        f"/listView/{sport_path}/{league_path}/all/all.json",
        {"useCombined": "true"},
    )
    if not data:
        return []
    return data.get("events", [])


def get_event_odds(event_id: str) -> dict | None:
    """Get all betoffers (markets) for a single event."""
    return _fetch(f"/betoffer/event/{event_id}.json")


def parse_event_odds(event_data: dict) -> list:
    """Parse Kambi betoffer response into odds rows."""
    events = event_data.get("events", [])
    offers = event_data.get("betOffers", [])

    if not events:
        return []

    ev = events[0]
    event_name = ev.get("name", "")
    event_id = ev.get("id")
    start = ev.get("start", "")
    sport = ev.get("sport", "")
    group = ev.get("group", "")

    # Parse teams from name — Kambi uses "Away @ Home"
    home, away = None, None
    if " @ " in event_name:
        parts = event_name.split(" @ ", 1)
        away, home = parts[0].strip(), parts[1].strip()
    elif " v " in event_name:
        parts = event_name.split(" v ", 1)
        away, home = parts[0].strip(), parts[1].strip()

    rows = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for bo in offers:
        criterion = bo.get("criterion", {})
        market_name = criterion.get("label", "") if isinstance(criterion, dict) else ""
        market_type = _classify_market(market_name)
        outcomes = bo.get("outcomes", [])

        for oc in outcomes:
            label = oc.get("label", "")
            kambi_odds = oc.get("odds")  # Decimal × 1000
            american_str = oc.get("oddsAmerican", "")
            fractional = oc.get("oddsFractional", "")
            status = oc.get("status", "")
            line = oc.get("line")  # Spread/total line (Kambi uses �� 1000)

            if status != "OPEN" or not kambi_odds:
                continue

            # Convert Kambi odds to standard decimal
            decimal_odds = kambi_odds / 1000.0

            # Parse American odds
            try:
                american_odds = int(american_str.replace("+", ""))
            except (ValueError, AttributeError):
                # Convert from decimal
                if decimal_odds >= 2.0:
                    american_odds = round((decimal_odds - 1) * 100)
                else:
                    american_odds = round(-100 / (decimal_odds - 1))

            # Parse line (Kambi stores as × 1000 for some markets)
            line_value = None
            if line is not None:
                line_value = line / 1000.0

            side = _determine_side(label, home, away, market_type)

            row = {
                "book": "BetRivers",
                "home_team": home,
                "away_team": away,
                "event_name": event_name,
                "market_type": market_type,
                "market_name": market_name,
                "selection": label,
                "side": side,
                "american_odds": american_odds,
                "decimal_odds": decimal_odds,
                "line": line_value,
                "start_time": start,
                "scraped_at": now_iso,
                "event_id": str(event_id),
                "kambi_outcome_id": oc.get("id"),
            }
            rows.append(row)

    return rows


def _classify_market(name: str) -> str:
    nl = name.lower()
    if "moneyline" in nl:
        return "moneyline"
    if "puck line" in nl or "run line" in nl or "spread" in nl or "handicap" in nl:
        return "spread"
    if "total" in nl:
        return "total"
    if "3-way" in nl or "regular time" in nl or "double chance" in nl:
        return "three_way"
    if "correct score" in nl:
        return "correct_score"
    if "period" in nl or "inning" in nl or "half" in nl:
        return "period"
    return "other"


def _determine_side(label: str, home: str, away: str, market_type: str) -> str:
    ll = label.lower()
    if "over" in ll:
        return "over"
    if "under" in ll:
        return "under"
    if home and home.lower() in ll:
        return "home"
    if away and away.lower() in ll:
        return "away"
    if ll in ("1", "1x"):
        return "home"
    if ll in ("2", "x2"):
        return "away"
    if ll in ("x",):
        return "draw"
    return label


def scrape_sport(sport_name: str) -> list:
    """Scrape all NCAA odds for a sport from BetRivers/Kambi."""
    cfg = SPORTS.get(sport_name)
    if not cfg:
        print(f"Unknown sport: {sport_name}")
        return []

    sport_path = cfg["sport_path"]
    league_path = cfg["league_path"]
    sport_key = cfg["sport_key"]

    print(f"\n[BetRivers/Kambi] NCAA {sport_name.title()} ({sport_path}/{league_path})")

    # Get event list
    events = list_events(sport_path, league_path)
    print(f"  Found {len(events)} events")

    all_rows = []
    for ev_wrap in events:
        event = ev_wrap.get("event", {})
        eid = event.get("id")
        ename = event.get("name", "")

        if not eid:
            continue

        # Fetch full odds for this event
        time.sleep(0.3)  # Be polite
        event_data = get_event_odds(str(eid))
        if not event_data:
            print(f"    {ename}: no data")
            continue

        rows = parse_event_odds(event_data)
        print(f"    {ename}: {len(rows)} odds rows")
        all_rows.extend(rows)

    # Tag with sport_key
    for r in all_rows:
        r["sport_key"] = sport_key

    return all_rows


def push_to_supabase(rows: list, sport_key: str):
    """Push odds rows to Supabase line_snapshots table."""
    try:
        from supabase import create_client
    except ImportError:
        print("ERROR: pip install supabase")
        return

    url = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")

    if not url or not key:
        env_path = Path(__file__).parent.parent / ".env.local"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k == "NEXT_PUBLIC_SUPABASE_URL":
                        url = url or v
                    elif k == "SUPABASE_SERVICE_ROLE_KEY":
                        key = key or v

    if not url or not key:
        print("ERROR: Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY")
        return

    sb = create_client(url, key)
    today = date.today().isoformat()

    if sport_key == "ncaa_baseball":
        res = sb.table("baseball_predictions").select(
            "game_id, home_name, away_name, prediction_date"
        ).eq("prediction_date", today).execute()
        game_lookup = {}
        for r in (res.data or []):
            game_lookup[(r["home_name"], r["away_name"])] = r["game_id"]
    else:
        res = sb.table("projections").select(
            "game_id, home_team, away_team, game_date"
        ).eq("sport", sport_key).eq("game_date", today).execute()
        game_lookup = {}
        for r in (res.data or []):
            game_lookup[(r["home_team"], r["away_team"])] = r["game_id"]

    print(f"\n  Supabase: {len(game_lookup)} games in {sport_key} for {today}")

    main_rows = [r for r in rows if r["market_type"] in ("moneyline", "total", "spread")]
    inserted = 0
    unmatched = set()

    for r in main_rows:
        home = r["home_team"]
        away = r["away_team"]
        game_id = game_lookup.get((home, away))

        if not game_id:
            game_id = _fuzzy_match_game(home, away, game_lookup)

        if not game_id:
            unmatched.add(f"{away} @ {home}")
            continue

        snapshot = {
            "game_id": game_id,
            "book": "BetRivers",
            "market_type": r["market_type"],
            "home_line": None, "away_line": None,
            "home_odds": None, "away_odds": None,
            "total_line": None, "over_odds": None, "under_odds": None,
            "scraped_at": r["scraped_at"],
        }

        if r["market_type"] == "moneyline":
            if r["side"] == "home":
                snapshot["home_odds"] = r["american_odds"]
            elif r["side"] == "away":
                snapshot["away_odds"] = r["american_odds"]
        elif r["market_type"] == "spread":
            if r["side"] == "home":
                snapshot["home_line"] = r["line"]
                snapshot["home_odds"] = r["american_odds"]
            elif r["side"] == "away":
                snapshot["away_line"] = r["line"]
                snapshot["away_odds"] = r["american_odds"]
        elif r["market_type"] == "total":
            snapshot["total_line"] = r["line"]
            if r["side"] == "over":
                snapshot["over_odds"] = r["american_odds"]
            elif r["side"] == "under":
                snapshot["under_odds"] = r["american_odds"]

        try:
            sb.table("line_snapshots").insert(snapshot).execute()
            inserted += 1
        except Exception as e:
            print(f"  Insert error: {e}")

    if unmatched:
        print(f"  Unmatched games ({len(unmatched)}): {unmatched}")
    print(f"  Pushed {inserted}/{len(main_rows)} main-market rows to Supabase")


def _fuzzy_match_game(home: str, away: str, lookup: dict) -> str | None:
    def _normalize(name: str) -> str:
        # Kambi uses short names like "Michigan State" — strip common suffixes
        parts = name.rsplit(" ", 1)
        if len(parts) == 2 and len(parts[1]) > 3:
            common_words = {"state", "tech", "city", "north", "south", "east", "west", "central"}
            if parts[1].lower() not in common_words:
                return parts[0]
        return name

    norm_home = _normalize(home)
    norm_away = _normalize(away)

    for (lh, la), gid in lookup.items():
        if (_normalize(lh) == norm_home and _normalize(la) == norm_away):
            return gid
        if (norm_home in lh or lh in norm_home) and (norm_away in la or la in norm_away):
            return gid
    return None


def main():
    parser = argparse.ArgumentParser(description="BetRivers/Kambi NCAA Odds Scraper")
    parser.add_argument("--sport", choices=["baseball", "hockey", "basketball", "lacrosse", "all"], default="all")
    parser.add_argument("--push", action="store_true", help="Push to Supabase")
    args = parser.parse_args()

    print("=" * 60)
    print("  BetRivers/Kambi NCAA Odds Scraper")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M %Z')}")
    print("=" * 60)

    sports_to_scrape = list(SPORTS.keys()) if args.sport == "all" else [args.sport]
    all_rows = []

    for sport in sports_to_scrape:
        rows = scrape_sport(sport)
        all_rows.extend(rows)

    print(f"\n{'=' * 60}")
    print(f"  TOTAL: {len(all_rows)} odds rows")

    by_type = {}
    for r in all_rows:
        by_type[r["market_type"]] = by_type.get(r["market_type"], 0) + 1
    for mt, count in sorted(by_type.items()):
        print(f"    {mt}: {count}")

    by_sport = {}
    for r in all_rows:
        by_sport[r.get("sport_key", "?")] = by_sport.get(r.get("sport_key", "?"), 0) + 1
    for sk, count in sorted(by_sport.items()):
        print(f"    {sk}: {count}")

    print("=" * 60)

    if args.push and all_rows:
        for sport in sports_to_scrape:
            cfg = SPORTS[sport]
            sport_rows = [r for r in all_rows if r.get("sport_key") == cfg["sport_key"]]
            if sport_rows:
                push_to_supabase(sport_rows, cfg["sport_key"])

    return all_rows


if __name__ == "__main__":
    main()
