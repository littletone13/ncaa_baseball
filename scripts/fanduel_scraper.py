#!/usr/bin/env python3
"""
FanDuel Direct API Scraper — NCAA Hockey + Baseball

Discovered via mitmproxy interception of FanDuel iOS app.
Uses the sbapi endpoints which return full odds in REST (no WebSocket needed).

Usage:
    python adapters/fanduel_scraper.py                 # dry run
    python adapters/fanduel_scraper.py --push           # push to Supabase
    python adapters/fanduel_scraper.py --sport hockey   # single sport

Key endpoints:
    /sbapi/content-managed-page?eventTypeId=X  → sport landing (all competitions)
    /sbapi/competition-page?competitionId=X    → competition (all events + markets)
    /sbapi/event-page?eventId=X                → single event (all markets + outcomes)
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone, date, timedelta
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────
BASE_URL = "https://api.sportsbook.fanduel.com"
API_KEY = "oN2groXWNuItc4hZ"  # Static app key from FanDuel iOS
PX_AUTH = None  # Set from captures if needed; currently not required for competition-page

HEADERS = {
    "User-Agent": "FanDuel-Sportsbook-iOS/2.139.4",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "x-sportsbook-region": "MO",
}

# PerimeterX headers — refresh from captures/ if expired
# These come from mitmproxy interception of the FanDuel iOS app
PX_HEADERS = {
    "x-px-authorization": "3:cea4a59d62347a9f3de9bcb68e76c858c8194c5b9ce80e9e90c8450134c4b3b7:uYLbeX7T4CnWAVFDGhCJGoSNbbTBkrxTW0w9izm1ZPI3AOwqusfEkOuLwNdxwupaM4E28Ee8TbvMnTeuE+2zUw==:1000:rbrEE+ydzOOYxK1J4t1vUdN1W3BAdRfjwJqxrs+pSFWaAbDiaNOJNHJSZumHbZnFQYc7aF/fudAo1X4A9c+a3M56PMRdqS+YRpbnHtUcn1RCtCKtMXFXozytwv2TMiQGAW53GZH/hBb2RNkCMM+TdRBiep07wlsCA/mKTWnVmUwJVKw7YUaAK5aBlb6rU76woJhVcHX+GNAXsC7UpD22KH21qUHQJfoRmXZxCnYOt4C/n3AD+g0WkgG4yAVj2bqYaMcoxpcZDF5EHm9W5q0lKal9nosYsKCq28XklwsblqpVjBCqfKwB6xpgiKXKqF5WA1jNmaVOEEMtO/70FBCZWCCPehYuYhq4Rj3s9FJoiXo=",
    "x-px-uuid": "872d3f5e-2a5e-11f1-958e-17162551c440",
    "x-px-vid": "72db58f3-ca6d-11ed-ac8a-2c9914764e16",
    "x-px-os": "iOS",
    "x-px-mobile-sdk-version": "3.2.6",
}

# ── Sport / Competition mapping ───────────────────────────────────
# eventTypeId → sport ; competitionId → league
SPORTS = {
    "hockey": {
        "event_type_id": "7524",
        "sport_key": "ncaa_hockey",
        "competitions": {
            "12634197": "NCAA Hockey - Frozen Four",
            # Add more as discovered — browse FanDuel app to find IDs
        },
    },
    "baseball": {
        "event_type_id": "7511",
        "sport_key": "ncaa_baseball",
        "competitions": {
            # Populate when FanDuel posts NCAA baseball markets
            # Check: /sbapi/content-managed-page?eventTypeId=7511
        },
    },
}


def _fetch(path: str, params: dict = None) -> dict | None:
    """Make authenticated FanDuel API request."""
    url = f"{BASE_URL}{path}"
    p = {"_ak": API_KEY, **(params or {})}
    hdrs = {**HEADERS, **PX_HEADERS}
    try:
        r = requests.get(url, params=p, headers=hdrs, timeout=15)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"  WARN: {r.status_code} for {path}")
            return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def discover_competitions(event_type_id: str) -> list:
    """Find all competitions for a sport (requires PerimeterX token for large pages)."""
    data = _fetch("/sbapi/content-managed-page", {
        "page": "SPORT",
        "eventTypeId": event_type_id,
        "timezone": "America/Chicago",
    })
    if not data:
        return []

    atts = data.get("attachments", {})
    comps = atts.get("competitions", {})
    result = []
    for cid, comp in comps.items():
        result.append({
            "id": cid,
            "name": comp.get("name", "?"),
        })
    return result


def scrape_competition(competition_id: str, sport_key: str) -> list:
    """Scrape all events + markets for a competition."""
    data = _fetch("/sbapi/competition-page", {
        "eventTypeId": SPORTS.get(sport_key, {}).get("event_type_id", "7524"),
        "competitionId": competition_id,
    })
    if not data:
        return []

    atts = data.get("attachments", {})
    events = atts.get("events", {})
    markets = atts.get("markets", {})

    # Build event lookup
    event_map = {}
    for eid, ev in events.items():
        participants = ev.get("participants", [])
        home = next((p["name"] for p in participants if p.get("venueRole") == "Home"), None)
        away = next((p["name"] for p in participants if p.get("venueRole") == "Away"), None)
        if not home:
            # Fall back to name parsing
            name = ev.get("name", "")
            parts = name.split(" @ ")
            if len(parts) == 2:
                away, home = parts[0].strip(), parts[1].strip()
            elif " v " in name:
                parts = name.split(" v ")
                away, home = parts[0].strip(), parts[1].strip()

        event_map[eid] = {
            "name": ev.get("name"),
            "home": home,
            "away": away,
            "start": ev.get("openDate"),
        }

    rows = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for mid, m in markets.items():
        eid = m.get("eventId")
        ev = event_map.get(str(eid), {})
        mname = m.get("marketName", "")
        mtype = m.get("marketType", "")
        runners = m.get("runners", [])

        if not ev.get("home") or not runners:
            continue

        # Classify market type
        market_type = None
        if "MONEY_LINE" in mtype and "3-WAY" not in mtype:
            market_type = "h2h"
        elif "HANDICAP" in mtype or "PUCK_LINE" in mtype or "MATCH_HANDICAP" in mtype:
            market_type = "spreads"
        elif "TOTAL" in mtype and "TEAM" not in mtype and "HOME" not in mtype and "AWAY" not in mtype:
            if "60_MIN" not in mtype and "1ST" not in mtype and "ALTERNATE" not in mtype:
                market_type = "totals"

        if not market_type:
            continue

        # Extract odds
        home_runner = None
        away_runner = None
        over_runner = None
        under_runner = None

        for r in runners:
            rname = r.get("runnerName", "")
            odds_data = r.get("winRunnerOdds", {}).get("americanDisplayOdds", {})
            american_odds = odds_data.get("americanOdds")
            handicap = r.get("handicap")

            if market_type == "h2h":
                if ev.get("home") and ev["home"] in rname:
                    home_runner = {"odds": american_odds}
                elif ev.get("away") and ev["away"] in rname:
                    away_runner = {"odds": american_odds}
            elif market_type == "spreads":
                if ev.get("home") and ev["home"] in rname:
                    home_runner = {"odds": american_odds, "line": handicap}
                elif ev.get("away") and ev["away"] in rname:
                    away_runner = {"odds": american_odds, "line": handicap}
            elif market_type == "totals":
                if "Over" in rname:
                    over_runner = {"odds": american_odds, "line": handicap}
                elif "Under" in rname:
                    under_runner = {"odds": american_odds, "line": handicap}

        if market_type == "h2h" and home_runner and away_runner:
            rows.append({
                "home_team": ev["home"],
                "away_team": ev["away"],
                "game_start": ev.get("start"),
                "book": "FanDuel",
                "market": "h2h",
                "home_price": int(home_runner["odds"]),
                "away_price": int(away_runner["odds"]),
                "line": None,
                "sport": sport_key,
                "source": "fanduel_api",
                "captured_at": now_iso,
            })
        elif market_type == "spreads" and home_runner and away_runner:
            rows.append({
                "home_team": ev["home"],
                "away_team": ev["away"],
                "game_start": ev.get("start"),
                "book": "FanDuel",
                "market": "spreads",
                "home_price": int(home_runner["odds"]),
                "away_price": int(away_runner["odds"]),
                "line": float(home_runner.get("line", 0)),
                "sport": sport_key,
                "source": "fanduel_api",
                "captured_at": now_iso,
            })
        elif market_type == "totals" and over_runner and under_runner:
            rows.append({
                "home_team": ev["home"],
                "away_team": ev["away"],
                "game_start": ev.get("start"),
                "book": "FanDuel",
                "market": "totals",
                "home_price": int(over_runner["odds"]),
                "away_price": int(under_runner["odds"]),
                "line": float(over_runner.get("line", 0)),
                "sport": sport_key,
                "source": "fanduel_api",
                "captured_at": now_iso,
            })

    return rows


def scrape_event(event_id: str, sport_key: str) -> list:
    """Scrape ALL markets for a single event (includes props, alternates, etc.)."""
    data = _fetch("/sbapi/event-page", {
        "eventId": event_id,
        "includeOBP": "true",
        "isChapiEnabled": "true",
        "useQuickBets": "true",
    })
    if not data:
        return []

    atts = data.get("attachments", {})
    events = atts.get("events", {})
    markets = atts.get("markets", {})

    rows = []
    now_iso = datetime.now(timezone.utc).isoformat()

    # Get event info
    ev = list(events.values())[0] if events else {}
    name = ev.get("name", "")
    participants = ev.get("participants", [])
    home = next((p["name"] for p in participants if p.get("venueRole") == "Home"), None)
    away = next((p["name"] for p in participants if p.get("venueRole") == "Away"), None)
    if not home:
        parts = name.split(" @ ")
        if len(parts) == 2:
            away, home = parts[0].strip(), parts[1].strip()

    for mid, m in markets.items():
        mname = m.get("marketName", "")
        mtype = m.get("marketType", "")
        runners = m.get("runners", [])

        # Build output for ALL markets
        runner_data = []
        for r in runners:
            odds_data = r.get("winRunnerOdds", {}).get("americanDisplayOdds", {})
            runner_data.append({
                "name": r.get("runnerName", "?"),
                "odds": odds_data.get("americanOdds"),
                "handicap": r.get("handicap"),
            })

        if runner_data:
            rows.append({
                "event_name": name,
                "home_team": home,
                "away_team": away,
                "market_name": mname,
                "market_type": mtype,
                "runners": runner_data,
                "book": "FanDuel",
                "sport": sport_key,
                "captured_at": now_iso,
            })

    return rows


def push_to_supabase(rows: list):
    """Push FanDuel odds to Supabase line_snapshots."""
    env_file = Path(__file__).parent.parent / ".env.local"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if "SUPABASE_URL" in k:
                        os.environ.setdefault("SUPABASE_URL", v)
                    elif "ANON_KEY" in k or "SERVICE_KEY" in k:
                        os.environ.setdefault("SUPABASE_SERVICE_KEY", v)

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("ERROR: Set SUPABASE_URL and SUPABASE_SERVICE_KEY")
        return

    from supabase import create_client
    sb = create_client(url, key)

    # Build game_id lookup from projections + baseball_predictions
    sports_in_data = set(r["sport"] for r in rows)
    today = date.today()
    dates = [today.isoformat(), (today + timedelta(days=1)).isoformat(),
             (today + timedelta(days=2)).isoformat()]
    home_lookup = {}

    for sport in sports_in_data:
        for d in dates:
            if sport != "ncaa_baseball":
                res = sb.table("projections").select(
                    "game_id, home_team, away_team, game_date"
                ).eq("sport", sport).eq("game_date", d).execute()
                for p in (res.data or []):
                    home_lookup[(sport, p["home_team"].lower().strip())] = p["game_id"]
            else:
                res = sb.table("baseball_predictions").select(
                    "game_id, home_name, away_name, prediction_date"
                ).eq("prediction_date", d).execute()
                for p in (res.data or []):
                    home_lookup[(sport, p["home_name"].lower().strip())] = p["game_id"]

    print(f"  Projection lookup: {len(home_lookup)} entries")

    snapshots = []
    matched = unmatched = 0

    for r in rows:
        home = r["home_team"].lower().strip()
        sport = r["sport"]

        # Try exact match + common variations
        game_id = home_lookup.get((sport, home))
        if not game_id:
            # Try without common suffixes
            for suffix in [" spartans", " badgers", " bobcats", " fighting hawks",
                           " bulldogs", " broncos", " wolverines"]:
                if home.endswith(suffix):
                    game_id = home_lookup.get((sport, home.replace(suffix, "").strip()))
                    if game_id:
                        break

        if not game_id:
            unmatched += 1
            print(f"  UNMATCHED: {r['away_team']} @ {r['home_team']}")
            continue

        matched += 1
        snapshots.append({
            "game_id": game_id,
            "sport": sport,
            "book": "FanDuel",
            "market": r["market"],
            "home_price": r["home_price"],
            "away_price": r.get("away_price"),
            "line": r.get("line"),
            "is_opening": False,
            "is_closing": False,
            "source": "fanduel_api",
            "captured_at": r["captured_at"],
        })

    print(f"  Matched: {matched}, Unmatched: {unmatched}")

    if snapshots:
        for i in range(0, len(snapshots), 100):
            batch = snapshots[i:i + 100]
            sb.table("line_snapshots").insert(batch).execute()
        print(f"  Pushed {len(snapshots)} FanDuel line snapshots to Supabase")
    else:
        print("  No snapshots to push")


def main():
    parser = argparse.ArgumentParser(description="FanDuel NCAA odds scraper")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--sport", choices=["hockey", "baseball", "all"], default="all")
    parser.add_argument("--discover", action="store_true", help="Discover competition IDs")
    parser.add_argument("--event", type=str, help="Scrape single event ID (all markets)")
    args = parser.parse_args()

    sports = list(SPORTS.keys()) if args.sport == "all" else [args.sport]

    print("=" * 60)
    print(f"FANDUEL SCRAPER — {', '.join(sports)}")
    print("=" * 60)

    all_rows = []

    if args.event:
        # Scrape single event with all markets
        rows = scrape_event(args.event, sports[0])
        print(f"\nEvent {args.event}: {len(rows)} markets")
        for r in rows:
            runners_str = ", ".join(
                f"{rd['name']}: {rd['odds']}" for rd in r["runners"][:4]
            )
            print(f"  {r['market_name']}: {runners_str}")
        return

    if args.discover:
        for sport in sports:
            cfg = SPORTS[sport]
            print(f"\n[{sport}] Discovering competitions...")
            comps = discover_competitions(cfg["event_type_id"])
            for c in comps:
                print(f"  {c['id']}: {c['name']}")
        return

    for sport in sports:
        cfg = SPORTS[sport]
        for comp_id, comp_name in cfg["competitions"].items():
            print(f"\n  [{sport}] {comp_name} (competition {comp_id})...")
            rows = scrape_competition(comp_id, cfg["sport_key"])
            all_rows.extend(rows)
            print(f"  [{sport}] Extracted {len(rows)} odds rows")
            time.sleep(0.5)

    # Summary
    from collections import Counter
    markets = Counter(r["market"] for r in all_rows)
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {len(all_rows)} odds rows")
    print(f"Markets: {dict(markets)}")

    if not args.push:
        print(f"\n--- DRY RUN ---")
        for r in all_rows:
            line_str = ""
            if r["market"] == "h2h":
                line_str = f"ML: {r['home_price']}/{r['away_price']}"
            elif r["market"] == "totals":
                line_str = f"T: {r['line']} O{r['home_price']}/U{r['away_price']}"
            elif r["market"] == "spreads":
                line_str = f"S: {r['line']} ({r['home_price']}/{r['away_price']})"
            print(f"  {r['away_team']:25s} @ {r['home_team']:25s} {line_str}")
    else:
        push_to_supabase(all_rows)


if __name__ == "__main__":
    main()
