#!/usr/bin/env python3
"""
DraftKings Direct API Scraper — NCAA Baseball, Hockey, Basketball

Discovered via mitmproxy interception of DraftKings iOS app.
Uses the sportscontent REST API which returns full odds including trueOdds (no-vig decimal).

Usage:
    python adapters/draftkings_scraper.py                       # dry run (all sports)
    python adapters/draftkings_scraper.py --push                 # push to Supabase
    python adapters/draftkings_scraper.py --sport hockey         # single sport
    python adapters/draftkings_scraper.py --sport baseball --push

Key endpoints:
    /sites/US-MO-SB/api/sportscontent/controldata/league/marketSelector/v1/markets
        ?eventsQuery=$filter=leagueId eq 'X'
    /sites/US-MO-SB/api/sportscontent/pagedata/league/v1/events
        ?leagueIds=[X]

DraftKings League IDs (discovered from mitmproxy captures):
    84813  = College Hockey
    41151  = College Baseball
    92483  = College Basketball (M)
    204466 = College Lacrosse
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
BASE_URL = "https://sportsbook-nash.draftkings.com"

HEADERS = {
    "User-Agent": "dksb/5.40.2 (iOS; iPhone15,4; iOS26.3.1)",
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "x-client-name": "sbios",
    "x-dk-device-appname": "sbios",
    "x-client-version": "5.40.2",
    "x-dk-device-version": "5.40.2",
}

# DraftKings League IDs
SPORTS = {
    "baseball": {
        "league_id": "41151",
        "sport_id": "7",
        "sport_key": "ncaa_baseball",
        "league_name": "College Baseball",
        "subcategory_id": "13680",
    },
    "hockey": {
        "league_id": "84813",
        "sport_id": "8",
        "sport_key": "ncaa_hockey",
        "league_name": "College Hockey",
        "subcategory_id": "4525",
    },
    "basketball": {
        "league_id": "92483",
        "sport_id": "2",
        "sport_key": "ncaa_basketball",
        "league_name": "College Basketball (M)",
        "subcategory_id": "13680",
    },
    "lacrosse": {
        "league_id": "204466",
        "sport_id": "245",
        "sport_key": "ncaa_lacrosse",
        "league_name": "College Lacrosse",
        "subcategory_id": "13680",
    },
}


def _fetch(path: str, params: dict = None) -> dict | None:
    """Make DraftKings API request."""
    url = f"{BASE_URL}{path}"
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"  WARN: {r.status_code} for {path}")
            return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def scrape_league_markets(league_id: str, subcategory_id: str = "13680") -> dict | None:
    """Fetch all events + markets + selections for a league."""
    # The marketSelector endpoint returns everything in one call
    # Must include subcategory filter and format=json
    params = {
        "version": "5.40.2",
        "appname": "sbios",
        "eventsQuery": f"$filter=leagueId eq '{league_id}' AND clientMetadata/Subcategories/any(s: s/Id eq '{subcategory_id}')",
        "marketsQuery": f"$filter=clientMetadata/subCategoryId eq '{subcategory_id}' AND tags/all(t: t ne 'SportcastBetBuilder')",
        "include": "Events",
        "format": "json",
    }
    return _fetch(
        "/sites/US-MO-SB/api/sportscontent/controldata/league/marketSelector/v1/markets",
        params,
    )


def scrape_event_markets(event_id: str) -> dict | None:
    """Fetch all markets for a single event (for props/alt lines)."""
    params = {
        "version": "5.40.2",
        "appname": "sbios",
        "eventsQuery": f"$filter=eventId eq '{event_id}'",
    }
    return _fetch(
        "/sites/US-MO-SB/api/sportscontent/controldata/event/marketSelector/v1/markets",
        params,
    )


def parse_market_data(data: dict) -> list:
    """Parse DraftKings marketSelector response into odds rows."""
    if not data:
        return []

    events = {str(e["id"]): e for e in data.get("events", [])}
    markets = {str(m["id"]): m for m in data.get("markets", [])}
    selections = data.get("selections", [])

    rows = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for sel in selections:
        mid = str(sel.get("marketId", ""))
        mkt = markets.get(mid, {})
        eid = str(mkt.get("eventId", ""))
        ev = events.get(eid, {})

        if not ev:
            continue

        # Parse event info
        event_name = ev.get("name", "")
        start_date = ev.get("startEventDate", "")
        participants = ev.get("participants", [])

        home, away = None, None
        for p in participants:
            ptype = p.get("type", "")
            pname = p.get("name", "")
            if ptype == "Team":
                if not away:
                    away = pname  # First participant is typically away
                elif not home:
                    home = pname

        # Parse from event name if participants didn't work
        if not home or not away:
            if " @ " in event_name:
                parts = event_name.split(" @ ", 1)
                away, home = parts[0].strip(), parts[1].strip()
            elif " vs " in event_name.lower():
                parts = re.split(r'\s+vs\.?\s+', event_name, flags=re.IGNORECASE)
                if len(parts) == 2:
                    away, home = parts[0].strip(), parts[1].strip()

        # Parse selection (odds)
        label = sel.get("label", "")
        display_odds = sel.get("displayOdds", {})
        american_odds_str = display_odds.get("american", "")
        true_odds = sel.get("trueOdds", 0)
        points = sel.get("points", None)
        outcome_type = sel.get("outcomeType", "")

        # Convert American odds string to int
        try:
            american_odds = int(american_odds_str.replace("+", "").replace("−", "-").replace("–", "-").replace("—", "-"))
        except (ValueError, AttributeError):
            # Try unicode minus signs
            cleaned = american_odds_str
            for char in "−–—":
                cleaned = cleaned.replace(char, "-")
            try:
                american_odds = int(cleaned.replace("+", ""))
            except (ValueError, AttributeError):
                continue

        # Market info
        market_name = mkt.get("name", "")
        market_type = _classify_market(market_name)

        # Determine side
        side = _determine_side(label, home, away, market_type)

        row = {
            "book": "DraftKings",
            "home_team": home,
            "away_team": away,
            "event_name": event_name,
            "market_type": market_type,
            "market_name": market_name,
            "selection": label,
            "side": side,
            "american_odds": american_odds,
            "true_odds": float(true_odds) if true_odds else None,
            "line": float(points) if points else None,
            "start_time": start_date,
            "scraped_at": now_iso,
            "event_id": eid,
        }
        rows.append(row)

    return rows


def _classify_market(name: str) -> str:
    nl = name.lower()
    if "moneyline" in nl or "money line" in nl:
        return "moneyline"
    if "spread" in nl or "run line" in nl or "puck line" in nl:
        return "spread"
    if "total" in nl:
        return "total"
    if "period" in nl or "half" in nl or "inning" in nl:
        return "period"
    if "player" in nl or "prop" in nl:
        return "player_prop"
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
    # For spread, first word is usually team abbreviation
    if home and ll.startswith(home.lower().split()[0]):
        return "home"
    if away and ll.startswith(away.lower().split()[0]):
        return "away"
    return label


def scrape_sport(sport_name: str) -> list:
    """Scrape all odds for a sport's NCAA league."""
    cfg = SPORTS.get(sport_name)
    if not cfg:
        print(f"Unknown sport: {sport_name}")
        return []

    league_id = cfg["league_id"]
    sport_key = cfg["sport_key"]
    league_name = cfg["league_name"]

    print(f"\n[DraftKings] {league_name} (leagueId={league_id})")

    subcategory_id = cfg.get("subcategory_id", "13680")
    data = scrape_league_markets(league_id, subcategory_id)
    if not data:
        print("  No data returned")
        return []

    n_events = len(data.get("events", []))
    n_markets = len(data.get("markets", []))
    n_selections = len(data.get("selections", []))
    print(f"  Events: {n_events}, Markets: {n_markets}, Selections: {n_selections}")

    rows = parse_market_data(data)
    print(f"  Parsed: {len(rows)} odds rows")

    # Tag with sport_key
    for r in rows:
        r["sport_key"] = sport_key

    # Print summary
    for ev_name in sorted(set(r["event_name"] for r in rows)):
        ev_rows = [r for r in rows if r["event_name"] == ev_name]
        mkts = sorted(set(r["market_name"] for r in ev_rows))
        print(f"    {ev_name}: {', '.join(mkts)}")

    return rows


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
    date_slug = today.replace("-", "")

    # Load game_ids from projections — same table for all sports
    res = sb.table("projections").select(
        "game_id, home_team, away_team, game_date"
    ).eq("sport", sport_key).eq("game_date", today).execute()

    # Index by game_id and by normalized pair for flexible matching
    game_id_set: set[str] = set()
    pair_lookup: dict[tuple[str, str], str] = {}
    for r in (res.data or []):
        gid = r["game_id"]
        game_id_set.add(gid)
        h = r["home_team"].lower().strip()
        a = r["away_team"].lower().strip()
        pair_lookup[(a, h)] = gid

    print(f"\n  Supabase: {len(game_id_set)} games in {sport_key} for {today}")

    # Sport prefix for game_id construction
    prefix_map = {
        "ncaa_baseball": "bsb",
        "ncaa_hockey": "ncaa_hky",
        "ncaa_basketball": "cbb",
        "ncaa_lacrosse": "lac",
    }
    prefix = prefix_map.get(sport_key, sport_key)

    # Filter to main markets
    main_rows = [r for r in rows if r["market_type"] in ("moneyline", "total", "spread")]

    inserted = 0
    unmatched = set()
    for r in main_rows:
        home = r["home_team"]
        away = r["away_team"]

        # Canonicalize names
        canon_away = _canonical_team(away)
        canon_home = _canonical_team(home)

        # Build candidate game_id (same logic as TS cron)
        if sport_key == "ncaa_hockey":
            candidate = f"ncaa_hky_{date_slug}_{canon_away.replace(' ', '_')}_at_{canon_home.replace(' ', '_')}"
        else:
            candidate = f"{prefix}_{date_slug}_{_slugify(canon_away)}_{_slugify(canon_home)}"

        # 1. Direct game_id match
        game_id = candidate if candidate in game_id_set else None

        # 2. Pair lookup (projections' own team names)
        if not game_id:
            game_id = pair_lookup.get((canon_away, canon_home))
        if not game_id:
            game_id = pair_lookup.get((away.lower().strip(), home.lower().strip()))

        if not game_id:
            unmatched.add(f"{away} @ {home}")
            continue

        snapshot = {
            "game_id": game_id,
            "book": "DraftKings",
            "market_type": r["market_type"],
            "home_line": None,
            "away_line": None,
            "home_odds": None,
            "away_odds": None,
            "total_line": None,
            "over_odds": None,
            "under_odds": None,
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


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


_TEAM_ALIASES: dict[str, str] = {
    "appalachian state": "app state",
    "appalachian state mountaineers": "app state",
    "southern mississippi": "southern miss.",
    "southern miss golden eagles": "southern miss.",
    "nc state wolfpack": "nc state",
    "georgia tech yellow jackets": "georgia tech",
    "texas tech red raiders": "texas tech",
    "tcu horned frogs": "tcu",
    "california golden bears": "california",
    "missouri state": "missouri st.",
    "missouri state bears": "missouri st.",
    "dallas baptist": "dbu",
    "dallas baptist patriots": "dbu",
    "oklahoma st cowboys": "oklahoma st.",
    "oklahoma state": "oklahoma st.",
    "oklahoma state cowboys": "oklahoma st.",
    "florida gulf coast": "fgcu",
    "florida gulf coast eagles": "fgcu",
    "uc santa barbara gauchos": "uc santa barbara",
    "usc trojans": "southern california",
    "usc": "southern california",
    "miami hurricanes": "miami (fl)",
    "miami fl hurricanes": "miami (fl)",
    "san jose state": "san jose st.",
    "san jose state spartans": "san jose st.",
    "san diego state": "san diego st.",
    "san diego state aztecs": "san diego st.",
    "florida state": "florida st.",
    "florida state seminoles": "florida st.",
    "ohio state": "ohio st.",
    "ohio state buckeyes": "ohio st.",
    "penn state": "penn st.",
    "penn state nittany lions": "penn st.",
    "oregon state": "oregon st.",
    "oregon state beavers": "oregon st.",
    "mississippi state": "mississippi st.",
    "mississippi state bulldogs": "mississippi st.",
    "fresno state": "fresno st.",
    "fresno state bulldogs": "fresno st.",
    "kansas state": "kansas st.",
    "kansas state wildcats": "kansas st.",
    "arizona state": "arizona st.",
    "arizona state sun devils": "arizona st.",
    "wichita state": "wichita st.",
    "wichita state shockers": "wichita st.",
    "boise state": "boise st.",
    "iowa state": "iowa st.",
    "grand canyon lopes": "grand canyon",
}

_MULTI_WORD_MASCOTS = [
    "golden eagles", "blue devils", "crimson tide", "fighting irish",
    "yellow jackets", "red raiders", "sun devils", "horned frogs",
    "golden bears", "demon deacons", "scarlet knights", "tar heels",
    "red wolves", "blue hose", "golden flashes", "purple aces",
    "golden panthers", "fighting camels", "mountain hawks", "red foxes",
    "thundering herd", "rainbow warriors", "wolf pack",
]


def _strip_mascot(name: str) -> str:
    lower = name.strip().lower()
    for mascot in _MULTI_WORD_MASCOTS:
        if lower.endswith(f" {mascot}"):
            return lower[: -(len(mascot) + 1)].strip()
    parts = lower.split()
    if len(parts) >= 2:
        return " ".join(parts[:-1])
    return lower


def _canonical_team(name: str) -> str:
    lower = name.strip().lower()
    if lower in _TEAM_ALIASES:
        return _TEAM_ALIASES[lower]
    stripped = _strip_mascot(name)
    if stripped in _TEAM_ALIASES:
        return _TEAM_ALIASES[stripped]
    return stripped


def main():
    parser = argparse.ArgumentParser(description="DraftKings NCAA Odds Scraper")
    parser.add_argument("--sport", choices=["baseball", "hockey", "basketball", "lacrosse", "all"], default="all")
    parser.add_argument("--push", action="store_true", help="Push to Supabase")
    args = parser.parse_args()

    print("=" * 60)
    print("  DraftKings NCAA Odds Scraper")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M %Z')}")
    print("=" * 60)

    sports_to_scrape = list(SPORTS.keys()) if args.sport == "all" else [args.sport]
    all_rows = []

    for sport in sports_to_scrape:
        rows = scrape_sport(sport)
        all_rows.extend(rows)
        if sport != sports_to_scrape[-1]:
            time.sleep(1)  # Be polite between sports

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
            sport_rows = [r for r in all_rows if r.get("sport_key") == SPORTS[sport]["sport_key"]]
            if sport_rows:
                push_to_supabase(sport_rows, SPORTS[sport]["sport_key"])

    return all_rows


if __name__ == "__main__":
    main()
