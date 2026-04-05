#!/usr/bin/env python3
"""
BetMGM Direct API Scraper — NCAA Baseball, Hockey, Basketball

Discovered via mitmproxy interception of BetMGM iOS app.
Uses the cds-api/bettingoffer endpoints which return full odds in REST JSON.

Usage:
    python adapters/betmgm_scraper.py                       # dry run (all sports)
    python adapters/betmgm_scraper.py --push                 # push to Supabase
    python adapters/betmgm_scraper.py --sport hockey         # single sport
    python adapters/betmgm_scraper.py --sport baseball --push

Key endpoints:
    /cds-api/bettingoffer/fixtures?sportIds=X&competitionIds=Y  → all events for a competition
    /cds-api/bettingoffer/fixture-view?fixtureId=X              → single event (all markets)
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
BASE_URL = "https://www.mo.betmgm.com"
ACCESS_ID = "Y2U3MWQ4YTItNjM2Ni00ZjYyLTg2ZjYtYmI3YzY0ZmIzYWU4"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.mo.betmgm.com",
    "Referer": "https://www.mo.betmgm.com/en/sports",
}

# ── BetMGM Sport & Competition IDs ────────────────────────────────
# Discovered from mitmproxy captures:
#   sportId 23 = Baseball    competitionId 5909  = College Baseball
#   sportId 12 = Hockey      competitionId 12500 = College Ice Hockey
#   sportId  7 = Basketball  competitionId 264   = College Basketball
SPORTS = {
    "baseball": {
        "sport_id": "23",
        "sport_key": "ncaa_baseball",
        "competitions": {
            "5909": "College Baseball",
        },
    },
    "hockey": {
        "sport_id": "12",
        "sport_key": "ncaa_hockey",
        "competitions": {
            "12500": "College Ice Hockey",
        },
    },
    "basketball": {
        "sport_id": "7",
        "sport_key": "ncaa_basketball",
        "competitions": {
            "264": "College Basketball",
        },
    },
}

# Common query params for all cds-api requests
COMMON_PARAMS = {
    "x-bwin-accessid": ACCESS_ID,
    "lang": "en-us",
    "country": "US",
    "userCountry": "US",
    "subdivision": "US-Missouri",
}


def _fetch(path: str, params: dict = None) -> dict | None:
    """Make BetMGM cds-api request."""
    url = f"{BASE_URL}{path}"
    p = {**COMMON_PARAMS, **(params or {})}
    try:
        r = requests.get(url, params=p, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"  WARN: {r.status_code} for {path}")
            if r.status_code == 403:
                print("  → CloudFlare/WAF block — may need fresh cookies or app User-Agent")
            return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def _safe_name(val) -> str:
    """Extract string from BetMGM name field (can be dict or str)."""
    if isinstance(val, dict):
        return val.get("value", "")
    return str(val) if val else ""


def scrape_fixtures(sport_id: str, competition_id: str = None) -> list:
    """Fetch all fixtures (events) for a sport/competition."""
    params = {
        "fixtureTypes": "Standard",
        "state": "Latest",
        "offerMapping": "Filtered",
        "offerCategories": "Gridable",
        "fixtureCategories": "Gridable",
        "sportIds": sport_id,
        "isPriceBoost": "false",
        "statisticsModes": "None",
        "skip": "0",
        "take": "100",
        "sortBy": "StartDate",
    }
    if competition_id:
        params["competitionIds"] = competition_id

    data = _fetch("/cds-api/bettingoffer/fixtures", params)
    if not data:
        return []

    return data.get("fixtures", [])


def scrape_fixture_detail(fixture_id: str) -> dict | None:
    """Fetch all markets for a single fixture."""
    data = _fetch("/cds-api/bettingoffer/fixture-view", {
        "offerMapping": "All",
        "fixtureId": fixture_id,
    })
    if not data:
        return None
    return data.get("fixture", {})


def parse_fixture_odds(fixture: dict) -> list:
    """Extract odds rows from a fixture's games (markets)."""
    name = _safe_name(fixture.get("name", {}))
    fixture_id = fixture.get("id")
    participants = fixture.get("participants", [])
    start_date = fixture.get("startDate")

    # Extract team names
    home, away = None, None
    for p in participants:
        pname = _safe_name(p.get("name", {}))
        # BetMGM lists away @ home — first participant is usually away
        # We'll match by checking source type or just take order
        if not away:
            away = pname
        elif not home:
            home = pname

    # In BetMGM the event name is "Away at Home" — parse for accuracy
    if " at " in name:
        parts = name.split(" at ", 1)
        away = parts[0].strip()
        home_part = parts[1].strip()
        # Remove suffixes like "(Neutral Venue)"
        home = re.sub(r'\s*\(.*?\)\s*$', '', home_part).strip()

    games = fixture.get("games", [])
    rows = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for game in games:
        market_name = _safe_name(game.get("name", {}))
        results = game.get("results", [])
        is_main = game.get("isMain", False)

        if not results:
            continue

        for result in results:
            american_odds = result.get("americanOdds")
            decimal_odds = result.get("odds")
            result_name = _safe_name(result.get("name", {}))
            visibility = result.get("visibility", "")
            attr = result.get("attr")  # spread value like "+1.5"
            totals_prefix = result.get("totalsPrefix")  # "Over" or "Under"

            if american_odds is None or visibility != "Visible":
                continue

            # Determine market type and details
            market_type = _classify_market(market_name, totals_prefix)
            line_value = None

            if market_type == "spread" and attr:
                try:
                    line_value = float(attr)
                except (ValueError, TypeError):
                    pass
            elif market_type == "total":
                # Extract total line from result name like "Over 6.5"
                m = re.search(r'[\d.]+', result_name)
                if m:
                    try:
                        line_value = float(m.group())
                    except ValueError:
                        pass

            # Determine which team this result belongs to
            side = _determine_side(result_name, home, away, totals_prefix)

            row = {
                "book": "BetMGM",
                "home_team": home,
                "away_team": away,
                "market_type": market_type,
                "market_name": market_name,
                "selection": result_name,
                "side": side,
                "american_odds": american_odds,
                "decimal_odds": decimal_odds,
                "line": line_value,
                "is_main": is_main,
                "start_time": start_date,
                "scraped_at": now_iso,
                "fixture_id": str(fixture_id),
            }
            rows.append(row)

    return rows


def _classify_market(name: str, totals_prefix: str = None) -> str:
    """Classify market into standard type."""
    nl = name.lower()
    if totals_prefix:
        return "total"
    if "moneyline" in nl or "money line" in nl:
        return "moneyline"
    if "spread" in nl or "handicap" in nl:
        return "spread"
    if "total" in nl or "over" in nl or "under" in nl:
        return "total"
    if "3-way" in nl or "result" in nl:
        return "three_way"
    if "period" in nl:
        return "period"
    if "player" in nl or "scorer" in nl:
        return "player_prop"
    return "other"


def _determine_side(result_name: str, home: str, away: str, totals_prefix: str = None) -> str:
    """Figure out which side (home/away/over/under) a result belongs to."""
    if totals_prefix:
        return totals_prefix.lower()  # "over" or "under"

    rn = result_name.lower()
    if home and home.lower() in rn:
        return "home"
    if away and away.lower() in rn:
        return "away"
    if "over" in rn:
        return "over"
    if "under" in rn:
        return "under"
    if "tie" in rn or "draw" in rn:
        return "draw"
    return result_name


def scrape_sport(sport_name: str, include_all_markets: bool = False) -> list:
    """Scrape all odds for a sport's NCAA competitions."""
    cfg = SPORTS.get(sport_name)
    if not cfg:
        print(f"Unknown sport: {sport_name}")
        return []

    sport_id = cfg["sport_id"]
    sport_key = cfg["sport_key"]
    all_rows = []

    for comp_id, comp_name in cfg["competitions"].items():
        print(f"\n[BetMGM] {comp_name} (sport={sport_id}, comp={comp_id})")

        fixtures = scrape_fixtures(sport_id, comp_id)
        print(f"  Found {len(fixtures)} fixtures")

        for fix in fixtures:
            fname = _safe_name(fix.get("name", {}))
            fid = fix.get("id")
            start = fix.get("startDate", "")

            # Quick odds from fixtures list (ML/Spread/Totals already included)
            rows = parse_fixture_odds(fix)
            print(f"  {fname} (id={fid}): {len(rows)} odds from grid")
            all_rows.extend(rows)

            # If we want all markets, fetch the detailed fixture view
            if include_all_markets and fid:
                time.sleep(0.5)  # Be polite
                detail = scrape_fixture_detail(str(fid))
                if detail:
                    detail_rows = parse_fixture_odds(detail)
                    # Only add markets we don't already have
                    existing = {(r["market_name"], r["selection"]) for r in rows}
                    new_rows = [r for r in detail_rows if (r["market_name"], r["selection"]) not in existing]
                    if new_rows:
                        print(f"    + {len(new_rows)} additional markets from fixture-view")
                        all_rows.extend(new_rows)

    # Tag all rows with sport_key
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
        # Try .env.local
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

    # Look up game_ids from projections/predictions tables
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

    # Filter to main markets only (ML + totals) for line_snapshots
    main_rows = [r for r in rows if r["market_type"] in ("moneyline", "total", "spread") and r["is_main"]]

    inserted = 0
    unmatched = set()
    for r in main_rows:
        home = r["home_team"]
        away = r["away_team"]
        game_id = game_lookup.get((home, away))

        if not game_id:
            # Try fuzzy match — strip common suffixes
            game_id = _fuzzy_match_game(home, away, game_lookup)

        if not game_id:
            unmatched.add(f"{away} @ {home}")
            continue

        snapshot = {
            "game_id": game_id,
            "book": "BetMGM",
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


def _fuzzy_match_game(home: str, away: str, lookup: dict) -> str | None:
    """Try to match team names with common variations stripped."""
    def _normalize(name: str) -> str:
        # Remove common college mascot suffixes
        # "Michigan State Spartans" → "Michigan State"
        parts = name.rsplit(" ", 1)
        if len(parts) == 2 and len(parts[1]) > 3:
            # Check if last word is likely a mascot (not a state/city name)
            common_words = {"state", "tech", "city", "north", "south", "east", "west", "central"}
            if parts[1].lower() not in common_words:
                return parts[0]
        return name

    norm_home = _normalize(home)
    norm_away = _normalize(away)

    for (lh, la), gid in lookup.items():
        if (_normalize(lh) == norm_home and _normalize(la) == norm_away):
            return gid
        # Also try substring match
        if (norm_home in lh or lh in norm_home) and (norm_away in la or la in norm_away):
            return gid

    return None


def main():
    parser = argparse.ArgumentParser(description="BetMGM NCAA Odds Scraper")
    parser.add_argument("--sport", choices=["baseball", "hockey", "basketball", "all"], default="all")
    parser.add_argument("--push", action="store_true", help="Push to Supabase")
    parser.add_argument("--all-markets", action="store_true", help="Fetch all markets (slower)")
    args = parser.parse_args()

    print("=" * 60)
    print("  BetMGM NCAA Odds Scraper")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M %Z')}")
    print("=" * 60)

    sports_to_scrape = list(SPORTS.keys()) if args.sport == "all" else [args.sport]
    all_rows = []

    for sport in sports_to_scrape:
        rows = scrape_sport(sport, include_all_markets=args.all_markets)
        all_rows.extend(rows)

    print(f"\n{'=' * 60}")
    print(f"  TOTAL: {len(all_rows)} odds rows")

    # Summary by market type
    by_type = {}
    for r in all_rows:
        by_type[r["market_type"]] = by_type.get(r["market_type"], 0) + 1
    for mt, count in sorted(by_type.items()):
        print(f"    {mt}: {count}")

    # Summary by sport
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
