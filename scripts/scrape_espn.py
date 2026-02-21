"""
Scrape NCAA baseball data from ESPN API.

One endpoint per game gives us:
  - Game result (scores, teams, date, venue)
  - Full boxscore (per-player batting + pitching with starter flags)
  - Play-by-play where available (~23% of games — biased toward lined games)
  - Line scores by inning

Usage:
    python3 scripts/scrape_espn.py --start 2024-02-14 --end 2024-06-30 --out data/raw/espn/games_2024.jsonl
    python3 scripts/scrape_espn.py --start 2025-02-14 --end 2025-06-30 --out data/raw/espn/games_2025.jsonl
    python3 scripts/scrape_espn.py --start 2026-02-14 --end 2026-02-20 --out data/raw/espn/games_2026.jsonl

Output: JSONL (one JSON object per game), each containing:
    {
        "event_id": "401749088",
        "date": "2025-05-01",
        "season": 2025,
        "home_team": {"id": "8", "name": "Arkansas Razorbacks", "abbreviation": "ARK"},
        "away_team": {"id": "126", "name": "Texas Longhorns", "abbreviation": "TEX"},
        "home_score": 9,
        "away_score": 0,
        "neutral_site": false,
        "venue": {"name": "Baum-Walker Stadium", "city": "Fayetteville", "state": "AR"},
        "pbp_available": true,
        "line_scores": {"home": [2,0,0,...], "away": [0,0,0,...]},
        "boxscore": {
            "home": {"batting": [...], "pitching": [...]},
            "away": {"batting": [...], "pitching": [...]}
        },
        "plays": [...],      # full PBP if available
        "at_bats": {...},    # at-bat groupings if available
        "run_events": {      # DERIVED: run-event counts for Mack model
            "home": {"run_1": N, "run_2": N, "run_3": N, "run_4": N},
            "away": {"run_1": N, "run_2": N, "run_3": N, "run_4": N}
        },
        "starters": {
            "home_pitcher": {"name": "...", "espn_id": "..."},
            "away_pitcher": {"name": "...", "espn_id": "..."}
        }
    }
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE = "https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"


def fetch_json(url: str, retries: int = 3, delay: float = 1.0) -> dict | None:
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": UA})
            with urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except HTTPError as e:
            if e.code == 404:
                return None
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    return None


def get_scoreboard(dt: date) -> list[dict]:
    """Fetch all games for a given date."""
    url = f"{BASE}/scoreboard?dates={dt.strftime('%Y%m%d')}&limit=200"
    data = fetch_json(url)
    if not data:
        return []
    return data.get("events", [])


def extract_run_events(plays: list[dict]) -> dict:
    """
    From PBP plays, count run-scoring events per team.
    A 'run event' is an at-bat where N runs scored (N=1,2,3,4+).
    This is the core input for Mack's log-linear NegBin model.

    Plays are grouped by atBatId so that score changes within a single
    at-bat are counted exactly once. Within each at-bat we take the max
    score seen (ESPN can reset scores to 0 on substitution plays) and
    compare against the running high-water mark.
    """
    home_runs = {1: 0, 2: 0, 3: 0, 4: 0}
    away_runs = {1: 0, 2: 0, 3: 0, 4: 0}

    ab_max: dict[str, tuple[int, int]] = {}
    ab_order: list[str] = []

    for play in plays:
        ab_id = play.get("atBatId")
        if ab_id is None:
            continue
        ab_id = str(ab_id)
        cur_home = play.get("homeScore", 0)
        cur_away = play.get("awayScore", 0)

        if ab_id not in ab_max:
            ab_max[ab_id] = (cur_home, cur_away)
            ab_order.append(ab_id)
        else:
            prev_h, prev_a = ab_max[ab_id]
            ab_max[ab_id] = (max(prev_h, cur_home), max(prev_a, cur_away))

    max_home = 0
    max_away = 0

    for ab_id in ab_order:
        h, a = ab_max[ab_id]

        if h > max_home:
            delta = h - max_home
            bucket = min(delta, 4)
            home_runs[bucket] += 1
            max_home = h

        if a > max_away:
            delta = a - max_away
            bucket = min(delta, 4)
            away_runs[bucket] += 1
            max_away = a

    return {
        "home": {f"run_{k}": v for k, v in home_runs.items()},
        "away": {f"run_{k}": v for k, v in away_runs.items()},
    }


def extract_starters(boxscore: dict, home_idx: int, away_idx: int) -> dict:
    """Extract starting pitchers from boxscore."""
    result = {"home_pitcher": None, "away_pitcher": None}
    players = boxscore.get("players", [])

    for team_section in players:
        team_id = team_section.get("team", {}).get("id")
        for stat_cat in team_section.get("statistics", []):
            labels = stat_cat.get("labels", [])
            if "IP" not in labels:
                continue  # This is pitching stats
            for athlete in stat_cat.get("athletes", []):
                if athlete.get("starter"):
                    info = {
                        "name": athlete["athlete"]["displayName"],
                        "espn_id": athlete["athlete"].get("id"),
                    }
                    if team_id == str(home_idx):
                        result["home_pitcher"] = info
                    elif team_id == str(away_idx):
                        result["away_pitcher"] = info
    return result


def extract_boxscore_players(boxscore: dict) -> dict:
    """Extract per-player batting and pitching stats."""
    out = {}
    for team_section in boxscore.get("players", []):
        team_abbr = team_section.get("team", {}).get("abbreviation", "?")
        team_data = {"batting": [], "pitching": []}

        for stat_cat in team_section.get("statistics", []):
            labels = stat_cat.get("labels", [])
            cat_type = "pitching" if "IP" in labels else "batting"
            for athlete in stat_cat.get("athletes", []):
                row = {
                    "name": athlete["athlete"]["displayName"],
                    "espn_id": athlete["athlete"].get("id"),
                    "starter": athlete.get("starter", False),
                    "stats": dict(zip(labels, athlete.get("stats", []))),
                }
                team_data[cat_type].append(row)

        out[team_abbr] = team_data
    return out


def process_game(event: dict) -> dict | None:
    """Fetch summary for a single game and extract all data."""
    event_id = event["id"]
    comp = event["competitions"][0]

    # Identify home/away
    competitors = comp.get("competitors", [])
    home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
    away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1] if len(competitors) > 1 else competitors[0])

    status = comp.get("status", {}).get("type", {}).get("name", "")
    if status not in ("STATUS_FINAL", "STATUS_FULL_TIME"):
        return None  # Skip non-final games

    home_score = int(home.get("score", 0)) if home.get("score") else None
    away_score = int(away.get("score", 0)) if away.get("score") else None
    if home_score is None or away_score is None:
        return None

    game_date = event.get("date", "")[:10]
    pbp_available = comp.get("playByPlayAvailable", False)

    record = {
        "event_id": event_id,
        "date": game_date,
        "season": event.get("season", {}).get("year"),
        "home_team": {
            "id": home["team"]["id"],
            "name": home["team"]["displayName"],
            "abbreviation": home["team"].get("abbreviation", ""),
        },
        "away_team": {
            "id": away["team"]["id"],
            "name": away["team"]["displayName"],
            "abbreviation": away["team"].get("abbreviation", ""),
        },
        "home_score": home_score,
        "away_score": away_score,
        "neutral_site": comp.get("neutralSite", False),
        "venue": None,
        "pbp_available": pbp_available,
        "line_scores": {"home": [], "away": []},
        "boxscore": {},
        "plays": [],
        "run_events": None,
        "starters": {"home_pitcher": None, "away_pitcher": None},
    }

    # Fetch full summary for boxscore + PBP
    summary = fetch_json(f"{BASE}/summary?event={event_id}")
    if summary:
        # Venue
        gi = summary.get("gameInfo", {})
        venue = gi.get("venue", {})
        if venue:
            addr = venue.get("address", {})
            record["venue"] = {
                "name": venue.get("fullName"),
                "city": addr.get("city"),
                "state": addr.get("state"),
            }

        # Line scores from header
        header = summary.get("header", {})
        hcomps = header.get("competitions", [{}])
        if hcomps:
            for c in hcomps[0].get("competitors", []):
                ls = [l.get("value", 0) for l in c.get("linescores", [])]
                if c.get("homeAway") == "home":
                    record["line_scores"]["home"] = ls
                else:
                    record["line_scores"]["away"] = ls

        # Boxscore
        bs = summary.get("boxscore", {})
        record["boxscore"] = extract_boxscore_players(bs)
        record["starters"] = extract_starters(bs, int(home["team"]["id"]), int(away["team"]["id"]))

        # PBP
        plays = summary.get("plays", [])
        if plays:
            record["plays"] = plays
            record["run_events"] = extract_run_events(plays)

    return record


def iter_dates(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(description="Scrape ESPN NCAA baseball data")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--sleep", type=float, default=0.3, help="Sleep between summary requests")
    parser.add_argument("--skip-summary", action="store_true", help="Only fetch scoreboard (no boxscore/PBP)")
    parser.add_argument("--resume", action="store_true", help="Skip games already in output file")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Load existing event IDs if resuming
    existing_ids = set()
    if args.resume and args.out.exists():
        with open(args.out) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing_ids.add(obj["event_id"])
                except Exception:
                    pass
        print(f"Resuming: {len(existing_ids)} games already scraped")

    mode = "a" if args.resume else "w"
    total_games = 0
    total_pbp = 0
    total_boxscore = 0

    with open(args.out, mode) as fout:
        for dt in iter_dates(start, end):
            events = get_scoreboard(dt)
            finals = [
                e for e in events
                if e["competitions"][0].get("status", {}).get("type", {}).get("name") in ("STATUS_FINAL", "STATUS_FULL_TIME")
            ]

            if not finals:
                continue

            day_games = 0
            day_pbp = 0
            for event in finals:
                if event["id"] in existing_ids:
                    continue

                if args.skip_summary:
                    # Quick mode: just scores from scoreboard
                    comp = event["competitions"][0]
                    competitors = comp.get("competitors", [])
                    home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
                    away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
                    record = {
                        "event_id": event["id"],
                        "date": event.get("date", "")[:10],
                        "season": event.get("season", {}).get("year"),
                        "home_team": {"id": home["team"]["id"], "name": home["team"]["displayName"], "abbreviation": home["team"].get("abbreviation", "")},
                        "away_team": {"id": away["team"]["id"], "name": away["team"]["displayName"], "abbreviation": away["team"].get("abbreviation", "")},
                        "home_score": int(home.get("score", 0)),
                        "away_score": int(away.get("score", 0)),
                        "neutral_site": comp.get("neutralSite", False),
                        "pbp_available": comp.get("playByPlayAvailable", False),
                    }
                else:
                    record = process_game(event)
                    time.sleep(args.sleep)

                if record:
                    fout.write(json.dumps(record) + "\n")
                    fout.flush()
                    day_games += 1
                    if record.get("run_events"):
                        day_pbp += 1
                    if record.get("boxscore"):
                        total_boxscore += 1

            total_games += day_games
            total_pbp += day_pbp
            if day_games:
                print(f"{dt}: {day_games} games ({day_pbp} PBP) | cumulative: {total_games} games, {total_pbp} PBP")

            time.sleep(0.2)  # Brief pause between days

    print(f"\nDone: {total_games} games, {total_pbp} with PBP, {total_boxscore} with boxscore -> {args.out}")


if __name__ == "__main__":
    main()
