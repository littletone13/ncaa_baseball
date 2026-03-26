"""
Scrape full pitching boxscores from NCAA API (ncaa-api.henrygd.me).

This API wraps ncaa.com and provides ~100% boxscore coverage for all D1 games,
including per-pitcher: IP, H, R, ER, BB, K, BF, strikes, starter flag, W/L/S.

ESPN only has ~34% boxscore coverage. This fills the gap.

Flow:
  1. Fetch scoreboard for each date -> list of game IDs
  2. For each game ID, fetch /game/{id}/boxscore
  3. Extract per-pitcher stats with team mapping
  4. Output JSONL (one game per line) with full pitching arrays

Usage:
  python3 scripts/scrape_ncaa_boxscores.py --start 2026-02-14 --end 2026-03-08 --out data/raw/ncaa/boxscores_2026.jsonl
  python3 scripts/scrape_ncaa_boxscores.py --start 2026-02-14 --end 2026-03-08 --out data/raw/ncaa/boxscores_2026.jsonl --resume

Rate limit: NCAA API allows 5 req/s. We use 0.25s delay between game requests.
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

BASE = "https://ncaa-api.henrygd.me"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"


def fetch_json(url: str, retries: int = 3, delay: float = 0.3) -> dict | None:
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


def get_game_ids_for_date(dt: date) -> list[dict]:
    """Fetch all D1 baseball game IDs for a given date."""
    url = f"{BASE}/scoreboard/baseball/d1/{dt.year}/{dt.month:02d}/{dt.day:02d}"
    data = fetch_json(url)
    if not data:
        return []
    games = data.get("games", [])
    results = []
    for raw in games:
        # Scoreboard wraps each game under a "game" key
        g = raw.get("game", raw)
        gid = g.get("gameID")
        if not gid:
            continue
        home = g.get("home", {})
        away = g.get("away", {})
        # names.full is often empty; prefer short, then char6
        home_names = home.get("names", {})
        away_names = away.get("names", {})
        results.append({
            "game_id": str(gid),
            "home_name": home_names.get("short", "") or home_names.get("full", "") or home_names.get("char6", ""),
            "home_short": home_names.get("short", "") or home_names.get("char6", ""),
            "away_name": away_names.get("short", "") or away_names.get("full", "") or away_names.get("char6", ""),
            "away_short": away_names.get("short", "") or away_names.get("char6", ""),
            "home_score": home.get("score"),
            "away_score": away.get("score"),
            "status": g.get("gameState", "") or g.get("currentPeriod", ""),
        })
    return results


def parse_ip(ip_str: str) -> float:
    """Parse innings pitched: '5' -> 5.0, '5.1' -> 5.333, '5.2' -> 5.667."""
    try:
        parts = str(ip_str).split(".")
        whole = int(parts[0])
        if len(parts) > 1:
            thirds = int(parts[1])
            return whole + thirds / 3.0
        return float(whole)
    except (ValueError, IndexError):
        return 0.0


def extract_pitching(boxscore: dict, game_info: dict) -> dict | None:
    """Extract per-pitcher stats from NCAA API boxscore response."""
    team_boxscores = boxscore.get("teamBoxscore", [])
    teams_info = boxscore.get("teams", [])

    # Build team_id -> team info map
    # NOTE: Normalize teamId to str because teams[] returns str IDs
    #       but teamBoxscore[] returns int IDs — type mismatch breaks dict lookup.
    team_map = {}
    for t in teams_info:
        tid = t.get("teamId")
        if tid:
            team_map[str(tid)] = {
                "name": t.get("nameFull", ""),
                "short": t.get("nameShort", "") or t.get("name6Char", ""),
                "is_home": t.get("isHome", False),
            }

    result = {
        "game_id": game_info["game_id"],
        "date": game_info.get("date", ""),
        "home_team": game_info.get("home_name", ""),
        "away_team": game_info.get("away_name", ""),
        "home_score": game_info.get("home_score"),
        "away_score": game_info.get("away_score"),
        "pitching": {"home": [], "away": []},
    }

    for tb in team_boxscores:
        tid = str(tb.get("teamId", ""))
        tinfo = team_map.get(tid, {})
        side = "home" if tinfo.get("is_home") else "away"

        for p in tb.get("playerStats", []):
            ps = p.get("pitcherStats")
            if ps is None:
                continue

            ip_raw = ps.get("inningsPitched", "0")
            pitcher = {
                "name": f"{p.get('firstName', '')} {p.get('lastName', '')}".strip(),
                "position": p.get("position", ""),
                "number": p.get("number"),
                "starter": p.get("starter", False),
                "ip": parse_ip(ip_raw),
                "ip_raw": str(ip_raw),
                "h": int(ps.get("hitsAllowed", 0) or 0),
                "r": int(ps.get("runsAllowed", 0) or 0),
                "er": int(ps.get("earnedRunsAllowed", 0) or 0),
                "bb": int(ps.get("walksAllowed", 0) or 0),
                "k": int(ps.get("strikeouts", 0) or 0),
                "bf": int(ps.get("battersFaced", 0) or 0),
                "strikes": int(ps.get("strikes", 0) or 0),
                "win": int(ps.get("win", 0) or 0),
                "loss": int(ps.get("loss", 0) or 0),
                "save": int(ps.get("save", 0) or 0),
            }
            result["pitching"][side].append(pitcher)

    # Only return if we actually got pitching data
    total_pitchers = len(result["pitching"]["home"]) + len(result["pitching"]["away"])
    if total_pitchers == 0:
        return None
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape full pitching boxscores from NCAA API for all D1 games.",
    )
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--resume", action="store_true", help="Skip games already in output file")
    parser.add_argument("--delay", type=float, default=0.25, help="Delay between API calls (seconds)")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Load already-scraped game IDs for --resume
    scraped_ids: set[str] = set()
    if args.resume and args.out.exists():
        with open(args.out) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    scraped_ids.add(str(d.get("game_id", "")))
                except json.JSONDecodeError:
                    pass
        print(f"Resume mode: {len(scraped_ids)} games already scraped.")

    total_games = 0
    total_with_pitching = 0
    total_pitchers = 0
    dt = start

    # Always append — never truncate the historical boxscore file
    with open(args.out, "a") as outf:
        while dt <= end:
            games = get_game_ids_for_date(dt)
            if games:
                print(f"{dt}: {len(games)} games", end="", flush=True)
            else:
                dt += timedelta(days=1)
                continue

            day_pitching = 0
            for g in games:
                total_games += 1
                gid = g["game_id"]
                if gid in scraped_ids:
                    continue

                time.sleep(args.delay)
                box_url = f"{BASE}/game/{gid}/boxscore"
                box = fetch_json(box_url)
                if not box:
                    continue

                g["date"] = dt.isoformat()
                result = extract_pitching(box, g)
                if result:
                    total_with_pitching += 1
                    day_pitching += 1
                    n_p = len(result["pitching"]["home"]) + len(result["pitching"]["away"])
                    total_pitchers += n_p
                    outf.write(json.dumps(result) + "\n")

            print(f" -> {day_pitching} with pitching data")
            dt += timedelta(days=1)

    print(f"\nDone: {total_games} games, {total_with_pitching} with pitching ({total_with_pitching/max(1,total_games)*100:.1f}%)")
    print(f"Total pitcher appearances: {total_pitchers}")
    print(f"Output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
