"""
Scrape inning-by-inning linescores from NCAA API for existing boxscore games.

Uses the /game/{id} endpoint (NOT /game/{id}/boxscore) which returns
linescores with per-inning runs for home and away teams.

Reads game IDs from existing boxscores_2026.jsonl, fetches linescores,
and outputs an augmented JSONL with linescore data added.

Usage:
  python3 scripts/scrape_ncaa_linescores.py
  python3 scripts/scrape_ncaa_linescores.py --resume  # skip already-fetched games
  python3 scripts/scrape_ncaa_linescores.py --input data/raw/ncaa/boxscores_2026.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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


def extract_linescores(game_data: dict) -> dict | None:
    """
    Extract inning-by-inning linescores from NCAA /game/{id} response.

    Returns dict with:
      home_innings: [runs_inn1, runs_inn2, ...] (integers)
      away_innings: [runs_inn1, runs_inn2, ...]
      home_hits: total hits
      away_hits: total hits
      home_errors: total errors
      away_errors: total errors
      n_innings: number of innings played
      quality: "full" | "partial_home_zeros" | "partial_away_zeros" | "both_zeros"
    """
    contests = game_data.get("contests", [])
    if not contests:
        return None

    contest = contests[0]
    linescores = contest.get("linescores", [])
    if not linescores:
        return None

    # Parse linescores: numbered periods are innings, "R"/"H"/"E" are totals
    home_innings = []
    away_innings = []
    home_r = home_h = home_e = 0
    away_r = away_h = away_e = 0

    for ls in linescores:
        period = str(ls.get("period", ""))
        home_val = str(ls.get("home", "0"))
        away_val = str(ls.get("visit", "0"))

        if period == "R":
            home_r = _safe_int(home_val)
            away_r = _safe_int(away_val)
        elif period == "H":
            home_h = _safe_int(home_val)
            away_h = _safe_int(away_val)
        elif period == "E":
            home_e = _safe_int(home_val)
            away_e = _safe_int(away_val)
        elif period.isdigit():
            home_innings.append(_safe_int(home_val))
            away_innings.append(_safe_int(away_val))

    if not home_innings:
        return None

    n_innings = len(home_innings)

    # Assess quality: do inning sums match R totals?
    home_sum = sum(home_innings)
    away_sum = sum(away_innings)
    home_ok = (home_sum == home_r)
    away_ok = (away_sum == away_r)

    if home_ok and away_ok:
        quality = "full"
    elif home_ok and not away_ok:
        quality = "partial_away_zeros"
    elif not home_ok and away_ok:
        quality = "partial_home_zeros"
    else:
        quality = "both_zeros"

    return {
        "home_innings": home_innings,
        "away_innings": away_innings,
        "home_runs_total": home_r,
        "away_runs_total": away_r,
        "home_hits": home_h,
        "away_hits": away_h,
        "home_errors": home_e,
        "away_errors": away_e,
        "n_innings": n_innings,
        "quality": quality,
    }


def _safe_int(v) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape inning-by-inning linescores from NCAA API.",
    )
    parser.add_argument("--input", type=Path,
                        default=Path("data/raw/ncaa/boxscores_2026.jsonl"),
                        help="Input boxscores JSONL (for game IDs)")
    parser.add_argument("--out", type=Path,
                        default=Path("data/raw/ncaa/linescores_2026.jsonl"),
                        help="Output linescores JSONL")
    parser.add_argument("--resume", action="store_true",
                        help="Skip games already in output file")
    parser.add_argument("--delay", type=float, default=0.25,
                        help="Delay between API calls (seconds)")
    args = parser.parse_args()

    # Load game IDs from boxscores
    game_ids = []
    game_meta = {}
    with args.input.open() as f:
        for line in f:
            try:
                g = json.loads(line.strip())
                gid = str(g.get("game_id", ""))
                if gid:
                    game_ids.append(gid)
                    game_meta[gid] = {
                        "date": g.get("date", ""),
                        "home_team": g.get("home_team", ""),
                        "away_team": g.get("away_team", ""),
                        "home_score": g.get("home_score"),
                        "away_score": g.get("away_score"),
                    }
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(game_ids)} game IDs from {args.input}")

    # Load already-scraped for --resume
    scraped_ids: set[str] = set()
    if args.resume and args.out.exists():
        with args.out.open() as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    scraped_ids.add(str(d.get("game_id", "")))
                except json.JSONDecodeError:
                    pass
        print(f"Resume mode: {len(scraped_ids)} games already scraped.")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    fetched = 0
    quality_counts = {"full": 0, "partial_home_zeros": 0,
                      "partial_away_zeros": 0, "both_zeros": 0, "failed": 0}

    with args.out.open("a" if args.resume else "w") as outf:
        for i, gid in enumerate(game_ids):
            if gid in scraped_ids:
                continue

            total += 1
            if total > 1:
                time.sleep(args.delay)

            url = f"{BASE}/game/{gid}"
            data = fetch_json(url)

            if data is None:
                quality_counts["failed"] += 1
                if total % 100 == 0:
                    print(f"  [{total}/{len(game_ids)}] {gid}: API failed")
                continue

            ls = extract_linescores(data)
            if ls is None:
                quality_counts["failed"] += 1
                continue

            fetched += 1
            quality_counts[ls["quality"]] += 1

            meta = game_meta.get(gid, {})
            record = {
                "game_id": gid,
                "date": meta.get("date", ""),
                "home_team": meta.get("home_team", ""),
                "away_team": meta.get("away_team", ""),
                "home_score": meta.get("home_score"),
                "away_score": meta.get("away_score"),
                **ls,
            }
            outf.write(json.dumps(record) + "\n")

            if total % 200 == 0:
                print(f"  [{total}/{len(game_ids)}] fetched={fetched}, "
                      f"full={quality_counts['full']}, "
                      f"partial={quality_counts['partial_home_zeros']+quality_counts['partial_away_zeros']}, "
                      f"failed={quality_counts['failed']}")

    print(f"\n=== NCAA Linescore Scraping Complete ===")
    print(f"Total attempted: {total}")
    print(f"Successfully fetched: {fetched}")
    print(f"Quality breakdown:")
    for q, n in quality_counts.items():
        print(f"  {q}: {n}")
    print(f"Output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
