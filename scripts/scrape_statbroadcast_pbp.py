#!/usr/bin/env python3
"""
scrape_statbroadcast_pbp.py — Scrape run_events from StatBroadcast completed games.

Extracts per-half-inning run counts with pitcher attribution from the Scoring
Summary + linescore views. Converts to the same run_event format used by the
Stan model (run_1, run_2, run_3, run_4 per team per game).

Non-headless Playwright required (StatBroadcast blocks headless with proof-of-work).

Usage:
  # Single game
  python3 scripts/scrape_statbroadcast_pbp.py --ids 631754

  # Batch from file (one ID per line)
  python3 scripts/scrape_statbroadcast_pbp.py --id-file data/statbroadcast_game_ids.txt

  # Output
  python3 scripts/scrape_statbroadcast_pbp.py --ids 631754,629941 --out data/processed/run_events_statbroadcast.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401


def parse_linescore(text: str) -> dict:
    """Parse the linescore table from StatBroadcast text.

    Returns dict with:
        away_team, home_team, away_runs_by_inning, home_runs_by_inning,
        away_total, home_total
    """
    result = {"away_team": "", "home_team": "", "away_inning_runs": [], "home_inning_runs": []}

    # Find linescore: TEAM  1  2  3  ...  R  H  E  L
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "\tTEAM\t" in line:
            # Next two lines are away and home
            if i + 2 < len(lines):
                away_line = lines[i + 1]
                home_line = lines[i + 2]

                away_parts = away_line.split("\t")
                home_parts = home_line.split("\t")

                # First non-empty part is team abbreviation
                result["away_team"] = away_parts[1].strip() if len(away_parts) > 1 else ""
                result["home_team"] = home_parts[1].strip() if len(home_parts) > 1 else ""

                # Extract per-inning runs (between team name and R/H/E totals)
                # The format has empty tabs for separation before R H E L
                for part in away_parts[2:]:
                    part = part.strip()
                    if part == "" or part == "R":
                        break
                    try:
                        result["away_inning_runs"].append(int(part))
                    except ValueError:
                        break

                for part in home_parts[2:]:
                    part = part.strip()
                    if part == "" or part == "R":
                        break
                    try:
                        result["home_inning_runs"].append(int(part))
                    except ValueError:
                        break

                break

    result["away_total"] = sum(result["away_inning_runs"])
    result["home_total"] = sum(result["home_inning_runs"])
    return result


def parse_scoring_summary(text: str) -> list[dict]:
    """Parse the Scoring Summary table from StatBroadcast.

    Each scoring play has: Team, Inn, Scoring Dec., Play, Batter, Pitcher, Outs
    Returns list of scoring play dicts.
    """
    plays = []

    # Find lines that match scoring play format:
    # \tTop 3\t1B MI 1RBI\tC. Goldin singled...\tC. Goldin\tPeterson\t2
    # Or: \tBot 1\t1B 8 1RBI\tSurowiec singled...\tSurowiec\tT. Rabe\t1
    for line in text.split("\n"):
        parts = line.split("\t")
        # Look for lines with inning indicator (Top/Bot N)
        for j, part in enumerate(parts):
            match = re.match(r"(Top|Bot)\s+(\d+)", part.strip())
            if match:
                half = "top" if match.group(1) == "Top" else "bottom"
                inning = int(match.group(2))

                # Extract RBI count from scoring decision
                scoring_dec = parts[j + 1].strip() if j + 1 < len(parts) else ""
                rbi_match = re.search(r"(\d+)RBI", scoring_dec)
                rbi = int(rbi_match.group(1)) if rbi_match else 1

                # Batter and pitcher are near the end
                pitcher = ""
                batter = ""
                if j + 4 < len(parts):
                    batter = parts[j + 3].strip()
                    pitcher = parts[j + 4].strip()
                elif j + 3 < len(parts):
                    batter = parts[j + 2].strip()
                    pitcher = parts[j + 3].strip()

                plays.append({
                    "half": half,
                    "inning": inning,
                    "scoring_dec": scoring_dec,
                    "rbi": rbi,
                    "batter": batter,
                    "pitcher": pitcher,
                })
                break

    return plays


def parse_pitcher_stats(box_text: str, team_label: str) -> list[dict]:
    """Parse pitcher stats from box score to get IP per pitcher."""
    pitchers = []
    # Look for pitcher lines: they have IP as a number with possible .1/.2
    # Format: p  #  Name  IP  H  R  ER  BB  K ...
    in_pitching = False
    for line in box_text.split("\n"):
        if "Pitching" in line or "PITCHER" in line.upper():
            in_pitching = True
            continue
        if in_pitching:
            parts = line.split("\t")
            if len(parts) >= 6:
                try:
                    name = parts[0].strip() or parts[1].strip()
                    ip_str = ""
                    for p in parts[1:]:
                        p = p.strip()
                        try:
                            float(p)
                            ip_str = p
                            break
                        except ValueError:
                            if p and not p.startswith("#"):
                                name = p
                    if ip_str:
                        pitchers.append({"name": name, "ip": ip_str})
                except (ValueError, IndexError):
                    pass
    return pitchers


def build_run_events_from_game(
    linescore: dict,
    scoring_plays: list[dict],
    game_id: str,
    game_date: str,
    home_team_full: str,
    away_team_full: str,
) -> list[dict]:
    """Convert StatBroadcast data to run_event format.

    For each half-inning, count total runs scored and attribute to the pitcher
    on the mound. This produces run_1/run_2/run_3/run_4 events per team.
    """
    # Build per-half-inning run counts from linescore
    away_innings = linescore["away_inning_runs"]
    home_innings = linescore["home_inning_runs"]

    # Build pitcher-per-inning mapping from scoring plays
    # For each half-inning, find the pitcher(s) who gave up runs
    pitcher_by_hi = {}  # (half, inning) → primary pitcher
    for play in scoring_plays:
        key = (play["half"], play["inning"])
        if key not in pitcher_by_hi and play["pitcher"]:
            pitcher_by_hi[key] = play["pitcher"]

    # Build run_events
    events = []

    # Away team batting (top of innings) — home pitcher on mound
    for i, runs in enumerate(away_innings):
        inning = i + 1
        pitcher = pitcher_by_hi.get(("top", inning), "")
        events.append({
            "game_id": f"SB_{game_id}",
            "game_date": game_date,
            "season": game_date[:4] if game_date else "",
            "batting_team": away_team_full,
            "pitching_team": home_team_full,
            "pitcher_name": pitcher,
            "inning": inning,
            "half": "top",
            "runs_scored": runs,
            "source": "statbroadcast",
        })

    # Home team batting (bottom of innings) — away pitcher on mound
    for i, runs in enumerate(home_innings):
        inning = i + 1
        pitcher = pitcher_by_hi.get(("bottom", inning), "")
        events.append({
            "game_id": f"SB_{game_id}",
            "game_date": game_date,
            "season": game_date[:4] if game_date else "",
            "batting_team": home_team_full,
            "pitching_team": away_team_full,
            "pitcher_name": pitcher,
            "inning": inning,
            "half": "bottom",
            "runs_scored": runs,
            "source": "statbroadcast",
        })

    return events


def scrape_game_pbp(page, game_id: str) -> dict | None:
    """Scrape a single StatBroadcast game for PBP/scoring data.

    Returns dict with linescore, scoring_plays, run_events, team names.
    """
    url = f"https://stats.statbroadcast.com/broadcast/?id={game_id}"

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=20000)
        time.sleep(8)  # proof-of-work challenge — needs 8+ seconds

        title = page.title()
        # If still showing the challenge page, wait longer and retry
        if "StatBroadcast Live" in title or "statbroadcast" in title.lower():
            time.sleep(8)
            title = page.title()

        if "403" in title or "Forbidden" in title:
            print(f"  {game_id}: 403 blocked", file=sys.stderr)
            return None

        if "StatBroadcast Live" in title or "statbroadcast" in title.lower():
            print(f"  {game_id}: challenge timeout (try again)", file=sys.stderr)
            return None

        # Check for completed game — look for score pattern or "Final"
        has_score = bool(re.search(r"\d+.*\d+", title))
        is_final = "final" in title.lower()
        is_pregame = "pregame" in title.lower()

        if is_pregame:
            print(f"  {game_id}: {title} (pregame)", file=sys.stderr)
            return None

        if not is_final and not has_score:
            print(f"  {game_id}: {title} (not final)", file=sys.stderr)
            return None

        # Get full team names from JS event object
        names = page.evaluate("""
            (() => {
                const ev = window.statBroadcast && window.statBroadcast.Event;
                return ev ? {home: ev.homename || '', away: ev.visitorname || ''} : {};
            })()
        """)
        home_name = names.get("home", "")
        away_name = names.get("away", "")

        # Switch to Scoring view
        page.evaluate("window.statBroadcast.Stats.setStatPage('scoring')")
        time.sleep(2)

        scoring_text = page.evaluate(
            '(document.querySelector("#stats") || document.body).innerText'
        )

        # Parse linescore and scoring summary
        linescore = parse_linescore(scoring_text)
        scoring_plays = parse_scoring_summary(scoring_text)

        if not linescore["away_inning_runs"]:
            print(f"  {game_id}: no linescore data", file=sys.stderr)
            return None

        # Extract game date from page if possible
        game_date = ""
        date_match = re.search(r"(\w+ \d+, \d{4})", scoring_text)
        if date_match:
            from datetime import datetime
            try:
                dt = datetime.strptime(date_match.group(1), "%B %d, %Y")
                game_date = dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

        # Build run events
        run_events = build_run_events_from_game(
            linescore, scoring_plays, game_id, game_date,
            home_name or linescore["home_team"],
            away_name or linescore["away_team"],
        )

        total_runs = sum(e["runs_scored"] for e in run_events)
        runs_with_pitcher = sum(1 for e in run_events if e["pitcher_name"] and e["runs_scored"] > 0)
        total_scoring_hi = sum(1 for e in run_events if e["runs_scored"] > 0)

        print(f"  {game_id}: {away_name} @ {home_name} — "
              f"{linescore['away_total']}-{linescore['home_total']}, "
              f"{len(run_events)} half-innings, "
              f"{runs_with_pitcher}/{total_scoring_hi} scoring HIs have pitcher",
              file=sys.stderr)

        return {
            "game_id": game_id,
            "home_team": home_name,
            "away_team": away_name,
            "linescore": linescore,
            "scoring_plays": scoring_plays,
            "run_events": run_events,
        }

    except Exception as e:
        print(f"  {game_id}: error — {e}", file=sys.stderr)
        return None


def convert_to_run_event_format(events: list[dict]) -> list[dict]:
    """Convert per-half-inning events to the run_1/run_2/run_3/run_4 format
    used by the Stan model (one row per team per game)."""
    from collections import defaultdict

    # Group by (game_id, batting_team)
    groups = defaultdict(lambda: {"run_1": 0, "run_2": 0, "run_3": 0, "run_4": 0})

    for e in events:
        key = (e["game_id"], e["game_date"], e["batting_team"], e["pitching_team"])
        runs = e["runs_scored"]
        if runs == 0:
            continue
        elif runs == 1:
            groups[key]["run_1"] += 1
        elif runs == 2:
            groups[key]["run_2"] += 1
        elif runs == 3:
            groups[key]["run_3"] += 1
        elif runs >= 4:
            groups[key]["run_4"] += 1

        # Store pitcher info for the highest-IP pitcher (starter usually)
        if "pitcher_name" not in groups[key] and e["pitcher_name"]:
            groups[key]["pitcher_name"] = e["pitcher_name"]

    rows = []
    for (gid, gdate, bat_team, pitch_team), counts in groups.items():
        rows.append({
            "game_id": gid,
            "game_date": gdate,
            "batting_team": bat_team,
            "pitching_team": pitch_team,
            "run_1": counts["run_1"],
            "run_2": counts["run_2"],
            "run_3": counts["run_3"],
            "run_4": counts["run_4"],
            "pitcher_name": counts.get("pitcher_name", ""),
            "source": "statbroadcast",
        })

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape StatBroadcast PBP for run_events.")
    parser.add_argument("--ids", help="Comma-separated StatBroadcast game IDs")
    parser.add_argument("--id-file", type=Path, help="File with one game ID per line")
    parser.add_argument("--out", type=Path, default=Path("data/processed/run_events_statbroadcast.csv"))
    parser.add_argument("--raw-out", type=Path, default=None, help="Save raw per-half-inning events")
    args = parser.parse_args()

    game_ids = []
    if args.ids:
        game_ids = [gid.strip() for gid in args.ids.split(",")]
    elif args.id_file and args.id_file.exists():
        game_ids = [line.strip() for line in args.id_file.read_text().splitlines() if line.strip()]
    else:
        print("Provide --ids or --id-file", file=sys.stderr)
        return 1

    print(f"Scraping {len(game_ids)} games...", file=sys.stderr)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("pip install playwright && playwright install chromium", file=sys.stderr)
        return 1

    all_events = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        for gid in game_ids:
            result = scrape_game_pbp(page, gid)
            if result:
                all_events.extend(result["run_events"])
            time.sleep(1)

        browser.close()

    if not all_events:
        print("No run events scraped.", file=sys.stderr)
        return 1

    # Save raw per-half-inning events
    if args.raw_out:
        args.raw_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.raw_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_events[0].keys()))
            w.writeheader()
            w.writerows(all_events)
        print(f"Wrote {len(all_events)} raw half-inning events → {args.raw_out}", file=sys.stderr)

    # Convert to run_event format (run_1/run_2/run_3/run_4 per team per game)
    run_events = convert_to_run_event_format(all_events)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(run_events[0].keys()))
        w.writeheader()
        w.writerows(run_events)

    n_games = len(set(e["game_id"] for e in run_events))
    n_with_pitcher = sum(1 for e in run_events if e["pitcher_name"])
    print(f"\nWrote {len(run_events)} run_event rows ({n_games} games, "
          f"{n_with_pitcher} with pitcher attribution) → {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
