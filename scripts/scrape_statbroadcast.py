#!/usr/bin/env python3
"""
scrape_statbroadcast.py — Scrape starting lineups from StatBroadcast.

StatBroadcast pages contain pre-game lineups with batting order, positions,
batter handedness (L/R/B), and the starting pitcher (spot 10, pos=p).

Lineups are available ~30-90 min before first pitch.

URL pattern: https://stats.statbroadcast.com/broadcast/?id={game_id}
Game IDs are linked from D1Baseball scores page.

This scraper uses Playwright to render the JavaScript-heavy page, switches
to the Lineups view, and parses the text output.

Usage:
  # Scrape specific StatBroadcast game IDs
  python3 scripts/scrape_statbroadcast.py --ids 631754,629941,628429

  # Scrape all games for a date (reads IDs from schedule or D1B)
  python3 scripts/scrape_statbroadcast.py --date 2026-04-02

  # Output to starter overrides
  python3 scripts/scrape_statbroadcast.py --ids 631754 --override-csv data/daily/2026-04-02/starter_overrides.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401


def parse_lineup_text(text: str) -> dict:
    """Parse StatBroadcast lineup text into structured data.

    Returns dict with:
        home_team, away_team, home_sp, away_sp, home_sp_hand, away_sp_hand,
        home_lineup (list of dicts), away_lineup (list of dicts)
    """
    result = {
        "home_team": "", "away_team": "",
        "home_sp": "", "away_sp": "",
        "home_sp_hand": "", "away_sp_hand": "",
        "home_lineup": [], "away_lineup": [],
    }

    # Split into lineup sections
    # Pattern: "{TEAM} Line Up\nSpot\tPos\t# Player\tBats..."
    lineup_pattern = r"(\w[\w\s.&'()-]*?) Line Up\s*\n\s*Spot\s+Pos\s+# Player\s+Bats.*?\n((?:\d+\t.*\n?)*)"
    matches = re.findall(lineup_pattern, text, re.MULTILINE)

    lineups = []
    for team_name, lineup_block in matches:
        team_name = team_name.strip()
        players = []
        sp_name = ""
        sp_hand = ""

        for line in lineup_block.strip().split("\n"):
            # Parse: "1\t3b\t27 Judd Utermark\tR\t0-0\t.299"
            parts = line.split("\t")
            if len(parts) < 4:
                continue

            spot = parts[0].strip()
            pos = parts[1].strip().lower()
            player_field = parts[2].strip()
            bats = parts[3].strip() if len(parts) > 3 else ""

            # Extract player number and name
            num_match = re.match(r"(\d+)\s+(.*)", player_field)
            if num_match:
                number = num_match.group(1)
                name = num_match.group(2).strip()
            else:
                number = ""
                name = player_field

            player = {
                "spot": int(spot) if spot.isdigit() else 0,
                "pos": pos,
                "number": number,
                "name": name,
                "bats": bats,
            }
            players.append(player)

            # Starting pitcher is position 'p' (typically spot 10)
            if pos == "p":
                sp_name = name
                sp_hand = bats  # R/L/B

        lineups.append({
            "team": team_name,
            "players": players,
            "sp_name": sp_name,
            "sp_hand": sp_hand,
        })

    # Extract full team names if available (from StatBroadcast event object)
    full_away_match = re.search(r"FULL_AWAY=(.+)", text)
    full_home_match = re.search(r"FULL_HOME=(.+)", text)
    full_away = full_away_match.group(1).strip() if full_away_match else ""
    full_home = full_home_match.group(1).strip() if full_home_match else ""

    if len(lineups) >= 2:
        # First lineup is away (visitor), second is home
        result["away_team"] = full_away or lineups[0]["team"]
        result["home_team"] = full_home or lineups[1]["team"]
        result["away_sp"] = lineups[0]["sp_name"]
        result["home_sp"] = lineups[1]["sp_name"]
        result["away_sp_hand"] = lineups[0]["sp_hand"]
        result["home_sp_hand"] = lineups[1]["sp_hand"]
        result["away_lineup"] = lineups[0]["players"]
        result["home_lineup"] = lineups[1]["players"]
    elif len(lineups) == 1:
        result["away_team"] = lineups[0]["team"]
        result["away_sp"] = lineups[0]["sp_name"]
        result["away_sp_hand"] = lineups[0]["sp_hand"]
        result["away_lineup"] = lineups[0]["players"]

    return result


def scrape_statbroadcast_ids(
    game_ids: list[str],
    headless: bool = True,
) -> list[dict]:
    """Scrape lineups from StatBroadcast for given game IDs.

    Uses Playwright to render the JS-heavy pages.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright not installed. Run: pip install playwright && playwright install chromium",
              file=sys.stderr)
        return []

    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        )
        page = context.new_page()

        for gid in game_ids:
            url = f"https://stats.statbroadcast.com/broadcast/?id={gid}"
            print(f"  Scraping {url}...", file=sys.stderr)

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=15000)
                # Wait for the proof-of-work challenge to complete
                page.wait_for_timeout(5000)

                # Check if page loaded (title should have team names)
                title = page.title()
                if "statbroadcast" in title.lower() and "at" not in title.lower():
                    # Still on loading page — wait more
                    page.wait_for_timeout(5000)
                    title = page.title()

                # Switch to Lineups view
                page.evaluate("window.statBroadcast && window.statBroadcast.Stats && window.statBroadcast.Stats.setStatPage('lineups')")
                page.wait_for_timeout(2000)

                # Get the lineup text + full team names from the StatBroadcast event object
                text = page.evaluate("""
                    (() => {
                        const el = document.querySelector('#stats') || document.body;
                        let text = el.innerText;
                        // Prepend full team names from the event object for matching
                        if (window.statBroadcast && window.statBroadcast.Event) {
                            const ev = window.statBroadcast.Event;
                            text = 'FULL_AWAY=' + (ev.visitorname || '') + '\\n' +
                                   'FULL_HOME=' + (ev.homename || '') + '\\n' + text;
                        }
                        return text;
                    })()
                """)

                if not text or "Line Up" not in text:
                    print(f"    No lineup data (game may not have started or lineups not submitted)",
                          file=sys.stderr)
                    continue

                parsed = parse_lineup_text(text)
                parsed["statbroadcast_id"] = gid
                parsed["url"] = url

                if parsed["home_sp"] or parsed["away_sp"]:
                    hand_h = f"({parsed['home_sp_hand']}HP)" if parsed["home_sp_hand"] else ""
                    hand_a = f"({parsed['away_sp_hand']}HP)" if parsed["away_sp_hand"] else ""
                    print(f"    {parsed['away_team']} @ {parsed['home_team']}: "
                          f"{parsed['away_sp']} {hand_a} vs {parsed['home_sp']} {hand_h}",
                          file=sys.stderr)
                    results.append(parsed)
                else:
                    print(f"    {title}: lineups loaded but no SP found", file=sys.stderr)

            except Exception as e:
                print(f"    Error: {e}", file=sys.stderr)

            time.sleep(1)  # rate limit

        browser.close()

    return results


def write_overrides(
    results: list[dict],
    schedule_csv: Path,
    override_csv: Path,
) -> int:
    """Write scraped starters to override CSV, matching to game numbers."""
    import pandas as pd

    if not results:
        return 0

    sched = pd.read_csv(schedule_csv, dtype=str)

    # Build lookup via canonical registry — exact match only, no fuzzy.
    # StatBroadcast full team names → canonical_id → schedule game_num
    canon_csv = Path("data/registries/canonical_teams_2026.csv")
    # StatBroadcast name → canonical_id: exact match from all registry name columns.
    # Also loads statbroadcast_names.csv for known aliases that differ from registry.
    name_to_cid: dict[str, str] = {}
    if canon_csv.exists():
        canon = pd.read_csv(canon_csv, dtype=str)
        for _, cr in canon.iterrows():
            cid = str(cr.get("canonical_id", "")).strip()
            for col in ("team_name", "odds_api_name", "espn_name"):
                n = str(cr.get(col, "")).strip()
                if n and n != "nan":
                    name_to_cid[n.lower()] = cid

    # Load StatBroadcast-specific aliases (for names that differ from canonical registry)
    sb_names_csv = Path("data/registries/statbroadcast_names.csv")
    if sb_names_csv.exists():
        with open(sb_names_csv) as f:
            for row in csv.DictReader(f):
                sb_name = str(row.get("statbroadcast_name", "")).strip()
                cid = str(row.get("canonical_id", "")).strip()
                if sb_name and cid:
                    name_to_cid[sb_name.lower()] = cid

    # Schedule lookup: (home_cid, away_cid) → game_num
    cid_to_game: dict[tuple[str, str], str] = {}
    h_col = "home_cid" if "home_cid" in sched.columns else "home_canonical_id"
    a_col = "away_cid" if "away_cid" in sched.columns else "away_canonical_id"
    for _, g in sched.iterrows():
        hc = str(g[h_col]).strip()
        ac = str(g[a_col]).strip()
        gn = str(g["game_num"])
        cid_to_game[(hc, ac)] = gn

    # Load existing overrides
    existing = {}
    if override_csv.exists():
        with open(override_csv) as f:
            for row in csv.DictReader(f):
                key = (row["game_num"], row["side"])
                existing[key] = row

    new_count = 0
    for r in results:
        h_name = r["home_team"].lower().strip()
        a_name = r["away_team"].lower().strip()

        # Resolve StatBroadcast team names → canonical_id via exact registry match
        h_cid = name_to_cid.get(h_name, "")
        a_cid = name_to_cid.get(a_name, "")

        if not h_cid:
            print(f"  WARNING: '{r['home_team']}' not in canonical registry — add mapping",
                  file=sys.stderr)
            continue
        if not a_cid:
            print(f"  WARNING: '{r['away_team']}' not in canonical registry — add mapping",
                  file=sys.stderr)
            continue

        # Look up game_num from schedule via canonical_id pair
        game_num = cid_to_game.get((h_cid, a_cid))
        if not game_num:
            print(f"  WARNING: {a_cid} @ {h_cid} not in schedule for this date",
                  file=sys.stderr)
            continue

        # Add overrides
        if r["home_sp"]:
            key = (game_num, "home")
            if key not in existing:
                existing[key] = {
                    "game_num": game_num,
                    "side": "home",
                    "pitcher_name": r["home_sp"],
                    "source": "statbroadcast",
                }
                new_count += 1

        if r["away_sp"]:
            key = (game_num, "away")
            if key not in existing:
                existing[key] = {
                    "game_num": game_num,
                    "side": "away",
                    "pitcher_name": r["away_sp"],
                    "source": "statbroadcast",
                }
                new_count += 1

    # Write merged overrides
    override_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(override_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["game_num", "side", "pitcher_name", "source"])
        w.writeheader()
        for key in sorted(existing.keys()):
            w.writerow(existing[key])

    print(f"  Wrote {len(existing)} overrides ({new_count} new) → {override_csv}",
          file=sys.stderr)
    return new_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape StatBroadcast lineups.")
    parser.add_argument("--ids", help="Comma-separated StatBroadcast game IDs")
    parser.add_argument("--date", help="Game date YYYY-MM-DD (auto-discover IDs)")
    parser.add_argument("--schedule", type=Path, help="Schedule CSV for game number matching")
    parser.add_argument("--override-csv", type=Path, help="Output override CSV path")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", action="store_true")
    args = parser.parse_args()

    if args.no_headless:
        args.headless = False

    game_ids = []
    if args.ids:
        game_ids = [gid.strip() for gid in args.ids.split(",")]
    elif args.date:
        # TODO: auto-discover game IDs from D1B scores page
        print(f"Auto-discovery not yet implemented. Use --ids.", file=sys.stderr)
        return 1
    else:
        print("Provide --ids or --date", file=sys.stderr)
        return 1

    results = scrape_statbroadcast_ids(game_ids, headless=args.headless)

    if results and args.override_csv:
        schedule = args.schedule or Path(f"data/daily/{args.date}/schedule.csv")
        if schedule.exists():
            write_overrides(results, schedule, args.override_csv)
        else:
            print(f"  No schedule at {schedule} — can't write overrides", file=sys.stderr)

    # Print summary
    for r in results:
        h = f"({r['home_sp_hand']}HP)" if r["home_sp_hand"] else ""
        a = f"({r['away_sp_hand']}HP)" if r["away_sp_hand"] else ""
        print(f"{r['away_team']} @ {r['home_team']}: {r['away_sp']} {a} vs {r['home_sp']} {h}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
