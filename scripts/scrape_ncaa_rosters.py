#!/usr/bin/env python3
"""
scrape_ncaa_rosters.py — Scrape B/T data from stats.ncaa.org roster pages.

Uses Playwright headless browser to navigate stats.ncaa.org/teams/{team_id}/roster
which shows player bats/throws data. This works where direct HTTP (baseballr, etc.)
gets 403 Forbidden because Playwright handles cookies/JS/CSRF properly.

Usage:
  # Scrape only teams missing B/T data (default)
  python3 scripts/scrape_ncaa_rosters.py

  # Scrape specific teams
  python3 scripts/scrape_ncaa_rosters.py --teams NCAA_614570 NCAA_614577

  # Test with N teams
  python3 scripts/scrape_ncaa_rosters.py --limit 3

  # Force re-scrape all teams
  python3 scripts/scrape_ncaa_rosters.py --all

Output:
  data/processed/ncaa_rosters.csv
  Columns: canonical_id, player_name, jersey_number, position, bats, throws, year_class
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd


OUTPUT_CSV = Path("data/processed/ncaa_rosters.csv")


def _find_missing_teams(registry_csv: Path, canonical_csv: Path) -> list[tuple[str, int]]:
    """Find teams with zero B/T data in the player registry."""
    if not registry_csv.exists():
        print(f"Player registry not found: {registry_csv}", file=sys.stderr)
        return []

    reg = pd.read_csv(registry_csv)
    canon = pd.read_csv(canonical_csv)

    # Teams with zero throws data
    team_throws = reg.groupby("canonical_id")["throws"].apply(
        lambda x: x.isin(["L", "R"]).sum()
    )
    zero_teams = set(team_throws[team_throws == 0].index)

    # Get ncaa_teams_id for missing teams
    missing = canon[canon["canonical_id"].isin(zero_teams)].copy()
    missing = missing.dropna(subset=["ncaa_teams_id"])
    missing["ncaa_teams_id"] = missing["ncaa_teams_id"].astype(int)

    return list(zip(missing["canonical_id"], missing["ncaa_teams_id"]))


def scrape_ncaa_roster(page, team_id: int, retries: int = 2) -> list[dict]:
    """
    Scrape a stats.ncaa.org roster page for player B/T data.

    The page at stats.ncaa.org/teams/{team_id}/roster typically has a table
    with columns: #, Name, Pos, Ht, Wt, Yr, Bat, Thr (or similar).

    Returns list of {player_name, jersey_number, position, bats, throws, year_class}.
    """
    url = f"https://stats.ncaa.org/teams/{team_id}/roster"

    for attempt in range(retries + 1):
        try:
            resp = page.goto(url, wait_until="networkidle", timeout=30000)
            if resp and resp.status == 403:
                if attempt < retries:
                    time.sleep(3 + attempt * 2)
                    continue
                return []
            # Wait for table to render
            time.sleep(2)
            break
        except Exception as e:
            if attempt < retries:
                time.sleep(3 + attempt * 2)
                continue
            print(f" error: {e}", file=sys.stderr)
            return []

    players = []

    # Strategy 1: Parse the roster table directly
    # stats.ncaa.org uses standard HTML tables
    try:
        tables = page.query_selector_all("table")
        for table in tables:
            headers = []
            header_row = table.query_selector("thead tr") or table.query_selector("tr:first-child")
            if header_row:
                for th in header_row.query_selector_all("th, td"):
                    headers.append(th.inner_text().strip().lower())

            # Map header names to indices
            col_map = {}
            for i, h in enumerate(headers):
                h_clean = h.replace(".", "").strip()
                if h_clean in ("name", "player", "full name"):
                    col_map["name"] = i
                elif h_clean in ("#", "no", "no.", "number", "jersey"):
                    col_map["number"] = i
                elif h_clean in ("pos", "pos.", "position"):
                    col_map["pos"] = i
                elif h_clean in ("bat", "bats", "b", "b/t"):
                    col_map["bats"] = i
                elif h_clean in ("thr", "throw", "throws", "t"):
                    col_map["throws"] = i
                elif h_clean in ("yr", "yr.", "year", "cl", "cl.", "class"):
                    col_map["year"] = i
                elif h_clean == "b/t":
                    col_map["bt"] = i

            if "name" not in col_map:
                continue

            # Parse data rows
            rows = table.query_selector_all("tbody tr") or table.query_selector_all("tr")
            for row in rows:
                cells = row.query_selector_all("td")
                if len(cells) < 2:
                    continue

                def get_cell(key):
                    idx = col_map.get(key)
                    if idx is not None and idx < len(cells):
                        return cells[idx].inner_text().strip()
                    return ""

                name = get_cell("name")
                if not name or name.lower() in headers:
                    continue

                # Handle B/T combined column
                bt_val = get_cell("bt")
                if bt_val and "/" in bt_val:
                    parts = bt_val.split("/")
                    bats = parts[0].strip().upper()
                    throws = parts[1].strip().upper() if len(parts) > 1 else ""
                else:
                    bats = get_cell("bats").upper()
                    throws = get_cell("throws").upper()

                # Normalize
                if bats not in ("L", "R", "S", "B"):
                    bats = ""
                if throws not in ("L", "R"):
                    throws = ""

                pos = get_cell("pos")
                number = get_cell("number")
                year_class = get_cell("year")

                if name and (bats or throws):
                    players.append({
                        "player_name": name,
                        "jersey_number": number,
                        "position": pos,
                        "bats": bats,
                        "throws": throws,
                        "year_class": year_class,
                    })

            if players:
                break  # Found data in this table, no need to check others

    except Exception as e:
        print(f" table parse error: {e}", file=sys.stderr)

    # Strategy 2: Fallback — regex parse the page body text
    if not players:
        try:
            body = page.inner_text("body")
            lines = body.split("\n")

            # Look for B/T patterns like "R/R", "L/R", "S/R" etc.
            for i, line in enumerate(lines):
                line_s = line.strip()
                bt_match = re.match(r'^([RLSB])/([RL])$', line_s)
                if bt_match:
                    # Search backwards for name
                    name = ""
                    pos = ""
                    for j in range(i - 1, max(0, i - 10), -1):
                        candidate = lines[j].strip()
                        if not candidate:
                            continue
                        if re.match(r'^\d+$', candidate):
                            continue
                        if re.match(r"^\d+'\d+", candidate) or re.match(r'^\d+ lbs', candidate):
                            continue
                        if re.match(r'^(Fr\.|So\.|Jr\.|Sr\.|R-Fr|R-So|R-Jr|Gr)', candidate):
                            continue
                        if re.match(r'^(RHP|LHP|C|1B|2B|3B|SS|OF|DH|IF|P|UT|INF)', candidate):
                            pos = candidate
                            continue
                        # Name: at least two words starting with capital
                        if re.match(r'^[A-Z][a-zA-Z]+[ ,]+[A-Z]', candidate) and len(candidate) < 60:
                            name = candidate
                            break

                    if name:
                        players.append({
                            "player_name": name,
                            "jersey_number": "",
                            "position": pos,
                            "bats": bt_match.group(1),
                            "throws": bt_match.group(2),
                            "year_class": "",
                        })
        except Exception as e:
            print(f" fallback parse error: {e}", file=sys.stderr)

    return players


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape B/T data from stats.ncaa.org roster pages."
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=Path("data/registries/canonical_teams_2026.csv"),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("data/processed/player_registry.csv"),
    )
    parser.add_argument("--teams", nargs="+", help="Specific canonical_ids to scrape")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N teams")
    parser.add_argument("--all", action="store_true", help="Scrape all teams, not just missing")
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_CSV,
    )
    args = parser.parse_args()

    canon = pd.read_csv(args.canonical)

    if args.teams:
        # Scrape specific teams
        team_ids = canon[canon["canonical_id"].isin(args.teams)].copy()
        team_ids = team_ids.dropna(subset=["ncaa_teams_id"])
        teams_to_scrape = list(
            zip(team_ids["canonical_id"], team_ids["ncaa_teams_id"].astype(int))
        )
    elif args.all:
        # Scrape all teams
        all_teams = canon.dropna(subset=["ncaa_teams_id"]).copy()
        all_teams["ncaa_teams_id"] = all_teams["ncaa_teams_id"].astype(int)
        teams_to_scrape = list(zip(all_teams["canonical_id"], all_teams["ncaa_teams_id"]))
    else:
        # Default: only teams missing B/T data
        teams_to_scrape = _find_missing_teams(args.registry, args.canonical)

    if not teams_to_scrape:
        print("No teams to scrape.", file=sys.stderr)
        return 0

    if args.limit > 0:
        teams_to_scrape = teams_to_scrape[: args.limit]

    print(f"Scraping {len(teams_to_scrape)} teams from stats.ncaa.org", file=sys.stderr)

    from playwright.sync_api import sync_playwright

    all_results = []
    success_count = 0
    fail_count = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 720},
        )
        page = context.new_page()

        for i, (cid, tid) in enumerate(teams_to_scrape):
            print(
                f"  [{i + 1}/{len(teams_to_scrape)}] {cid} (ncaa_id={tid})...",
                end="",
                file=sys.stderr,
            )

            players = scrape_ncaa_roster(page, tid)

            for player in players:
                player["canonical_id"] = cid
                player["ncaa_teams_id"] = tid

            all_results.extend(players)

            n_bt = sum(1 for p in players if p.get("throws") in ("R", "L"))
            if n_bt > 0:
                success_count += 1
                print(f" ✓ {len(players)} players ({n_bt} with B/T)", file=sys.stderr)
            else:
                fail_count += 1
                print(f" ✗ no B/T data found", file=sys.stderr)

            # Rate limit: 2-3s between requests
            time.sleep(2.5)

        browser.close()

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)

        # Merge with existing if present
        if args.out.exists():
            old = pd.read_csv(args.out, dtype=str)
            # Remove old data for teams we just re-scraped
            scraped_cids = set(df["canonical_id"])
            old = old[~old["canonical_id"].isin(scraped_cids)]
            df = pd.concat([old, df], ignore_index=True)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)

        n_with_bt = (df["throws"].isin(["R", "L"])).sum()
        n_teams = df["canonical_id"].nunique()
        print(
            f"\nSaved {len(df)} players ({n_with_bt} with throws) from {n_teams} teams → {args.out}",
            file=sys.stderr,
        )
    else:
        print("\nNo data scraped.", file=sys.stderr)

    print(f"\nSummary: {success_count} teams succeeded, {fail_count} teams failed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
