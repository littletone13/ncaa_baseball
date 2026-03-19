#!/usr/bin/env python3
"""
scrape_sidearm_handedness.py — Scrape pitcher handedness from Sidearm team roster pages.

School athletics sites (Sidearm-powered) show B/T (Bats/Throws) for every player.
These are free, no login required, and work with headless browsers.

Usage:
  # Scrape all 308 teams
  python3 scripts/scrape_sidearm_handedness.py

  # Test with N teams
  python3 scripts/scrape_sidearm_handedness.py --limit 10

  # Resume from where we left off (skips teams already scraped)
  python3 scripts/scrape_sidearm_handedness.py --resume

Output:
  data/processed/sidearm_rosters.csv
  Columns: canonical_id, player_name, position, bats, throws, source_url
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd


OUTPUT_CSV = Path("data/processed/sidearm_rosters.csv")


def get_sidearm_urls(canonical_csv: Path) -> dict[str, str]:
    """
    Build canonical_id → roster URL mapping from sidearm_urls module.

    Uses the comprehensive 308-team mapping in scripts/sidearm_urls.py.
    Returns {canonical_id: full_roster_url}.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from sidearm_urls import get_all_roster_urls
    urls = get_all_roster_urls()

    return urls


def scrape_roster_page(page, url: str) -> list[dict]:
    """
    Scrape a Sidearm roster page for player names and B/T data.

    Returns list of {player_name, position, bats, throws}.
    """
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=20000)
        time.sleep(3)
    except Exception as e:
        return []

    body = page.inner_text("body")
    players = []

    # Strategy 1: Parse structured roster table
    # Sidearm rosters typically have: #, Name, Pos, B/T, Ht, Wt, Class, Hometown
    tables = page.query_selector_all("table")
    for table in tables:
        rows = table.query_selector_all("tr")
        for row in rows:
            text = row.inner_text()
            # Look for B/T pattern in the row
            bt_match = re.search(r'([RLSB])\s*/\s*([RL])', text)
            if bt_match:
                # Extract name — typically the longest non-numeric text before B/T
                parts = text.split("\t")
                if len(parts) < 2:
                    parts = re.split(r'\s{2,}', text)
                name = ""
                pos = ""
                for part in parts:
                    part = part.strip()
                    if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]', part) and not name:
                        name = part
                    elif re.match(r'^(RHP|LHP|C|1B|2B|3B|SS|OF|DH|IF|P|UT)', part) and not pos:
                        pos = part
                if name:
                    players.append({
                        "player_name": name,
                        "position": pos,
                        "bats": bt_match.group(1),
                        "throws": bt_match.group(2),
                    })

    # Strategy 2: Fallback — find B/T patterns in body text with nearby names
    if not players:
        lines = body.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            bt_match = re.search(r'^([RLSB])/([RL])$', line)
            if bt_match:
                # Look backwards for a name
                name = ""
                for j in range(max(0, i - 5), i):
                    candidate = lines[j].strip()
                    if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]', candidate) and len(candidate) < 50:
                        name = candidate
                if name:
                    players.append({
                        "player_name": name,
                        "position": "",
                        "bats": bt_match.group(1),
                        "throws": bt_match.group(2),
                    })
            i += 1

    # Strategy 3: card-based layout (Sidearm uses cards with fields on separate lines)
    # Pattern: Name line ... Position line ... "Custom Field 1" or direct B/T line
    if not players:
        lines = body.split("\n")
        for i, line in enumerate(lines):
            line_s = line.strip()
            bt_match = re.match(r'^([RLSB])/([RL])$', line_s)
            if bt_match:
                # Search backwards for name (multi-word, capitalized)
                name = ""
                pos = ""
                for j in range(i - 1, max(0, i - 20), -1):
                    candidate = lines[j].strip()
                    # Skip field labels and values
                    if candidate in ("Position", "Height", "Weight", "Hometown",
                                     "Last School", "Academic Year", "Custom Field 1",
                                     "Custom Field 2", "High School", "Number", ""):
                        continue
                    if re.match(r"^\d+$", candidate):  # jersey number
                        continue
                    if re.match(r"^\d+'\s*\d+", candidate):  # height
                        continue
                    if re.match(r"^\d+ lbs", candidate):  # weight
                        continue
                    if re.match(r"^(Fr\.|So\.|Jr\.|Sr\.|R-Fr\.|R-So\.|R-Jr\.|Gr\.)", candidate):
                        continue
                    # Position detection
                    if re.match(r"^(RHP|LHP|C|1B|2B|3B|SS|OF|DH|IF|P|UT|INF|UTIL|LHSP|RHSP|LF|RF|CF)$", candidate):
                        pos = candidate
                        continue
                    # Name detection: two+ words, starts with capital
                    if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]', candidate) and not name:
                        name = candidate
                        break

                if name:
                    players.append({
                        "player_name": name,
                        "position": pos,
                        "bats": bt_match.group(1),
                        "throws": bt_match.group(2),
                    })

    # Strategy 4: RHP/LHP position labels (sites without B/T field)
    # Many Sidearm sites list position as "RHP" or "LHP" — we can infer throws from that
    if not players:
        lines = body.split("\n")
        for i, line in enumerate(lines):
            line_s = line.strip()
            if line_s in ("RHP", "LHP", "RHSP", "LHSP"):
                throws = "R" if line_s.startswith("R") else "L"
                # Search backwards for name
                name = ""
                for j in range(i - 1, max(0, i - 15), -1):
                    candidate = lines[j].strip()
                    if candidate in ("Position", "Height", "Weight", "Hometown",
                                     "Last School", "Academic Year", "Custom Field 1",
                                     "Number", "", "Roster", "ROSTER"):
                        continue
                    if re.match(r"^\d+$", candidate):
                        continue
                    if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]', candidate) and len(candidate) < 50:
                        name = candidate
                        break
                if name:
                    players.append({
                        "player_name": name,
                        "position": line_s,
                        "bats": "",  # can't infer bats from position
                        "throws": throws,
                    })

    # Strategy 5: Inline format — "Position Height Weight B/T" on one line, name nearby
    # Example: "Infielder 6'1" 180 lbs R/R" with name on next non-empty line
    if not players:
        lines = body.split("\n")
        for i, line in enumerate(lines):
            ls = line.strip()
            # Match lines like "RHP 6'2" 195 lbs R/R" or "Infielder 6'1" 180 lbs R/R"
            m = re.search(r'([RLSB])/([RL])\s*$', ls)
            if m and len(ls) > 10:
                bats, throws = m.group(1), m.group(2)
                # Extract position from start of line
                pos_match = re.match(r'^(RHP|LHP|RHSP|LHSP|Pitcher|Infielder|Outfielder|Catcher|Utility|IF|OF|C|P|UT)', ls)
                pos = pos_match.group(1) if pos_match else ""
                if pos in ("RHP", "LHP", "RHSP", "LHSP"):
                    throws = "R" if pos.startswith("R") else "L"
                # Look forward and backward for name
                name = ""
                for j in range(i + 1, min(len(lines), i + 8)):
                    c = lines[j].strip()
                    if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]', c) and len(c) < 50 and len(c) > 4:
                        name = c
                        break
                if not name:
                    for j in range(i - 1, max(0, i - 5), -1):
                        c = lines[j].strip()
                        if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]', c) and len(c) < 50 and len(c) > 4:
                            name = c
                            break
                if name:
                    players.append({
                        "player_name": name,
                        "position": pos,
                        "bats": bats,
                        "throws": throws,
                    })

    return players


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape handedness from Sidearm roster pages.")
    parser.add_argument("--canonical", type=Path,
                        default=Path("data/registries/canonical_teams_2026.csv"))
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to N teams (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip teams already in output CSV")
    args = parser.parse_args()

    sidearm_urls = get_sidearm_urls(args.canonical)
    print(f"Sidearm URLs configured: {len(sidearm_urls)} teams", file=sys.stderr)

    # Load existing data for resume
    scraped_teams = set()
    if args.resume and OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV, dtype=str)
        scraped_teams = set(existing["canonical_id"].unique())
        print(f"Already scraped: {len(scraped_teams)} teams", file=sys.stderr)

    teams_to_scrape = [
        (cid, url) for cid, url in sidearm_urls.items()
        if cid not in scraped_teams
    ]

    if args.limit > 0:
        teams_to_scrape = teams_to_scrape[:args.limit]

    print(f"Scraping: {len(teams_to_scrape)} teams", file=sys.stderr)

    from playwright.sync_api import sync_playwright

    all_results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for i, (cid, url) in enumerate(teams_to_scrape):
            print(f"  [{i+1}/{len(teams_to_scrape)}] {cid}...", end="", file=sys.stderr)
            players = scrape_roster_page(page, url)
            for player in players:
                player["canonical_id"] = cid
                player["source_url"] = url
            all_results.extend(players)
            n_p = sum(1 for p in players if p.get("throws") in ("R", "L"))
            print(f" {n_p} pitchers with handedness", file=sys.stderr)
            time.sleep(1.5)

        browser.close()

    # Save results
    df = pd.DataFrame(all_results)
    if not df.empty:
        if args.resume and OUTPUT_CSV.exists():
            old = pd.read_csv(OUTPUT_CSV, dtype=str)
            df = pd.concat([old, df], ignore_index=True)

        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)

        n_with = (df["throws"].isin(["R", "L"])).sum()
        n_teams = df["canonical_id"].nunique()
        print(f"\nSaved {len(df)} players ({n_with} with throws) from {n_teams} teams → {OUTPUT_CSV}",
              file=sys.stderr)
    else:
        print("No data scraped.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
