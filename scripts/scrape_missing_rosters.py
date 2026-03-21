#!/usr/bin/env python3
"""
scrape_missing_rosters.py — Re-scrape Sidearm roster pages for teams with zero B/T data.

The original scrape_sidearm_handedness.py missed 54 teams because their Sidearm sites
use a card layout where B/T is embedded in a combined line with class/hometown, e.g.:
  "Freshman R/R Castle Rock, Colo. Castle View HS"

This script uses a more robust parser that handles:
  - B/T embedded in combined class/hometown lines
  - Inline position/height/weight lines before name
  - Various Sidearm card layout variants

Usage:
  python3 scripts/scrape_missing_rosters.py           # scrape all 54 missing teams
  python3 scripts/scrape_missing_rosters.py --limit 5  # test with 5 teams
  python3 scripts/scrape_missing_rosters.py --merge     # merge into sidearm_rosters.csv

Output:
  data/processed/ncaa_rosters.csv (new file for NCAA-sourced roster data)
  With --merge: also updates data/processed/sidearm_rosters.csv
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd


OUTPUT_CSV = Path("data/processed/ncaa_rosters.csv")
SIDEARM_CSV = Path("data/processed/sidearm_rosters.csv")


def _find_missing_teams(
    registry_csv: Path, canonical_csv: Path
) -> list[tuple[str, str]]:
    """Find teams with zero B/T data. Returns [(canonical_id, roster_url), ...]."""
    reg = pd.read_csv(registry_csv)
    team_throws = reg.groupby("canonical_id")["throws"].apply(
        lambda x: x.isin(["L", "R"]).sum()
    )
    zero_teams = set(team_throws[team_throws == 0].index)

    # Get roster URLs from sidearm_urls module
    sys.path.insert(0, str(Path(__file__).parent))
    from sidearm_urls import get_all_roster_urls

    urls = get_all_roster_urls()
    return [(cid, urls[cid]) for cid in zero_teams if cid in urls]


def scrape_roster_robust(page, url: str) -> list[dict]:
    """
    Robustly scrape a Sidearm roster page for player B/T data.

    Handles multiple Sidearm layout variants including:
    - Card layout with B/T in combined class/hometown line
    - Table layout with separate B/T columns
    - List layout with inline B/T
    """
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=25000)
        time.sleep(4)  # Sidearm sites are JS-heavy
    except Exception as e:
        print(f" navigation error: {e}", file=sys.stderr)
        return []

    body = page.inner_text("body")
    lines = body.split("\n")
    players = []

    # ── Strategy A: Card layout — B/T in combined line ──
    # Pattern: position/height line, jersey#, blanks, name, blank, "Class B/T Hometown"
    # Example:
    #   Outfielder 6'4" 210 lbs
    #   1
    #   (blanks)
    #   Sam Harry
    #   (blank)
    #   Freshman R/R Castle Rock, Colo. Castle View HS
    for i, line in enumerate(lines):
        line_s = line.strip()
        # Look for B/T pattern embedded in line (not at start/end exclusively)
        bt_match = re.search(r'\b([RLSB])/([RL])\b', line_s)
        if not bt_match:
            continue

        bats, throws = bt_match.group(1), bt_match.group(2)

        # Search backwards for name (multi-word, starts with capital, not a position/stat)
        name = ""
        pos = ""
        for j in range(i - 1, max(0, i - 12), -1):
            candidate = lines[j].strip()
            if not candidate:
                continue
            # Skip jersey numbers
            if re.match(r"^\d{1,3}$", candidate):
                continue
            # Skip "Full Bio", "Hide/Show..."
            if candidate.startswith("Full Bio") or candidate.startswith("Hide/Show"):
                continue
            # Skip "Players", "Go", navigation items
            if candidate in (
                "Players", "Go", "Coaches", "Roster", "ROSTER",
                "Schedule", "News", "Stats",
            ):
                continue
            # Position line: "Outfielder 6'4" 210 lbs" or "Pitcher 6'1" 205 lbs"
            if re.match(
                r"^(Pitcher|Outfielder|Infielder|Catcher|Utility|"
                r"RHP|LHP|RHSP|LHSP|C|1B|2B|3B|SS|OF|DH|IF|P|UT|INF|UTIL|"
                r"First Baseman|Second Baseman|Third Baseman|Shortstop|"
                r"Left Fielder|Right Fielder|Center Fielder|"
                r"Designated Hitter|Relief Pitcher|Starting Pitcher)",
                candidate,
                re.IGNORECASE,
            ):
                # Extract just the position word(s) before height
                pos_m = re.match(
                    r"^([\w\s]+?)(?:\s+\d+['\u2019\u2032]|\s*$)", candidate
                )
                if pos_m:
                    pos = pos_m.group(1).strip()
                continue

            # Name detection: 2+ words, starts with capital letter
            # Must not be a class year or hometown
            if re.match(r"^(Freshman|Sophomore|Junior|Senior|R-Freshman|R-Sophomore|R-Junior|Graduate|Fr\.|So\.|Jr\.|Sr\.|Gr\.)", candidate):
                continue
            if re.match(r"^[A-Z][a-zA-Z'\u2019-]+([ ,]+[A-Z][a-zA-Z'\u2019-]+)+$", candidate) and len(candidate) < 50:
                # Could be name or hometown; names typically 2-3 words, no comma
                if "," not in candidate and len(candidate.split()) <= 4:
                    name = candidate
                    break

        if name:
            players.append({
                "player_name": name,
                "position": pos,
                "bats": bats,
                "throws": throws,
            })

    # ── Strategy B: Table-based layout ──
    if not players:
        try:
            tables = page.query_selector_all("table")
            for table in tables:
                rows = table.query_selector_all("tr")
                if len(rows) < 3:
                    continue

                # Try to find header row
                header_cells = rows[0].query_selector_all("th, td")
                headers = [c.inner_text().strip().lower() for c in header_cells]

                bt_col = None
                name_col = None
                pos_col = None
                for idx, h in enumerate(headers):
                    if h in ("b/t", "bt"):
                        bt_col = idx
                    elif h in ("name", "player"):
                        name_col = idx
                    elif h in ("pos", "pos.", "position"):
                        pos_col = idx

                if bt_col is None or name_col is None:
                    continue

                for row in rows[1:]:
                    cells = row.query_selector_all("td")
                    if len(cells) <= max(bt_col, name_col):
                        continue
                    name = cells[name_col].inner_text().strip()
                    bt_val = cells[bt_col].inner_text().strip()
                    pos_val = cells[pos_col].inner_text().strip() if pos_col and pos_col < len(cells) else ""

                    bt_m = re.search(r"([RLSB])/([RL])", bt_val)
                    if name and bt_m:
                        players.append({
                            "player_name": name,
                            "position": pos_val,
                            "bats": bt_m.group(1),
                            "throws": bt_m.group(2),
                        })

                if players:
                    break
        except Exception as e:
            pass

    # ── Strategy C: Standalone B/T lines (original format) ──
    if not players:
        for i, line in enumerate(lines):
            line_s = line.strip()
            if not re.match(r"^[RLSB]/[RL]$", line_s):
                continue
            bats, throws = line_s[0], line_s[2]
            name = ""
            pos = ""
            for j in range(i - 1, max(0, i - 15), -1):
                candidate = lines[j].strip()
                if not candidate:
                    continue
                if re.match(r"^\d+$", candidate):
                    continue
                if re.match(r"^(RHP|LHP|C|1B|2B|3B|SS|OF|DH|IF|P|UT)", candidate):
                    pos = candidate
                    continue
                if re.match(r"^[A-Z][a-zA-Z'\u2019-]+ [A-Z]", candidate) and len(candidate) < 50:
                    name = candidate
                    break
            if name:
                players.append({
                    "player_name": name,
                    "position": pos,
                    "bats": bats,
                    "throws": throws,
                })

    # ── Strategy D: "Right-Handed Pitcher" / "Left-Handed Pitcher" position labels ──
    # Some Sidearm sites don't have B/T field but spell out handedness in position
    if not players:
        for i, line in enumerate(lines):
            line_s = line.strip()
            # Match "Right-Handed Pitcher 6'3"" or "Left-Handed Pitcher 6'0""
            hand_match = re.match(
                r"^(Right|Left)-Handed Pitcher",
                line_s,
                re.IGNORECASE,
            )
            if not hand_match:
                continue
            throws = "R" if hand_match.group(1).lower() == "right" else "L"
            # Search forward for name (Sidearm card: position → jersey# → blanks → name)
            name = ""
            for j in range(i + 1, min(len(lines), i + 10)):
                candidate = lines[j].strip()
                if not candidate or re.match(r"^\d{1,3}$", candidate):
                    continue
                if candidate.startswith("Full Bio") or candidate.startswith("Hide/Show"):
                    break
                if re.match(r"^[A-Z][a-zA-Z'\u2019-]+([ ]+[A-Z][a-zA-Z'\u2019-]+)+$", candidate) and len(candidate) < 50:
                    name = candidate
                    break
            if name:
                players.append({
                    "player_name": name,
                    "position": "P",
                    "bats": "",
                    "throws": throws,
                })

        # Also get non-pitcher position labels for general roster coverage
        if players:  # Only if we found at least one pitcher this way
            for i, line in enumerate(lines):
                line_s = line.strip()
                pos_match = re.match(
                    r"^(Outfielder|Infielder|Catcher|Utility|First Baseman|"
                    r"Second Baseman|Third Baseman|Shortstop|"
                    r"Left Fielder|Right Fielder|Center Fielder|"
                    r"Designated Hitter|Infielder/Outfielder|"
                    r"Outfielder/Right-Handed Pitcher|Outfielder/Left-Handed Pitcher)\s",
                    line_s,
                    re.IGNORECASE,
                )
                if not pos_match:
                    continue
                pos = pos_match.group(1)
                # Infer throws from combo positions
                throws = ""
                if "Right-Handed" in pos:
                    throws = "R"
                elif "Left-Handed" in pos:
                    throws = "L"
                # Search forward for name
                name = ""
                for j in range(i + 1, min(len(lines), i + 10)):
                    candidate = lines[j].strip()
                    if not candidate or re.match(r"^\d{1,3}$", candidate):
                        continue
                    if candidate.startswith("Full Bio") or candidate.startswith("Hide/Show"):
                        break
                    if re.match(r"^[A-Z][a-zA-Z'\u2019-]+([ ]+[A-Z][a-zA-Z'\u2019-]+)+$", candidate) and len(candidate) < 50:
                        name = candidate
                        break
                if name:
                    players.append({
                        "player_name": name,
                        "position": pos,
                        "bats": "",
                        "throws": throws,
                    })

    return players


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-scrape roster pages for teams missing B/T data."
    )
    parser.add_argument("--registry", type=Path, default=Path("data/processed/player_registry.csv"))
    parser.add_argument("--canonical", type=Path, default=Path("data/registries/canonical_teams_2026.csv"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--merge", action="store_true", help="Merge results into sidearm_rosters.csv")
    parser.add_argument("--out", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()

    teams = _find_missing_teams(args.registry, args.canonical)
    if not teams:
        print("No teams missing B/T data.", file=sys.stderr)
        return 0

    if args.limit > 0:
        teams = teams[:args.limit]

    print(f"Scraping {len(teams)} teams with missing B/T data", file=sys.stderr)

    from playwright.sync_api import sync_playwright

    all_results = []
    success = 0
    fail = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )
        page = ctx.new_page()

        for i, (cid, url) in enumerate(teams):
            print(f"  [{i+1}/{len(teams)}] {cid} → {url}...", end="", file=sys.stderr)

            players = scrape_roster_robust(page, url)
            for pl in players:
                pl["canonical_id"] = cid
                pl["source_url"] = url

            all_results.extend(players)
            n_bt = sum(1 for pl in players if pl.get("throws") in ("R", "L"))

            if n_bt > 0:
                success += 1
                print(f" ✓ {len(players)} players ({n_bt} with B/T)", file=sys.stderr)
            else:
                fail += 1
                print(f" ✗ no B/T", file=sys.stderr)

            time.sleep(2)

        browser.close()

    if not all_results:
        print("\nNo data scraped.", file=sys.stderr)
        return 1

    df = pd.DataFrame(all_results)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Save to ncaa_rosters.csv
    if args.out.exists():
        old = pd.read_csv(args.out, dtype=str)
        scraped_cids = set(df["canonical_id"])
        old = old[~old["canonical_id"].isin(scraped_cids)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(args.out, index=False)

    n_bt = (df["throws"].isin(["R", "L"])).sum()
    n_teams = df["canonical_id"].nunique()
    print(f"\nSaved {len(df)} players ({n_bt} with B/T) from {n_teams} teams → {args.out}", file=sys.stderr)
    print(f"Success: {success}, Failed: {fail}", file=sys.stderr)

    # Optionally merge into sidearm_rosters.csv
    if args.merge and SIDEARM_CSV.exists():
        sidearm = pd.read_csv(SIDEARM_CSV, dtype=str)
        # Only add players from teams NOT already in sidearm
        existing_cids = set(sidearm["canonical_id"].unique())
        new_rows = df[~df["canonical_id"].isin(existing_cids)]
        if not new_rows.empty:
            # Align columns
            for col in ("bats", "throws", "position", "player_name", "canonical_id", "source_url"):
                if col not in new_rows.columns:
                    new_rows[col] = ""
            new_sidearm = new_rows[["canonical_id", "player_name", "position", "bats", "throws", "source_url"]]
            merged = pd.concat([sidearm, new_sidearm], ignore_index=True)
            merged.to_csv(SIDEARM_CSV, index=False)
            print(f"Merged {len(new_sidearm)} new rows into {SIDEARM_CSV}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
