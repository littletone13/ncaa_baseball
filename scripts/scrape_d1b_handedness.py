#!/usr/bin/env python3
"""
scrape_d1b_handedness.py — Scrape pitcher handedness from D1Baseball player pages.

Uses Playwright with persistent browser profile. First-time setup requires
running --login to authenticate with D1Baseball (uses credentials from .env).

Player pages show BAT/THRW: R/R format.

Usage:
  # First time: login interactively
  python3 scripts/scrape_d1b_handedness.py --login

  # Scrape handedness for all pitchers in pitcher_table
  python3 scripts/scrape_d1b_handedness.py --scrape

  # Scrape only pitchers missing handedness
  python3 scripts/scrape_d1b_handedness.py --scrape --missing-only

  # Limit to N pitchers (for testing)
  python3 scripts/scrape_d1b_handedness.py --scrape --limit 50

Output:
  data/processed/pitcher_handedness.csv
  Columns: pitcher_name, canonical_id, throws, bats, source, d1b_slug
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


BROWSER_DATA_DIR = Path.home() / ".cache" / "ncaa_baseball_browser"
OUTPUT_CSV = Path("data/processed/pitcher_handedness.csv")


def do_login():
    """Automated D1Baseball login using credentials from .env."""
    from playwright.sync_api import sync_playwright

    load_dotenv()
    email = os.getenv("D1B_USER")
    password = os.getenv("D1B_PASS")

    if not email or not password:
        print("Set D1B_USER and D1B_PASS in .env", file=sys.stderr)
        sys.exit(1)

    BROWSER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(BROWSER_DATA_DIR),
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
            ignore_default_args=["--enable-automation"],
        )
        page = context.new_page()
        page.goto("https://d1baseball.com/login/", wait_until="domcontentloaded", timeout=30000)
        time.sleep(5)

        body = page.inner_text("body")[:500]
        if "Log Out" in body or "My Account" in body:
            print("Already logged in!")
            context.close()
            return

        # Fill login form
        try:
            page.fill('input[name="log"], input[type="email"], #user_login', email)
            page.fill('input[name="pwd"], input[type="password"], #user_pass', password)
            page.click('input[type="submit"], button[type="submit"]')
            time.sleep(5)

            body = page.inner_text("body")[:500]
            if "Log Out" in body or "My Account" in body:
                print("Login successful!")
            else:
                print("Login may have failed — check browser. Session saved regardless.")
        except Exception as e:
            print(f"Auto-login failed: {e}", file=sys.stderr)
            print("Falling back to manual login. Please log in and close the browser.")
            try:
                page.wait_for_event("close", timeout=300000)
            except:
                pass

        context.close()


def scrape_team_roster(page, slug: str) -> list[dict]:
    """
    Scrape a team's roster page for player handedness.

    D1B roster pages show players with BAT/THRW or B/T info.
    Returns list of {name, throws, bats} dicts.
    """
    url = f"https://d1baseball.com/team/{slug}/roster/"
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=20000)
        time.sleep(2)
    except Exception as e:
        print(f"    Failed to load {url}: {e}", file=sys.stderr)
        return []

    body = page.inner_text("body")

    # Look for player entries with handedness
    # D1B roster format varies but typically:
    #   Name | Pos | B/T | Ht | Wt | Class | Hometown
    players = []

    # Try table parsing
    tables = page.query_selector_all("table")
    for table in tables:
        rows = table.query_selector_all("tr")
        header_cells = []
        for row in rows:
            cells = row.query_selector_all("td, th")
            cell_texts = [c.inner_text().strip() for c in cells]

            if not header_cells and any("B/T" in c or "Throws" in c or "THR" in c for c in cell_texts):
                header_cells = cell_texts
                continue

            if header_cells and len(cell_texts) >= len(header_cells):
                entry = dict(zip(header_cells, cell_texts))
                name = entry.get("Name", entry.get("Player", ""))
                bt = entry.get("B/T", entry.get("Throws", entry.get("THR", "")))
                if name and bt:
                    # Parse B/T format: "R/R", "L/L", "S/R", etc.
                    bt_match = re.match(r"([RLSB])\s*/\s*([RL])", bt)
                    if bt_match:
                        players.append({
                            "name": name,
                            "bats": bt_match.group(1),
                            "throws": bt_match.group(2),
                        })

    # Fallback: regex search through body text
    if not players:
        # Look for patterns like "John Smith RHP R/R" or "B/T: R/R"
        for match in re.finditer(
            r"([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:RHP|LHP|RHSP|LHSP)\s+([RLSB])/([RL])",
            body,
        ):
            players.append({
                "name": match.group(1),
                "bats": match.group(2),
                "throws": match.group(3),
            })

    return players


def scrape_handedness(
    pitcher_table_csv: Path,
    crosswalk_csv: Path,
    limit: int = 0,
    missing_only: bool = False,
) -> pd.DataFrame:
    """Scrape handedness for pitchers in pitcher_table."""
    from playwright.sync_api import sync_playwright

    pt = pd.read_csv(pitcher_table_csv, dtype=str)
    xw = pd.read_csv(crosswalk_csv, dtype=str)

    # Build canonical_id → d1b_slug mapping
    cid_to_slug: dict[str, str] = {}
    for _, r in xw.iterrows():
        cid = str(r.get("canonical_id", "")).strip()
        d1b_name = str(r.get("d1baseball_name", "")).strip()
        if cid and d1b_name:
            slug = re.sub(r"[^a-z0-9]+", "-", d1b_name.lower()).strip("-")
            slug = slug.replace("\u2019", "").replace("'", "")
            cid_to_slug[cid] = slug

    # Load existing handedness data if available
    existing: dict[tuple[str, str], dict] = {}
    if OUTPUT_CSV.exists():
        ex_df = pd.read_csv(OUTPUT_CSV, dtype=str)
        for _, r in ex_df.iterrows():
            key = (str(r.get("pitcher_name", "")).strip().lower(),
                   str(r.get("canonical_id", "")).strip())
            existing[key] = {"throws": r.get("throws", ""), "bats": r.get("bats", "")}

    # Get unique teams to scrape
    teams_needed = set()
    pitchers_to_resolve = []
    for _, r in pt.iterrows():
        cid = str(r.get("team_canonical_id", r.get("canonical_id", ""))).strip()
        name = str(r.get("pitcher_name", "")).strip()
        has_throws = str(r.get("throws", "")).strip() in ("R", "L")
        has_existing = (name.lower(), cid) in existing

        if missing_only and (has_throws or has_existing):
            continue

        if cid in cid_to_slug:
            teams_needed.add(cid)
            pitchers_to_resolve.append({"name": name, "canonical_id": cid})

    print(f"Teams to scrape: {len(teams_needed)}", file=sys.stderr)
    print(f"Pitchers to resolve: {len(pitchers_to_resolve)}", file=sys.stderr)

    if limit > 0:
        # Limit by teams, not pitchers
        teams_needed = set(list(teams_needed)[:limit])

    BROWSER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_players: dict[str, list[dict]] = {}  # cid → [{name, throws, bats}]

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(BROWSER_DATA_DIR),
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
            ignore_default_args=["--enable-automation"],
        )
        page = context.new_page()

        for i, cid in enumerate(sorted(teams_needed)):
            slug = cid_to_slug[cid]
            print(f"  [{i+1}/{len(teams_needed)}] {cid} → {slug}...", file=sys.stderr, end="")
            players = scrape_team_roster(page, slug)
            all_players[cid] = players
            print(f" {len(players)} players", file=sys.stderr)
            time.sleep(2)  # Rate limit

        context.close()

    # Match scraped players to our pitcher table
    results = []
    for entry in pitchers_to_resolve:
        name = entry["name"]
        cid = entry["canonical_id"]
        name_lower = name.lower()

        # Check existing
        if (name_lower, cid) in existing:
            ex = existing[(name_lower, cid)]
            results.append({
                "pitcher_name": name,
                "canonical_id": cid,
                "throws": ex["throws"],
                "bats": ex["bats"],
                "source": "cached",
            })
            continue

        # Match against scraped roster
        team_players = all_players.get(cid, [])
        matched = False
        for tp in team_players:
            # Fuzzy name match (last name + first initial)
            tp_parts = tp["name"].lower().split()
            name_parts = name_lower.split()
            if len(tp_parts) >= 2 and len(name_parts) >= 2:
                if tp_parts[-1] == name_parts[-1] and tp_parts[0][0] == name_parts[0][0]:
                    results.append({
                        "pitcher_name": name,
                        "canonical_id": cid,
                        "throws": tp["throws"],
                        "bats": tp["bats"],
                        "source": "d1baseball_roster",
                    })
                    matched = True
                    break

        if not matched:
            results.append({
                "pitcher_name": name,
                "canonical_id": cid,
                "throws": "",
                "bats": "",
                "source": "not_found",
            })

    df = pd.DataFrame(results)
    if not df.empty:
        # Merge with existing and save
        if OUTPUT_CSV.exists():
            old = pd.read_csv(OUTPUT_CSV, dtype=str)
            df = pd.concat([old, df], ignore_index=True)
            df = df.drop_duplicates(
                subset=["pitcher_name", "canonical_id"], keep="last"
            )
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        found = (df["throws"].isin(["R", "L"])).sum()
        print(f"\nSaved {len(df)} pitchers ({found} with handedness) → {OUTPUT_CSV}",
              file=sys.stderr)

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape pitcher handedness from D1Baseball.")
    parser.add_argument("--login", action="store_true", help="Open browser for D1B login")
    parser.add_argument("--scrape", action="store_true", help="Scrape handedness data")
    parser.add_argument("--missing-only", action="store_true",
                        help="Only scrape pitchers without handedness")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to N teams (for testing)")
    parser.add_argument("--pitcher-table", type=Path,
                        default=Path("data/processed/pitcher_table.csv"))
    parser.add_argument("--crosswalk", type=Path,
                        default=Path("data/registries/d1baseball_crosswalk.csv"))
    args = parser.parse_args()

    if args.login:
        do_login()
        return 0

    if args.scrape:
        scrape_handedness(
            pitcher_table_csv=args.pitcher_table,
            crosswalk_csv=args.crosswalk,
            limit=args.limit,
            missing_only=args.missing_only,
        )
        return 0

    print("Specify --login or --scrape", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
