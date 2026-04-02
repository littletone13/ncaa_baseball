#!/usr/bin/env python3
"""
parse_d1b_rotations_html.py — Parse D1Baseball weekly rotations from HTML.

Accepts the raw HTML from the D1Baseball "Projected Weekend Rotations" article
(either pasted to stdin, from a file, or fetched via browser).

The HTML uses structured divs with classes:
  .starters-matrix__starter — pitcher name (with optional * for unconfirmed)
  .starter-top .sm-text — handedness (L/R)
  .starters-matrix__team — team abbreviation (from data-id attribute)
  .starters-matrix__item .sm-text — IP, ERA, opponent, matchup

Usage:
  # From file (user saves the page source)
  python3 scripts/parse_d1b_rotations_html.py --html rotations.html

  # From stdin (pipe from curl or copy-paste)
  cat rotations.html | python3 scripts/parse_d1b_rotations_html.py --stdin

  # From the user message content (JSON with d1-dynamic-post-content key)
  python3 scripts/parse_d1b_rotations_html.py --json-content content.json

Output: data/processed/d1baseball_rotations.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from html import unescape
from pathlib import Path

import _bootstrap  # noqa: F401


def parse_rotations_html(html: str) -> list[dict]:
    """Parse D1Baseball rotation HTML into structured records."""
    records = []

    # Find conference headings to track which conference each team belongs to
    conf_pattern = r'<h2[^>]*id="h-([^"]+)"[^>]*>([^<]+)</h2>'
    conferences = re.findall(conf_pattern, html)
    conf_map = {slug: name for slug, name in conferences}

    # Find each team row: <div class="flex-row myleagues__proteam" data-id="TEAM">
    team_pattern = r'<div class="flex-row myleagues__proteam" data-id="([^"]+)">'
    team_positions = [(m.start(), m.group(1)) for m in re.finditer(team_pattern, html)]

    for i, (pos, team_abbr) in enumerate(team_positions):
        # Get the HTML chunk for this team (up to next team or end)
        end_pos = team_positions[i + 1][0] if i + 1 < len(team_positions) else len(html)
        chunk = html[pos:end_pos]

        # Find all starters-matrix__item divs (one per game)
        item_pattern = r'<div class="starters-matrix__item">(.*?)</div>\s*</div>'
        # Use a simpler approach: find all starter names and their context
        starter_pattern = r'<a class="starters-matrix__starter[^"]*"[^>]*>([^<]+)</a>\s*<span class="sm-text">([^<]*)</span>'
        starters = re.findall(starter_pattern, chunk)

        # Find opponent info
        opp_pattern = r'<div class="sm-text">(?:@\s*|vs\.?\s*)([^<]+)</div>'
        opponents = re.findall(opp_pattern, chunk)

        game_days = ["fri", "sat", "sun"]
        for j, (pitcher_name, hand_raw) in enumerate(starters):
            pitcher_name = unescape(pitcher_name).strip()
            hand_raw = hand_raw.strip()

            if pitcher_name.upper() == "TBA" or not pitcher_name:
                continue

            # Clean name (remove asterisk tracking)
            confirmed = "*" not in pitcher_name
            pitcher_clean = pitcher_name.replace("*", "").strip()

            # Handedness
            hand = ""
            if hand_raw == "L":
                hand = "LHP"
            elif hand_raw == "R":
                hand = "RHP"

            day = game_days[j] if j < len(game_days) else f"gm{j+1}"

            # Opponent
            opponent = opponents[j].strip() if j < len(opponents) else ""

            records.append({
                "team_abbr": team_abbr,
                "pitcher_name": pitcher_clean,
                "hand": hand,
                "day": day,
                "opponent": opponent,
                "confirmed": "yes" if confirmed else "unconfirmed",
                "source": "d1baseball",
            })

    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse D1Baseball rotation HTML.")
    parser.add_argument("--html", type=Path, help="Path to saved HTML file")
    parser.add_argument("--stdin", action="store_true", help="Read HTML from stdin")
    parser.add_argument("--json-content", type=Path, help="JSON file with d1-dynamic-post-content key")
    parser.add_argument("--out", type=Path, default=Path("data/processed/d1baseball_rotations.csv"))
    args = parser.parse_args()

    html = ""
    if args.json_content and args.json_content.exists():
        with open(args.json_content) as f:
            data = json.load(f)
        html = data.get("content", {}).get("d1-dynamic-post-content", "")
    elif args.html and args.html.exists():
        html = args.html.read_text()
    elif args.stdin:
        html = sys.stdin.read()
    else:
        print("Provide --html, --stdin, or --json-content", file=sys.stderr)
        return 1

    if not html:
        print("No HTML content found.", file=sys.stderr)
        return 1

    records = parse_rotations_html(html)

    if not records:
        print("No rotation data parsed from HTML.", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["team_abbr", "pitcher_name", "hand", "day", "opponent", "confirmed", "source"])
        w.writeheader()
        w.writerows(records)

    n_teams = len(set(r["team_abbr"] for r in records))
    n_fri = sum(1 for r in records if r["day"] == "fri")
    n_sat = sum(1 for r in records if r["day"] == "sat")
    n_sun = sum(1 for r in records if r["day"] == "sun")
    n_lhp = sum(1 for r in records if r["hand"] == "LHP")
    print(f"Wrote {len(records)} rotation entries → {args.out}", file=sys.stderr)
    print(f"  Teams: {n_teams} | Fri: {n_fri} | Sat: {n_sat} | Sun: {n_sun} | LHP: {n_lhp}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
