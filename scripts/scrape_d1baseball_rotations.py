#!/usr/bin/env python3
"""
Parse D1Baseball weekend rotation markdown (from firecrawl) into structured CSV.

Usage:
    .venv/bin/python3 scripts/scrape_d1baseball_rotations.py
    .venv/bin/python3 scripts/scrape_d1baseball_rotations.py --input .firecrawl/d1baseball-weekend-rotations-week5.md --week week5
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Team-name matching helpers
# ---------------------------------------------------------------------------

def load_canonical_teams(registry_path: str) -> dict[str, str]:
    """Build lookup: lowercase team-name variant -> canonical_id.

    Checks team_name, display_name (if present), and odds_api_name columns.
    Returns dict mapping lowered name -> canonical_id.
    """
    lookup: dict[str, str] = {}
    with open(registry_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["canonical_id"]
            for col in ("team_name", "odds_api_name"):
                val = row.get(col, "").strip()
                if val:
                    lookup[val.lower()] = cid
                    # Also store without trailing mascot for odds_api_name
                    # e.g. "Clemson Tigers" -> also try "Clemson"
                    if col == "odds_api_name" and " " in val:
                        first_word = val.split()[0]
                        # Only add if unambiguous (don't overwrite)
                        lookup.setdefault(first_word.lower(), cid)
    return lookup


def resolve_team(name: str, lookup: dict[str, str]) -> str | None:
    """Try to match a D1Baseball team name to a canonical_id."""
    key = name.strip().lower()
    if key in lookup:
        return lookup[key]

    # Try common substitutions
    subs = {
        "florida st.": "florida st.",
        "nc state": "nc state",
        "miami (fl)": "miami",
        "southern california": "usc",
        "arizona st.": "arizona st.",
        "kansas st.": "kansas st.",
        "oklahoma st.": "oklahoma st.",
        "michigan st.": "michigan st.",
        "penn st.": "penn st.",
        "south fla.": "south fla.",
        "uc santa barbara": "uc santa barbara",
        "ohio st.": "ohio st.",
        "dbu": "dallas baptist",
    }
    alt = subs.get(key)
    if alt and alt.lower() in lookup:
        return lookup[alt.lower()]

    # Try without trailing period abbreviation patterns
    # e.g. "Florida St." -> look for both as-is and expanded
    return None


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------

_MATCHUP_RE = re.compile(
    r"^##\s+"
    r"(?:\*\*)?(?:\[.*?\]\(.*?\))?\s*"  # optional bold / link wrapping
    r"(\d*)"                             # optional ranking for team1
    r"(.+?)\s*"                          # team1 name
    r"\([\d]+-[\d]+\)"                   # record
    r"\s+vs\.\s+"
    r"(\d*)"                             # optional ranking for team2
    r"(.+?)\s*"                          # team2 name
    r"\([\d]+-[\d]+\)"                   # record
    r"\s*$"
)

_AT_RE = re.compile(r"^###\s+At:\s+(.+)$")


def parse_pitcher_cell(cell: str) -> dict | None:
    """Parse a single table cell into pitcher info.

    Returns dict with keys: pitcher_name, hand, era, ip, k_bb
    or None if TBA / unparseable.
    """
    cell = cell.strip()
    if not cell or cell.startswith("**TBA**") or cell == "—":
        return None

    result: dict = {}

    # Pitcher name: **[Name](url)** or **Name**
    name_m = re.search(r"\*\*\[([^\]]+)\]\([^)]*\)\*\*", cell)
    if not name_m:
        name_m = re.search(r"\*\*([^*]+)\*\*", cell)
    if name_m:
        result["pitcher_name"] = name_m.group(1).strip()
    else:
        return None

    if result["pitcher_name"] == "TBA":
        return None

    # Split on <br> to get info lines
    parts = re.split(r"<br\s*/?>", cell)

    # Hand: look for RHP or LHP
    hand_m = re.search(r"(RHP|LHP)", cell)
    result["hand"] = hand_m.group(1) if hand_m else ""

    # ERA: look for X.XX ERA
    era_m = re.search(r"([\d]+\.[\d]+)\s+ERA", cell)
    result["era"] = era_m.group(1) if era_m else ""

    # IP: look for XX.X IP
    ip_m = re.search(r"([\d]+\.[\d]+)\s+IP", cell)
    if not ip_m:
        ip_m = re.search(r"([\d]+\.[\d])\s+IP", cell)
    result["ip"] = ip_m.group(1) if ip_m else ""

    # K-BB: look for XX-XX K-BB
    kbb_m = re.search(r"([\d]+-[\d]+)\s+K-BB", cell)
    result["k_bb"] = kbb_m.group(1) if kbb_m else ""

    return result


def parse_matchups(md_text: str) -> list[dict]:
    """Parse the full markdown into a list of matchup dicts.

    Each matchup dict has:
      team1, team2, home_team,
      pitchers: list of (day, team1_pitcher, team2_pitcher)
    """
    lines = md_text.split("\n")
    matchups: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for matchup header
        m = _MATCHUP_RE.match(line)
        if not m:
            i += 1
            continue

        team1 = m.group(2).strip()
        team2 = m.group(4).strip()

        # Find "At:" line
        home_team = None
        j = i + 1
        while j < len(lines) and j < i + 5:
            at_m = _AT_RE.match(lines[j].strip())
            if at_m:
                home_team = at_m.group(1).strip()
                break
            j += 1

        if home_team is None:
            i += 1
            continue

        # Determine which team is home / away
        # The table columns are always: team1 | team2 (matching header order)
        # home_team tells us which one is home
        # Normalize for comparison
        home_is_team1 = _fuzzy_match(home_team, team1)

        # Find table rows (lines starting with |, skip header and separator)
        table_rows: list[str] = []
        k = j + 1
        while k < len(lines) and k < j + 20:
            tl = lines[k].strip()
            if tl.startswith("|") and "---" not in tl:
                # Skip the header row (contains team names)
                # Check if this is data (contains ** or TBA)
                if "**" in tl or "TBA" in tl:
                    table_rows.append(tl)
            elif tl.startswith("[") and "source" in tl.lower():
                break
            elif tl.startswith("##"):
                break
            k += 1

        days = ["fri", "sat", "sun"]
        pitchers: list[tuple] = []
        for idx, row in enumerate(table_rows[:3]):
            day = days[idx] if idx < 3 else f"day{idx+1}"
            # Split row by | — first and last are empty
            cells = row.split("|")
            # Filter out empty edge cells
            cells = [c for c in cells if c.strip()]
            if len(cells) >= 2:
                p1 = parse_pitcher_cell(cells[0])
                p2 = parse_pitcher_cell(cells[1])
            else:
                p1, p2 = None, None

            # Assign home/away based on table column order
            if home_is_team1:
                pitchers.append((day, "home", team1, p1))
                pitchers.append((day, "away", team2, p2))
            else:
                pitchers.append((day, "away", team1, p1))
                pitchers.append((day, "home", team2, p2))

        matchups.append({
            "team1": team1,
            "team2": team2,
            "home_team": home_team,
            "pitchers": pitchers,
        })

        i = k

    return matchups


def _fuzzy_match(name_a: str, name_b: str) -> bool:
    """Check if two team names refer to the same team (fuzzy)."""
    a = name_a.strip().lower()
    b = name_b.strip().lower()
    if a == b:
        return True
    # One might be a substring
    if a in b or b in a:
        return True
    # Handle abbreviation differences
    # e.g. "Florida St." vs "Florida St."
    return a.replace(".", "") == b.replace(".", "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse D1Baseball weekend rotation markdown into CSV"
    )
    parser.add_argument(
        "--input",
        default=".firecrawl/d1baseball-weekend-rotations-week5.md",
        help="Path to scraped markdown file",
    )
    parser.add_argument(
        "--output",
        default="data/processed/d1baseball_rotations.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--week",
        default="week5",
        help="Week label for source column",
    )
    parser.add_argument(
        "--registry",
        default="data/registries/canonical_teams_2026.csv",
        help="Path to canonical teams registry",
    )
    args = parser.parse_args()

    # Load team registry
    registry_path = Path(args.registry)
    if not registry_path.exists():
        print(f"ERROR: Registry not found: {registry_path}", file=sys.stderr)
        sys.exit(1)

    lookup = load_canonical_teams(str(registry_path))

    # Read markdown
    md_path = Path(args.input)
    if not md_path.exists():
        print(f"ERROR: Input file not found: {md_path}", file=sys.stderr)
        sys.exit(1)

    md_text = md_path.read_text(encoding="utf-8")
    matchups = parse_matchups(md_text)

    # Build output rows
    rows: list[dict] = []
    unresolved: set[str] = set()

    for matchup in matchups:
        for day, side, team_name, pitcher in matchup["pitchers"]:
            if pitcher is None:
                continue

            cid = resolve_team(team_name, lookup)
            if cid is None:
                unresolved.add(team_name)
                cid = f"UNRESOLVED_{team_name}"

            rows.append({
                "canonical_id": cid,
                "team_name": team_name,
                "day": day,
                "pitcher_name": pitcher["pitcher_name"],
                "hand": pitcher["hand"],
                "era": pitcher["era"],
                "ip": pitcher["ip"],
                "k_bb": pitcher["k_bb"],
                "source": f"d1baseball_{args.week}",
            })

    # Write CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "canonical_id", "team_name", "day", "pitcher_name",
        "hand", "era", "ip", "k_bb", "source",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    n_pitchers = len(rows)
    n_matchups = len(matchups)
    n_tba = sum(
        1 for m in matchups
        for _, _, _, p in m["pitchers"]
        if p is None
    )
    print(f"Parsed {n_matchups} matchups, {n_pitchers} pitcher entries ({n_tba} TBA slots skipped)")
    print(f"Output: {out_path}")

    if unresolved:
        print(f"\nWARNING: {len(unresolved)} team(s) could not be resolved to canonical_id:")
        for t in sorted(unresolved):
            print(f"  - {t}")


if __name__ == "__main__":
    main()
