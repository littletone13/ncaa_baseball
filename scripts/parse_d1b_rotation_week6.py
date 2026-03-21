#!/usr/bin/env python3
"""
Parse D1Baseball weekend rotation markdown (Week 6+ format) into structured CSV.

This handles the conference-grouped table format where each team block has:
  ![ABBR](https://www.ncaa.com/.../schools/bgl/{logo-slug}.svg)ABBR
  [Pitcher Name](url) R/L
  IP, ERA
  vs./@ Opponent
  vs. [Opp Pitcher](url) R/L
  ... (repeated 3x for Fri/Sat/Sun)

Uses the NCAA logo URL slug (e.g., "south-carolina", "oklahoma-st") as the
primary team identifier — this is unique and unambiguous, unlike short abbreviations
which collide (OSU = Ohio St/Oklahoma St/Oregon St, USC = USC/South Carolina, etc.).

Usage:
    .venv/bin/python3 scripts/parse_d1b_rotation_week6.py \
        --input .firecrawl/d1b-rotations-week6.md \
        --week week6
"""
import argparse
import csv
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# NCAA logo slug → canonical_id mapping
# ---------------------------------------------------------------------------

# NCAA logo URLs follow pattern: .../schools/bgl/{slug}.svg
# These slugs are unique per school (unlike D1B abbreviations)
LOGO_SLUG_TO_TEAM_NAME: dict[str, str] = {
    # ACC
    "boston-college": "boston college",
    "california": "california",
    "clemson": "clemson",
    "duke": "duke",
    "florida-st": "florida st.",
    "georgia-tech": "georgia tech",
    "louisville": "louisville",
    "miami-fl": "miami (fl)",
    "north-carolina": "north carolina",
    "north-carolina-st": "nc state",
    "notre-dame": "notre dame",
    "pittsburgh": "pittsburgh",
    "smu": "smu",
    "stanford": "stanford",
    "syracuse": "syracuse",
    "virginia": "virginia",
    "virginia-tech": "virginia tech",
    "wake-forest": "wake forest",
    # Big Ten
    "illinois": "illinois",
    "indiana": "indiana",
    "iowa": "iowa",
    "maryland": "maryland",
    "michigan": "michigan",
    "michigan-st": "michigan st.",
    "minnesota": "minnesota",
    "nebraska": "nebraska",
    "northwestern": "northwestern",
    "ohio-st": "ohio st.",
    "oregon": "oregon",
    "penn-st": "penn st.",
    "purdue": "purdue",
    "rutgers": "rutgers",
    "ucla": "ucla",
    "southern-california": "southern california",
    "washington": "washington",
    "wisconsin": "wisconsin",
    # Big 12
    "arizona": "arizona",
    "arizona-st": "arizona st.",
    "baylor": "baylor",
    "byu": "byu",
    "cincinnati": "cincinnati",
    "colorado": "colorado",
    "houston": "houston",
    "iowa-st": "iowa st.",
    "kansas": "kansas",
    "kansas-st": "kansas st.",
    "oklahoma-st": "oklahoma st.",
    "tcu": "tcu",
    "texas-tech": "texas tech",
    "ucf": "ucf",
    "utah": "utah",
    "west-virginia": "west virginia",
    # SEC
    "alabama": "alabama",
    "arkansas": "arkansas",
    "auburn": "auburn",
    "florida": "florida",
    "georgia": "georgia",
    "kentucky": "kentucky",
    "lsu": "lsu",
    "mississippi-st": "mississippi st.",
    "missouri": "missouri",
    "oklahoma": "oklahoma",
    "ole-miss": "ole miss",
    "south-carolina": "south carolina",
    "tennessee": "tennessee",
    "texas": "texas",
    "texas-am": "texas a&m",
    "vanderbilt": "vanderbilt",
    # Sun Belt
    "southern-miss": "southern miss.",
    "coastal-caro": "coastal carolina",
    "troy": "troy",
    "ga-southern": "ga. southern",
    "james-madison": "james madison",
    "appalachian-st": "app state",
    "arkansas-st": "arkansas st.",
    "georgia-st": "georgia st.",
    "la-lafayette": "louisiana",
    "marshall": "marshall",
    "old-dominion": "old dominion",
    "south-ala": "south alabama",
    "texas-st": "texas st.",
    "la-monroe": "ulm",
    # American
    "charlotte": "charlotte",
    "east-carolina": "east carolina",
    "fla-atlantic": "fla. atlantic",
    "memphis": "memphis",
    "rice": "rice",
    "tulane": "tulane",
    "uab": "uab",
    "south-fla": "south fla.",
    "utsa": "utsa",
    "wichita-st": "wichita st.",
    "navy": "navy",
    "temple": "temple",
    # CUSA / Independent / Other
    "dallas-baptist": "dbu",
    "delaware": "delaware",
    "fiu": "fiu",
    "jacksonville-st": "jacksonville st.",
    "kennesaw-st": "kennesaw st.",
    "liberty": "liberty",
    "louisiana-tech": "louisiana tech",
    "middle-tenn": "middle tenn.",
    "missouri-st": "missouri st.",
    "new-mexico-st": "new mexico st.",
    "sam-houston-st": "sam houston",
    "western-ky": "western ky.",
    "oregon-st": "oregon st.",
    "uc-santa-barbara": "uc santa barbara",
}


def build_logo_slug_lookup(registry_path: str) -> dict[str, str]:
    """Build lookup: NCAA logo slug → canonical_id.

    First builds team_name → canonical_id from registry,
    then maps logo slugs through LOGO_SLUG_TO_TEAM_NAME.
    """
    name_to_cid: dict[str, str] = {}
    with open(registry_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["canonical_id"]
            for col in ("team_name", "odds_api_name", "espn_name"):
                val = row.get(col, "").strip()
                if val:
                    name_to_cid[val.lower()] = cid

    slug_to_cid: dict[str, str] = {}
    for slug, team_name in LOGO_SLUG_TO_TEAM_NAME.items():
        cid = name_to_cid.get(team_name.lower())
        if cid:
            slug_to_cid[slug] = cid
        else:
            print(f"  WARNING: logo slug '{slug}' → team '{team_name}' not found in registry",
                  file=sys.stderr)

    return slug_to_cid


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------

# Matches team block: ![ABBR](https://.../schools/bgl/{slug}.svg)ABBR
# Also matches bgd variant
TEAM_RE = re.compile(
    r"^!\[([A-Z&\s\.]+)\]\(https://www\.ncaa\.com/sites/default/files/images/logos/schools/bg[ld]/([a-z0-9-]+)\.svg\)"
)

# Matches pitcher line: [Name](url) R/L  (with optional asterisk for unconfirmed)
PITCHER_RE = re.compile(
    r"^\[([^\]]+?)(?:\\\*)?\]\(https://d1baseball\.com/player/[^)]+\)\s*([RL])?$"
)

# Matches stats line: 27.2 IP, 3.58 ERA
STATS_RE = re.compile(r"^([\d]+\.?\d*)\s*IP,\s*([\d]+\.?\d*)\s*ERA$")

# Matches TBA pitcher (links back to rotation page itself)
TBA_RE = re.compile(r"^\[TBA\]\(")


def parse_rotation_md(md_text: str, slug_to_cid: dict[str, str]) -> list[dict]:
    """Parse the firecrawl markdown into a flat list of pitcher assignments."""
    lines = md_text.split("\n")
    results: list[dict] = []
    unresolved_slugs: set[str] = set()
    days = ["fri", "sat", "sun"]

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for team block start
        m = TEAM_RE.match(line)
        if not m:
            i += 1
            continue

        team_abbrev = m.group(1).strip()
        logo_slug = m.group(2)

        cid = slug_to_cid.get(logo_slug)
        if cid is None:
            unresolved_slugs.add(f"{logo_slug} ({team_abbrev})")
            i += 1
            continue

        # Now parse up to 3 game-day pitcher entries for this team
        i += 1
        game_idx = 0
        while game_idx < 3 and i < len(lines):
            line = lines[i].strip()

            # Skip empty lines
            if not line:
                i += 1
                continue

            # Check if we hit the next team block or conference header
            if TEAM_RE.match(line) or line.startswith("## "):
                break

            # Try to match a pitcher name
            pm = PITCHER_RE.match(line)
            if pm:
                pitcher_name = pm.group(1).strip()
                # Remove trailing \* for unconfirmed
                pitcher_name = pitcher_name.rstrip("*").rstrip("\\").strip()
                hand = pm.group(2) or ""

                # Look ahead for stats
                era = ""
                ip = ""
                j = i + 1
                while j < len(lines) and j < i + 3:
                    sline = lines[j].strip()
                    sm = STATS_RE.match(sline)
                    if sm:
                        ip = sm.group(1)
                        era = sm.group(2)
                        break
                    j += 1

                day = days[game_idx] if game_idx < 3 else f"day{game_idx+1}"
                results.append({
                    "canonical_id": cid,
                    "team_abbrev": team_abbrev,
                    "logo_slug": logo_slug,
                    "day": day,
                    "pitcher_name": pitcher_name,
                    "hand": f"{hand}HP" if hand else "",
                    "era": era,
                    "ip": ip,
                })
                game_idx += 1
                i += 1
                continue

            # Check for TBA
            if TBA_RE.match(line):
                game_idx += 1
                i += 1
                continue

            # Check for a stats line or matchup line (skip)
            if STATS_RE.match(line) or line.startswith("vs.") or line.startswith("@") or line == "—":
                i += 1
                continue

            # Unknown line — skip
            i += 1

    if unresolved_slugs:
        print(f"\n  Unresolved logo slugs:", file=sys.stderr)
        for s in sorted(unresolved_slugs):
            print(f"    - {s}", file=sys.stderr)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Parse D1Baseball weekend rotation (Week 6+ format) into CSV"
    )
    parser.add_argument(
        "--input",
        default=".firecrawl/d1b-rotations-week6.md",
        help="Path to scraped markdown file",
    )
    parser.add_argument(
        "--output",
        default="data/processed/d1baseball_rotations.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--week",
        default="week6",
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

    slug_to_cid = build_logo_slug_lookup(str(registry_path))
    print(f"Loaded {len(slug_to_cid)} logo slug → canonical_id mappings", file=sys.stderr)

    # Read markdown
    md_path = Path(args.input)
    if not md_path.exists():
        print(f"ERROR: Input file not found: {md_path}", file=sys.stderr)
        sys.exit(1)

    md_text = md_path.read_text(encoding="utf-8")
    entries = parse_rotation_md(md_text, slug_to_cid)

    # Build output rows
    rows: list[dict] = []
    for entry in entries:
        rows.append({
            "canonical_id": entry["canonical_id"],
            "team_name": entry["team_abbrev"],
            "day": entry["day"],
            "pitcher_name": entry["pitcher_name"],
            "hand": entry["hand"],
            "era": entry["era"],
            "ip": entry["ip"],
            "k_bb": "",
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
    teams_resolved = len(set(r["canonical_id"] for r in rows))
    print(f"\nParsed {len(entries)} pitcher entries for {teams_resolved} teams")
    print(f"{len(rows)} rows written → {out_path}")

    # Spot-check key matchups
    print("\nSample entries (Fri starters):")
    for r in rows:
        if r["day"] == "fri":
            print(f"  {r['canonical_id']:25s} {r['pitcher_name']:20s} {r['hand']:3s} {r['era']}")


if __name__ == "__main__":
    main()
