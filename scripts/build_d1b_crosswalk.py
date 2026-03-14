#!/usr/bin/env python3
"""
Build a crosswalk mapping D1Baseball team names → canonical_id.

D1Baseball uses full names ("Florida State", "Cal State Fullerton") while
the NCAA canonical_teams_2026.csv uses NCAA abbreviations ("Fla. State",
"Cal St. Fullerton"). This script:

1. Direct-matches where names are identical
2. Expands NCAA abbreviations to try matching full names
3. Applies manual overrides for edge cases
4. Writes data/registries/d1baseball_crosswalk.csv
"""
from __future__ import annotations

import csv
from pathlib import Path


# NCAA team_name abbreviations → D1Baseball full names
ABBREV_MAP = {
    "St.": "State",
    "Fla.": "Florida",
    "Mich.": "Michigan",
    "Ill.": "Illinois",
    "Conn.": "Connecticut",
    "Miss.": "Mississippi",
    "La.": "Louisiana",
    "Ala.": "Alabama",
    "Ga.": "Georgia",
    "Tenn.": "Tennessee",
    "Wis.": "Wisconsin",
    "Minn.": "Minnesota",
    "Ind.": "Indiana",
    "Ark.": "Arkansas",
    "N.C.": "North Carolina",
    "S.C.": "South Carolina",
}

# Manual overrides for cases abbreviation expansion can't handle
# d1baseball_name → ncaa_team_name (as it appears in canonical_teams_2026.csv)
MANUAL_MAP = {
    "Albany": "UAlbany",
    "Alcorn State": "Alcorn",
    "Appalachian State": "App State",
    "Army": "Army West Point",
    "Cal State Northridge": "CSUN",
    "Central Connecticut": "Central Conn. St.",
    "Charleston Southern": "Charleston So.",
    "College of Charleston": "Col. of Charleston",
    "Connecticut": "UConn",
    "Dallas Baptist": "DBU",
    "East Tennessee State": "ETSU",
    "Eastern Kentucky": "Eastern Ky.",
    "Fairleigh Dickinson": "FDU",
    "Florida Gulf Coast": "FGCU",
    "Florida International": "FIU",
    "Illinois-Chicago": "UIC",
    "Incarnate Word": "UIW",
    "Lamar": "Lamar University",
    "Long Island": "LIU",
    "Loyola Marymount": "LMU (CA)",
    "Miami": "Miami (FL)",
    "Mississippi Valley State": "Mississippi Val.",
    "Mount St. Mary's": "Mt. St. Mary's",
    "New Jersey Tech": "NJIT",
    "Northern Colorado": "Northern Colo.",
    "Northern Illinois": "NIU",
    "Northern Kentucky": "Northern Ky.",
    "Pennsylvania": "Penn",
    "Sacramento State": "Sacramento St.",
    "Saint Joseph's": "Saint Joseph's",
    "Saint Mary's": "Saint Mary's (CA)",
    "Saint Peter's": "St. Peter's",
    "SIU Edwardsville": "SIUE",
    "Seattle": "Seattle U",
    "Southeast Missouri State": "Southeast Mo. St.",
    "Southern": "Southern U.",
    "Southern Miss": "Southern Miss.",
    "St. John's": "St. John's (NY)",
    "St. Thomas": "St. Thomas (MN)",
    "Stephen F. Austin": "SFA",
    "Tarleton": "Tarleton St.",
    "Tennessee-Martin": "UT Martin",
    "Texas A&M-Corpus Christi": "A&M-Corpus Christi",
    "UL Monroe": "ULM",
    "UNC Wilmington": "UNCW",
    "UT Rio Grande Valley": "UTRGV",
    "West Georgia": "West Ga.",
    "Western Carolina": "Western Caro.",
    "Western Kentucky": "Western Ky.",
}


def load_canonical_teams(csv_path: Path) -> dict[str, str]:
    """Load team_name → canonical_id mapping."""
    teams = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            teams[row["team_name"]] = row["canonical_id"]
    return teams


def normalize_apostrophe(s: str) -> str:
    """Replace curly/smart apostrophes with straight ones."""
    return s.replace("\u2019", "'").replace("\u2018", "'")


def load_d1b_team_names(tsv_dir: Path) -> set[str]:
    """Get unique team names from D1Baseball TSV files."""
    names = set()
    for tsv_file in tsv_dir.glob("*.tsv"):
        with open(tsv_file) as f:
            for line in f:
                cols = line.strip().split("\t")
                if cols[0] == "Qual." or len(cols) < 3:
                    continue
                names.add(normalize_apostrophe(cols[2]))
    return names


def expand_abbrevs(name: str) -> str:
    """Expand NCAA abbreviations to full words."""
    result = name
    for abbr, full in ABBREV_MAP.items():
        result = result.replace(abbr, full)
    return result.strip()


def build_crosswalk(
    canonical_csv: Path,
    d1b_dir: Path,
) -> list[dict]:
    """Build d1baseball_name → canonical_id crosswalk."""
    canon = load_canonical_teams(canonical_csv)
    d1b_teams = load_d1b_team_names(d1b_dir)

    # Build reverse lookup: expanded_name → (ncaa_name, canonical_id)
    expanded_lookup = {}
    for ncaa_name, cid in canon.items():
        expanded = expand_abbrevs(ncaa_name)
        expanded_lookup[expanded] = (ncaa_name, cid)

    # Also add manual map targets to lookup
    manual_ncaa_to_cid = {}
    for d1b_name, ncaa_name in MANUAL_MAP.items():
        if ncaa_name in canon:
            manual_ncaa_to_cid[d1b_name] = (ncaa_name, canon[ncaa_name])

    results = []
    unmatched = []

    for d1b_name in sorted(d1b_teams):
        # 1. Direct match
        if d1b_name in canon:
            results.append({
                "d1baseball_name": d1b_name,
                "ncaa_team_name": d1b_name,
                "canonical_id": canon[d1b_name],
                "match_method": "direct",
            })
            continue

        # 2. Manual override
        if d1b_name in manual_ncaa_to_cid:
            ncaa_name, cid = manual_ncaa_to_cid[d1b_name]
            results.append({
                "d1baseball_name": d1b_name,
                "ncaa_team_name": ncaa_name,
                "canonical_id": cid,
                "match_method": "manual",
            })
            continue

        # 3. Abbreviation expansion
        if d1b_name in expanded_lookup:
            ncaa_name, cid = expanded_lookup[d1b_name]
            results.append({
                "d1baseball_name": d1b_name,
                "ncaa_team_name": ncaa_name,
                "canonical_id": cid,
                "match_method": "abbrev_expand",
            })
            continue

        unmatched.append(d1b_name)

    return results, unmatched


def main():
    base = Path(".")
    canonical_csv = base / "data/registries/canonical_teams_2026.csv"
    d1b_dir = base / "data/raw/d1baseball"
    out_csv = base / "data/registries/d1baseball_crosswalk.csv"

    results, unmatched = build_crosswalk(canonical_csv, d1b_dir)

    # Write crosswalk
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "d1baseball_name", "ncaa_team_name", "canonical_id", "match_method",
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"Crosswalk written to {out_csv}")
    print(f"  Matched: {len(results)}")
    print(f"  Unmatched: {len(unmatched)}")

    if unmatched:
        print("\nUnmatched D1Baseball teams:")
        for t in unmatched:
            print(f"  {t}")


if __name__ == "__main__":
    main()
