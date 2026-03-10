"""
Expand stadium_orientations.csv to cover all 308 D1 teams.

Uses Nominatim (free) geocoder to look up approximate lat/lon for each
team's university location. For weather/wind calculations, we just need
to be within a few miles of the actual stadium.

Preserves existing stadium data (which may have known bearings from satellite).
New entries default to hp_bearing_deg=67 (the conventional MLB orientation).

Usage:
  python3 scripts/expand_stadium_orientations.py
  python3 scripts/expand_stadium_orientations.py --dry-run  # just show what would be added
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# ──────────────────────────────────────────────────────────────────────────────
# Manual overrides for abbreviated/ambiguous team names
# Maps team_name → geocoding query
# ──────────────────────────────────────────────────────────────────────────────

TEAM_QUERY_OVERRIDES: dict[str, str] = {
    # Abbreviated names
    "CSUN": "California State University Northridge",
    "CSU Bakersfield": "California State University Bakersfield",
    "Cal St. Fullerton": "California State University Fullerton",
    "UCF": "University of Central Florida Orlando",
    "UNLV": "University of Nevada Las Vegas",
    "UTSA": "University of Texas San Antonio",
    "UTRGV": "University of Texas Rio Grande Valley Edinburg",
    "UIC": "University of Illinois Chicago",
    "UMass Lowell": "University of Massachusetts Lowell",
    "UMass": "University of Massachusetts Amherst",
    "UMBC": "University of Maryland Baltimore County",
    "SIU Edwardsville": "Southern Illinois University Edwardsville",
    "LIU": "Long Island University Brooklyn",
    "FDU": "Fairleigh Dickinson University Teaneck",
    "NJIT": "New Jersey Institute of Technology Newark",
    "VCU": "Virginia Commonwealth University Richmond",
    "UAB": "University of Alabama Birmingham",
    "USC Upstate": "University of South Carolina Upstate Spartanburg",
    "UNC Wilmington": "University of North Carolina Wilmington",
    "UNC Greensboro": "University of North Carolina Greensboro",
    "UNC Asheville": "University of North Carolina Asheville",
    "Fla. Atlantic": "Florida Atlantic University Boca Raton",
    "FIU": "Florida International University Miami",
    "FGCU": "Florida Gulf Coast University Fort Myers",
    "Fla. Gulf Coast": "Florida Gulf Coast University Fort Myers",
    "Fla. A&M": "Florida A&M University Tallahassee",
    "UALR": "University of Arkansas Little Rock",
    "Ark.-Pine Bluff": "University of Arkansas Pine Bluff",
    "Bethune-Cookman": "Bethune-Cookman University Daytona Beach",
    "McNeese": "McNeese State University Lake Charles",
    "SFA": "Stephen F. Austin State University Nacogdoches",
    "Sam Houston": "Sam Houston State University Huntsville Texas",
    "Lamar University": "Lamar University Beaumont Texas",
    "Lamar": "Lamar University Beaumont Texas",
    "BYU": "Brigham Young University Provo Utah",
    "Le Moyne": "Le Moyne College Syracuse",
    "St. John's": "St. John's University Queens New York",
    "St. Thomas": "University of St. Thomas St Paul Minnesota",
    "St. Bonaventure": "St. Bonaventure University",
    "St. Joseph's": "Saint Joseph's University Philadelphia",
    "Central Conn. St.": "Central Connecticut State University New Britain",
    "North Carolina": "University of North Carolina Chapel Hill",
    "NC State": "North Carolina State University Raleigh",
    "NC A&T": "North Carolina A&T State University Greensboro",
    "NC Central": "North Carolina Central University Durham",
    "Eastern Ky.": "Eastern Kentucky University Richmond",
    "Western Ky.": "Western Kentucky University Bowling Green",
    "Northern Ky.": "Northern Kentucky University Highland Heights",
    "Murray St.": "Murray State University Kentucky",
    "Morehead St.": "Morehead State University Kentucky",
    "Austin Peay": "Austin Peay State University Clarksville Tennessee",
    "Middle Tenn.": "Middle Tennessee State University Murfreesboro",
    "Tenn. Tech": "Tennessee Tech University Cookeville",
    "Eastern Ill.": "Eastern Illinois University Charleston",
    "Southern Ill.": "Southern Illinois University Carbondale",
    "Illinois St.": "Illinois State University Normal",
    "Indiana St.": "Indiana State University Terre Haute",
    "Purdue Fort Wayne": "Purdue University Fort Wayne Indiana",
    "Youngstown St.": "Youngstown State University Ohio",
    "Cleveland St.": "Cleveland State University Ohio",
    "Wright St.": "Wright State University Dayton Ohio",
    "Oakland": "Oakland University Rochester Hills Michigan",
    "Green Bay": "University of Wisconsin Green Bay",
    "Milwaukee": "University of Wisconsin Milwaukee",
    "Long Beach St.": "California State University Long Beach",
    "Sacramento St.": "California State University Sacramento",
    "Fresno St.": "California State University Fresno",
    "San Jose St.": "San Jose State University",
    "San Diego St.": "San Diego State University",
    "San Diego": "University of San Diego",
    "Wichita St.": "Wichita State University Kansas",
    "Kansas St.": "Kansas State University Manhattan",
    "Iowa St.": "Iowa State University Ames",
    "Michigan St.": "Michigan State University East Lansing",
    "Ohio St.": "Ohio State University Columbus",
    "Penn St.": "Pennsylvania State University State College",
    "Oregon St.": "Oregon State University Corvallis",
    "Boise St.": "Boise State University Idaho",
    "Utah St.": "Utah State University Logan",
    "Arizona St.": "Arizona State University Tempe",
    "Kennesaw St.": "Kennesaw State University Georgia",
    "Georgia St.": "Georgia State University Atlanta",
    "Georgia Southern": "Georgia Southern University Statesboro",
    "Jacksonville St.": "Jacksonville State University Alabama",
    "Nicholls": "Nicholls State University Thibodaux Louisiana",
    "Nicholls St.": "Nicholls State University Thibodaux Louisiana",
    "Northwestern St.": "Northwestern State University Natchitoches Louisiana",
    "Southeastern La.": "Southeastern Louisiana University Hammond",
    "Southern U.": "Southern University Baton Rouge Louisiana",
    "Grambling": "Grambling State University Louisiana",
    "Alcorn": "Alcorn State University Lorman Mississippi",
    "Alcorn St.": "Alcorn State University Lorman Mississippi",
    "Jackson St.": "Jackson State University Mississippi",
    "Miss. Valley St.": "Mississippi Valley State University Itta Bena",
    "Prairie View": "Prairie View A&M University Texas",
    "Tex. Southern": "Texas Southern University Houston",
    "Alabama St.": "Alabama State University Montgomery",
    "Alabama A&M": "Alabama A&M University Huntsville",
    "MVSU": "Mississippi Valley State University Itta Bena",
    "Maine": "University of Maine Orono",
    "Stony Brook": "Stony Brook University New York",
    "Binghamton": "Binghamton University New York",
    "Albany": "University at Albany New York",
    "Hartford": "University of Hartford Connecticut",
    "Bryant": "Bryant University Smithfield Rhode Island",
    "Army West Point": "United States Military Academy West Point",
    "Army": "United States Military Academy West Point",
    "Navy": "United States Naval Academy Annapolis",
    "Air Force": "United States Air Force Academy Colorado Springs",
    "Cal Poly": "California Polytechnic State University San Luis Obispo",
    "Charleston So.": "Charleston Southern University",
    "High Point": "High Point University North Carolina",
    "Winthrop": "Winthrop University Rock Hill South Carolina",
    "Gardner-Webb": "Gardner-Webb University Boiling Springs North Carolina",
    "Radford": "Radford University Virginia",
    "Presbyterian": "Presbyterian College Clinton South Carolina",
    "UNC": "University of North Carolina Chapel Hill",
    "The Citadel": "The Citadel Charleston South Carolina",
    "Samford": "Samford University Birmingham Alabama",
    "Mercer": "Mercer University Macon Georgia",
    "Furman": "Furman University Greenville South Carolina",
    "Wofford": "Wofford College Spartanburg South Carolina",
    "VMI": "Virginia Military Institute Lexington",
    "W. Carolina": "Western Carolina University Cullowhee North Carolina",
    "ETSU": "East Tennessee State University Johnson City",
    "Elon": "Elon University North Carolina",
    "William & Mary": "College of William and Mary Williamsburg Virginia",
    "Towson": "Towson University Maryland",
    "Hofstra": "Hofstra University Hempstead New York",
    "Delaware": "University of Delaware Newark",
    "Drexel": "Drexel University Philadelphia",
    "N.C. A&T": "North Carolina A&T State University Greensboro",
    "Northeastern": "Northeastern University Boston",
    "Charleston": "College of Charleston South Carolina",
    "James Madison": "James Madison University Harrisonburg Virginia",
    "Hampton": "Hampton University Virginia",
    "Norfolk St.": "Norfolk State University Virginia",
    "Coppin St.": "Coppin State University Baltimore Maryland",
    "Md.-Eastern Shore": "University of Maryland Eastern Shore Princess Anne",
    "Morgan St.": "Morgan State University Baltimore Maryland",
    "Delaware St.": "Delaware State University Dover",
    "Howard": "Howard University Washington DC",
    "SC Upstate": "University of South Carolina Upstate Spartanburg",
    "Cal Baptist": "California Baptist University Riverside",
    "Grand Canyon": "Grand Canyon University Phoenix Arizona",
    "Abilene Christian": "Abilene Christian University Texas",
    "Tarleton St.": "Tarleton State University Stephenville Texas",
    "Utah Valley": "Utah Valley University Orem",
    "Dixie St.": "Dixie State University St George Utah",
    "Seattle U": "Seattle University Washington",
    "Pacific": "University of the Pacific Stockton California",
    "Santa Clara": "Santa Clara University California",
    "San Francisco": "University of San Francisco California",
    "Gonzaga": "Gonzaga University Spokane Washington",
    "Portland": "University of Portland Oregon",
    "Loyola Marymount": "Loyola Marymount University Los Angeles",
    "Pepperdine": "Pepperdine University Malibu California",
    "Saint Mary's": "Saint Mary's College Moraga California",
    "Hawaii": "University of Hawaii Manoa Honolulu",
    "Hawai'i": "University of Hawaii Manoa Honolulu",
    "Texas A&M-CC": "Texas A&M University Corpus Christi",
    "Texas A&M-Corpus Christi": "Texas A&M University Corpus Christi",
    "SE Missouri St.": "Southeast Missouri State University Cape Girardeau",
    "SE Louisiana": "Southeastern Louisiana University Hammond",
    "Lipscomb": "Lipscomb University Nashville Tennessee",
    "Bellarmine": "Bellarmine University Louisville Kentucky",
    "Queens": "Queens University Charlotte North Carolina",
    "Lindenwood": "Lindenwood University St Charles Missouri",
    "W. Michigan": "Western Michigan University Kalamazoo",
    "E. Michigan": "Eastern Michigan University Ypsilanti",
    "N. Illinois": "Northern Illinois University DeKalb",
    "S. Illinois": "Southern Illinois University Carbondale",
    "W. Illinois": "Western Illinois University Macomb",
    "SE Missouri": "Southeast Missouri State University Cape Girardeau",
    "N. Dakota St.": "North Dakota State University Fargo",
    "S. Dakota St.": "South Dakota State University Brookings",
    "Oral Roberts": "Oral Roberts University Tulsa Oklahoma",
    "Omaha": "University of Nebraska Omaha",
    "N. Dakota": "University of North Dakota Grand Forks",
    "S. Dakota": "University of South Dakota Vermillion",
    "Montana St.": "Montana State University Bozeman",
    "Weber St.": "Weber State University Ogden Utah",
    "Idaho St.": "Idaho State University Pocatello",
    "New Mexico St.": "New Mexico State University Las Cruces",
    "Sul Ross St.": "Sul Ross State University Alpine Texas",
    "Sam Houston St.": "Sam Houston State University Huntsville Texas",
    "SLU": "Saint Louis University Missouri",
    "Incarnate Word": "University of the Incarnate Word San Antonio Texas",
    "UAPB": "University of Arkansas Pine Bluff",
    "Little Rock": "University of Arkansas Little Rock",
    "Cent. Michigan": "Central Michigan University Mount Pleasant",
    "N. Kentucky": "Northern Kentucky University Highland Heights",
    "Southern Miss.": "University of Southern Mississippi Hattiesburg",
    "South Alabama": "University of South Alabama Mobile",
    "South Fla.": "University of South Florida Tampa",
    "Stephen F. Austin": "Stephen F. Austin State University Nacogdoches",
    "Troy": "Troy University Alabama",
    "Louisiana": "University of Louisiana Lafayette",
    "Louisiana Tech": "Louisiana Tech University Ruston",
    "UL Monroe": "University of Louisiana Monroe",
    "App State": "Appalachian State University Boone North Carolina",
    "Appalachian St.": "Appalachian State University Boone North Carolina",
    "Marshall": "Marshall University Huntington West Virginia",
    "Rider": "Rider University Lawrenceville New Jersey",
    "Iona": "Iona University New Rochelle New York",
    "Marist": "Marist College Poughkeepsie New York",
    "Manhattan": "Manhattan College Riverdale New York",
    "Siena": "Siena College Loudonville New York",
    "Monmouth": "Monmouth University West Long Branch New Jersey",
    "Quinnipiac": "Quinnipiac University Hamden Connecticut",
    "Sacred Heart": "Sacred Heart University Fairfield Connecticut",
    "Fairfield": "Fairfield University Connecticut",
    "Wagner": "Wagner College Staten Island New York",
    "Mount St. Mary's": "Mount St. Mary's University Emmitsburg Maryland",
    "Merrimack": "Merrimack College North Andover Massachusetts",
    "Stonehill": "Stonehill College Easton Massachusetts",
    "Chicago St.": "Chicago State University",
    "Tex. A&M-CC": "Texas A&M University Corpus Christi",
    "La.-Monroe": "University of Louisiana Monroe",
    "Ark. St.": "Arkansas State University Jonesboro",
    "Cent. Ark.": "University of Central Arkansas Conway",
}


def build_geocode_query(team_name: str) -> str:
    """Build a geocoding query from team name."""
    # Check manual overrides first
    if team_name in TEAM_QUERY_OVERRIDES:
        return TEAM_QUERY_OVERRIDES[team_name]

    # Common patterns
    name = team_name.strip()

    # If it ends with "St." → State University
    if name.endswith(" St."):
        return name.replace(" St.", " State University")

    # Direct university name (most common case)
    # "Florida" → "University of Florida"
    # "Texas" → "University of Texas"
    return f"University of {name}" if not any(
        name.startswith(p) for p in ["University", "College"]
    ) else name


def geocode_team(
    geolocator: Nominatim,
    team_name: str,
    canonical_id: str,
) -> dict | None:
    """Geocode a team to get approximate lat/lon."""
    queries = []

    # Primary query
    primary = build_geocode_query(team_name)
    queries.append(primary)

    # Fallback: team name + "university"
    if "University" not in primary and "College" not in primary:
        queries.append(f"{team_name} University")

    # Fallback: just the team name
    queries.append(team_name)

    for query in queries:
        try:
            location = geolocator.geocode(query, timeout=10, country_codes="us")
            if location:
                return {
                    "lat": round(location.latitude, 4),
                    "lon": round(location.longitude, 4),
                    "geocoded_address": location.address,
                    "query_used": query,
                }
            # Special case: Hawaii
            if "Hawaii" in query or "Hawai" in query:
                location = geolocator.geocode(query, timeout=10)
                if location:
                    return {
                        "lat": round(location.latitude, 4),
                        "lon": round(location.longitude, 4),
                        "geocoded_address": location.address,
                        "query_used": query,
                    }
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"  Geocoding error for '{query}': {e}", file=sys.stderr)
            time.sleep(2)  # Extra delay on error
            continue

    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Expand stadium_orientations.csv to all 308 D1 teams.",
    )
    parser.add_argument("--canonical", type=Path,
                        default=Path("data/registries/canonical_teams_2026.csv"))
    parser.add_argument("--existing", type=Path,
                        default=Path("data/registries/stadium_orientations.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/registries/stadium_orientations.csv"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be added without writing")
    parser.add_argument("--delay", type=float, default=1.1,
                        help="Delay between geocoding requests (Nominatim requires >= 1s)")
    args = parser.parse_args()

    # Load canonical teams
    canonical = pd.read_csv(args.canonical)
    all_teams = canonical[["canonical_id", "team_name"]].drop_duplicates("canonical_id")
    print(f"Total D1 teams: {len(all_teams)}")

    # Load existing stadium data
    existing = pd.DataFrame(columns=["canonical_id", "venue_name", "lat", "lon",
                                      "hp_bearing_deg", "source"])
    if args.existing.exists():
        existing = pd.read_csv(args.existing)
    existing_ids = set(existing["canonical_id"].values)
    print(f"Existing stadiums: {len(existing_ids)}")

    # Find teams needing geocoding
    need_geocoding = all_teams[~all_teams["canonical_id"].isin(existing_ids)]
    print(f"Teams needing geocoding: {len(need_geocoding)}")

    if need_geocoding.empty:
        print("All teams already have stadium data!")
        return 0

    # Initialize geocoder
    geolocator = Nominatim(
        user_agent="ncaa-baseball-model/1.0 (research)",
        timeout=10,
    )

    new_rows = []
    failed = []

    for i, (_, row) in enumerate(need_geocoding.iterrows()):
        cid = row["canonical_id"]
        tname = row["team_name"]

        if i > 0:
            time.sleep(args.delay)  # Rate limit

        result = geocode_team(geolocator, tname, cid)

        if result:
            new_rows.append({
                "canonical_id": cid,
                "venue_name": f"{tname} Baseball Stadium",
                "lat": result["lat"],
                "lon": result["lon"],
                "hp_bearing_deg": 67,
                "source": f"geocoded_nominatim",
            })
            print(f"  [{i+1}/{len(need_geocoding)}] {cid} ({tname}): "
                  f"({result['lat']}, {result['lon']}) [{result['query_used']}]")
        else:
            failed.append((cid, tname))
            print(f"  [{i+1}/{len(need_geocoding)}] {cid} ({tname}): FAILED")

    print(f"\nGeocoded: {len(new_rows)}, Failed: {len(failed)}")
    if failed:
        print("Failed teams:")
        for cid, tname in failed:
            print(f"  {cid}: {tname}")

    if args.dry_run:
        print("\n(Dry run — not writing)")
        return 0

    # Combine existing + new
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates("canonical_id", keep="first")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"\nWritten {len(combined)} stadiums to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
