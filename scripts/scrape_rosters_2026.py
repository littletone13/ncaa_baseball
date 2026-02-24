"""
Scrape 2026 (2025-26) rosters from stats.ncaa.org for all D1 teams in canonical registry.

Uses roster year_id 614802 from data/registries/ncaa_roster_year_id_lu.csv.
Writes one CSV per team: data/raw/rosters/roster_2026_<ncaa_teams_id>.csv

Usage:
  pip install beautifulsoup4  # if not already installed
  python3 scripts/scrape_rosters_2026.py
  python3 scripts/scrape_rosters_2026.py --limit 5 --sleep 2   # test run

Note: stats.ncaa.org often returns 403 for server-side requests (curl/requests).
If you see "403 blocked" for all teams, try: (1) browser-based scraping (Playwright/Selenium),
(2) running from a different network/VPN, or (3) manual export + import. The script is
ready once the site allows requests (e.g. from a residential IP or browser context).
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROSTER_BASE = "https://stats.ncaa.org/team/{ncaa_teams_id}/roster/{roster_year_id}"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def load_roster_year_id(lookup_path: Path, academic_year: int = 2026) -> int:
    """Get ncaa_roster_year_id for the given academic year."""
    df = pd.read_csv(lookup_path)
    row = df[df["academic_year"] == academic_year]
    if row.empty:
        raise SystemExit(f"No roster year_id for academic_year={academic_year} in {lookup_path}")
    return int(row.iloc[0]["ncaa_roster_year_id"])


def load_team_ids(canonical_path: Path) -> list[tuple[int, str, str]]:
    """Load (ncaa_teams_id, team_name, canonical_id) from canonical_teams_2026.csv."""
    df = pd.read_csv(canonical_path)
    out = []
    for _, r in df.iterrows():
        out.append((int(r["ncaa_teams_id"]), str(r.get("team_name", "")).strip(), str(r.get("canonical_id", "")).strip()))
    return out


def fetch_roster_page(ncaa_teams_id: int, roster_year_id: int, session: requests.Session) -> str:
    """GET roster HTML; return body or raise."""
    url = ROSTER_BASE.format(ncaa_teams_id=ncaa_teams_id, roster_year_id=roster_year_id)
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def parse_roster_html(html: str) -> pd.DataFrame | None:
    """
    Parse NCAA roster table. First table is the roster.
    Returns DataFrame with columns from table (Player -> player_name, player_id from links).
    """
    if not html or "Invalid" in html[:200]:
        return None
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        return None
    tbl = tables[0]
    # Player links: href="/players/123"
    link_map = {}  # player_name -> player_id
    for a in tbl.find_all("a", href=True):
        m = re.match(r"^/players/(\d+)$", a.get("href", "").strip())
        if m:
            name = (a.get_text() or "").strip()
            if name:
                link_map[name] = m.group(1)
    # Table to rows
    rows = []
    trs = tbl.find_all("tr")
    if len(trs) < 2:
        return None
    header = [th.get_text(strip=True) or f"col_{i}" for i, th in enumerate(trs[0].find_all(["th", "td"]))]
    if "Player" not in header:
        return None
    for tr in trs[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all(["td"])]
        if len(cells) != len(header):
            continue
        row = dict(zip(header, cells))
        row["player_id"] = link_map.get(row.get("Player", ""), "")
        if "Player" in row:
            row["player_name"] = row["Player"]
            del row["Player"]
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape 2026 rosters from stats.ncaa.org for canonical D1 teams.",
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=Path("data/registries/canonical_teams_2026.csv"),
        help="Canonical teams CSV",
    )
    parser.add_argument(
        "--roster-lu",
        type=Path,
        default=Path("data/registries/ncaa_roster_year_id_lu.csv"),
        help="Roster year_id lookup CSV",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw/rosters"),
        help="Output directory for roster CSVs",
    )
    parser.add_argument(
        "--academic-year",
        type=int,
        default=2026,
        help="Academic year (default 2026)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between requests",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of teams to scrape (0 = all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing roster files",
    )
    args = parser.parse_args()

    roster_year_id = load_roster_year_id(args.roster_lu, args.academic_year)
    teams = load_team_ids(args.canonical)
    if args.limit:
        teams = teams[: args.limit]
    print(f"Roster year_id for {args.academic_year}: {roster_year_id}")
    print(f"Teams to scrape: {len(teams)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    ok = 0
    blocked = 0
    empty = 0
    for i, (ncaa_id, team_name, canonical_id) in enumerate(teams):
        out_path = args.out_dir / f"roster_{args.academic_year}_{ncaa_id}.csv"
        if out_path.exists() and not args.overwrite:
            ok += 1
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(teams)} ...")
            continue
        try:
            html = fetch_roster_page(ncaa_id, roster_year_id, session)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                blocked += 1
                print(f"  403 blocked: {team_name} ({ncaa_id})")
            else:
                print(f"  HTTP error {team_name}: {e}")
            time.sleep(args.sleep)
            continue
        except Exception as e:
            print(f"  Error {team_name}: {e}")
            time.sleep(args.sleep)
            continue

        df = parse_roster_html(html)
        if df is None or df.empty:
            empty += 1
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(teams)} (empty: {team_name})")
        else:
            df["year"] = args.academic_year
            df["team_id"] = ncaa_id
            df["team_name"] = team_name
            df["canonical_id"] = canonical_id
            df.to_csv(out_path, index=False)
            ok += 1
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(teams)} ...")

        time.sleep(args.sleep)

    print(f"Done. Wrote/skipped: {ok}, empty: {empty}, blocked: {blocked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
