#!/usr/bin/env python3
"""
scrape_sidearm_boxscores.py — Scrape box scores from Sidearm-powered team athletics sites.

Extracts pitcher appearances, lineups (with bats L/R/S), linescores, game info
(umpires, weather, attendance) from every completed game on a team's schedule.

Sidearm sites use 3 URL patterns for box scores:
  1. Standard:  /sports/baseball/stats/2026/{opponent}/boxscore/{game_id}
  2. ASPX:      /boxscore.aspx?id={game_id}
  3. Short:     /boxscore/{game_id}

All three render the same table structure (linescore, batting, pitching tables).

Usage:
  # Scrape all completed games for a team
  python3 scripts/scrape_sidearm_boxscores.py --team BSB_TEXAS

  # Scrape a single box score URL
  python3 scripts/scrape_sidearm_boxscores.py --url https://texassports.com/sports/baseball/stats/2026/baylor/boxscore/17043

  # Scrape all teams, limit N games per team
  python3 scripts/scrape_sidearm_boxscores.py --all --limit 3

  # Resume (skip already-scraped game IDs)
  python3 scripts/scrape_sidearm_boxscores.py --team BSB_TEXAS --resume

Output:
  data/raw/sidearm/pitcher_appearances.csv
  data/raw/sidearm/lineups.csv
  data/raw/sidearm/game_info.csv
  data/raw/sidearm/run_events.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd


# ── Paths ────────────────────────────────────────────────────────────────────

OUT_DIR = Path("data/raw/sidearm")
PITCHER_CSV = OUT_DIR / "pitcher_appearances.csv"
LINEUP_CSV = OUT_DIR / "lineups.csv"
GAME_INFO_CSV = OUT_DIR / "game_info.csv"
RUN_EVENT_CSV = OUT_DIR / "run_events.csv"

ROSTER_CSV = Path("data/processed/sidearm_rosters.csv")
SIDEARM_URLS_MODULE = Path("scripts/sidearm_urls.py")
CANONICAL_CSV = Path("data/registries/canonical_teams_2026.csv")

RATE_LIMIT_SEC = 1.5
PAGE_LOAD_WAIT = 3.0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_sidearm_urls() -> dict[str, str]:
    """Load SIDEARM_URLS from scripts/sidearm_urls.py."""
    sys.path.insert(0, str(Path(__file__).parent))
    from sidearm_urls import SIDEARM_URLS  # type: ignore[import-untyped]
    return SIDEARM_URLS


def _load_roster_bats() -> dict[tuple[str, str], str]:
    """Load (canonical_id, player_name_lower) → bats from sidearm_rosters.csv."""
    if not ROSTER_CSV.exists():
        return {}
    try:
        df = pd.read_csv(ROSTER_CSV, dtype=str)
        lookup = {}
        for _, row in df.iterrows():
            cid = row.get("canonical_id", "")
            name = (row.get("player_name", "") or "").strip().lower()
            bats = (row.get("bats", "") or "").strip().upper()
            if cid and name and bats in ("L", "R", "S"):
                lookup[(cid, name)] = bats
        return lookup
    except Exception:
        return {}


def _normalize_name(name: str) -> str:
    """Normalize player name for ID generation."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _parse_ip(ip_str: str) -> float:
    """Convert IP string like '3.1' or '6.2' to fractional innings.

    In baseball notation, .1 = 1/3, .2 = 2/3.
    """
    try:
        parts = ip_str.strip().split(".")
        whole = int(parts[0])
        if len(parts) > 1:
            thirds = int(parts[1])
            return whole + thirds / 3.0
        return float(whole)
    except (ValueError, IndexError):
        return 0.0


def _safe_int(val: str) -> int:
    """Parse integer, return 0 on failure."""
    try:
        return int(val.strip())
    except (ValueError, AttributeError):
        return 0


def _extract_game_id(url: str) -> str:
    """Extract a game identifier from a Sidearm box score URL."""
    # Standard: /sports/baseball/stats/2026/baylor/boxscore/17043
    m = re.search(r"/boxscore/(\d+)", url)
    if m:
        return m.group(1)
    # ASPX: /boxscore.aspx?id=14541
    m = re.search(r"[?&]id=(\d+)", url)
    if m:
        return m.group(1)
    # Fallback: last path segment
    return url.rstrip("/").rsplit("/", 1)[-1]


MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _extract_date_from_text(text: str) -> str:
    """Extract game date from page title or body text.

    Handles formats:
      - 'Baseball vs Baylor on 2/28/2026' (standard Sidearm title)
      - 'March 27, 2026' or 'Mar 27, 2026' (ASPX body text)
      - 'Fri, Feb. 13 (2026)' (some Sidearm variants)
    """
    # Try M/D/YYYY
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"

    # Try "Month DD, YYYY" or "Mon DD, YYYY"
    m = re.search(
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\b\.?\s+(\d{1,2}),?\s+(\d{4})",
        text, re.IGNORECASE,
    )
    if m:
        month = MONTH_MAP.get(m.group(1).lower().rstrip("."), 0)
        if month:
            return f"{m.group(3)}-{month:02d}-{int(m.group(2)):02d}"

    # Try "Mon. DD (YYYY)" format
    m = re.search(
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\b\.?\s+(\d{1,2})\s*\((\d{4})\)",
        text, re.IGNORECASE,
    )
    if m:
        month = MONTH_MAP.get(m.group(1).lower().rstrip("."), 0)
        if month:
            return f"{m.group(3)}-{month:02d}-{int(m.group(2)):02d}"

    # Try "Month DD" without year — assume current season (2026)
    m = re.search(
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\b\.?\s+(\d{1,2})\b",
        text, re.IGNORECASE,
    )
    if m:
        month = MONTH_MAP.get(m.group(1).lower().rstrip("."), 0)
        if month:
            return f"2026-{month:02d}-{int(m.group(2)):02d}"

    return ""


# ── Schedule Discovery ───────────────────────────────────────────────────────

def discover_box_score_urls(
    page, base_url: str, domain: str,
    date_after: str = "", date_before: str = "9999-99-99",
) -> list[str]:
    """Find box score URLs from a team's schedule page.

    Returns absolute URLs, optionally filtered by date range using the
    link title attribute (e.g., 'Box Score ... on March 29').
    """
    schedule_url = f"https://{domain}/sports/baseball/schedule"
    try:
        page.goto(schedule_url, wait_until="domcontentloaded", timeout=20000)
    except Exception as e:
        print(f"    Schedule load failed: {e}", file=sys.stderr)
        return []
    time.sleep(PAGE_LOAD_WAIT)

    raw_links = page.evaluate("""
    (() => {
        return Array.from(document.querySelectorAll('a'))
            .filter(a => {
                const href = (a.getAttribute('href') || '').toLowerCase();
                const text = (a.textContent || '').toLowerCase().trim();
                return (text === 'box score' || text === 'boxscore') ||
                       href.includes('boxscore');
            })
            .map(a => {
                // Collect date hints from multiple sources
                let dateHint = a.getAttribute('title') || a.getAttribute('aria-label') || '';

                // Also try href slug (e.g., /boxscore/baseball-vs-texas-4-2-26/)
                const href = a.getAttribute('href') || '';

                // Walk up DOM to find <time> element or date in container text
                if (!dateHint) {
                    let el = a;
                    for (let i = 0; i < 20 && el; i++) {
                        el = el.parentElement;
                        if (!el) break;
                        // Check <time> element
                        const timeEl = el.querySelector('time');
                        if (timeEl) {
                            dateHint = timeEl.getAttribute('datetime') || timeEl.textContent || '';
                            break;
                        }
                        // Check container text for date patterns (e.g., "FriFeb 13" or "Mar 29")
                        const text = (el.textContent || '').replace(/\\s+/g, ' ').substring(0, 300);
                        const dm = text.match(/((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\\s*\\.?\\s*(\\d{1,2})/i);
                        if (dm && text.length < 500) {
                            dateHint = dm[1] + ' ' + dm[2];
                            break;
                        }
                    }
                }

                // Try extracting date from URL slug (e.g., -3-29-26 or -4-2-26)
                if (!dateHint) {
                    const m = href.match(/(\\d{1,2})-(\\d{1,2})-(\\d{2})\\/?$/);
                    if (m) {
                        dateHint = `${m[1]}/${m[2]}/20${m[3]}`;
                    }
                }

                return {href, dateHint};
            })
            .filter(o => o.href && !o.href.startsWith('javascript:'));
    })()
    """)

    # Deduplicate and make absolute, with date pre-filtering
    seen = set()
    urls = []
    have_date_filter = date_after or date_before != "9999-99-99"

    for item in raw_links:
        href = item["href"]
        if href.startswith("http"):
            full_url = href
        else:
            full_url = f"https://{domain}{href}"
        if full_url in seen:
            continue
        seen.add(full_url)

        # Pre-filter by date using hints from title, <time>, or URL slug
        if have_date_filter and item["dateHint"]:
            hint_date = _extract_date_from_text(item["dateHint"])
            if hint_date:
                if hint_date < date_after or hint_date > date_before:
                    continue  # Skip — outside date range

        urls.append(full_url)

    return urls


# ── Box Score Parsing ────────────────────────────────────────────────────────

def parse_box_score(
    page, url: str, host_canonical_id: str,
    roster_bats: dict[tuple[str, str], str] | None = None,
) -> dict:
    """Parse a single Sidearm box score page.

    Returns dict with keys: pitchers, lineups, game_info, linescore
    """
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=20000)
    except Exception as e:
        return {"error": str(e)}
    time.sleep(PAGE_LOAD_WAIT)

    title = page.title()
    game_date = _extract_date_from_text(title)
    game_id = _extract_game_id(url)
    event_id = f"sidearm_{game_id}"

    # Fallback: extract date from page body if title didn't have it
    if not game_date:
        body_snippet = page.evaluate("document.body.innerText.substring(0, 2000)")
        game_date = _extract_date_from_text(body_snippet)

    # ── Extract all tables ────────────────────────────────────────────────
    # Sidearm puts player names in <th> within body rows, stats in <td>.
    # We extract both separately so parsers can reconstruct correctly.
    tables_data = page.evaluate("""
    (() => {
        const tables = document.querySelectorAll('table');
        return Array.from(tables).map(t => {
            // Column headers from thead
            const theadThs = Array.from(t.querySelectorAll('thead th'))
                .map(th => th.textContent.trim());
            // Body rows: capture both th (player name) and td (stats)
            const rows = Array.from(t.querySelectorAll('tbody tr')).map(tr => {
                const ths = Array.from(tr.querySelectorAll('th'))
                    .map(th => th.textContent.trim());
                const tds = Array.from(tr.querySelectorAll('td'))
                    .map(td => td.textContent.trim());
                return {ths, tds};
            });
            return {headers: theadThs, rows};
        });
    })()
    """)

    # ── Extract team names from linescore (table 0) ──────────────────────
    away_team_name = ""
    home_team_name = ""
    linescore_innings = {}

    for tbl in tables_data:
        hdrs = [h.lower() for h in tbl["headers"]]
        # Linescore table has "Team" header + numeric inning columns (1,2,3...) + R,H,E
        # Note: Sidearm spells it "innning" (triple n) in some templates
        has_team = "team" in hdrs
        # Check for inning-like headers: "1st innning 1", "2nd innning 2", or just "1", "2"
        has_innings = sum(1 for h in hdrs if re.search(r"\b[1-9]\b", h)) >= 3
        if has_team and has_innings and "pos" not in hdrs and "ab" not in hdrs:
            # This is the linescore table (not batting or scoring summary)
            if len(tbl["rows"]) >= 2:
                # In linescore, team name may be in th or first td
                def _get_linescore_row(row_data):
                    """Combine th + td into a flat list for linescore row."""
                    name = row_data["ths"][0] if row_data["ths"] else ""
                    cells = row_data["tds"]
                    if not name and cells:
                        name = cells[0]
                        cells = cells[1:]
                    return name, cells

                away_name, away_cells = _get_linescore_row(tbl["rows"][0])
                home_name, home_cells = _get_linescore_row(tbl["rows"][1])
                away_team_name = away_name
                home_team_name = home_name

                # Parse per-inning runs
                # Headers: Team, 1st/1, 2nd/2, ..., R, H, E
                inning_hdrs = tbl["headers"][1:]  # skip "Team"
                for side, cells in [("away", away_cells), ("home", home_cells)]:
                    innings = {}
                    for i, (hdr, val) in enumerate(zip(inning_hdrs, cells)):
                        # Extract inning number from headers like "1st innning 1"
                        # or just "1", "2", etc. Skip R/H/E columns.
                        hdr_lower = hdr.lower().strip()
                        if hdr_lower.startswith(("runs", "hits", "errors", "r", "h", "e")):
                            continue
                        # Find the last digit in the header (e.g., "1st innning 1" → 1)
                        digits = re.findall(r"\d+", hdr)
                        if digits:
                            inning_num = int(digits[-1])
                            runs = _safe_int(val) if val.upper() != "X" else 0
                            innings[inning_num] = runs
                    linescore_innings[side] = innings
            break

    # ── Identify pitching and batting tables ────────────────────────────
    pitching_tables = []
    batting_tables = []

    for tbl in tables_data:
        hdrs_lower = [h.lower() for h in tbl["headers"]]
        if "ip" in hdrs_lower and "er" in hdrs_lower:
            pitching_tables.append(tbl)
        elif "pos" in hdrs_lower and "ab" in hdrs_lower:
            batting_tables.append(tbl)

    # ── Parse pitcher appearances ────────────────────────────────────────
    pitchers = []

    # First pitching table = away team, second = home team
    for idx, tbl in enumerate(pitching_tables[:2]):
        side = "away" if idx == 0 else "home"
        team_name = away_team_name if side == "away" else home_team_name
        team_cid = host_canonical_id if side == "home" else ""

        # Map header positions — headers are column names from thead
        # In Sidearm, "Player" is a header but the actual player name is in
        # the body row's <th>, and stats are in <td> cells.
        # So we map stat columns against the td-only headers (skip "Player").
        hdrs_lower = [h.lower() for h in tbl["headers"]]
        # Build col_map for td cells (stats only, player name is in th)
        stat_headers = [h for h in hdrs_lower if h != "player"]
        col_map = {}
        for col_name in ("ip", "h", "r", "er", "bb", "so", "wp", "bk", "hbp", "ibb", "ab", "bf", "fo", "go", "np"):
            if col_name in stat_headers:
                col_map[col_name] = stat_headers.index(col_name)

        for row_idx, row_data in enumerate(tbl["rows"]):
            # Player name is in th, stats in td
            player_name = row_data["ths"][0].strip() if row_data["ths"] else ""
            cells = row_data["tds"]

            # If no th, try first td as player name
            if not player_name and cells:
                player_name = cells[0].strip()
                cells = cells[1:]

            if not player_name or player_name.lower() in ("totals", "total"):
                continue

            def _get(col: str) -> str:
                idx = col_map.get(col, -1)
                return cells[idx] if 0 <= idx < len(cells) else "0"

            pitcher = {
                "event_id": event_id,
                "game_date": game_date,
                "season": 2026,
                "pitcher_espn_id": "",
                "pitcher_id": f"SIDEARM_{_normalize_name(player_name)}__{team_cid or 'AWAY'}",
                "pitcher_name": player_name,
                "team_canonical_id": team_cid,
                "team_name": team_name,
                "side": side,
                "starter": row_idx == 0,
                "role": "starter" if row_idx == 0 else "reliever",
                "ip": _parse_ip(_get("ip")),
                "h": _safe_int(_get("h")),
                "r": _safe_int(_get("r")),
                "er": _safe_int(_get("er")),
                "bb": _safe_int(_get("bb")),
                "k": _safe_int(_get("so")),
                "hr": 0,  # Not always in Sidearm pitching table
                "pc": _safe_int(_get("np")),
            }
            pitchers.append(pitcher)

    # ── Parse lineups (batting tables) ───────────────────────────────────
    lineups = []

    for idx, tbl in enumerate(batting_tables[:2]):
        side = "away" if idx == 0 else "home"
        team_name = away_team_name if side == "away" else home_team_name
        team_cid = host_canonical_id if side == "home" else ""

        # Batting table: player name in <th>, pos + stats in <td>
        # Headers (thead): Player, Pos, AB, R, H, ...
        hdrs_lower = [h.lower() for h in tbl["headers"]]
        stat_headers = [h for h in hdrs_lower if h != "player"]
        pos_col = stat_headers.index("pos") if "pos" in stat_headers else None

        batting_order = 0
        for row_data in tbl["rows"]:
            player_name = row_data["ths"][0].strip() if row_data["ths"] else ""
            cells = row_data["tds"]

            # If no th, try first td
            if not player_name and cells:
                player_name = cells[0].strip()
                cells = cells[1:]

            if not player_name or player_name.lower() in ("totals", "total"):
                continue

            # Check if substitute (leading whitespace in original name)
            raw_name = row_data["ths"][0] if row_data["ths"] else ""
            is_sub = raw_name != raw_name.lstrip()

            position = ""
            if pos_col is not None and pos_col < len(cells):
                position = cells[pos_col].strip()

            # Only count non-pitcher non-sub entries for batting order
            if position.upper() != "P" and not is_sub:
                batting_order += 1

            lineup_entry = {
                "event_id": event_id,
                "game_date": game_date,
                "team_canonical_id": team_cid,
                "team_name": team_name,
                "side": side,
                "batting_order": batting_order if not is_sub else 0,
                "player_name": player_name,
                "position": position,
                "bats": (roster_bats or {}).get(
                    (team_cid, player_name.strip().lower()), "",
                ),
                "is_substitute": is_sub,
            }
            lineups.append(lineup_entry)

    # ── Parse game info from page text ───────────────────────────────────
    page_text = page.evaluate("document.body.innerText")

    game_info = {
        "event_id": event_id,
        "game_date": game_date,
        "away_team": away_team_name,
        "home_team": home_team_name,
        "source_url": url,
        "host_canonical_id": host_canonical_id,
    }

    # Extract win/loss/save
    wls_match = re.search(r"Win:\s*(.+?)(?:\n|Loss:)", page_text)
    if wls_match:
        game_info["win_pitcher"] = wls_match.group(1).strip()
    loss_match = re.search(r"Loss:\s*(.+?)(?:\n|Save:|$)", page_text)
    if loss_match:
        game_info["loss_pitcher"] = loss_match.group(1).strip()
    save_match = re.search(r"Save:\s*(.+?)(?:\n|$)", page_text)
    if save_match:
        game_info["save_pitcher"] = save_match.group(1).strip()

    # ── Build run events from linescore ──────────────────────────────────
    run_events = []
    for side in ("away", "home"):
        innings = linescore_innings.get(side, {})
        for inning_num, runs in innings.items():
            if runs > 0:
                run_events.append({
                    "event_id": event_id,
                    "game_date": game_date,
                    "season": 2026,
                    "team_name": away_team_name if side == "away" else home_team_name,
                    "team_canonical_id": "" if side == "away" else host_canonical_id,
                    "side": side,
                    "inning": inning_num,
                    "half": "top" if side == "away" else "bottom",
                    "runs": runs,
                    "run_event": f"run_{min(runs, 4)}",
                })

    return {
        "pitchers": pitchers,
        "lineups": lineups,
        "game_info": game_info,
        "run_events": run_events,
        "game_date": game_date,
        "event_id": event_id,
    }


# ── CSV Writers ──────────────────────────────────────────────────────────────

PITCHER_COLS = [
    "event_id", "game_date", "season", "pitcher_espn_id", "pitcher_id",
    "pitcher_name", "team_canonical_id", "team_name", "side", "starter",
    "role", "ip", "h", "r", "er", "bb", "k", "hr", "pc",
]

LINEUP_COLS = [
    "event_id", "game_date", "team_canonical_id", "team_name", "side",
    "batting_order", "player_name", "position", "bats", "is_substitute",
]

GAME_INFO_COLS = [
    "event_id", "game_date", "away_team", "home_team", "source_url",
    "host_canonical_id", "win_pitcher", "loss_pitcher", "save_pitcher",
]

RUN_EVENT_COLS = [
    "event_id", "game_date", "season", "team_name", "team_canonical_id",
    "side", "inning", "half", "runs", "run_event",
]


def _append_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Append rows to a CSV file, creating with header if needed."""
    write_header = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _load_scraped_event_ids(path: Path) -> set[str]:
    """Load already-scraped event IDs from a CSV."""
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["event_id"], dtype=str)
        return set(df["event_id"].dropna().unique())
    except Exception:
        return set()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape Sidearm box scores")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--team", help="Canonical ID of team to scrape (e.g., BSB_TEXAS)")
    group.add_argument("--url", help="Direct box score URL to scrape")
    group.add_argument("--all", action="store_true", help="Scrape all teams")
    parser.add_argument("--limit", type=int, default=0, help="Max games per team (0 = all)")
    parser.add_argument("--after", help="Only scrape games on or after this date (YYYY-MM-DD)")
    parser.add_argument("--before", help="Only scrape games on or before this date (YYYY-MM-DD)")
    parser.add_argument("--resume", action="store_true", help="Skip already-scraped games")
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless (default)")
    parser.add_argument("--no-headless", dest="headless", action="store_false", help="Show browser")
    args = parser.parse_args()

    sidearm_urls = _get_sidearm_urls()
    roster_bats = _load_roster_bats()
    print(f"Sidearm URLs loaded: {len(sidearm_urls)} teams", file=sys.stderr)
    if roster_bats:
        print(f"Roster bats loaded: {len(roster_bats)} players", file=sys.stderr)

    # Determine which teams to scrape
    if args.url:
        teams_to_scrape = []  # Will handle single URL below
    elif args.team:
        if args.team not in sidearm_urls:
            print(f"Error: {args.team} not in sidearm_urls.py", file=sys.stderr)
            return 1
        teams_to_scrape = [(args.team, sidearm_urls[args.team])]
    else:  # --all
        teams_to_scrape = list(sidearm_urls.items())

    # Date filtering
    date_after = args.after or ""
    date_before = args.before or "9999-99-99"
    if args.after or args.before:
        print(f"Date filter: {date_after or '*'} to {date_before if args.before else '*'}",
              file=sys.stderr)

    # Load already-scraped event IDs for resume
    scraped_ids = set()
    if args.resume:
        scraped_ids = _load_scraped_event_ids(PITCHER_CSV)
        print(f"Already scraped: {len(scraped_ids)} games", file=sys.stderr)

    from playwright.sync_api import sync_playwright

    total_pitchers = 0
    total_games = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        page = browser.new_page()

        if args.url:
            # Single URL mode
            print(f"Scraping: {args.url}", file=sys.stderr)
            # Guess canonical_id from URL domain
            from urllib.parse import urlparse
            parsed = urlparse(args.url)
            domain = parsed.netloc.lstrip("www.")
            host_cid = ""
            for cid, dom in sidearm_urls.items():
                if dom == domain or dom in domain:
                    host_cid = cid
                    break

            result = parse_box_score(page, args.url, host_cid, roster_bats)
            if "error" in result:
                print(f"Error: {result['error']}", file=sys.stderr)
                return 1

            _write_results(result)
            total_games = 1
            total_pitchers = len(result["pitchers"])
        else:
            # ── Phase 1: Discover box score URLs from schedule pages ─────
            print(f"\n--- Phase 1: Discovering box score URLs from "
                  f"{len(teams_to_scrape)} teams ---", file=sys.stderr)

            # Collect (url, canonical_id) pairs
            all_box_urls: list[tuple[str, str]] = []

            for team_idx, (cid, domain) in enumerate(teams_to_scrape):
                if (team_idx + 1) % 20 == 0 or team_idx == 0:
                    print(f"  [{team_idx+1}/{len(teams_to_scrape)}] "
                          f"Scanning schedules... ({len(all_box_urls)} URLs so far)",
                          file=sys.stderr)

                box_urls = discover_box_score_urls(
                    page, f"https://{domain}", domain, date_after, date_before,
                )

                for u in box_urls:
                    eid = f"sidearm_{_extract_game_id(u)}"
                    if eid not in scraped_ids:
                        all_box_urls.append((u, cid))

                if args.limit > 0 and len(all_box_urls) >= args.limit * len(teams_to_scrape):
                    break

                time.sleep(0.5)  # Light rate limit for schedule pages

            print(f"\n--- Phase 2: Scraping {len(all_box_urls)} box scores ---",
                  file=sys.stderr)

            # ── Phase 2: Scrape each box score ───────────────────────────
            for game_idx, (box_url, cid) in enumerate(all_box_urls):
                game_id = _extract_game_id(box_url)
                event_id = f"sidearm_{game_id}"

                print(f"  [{game_idx+1}/{len(all_box_urls)}] {game_id}...",
                      end="", file=sys.stderr)

                result = parse_box_score(page, box_url, cid, roster_bats)
                if "error" in result:
                    print(f" ERROR: {result['error']}", file=sys.stderr)
                    time.sleep(RATE_LIMIT_SEC)
                    continue

                gd = result.get("game_date", "")
                n_p = len(result["pitchers"])

                # Post-filter by date (for URLs that passed pre-filter without dates)
                if gd and (gd < date_after or gd > date_before):
                    print(f" {gd} — skip (outside date range)",
                          file=sys.stderr)
                    time.sleep(RATE_LIMIT_SEC)
                    continue

                print(f" {gd or '?'} — {n_p} pitchers", file=sys.stderr)

                _write_results(result)
                total_games += 1
                total_pitchers += n_p
                scraped_ids.add(event_id)

                time.sleep(RATE_LIMIT_SEC)

        browser.close()

    print(f"\nDone: {total_games} games, {total_pitchers} pitcher appearances",
          file=sys.stderr)
    print(f"Output: {OUT_DIR}/", file=sys.stderr)
    return 0


def _write_results(result: dict) -> None:
    """Write parsed box score data to CSV files."""
    if result.get("pitchers"):
        _append_csv(PITCHER_CSV, result["pitchers"], PITCHER_COLS)
    if result.get("lineups"):
        _append_csv(LINEUP_CSV, result["lineups"], LINEUP_COLS)
    if result.get("game_info"):
        _append_csv(GAME_INFO_CSV, [result["game_info"]], GAME_INFO_COLS)
    if result.get("run_events"):
        _append_csv(RUN_EVENT_CSV, result["run_events"], RUN_EVENT_COLS)


if __name__ == "__main__":
    raise SystemExit(main())
