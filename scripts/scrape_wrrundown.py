#!/usr/bin/env python3
"""
scrape_wrrundown.py — Scrape WR Rundown daily college baseball picks & analysis.

URL pattern: https://wrrundown.com/YYYY/MM/DD/M-DD-college-baseball/

Extracts:
  - Individual game picks (team, side, odds, units)
  - Parlays (legs + combined odds)
  - Pick 3 / best bets
  - Full write-up analysis per pick (injury intel, SIERA, wRC+, situational angles)

Output: data/daily/{date}/wrrundown_intel.csv
  Columns: pick_num, pick_raw, team, side, odds, units, parlay_group, analysis, pick_type

Usage:
    .venv/bin/python3 scripts/scrape_wrrundown.py --date 2026-03-20
    .venv/bin/python3 scripts/scrape_wrrundown.py --date 2026-03-20 --no-cache
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


def build_url(date: str) -> str:
    """Build WR Rundown URL from date string YYYY-MM-DD."""
    parts = date.split("-")
    y, m, d = parts[0], parts[1], parts[2]
    # URL pattern: /YYYY/MM/DD/M-DD-college-baseball/
    # Month without leading zero for the slug portion
    m_short = str(int(m))
    d_short = str(int(d))
    return f"https://wrrundown.com/{y}/{m}/{d}/{m_short}-{d_short}-college-baseball/"


def scrape_page(url: str, cache_path: Path, no_cache: bool = False) -> str | None:
    """Scrape the WR Rundown page using firecrawl CLI."""
    if cache_path.exists() and not no_cache:
        print(f"  Using cached scrape: {cache_path}", file=sys.stderr)
        return cache_path.read_text(encoding="utf-8")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Scraping {url}...", file=sys.stderr)

    try:
        result = subprocess.run(
            ["firecrawl", "scrape", url, "-o", str(cache_path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"  firecrawl error: {result.stderr[:200]}", file=sys.stderr)
            return None
    except FileNotFoundError:
        print("  firecrawl CLI not found — install with: npm install -g firecrawl-cli",
              file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print("  firecrawl timed out", file=sys.stderr)
        return None

    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Matches pick lines like "TCU -125", "Minnesota +100", "1.25u: Arkansas + Florida +133 CZR"
PICK_LINE_RE = re.compile(
    r"^(?:(\d+\.?\d*)u:\s*)?"           # optional units prefix "1.25u: "
    r"(.+?)"                             # team(s) / matchup
    r"\s+([-+]\d+)"                      # odds
    r"(?:\s+(\w+))?"                     # optional book (CZR, B365, etc.)
    r"\s*$"
)

# Matches total/under lines: "Texas / Auburn u11.5 -115"
TOTAL_LINE_RE = re.compile(
    r"^(?:(\d+\.?\d*)u:\s*)?"
    r"(.+?)\s+"
    r"([ou])([\d.]+)"                    # o/u + number
    r"\s+([-+]\d+)"                      # odds
    r"(?:\s+(\w+))?"                     # optional book
    r"\s*$"
)

# Matches Pick 3 / best bet section items
PICK3_RE = re.compile(
    r"^(.+?)\s+(ML[P]?|[-+]\d+)"        # team(s) + ML/MLP/odds
    r"\s*$"
)


def extract_analysis(lines: list[str], start_idx: int) -> str:
    """Extract the bullet-point analysis following a pick line.

    Skips blank lines between the pick and its analysis bullets.
    """
    analysis_parts = []
    i = start_idx
    # Skip blank lines between pick and analysis
    while i < len(lines) and not lines[i].strip():
        i += 1
    while i < len(lines):
        line = lines[i].strip()
        # Analysis lines start with "- " (bullet points)
        if line.startswith("- ") or line.startswith("* "):
            analysis_parts.append(line[2:].strip())
        elif line.startswith("-") and len(line) > 1 and line[1] != "-":
            analysis_parts.append(line[1:].strip())
        elif analysis_parts and line and not line.startswith("**") and not line.startswith("#"):
            # Continuation of previous bullet
            if not any(c in line for c in ["**", "Pick 3", "---"]):
                analysis_parts[-1] += " " + line
            else:
                break
        elif not line:
            # Blank line after analysis — stop
            if analysis_parts:
                break
        else:
            break
        i += 1
    return " | ".join(analysis_parts)


def parse_wrrundown(md_text: str) -> list[dict]:
    """Parse WR Rundown markdown into structured picks."""
    # Find the plays section
    lines = md_text.split("\n")
    picks: list[dict] = []
    pick_num = 0
    in_plays = False
    in_pick3 = False
    parlay_group = 0

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Detect start of plays section
        if "**3/" in line and "Plays**" in line:
            in_plays = True
            i += 1
            continue

        # Detect Pick 3 section
        if "**Pick 3" in line or "**pick 3" in line.lower():
            in_pick3 = True
            i += 1
            continue

        # Detect end of content
        if line.startswith("### Share") or line.startswith("### Leave"):
            break

        if not in_plays:
            i += 1
            continue

        # Skip empty lines and non-pick lines
        if not line or line.startswith("![") or line.startswith("[") or line == "---":
            i += 1
            continue

        # Bold lines are pick headers
        if line.startswith("**") and line.endswith("**"):
            i += 1
            continue

        # Try to match a total/under line first (more specific)
        tm = TOTAL_LINE_RE.match(line)
        if tm:
            pick_num += 1
            units = float(tm.group(1)) if tm.group(1) else 1.0
            matchup = tm.group(2).strip()
            ou_side = "over" if tm.group(3) == "o" else "under"
            ou_line = float(tm.group(4))
            odds = int(tm.group(5))
            book = tm.group(6) or ""
            analysis = extract_analysis(lines, i + 1)

            picks.append({
                "pick_num": pick_num,
                "pick_raw": line,
                "team": matchup,
                "side": f"{ou_side} {ou_line}",
                "odds": odds,
                "units": units,
                "book": book,
                "parlay_group": 0,
                "analysis": analysis,
                "pick_type": "total",
            })
            i += 1
            continue

        # Try Pick 3 items
        if in_pick3:
            p3m = PICK3_RE.match(line)
            if p3m:
                pick_num += 1
                picks.append({
                    "pick_num": pick_num,
                    "pick_raw": line,
                    "team": p3m.group(1).strip(),
                    "side": p3m.group(2).strip(),
                    "odds": 0,
                    "units": 0,
                    "book": "",
                    "parlay_group": -1,  # pick3 marker
                    "analysis": "",
                    "pick_type": "pick3",
                })
            i += 1
            continue

        # Try standard pick line
        pm = PICK_LINE_RE.match(line)
        if pm:
            pick_num += 1
            units = float(pm.group(1)) if pm.group(1) else 1.0
            team_str = pm.group(2).strip()
            odds = int(pm.group(3))
            book = pm.group(4) or ""
            analysis = extract_analysis(lines, i + 1)

            # Detect parlays: "Team1 + Team2 +133"
            is_parlay = "+" in team_str and not team_str.startswith("+")
            if is_parlay:
                parlay_group += 1
                legs = [t.strip() for t in team_str.split("+")]
                # Split analysis per leg if possible
                analysis_parts = analysis.split("|") if "|" in analysis else [analysis] * len(legs)

                for j, leg in enumerate(legs):
                    # Clean up leg — remove trailing odds/book from last leg
                    leg = leg.strip()
                    if not leg:
                        continue
                    picks.append({
                        "pick_num": pick_num,
                        "pick_raw": line,
                        "team": leg,
                        "side": "ml",
                        "odds": odds,
                        "units": units,
                        "book": book,
                        "parlay_group": parlay_group,
                        "analysis": analysis_parts[j].strip() if j < len(analysis_parts) else "",
                        "pick_type": "parlay",
                    })
            else:
                # Single game pick — extract team and side
                # Could be "TCU -125" (ML implied) or "Kentucky +122 B365"
                side = "ml"
                picks.append({
                    "pick_num": pick_num,
                    "pick_raw": line,
                    "team": team_str,
                    "side": side,
                    "odds": odds,
                    "units": units,
                    "book": book,
                    "parlay_group": 0,
                    "analysis": analysis,
                    "pick_type": "single",
                })
            i += 1
            continue

        # Unmatched line — might be analysis continuation, skip
        i += 1

    return picks


def write_csv(picks: list[dict], out_csv: Path) -> None:
    """Write parsed picks to CSV."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pick_num", "pick_raw", "team", "side", "odds", "units",
        "book", "parlay_group", "analysis", "pick_type",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(picks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape WR Rundown college baseball picks.")
    parser.add_argument("--date", required=True, help="Game date YYYY-MM-DD")
    parser.add_argument("--no-cache", action="store_true", help="Force re-scrape even if cached")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path override")
    args = parser.parse_args()

    url = build_url(args.date)
    print(f"WR Rundown: {url}", file=sys.stderr)

    cache_dir = Path(".firecrawl")
    cache_path = cache_dir / f"wrrundown-{args.date}.md"

    md_text = scrape_page(url, cache_path, no_cache=args.no_cache)
    if not md_text:
        print("  No content scraped — page may not exist yet", file=sys.stderr)
        return 1

    picks = parse_wrrundown(md_text)
    if not picks:
        print("  No picks found in page", file=sys.stderr)
        return 0

    out_csv = args.out or Path(f"data/daily/{args.date}/wrrundown_intel.csv")
    write_csv(picks, out_csv)

    # Summary
    singles = sum(1 for p in picks if p["pick_type"] == "single")
    parlays = len(set(p["parlay_group"] for p in picks if p["pick_type"] == "parlay"))
    totals = sum(1 for p in picks if p["pick_type"] == "total")
    pick3s = sum(1 for p in picks if p["pick_type"] == "pick3")
    with_analysis = sum(1 for p in picks if p["analysis"])

    print(f"\n  {len(picks)} picks parsed: {singles} singles, {parlays} parlays, "
          f"{totals} totals, {pick3s} pick3", file=sys.stderr)
    print(f"  {with_analysis} picks have analysis write-ups", file=sys.stderr)
    print(f"  Output: {out_csv}", file=sys.stderr)

    # Print picks summary
    print(f"\n  {'#':>2s}  {'Type':>7s}  {'Team':>25s}  {'Side':>12s}  {'Odds':>5s}  {'Units':>5s}", file=sys.stderr)
    print("  " + "-" * 70, file=sys.stderr)
    for p in picks:
        print(f"  {p['pick_num']:>2d}  {p['pick_type']:>7s}  {p['team']:>25s}  "
              f"{p['side']:>12s}  {p['odds']:>+5d}  {p['units']:>5.2f}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
