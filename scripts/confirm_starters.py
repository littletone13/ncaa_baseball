#!/usr/bin/env python3
"""
confirm_starters.py — Multi-source midweek starter confirmation.

Searches team websites (SidearmSports), Twitter/X, and web sources to
confirm or discover starting pitchers for games where D1Baseball rotations
don't cover (Mon-Thu midweek games).

Sources (in priority order):
  1. SidearmSports team athletic sites — game previews/notes
  2. Twitter/X team accounts + beat reporters — day-of announcements
  3. Google web search — catches press releases, local media
  4. Manual overrides CSV — user-provided confirmations

Usage:
  python3 scripts/confirm_starters.py --date 2026-03-31
  python3 scripts/confirm_starters.py --date 2026-03-31 --schedule data/daily/2026-03-31/schedule.csv
  python3 scripts/confirm_starters.py --date 2026-03-31 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote_plus

import requests

import _bootstrap  # noqa: F401

# ── Starter-related keywords ────────────────────────────────────────────────
STARTER_PATTERNS = [
    r"(?:probable|projected|expected|scheduled)\s+(?:starter|starting pitcher)[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)",
    r"([A-Z][a-z]+ [A-Z][a-z]+)\s+(?:gets the start|will start|on the mound|takes the bump|gets the ball|to start)",
    r"(?:starting pitcher|SP|starter)[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)",
    r"(?:RHP|LHP)\s+([A-Z][a-z]+ [A-Z][a-z]+)\s+(?:starts?|gets|on the|takes|will)",
    r"([A-Z][a-z]+ [A-Z][a-z]+)\s+\((?:RHP|LHP)\)\s+(?:vs|@|at)",
]


def _extract_pitcher_name(text: str) -> list[str]:
    """Extract probable pitcher names from text using regex patterns."""
    names = []
    for pattern in STARTER_PATTERNS:
        for match in re.finditer(pattern, text):
            name = match.group(1).strip()
            # Filter out common false positives
            if name.lower() not in ("the game", "this week", "head coach", "game day",
                                     "the series", "home run", "first pitch", "game one"):
                names.append(name)
    return list(dict.fromkeys(names))  # dedupe preserving order


def _load_sidearm_registry(path: Path) -> dict[str, str]:
    """Load canonical_id → sidearm base URL mapping."""
    if not path.exists():
        return {}
    registry = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            cid = row.get("canonical_id", "").strip()
            url = row.get("sidearm_url", "").strip()
            if cid and url:
                registry[cid] = url
    return registry


def search_sidearm_site(base_url: str, team_name: str, date: str) -> list[dict]:
    """Search a SidearmSports team site for game preview/notes mentioning starters."""
    results = []
    # SidearmSports news feeds typically at /news or /sports/baseball/news
    for news_path in ["/sports/baseball/news", "/news"]:
        url = f"{base_url.rstrip('/')}{news_path}"
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "NCAA-Baseball-Model/1.0"})
            if resp.status_code != 200:
                continue
            text = resp.text
            # Look for game preview articles mentioning today's date or opponent
            names = _extract_pitcher_name(text)
            if names:
                results.append({
                    "source": "sidearm",
                    "url": url,
                    "names": names,
                    "confidence": "medium",
                })
        except Exception:
            continue
    return results


def search_twitter_web(team_name: str, date: str) -> list[dict]:
    """
    Search for starting pitcher announcements via web search of Twitter/X.
    Uses Google search with site:twitter.com OR site:x.com filter.
    """
    results = []
    # Format date for search
    month_day = f"{date[5:7]}/{date[8:10]}"
    queries = [
        f'site:twitter.com "{team_name}" "starting pitcher" OR "gets the start" OR "on the mound" {date[5:]}',
        f'site:x.com "{team_name}" "starting pitcher" OR "gets the start" OR "on the mound" {date[5:]}',
    ]
    for query in queries:
        try:
            # Use Google search API or web search
            search_url = f"https://www.google.com/search?q={quote_plus(query)}&num=5"
            resp = requests.get(
                search_url,
                timeout=10,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                },
            )
            if resp.status_code == 200:
                names = _extract_pitcher_name(resp.text)
                if names:
                    results.append({
                        "source": "twitter_web_search",
                        "query": query,
                        "names": names,
                        "confidence": "low",
                    })
            time.sleep(1)  # rate limit Google
        except Exception:
            continue
    return results


def search_web_general(team_name: str, opponent_name: str, date: str) -> list[dict]:
    """General web search for starting pitcher info."""
    results = []
    query = f'"{team_name}" baseball "starting pitcher" OR "probable starter" "{date}"'
    try:
        search_url = f"https://www.google.com/search?q={quote_plus(query)}&num=5"
        resp = requests.get(
            search_url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            },
        )
        if resp.status_code == 200:
            names = _extract_pitcher_name(resp.text)
            if names:
                results.append({
                    "source": "web_search",
                    "query": query,
                    "names": names,
                    "confidence": "low",
                })
    except Exception:
        pass
    return results


def load_overrides(date: str) -> dict[tuple[str, str], dict]:
    """Load manual starter overrides for a date."""
    override_path = Path(f"data/daily/{date}/starter_overrides.csv")
    overrides = {}
    if override_path.exists():
        with open(override_path) as f:
            for row in csv.DictReader(f):
                game_num = row.get("game_num", "").strip()
                side = row.get("side", "").strip()
                name = row.get("pitcher_name", "").strip()
                source = row.get("source", "manual").strip()
                if game_num and side and name:
                    overrides[(game_num, side)] = {
                        "pitcher_name": name,
                        "source": source,
                        "confidence": "high",
                    }
    return overrides


def confirm_starters(
    date: str,
    schedule_csv: Path,
    starters_csv: Path | None = None,
    sidearm_registry_csv: Path = Path("data/registries/sidearm_urls.csv"),
    dry_run: bool = False,
) -> list[dict]:
    """
    Run multi-source starter confirmation for a day's games.

    Returns list of confirmation records with:
      game_num, side, current_starter, confirmed_starter, source, confidence, changed
    """
    import pandas as pd

    schedule = pd.read_csv(schedule_csv, dtype=str)
    starters = pd.read_csv(starters_csv, dtype=str) if starters_csv and starters_csv.exists() else None

    # Load team name mapping
    canon = pd.read_csv("data/registries/canonical_teams_2026.csv", dtype=str)
    cid_to_name = {}
    for _, r in canon.iterrows():
        cid = str(r.get("canonical_id", "")).strip()
        name = str(r.get("team_name", "")).strip()
        if cid and name:
            cid_to_name[cid] = name

    sidearm_registry = _load_sidearm_registry(sidearm_registry_csv)
    overrides = load_overrides(date)

    confirmations = []
    n_games = len(schedule)
    print(f"Confirming starters for {n_games} games on {date}...", file=sys.stderr)

    for _, game in schedule.iterrows():
        game_num = str(game.get("game_num", ""))
        h_cid = str(game.get("home_canonical_id", game.get("home_cid", ""))).strip()
        a_cid = str(game.get("away_canonical_id", game.get("away_cid", ""))).strip()
        h_name = cid_to_name.get(h_cid, h_cid)
        a_name = cid_to_name.get(a_cid, a_cid)

        # Get current starters from starters.csv
        current_hp = current_ap = "unknown"
        if starters is not None:
            row = starters[starters["game_num"].astype(str) == game_num]
            if not row.empty:
                current_hp = str(row.iloc[0].get("home_starter", "unknown"))
                current_ap = str(row.iloc[0].get("away_starter", "unknown"))

        for side, cid, name, opponent, current in [
            ("home", h_cid, h_name, a_name, current_hp),
            ("away", a_cid, a_name, h_name, current_ap),
        ]:
            rec = {
                "game_num": game_num,
                "side": side,
                "canonical_id": cid,
                "team_name": name,
                "current_starter": current,
                "confirmed_starter": None,
                "source": None,
                "confidence": None,
                "changed": False,
            }

            # Priority 1: Manual overrides
            override = overrides.get((game_num, side))
            if override:
                rec["confirmed_starter"] = override["pitcher_name"]
                rec["source"] = override["source"]
                rec["confidence"] = "high"
                rec["changed"] = override["pitcher_name"].lower() != current.lower()
                confirmations.append(rec)
                continue

            # Priority 2: SidearmSports (if URL registered)
            sidearm_url = sidearm_registry.get(cid)
            if sidearm_url and not dry_run:
                hits = search_sidearm_site(sidearm_url, name, date)
                if hits:
                    rec["confirmed_starter"] = hits[0]["names"][0]
                    rec["source"] = "sidearm"
                    rec["confidence"] = "medium"
                    rec["changed"] = hits[0]["names"][0].lower() != current.lower()
                    confirmations.append(rec)
                    continue

            # Priority 3: Twitter/X search
            if not dry_run:
                hits = search_twitter_web(name, date)
                if hits:
                    rec["confirmed_starter"] = hits[0]["names"][0]
                    rec["source"] = "twitter"
                    rec["confidence"] = "low"
                    rec["changed"] = hits[0]["names"][0].lower() != current.lower()
                    confirmations.append(rec)
                    continue

            # Priority 4: General web search
            if not dry_run:
                hits = search_web_general(name, opponent, date)
                if hits:
                    rec["confirmed_starter"] = hits[0]["names"][0]
                    rec["source"] = "web"
                    rec["confidence"] = "low"
                    rec["changed"] = hits[0]["names"][0].lower() != current.lower()
                    confirmations.append(rec)
                    continue

            # No confirmation found
            rec["source"] = "none"
            rec["confidence"] = "unconfirmed"
            confirmations.append(rec)

    return confirmations


def main() -> int:
    parser = argparse.ArgumentParser(description="Confirm midweek starting pitchers.")
    parser.add_argument("--date", required=True, help="Game date YYYY-MM-DD")
    parser.add_argument("--schedule", type=Path, default=None,
                        help="Schedule CSV (default: data/daily/{date}/schedule.csv)")
    parser.add_argument("--starters", type=Path, default=None,
                        help="Current starters CSV (default: data/daily/{date}/starters.csv)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output CSV (default: data/daily/{date}/starter_confirmations.csv)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without making web requests")
    args = parser.parse_args()

    daily_dir = Path(f"data/daily/{args.date}")
    schedule_csv = args.schedule or daily_dir / "schedule.csv"
    starters_csv = args.starters or daily_dir / "starters.csv"
    out_csv = args.out or daily_dir / "starter_confirmations.csv"

    if not schedule_csv.exists():
        print(f"Schedule not found: {schedule_csv}", file=sys.stderr)
        return 1

    confirmations = confirm_starters(
        date=args.date,
        schedule_csv=schedule_csv,
        starters_csv=starters_csv,
        dry_run=args.dry_run,
    )

    # Report
    import pandas as pd
    df = pd.DataFrame(confirmations)

    n_confirmed = (df["confidence"].isin(["high", "medium"])).sum()
    n_low = (df["confidence"] == "low").sum()
    n_unconfirmed = (df["confidence"] == "unconfirmed").sum()
    n_changed = df["changed"].sum()

    print(f"\nStarter Confirmation Report — {args.date}", file=sys.stderr)
    print(f"  Total slots: {len(df)}", file=sys.stderr)
    print(f"  Confirmed (high/medium): {n_confirmed}", file=sys.stderr)
    print(f"  Low confidence: {n_low}", file=sys.stderr)
    print(f"  Unconfirmed: {n_unconfirmed}", file=sys.stderr)
    print(f"  Changes from current: {n_changed}", file=sys.stderr)

    if n_changed > 0:
        changed = df[df["changed"]]
        print(f"\n  STARTER CHANGES:", file=sys.stderr)
        for _, r in changed.iterrows():
            print(f"    Game {r['game_num']} {r['side']}: {r['current_starter']} → {r['confirmed_starter']} ({r['source']})",
                  file=sys.stderr)

    # Write output
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(df)} rows → {out_csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
