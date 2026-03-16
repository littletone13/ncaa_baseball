#!/usr/bin/env python3
"""
scrape_gameday_starters.py — Scrape actual game-day starters from NCAA Sidearm live stats.

NCAA schools use Sidearm Sports for their live stats pages. These pages are publicly
accessible and show lineups + starting pitchers typically 30-60 minutes before first pitch.

The JSON data behind these pages is available at:
  https://{school_domain}/services/schedule_txt.ashx?format=json&schedule={schedule_id}
or the live stats page itself parses lineup data.

This script:
  1. Takes today's schedule (schedule.csv)
  2. For each game, attempts to find the sidearm live stats URL
  3. Scrapes lineup and starting pitcher info
  4. Outputs corrections to starters.csv when our projection was wrong

Usage:
  python3 scripts/scrape_gameday_starters.py --date 2026-03-16
  python3 scripts/scrape_gameday_starters.py --date 2026-03-16 --schedule data/daily/2026-03-16/schedule.csv

For manual override (when you can see the lineups on the sidearm page):
  python3 scripts/scrape_gameday_starters.py --date 2026-03-16 --manual \\
    --game 1 --home-starter "Colton Kennedy" --away-starter "Dillon Schueler"
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd


# ── ESPN In-Game Starters ────────────────────────────────────────────────────

def scrape_espn_starters(date: str) -> dict[str, dict]:
    """
    Once games are in progress or completed, ESPN shows actual starters.
    Returns {espn_event_id: {"home_starter": name, "away_starter": name, "status": str}}
    """
    date_fmt = date.replace("-", "")
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/baseball/"
        f"college-baseball/scoreboard?dates={date_fmt}&limit=200"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ncaa-baseball-model/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
        print(f"ESPN API error: {e}", file=sys.stderr)
        return {}

    results = {}
    for ev in data.get("events", []):
        event_id = ev.get("id", "")
        comp = ev.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {}).get("description", "")

        if status not in ("In Progress", "Final", "End of Period"):
            continue

        # Get game detail for pitcher info
        detail_url = (
            f"https://site.api.espn.com/apis/site/v2/sports/baseball/"
            f"college-baseball/summary?event={event_id}"
        )
        try:
            req2 = urllib.request.Request(detail_url, headers={"User-Agent": "ncaa-baseball-model/1.0"})
            with urllib.request.urlopen(req2, timeout=15) as resp2:
                detail = json.loads(resp2.read().decode())
        except Exception:
            continue

        boxscore = detail.get("boxscore", {})
        players = boxscore.get("players", [])

        home_starter = ""
        away_starter = ""
        home_team = ""
        away_team = ""

        for p in players:
            team_name = p.get("team", {}).get("displayName", "")
            home_away = p.get("homeAway", "")

            if home_away == "home":
                home_team = team_name
            else:
                away_team = team_name

            for stat_group in p.get("statistics", []):
                if stat_group.get("type") == "pitching":
                    for athlete in stat_group.get("athletes", []):
                        if athlete.get("starter"):
                            name = athlete.get("athlete", {}).get("displayName", "")
                            if home_away == "home":
                                home_starter = name
                            else:
                                away_starter = name

        if home_starter or away_starter:
            results[event_id] = {
                "home_team": home_team,
                "away_team": away_team,
                "home_starter": home_starter,
                "away_starter": away_starter,
                "status": status,
            }

    return results


def update_starters_csv(
    starters_csv: Path,
    corrections: dict[int, dict],
    pitcher_table_csv: Path = Path("data/processed/pitcher_table.csv"),
) -> pd.DataFrame:
    """
    Apply starter corrections to starters.csv.

    corrections: {game_num: {"home_starter": name, "away_starter": name}}
    """
    st = pd.read_csv(starters_csv, dtype=str)
    pt = pd.read_csv(pitcher_table_csv, dtype=str)
    pt["pitcher_idx"] = pd.to_numeric(pt["pitcher_idx"], errors="coerce").fillna(0).astype(int)

    changes = []
    for game_num, corr in corrections.items():
        mask = st["game_num"].astype(str) == str(game_num)
        if not mask.any():
            print(f"  WARNING: game_num {game_num} not in starters.csv", file=sys.stderr)
            continue

        row_idx = mask.idxmax()
        h_cid = str(st.loc[row_idx, "home_canonical_id"]).strip()
        a_cid = str(st.loc[row_idx, "away_canonical_id"]).strip()

        for side, cid, col_starter, col_idx in [
            ("home", h_cid, "home_starter", "home_starter_idx"),
            ("away", a_cid, "away_starter", "away_starter_idx"),
        ]:
            new_name = corr.get(f"{side}_starter", "").strip()
            old_name = str(st.loc[row_idx, col_starter]).strip()
            if not new_name or new_name.lower() == old_name.lower():
                continue

            # Find pitcher in pitcher_table
            matches = pt[
                (pt["pitcher_name"].str.lower().str.contains(new_name.lower(), na=False))
                & (pt["team_canonical_id"] == cid)
            ]
            if matches.empty:
                # Try just last name
                last = new_name.split()[-1]
                matches = pt[
                    (pt["pitcher_name"].str.lower().str.contains(last.lower(), na=False))
                    & (pt["team_canonical_id"] == cid)
                ]

            new_idx = 0
            if not matches.empty:
                best = matches.sort_values("pitcher_idx", ascending=False).iloc[0]
                new_idx = int(best["pitcher_idx"])

            st.loc[row_idx, col_starter] = new_name
            st.loc[row_idx, col_idx] = str(new_idx)
            st.loc[row_idx, f"{side}_resolution_method"] = "gameday_override"

            changes.append({
                "game_num": game_num,
                "side": side,
                "old": old_name,
                "new": new_name,
                "idx": new_idx,
            })
            print(f"  Game {game_num} {side}: {old_name} → {new_name} (idx={new_idx})",
                  file=sys.stderr)

    if changes:
        st.to_csv(starters_csv, index=False)
        print(f"\n  Updated {len(changes)} starters in {starters_csv}", file=sys.stderr)
    else:
        print("  No starter changes needed.", file=sys.stderr)

    return st


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape game-day starters and update starters.csv"
    )
    parser.add_argument("--date", type=str, required=True, help="Game date YYYY-MM-DD")
    parser.add_argument("--schedule", type=Path, default=None)
    parser.add_argument("--starters", type=Path, default=None)
    parser.add_argument("--pitcher-table", type=Path,
                        default=Path("data/processed/pitcher_table.csv"))

    # Manual override mode
    parser.add_argument("--manual", action="store_true", help="Manual starter override mode")
    parser.add_argument("--game", type=int, help="Game number to override")
    parser.add_argument("--home-starter", type=str, default="")
    parser.add_argument("--away-starter", type=str, default="")

    args = parser.parse_args()
    repo_root = Path(__file__).parent.parent
    date = args.date

    starters_csv = args.starters or Path(f"data/daily/{date}/starters.csv")
    starters_csv = repo_root / starters_csv if not starters_csv.is_absolute() else starters_csv
    pt_csv = repo_root / args.pitcher_table if not args.pitcher_table.is_absolute() else args.pitcher_table

    if not starters_csv.exists():
        print(f"No starters.csv at {starters_csv} — run resolve_starters.py first", file=sys.stderr)
        return 1

    if args.manual:
        if args.game is None:
            print("--manual requires --game N", file=sys.stderr)
            return 1
        corrections = {args.game: {
            "home_starter": args.home_starter,
            "away_starter": args.away_starter,
        }}
        update_starters_csv(starters_csv, corrections, pt_csv)
        return 0

    # Auto mode: scrape ESPN for in-progress/final games
    print(f"Checking ESPN for game-day starters ({date})...", file=sys.stderr)
    espn_starters = scrape_espn_starters(date)

    if not espn_starters:
        print("  No in-progress or final games found on ESPN.", file=sys.stderr)
        print("  For pre-game starters, use --manual mode:", file=sys.stderr)
        print(f"    python3 scripts/scrape_gameday_starters.py --date {date} --manual \\", file=sys.stderr)
        print(f"      --game 0 --home-starter 'Name' --away-starter 'Name'", file=sys.stderr)
        return 0

    # Match ESPN starters to our schedule
    st = pd.read_csv(starters_csv, dtype=str)
    corrections = {}

    for eid, info in espn_starters.items():
        print(f"\n  ESPN: {info['away_team']} @ {info['home_team']} ({info['status']})")
        print(f"    Home SP: {info['home_starter']}")
        print(f"    Away SP: {info['away_starter']}")

        # Find matching game in our schedule
        for _, row in st.iterrows():
            gn = int(row["game_num"])
            # Simple name matching
            h_cid = str(row.get("home_canonical_id", ""))
            old_hp = str(row.get("home_starter", ""))
            old_ap = str(row.get("away_starter", ""))

            # Check if this is the same game (match on team names)
            espn_home = info["home_team"].lower()
            espn_away = info["away_team"].lower()

            # Match by checking if any canonical ID words appear in ESPN name
            h_words = h_cid.lower().replace("bsb_", "").replace("ncaa_", "").split("_")
            if any(w in espn_home for w in h_words if len(w) > 3):
                corr = {}
                if info["home_starter"] and info["home_starter"].lower() != old_hp.lower():
                    corr["home_starter"] = info["home_starter"]
                if info["away_starter"] and info["away_starter"].lower() != old_ap.lower():
                    corr["away_starter"] = info["away_starter"]
                if corr:
                    corrections[gn] = corr
                break

    if corrections:
        update_starters_csv(starters_csv, corrections, pt_csv)
    else:
        print("\n  All starters match our projections.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
