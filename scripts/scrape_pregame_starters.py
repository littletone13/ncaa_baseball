#!/usr/bin/env python3
"""
scrape_pregame_starters.py — Fetch probable starters from ESPN pregame data.

ESPN publishes probable starters via their scoreboard API once lineups are
available (typically 1-4 hours before game time). This script checks all
games for a given date and updates starters.csv with confirmed starters.

Usage:
  # Check all games and update starters
  python3 scripts/scrape_pregame_starters.py --date 2026-03-17

  # Dry run (show what would change, don't write)
  python3 scripts/scrape_pregame_starters.py --date 2026-03-17 --dry-run

Data sources:
  1. ESPN scoreboard API (probable pitchers, when published)
  2. Manual overrides file (data/daily/{date}/starter_overrides.csv)

The override file format (CSV):
  game_num,side,pitcher_name,source
  5,home,Dillon Schueler,sidearm
  12,away,Jacob Thompson,twitter
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/scoreboard"


def fetch_espn_starters(date: str) -> list[dict]:
    """
    Fetch probable/actual starters from ESPN scoreboard API.

    Returns list of dicts:
        {espn_game_id, home_team, away_team, home_starter, away_starter,
         home_starter_id, away_starter_id, status}
    """
    params = {"dates": date.replace("-", ""), "limit": 200}
    try:
        resp = requests.get(ESPN_SCOREBOARD, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  ESPN API error: {e}", file=sys.stderr)
        return []

    results = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {}).get("name", "")

        teams = {}
        for team_entry in comp.get("competitors", []):
            ht = team_entry.get("homeAway", "")
            team_name = team_entry.get("team", {}).get("displayName", "")
            # Look for probable pitcher in linescores or roster
            prob_pitcher = None
            prob_pitcher_id = None
            for leader in team_entry.get("leaders", []):
                if leader.get("name") == "probablePitchers":
                    athletes = leader.get("leaders", [])
                    if athletes:
                        ath = athletes[0].get("athlete", {})
                        prob_pitcher = ath.get("displayName", "")
                        prob_pitcher_id = ath.get("id", "")
            # Also check probables at competition level
            for prob in comp.get("probables", []):
                if prob.get("homeAway") == ht:
                    ath = prob.get("athlete", {})
                    prob_pitcher = ath.get("displayName", prob_pitcher)
                    prob_pitcher_id = ath.get("id", prob_pitcher_id)

            teams[ht] = {
                "team": team_name,
                "starter": prob_pitcher,
                "starter_id": prob_pitcher_id,
            }

        if "home" in teams and "away" in teams:
            results.append({
                "espn_game_id": event.get("id"),
                "home_team": teams["home"]["team"],
                "away_team": teams["away"]["team"],
                "home_starter": teams["home"]["starter"],
                "away_starter": teams["away"]["starter"],
                "home_starter_espn_id": teams["home"]["starter_id"],
                "away_starter_espn_id": teams["away"]["starter_id"],
                "status": status,
            })

    return results


def load_overrides(override_csv: Path) -> dict:
    """Load manual starter overrides. Returns {(game_num, side): pitcher_name}."""
    if not override_csv.exists():
        return {}
    df = pd.read_csv(override_csv, dtype=str)
    overrides = {}
    for _, r in df.iterrows():
        gn = int(r["game_num"])
        side = r["side"]  # "home" or "away"
        overrides[(gn, side)] = {
            "name": r["pitcher_name"],
            "source": r.get("source", "manual"),
        }
    return overrides


def update_starters(
    date: str,
    daily_dir: Path,
    pitcher_table_csv: Path,
    canonical_csv: Path,
    dry_run: bool = False,
) -> int:
    """
    Update starters.csv with ESPN probable pitchers and manual overrides.

    Returns number of starters updated.
    """
    starters_csv = daily_dir / "starters.csv"
    if not starters_csv.exists():
        print(f"  No starters.csv at {starters_csv}", file=sys.stderr)
        return 0

    starters = pd.read_csv(starters_csv, dtype=str)

    # Load pitcher table for index lookups
    pt = pd.read_csv(pitcher_table_csv, dtype=str)
    # Build name+team → pitcher_idx lookup
    pitcher_lookup: dict[tuple[str, str], dict] = {}
    for _, r in pt.iterrows():
        name = str(r.get("pitcher_name", "")).strip().lower()
        team = str(r.get("canonical_id", "")).strip()
        idx = int(r.get("pitcher_idx", 0))
        if name and team:
            pitcher_lookup[(name, team)] = {
                "idx": idx,
                "throws": r.get("throws", ""),
                "fb_sens": r.get("fb_sensitivity", "1.0"),
            }

    # Load ESPN name → canonical mapping
    canon = pd.read_csv(canonical_csv, dtype=str)
    espn_to_canon = {}
    for _, r in canon.iterrows():
        en = str(r.get("espn_name", "")).strip()
        cid = str(r.get("canonical_id", "")).strip()
        if en:
            espn_to_canon[en] = cid

    # Fetch ESPN starters
    espn_starters = fetch_espn_starters(date)
    print(f"  ESPN: {len(espn_starters)} games found", file=sys.stderr)

    # Map ESPN starters to our games
    espn_map: dict[str, dict] = {}  # canonical_id → starter info
    for es in espn_starters:
        for side in ["home", "away"]:
            team = es[f"{side}_team"]
            starter = es[f"{side}_starter"]
            if starter and team:
                cid = espn_to_canon.get(team)
                if cid:
                    espn_map[cid] = {
                        "name": starter,
                        "espn_id": es.get(f"{side}_starter_espn_id", ""),
                    }

    n_espn = sum(1 for v in espn_map.values() if v["name"])
    print(f"  ESPN: {n_espn} starters resolved to canonical teams", file=sys.stderr)

    # Load manual overrides
    override_csv = daily_dir / "starter_overrides.csv"
    overrides = load_overrides(override_csv)
    if overrides:
        print(f"  Overrides: {len(overrides)} manual entries", file=sys.stderr)

    # Apply updates
    updated = 0
    for idx, row in starters.iterrows():
        gn = int(row["game_num"])
        for side, cid_col, name_col, idx_col in [
            ("home", "home_cid", "home_starter", "home_starter_idx"),
            ("away", "away_cid", "away_starter", "away_starter_idx"),
        ]:
            cid = str(row.get(cid_col, "")).strip()
            current_name = str(row.get(name_col, "")).strip()

            # Priority: manual override > ESPN > existing
            new_name = None
            new_source = None

            if (gn, side) in overrides:
                new_name = overrides[(gn, side)]["name"]
                new_source = overrides[(gn, side)]["source"]
            elif cid in espn_map and espn_map[cid]["name"]:
                new_name = espn_map[cid]["name"]
                new_source = "espn_pregame"

            if new_name and new_name.lower() != current_name.lower():
                old_name = current_name
                starters.at[idx, name_col] = new_name
                starters.at[idx, f"{side[0]}p_resolution"] = new_source

                # Try to find pitcher index
                lookup_key = (new_name.lower(), cid)
                if lookup_key in pitcher_lookup:
                    pi = pitcher_lookup[lookup_key]
                    starters.at[idx, idx_col] = str(pi["idx"])

                if dry_run:
                    print(f"  [DRY] Game {gn} {side}: {old_name} → {new_name} ({new_source})")
                else:
                    print(f"  Game {gn} {side}: {old_name} → {new_name} ({new_source})")
                updated += 1

    if not dry_run and updated > 0:
        starters.to_csv(starters_csv, index=False)
        print(f"\n  Updated {updated} starters in {starters_csv}", file=sys.stderr)

    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch pregame starters from ESPN + overrides.")
    parser.add_argument("--date", required=True, help="Game date YYYY-MM-DD")
    parser.add_argument("--pitcher-table", type=Path,
                        default=Path("data/processed/pitcher_table.csv"))
    parser.add_argument("--canonical", type=Path,
                        default=Path("data/registries/canonical_teams_2026.csv"))
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()

    daily_dir = Path(f"data/daily/{args.date}")
    if not daily_dir.exists():
        print(f"No daily directory: {daily_dir}", file=sys.stderr)
        return 1

    n = update_starters(
        date=args.date,
        daily_dir=daily_dir,
        pitcher_table_csv=args.pitcher_table,
        canonical_csv=args.canonical,
        dry_run=args.dry_run,
    )

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Total starters updated: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
