#!/usr/bin/env python3
"""
integrate_ncaa_boxscores.py — Merge NCAA boxscore pitcher appearances into
the main pitcher_appearances.csv pipeline.

ESPN only covers ~15% of D1 games with full boxscore data. NCAA stats
(ncaa-api.henrygd.me) covers ~100%. This script:

1. Reads NCAA boxscore JSONL (from scrape_ncaa_boxscores.py)
2. Resolves team names → canonical_id
3. Converts to pitcher_appearances.csv format
4. Merges with existing ESPN-sourced appearances (deduplicates by event_id)

Usage:
  python3 scripts/integrate_ncaa_boxscores.py
  python3 scripts/integrate_ncaa_boxscores.py --ncaa-boxscores data/raw/ncaa/boxscores_2026.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def resolve_team_name(
    team_name: str,
    canonical_map: dict[str, str],
) -> str:
    """Resolve NCAA team display name → canonical_id."""
    if not team_name:
        return ""
    # Try exact match first
    key = team_name.strip().lower()
    if key in canonical_map:
        return canonical_map[key]
    # Try without common suffixes
    for suffix in [" st.", " st", " state"]:
        if key.endswith(suffix):
            short = key[: -len(suffix)]
            if short in canonical_map:
                return canonical_map[short]
    return ""


def build_ncaa_team_map(canonical_csv: Path) -> dict[str, str]:
    """Build name → canonical_id map from all available name columns."""
    ct = pd.read_csv(canonical_csv, dtype=str)
    name_map: dict[str, str] = {}

    for _, row in ct.iterrows():
        cid = str(row.get("canonical_id", "")).strip()
        if not cid:
            continue
        # Index team_name, odds_api_name, espn_name, baseballr_team_name
        for col in ["team_name", "odds_api_name", "espn_name", "baseballr_team_name"]:
            val = str(row.get(col, "")).strip()
            if val and val != "nan":
                name_map[val.lower()] = cid
                # Also without mascot (e.g., "Florida Gators" → "Florida")
                parts = val.split()
                if len(parts) >= 2:
                    # First word only
                    name_map[parts[0].lower()] = cid

    # Load NCAA scoreboard team names from existing appearances for
    # additional resolution hints.
    return name_map


def load_ncaa_boxscores(
    ncaa_jsonl: Path,
    canonical_csv: Path,
) -> pd.DataFrame:
    """
    Parse NCAA boxscore JSONL into pitcher_appearances format.

    Returns DataFrame matching pitcher_appearances.csv columns.
    """
    team_map = build_ncaa_team_map(canonical_csv)

    # Also build map from ncaa_teams_id → canonical_id for numeric ID resolution
    ct = pd.read_csv(canonical_csv, dtype=str)
    ncaa_id_map: dict[str, str] = {}
    for _, row in ct.iterrows():
        cid = str(row.get("canonical_id", "")).strip()
        ncaa_id = str(row.get("ncaa_teams_id", "")).strip()
        if cid and ncaa_id and ncaa_id != "nan":
            ncaa_id_map[ncaa_id] = cid

    rows: list[dict] = []
    n_resolved = 0
    n_unresolved = 0
    unresolved_names: set[str] = set()

    with open(ncaa_jsonl) as f:
        for line in f:
            try:
                game = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            game_id = str(game.get("game_id", ""))
            game_date = game.get("date", "")
            if not game_date:
                continue

            # Determine season from date
            try:
                year = int(game_date[:4])
                season = year
            except (ValueError, IndexError):
                season = 2026

            pitching = game.get("pitching", {})

            for side in ["home", "away"]:
                team_name = str(game.get(f"{side}_team", "")).strip()

                # Resolve team
                cid = resolve_team_name(team_name, team_map)
                if not cid:
                    # Try NCAA ID if embedded
                    team_id = str(game.get(f"{side}_team_id", "")).strip()
                    cid = ncaa_id_map.get(team_id, "")

                if not cid:
                    unresolved_names.add(team_name)
                    n_unresolved += 1
                    continue

                n_resolved += 1
                pitchers = pitching.get(side, [])

                for p in pitchers:
                    name = str(p.get("name", "")).strip()
                    if not name:
                        continue

                    starter = bool(p.get("starter", False))
                    ip = float(p.get("ip", 0) or 0)

                    # Generate a pitcher_id in NCAA_ format
                    pitcher_id = f"NCAA_{game_id}_{name.replace(' ', '_')}"

                    rows.append({
                        "event_id": f"ncaa_{game_id}",
                        "game_date": game_date,
                        "season": season,
                        "pitcher_espn_id": "",  # No ESPN ID for NCAA-sourced data
                        "pitcher_id": pitcher_id,
                        "pitcher_name": name,
                        "team_canonical_id": cid,
                        "team_name": team_name,
                        "side": side,
                        "starter": starter,
                        "role": "starter" if starter else "reliever",
                        "ip": ip,
                        "h": int(p.get("h", 0) or 0),
                        "r": int(p.get("r", 0) or 0),
                        "er": int(p.get("er", 0) or 0),
                        "bb": int(p.get("bb", 0) or 0),
                        "k": int(p.get("k", 0) or 0),
                        "hr": 0,
                        "pc": int(p.get("strikes", 0) or 0),
                    })

    df = pd.DataFrame(rows)

    print(f"  NCAA boxscores: {n_resolved} team-games resolved, "
          f"{n_unresolved} unresolved", file=sys.stderr)
    if unresolved_names:
        top_unresolved = sorted(unresolved_names)[:20]
        print(f"  Top unresolved: {', '.join(top_unresolved)}", file=sys.stderr)

    return df


def merge_appearances(
    espn_appearances: pd.DataFrame,
    ncaa_appearances: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge ESPN and NCAA appearances, deduplicating.

    ESPN data is preferred when both sources cover the same game
    (ESPN has ESPN pitcher IDs). Dedup by (game_date, team, pitcher_name, starter).
    """
    # Mark source
    espn_appearances = espn_appearances.copy()
    ncaa_appearances = ncaa_appearances.copy()
    espn_appearances["_source"] = "espn"
    ncaa_appearances["_source"] = "ncaa"

    # For dedup: create a match key
    def make_key(df: pd.DataFrame) -> pd.Series:
        return (
            df["game_date"].astype(str).str[:10] + "|"
            + df["team_canonical_id"].astype(str) + "|"
            + df["pitcher_name"].astype(str).str.lower().str.strip() + "|"
            + df["starter"].astype(str)
        )

    espn_keys = set(make_key(espn_appearances))

    # Only keep NCAA rows that don't overlap with ESPN
    ncaa_key_series = make_key(ncaa_appearances)
    ncaa_novel = ncaa_appearances[~ncaa_key_series.isin(espn_keys)].copy()

    n_overlap = len(ncaa_appearances) - len(ncaa_novel)
    print(f"  Merge: {len(espn_appearances)} ESPN + {len(ncaa_novel)} novel NCAA "
          f"({n_overlap} overlapping removed)", file=sys.stderr)

    # Combine
    combined = pd.concat([espn_appearances, ncaa_novel], ignore_index=True)
    combined = combined.drop(columns=["_source"], errors="ignore")

    # Sort by date desc
    combined = combined.sort_values("game_date", ascending=False).reset_index(drop=True)

    return combined


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Integrate NCAA boxscore pitcher appearances into pipeline."
    )
    parser.add_argument(
        "--ncaa-boxscores", type=Path, nargs="+",
        default=[
            Path("data/raw/ncaa/boxscores_2026.jsonl"),
        ],
    )
    parser.add_argument(
        "--espn-appearances", type=Path,
        default=Path("data/processed/pitcher_appearances.csv"),
    )
    parser.add_argument(
        "--canonical", type=Path,
        default=Path("data/registries/canonical_teams_2026.csv"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/processed/pitcher_appearances.csv"),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent

    # Load existing ESPN appearances
    espn_path = repo_root / args.espn_appearances
    print(f"Loading ESPN appearances from {espn_path}...", file=sys.stderr)
    espn_app = pd.read_csv(espn_path, dtype=str)
    # Convert IP to float for merge
    espn_app["ip"] = pd.to_numeric(espn_app["ip"], errors="coerce").fillna(0)
    print(f"  {len(espn_app)} ESPN rows", file=sys.stderr)

    # Load and parse NCAA boxscores
    canonical_csv = repo_root / args.canonical
    all_ncaa = []
    for bp in args.ncaa_boxscores:
        bp_full = repo_root / bp if not bp.is_absolute() else bp
        if not bp_full.exists():
            print(f"  Skipping {bp_full} (not found)", file=sys.stderr)
            continue
        print(f"Loading NCAA boxscores from {bp_full}...", file=sys.stderr)
        ncaa_df = load_ncaa_boxscores(bp_full, canonical_csv)
        all_ncaa.append(ncaa_df)
        print(f"  {len(ncaa_df)} pitcher appearances from NCAA", file=sys.stderr)

    if not all_ncaa:
        print("No NCAA boxscore data found.", file=sys.stderr)
        return 1

    ncaa_combined = pd.concat(all_ncaa, ignore_index=True)

    # Merge
    merged = merge_appearances(espn_app, ncaa_combined)

    # Write
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print(f"\nWrote {len(merged)} rows → {out_path}", file=sys.stderr)

    # Stats
    merged["game_date_dt"] = pd.to_datetime(merged["game_date"], errors="coerce")
    n_teams = merged["team_canonical_id"].nunique()
    n_dates = merged["game_date_dt"].nunique()
    latest = merged["game_date_dt"].max()
    print(f"  {n_teams} teams, {n_dates} dates, latest: {latest.date() if pd.notna(latest) else '?'}",
          file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
