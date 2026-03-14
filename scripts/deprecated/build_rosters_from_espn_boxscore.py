"""
Build roster tables from ESPN game JSONL boxscore data.

Aggregates every player who appears in a game boxscore (batting or pitching)
per team per season. Resolves team names to canonical_teams_2026. Use as
fallback when NCAA roster scraping returns 403.

- Input: data/raw/espn/games_2024.jsonl, games_2025.jsonl, etc.
- Output: data/processed/rosters/rosters_espn_boxscore.csv
  Columns: canonical_id, team_name, season, player_name, espn_id, position, roles
  Position: from ESPN when present, else inferred (P / B / P/B from batting/pitching role).

Usage:
  python3 scripts/build_rosters_from_espn_boxscore.py \\
    --espn-dir data/raw/espn \\
    --canonical data/registries/canonical_teams_2026.csv \\
    --out data/processed/rosters/rosters_espn_boxscore.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from ncaa_baseball.phase1 import (
    _normalize_for_match,
    build_odds_name_to_canonical,
    load_canonical_teams,
)


def _collect_players_from_boxscore(box: dict, team_abbr: str) -> list[tuple[str, str | None, str, str]]:
    """Return list of (player_name, espn_id, role, position) for one team's boxscore section."""
    out = []
    section = box.get(team_abbr) or {}
    for cat, role in ("batting", "batting"), ("pitching", "pitching"):
        for athlete in section.get(cat, []):
            name = (athlete.get("name") or "").strip()
            if not name:
                continue
            espn_id = athlete.get("espn_id")
            if espn_id is not None:
                try:
                    espn_id = str(int(espn_id))
                except (TypeError, ValueError):
                    espn_id = str(espn_id) if espn_id else None
            pos = (athlete.get("position") or "").strip()
            if pos and "unspecified" in pos.lower():
                pos = ""
            out.append((name, espn_id, role, pos))
    return out


def _infer_position(roles: set[str], positions_seen: set[str]) -> str:
    """Infer position from roles when ESPN position is missing or UN. positions_seen = non-empty from API."""
    if positions_seen:
        # Use most common: take first (we could count; for now join if multiple)
        return ",".join(sorted(positions_seen))
    if roles == {"pitching"}:
        return "P"
    if roles == {"batting"}:
        return "B"
    if roles == {"batting", "pitching"}:
        return "P/B"
    return ""


def aggregate_rosters_from_espn_jsonl(
    espn_dir: Path,
) -> dict[tuple[str, str, int], dict[tuple[str, str | None], tuple[set[str], set[str]]]]:
    """
    Stream ESPN game JSONL files and aggregate (espn_team_id, espn_team_name, season)
    -> { (player_name, espn_id): (roles, positions_seen) }.
    """
    # (espn_team_id, espn_team_name, season) -> (player_name, espn_id) -> (roles, positions)
    rosters: dict[
        tuple[str, str, int],
        dict[tuple[str, str | None], tuple[set[str], set[str]]],
    ] = {}

    for path in sorted(espn_dir.glob("games_*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    g = json.loads(line)
                except json.JSONDecodeError:
                    continue
                box = g.get("boxscore") or {}
                if not box:
                    continue
                season = g.get("season")
                if season is None:
                    continue
                try:
                    season = int(season)
                except (TypeError, ValueError):
                    continue

                home = g.get("home_team") or {}
                away = g.get("away_team") or {}
                home_id = str((home.get("id") or ""))
                away_id = str((away.get("id") or ""))
                home_name = (home.get("name") or "").strip()
                away_name = (away.get("name") or "").strip()
                home_abbr = (home.get("abbreviation") or "").strip()
                away_abbr = (away.get("abbreviation") or "").strip()
                if not home_id or not away_id:
                    continue

                for team_id, team_name, abbr in (
                    (home_id, home_name, home_abbr),
                    (away_id, away_name, away_abbr),
                ):
                    if not team_name or not abbr:
                        continue
                    key = (team_id, team_name, season)
                    if key not in rosters:
                        rosters[key] = {}
                    for name, espn_id, role, pos in _collect_players_from_boxscore(box, abbr):
                        player_key = (name, espn_id)
                        if player_key not in rosters[key]:
                            rosters[key][player_key] = (set(), set())
                        roles_set, positions_set = rosters[key][player_key]
                        roles_set.add(role)
                        if pos:
                            positions_set.add(pos)
                        rosters[key][player_key] = (roles_set, positions_set)

    return rosters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build rosters from ESPN game boxscores (fallback when NCAA is blocked)"
    )
    parser.add_argument(
        "--espn-dir",
        type=Path,
        default=Path("data/raw/espn"),
        help="Directory containing games_*.jsonl",
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=Path("data/registries/canonical_teams_2026.csv"),
        help="Canonical teams CSV for name resolution",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/rosters/rosters_espn_boxscore.csv"),
        help="Output roster CSV",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    canonical = load_canonical_teams(args.canonical)
    name_to_canonical = build_odds_name_to_canonical(canonical)

    # Resolve ESPN display name -> (canonical_id, ncaa_teams_id); we only need canonical_id and team_name
    def resolve(espn_name: str) -> tuple[str, str] | None:
        n = _normalize_for_match(espn_name)
        if n in name_to_canonical:
            cid, _ = name_to_canonical[n]
            row = canonical[canonical["canonical_id"] == cid].iloc[0]
            return (cid, (row.get("team_name") or cid))
        # Longest-prefix match
        rows_sorted = sorted(
            [
                (len((row.get("team_name") or "").strip()), row)
                for _, row in canonical.iterrows()
                if (row.get("team_name") or "").strip()
            ],
            key=lambda x: -x[0],
        )
        for _, row in rows_sorted:
            tn = (row.get("team_name") or "").strip()
            if tn and n.startswith(_normalize_for_match(tn)):
                return (row["canonical_id"], tn)
        return None

    rosters = aggregate_rosters_from_espn_jsonl(args.espn_dir)

    rows = []
    resolved_teams = 0
    for (espn_id, espn_name, season), players in rosters.items():
        res = resolve(espn_name)
        if res is None:
            continue
        canonical_id, team_name = res
        resolved_teams += 1
        for (player_name, espn_id_str), (roles_set, positions_set) in players.items():
            roles_str = ",".join(sorted(roles_set))
            position = _infer_position(roles_set, positions_set)
            rows.append({
                "canonical_id": canonical_id,
                "team_name": team_name,
                "season": season,
                "player_name": player_name,
                "espn_id": espn_id_str or "",
                "position": position,
                "roles": roles_str,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=["canonical_id", "team_name", "season", "player_name", "espn_id", "position", "roles"]
        )
    else:
        df = df.sort_values(["canonical_id", "season", "player_name"]).drop_duplicates()
    df.to_csv(args.out, index=False)
    n_teams = df["canonical_id"].nunique() if not df.empty else 0
    print(f"Wrote {len(df)} player-season rows for {n_teams} teams to {args.out}")
    print(f"Resolved {resolved_teams} ESPN team-season keys to canonical.")


if __name__ == "__main__":
    main()
