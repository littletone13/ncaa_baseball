"""
Build a game-result table from ESPN JSONL (games_2024.jsonl, games_2025.jsonl).

Resolves team names to canonical_teams_2026. Output: data/processed/games_espn.csv
for use in Phase 1 Elo/BTM fitting.

Usage:
  python3 scripts/build_games_from_espn.py --espn-dir data/raw/espn --canonical data/registries/canonical_teams_2026.csv --out data/processed/games_espn.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from ncaa_baseball.phase1 import (
    build_odds_name_to_canonical,
    load_canonical_teams,
    resolve_odds_teams,
)


def stream_espn_games(path: Path):
    """Yield one game dict per line (only fields we need)."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                g = json.loads(line)
            except json.JSONDecodeError:
                continue
            home = g.get("home_team") or {}
            away = g.get("away_team") or {}
            home_name = (home.get("name") or "").strip()
            away_name = (away.get("name") or "").strip()
            home_score = g.get("home_score")
            away_score = g.get("away_score")
            if not home_name or not away_name:
                continue
            if home_score is None or away_score is None:
                continue
            try:
                home_score = int(home_score)
                away_score = int(away_score)
            except (TypeError, ValueError):
                continue
            yield {
                "date": (g.get("date") or "").strip(),
                "season": g.get("season"),
                "home_team_name": home_name,
                "away_team_name": away_name,
                "home_score": home_score,
                "away_score": away_score,
                "winner_home": 1 if home_score > away_score else 0,
            }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build games_espn.csv from ESPN JSONL with canonical resolution.",
    )
    parser.add_argument(
        "--espn-dir",
        type=Path,
        default=Path("data/raw/espn"),
        help="Directory containing games_YYYY.jsonl",
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=Path("data/registries/canonical_teams_2026.csv"),
        help="Canonical teams CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/games_espn.csv"),
        help="Output games CSV",
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default="2024,2025",
        help="Comma-separated seasons to include (e.g. 2024,2025)",
    )
    args = parser.parse_args()

    canonical = load_canonical_teams(args.canonical)
    name_to_canonical = build_odds_name_to_canonical(canonical)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]

    rows = []
    for season in seasons:
        path = args.espn_dir / f"games_{season}.jsonl"
        if not path.exists():
            print(f"Skip (not found): {path}")
            continue
        n = 0
        for g in stream_espn_games(path):
            home_t, away_t = resolve_odds_teams(
                g["home_team_name"],
                g["away_team_name"],
                canonical,
                name_to_canonical,
            )
            row = {
                "date": g["date"],
                "season": g["season"],
                "home_team_name": g["home_team_name"],
                "away_team_name": g["away_team_name"],
                "home_score": g["home_score"],
                "away_score": g["away_score"],
                "winner_home": g["winner_home"],
                "canonical_home_id": home_t[0] if home_t else None,
                "canonical_away_id": away_t[0] if away_t else None,
            }
            rows.append(row)
            n += 1
        print(f"  {path.name}: {n} games")

    if not rows:
        print("No games collected.")
        return 1

    df = pd.DataFrame(rows)
    resolved = df["canonical_home_id"].notna() & df["canonical_away_id"].notna()
    print(f"Total: {len(df)} games, resolved: {resolved.sum()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
