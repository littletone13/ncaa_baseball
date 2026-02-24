"""
Build pitching lines table from ESPN game JSONL.

One row per pitcher appearance: game_date, team, pitcher, starter, IP, ER, etc.
Used to compute team SP/RP strength and pitcher workload (stamina).

Usage:
  python3 scripts/build_pitching_from_espn.py --espn-dir data/raw/espn --canonical data/registries/canonical_teams_2026.csv --out data/processed/pitching_lines_espn.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import pandas as pd

import _bootstrap  # noqa: F401
from ncaa_baseball.phase1 import (
    build_odds_name_to_canonical,
    load_canonical_teams,
    resolve_odds_teams,
)


def _safe_float(x, default=None):
    if x is None or x == "":
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _collect_pitching_rows(
    game_date: str,
    event_id: str,
    season: int,
    home_team: dict,
    away_team: dict,
    box: dict,
    resolve_team: Callable[[str], str | None],
) -> list[dict]:
    rows = []
    for (team_id, team_name, abbr, home_away) in (
        (home_team.get("id"), home_team.get("name"), home_team.get("abbreviation"), "home"),
        (away_team.get("id"), away_team.get("name"), away_team.get("abbreviation"), "away"),
    ):
        section = box.get(abbr or "") or box.get(str(team_id or "")) or {}
        for athlete in section.get("pitching", []):
            stats = athlete.get("stats") or {}
            ip = _safe_float(stats.get("IP"))
            er = _safe_float(stats.get("ER"))
            r = _safe_float(stats.get("R"), er)
            bb = _safe_float(stats.get("BB"), 0)
            k = _safe_float(stats.get("K"), 0)
            hr = _safe_float(stats.get("HR"), 0)
            pc = _safe_float(stats.get("PC"))
            espn_id = athlete.get("espn_id")
            if espn_id is not None:
                try:
                    espn_id = str(int(espn_id))
                except (TypeError, ValueError):
                    espn_id = str(espn_id)
            name = (athlete.get("name") or "").strip()
            starter = bool(athlete.get("starter"))
            canonical_id = (resolve_team((team_name or "").strip()) or "") if team_name else ""
            rows.append({
                "game_date": game_date,
                "event_id": str(event_id),
                "season": season,
                "team_abbr": (abbr or "").strip(),
                "team_name": (team_name or "").strip(),
                "canonical_id": canonical_id or "",
                "home_away": home_away,
                "pitcher_espn_id": espn_id or "",
                "pitcher_name": name,
                "starter": starter,
                "IP": ip,
                "ER": er,
                "R": r,
                "BB": bb,
                "K": k,
                "HR": hr,
                "PC": pc,
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build pitching lines from ESPN JSONL.")
    parser.add_argument("--espn-dir", type=Path, default=Path("data/raw/espn"))
    parser.add_argument("--canonical", type=Path, default=Path("data/registries/canonical_teams_2026.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/pitching_lines_espn.csv"))
    parser.add_argument("--seasons", type=str, default="2024,2025,2026", help="Comma-separated seasons")
    args = parser.parse_args()

    canonical = load_canonical_teams(args.canonical)
    name_to_canonical = build_odds_name_to_canonical(canonical)

    def resolve_canonical(team_name: str) -> str | None:
        t = resolve_odds_teams(team_name, team_name, canonical, name_to_canonical)[0]
        return t[0] if t else None

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    all_rows = []
    for season in seasons:
        path = args.espn_dir / f"games_{season}.jsonl"
        if not path.exists():
            continue
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
                game_date = (g.get("date") or "")[:10]
                event_id = g.get("event_id") or g.get("id") or ""
                try:
                    sy = int(g.get("season") or season)
                except (TypeError, ValueError):
                    sy = int(season) if isinstance(season, int) else 0
                home = g.get("home_team") or {}
                away = g.get("away_team") or {}
                for r in _collect_pitching_rows(
                    game_date, event_id, sy, home, away, box, resolve_canonical
                ):
                    all_rows.append(r)

    df = pd.DataFrame(all_rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "game_date", "event_id", "season", "team_abbr", "team_name", "canonical_id",
            "home_away", "pitcher_espn_id", "pitcher_name", "starter", "IP", "ER", "R", "BB", "K", "HR", "PC",
        ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n_games = df[["game_date", "event_id"]].drop_duplicates().shape[0]
    print(f"Wrote {len(df)} pitching lines ({n_games} games) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
