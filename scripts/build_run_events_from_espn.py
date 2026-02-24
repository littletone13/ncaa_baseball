"""
Build run_events table from ESPN game JSONL (Path A step 6.1).

Only games with non-null run_events (PBP available) are included. Output is used by
the Stan run-event model (team and pitcher indices built separately in step 6.2).

Usage:
  python3 scripts/build_run_events_from_espn.py --espn-dir data/raw/espn --canonical data/registries/canonical_teams_2026.csv --out data/processed/run_events.csv
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build run_events.csv from ESPN JSONL (games with PBP run_events).",
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
        help="Canonical teams CSV for name resolution",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/run_events.csv"),
        help="Output run_events table",
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default="2024,2025,2026",
        help="Comma-separated seasons to include",
    )
    args = parser.parse_args()

    canonical = load_canonical_teams(args.canonical)
    name_to_canonical = build_odds_name_to_canonical(canonical)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]

    rows = []
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
                re = g.get("run_events")
                if not re or not isinstance(re, dict):
                    continue
                home_re = re.get("home") or {}
                away_re = re.get("away") or {}
                if not home_re or not away_re:
                    continue

                event_id = g.get("event_id") or g.get("id") or ""
                game_date = (g.get("date") or "")[:10]
                try:
                    yr = int(g.get("season") or season)
                except (TypeError, ValueError):
                    yr = int(season) if season.isdigit() else 0

                home = g.get("home_team") or {}
                away = g.get("away_team") or {}
                home_name = (home.get("name") or "").strip()
                away_name = (away.get("name") or "").strip()
                if not home_name or not away_name:
                    continue

                home_t, away_t = resolve_odds_teams(
                    home_name, away_name, canonical, name_to_canonical,
                )
                home_canonical = home_t[0] if home_t else ""
                away_canonical = away_t[0] if away_t else ""

                starters = g.get("starters") or {}
                hp = starters.get("home_pitcher") or {}
                ap = starters.get("away_pitcher") or {}
                home_pitcher_espn_id = hp.get("espn_id") or hp.get("id")
                away_pitcher_espn_id = ap.get("espn_id") or ap.get("id")
                if home_pitcher_espn_id is not None:
                    home_pitcher_espn_id = str(home_pitcher_espn_id)
                if away_pitcher_espn_id is not None:
                    away_pitcher_espn_id = str(away_pitcher_espn_id)

                home_score = g.get("home_score")
                away_score = g.get("away_score")
                if home_score is not None:
                    try:
                        home_score = int(home_score)
                    except (TypeError, ValueError):
                        home_score = None
                if away_score is not None:
                    try:
                        away_score = int(away_score)
                    except (TypeError, ValueError):
                        away_score = None

                def get_count(d: dict, key: str) -> int:
                    v = d.get(key)
                    if v is None:
                        return 0
                    try:
                        return int(v)
                    except (TypeError, ValueError):
                        return 0

                rows.append({
                    "event_id": event_id,
                    "game_date": game_date,
                    "season": yr,
                    "home_canonical_id": home_canonical,
                    "away_canonical_id": away_canonical,
                    "home_pitcher_espn_id": home_pitcher_espn_id or "",
                    "away_pitcher_espn_id": away_pitcher_espn_id or "",
                    "home_run_1": get_count(home_re, "run_1"),
                    "home_run_2": get_count(home_re, "run_2"),
                    "home_run_3": get_count(home_re, "run_3"),
                    "home_run_4": get_count(home_re, "run_4"),
                    "away_run_1": get_count(away_re, "run_1"),
                    "away_run_2": get_count(away_re, "run_2"),
                    "away_run_3": get_count(away_re, "run_3"),
                    "away_run_4": get_count(away_re, "run_4"),
                    "home_score": home_score,
                    "away_score": away_score,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "event_id", "game_date", "season",
            "home_canonical_id", "away_canonical_id",
            "home_pitcher_espn_id", "away_pitcher_espn_id",
            "home_run_1", "home_run_2", "home_run_3", "home_run_4",
            "away_run_1", "away_run_2", "away_run_3", "away_run_4",
            "home_score", "away_score",
        ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n_resolved = (df["home_canonical_id"] != "") & (df["away_canonical_id"] != "")
    print(f"Wrote {len(df)} run-event games -> {args.out}")
    print(f"  Both teams resolved to canonical: {n_resolved.sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
