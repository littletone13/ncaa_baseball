"""
Single-pass ESPN JSONL extractor.

Replaces 6 separate build_* scripts that each independently parse the same raw data:
  build_run_events_from_espn.py  → run_events.csv
  build_pitcher_registry.py      → pitcher_appearances.csv
  build_games_from_espn.py       → games_espn.csv
  build_pitching_from_espn.py    → pitching_lines_espn.csv
  build_park_factors.py          → park_factors.csv
  build_bullpen_fatigue.py       → bullpen_quality.csv

This script does ONE pass through each JSONL file and outputs 4 CSVs + a manifest.

Usage:
  python3 scripts/extract_espn.py
  python3 scripts/extract_espn.py --espn-dir data/raw/espn --out-dir data/processed --seasons 2024,2025,2026
  python3 scripts/extract_espn.py --out-dir data/processed/extracted_test
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from ncaa_baseball.phase1 import (
    build_odds_name_to_canonical,
    load_canonical_teams,
    resolve_odds_teams,
)


# ---------------------------------------------------------------------------
# IP parsing: ESPN encodes partial innings as .1 = 1/3, .2 = 2/3
# ---------------------------------------------------------------------------

def parse_ip(raw: str | float | None) -> float | None:
    """Convert ESPN IP string to decimal innings.

    "5.0" -> 5.0, "5.1" -> 5.333, "5.2" -> 5.667.
    """
    if raw is None or raw == "":
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    whole = int(val)
    frac = round(val - whole, 1)
    if abs(frac - 0.1) < 0.01:
        return whole + 1.0 / 3.0
    if abs(frac - 0.2) < 0.01:
        return whole + 2.0 / 3.0
    return val


def _safe_int(val) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _safe_float(x, default=None):
    if x is None or x == "":
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _parse_pc(stats: dict) -> float | None:
    """Extract pitch count from stats dict. Prefer 'PC'; fall back to 'PC-ST'."""
    pc = _safe_float(stats.get("PC"))
    if pc is not None:
        return pc
    pcst = (stats.get("PC-ST") or "").strip()
    if pcst:
        try:
            return float(pcst.split("-")[0])
        except (ValueError, IndexError):
            pass
    return None


# ---------------------------------------------------------------------------
# Team resolution helpers
# ---------------------------------------------------------------------------

def build_resolver(canonical: pd.DataFrame, name_to_canonical: dict):
    """Return a closure that maps a full ESPN team name to canonical_id or ''."""
    def resolve(name: str) -> str:
        if not name:
            return ""
        t, _ = resolve_odds_teams(name, name, canonical, name_to_canonical)
        return t[0] if t else ""
    return resolve


# ---------------------------------------------------------------------------
# Venue accumulator (for venue_stats)
# ---------------------------------------------------------------------------

RUN_EVENT_KEYS = ("run_1", "run_2", "run_3", "run_4")


class VenueAccum:
    """Accumulates per-venue totals for the venue_stats output."""

    __slots__ = (
        "games", "total_home_runs", "total_away_runs",
        "re_games",
        "home_re_totals", "away_re_totals",
        "home_teams",   # Counter of home_canonical_id
    )

    def __init__(self):
        self.games: int = 0
        self.total_home_runs: int = 0
        self.total_away_runs: int = 0
        self.re_games: int = 0
        self.home_re_totals: dict[str, int] = {k: 0 for k in RUN_EVENT_KEYS}
        self.away_re_totals: dict[str, int] = {k: 0 for k in RUN_EVENT_KEYS}
        self.home_teams: Counter = Counter()


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract(
    espn_dir: Path,
    seasons: list[str],
    canonical: pd.DataFrame,
    name_to_canonical: dict,
) -> tuple[list[dict], list[dict], list[dict], dict[tuple, VenueAccum]]:
    """Single pass through all JSONL files.

    Returns:
      games_rows       — one row per game
      run_events_rows  — one row per game with run_events data
      pitcher_rows     — one row per pitcher per game (from boxscore pitching)
      venue_accums     — per-venue accumulator dicts (keyed by (name, city, state))
    """
    resolve = build_resolver(canonical, name_to_canonical)

    games_rows: list[dict] = []
    run_events_rows: list[dict] = []
    pitcher_rows: list[dict] = []
    venue_accums: dict[tuple, VenueAccum] = defaultdict(VenueAccum)

    total_lines = 0
    parse_errors = 0

    for season in seasons:
        path = espn_dir / f"games_{season}.jsonl"
        if not path.exists():
            print(f"  skip (not found): {path}")
            continue

        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_lines += 1

                try:
                    g = json.loads(line)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue

                # ── Core identifiers ──────────────────────────────────────────
                event_id = str(g.get("event_id") or g.get("id") or "")
                game_date = (g.get("date") or "")[:10]
                try:
                    yr = int(g.get("season") or season)
                except (TypeError, ValueError):
                    yr = int(season) if season.isdigit() else 0

                home = g.get("home_team") or {}
                away = g.get("away_team") or {}
                home_name = (home.get("name") or "").strip()
                away_name = (away.get("name") or "").strip()

                # Resolve team names once per game
                home_canonical = resolve(home_name)
                away_canonical = resolve(away_name)

                # Scores
                home_score = _safe_int(g.get("home_score"))
                away_score = _safe_int(g.get("away_score"))

                # Venue
                venue = g.get("venue") or {}
                venue_name = (venue.get("name") or "").strip()
                venue_city = (venue.get("city") or "").strip()
                venue_state = (venue.get("state") or "").strip()

                neutral_site = bool(g.get("neutral_site"))

                # Starters
                starters = g.get("starters") or {}
                hp = starters.get("home_pitcher") or {}
                ap = starters.get("away_pitcher") or {}
                home_pitcher_espn_id = str(hp.get("espn_id") or hp.get("id") or "") or None
                away_pitcher_espn_id = str(ap.get("espn_id") or ap.get("id") or "") or None
                home_pitcher_name = (hp.get("name") or "").strip() or None
                away_pitcher_name = (ap.get("name") or "").strip() or None

                # Run events
                re = g.get("run_events")
                has_run_events = bool(re and isinstance(re, dict))
                home_re = (re.get("home") or {}) if has_run_events else {}
                away_re = (re.get("away") or {}) if has_run_events else {}
                if not (home_re and away_re):
                    has_run_events = False

                # Boxscore pitching
                box = g.get("boxscore") or {}
                has_boxscore = bool(box)

                # ── 1. games row ──────────────────────────────────────────────
                games_rows.append({
                    "event_id": event_id,
                    "game_date": game_date,
                    "season": yr,
                    "home_name": home_name,
                    "away_name": away_name,
                    "home_canonical_id": home_canonical,
                    "away_canonical_id": away_canonical,
                    "home_score": home_score,
                    "away_score": away_score,
                    "winner_home": (
                        bool(home_score is not None and away_score is not None
                             and home_score > away_score)
                        if (home_score is not None and away_score is not None) else None
                    ),
                    "venue_name": venue_name,
                    "venue_city": venue_city,
                    "venue_state": venue_state,
                    "neutral_site": neutral_site,
                    "home_pitcher_espn_id": home_pitcher_espn_id,
                    "away_pitcher_espn_id": away_pitcher_espn_id,
                    "home_pitcher_name": home_pitcher_name,
                    "away_pitcher_name": away_pitcher_name,
                    "has_run_events": has_run_events,
                    "has_boxscore": has_boxscore,
                })

                # ── 2. run_events row (only when PBP available) ───────────────
                if has_run_events:
                    def get_count(d: dict, key: str) -> int:
                        v = d.get(key)
                        if v is None:
                            return 0
                        try:
                            return int(v)
                        except (TypeError, ValueError):
                            return 0

                    run_events_rows.append({
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

                # ── 3. pitcher_appearances rows (from boxscore pitching) ───────
                # The boxscore is keyed by team abbreviation (or id).
                # We try both abbreviations.
                home_abbr = (home.get("abbreviation") or "").strip()
                away_abbr = (away.get("abbreviation") or "").strip()
                home_id_str = str(home.get("id") or "")
                away_id_str = str(away.get("id") or "")

                team_side_map = [
                    (home_abbr, home_id_str, home_name, home_canonical, "home"),
                    (away_abbr, away_id_str, away_name, away_canonical, "away"),
                ]
                for abbr, team_id_str, team_name, team_canonical_id, side in team_side_map:
                    section = box.get(abbr) or box.get(team_id_str) or {}
                    for athlete in section.get("pitching", []):
                        stats = athlete.get("stats") or {}
                        espn_id = athlete.get("espn_id")
                        if espn_id is not None:
                            try:
                                espn_id = str(int(espn_id))
                            except (TypeError, ValueError):
                                espn_id = str(espn_id)
                        pitcher_name = (athlete.get("name") or "").strip()
                        starter = bool(athlete.get("starter"))
                        role = "starter" if starter else "reliever"
                        ip_raw = stats.get("IP")
                        ip = parse_ip(ip_raw)
                        h = _safe_int(stats.get("H"))
                        r = _safe_int(stats.get("R"))
                        er = _safe_int(stats.get("ER"))
                        bb = _safe_int(stats.get("BB"))
                        k = _safe_int(stats.get("K"))
                        hr = _safe_int(stats.get("HR"))
                        pc = _parse_pc(stats)

                        pitcher_rows.append({
                            "event_id": event_id,
                            "game_date": game_date,
                            "season": yr,
                            "pitcher_espn_id": espn_id or "",
                            "pitcher_id": f"ESPN_{espn_id}" if espn_id else "",
                            "pitcher_name": pitcher_name,
                            "team_canonical_id": team_canonical_id,
                            "team_name": team_name,
                            "side": side,
                            "starter": starter,
                            "role": role,
                            "ip": ip,
                            "h": h,
                            "r": r,
                            "er": er,
                            "bb": bb,
                            "k": k,
                            "hr": hr,
                            "pc": pc,
                        })

                # ── 4. venue accumulation (non-neutral, scored games only) ─────
                if (
                    not neutral_site
                    and venue_name
                    and home_score is not None
                    and away_score is not None
                ):
                    vkey = (venue_name, venue_city, venue_state)
                    acc = venue_accums[vkey]
                    acc.games += 1
                    acc.total_home_runs += home_score
                    acc.total_away_runs += away_score
                    if home_canonical:
                        acc.home_teams[home_canonical] += 1
                    if has_run_events:
                        acc.re_games += 1
                        for rk in RUN_EVENT_KEYS:
                            acc.home_re_totals[rk] += get_count(home_re, rk)
                            acc.away_re_totals[rk] += get_count(away_re, rk)

    print(f"  total lines parsed: {total_lines}")
    print(f"  parse errors:       {parse_errors}")
    return games_rows, run_events_rows, pitcher_rows, venue_accums


# ---------------------------------------------------------------------------
# venue_stats builder (aggregation step, runs after the pass)
# ---------------------------------------------------------------------------

def build_venue_stats(venue_accums: dict[tuple, VenueAccum]) -> list[dict]:
    rows = []
    for (vname, vcity, vstate), acc in venue_accums.items():
        if acc.games == 0:
            continue
        total_runs = acc.total_home_runs + acc.total_away_runs
        rpg = total_runs / acc.games

        row: dict = {
            "venue_name": vname,
            "venue_city": vcity,
            "venue_state": vstate,
            "home_canonical_id": (
                acc.home_teams.most_common(1)[0][0] if acc.home_teams else ""
            ),
            "n_games": acc.games,
            "total_home_runs": acc.total_home_runs,
            "total_away_runs": acc.total_away_runs,
            "rpg": round(rpg, 4),
        }

        # Per-run-type averages (only for games with PBP)
        for rk in RUN_EVENT_KEYS:
            if acc.re_games > 0:
                row[f"home_{rk}_avg"] = round(acc.home_re_totals[rk] / acc.re_games, 4)
                row[f"away_{rk}_avg"] = round(acc.away_re_totals[rk] / acc.re_games, 4)
            else:
                row[f"home_{rk}_avg"] = None
                row[f"away_{rk}_avg"] = None

        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Single-pass ESPN JSONL extractor. Produces games, run_events, "
                    "pitcher_appearances, and venue_stats CSVs.",
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
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for all CSV files and manifest",
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default="2024,2025,2026",
        help="Comma-separated seasons to include",
    )
    args = parser.parse_args()

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    print(f"Loading canonical teams from {args.canonical}")
    canonical = load_canonical_teams(args.canonical)
    name_to_canonical = build_odds_name_to_canonical(canonical)

    print(f"Extracting ESPN JSONL from {args.espn_dir} (seasons: {', '.join(seasons)})")

    # ── Single pass ──────────────────────────────────────────────────────────
    games_rows, run_events_rows, pitcher_rows, venue_accums = extract(
        args.espn_dir, seasons, canonical, name_to_canonical,
    )

    # ── Build venue_stats from accumulators ──────────────────────────────────
    venue_rows = build_venue_stats(venue_accums)

    # ── Write outputs ─────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # games.csv
    games_path = args.out_dir / "games.csv"
    games_df = pd.DataFrame(games_rows, columns=[
        "event_id", "game_date", "season",
        "home_name", "away_name",
        "home_canonical_id", "away_canonical_id",
        "home_score", "away_score", "winner_home",
        "venue_name", "venue_city", "venue_state", "neutral_site",
        "home_pitcher_espn_id", "away_pitcher_espn_id",
        "home_pitcher_name", "away_pitcher_name",
        "has_run_events", "has_boxscore",
    ])
    games_df.to_csv(games_path, index=False)
    print(f"  games.csv:               {len(games_df):>6d} rows  →  {games_path}")

    # run_events.csv — schema must be identical to existing file
    run_events_path = args.out_dir / "run_events.csv"
    run_events_df = pd.DataFrame(run_events_rows, columns=[
        "event_id", "game_date", "season",
        "home_canonical_id", "away_canonical_id",
        "home_pitcher_espn_id", "away_pitcher_espn_id",
        "home_run_1", "home_run_2", "home_run_3", "home_run_4",
        "away_run_1", "away_run_2", "away_run_3", "away_run_4",
        "home_score", "away_score",
    ])
    run_events_df.to_csv(run_events_path, index=False)
    print(f"  run_events.csv:          {len(run_events_df):>6d} rows  →  {run_events_path}")

    # pitcher_appearances.csv
    pitcher_path = args.out_dir / "pitcher_appearances.csv"
    pitcher_df = pd.DataFrame(pitcher_rows, columns=[
        "event_id", "game_date", "season",
        "pitcher_espn_id", "pitcher_id", "pitcher_name",
        "team_canonical_id", "team_name", "side", "starter", "role",
        "ip", "h", "r", "er", "bb", "k", "hr", "pc",
    ])
    pitcher_df.to_csv(pitcher_path, index=False)
    print(f"  pitcher_appearances.csv: {len(pitcher_df):>6d} rows  →  {pitcher_path}")

    # venue_stats.csv
    venue_path = args.out_dir / "venue_stats.csv"
    venue_cols = [
        "venue_name", "venue_city", "venue_state", "home_canonical_id",
        "n_games", "total_home_runs", "total_away_runs", "rpg",
        "home_run_1_avg", "home_run_2_avg", "home_run_3_avg", "home_run_4_avg",
        "away_run_1_avg", "away_run_2_avg", "away_run_3_avg", "away_run_4_avg",
    ]
    venue_df = pd.DataFrame(venue_rows, columns=venue_cols)
    venue_df = venue_df.sort_values("n_games", ascending=False).reset_index(drop=True)
    venue_df.to_csv(venue_path, index=False)
    print(f"  venue_stats.csv:         {len(venue_df):>6d} rows  →  {venue_path}")

    # Resolution stats
    re_resolved = (
        (run_events_df["home_canonical_id"] != "")
        & (run_events_df["away_canonical_id"] != "")
    ).sum()
    g_resolved = (
        (games_df["home_canonical_id"] != "")
        & (games_df["away_canonical_id"] != "")
    ).sum()
    print(f"\n  games with both teams resolved:     {g_resolved}/{len(games_df)}")
    print(f"  run_events with both teams resolved: {re_resolved}/{len(run_events_df)}")
    print(f"  unique venues:                       {len(venue_df)}")

    # ── Manifest ─────────────────────────────────────────────────────────────
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seasons": seasons,
        "espn_dir": str(args.espn_dir),
        "canonical": str(args.canonical),
        "outputs": {
            "games": str(games_path),
            "run_events": str(run_events_path),
            "pitcher_appearances": str(pitcher_path),
            "venue_stats": str(venue_path),
        },
        "counts": {
            "games": len(games_df),
            "run_events": len(run_events_df),
            "pitcher_appearances": len(pitcher_df),
            "venues": len(venue_df),
            "games_both_resolved": int(g_resolved),
            "run_events_both_resolved": int(re_resolved),
        },
    }
    manifest_path = args.out_dir / "extract_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  extract_manifest.json             →  {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
