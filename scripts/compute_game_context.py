#!/usr/bin/env python3
"""
compute_game_context.py — Compute contextual adjustments for each game.

Layers computed:
  1. Rest/schedule density — games-in-last-N-days fatigue for position players
  2. Day/night — time-of-day scoring adjustment
  3. Surface type — turf vs grass
  4. Conference strength — non-conference opponent adjustment
  5. Travel distance — away team travel fatigue
  6. Recent form — 7-day scoring momentum

All adjustments are on the log-rate scale (additive to the run-event lambda).
Positive = more runs, negative = fewer runs.

Usage:
  python3 scripts/compute_game_context.py --date 2026-03-31 \\
      --schedule data/daily/2026-03-31/schedule.csv \\
      --out data/daily/2026-03-31/context.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — all on log-rate scale
# ═══════════════════════════════════════════════════════════════════════════════

# ── Rest / schedule density ──────────────────────────────────────────────────
# Teams playing their 4th+ game in 4 days have fatigued bats.
# ~5-8% scoring drop = -0.05 to -0.08 log-rate per game beyond 3-in-4.
REST_WINDOW_DAYS = 4
REST_GAMES_THRESHOLD = 3  # 3 games in 4 days is normal (weekend series)
REST_PENALTY_PER_EXTRA_GAME = -0.025  # per game beyond threshold
REST_DAYS_OFF_BONUS = 0.012  # bonus for 2+ days rest (fresh)
REST_DAYS_OFF_THRESHOLD = 2  # days off to trigger bonus

# ── Day/night ────────────────────────────────────────────────────────────────
# Afternoon games (before 5pm local) score ~0.3 fewer runs than evening games.
# Hitters see the ball worse in afternoon sun, especially at east-facing stadiums.
# Effect is relative to evening baseline (model is calibrated to avg mix).
DAY_GAME_ADJ = -0.015       # ~1.5% fewer runs in day games
NIGHT_GAME_ADJ = 0.008      # ~0.8% more runs in night games
DAY_CUTOFF_HOUR_LOCAL = 17  # before 5pm local = "day game"

# ── Surface type ─────────────────────────────────────────────────────────────
# Turf fields produce ~3-4% more runs than grass (faster grounders, harder hops).
TURF_ADJ = 0.018            # ~1.8% more runs on turf
GRASS_ADJ = 0.0             # baseline

# ── Conference strength (non-conference adjustment) ──────────────────────────
# When a strong-conference team hosts a weak-conference opponent in non-conf play,
# the stats from that game are inflated. We scale the weaker team's offensive
# contribution down slightly to reflect this.
# Uses conference_rank (1=strongest, 30+=weakest) differential.
CONF_STRENGTH_SCALE = 0.003  # per rank differential, applied to weaker team's offense

# ── Travel distance ──────────────────────────────────────────────────────────
# Long-distance travel (500+ miles) fatigues the away team, especially midweek.
# Cross-country trips (1500+ miles) have a measurable scoring penalty.
TRAVEL_PENALTY_THRESHOLD_MILES = 500
TRAVEL_PENALTY_PER_500MI = -0.008  # ~0.8% fewer runs per 500 miles beyond threshold
TRAVEL_MAX_PENALTY = -0.025  # cap at ~2.5% penalty

# ── Recent form / momentum ───────────────────────────────────────────────────
# 7-day scoring rate vs season average. Regressed toward mean (50% weight).
FORM_WINDOW_DAYS = 7
FORM_MIN_GAMES = 3  # need at least 3 games in window
FORM_REGRESSION_WEIGHT = 0.5  # blend: 50% recent, 50% season
FORM_SCALE = 0.3  # scale factor: don't overweight hot/cold streaks


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1: REST / SCHEDULE DENSITY
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rest_fatigue(
    games_csv: Path,
    game_date: str,
    team_ids: list[str],
) -> dict[str, dict]:
    """Compute rest/schedule density for each team."""
    games = pd.read_csv(games_csv, dtype=str)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")

    target = pd.Timestamp(game_date)
    window_start = target - timedelta(days=REST_WINDOW_DAYS)

    # Build team game history
    home = games[["game_date", "home_canonical_id"]].rename(columns={"home_canonical_id": "cid"})
    away = games[["game_date", "away_canonical_id"]].rename(columns={"away_canonical_id": "cid"})
    all_games = pd.concat([home, away]).dropna(subset=["game_date"])

    results = {}
    for cid in team_ids:
        team_recent = all_games[
            (all_games["cid"] == cid) &
            (all_games["game_date"] >= window_start) &
            (all_games["game_date"] < target)
        ]
        n_games = len(team_recent)
        if n_games == 0:
            days_since_last = 99
        else:
            last_game = team_recent["game_date"].max()
            days_since_last = (target - last_game).days

        # Fatigue: penalty for dense schedule
        extra_games = max(0, n_games - REST_GAMES_THRESHOLD)
        fatigue_adj = extra_games * REST_PENALTY_PER_EXTRA_GAME

        # Fresh bonus: 2+ days rest
        rest_bonus = REST_DAYS_OFF_BONUS if days_since_last >= REST_DAYS_OFF_THRESHOLD else 0.0

        total_adj = round(fatigue_adj + rest_bonus, 4)
        results[cid] = {
            "games_in_window": n_games,
            "days_since_last": days_since_last,
            "rest_adj": total_adj,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2: DAY/NIGHT
# ═══════════════════════════════════════════════════════════════════════════════

def compute_day_night_adj(
    commence_time_utc: str | None,
    timezone_str: str | None,
) -> dict:
    """Compute day/night adjustment from game commence time."""
    if not commence_time_utc or not isinstance(commence_time_utc, str) or len(commence_time_utc) < 13:
        return {"day_night": "unknown", "day_night_adj": 0.0}

    try:
        utc_hour = int(commence_time_utc[11:13])
        # Convert to approximate local time
        # Use timezone offset if available, otherwise estimate from UTC hour
        tz_offset = -5  # default Eastern
        if timezone_str:
            tz_map = {
                "US/Eastern": -4, "America/New_York": -4,
                "US/Central": -5, "America/Chicago": -5,
                "US/Mountain": -6, "America/Denver": -6,
                "US/Pacific": -7, "America/Los_Angeles": -7,
                "US/Arizona": -7, "America/Phoenix": -7,
            }
            tz_offset = tz_map.get(timezone_str, -5)

        local_hour = (utc_hour + tz_offset) % 24

        if local_hour < DAY_CUTOFF_HOUR_LOCAL:
            return {"day_night": "day", "day_night_adj": DAY_GAME_ADJ}
        else:
            return {"day_night": "night", "day_night_adj": NIGHT_GAME_ADJ}
    except (ValueError, IndexError):
        return {"day_night": "unknown", "day_night_adj": 0.0}


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3: SURFACE TYPE
# ═══════════════════════════════════════════════════════════════════════════════

def load_surface_registry(path: Path) -> dict[str, str]:
    """Load canonical_id → surface type (turf/grass) mapping."""
    if not path.exists():
        return {}
    registry = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            cid = row.get("canonical_id", "").strip()
            surface = row.get("surface", "grass").strip().lower()
            if cid:
                registry[cid] = surface
    return registry


def compute_surface_adj(surface: str) -> float:
    """Return surface adjustment on log-rate scale."""
    if surface in ("turf", "artificial", "fieldturf", "astroturf"):
        return TURF_ADJ
    return GRASS_ADJ


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 4: CONFERENCE STRENGTH (NON-CONF)
# ═══════════════════════════════════════════════════════════════════════════════

# Conference power rankings (approximate, based on RPI/NET strength)
CONF_POWER_RANK = {
    "SEC": 1, "ACC": 2, "Big 12": 3, "Big Ten": 4,
    "Sun Belt": 5, "AAC": 6, "Conference USA": 7, "West Coast": 8,
    "Missouri Valley": 9, "CAA": 10, "Atlantic 10": 11,
    "Big East": 12, "Mountain West": 13, "Big West": 14,
    "Southern": 15, "MAC": 16, "Pac-12": 17, "WAC": 18,
    "ASUN": 19, "Horizon": 20, "Patriot": 21, "America East": 22,
    "NEC": 23, "Southland": 24, "Summit": 25, "Ohio Valley": 26,
    "MEAC": 27, "SWAC": 28, "Ivy": 29, "Big South": 30,
}


def compute_conf_strength_adj(
    home_conf: str,
    away_conf: str,
    is_conference_game: bool,
) -> dict[str, float]:
    """
    Compute conference strength differential adjustment.
    Only applies to non-conference games. The weaker team gets a penalty
    (their offense is slightly inflated by playing weaker opponents all season).
    """
    if is_conference_game:
        return {"home_conf_adj": 0.0, "away_conf_adj": 0.0, "conf_diff": 0}

    h_rank = CONF_POWER_RANK.get(home_conf, 15)
    a_rank = CONF_POWER_RANK.get(away_conf, 15)
    diff = abs(h_rank - a_rank)

    if diff < 3:
        # Similar conference strength — no adjustment
        return {"home_conf_adj": 0.0, "away_conf_adj": 0.0, "conf_diff": diff}

    # Weaker conference team gets negative offensive adjustment
    # (their stats are inflated from playing weaker opponents)
    penalty = min(diff * CONF_STRENGTH_SCALE, 0.03)  # cap at 3%

    if h_rank > a_rank:
        # Home team is from weaker conference
        return {"home_conf_adj": -penalty, "away_conf_adj": 0.0, "conf_diff": diff}
    else:
        # Away team is from weaker conference
        return {"home_conf_adj": 0.0, "away_conf_adj": -penalty, "conf_diff": diff}


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 5: TRAVEL DISTANCE
# ═══════════════════════════════════════════════════════════════════════════════

def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in miles."""
    R = 3959  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def compute_travel_adj(
    away_lat: float | None,
    away_lon: float | None,
    venue_lat: float | None,
    venue_lon: float | None,
) -> dict:
    """Compute travel distance fatigue for away team."""
    if any(v is None for v in [away_lat, away_lon, venue_lat, venue_lon]):
        return {"travel_miles": None, "travel_adj": 0.0}

    distance = _haversine_miles(away_lat, away_lon, venue_lat, venue_lon)

    if distance < TRAVEL_PENALTY_THRESHOLD_MILES:
        return {"travel_miles": round(distance), "travel_adj": 0.0}

    penalty = ((distance - TRAVEL_PENALTY_THRESHOLD_MILES) / 500) * TRAVEL_PENALTY_PER_500MI
    penalty = max(penalty, TRAVEL_MAX_PENALTY)  # cap

    return {"travel_miles": round(distance), "travel_adj": round(penalty, 4)}


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 6: RECENT FORM / MOMENTUM
# ═══════════════════════════════════════════════════════════════════════════════

def compute_recent_form(
    games_csv: Path,
    game_date: str,
    team_ids: list[str],
) -> dict[str, dict]:
    """Compute 7-day scoring form relative to season average."""
    games = pd.read_csv(games_csv, dtype=str)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
    games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")
    games = games[games["home_score"].notna()].copy()

    target = pd.Timestamp(game_date)
    season_start = target - timedelta(days=120)
    form_start = target - timedelta(days=FORM_WINDOW_DAYS)

    # Build per-team scoring history
    home_scoring = games[["game_date", "home_canonical_id", "home_score"]].rename(
        columns={"home_canonical_id": "cid", "home_score": "runs"}
    )
    away_scoring = games[["game_date", "away_canonical_id", "away_score"]].rename(
        columns={"away_canonical_id": "cid", "away_score": "runs"}
    )
    all_scoring = pd.concat([home_scoring, away_scoring])

    results = {}
    for cid in team_ids:
        team = all_scoring[
            (all_scoring["cid"] == cid) &
            (all_scoring["game_date"] >= season_start) &
            (all_scoring["game_date"] < target)
        ]
        if team.empty:
            results[cid] = {"form_adj": 0.0, "recent_rpg": None, "season_rpg": None}
            continue

        season_rpg = team["runs"].mean()
        recent = team[team["game_date"] >= form_start]

        if len(recent) < FORM_MIN_GAMES:
            results[cid] = {"form_adj": 0.0, "recent_rpg": None, "season_rpg": round(season_rpg, 2)}
            continue

        recent_rpg = recent["runs"].mean()

        # Regressed form: blend recent with season
        blended = FORM_REGRESSION_WEIGHT * recent_rpg + (1 - FORM_REGRESSION_WEIGHT) * season_rpg

        # Convert to log-rate adjustment relative to season average
        if season_rpg > 0:
            form_adj = FORM_SCALE * np.log(blended / season_rpg)
        else:
            form_adj = 0.0

        # Clip to prevent extreme swings
        form_adj = float(np.clip(form_adj, -0.04, 0.04))

        results[cid] = {
            "form_adj": round(form_adj, 4),
            "recent_rpg": round(recent_rpg, 2),
            "season_rpg": round(season_rpg, 2),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN: Compute all context for a day's games
# ═══════════════════════════════════════════════════════════════════════════════

def compute_game_context(
    date: str,
    schedule_csv: Path,
    games_csv: Path = Path("data/processed/games.csv"),
    stadium_csv: Path = Path("data/registries/stadium_orientations.csv"),
    surface_csv: Path = Path("data/registries/surface_types.csv"),
    team_table_csv: Path = Path("data/processed/team_table.csv"),
    odds_log: Path = Path("data/raw/odds/odds_pull_log.jsonl"),
    out_csv: Path | None = None,
) -> pd.DataFrame:
    """Compute all contextual adjustments for each game on a date."""

    schedule = pd.read_csv(schedule_csv, dtype=str)
    print(f"Computing game context for {len(schedule)} games on {date}...", file=sys.stderr)

    # Collect all team IDs
    h_cid_col = "home_cid" if "home_cid" in schedule.columns else "home_canonical_id"
    a_cid_col = "away_cid" if "away_cid" in schedule.columns else "away_canonical_id"
    all_team_ids = list(set(
        schedule[h_cid_col].tolist() + schedule[a_cid_col].tolist()
    ))

    # ── Load reference data ──────────────────────────────────────────────
    # Stadium lat/lon for travel distance
    stadium_locs: dict[str, tuple[float, float]] = {}
    tz_by_cid: dict[str, str] = {}
    if stadium_csv.exists():
        stads = pd.read_csv(stadium_csv, dtype=str)
        for _, r in stads.iterrows():
            cid = str(r.get("canonical_id", "")).strip()
            try:
                lat = float(r["lat"])
                lon = float(r["lon"])
                stadium_locs[cid] = (lat, lon)
            except (ValueError, TypeError, KeyError):
                pass
            tz = str(r.get("timezone", "")).strip()
            if tz and tz != "nan":
                tz_by_cid[cid] = tz

    # Surface types
    surface_map = load_surface_registry(surface_csv)

    # Conference for each team
    conf_by_cid: dict[str, str] = {}
    if team_table_csv.exists():
        tt = pd.read_csv(team_table_csv, dtype=str)
        for _, r in tt.iterrows():
            cid = str(r.get("canonical_id", "")).strip()
            conf = str(r.get("conference", "")).strip()
            if cid and conf and conf != "nan":
                conf_by_cid[cid] = conf

    # Game times from odds data
    game_times: dict[str, str] = {}  # "home_away" → commence_time
    if odds_log.exists():
        with open(odds_log) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    ct = row.get("commence_time", "")
                    if not ct.startswith(date[:4]):
                        continue
                    home = row.get("home_team", "")
                    away = row.get("away_team", "")
                    if home and ct:
                        game_times[f"{home}_{away}"] = ct
                except:
                    pass

    # ── Compute batch layers ─────────────────────────────────────────────
    rest_data = compute_rest_fatigue(games_csv, date, all_team_ids)
    form_data = compute_recent_form(games_csv, date, all_team_ids)

    # ── Per-game context ─────────────────────────────────────────────────
    rows = []
    for _, game in schedule.iterrows():
        game_num = str(game.get("game_num", ""))
        h_cid = str(game[h_cid_col]).strip()
        a_cid = str(game[a_cid_col]).strip()
        home_name = str(game.get("home", game.get("home_name", ""))).strip()
        away_name = str(game.get("away", game.get("away_name", ""))).strip()

        rec = {"game_num": game_num, "home_cid": h_cid, "away_cid": a_cid}

        # Layer 1: Rest/schedule density
        h_rest = rest_data.get(h_cid, {"rest_adj": 0.0, "games_in_window": 0, "days_since_last": 99})
        a_rest = rest_data.get(a_cid, {"rest_adj": 0.0, "games_in_window": 0, "days_since_last": 99})
        rec["home_rest_adj"] = h_rest["rest_adj"]
        rec["away_rest_adj"] = a_rest["rest_adj"]
        rec["home_games_in_4d"] = h_rest["games_in_window"]
        rec["away_games_in_4d"] = a_rest["games_in_window"]
        rec["home_days_rest"] = h_rest["days_since_last"]
        rec["away_days_rest"] = a_rest["days_since_last"]

        # Layer 2: Day/night
        # Try to find game time from odds data
        commence = game_times.get(f"{home_name}_{away_name}", "")
        if not commence:
            # Try reverse lookup with odds_api_name
            commence = str(game.get("commence_time", ""))
        tz = tz_by_cid.get(h_cid, None)
        dn = compute_day_night_adj(commence, tz)
        rec["day_night"] = dn["day_night"]
        rec["day_night_adj"] = dn["day_night_adj"]

        # Layer 3: Surface type
        surface = surface_map.get(h_cid, "grass")
        rec["surface"] = surface
        rec["surface_adj"] = compute_surface_adj(surface)

        # Layer 4: Conference strength
        h_conf = conf_by_cid.get(h_cid, "")
        a_conf = conf_by_cid.get(a_cid, "")
        is_conf_game = h_conf == a_conf and h_conf != ""
        conf = compute_conf_strength_adj(h_conf, a_conf, is_conf_game)
        rec["home_conf"] = h_conf
        rec["away_conf"] = a_conf
        rec["is_conference_game"] = int(is_conf_game)
        rec["home_conf_adj"] = conf["home_conf_adj"]
        rec["away_conf_adj"] = conf["away_conf_adj"]
        rec["conf_rank_diff"] = conf["conf_diff"]

        # Layer 5: Travel distance
        h_loc = stadium_locs.get(h_cid)
        a_loc = stadium_locs.get(a_cid)
        travel = compute_travel_adj(
            a_loc[0] if a_loc else None, a_loc[1] if a_loc else None,
            h_loc[0] if h_loc else None, h_loc[1] if h_loc else None,
        )
        rec["travel_miles"] = travel["travel_miles"]
        rec["away_travel_adj"] = travel["travel_adj"]

        # Layer 6: Recent form
        h_form = form_data.get(h_cid, {"form_adj": 0.0})
        a_form = form_data.get(a_cid, {"form_adj": 0.0})
        rec["home_form_adj"] = h_form["form_adj"]
        rec["away_form_adj"] = a_form["form_adj"]
        rec["home_recent_rpg"] = h_form.get("recent_rpg")
        rec["away_recent_rpg"] = a_form.get("recent_rpg")

        # ── Combined context adjustment ──────────────────────────────────
        # Home team: rest + day/night + surface + conf + form (no travel — they're home)
        rec["home_context_adj"] = round(
            h_rest["rest_adj"] + dn["day_night_adj"] + compute_surface_adj(surface)
            + conf["home_conf_adj"] + h_form["form_adj"],
            4,
        )
        # Away team: rest + day/night + surface + conf + travel + form
        rec["away_context_adj"] = round(
            a_rest["rest_adj"] + dn["day_night_adj"] + compute_surface_adj(surface)
            + conf["away_conf_adj"] + travel["travel_adj"] + a_form["form_adj"],
            4,
        )

        rows.append(rec)

    result = pd.DataFrame(rows)

    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_csv, index=False)
        print(f"  Wrote {len(result)} rows → {out_csv}", file=sys.stderr)

    # Summary
    n_rest = sum(1 for r in rows if r["home_rest_adj"] != 0 or r["away_rest_adj"] != 0)
    n_day = sum(1 for r in rows if r["day_night"] == "day")
    n_night = sum(1 for r in rows if r["day_night"] == "night")
    n_turf = sum(1 for r in rows if r["surface"] == "turf")
    n_nonconf = sum(1 for r in rows if not r["is_conference_game"])
    n_travel = sum(1 for r in rows if r["away_travel_adj"] != 0)
    n_form = sum(1 for r in rows if r["home_form_adj"] != 0 or r["away_form_adj"] != 0)
    print(f"  Context: rest={n_rest}, day={n_day}/night={n_night}, turf={n_turf}, "
          f"nonconf={n_nonconf}, travel={n_travel}, form={n_form}",
          file=sys.stderr)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute game context adjustments.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or Path(f"data/daily/{args.date}/context.csv")
    compute_game_context(date=args.date, schedule_csv=args.schedule, out_csv=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
