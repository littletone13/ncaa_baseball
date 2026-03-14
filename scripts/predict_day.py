"""
Fast daily predictions: fetch today's games, resolve starters, fetch weather, simulate.

Includes:
  ✓ Park factors (static, from model fit)
  ✓ Bullpen quality (composite z-score)
  ✓ Starting pitchers (rotation inference from pitcher_appearances.csv)
  ✓ Game-time weather (wind + temperature via Open-Meteo)

Usage:
  python3 scripts/predict_day.py --date 2026-03-09
  python3 scripts/predict_day.py --date 2026-03-09 --N 5000 --json
  python3 scripts/predict_day.py --date 2026-03-09 --no-weather  # skip weather API
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from ncaa_baseball.phase1 import (
    build_odds_name_to_canonical,
    load_canonical_teams,
    resolve_odds_teams,
)
from simulate_run_event_game import prob_to_american
from lookup_starters import StarterLookup
from weather_park_adjustment import get_weather_park_adj, fetch_hourly_weather, load_stadium_data, get_stadium_info
from fb_sensitivity import FBSensitivityLookup
from platoon_adjustment import PlatoonLookup


def resolve_team(name: str, team_idx_map: dict, name_to_cid: dict,
                 canonical: pd.DataFrame, name_to_canonical: dict) -> tuple[str, int]:
    """Resolve ESPN/NCAA team name to (canonical_id, team_idx)."""
    name = name.strip()
    if not name:
        return "", 0
    # Direct canonical_id
    if name in team_idx_map:
        return name, team_idx_map[name]
    # By name (lower)
    cid = name_to_cid.get(name.lower())
    if cid and cid in team_idx_map:
        return cid, team_idx_map[cid]
    # Fuzzy match via resolve_odds_teams
    h_t, _ = resolve_odds_teams(name, "", canonical, name_to_canonical)
    if h_t:
        cid = h_t[0]
        if cid in team_idx_map:
            return cid, team_idx_map[cid]
    # Partial match
    nl = name.lower()
    for cid, idx in team_idx_map.items():
        if nl in cid.lower():
            return cid, idx
    return "", 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast daily game predictions.")
    parser.add_argument("--date", type=str, required=True, help="Date (YYYY-MM-DD)")
    parser.add_argument("--posterior", type=Path, default=Path("data/processed/run_event_posterior_2k.csv"))
    parser.add_argument("--meta", type=Path, default=Path("data/processed/run_event_fit_meta.json"))
    parser.add_argument("--team-index", type=Path, default=Path("data/processed/run_event_team_index.csv"))
    parser.add_argument("--pitcher-index", type=Path, default=Path("data/processed/run_event_pitcher_index.csv"))
    parser.add_argument("--canonical", type=Path, default=Path("data/registries/canonical_teams_2026.csv"))
    parser.add_argument("--park-factors", type=Path, default=Path("data/processed/park_factors.csv"))
    parser.add_argument("--bullpen-quality", type=Path, default=Path("data/processed/bullpen_quality.csv"))
    parser.add_argument("--appearances", type=Path, default=Path("data/processed/pitcher_appearances.csv"))
    parser.add_argument("--pitcher-registry", type=Path, default=Path("data/processed/pitcher_registry.csv"))
    parser.add_argument("--stadium-csv", type=Path, default=Path("data/registries/stadium_orientations.csv"))
    parser.add_argument("--N", type=int, default=5000, help="Simulations per game")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--out", type=Path, default=None, help="Save CSV of results")
    parser.add_argument("--no-weather", action="store_true", help="Skip weather API calls")
    args = parser.parse_args()

    for p in (args.posterior, args.meta, args.team_index, args.pitcher_index):
        if not p.exists():
            print(f"Missing: {p}")
            return 1

    # ── Load model meta + indices ──────────────────────────────────────────
    with open(args.meta) as f:
        meta = json.load(f)
    N_teams = meta["N_teams"]
    N_pitchers = meta["N_pitchers"]

    team_df = pd.read_csv(args.team_index, dtype=str)
    team_idx_map = {str(r["canonical_id"]).strip(): int(r["team_idx"])
                    for _, r in team_df.iterrows() if str(r.get("canonical_id", "")).strip()}

    canonical = load_canonical_teams(args.canonical)
    name_to_canonical = build_odds_name_to_canonical(canonical)
    name_to_cid: dict[str, str] = {}
    for _, row in canonical.iterrows():
        tname = str(row.get("team_name", "")).strip()
        cid = str(row.get("canonical_id", "")).strip()
        if tname and cid:
            name_to_cid[tname.lower()] = cid
            for word in tname.split():
                w = word.lower().strip()
                if len(w) > 3 and w not in name_to_cid:
                    name_to_cid[w] = cid

    # Park factors
    pf_map: dict[str, float] = {}
    if args.park_factors.exists():
        for _, r in pd.read_csv(args.park_factors).iterrows():
            htid = str(r.get("home_team_id", "")).strip()
            adj = r.get("adjusted_pf")
            if htid and adj is not None and not (isinstance(adj, float) and math.isnan(adj)):
                pf_map[htid] = math.log(float(adj))

    # Bullpen quality
    bp_map: dict[tuple[str, int], float] = {}
    if args.bullpen_quality.exists():
        for _, r in pd.read_csv(args.bullpen_quality).iterrows():
            cid = str(r.get("team_canonical_id", "")).strip()
            season = int(r.get("season", 0))
            score = r.get("bullpen_depth_score")
            if cid and season and score is not None and not (isinstance(score, float) and math.isnan(score)):
                bp_map[(cid, season)] = -float(score) * 0.1

    # ── Starting pitcher lookup ────────────────────────────────────────────
    print("Loading starter lookup...", file=sys.stderr)
    starter_lookup = StarterLookup(
        appearances_csv=args.appearances,
        registry_csv=args.pitcher_registry,
        pitcher_index_csv=args.pitcher_index,
    )

    # ── Fly-ball sensitivity lookup ──────────────────────────────────────
    fb_lookup = FBSensitivityLookup()
    print(f"  {fb_lookup.summary()}", file=sys.stderr)

    # ── Platoon (LHP/RHP) lookup ──────────────────────────────────────
    platoon_lookup = PlatoonLookup()
    print(f"  {platoon_lookup.summary()}", file=sys.stderr)

    # ── ERA-based pitcher ability fallback (D1Baseball) ──────────────────
    # When a pitcher has no run_event posterior data (idx=0), use their
    # D1Baseball ERA to estimate ability instead of league-average prior.
    # Calibration: ability ≈ 0.027 * log(ERA / 3.4) - 0.01, R²=0.07
    # (weak but directionally correct — better than assuming everyone is average)
    ERA_SLOPE = 0.027
    ERA_INTERCEPT = -0.01
    ERA_BASELINE = 3.4  # median ERA in D1B rotation data
    era_ability_map: dict[tuple[str, str], float] = {}  # (canonical_id, pitcher_name) → ability
    d1b_era_path = Path("data/processed/d1baseball_rotations.csv")
    if d1b_era_path.exists():
        d1b_era_df = pd.read_csv(d1b_era_path, dtype=str)
        for _, r in d1b_era_df.iterrows():
            cid = str(r.get("canonical_id", "")).strip()
            pname = str(r.get("pitcher_name", "")).strip()
            era_str = str(r.get("era", "")).strip()
            if cid and pname and era_str:
                try:
                    era_val = float(era_str)
                    if era_val > 0:
                        ability_est = ERA_SLOPE * math.log(era_val / ERA_BASELINE) + ERA_INTERCEPT
                        era_ability_map[(cid, pname.lower())] = ability_est
                except (ValueError, ZeroDivisionError):
                    pass
        print(f"  ERA-based ability fallback: {len(era_ability_map)} pitchers from D1Baseball",
              file=sys.stderr)

    # ── Pre-extract posterior into numpy arrays ────────────────────────────
    print("Loading posterior...", file=sys.stderr)
    draws_df = pd.read_csv(args.posterior)
    n_draws = len(draws_df)

    int_run = np.zeros((n_draws, 4))
    theta_run = np.zeros((n_draws, 2))
    home_adv = np.zeros(n_draws)
    beta_park = np.ones(n_draws)
    beta_bullpen = np.zeros(n_draws)

    for k in range(4):
        int_run[:, k] = draws_df[f"int_run_{k+1}"].values
    for k in range(2):
        theta_run[:, k] = draws_df[f"theta_run_{k+1}"].values
    home_adv[:] = draws_df["home_advantage"].values
    if "beta_park" in draws_df.columns:
        beta_park[:] = draws_df["beta_park"].values
    if "beta_bullpen" in draws_df.columns:
        beta_bullpen[:] = draws_df["beta_bullpen"].values

    att = np.zeros((n_draws, N_teams + 1, 4))
    def_ = np.zeros((n_draws, N_teams + 1, 4))
    for k in range(4):
        for t in range(1, N_teams + 1):
            col_a = f"att_run_{k+1}[{t}]"
            col_d = f"def_run_{k+1}[{t}]"
            if col_a in draws_df.columns:
                att[:, t, k] = draws_df[col_a].values
            if col_d in draws_df.columns:
                def_[:, t, k] = draws_df[col_d].values

    pitcher_ab = np.zeros((n_draws, N_pitchers + 1))
    for p in range(1, N_pitchers + 1):
        col = f"pitcher_ability[{p}]"
        if col in draws_df.columns:
            pitcher_ab[:, p] = draws_df[col].values

    del draws_df
    print(f"  {n_draws} draws, {N_teams} teams, {N_pitchers} pitchers", file=sys.stderr)

    # ── Fetch game schedule (NCAA API primary, ESPN fallback) ──────────────
    # matchups: list of (home_name, away_name, start_utc_iso_or_None)
    matchups = []
    dt_parts = args.date.split("-")
    ncaa_url = f"https://ncaa-api.henrygd.me/scoreboard/baseball/d1/{dt_parts[0]}/{dt_parts[1]}/{dt_parts[2]}"
    try:
        req = Request(ncaa_url, headers={"User-Agent": "Mozilla/5.0 (Macintosh)"})
        ncaa_data = json.loads(urlopen(req, timeout=15).read())
        for g in ncaa_data.get("games", []):
            game = g.get("game", {})
            home = game.get("home", {})
            away = game.get("away", {})
            h_names = home.get("names", {})
            a_names = away.get("names", {})
            h_name = h_names.get("full", "").strip() or h_names.get("short", "").strip() or "?"
            a_name = a_names.get("full", "").strip() or a_names.get("short", "").strip() or "?"
            start_time = game.get("startTimeEpoch")  # NCAA API may have epoch
            if h_name != "?" and a_name != "?":
                matchups.append((h_name, a_name, None))  # NCAA API doesn't reliably give times
        print(f"\n{len(matchups)} games on {args.date} (NCAA API)\n", file=sys.stderr)
    except Exception as ex:
        print(f"NCAA API failed ({ex}), falling back to ESPN...", file=sys.stderr)

    # Always also fetch ESPN to get start times
    # Two lookups: by raw name pair AND by canonical_id pair (for fuzzy merge)
    espn_times: dict[tuple[str, str], str] = {}      # (home_name, away_name) → UTC ISO
    espn_times_cid: dict[tuple[str, str], str] = {}   # (home_cid, away_cid) → UTC ISO
    try:
        dt = args.date.replace("-", "")
        espn_url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/scoreboard?dates={dt}&limit=200"
        req = Request(espn_url, headers={"User-Agent": "Mozilla/5.0 (Macintosh)"})
        data = json.loads(urlopen(req, timeout=15).read())
        for e in data.get("events", []):
            start_utc = e.get("date")  # ISO 8601 UTC
            comps = e.get("competitions", [{}])
            if not comps:
                continue
            competitors = comps[0].get("competitors", [])
            if len(competitors) != 2:
                continue
            home_c = next((x for x in competitors if x.get("homeAway") == "home"), None)
            away_c = next((x for x in competitors if x.get("homeAway") == "away"), None)
            if home_c and away_c:
                h_espn = home_c.get("team", {}).get("displayName", "?")
                a_espn = away_c.get("team", {}).get("displayName", "?")
                if start_utc:
                    espn_times[(h_espn, a_espn)] = start_utc
                    # Also resolve ESPN names to canonical_ids for fuzzy merge
                    h_cid_e, _ = resolve_team(h_espn, team_idx_map, name_to_cid, canonical, name_to_canonical)
                    a_cid_e, _ = resolve_team(a_espn, team_idx_map, name_to_cid, canonical, name_to_canonical)
                    if h_cid_e and a_cid_e:
                        espn_times_cid[(h_cid_e, a_cid_e)] = start_utc
                if not matchups:  # ESPN fallback for schedule
                    matchups.append((h_espn, a_espn, start_utc))
        if matchups and matchups[0][2] is None:
            print(f"  (ESPN provided {len(espn_times)} start times, {len(espn_times_cid)} resolved to canonical)", file=sys.stderr)
    except Exception as ex2:
        print(f"  ESPN time fetch failed: {ex2}", file=sys.stderr)

    if not matchups and espn_times:
        for (h, a), t in espn_times.items():
            matchups.append((h, a, t))
        print(f"\n{len(matchups)} games on {args.date} (ESPN fallback)\n", file=sys.stderr)

    # ── Load odds commence times as additional start time source ───────────
    odds_times_cid: dict[tuple[str, str], str] = {}
    try:
        odds_latest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "data", "raw", "odds", "odds_latest.jsonl")
        if os.path.exists(odds_latest):
            # Build odds_api_name → canonical_id lookup
            odds_name_to_cid: dict[str, str] = {}
            for _, row in canonical.iterrows():
                oname = str(row.get("odds_api_name", "")).strip()
                if oname:
                    odds_name_to_cid[oname] = row["canonical_id"]
            with open(odds_latest) as fod:
                for line in fod:
                    og = json.loads(line)
                    ct = og.get("commence_time")
                    if not ct:
                        continue
                    hc = odds_name_to_cid.get(og.get("home_team", ""))
                    ac = odds_name_to_cid.get(og.get("away_team", ""))
                    if hc and ac:
                        odds_times_cid[(hc, ac)] = ct
            if odds_times_cid:
                print(f"  (Odds API provided {len(odds_times_cid)} commence times)", file=sys.stderr)
    except Exception as ex3:
        print(f"  Odds commence time load failed: {ex3}", file=sys.stderr)

    # ── Resolve teams, starters, weather for each game ─────────────────────
    rng = np.random.default_rng(args.seed)
    all_results = []

    for game_num, (h_name, a_name, start_utc) in enumerate(matchups):

        h_cid, h_idx = resolve_team(h_name, team_idx_map, name_to_cid, canonical, name_to_canonical)
        a_cid, a_idx = resolve_team(a_name, team_idx_map, name_to_cid, canonical, name_to_canonical)

        # ── Resolve game start time ──────────────────────────────────────
        # Try to match ESPN time: first by raw names, then by canonical_id
        if start_utc is None:
            start_utc = espn_times.get((h_name, a_name))
        if start_utc is None and h_cid and a_cid:
            start_utc = espn_times_cid.get((h_cid, a_cid))
        # Odds API commence time as fallback before 6pm default
        if start_utc is None and h_cid and a_cid:
            start_utc = odds_times_cid.get((h_cid, a_cid))

        # Parse start time → local hour for hourly weather
        game_start_hour = None
        sdf_temp_local = None
        if 'sdf_temp' not in dir():
            sdf_temp = load_stadium_data(args.stadium_csv)
        sinfo = get_stadium_info(h_cid, sdf_temp) if h_cid else None
        if start_utc:
            try:
                from datetime import datetime, timezone, timedelta
                utc_dt = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
                # Rough local time: use longitude to estimate timezone offset
                # (Every 15° of longitude ≈ 1 hour offset from UTC)
                if sinfo:
                    lon_offset_hrs = round(sinfo["lon"] / 15.0)
                    local_dt = utc_dt + timedelta(hours=lon_offset_hrs)
                    game_start_hour = local_dt.hour
            except Exception:
                pass

        # Default start hour: 18 (6pm local) for games without ESPN times
        # Most college baseball games start between 5-7pm; 6pm is a safe default
        # for forecast-based weather rather than current conditions
        if game_start_hour is None and sinfo:
            game_start_hour = 18  # 6pm local default

        # ── Starting pitchers ──────────────────────────────────────────────
        hp_name, hp_id, hp_idx = starter_lookup.get_starter(h_cid, args.date)
        ap_name, ap_id, ap_idx = starter_lookup.get_starter(a_cid, args.date)

        # Clamp pitcher indices to posterior size (new pitchers not in model → league avg)
        if hp_idx >= N_pitchers + 1:
            print(f"    WARNING: home pitcher idx {hp_idx} exceeds posterior ({N_pitchers}), using league avg", file=sys.stderr)
            hp_idx = 0
        if ap_idx >= N_pitchers + 1:
            print(f"    WARNING: away pitcher idx {ap_idx} exceeds posterior ({N_pitchers}), using league avg", file=sys.stderr)
            ap_idx = 0

        # ── ERA-based ability override for pitchers without posterior data ─
        # When idx=0, pitcher_ab[d, 0] ≈ 0 (league average prior).
        # If D1Baseball ERA is available, use it for a better estimate.
        hp_era_adj = 0.0
        ap_era_adj = 0.0
        if hp_idx == 0 and hp_name != "unknown":
            era_est = era_ability_map.get((h_cid, hp_name.lower()))
            if era_est is not None:
                hp_era_adj = era_est
        if ap_idx == 0 and ap_name != "unknown":
            era_est = era_ability_map.get((a_cid, ap_name.lower()))
            if era_est is not None:
                ap_era_adj = era_est

        # ── Pitcher handedness (platoon effect) ────────────────────────────
        hp_hand = platoon_lookup.get_hand(h_cid, hp_name)
        ap_hand = platoon_lookup.get_hand(a_cid, ap_name)
        # Home team faces away pitcher → platoon adj based on away P hand
        platoon_h = platoon_lookup.platoon_adj(ap_hand)
        # Away team faces home pitcher → platoon adj based on home P hand
        platoon_a = platoon_lookup.platoon_adj(hp_hand)

        hp_hand_tag = f" ({hp_hand}HP)" if hp_hand else ""
        ap_hand_tag = f" ({ap_hand}HP)" if ap_hand else ""
        hp_era_tag = f", ERA→{hp_era_adj:+.3f}" if hp_era_adj != 0 else ""
        ap_era_tag = f", ERA→{ap_era_adj:+.3f}" if ap_era_adj != 0 else ""
        hp_tag = f"{hp_name}{hp_hand_tag} (idx={hp_idx}{hp_era_tag})" if hp_idx > 0 else f"{hp_name}{hp_hand_tag} (ERA adj={hp_era_adj:+.3f})" if hp_era_adj != 0 else f"{hp_name}{hp_hand_tag} (no model data)"
        ap_tag = f"{ap_name}{ap_hand_tag} (idx={ap_idx}{ap_era_tag})" if ap_idx > 0 else f"{ap_name}{ap_hand_tag} (ERA adj={ap_era_adj:+.3f})" if ap_era_adj != 0 else f"{ap_name}{ap_hand_tag} (no model data)"
        print(f"  Game {game_num+1}: {a_name} @ {h_name}", file=sys.stderr)
        if start_utc:
            print(f"    Start: {start_utc} (local hr≈{game_start_hour})", file=sys.stderr)
        print(f"    Starters: {ap_tag} vs {hp_tag}", file=sys.stderr)
        if platoon_h != 0.0 or platoon_a != 0.0:
            print(f"    Platoon: home={platoon_h:+.3f} (vs {ap_hand or '?'}HP), "
                  f"away={platoon_a:+.3f} (vs {hp_hand or '?'}HP)", file=sys.stderr)

        # ── Park factor + weather ──────────────────────────────────────────
        # Wind sensitivity is per-pitcher: the pitcher on the mound determines
        # how much wind affects scoring against them.
        #   Home batting faces away pitcher → ap's FB% scales wind for home runs
        #   Away batting faces home pitcher → hp's FB% scales wind for away runs
        # Bullpen wind uses team-average FB% (bullpen = staff average).
        # Temp + altitude (non_wind_adj) are symmetric — affect all batted balls.
        pf = pf_map.get(h_cid, 0.0)
        weather_adj = 0.0
        wind_adj_raw = 0.0
        non_wind_adj = 0.0
        weather_info = {}

        # Per-pitcher FB sensitivity (starters)
        hp_fb_sens = fb_lookup.pitcher_sensitivity(h_cid, hp_name)
        ap_fb_sens = fb_lookup.pitcher_sensitivity(a_cid, ap_name)
        # Bullpen FB sensitivity (team averages)
        hp_bp_fb_sens = fb_lookup.bullpen_sensitivity(h_cid)
        ap_bp_fb_sens = fb_lookup.bullpen_sensitivity(a_cid)

        if not args.no_weather:
            try:
                w = get_weather_park_adj(
                    canonical_id=h_cid,
                    stadium_csv=args.stadium_csv,
                    game_date=args.date,
                    game_start_hour=game_start_hour,
                )
                if "error" not in w:
                    wind_adj_raw = w["wind_adj_raw"]
                    non_wind_adj = w["non_wind_adj"]
                    # Display-level: average of both starters for summary
                    weather_adj = wind_adj_raw * (hp_fb_sens + ap_fb_sens) / 2.0 + non_wind_adj
                    weather_info = w
                    mode = w.get("weather_mode", "current")
                    wind_detail = ""
                    if mode == "hourly_avg" and "hourly_wind" in w:
                        hrs = w["hourly_wind"]
                        parts = [f"hr{h['hour_offset']}:{h['wind_out_mph']:+.0f}" for h in hrs]
                        wind_detail = f" [{', '.join(parts)}]"
                    fb_tag = f" [hp_fb={hp_fb_sens:.2f}, ap_fb={ap_fb_sens:.2f}]"
                    print(f"    Weather ({mode}): {w['temp_f']:.0f}°F, wind {w['wind_speed_mph']:.0f}mph "
                          f"(out: {w['wind_out_mph']:+.1f}), wind_raw={wind_adj_raw:+.4f}, "
                          f"non_wind={non_wind_adj:+.4f}{fb_tag}{wind_detail}",
                          file=sys.stderr)
                else:
                    print(f"    Weather: {w['error']}", file=sys.stderr)
            except Exception as e:
                print(f"    Weather: failed ({e})", file=sys.stderr)

        # Per-team park+weather effects:
        # base_pf = static park factor + temp/altitude (symmetric)
        # wind is per-pitcher: scaled by the FB% of the pitcher on the mound
        #
        # Starters pitch ~5.5 of 9 innings (~61%), bullpen covers ~3.5 (~39%).
        # Blend starter and bullpen FB sensitivity proportionally.
        STARTER_IP_FRAC = 0.61  # ~5.5 / 9
        BULLPEN_IP_FRAC = 1.0 - STARTER_IP_FRAC

        base_pf = pf + non_wind_adj

        # Home scoring: away pitcher on mound
        #   starter innings → away starter FB sens, bullpen innings → away bullpen FB sens
        ap_blended_sens = STARTER_IP_FRAC * ap_fb_sens + BULLPEN_IP_FRAC * ap_bp_fb_sens
        wind_adj_home = wind_adj_raw * ap_blended_sens

        # Away scoring: home pitcher on mound
        hp_blended_sens = STARTER_IP_FRAC * hp_fb_sens + BULLPEN_IP_FRAC * hp_bp_fb_sens
        wind_adj_away = wind_adj_raw * hp_blended_sens

        # Pure bullpen wind (for extra innings)
        wind_adj_home_bp = wind_adj_raw * ap_bp_fb_sens
        wind_adj_away_bp = wind_adj_raw * hp_bp_fb_sens

        h_bp = bp_map.get((h_cid, 2026), 0.0)
        a_bp = bp_map.get((a_cid, 2026), 0.0)

        # ── Simulate ───────────────────────────────────────────────────────
        wins_home = 0
        exp_h_sum, exp_a_sum = 0.0, 0.0
        home_rl_cover = 0
        away_rl_cover = 0
        overs = 0
        total_line = 11.5

        for _ in range(args.N):
            d = rng.integers(0, n_draws)
            base_park_eff = beta_park[d] * base_pf
            # Per-team park effects: base (symmetric) + per-pitcher wind
            park_eff_h = base_park_eff + wind_adj_home   # home scoring (away P on mound)
            park_eff_a = base_park_eff + wind_adj_away   # away scoring (home P on mound)
            bp_h_eff = beta_bullpen[d] * a_bp
            bp_a_eff = beta_bullpen[d] * h_bp

            home_runs_sim, away_runs_sim = 0, 0
            eh, ea = 0.0, 0.0

            for k in range(4):
                log_lam_h = (int_run[d, k] + att[d, h_idx, k] + def_[d, a_idx, k]
                             + home_adv[d] + pitcher_ab[d, ap_idx] + ap_era_adj
                             + park_eff_h + bp_h_eff + platoon_h)
                log_lam_a = (int_run[d, k] + att[d, a_idx, k] + def_[d, h_idx, k]
                             + pitcher_ab[d, hp_idx] + hp_era_adj
                             + park_eff_a + bp_a_eff + platoon_a)
                mu_h = np.exp(log_lam_h)
                mu_a = np.exp(log_lam_a)
                eh += (k + 1) * mu_h
                ea += (k + 1) * mu_a

                if k <= 1:
                    theta = max(1e-6, theta_run[d, k])
                    p_h = theta / (theta + max(1e-8, mu_h))
                    p_a = theta / (theta + max(1e-8, mu_a))
                    home_runs_sim += (k + 1) * rng.negative_binomial(n=theta, p=p_h)
                    away_runs_sim += (k + 1) * rng.negative_binomial(n=theta, p=p_a)
                else:
                    home_runs_sim += (k + 1) * rng.poisson(lam=max(1e-8, mu_h))
                    away_runs_sim += (k + 1) * rng.poisson(lam=max(1e-8, mu_a))

            exp_h_sum += eh
            exp_a_sum += ea

            # Extra innings (bullpen pitching → use bullpen FB sensitivity for wind)
            park_eff_h_bp = base_park_eff + wind_adj_home_bp
            park_eff_a_bp = base_park_eff + wind_adj_away_bp
            extra = 0
            while home_runs_sim == away_runs_sim and extra < 20:
                for k in range(4):
                    log_lam_h = (int_run[d, k] + att[d, h_idx, k] + def_[d, a_idx, k]
                                 + home_adv[d] + pitcher_ab[d, ap_idx] + park_eff_h_bp + bp_h_eff)
                    log_lam_a = (int_run[d, k] + att[d, a_idx, k] + def_[d, h_idx, k]
                                 + pitcher_ab[d, hp_idx] + park_eff_a_bp + bp_a_eff)
                    mu_h = np.exp(log_lam_h) / 9.0
                    mu_a = np.exp(log_lam_a) / 9.0
                    if k <= 1:
                        theta = max(1e-6, theta_run[d, k])
                        p_h = theta / (theta + max(1e-8, mu_h))
                        p_a = theta / (theta + max(1e-8, mu_a))
                        home_runs_sim += (k + 1) * rng.negative_binomial(n=theta, p=p_h)
                        away_runs_sim += (k + 1) * rng.negative_binomial(n=theta, p=p_a)
                    else:
                        home_runs_sim += (k + 1) * rng.poisson(lam=max(1e-8, mu_h))
                        away_runs_sim += (k + 1) * rng.poisson(lam=max(1e-8, mu_a))
                extra += 1
            if home_runs_sim == away_runs_sim:
                if rng.random() < 0.5:
                    home_runs_sim += 1
                else:
                    away_runs_sim += 1

            margin = home_runs_sim - away_runs_sim
            if margin > 0:
                wins_home += 1
            if margin > 1.5:
                home_rl_cover += 1
            if margin < -1.5:
                away_rl_cover += 1
            if (home_runs_sim + away_runs_sim) > total_line:
                overs += 1

        N = args.N
        win_prob = wins_home / N
        exp_h = exp_h_sum / N
        exp_a = exp_a_sum / N

        result = {
            "game": game_num + 1,
            "away": a_name,
            "home": h_name,
            "home_cid": h_cid,
            "away_cid": a_cid,
            "home_starter": hp_name,
            "away_starter": ap_name,
            "home_starter_idx": hp_idx,
            "away_starter_idx": ap_idx,
            "hp_throws": hp_hand or "",
            "ap_throws": ap_hand or "",
            "platoon_adj_home": round(platoon_h, 4),
            "platoon_adj_away": round(platoon_a, 4),
            "home_win_prob": win_prob,
            "away_win_prob": 1 - win_prob,
            "ml_home": prob_to_american(win_prob),
            "ml_away": prob_to_american(1 - win_prob),
            "exp_home": exp_h,
            "exp_away": exp_a,
            "exp_total": exp_h + exp_a,
            "home_rl_cover": home_rl_cover / N,
            "away_rl_cover": away_rl_cover / N,
            "over_prob": overs / N,
            "park_factor": pf,
            "weather_adj": round(weather_adj, 4),
            "wind_adj_raw": round(wind_adj_raw, 4),
            "non_wind_adj": round(non_wind_adj, 4),
            "hp_fb_sens": round(hp_fb_sens, 3),
            "ap_fb_sens": round(ap_fb_sens, 3),
            "hp_bp_fb_sens": round(hp_bp_fb_sens, 3),
            "ap_bp_fb_sens": round(ap_bp_fb_sens, 3),
            "temp_f": weather_info.get("temp_f"),
            "wind_mph": weather_info.get("wind_speed_mph"),
            "wind_out_mph": weather_info.get("wind_out_mph"),
            "weather_mode": weather_info.get("weather_mode"),
            "hp_era_adj": round(hp_era_adj, 4) if hp_era_adj != 0 else None,
            "ap_era_adj": round(ap_era_adj, 4) if ap_era_adj != 0 else None,
            "wind_out_hr0": None,
            "wind_out_hr1": None,
            "wind_out_hr2": None,
        }
        # Add per-hour wind detail if available
        if weather_info.get("hourly_wind"):
            for h in weather_info["hourly_wind"]:
                key = f"wind_out_hr{h['hour_offset']}"
                result[key] = h["wind_out_mph"]
        all_results.append(result)

    # ── Output ─────────────────────────────────────────────────────────────
    if args.json:
        print(json.dumps(all_results, indent=2))
        return 0

    # Sort by confidence (biggest edge first)
    all_results.sort(key=lambda r: abs(r["home_win_prob"] - 0.5), reverse=True)

    print(f"\n{'='*90}")
    print(f"  NCAA BASEBALL PREDICTIONS — {args.date}")
    print(f"  {len(all_results)} games | {n_draws} posterior draws | {args.N} sims/game")
    feat = "starters + bullpen + park + weather" if not args.no_weather else "starters + bullpen + park"
    print(f"  Features: {feat}")
    print(f"{'='*90}\n")

    # ── Matchup cards with starters ────────────────────────────────────────
    for i, r in enumerate(all_results):
        conf = max(r["home_win_prob"], r["away_win_prob"])
        if conf >= 0.65:
            tier = "★★★" if conf >= 0.75 else " ★★"
        elif abs(r["home_win_prob"] - 0.5) >= 0.05:
            tier = "  ◆"
        else:
            tier = "  ○"

        fav_name = r["home"] if r["home_win_prob"] > 0.5 else r["away"]
        fav_prob = max(r["home_win_prob"], r["away_win_prob"])

        print(f"  {tier}  {r['away']:>22s}  @  {r['home']:<22s}   {fav_name} {fav_prob:.0%}")

        # Starters line (with handedness)
        as_lbl = r["away_starter"] if r["away_starter"] != "unknown" else "??"
        hs_lbl = r["home_starter"] if r["home_starter"] != "unknown" else "??"
        as_model = "✓" if r["away_starter_idx"] > 0 else "✗"
        hs_model = "✓" if r["home_starter_idx"] > 0 else "✗"
        as_hand = f" ({r['ap_throws']}HP)" if r.get("ap_throws") else ""
        hs_hand = f" ({r['hp_throws']}HP)" if r.get("hp_throws") else ""
        plat_tag = ""
        if r.get("platoon_adj_home", 0) != 0 or r.get("platoon_adj_away", 0) != 0:
            plat_parts = []
            if r.get("platoon_adj_home", 0) != 0:
                plat_parts.append(f"home={r['platoon_adj_home']:+.3f}")
            if r.get("platoon_adj_away", 0) != 0:
                plat_parts.append(f"away={r['platoon_adj_away']:+.3f}")
            plat_tag = f"  platoon[{', '.join(plat_parts)}]"
        print(f"       SP: {as_lbl}{as_hand} [{as_model}]  vs  {hs_lbl}{hs_hand} [{hs_model}]{plat_tag}")

        # Weather line
        if r.get("temp_f") is not None:
            wind_str = f"wind out {r['wind_out_mph']:+.0f}mph" if r.get("wind_out_mph") else ""
            fb_str = f"  fb[hp={r['hp_fb_sens']:.2f},ap={r['ap_fb_sens']:.2f}]" if r.get("hp_fb_sens") else ""
            print(f"       Wx: {r['temp_f']:.0f}°F  {wind_str}  (adj={r['weather_adj']:+.4f}){fb_str}")
        elif not args.no_weather:
            print(f"       Wx: unavailable")

        # Line
        print(f"       ML: Home {r['ml_home']:+d}  Away {r['ml_away']:+d}  |  "
              f"Total {r['exp_total']:.1f}  O/U {r['over_prob']:.0%}")
        print()

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"{'─'*90}")
    print(f"  SUMMARY TABLE")
    print(f"{'─'*90}")
    print(f"  {'#':>2s}  {'Away':>20s}  {'A.SP':>12s}  {'@':>1s}  {'Home':>20s}  {'H.SP':>12s}  "
          f"{'P(H)':>6s}  {'ML.H':>6s}  {'ML.A':>6s}  {'Tot':>5s}  {'Wx':>5s}")
    print(f"  {'─'*2}  {'─'*20}  {'─'*12}  {'─'*1}  {'─'*20}  {'─'*12}  "
          f"{'─'*6}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*5}")
    for r in sorted(all_results, key=lambda x: x["game"]):
        wp = r["home_win_prob"]
        as_short = r["away_starter"][:12] if r["away_starter"] != "unknown" else "??"
        hs_short = r["home_starter"][:12] if r["home_starter"] != "unknown" else "??"
        temp = f"{r['temp_f']:.0f}°" if r.get("temp_f") else "  - "
        print(f"  {r['game']:2d}  {r['away']:>20s}  {as_short:>12s}  @  {r['home']:>20s}  {hs_short:>12s}  "
              f"{wp:>5.0%}  {r['ml_home']:>+6d}  {r['ml_away']:>+6d}  {r['exp_total']:>5.1f}  {temp:>5s}")

    # ── Starter coverage report ────────────────────────────────────────────
    n_sp_posterior = sum(1 for r in all_results if r["home_starter_idx"] > 0) + \
                     sum(1 for r in all_results if r["away_starter_idx"] > 0)
    n_sp_era = sum(1 for r in all_results if r.get("hp_era_adj")) + \
               sum(1 for r in all_results if r.get("ap_era_adj"))
    n_sp_total = 2 * len(all_results)
    n_wx = sum(1 for r in all_results if r.get("temp_f") is not None)
    print(f"\n  Coverage: starters {n_sp_posterior}/{n_sp_total} posterior | "
          f"{n_sp_era}/{n_sp_total} ERA-fallback | "
          f"weather {n_wx}/{len(all_results)} games")

    if args.out:
        pd.DataFrame(all_results).to_csv(args.out, index=False)
        print(f"\n  Saved to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
