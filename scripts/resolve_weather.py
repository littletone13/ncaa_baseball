"""
Standalone weather + park factor resolver for the daily prediction pipeline.

Reads a schedule CSV (produced by resolve_schedule.py or similar) and fetches
live/forecast weather for each game's home stadium, combining with static park
factors from park_factors.csv.

Output schema (weather.csv):
    game_num        - matches schedule.csv row index
    home_cid        - home team canonical_id
    park_factor     - static log-scale park factor (from park_factors.csv)
    wind_adj_raw    - wind adjustment (log-rate, unscaled by FB%)
    non_wind_adj    - temperature + altitude adjustment (log-rate)
    wind_out_mph    - effective (arc-weighted) wind out component
    wind_out_lf     - wind out toward left field
    wind_out_cf     - wind out toward center field
    wind_out_rf     - wind out toward right field
    temp_f          - temperature in Fahrenheit
    wind_mph        - wind speed
    wind_dir_deg    - wind direction (meteorological, FROM direction)
    rain_chance_pct - precipitation probability (0-100)
    weather_mode    - "hourly_avg" or "current"
    weather_status  - quality flag: ok_hourly/ok_current/missing_stadium/api_error/...
    elevation_ft    - stadium elevation in feet

Usage:
    python3 scripts/resolve_weather.py --schedule data/daily/2026-03-14/schedule.csv --date 2026-03-14
    python3 scripts/resolve_weather.py --schedule data/daily/2026-03-14/schedule.csv --date 2026-03-14 --out data/daily/2026-03-14/weather.csv
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from weather_park_adjustment import (
    get_weather_park_adj, load_stadium_data, get_stadium_info,
    air_density_ratio, ALTITUDE_COEFF,
)


# ──────────────────────────────────────────────────────────────────────────────
# Park factor loading
# ──────────────────────────────────────────────────────────────────────────────

def load_park_factors(park_factors_csv: Path) -> dict[str, float]:
    """Load park_factors.csv → {home_team_id: log(adjusted_pf)}.

    Returns empty dict if file does not exist. Teams not in the file get 0.0
    (neutral) when callers do pf_map.get(cid, 0.0).
    """
    pf_map: dict[str, float] = {}
    if not park_factors_csv.exists():
        print(f"Warning: park factors file not found: {park_factors_csv}", file=sys.stderr)
        return pf_map
    for _, r in pd.read_csv(park_factors_csv).iterrows():
        htid = str(r.get("home_team_id", "")).strip()
        adj = r.get("adjusted_pf")
        if htid and adj is not None and not (isinstance(adj, float) and math.isnan(adj)):
            pf_map[htid] = math.log(float(adj))
    return pf_map


# ──────────────────────────────────────────────────────────────────────────────
# Main resolver
# ──────────────────────────────────────────────────────────────────────────────

def resolve_weather(
    schedule_csv: Path,
    stadium_csv: Path = Path("data/registries/stadium_orientations.csv"),
    park_factors_csv: Path = Path("data/processed/park_factors.csv"),
    date: str = "",
    out_csv: Path | None = None,
) -> pd.DataFrame:
    """Fetch weather and park factors for each game in schedule_csv.

    Args:
        schedule_csv: Path to schedule CSV with at minimum columns:
            game_num, home_cid, start_local_hour
        stadium_csv: Path to stadium orientations CSV.
        park_factors_csv: Path to park factors CSV.
        date: Game date string (YYYY-MM-DD) — used for hourly weather forecasts.
        out_csv: Optional output path; if given, writes CSV there.

    Returns:
        DataFrame with one row per game containing weather and park factor data.
    """
    # Load inputs
    schedule = pd.read_csv(schedule_csv)
    pf_map = load_park_factors(park_factors_csv)
    stadium_df = load_stadium_data(stadium_csv)

    total = len(schedule)
    rows: list[dict] = []

    for _, game_row in schedule.iterrows():
        game_num = int(game_row["game_num"])
        home_cid = str(game_row.get("home_cid", "")).strip()
        start_local_hour = game_row.get("start_local_hour")
        if pd.isna(start_local_hour):
            start_local_hour = None
        else:
            start_local_hour = int(start_local_hour)

        # Look up stadium info for progress logging
        sinfo = get_stadium_info(home_cid, stadium_df) if home_cid else None
        venue_name = sinfo["venue_name"] if sinfo else "unknown"

        print(
            f"Weather: {game_num + 1}/{total} {home_cid} ({venue_name})",
            file=sys.stderr,
        )

        # Static park factor (log-scale, 0.0 = neutral)
        park_factor = pf_map.get(home_cid, 0.0)

        # Default weather values (neutral — used on failure or missing stadium)
        wind_adj_raw = 0.0
        non_wind_adj = 0.0
        wind_out_mph = 0.0
        wind_out_lf = 0.0
        wind_out_cf = 0.0
        wind_out_rf = 0.0
        temp_f = 72.0
        wind_mph = 0.0
        wind_dir_deg = 0.0
        wind_gusts_mph = 0.0
        rain_chance_pct = 0.0
        humidity_pct = 50.0
        weather_mode = "current"
        weather_status = "ok_current"
        weather_error = ""
        elevation_ft = 0.0
        is_dome = False

        if not home_cid:
            print(f"  Warning: no home_cid for game {game_num} — using neutral weather", file=sys.stderr)
            weather_status = "missing_home_cid"
            weather_error = "no_home_cid"
        elif sinfo is None:
            print(f"  Warning: no stadium data for {home_cid} — using neutral weather", file=sys.stderr)
            weather_status = "missing_stadium"
            weather_error = "no_stadium_data"
        else:
            elevation_ft = sinfo.get("elevation_ft", 0.0)
            is_dome = sinfo.get("is_dome", False)
            try:
                w = get_weather_park_adj(
                    canonical_id=home_cid,
                    stadium_csv=stadium_csv,
                    game_date=date if date else None,
                    game_start_hour=start_local_hour,
                )
                if "error" not in w:
                    wind_adj_raw = w.get("wind_adj_raw", 0.0)
                    non_wind_adj = w.get("non_wind_adj", 0.0)
                    wind_out_mph = w.get("wind_out_mph", 0.0)
                    wind_out_lf = w.get("wind_out_lf_mph", 0.0)
                    wind_out_cf = w.get("wind_out_cf_mph", 0.0)
                    wind_out_rf = w.get("wind_out_rf_mph", 0.0)
                    temp_f = w.get("temp_f", 72.0)
                    wind_mph = w.get("wind_speed_mph", 0.0)
                    wind_dir_deg = w.get("wind_dir_deg", 0.0)
                    wind_gusts_mph = w.get("wind_gusts_mph", 0.0)
                    rain_chance_pct = w.get("precip_prob_pct", 0.0)
                    humidity_pct = w.get("humidity_pct", 50.0)
                    weather_mode = w.get("weather_mode", "current")
                    if w.get("is_dome"):
                        weather_status = "dome_bypass"
                    else:
                        weather_status = "ok_hourly" if weather_mode == "hourly_avg" else "ok_current"
                    elevation_ft = w.get("elevation_ft", elevation_ft)
                else:
                    print(f"  Weather warning: {w['error']}", file=sys.stderr)
                    weather_status = "api_error"
                    weather_error = str(w.get("error", "api_error"))
            except Exception as e:
                print(f"  Weather failed for {home_cid}: {e}", file=sys.stderr)
                weather_status = "exception"
                weather_error = str(e)

        # Altitude double-counting guard: park factors computed from game data
        # already embed the altitude effect. Subtract expected altitude component
        # to avoid counting it twice (once in park_factor, once in alt_adj).
        # Only adjust for non-trivial elevations where the correction matters.
        if park_factor != 0.0 and elevation_ft > 500:
            density_r = air_density_ratio(elevation_ft)
            expected_alt_in_pf = ALTITUDE_COEFF * (1.0 - density_r)
            park_factor = park_factor - expected_alt_in_pf

        rows.append(
            {
                "game_num": game_num,
                "home_cid": home_cid,
                "park_factor": round(park_factor, 6),
                "wind_adj_raw": round(wind_adj_raw, 6),
                "non_wind_adj": round(non_wind_adj, 6),
                "wind_out_mph": round(wind_out_mph, 2),
                "wind_out_lf": round(wind_out_lf, 2),
                "wind_out_cf": round(wind_out_cf, 2),
                "wind_out_rf": round(wind_out_rf, 2),
                "temp_f": round(temp_f, 1),
                "wind_mph": round(wind_mph, 1),
                "wind_dir_deg": round(wind_dir_deg, 1),
                "wind_gusts_mph": round(wind_gusts_mph, 1),
                "rain_chance_pct": round(rain_chance_pct, 1),
                "humidity_pct": round(humidity_pct, 1),
                "weather_mode": weather_mode,
                "weather_status": weather_status,
                "weather_error": weather_error,
                "elevation_ft": round(elevation_ft, 1),
                "is_dome": is_dome,
            }
        )

    df = pd.DataFrame(rows)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"Wrote {len(df)} rows to {out_csv}", file=sys.stderr)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch weather + park factors for each game in a schedule CSV.",
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        required=True,
        help="Path to schedule CSV (must have game_num, home_cid, start_local_hour columns)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="",
        help="Game date (YYYY-MM-DD) — used for hourly weather forecasts",
    )
    parser.add_argument(
        "--stadium-csv",
        type=Path,
        default=Path("data/registries/stadium_orientations.csv"),
        help="Path to stadium orientations CSV",
    )
    parser.add_argument(
        "--park-factors",
        type=Path,
        default=Path("data/processed/park_factors.csv"),
        help="Path to park factors CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: data/daily/{date}/weather.csv if --date given)",
    )
    args = parser.parse_args()

    out_csv = args.out
    if out_csv is None and args.date:
        out_csv = Path(f"data/daily/{args.date}/weather.csv")

    df = resolve_weather(
        schedule_csv=args.schedule,
        stadium_csv=args.stadium_csv,
        park_factors_csv=args.park_factors,
        date=args.date,
        out_csv=out_csv,
    )

    # Print summary to stdout
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
