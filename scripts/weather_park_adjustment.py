"""
Live weather + wind park factor adjustment for game simulations.

Fetches current weather from Open-Meteo (free, no API key needed) using stadium
lat/lon, then computes a wind-based park factor adjustment based on the wind component
blowing out from home plate toward center field.

Stadium orientations (home plate → center field compass bearing) and lat/lon are
loaded from data/registries/stadium_orientations.csv.

Physics model:
  - Wind blowing out → carries fly balls further → more runs
  - Wind blowing in → suppresses fly balls → fewer runs
  - Empirical effect: ~0.04 log-rate change per mph of wind-out component
    (calibrated from MLB data: 10 mph tailwind ≈ +0.5 runs/game ≈ +0.04*10 = 0.4 log-scale)
  - Cross-wind and temperature effects are secondary but included

Usage:
  from weather_park_adjustment import get_weather_park_adj
  adj = get_weather_park_adj("NCAA_614704")  # Oregon St
  # Returns: {"wind_adj": 0.032, "temp_adj": -0.015, "total_adj": 0.017,
  #           "wind_out_mph": 3.2, "wind_speed": 8.5, "wind_dir": 225, "temp_f": 58}

  # Or from CLI:
  python3 scripts/weather_park_adjustment.py --team NCAA_614704
  python3 scripts/weather_park_adjustment.py --lat 44.5646 --lon -123.2620 --bearing 67
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Wind effect: log-rate change per mph of wind blowing out from home plate
# Empirical: 10 mph tailwind → ~8% more runs per team → log(1.08)/10 ≈ 0.008/mph
# Effect is primarily on fly-ball distance (~3-4 ft/mph), translating to HR and XBH
WIND_OUT_COEFF = 0.008

# Temperature effect: log-rate change per degree F above/below 72°F baseline
# Warmer air = less dense = balls carry farther
TEMP_COEFF = 0.002  # ~0.02 runs/game per 10°F above baseline
TEMP_BASELINE_F = 72.0

# Altitude effect: log-rate change per unit of air density reduction
# Based on Bahill et al. (2009) and Nathan's empirical data:
#   - Every 10% air density decrease → ~4% more fly ball distance
#   - Denver (5280 ft, 18% density reduction) → ~5% more distance, ~7.5% carry
#   - Coefficient maps (1 - density_ratio) to log-rate scoring change
#   - 0.25 calibrated so Denver: 0.25 * 0.18 = 0.045 → exp(0.045) = +4.6% runs
ALTITUDE_COEFF = 0.25

# Default home plate bearing if not in stadium data (MLB rule: ENE ~67°)
DEFAULT_HP_BEARING = 67.0

# Wind speed below which direction is negligible
CALM_WIND_MPH = 3.0


# ──────────────────────────────────────────────────────────────────────────────
# Stadium orientation data
# ──────────────────────────────────────────────────────────────────────────────

def load_stadium_data(csv_path: Path) -> pd.DataFrame:
    """Load stadium orientations CSV with columns:
    canonical_id, venue_name, lat, lon, hp_bearing_deg, source
    """
    if not csv_path.exists():
        return pd.DataFrame(columns=["canonical_id", "venue_name", "lat", "lon",
                                      "hp_bearing_deg", "source"])
    return pd.read_csv(csv_path)


def get_stadium_info(
    canonical_id: str,
    stadium_df: pd.DataFrame,
) -> dict | None:
    """Look up stadium lat/lon/bearing for a team's home field."""
    if stadium_df.empty:
        return None
    row = stadium_df[stadium_df["canonical_id"] == canonical_id]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "lat": float(r["lat"]),
        "lon": float(r["lon"]),
        "hp_bearing": float(r.get("hp_bearing_deg", DEFAULT_HP_BEARING)),
        "venue_name": str(r.get("venue_name", "")),
        "elevation_ft": float(r.get("elevation_ft", 0)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Weather API (Open-Meteo — free, no API key)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_current_weather(lat: float, lon: float) -> dict | None:
    """
    Fetch current weather from Open-Meteo API.
    Returns dict with wind_speed_mph, wind_direction_deg, temperature_f, wind_gusts_mph.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m"
        f"&temperature_unit=fahrenheit"
        f"&wind_speed_unit=mph"
        f"&timezone=auto"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ncaa-baseball-model/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
        print(f"Weather API error: {e}", file=sys.stderr)
        return None

    current = data.get("current", {})
    return {
        "wind_speed_mph": float(current.get("wind_speed_10m", 0)),
        "wind_direction_deg": float(current.get("wind_direction_10m", 0)),
        "temperature_f": float(current.get("temperature_2m", TEMP_BASELINE_F)),
        "wind_gusts_mph": float(current.get("wind_gusts_10m", 0)),
    }


def fetch_hourly_weather(lat: float, lon: float, date: str) -> list[dict] | None:
    """
    Fetch full day of hourly forecasts from Open-Meteo API.

    Args:
        lat, lon: stadium coordinates
        date: YYYY-MM-DD

    Returns list of 24 hourly dicts with keys:
        time (ISO str), wind_speed_mph, wind_direction_deg, temperature_f, wind_gusts_mph
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m"
        f"&temperature_unit=fahrenheit"
        f"&wind_speed_unit=mph"
        f"&timezone=auto"
        f"&start_date={date}&end_date={date}"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ncaa-baseball-model/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
        print(f"Hourly weather API error: {e}", file=sys.stderr)
        return None

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return None

    results = []
    for i, t in enumerate(times):
        results.append({
            "time": t,
            "wind_speed_mph": float(hourly.get("wind_speed_10m", [0])[i]),
            "wind_direction_deg": float(hourly.get("wind_direction_10m", [0])[i]),
            "temperature_f": float(hourly.get("temperature_2m", [TEMP_BASELINE_F])[i]),
            "wind_gusts_mph": float(hourly.get("wind_gusts_10m", [0])[i]),
        })
    return results


def get_weather_at_hours(
    hourly_data: list[dict],
    start_hour_local: int,
    offsets: tuple[int, ...] = (0, 1, 2),
) -> list[dict]:
    """
    Extract weather at specific hour offsets from game start.

    Args:
        hourly_data: list of 24 hourly weather dicts (from fetch_hourly_weather)
        start_hour_local: local hour (0-23) of game start
        offsets: hours after start to sample (default: start, +1hr, +2hr)

    Returns list of weather dicts for each offset hour.
    """
    result = []
    for off in offsets:
        idx = start_hour_local + off
        if 0 <= idx < len(hourly_data):
            result.append(hourly_data[idx])
    return result


def average_weather(weather_list: list[dict]) -> dict:
    """Average multiple hourly weather readings into one composite."""
    if not weather_list:
        return {
            "wind_speed_mph": 0,
            "wind_direction_deg": 0,
            "temperature_f": TEMP_BASELINE_F,
            "wind_gusts_mph": 0,
        }
    n = len(weather_list)
    # For wind direction, use vector averaging to handle wraparound (e.g., 350° + 10°)
    sin_sum = sum(math.sin(math.radians(w["wind_direction_deg"])) for w in weather_list)
    cos_sum = sum(math.cos(math.radians(w["wind_direction_deg"])) for w in weather_list)
    avg_dir = math.degrees(math.atan2(sin_sum / n, cos_sum / n)) % 360

    return {
        "wind_speed_mph": sum(w["wind_speed_mph"] for w in weather_list) / n,
        "wind_direction_deg": round(avg_dir, 1),
        "temperature_f": sum(w["temperature_f"] for w in weather_list) / n,
        "wind_gusts_mph": sum(w["wind_gusts_mph"] for w in weather_list) / n,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Altitude / air density
# ──────────────────────────────────────────────────────────────────────────────

def air_density_ratio(elevation_ft: float) -> float:
    """
    Ratio of air density at elevation vs sea level (standard atmosphere).

    Uses the barometric formula:
        ρ/ρ₀ = (1 - L·h/T₀)^(g·M/(R·L))

    Where L=lapse rate, h=altitude, T₀=288.15K, g=9.80665, M=0.0289644, R=8.31447.
    Simplified: (1 - 2.25577e-5 × elevation_m)^5.25588

    Returns 1.0 at sea level, ~0.82 at Denver (5280 ft), ~0.78 at 7000 ft.
    """
    elevation_m = elevation_ft * 0.3048
    return (1 - 2.25577e-5 * elevation_m) ** 5.25588


# ──────────────────────────────────────────────────────────────────────────────
# Wind component calculation
# ──────────────────────────────────────────────────────────────────────────────

def wind_out_component(
    wind_dir_deg: float,
    wind_speed_mph: float,
    hp_bearing_deg: float,
) -> float:
    """
    Compute wind component blowing OUT from home plate toward center field.

    Args:
        wind_dir_deg: meteorological direction wind comes FROM (0=N, 90=E, 180=S, 270=W)
        wind_speed_mph: wind speed
        hp_bearing_deg: compass bearing from home plate to center field

    Returns:
        positive = blowing out (balls carry farther)
        negative = blowing in (balls suppressed)
    """
    if wind_speed_mph < CALM_WIND_MPH:
        return 0.0

    # Wind blows FROM wind_dir, so it travels TOWARD (wind_dir + 180)
    wind_toward = (wind_dir_deg + 180.0) % 360.0

    # Angle between wind travel direction and home→CF bearing
    angle_diff = math.radians(wind_toward - hp_bearing_deg)

    return wind_speed_mph * math.cos(angle_diff)


# ──────────────────────────────────────────────────────────────────────────────
# Combined adjustment
# ──────────────────────────────────────────────────────────────────────────────

def compute_weather_adjustment(
    wind_speed_mph: float,
    wind_direction_deg: float,
    temperature_f: float,
    hp_bearing_deg: float,
    elevation_ft: float = 0.0,
) -> dict:
    """
    Compute log-scale park factor adjustment from current weather + altitude.

    Returns dict with:
      wind_out_mph: component of wind blowing out (+ = out, - = in)
      wind_adj: log-rate adjustment from wind
      temp_adj: log-rate adjustment from temperature
      alt_adj: log-rate adjustment from altitude (air density reduction)
      total_adj: combined log-rate adjustment (add to existing park_factor)
    """
    w_out = wind_out_component(wind_direction_deg, wind_speed_mph, hp_bearing_deg)
    wind_adj = WIND_OUT_COEFF * w_out
    temp_adj = TEMP_COEFF * (temperature_f - TEMP_BASELINE_F)

    # Altitude: lower air density → less drag → more carry
    density_ratio = air_density_ratio(elevation_ft)
    alt_adj = ALTITUDE_COEFF * (1.0 - density_ratio)

    return {
        "wind_out_mph": round(w_out, 1),
        "wind_adj": round(wind_adj, 4),
        "temp_adj": round(temp_adj, 4),
        "alt_adj": round(alt_adj, 4),
        "density_ratio": round(density_ratio, 4),
        "elevation_ft": round(elevation_ft),
        "total_adj": round(wind_adj + temp_adj + alt_adj, 4),
    }


def get_weather_park_adj(
    canonical_id: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    hp_bearing: float | None = None,
    stadium_csv: Path = Path("data/registries/stadium_orientations.csv"),
    game_date: str | None = None,
    game_start_hour: int | None = None,
) -> dict:
    """
    Full pipeline: look up stadium → fetch weather → compute adjustment.

    Provide either canonical_id (to look up from CSV) or lat/lon/hp_bearing directly.

    If game_date and game_start_hour are provided, uses hourly forecast data
    averaged over start, +1hr, and +2hr to account for wind direction changes
    during the game. Otherwise falls back to current conditions.
    """
    venue_name = ""
    elevation_ft = 0.0
    if canonical_id and (lat is None or lon is None):
        sdf = load_stadium_data(stadium_csv)
        info = get_stadium_info(canonical_id, sdf)
        if info is None:
            return {"error": f"No stadium data for {canonical_id}", "total_adj": 0.0}
        lat = info["lat"]
        lon = info["lon"]
        hp_bearing = hp_bearing or info["hp_bearing"]
        venue_name = info["venue_name"]
        elevation_ft = info.get("elevation_ft", 0.0)

    if lat is None or lon is None:
        return {"error": "No lat/lon provided", "total_adj": 0.0}
    hp_bearing = hp_bearing or DEFAULT_HP_BEARING

    # ── Hourly mode: average weather over game duration ──────────────────
    hourly_detail = []
    if game_date and game_start_hour is not None:
        hourly_data = fetch_hourly_weather(lat, lon, game_date)
        if hourly_data:
            hours = get_weather_at_hours(hourly_data, game_start_hour, offsets=(0, 1, 2))
            if hours:
                # Store per-hour wind detail for output
                for i, h in enumerate(hours):
                    w_out = wind_out_component(h["wind_direction_deg"],
                                                h["wind_speed_mph"], hp_bearing)
                    hourly_detail.append({
                        "hour_offset": i,
                        "time": h.get("time", ""),
                        "wind_speed_mph": round(h["wind_speed_mph"], 1),
                        "wind_dir_deg": round(h["wind_direction_deg"]),
                        "wind_out_mph": round(w_out, 1),
                        "temp_f": round(h["temperature_f"], 1),
                        "wind_gusts_mph": round(h["wind_gusts_mph"], 1),
                    })
                weather = average_weather(hours)
            else:
                weather = fetch_current_weather(lat, lon)
        else:
            weather = fetch_current_weather(lat, lon)
    else:
        weather = fetch_current_weather(lat, lon)

    if weather is None:
        return {"error": "Weather API failed", "total_adj": 0.0}

    adj = compute_weather_adjustment(
        weather["wind_speed_mph"],
        weather["wind_direction_deg"],
        weather["temperature_f"],
        hp_bearing,
        elevation_ft=elevation_ft,
    )
    adj.update({
        "wind_speed_mph": weather["wind_speed_mph"],
        "wind_dir_deg": weather["wind_direction_deg"],
        "wind_gusts_mph": weather.get("wind_gusts_mph", 0),
        "temp_f": weather["temperature_f"],
        "hp_bearing_deg": hp_bearing,
        "venue_name": venue_name,
        "lat": lat,
        "lon": lon,
    })
    if hourly_detail:
        adj["hourly_wind"] = hourly_detail
        adj["weather_mode"] = "hourly_avg"
    else:
        adj["weather_mode"] = "current"
    return adj


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Get live weather park factor adjustment for a stadium.",
    )
    parser.add_argument("--team", type=str, help="Team canonical_id (e.g. NCAA_614704)")
    parser.add_argument("--lat", type=float, help="Stadium latitude")
    parser.add_argument("--lon", type=float, help="Stadium longitude")
    parser.add_argument("--bearing", type=float, help="Home plate → CF bearing (degrees)")
    parser.add_argument("--stadium-csv", type=Path,
                        default=Path("data/registries/stadium_orientations.csv"))
    args = parser.parse_args()

    result = get_weather_park_adj(
        canonical_id=args.team,
        lat=args.lat,
        lon=args.lon,
        hp_bearing=args.bearing,
        stadium_csv=args.stadium_csv,
    )

    if "error" in result:
        print(f"Warning: {result['error']}")
    else:
        print(f"Stadium: {result.get('venue_name', 'Custom location')}")
        print(f"Location: ({result['lat']:.4f}, {result['lon']:.4f})")
        print(f"HP bearing: {result['hp_bearing_deg']:.0f}° (home plate → center field)")
        print(f"Weather: {result['temp_f']:.0f}°F, wind {result['wind_speed_mph']:.0f} mph "
              f"from {result['wind_dir_deg']:.0f}° (gusts {result['wind_gusts_mph']:.0f})")
        print(f"Wind out component: {result['wind_out_mph']:.1f} mph")
        print(f"Elevation: {result.get('elevation_ft', 0):.0f} ft "
              f"(air density ratio: {result.get('density_ratio', 1.0):.3f})")
        print(f"Wind adjustment:     {result['wind_adj']:+.4f} (log-rate)")
        print(f"Temp adjustment:     {result['temp_adj']:+.4f} (log-rate)")
        print(f"Altitude adjustment: {result.get('alt_adj', 0):+.4f} (log-rate)")
        print(f"TOTAL adjustment:    {result['total_adj']:+.4f} (log-rate)")
        total_exp = math.exp(result['total_adj'])
        print(f"  → {total_exp:.3f}x run multiplier ({(total_exp-1)*100:+.1f}% runs)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
