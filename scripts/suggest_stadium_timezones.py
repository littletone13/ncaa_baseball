"""
Suggest and apply stadium timezones from lat/lon.

Uses Open-Meteo's geotimezone resolution via forecast metadata to return
IANA timezone names (e.g. America/Chicago).

Usage:
  python3 scripts/suggest_stadium_timezones.py
  python3 scripts/suggest_stadium_timezones.py --apply
  python3 scripts/suggest_stadium_timezones.py --stadium-csv data/registries/stadium_orientations.csv --apply
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd


def _fetch_timezone(lat: float, lon: float, timeout_sec: float = 12.0) -> tuple[str, int | None] | None:
    """Return (timezone_name, utc_offset_seconds) from Open-Meteo metadata."""
    # Keep payload minimal; timezone metadata is returned in root object.
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=weather_code"
        "&timezone=auto"
        "&forecast_days=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "ncaa-baseball-model/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    tz = str(data.get("timezone", "")).strip()
    if not tz:
        return None
    offset = data.get("utc_offset_seconds")
    try:
        offset_i = int(offset) if offset is not None else None
    except (TypeError, ValueError):
        offset_i = None
    return tz, offset_i


def _is_valid_tz(tz_name: str) -> bool:
    try:
        ZoneInfo(tz_name)
        return True
    except ZoneInfoNotFoundError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Suggest/apply stadium timezones from lat/lon.")
    parser.add_argument(
        "--stadium-csv",
        type=Path,
        default=Path("data/registries/stadium_orientations.csv"),
        help="Stadium registry CSV with lat/lon.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/stadium_timezone_suggestions.csv"),
        help="Where to write suggestion table.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply suggested timezone values into the stadium CSV.",
    )
    parser.add_argument(
        "--sleep-ms",
        type=int,
        default=120,
        help="Sleep between API calls in milliseconds (default: 120).",
    )
    args = parser.parse_args()

    if not args.stadium_csv.exists():
        raise SystemExit(f"Missing stadium CSV: {args.stadium_csv}")

    df = pd.read_csv(args.stadium_csv, dtype=str)
    for col in ("lat", "lon"):
        if col not in df.columns:
            raise SystemExit(f"Stadium CSV missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Reuse lookups for identical coordinates.
    cache: dict[tuple[float, float], tuple[str, int | None] | None] = {}
    rows: list[dict] = []
    n_ok = 0
    n_fail = 0

    for _, r in df.iterrows():
        cid = str(r.get("canonical_id", "")).strip()
        venue = str(r.get("venue_name", "")).strip()
        lat = r.get("lat")
        lon = r.get("lon")
        existing_tz = str(r.get("timezone", "")).strip()

        if pd.isna(lat) or pd.isna(lon):
            rows.append(
                {
                    "canonical_id": cid,
                    "venue_name": venue,
                    "lat": lat,
                    "lon": lon,
                    "timezone_existing": existing_tz,
                    "timezone_suggested": "",
                    "utc_offset_seconds": "",
                    "status": "missing_latlon",
                }
            )
            n_fail += 1
            continue

        key = (round(float(lat), 6), round(float(lon), 6))
        if key not in cache:
            result = None
            for attempt in range(3):
                try:
                    result = _fetch_timezone(float(lat), float(lon))
                    break
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                    if attempt < 2:
                        time.sleep(0.35 * (attempt + 1))
            cache[key] = result
            time.sleep(max(args.sleep_ms, 0) / 1000.0)

        result = cache.get(key)
        if result is None:
            rows.append(
                {
                    "canonical_id": cid,
                    "venue_name": venue,
                    "lat": lat,
                    "lon": lon,
                    "timezone_existing": existing_tz,
                    "timezone_suggested": "",
                    "utc_offset_seconds": "",
                    "status": "lookup_failed",
                }
            )
            n_fail += 1
            continue

        tz_name, offset = result
        valid = _is_valid_tz(tz_name)
        status = "ok" if valid else "invalid_tz"
        if valid:
            n_ok += 1
        else:
            n_fail += 1
        rows.append(
            {
                "canonical_id": cid,
                "venue_name": venue,
                "lat": lat,
                "lon": lon,
                "timezone_existing": existing_tz,
                "timezone_suggested": tz_name,
                "utc_offset_seconds": offset if offset is not None else "",
                "status": status,
            }
        )

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    if args.apply:
        if "timezone" not in df.columns:
            df["timezone"] = ""
        if "timezone_source" not in df.columns:
            df["timezone_source"] = ""

        sug_map = {
            (str(r["canonical_id"]).strip()): str(r["timezone_suggested"]).strip()
            for _, r in out_df.iterrows()
            if str(r.get("status", "")).strip() == "ok"
        }
        applied = 0
        for i, r in df.iterrows():
            cid = str(r.get("canonical_id", "")).strip()
            tz = sug_map.get(cid, "")
            if not tz:
                continue
            old = str(r.get("timezone", "")).strip()
            if old != tz:
                df.at[i, "timezone"] = tz
                df.at[i, "timezone_source"] = "open-meteo-latlon"
                applied += 1

        df.to_csv(args.stadium_csv, index=False)
        print(f"Applied timezone updates to {args.stadium_csv} (changed rows: {applied})")

    print(f"Wrote suggestions: {args.out}")
    print(f"Lookup ok: {n_ok} | failed/invalid: {n_fail} | total: {len(out_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

