"""
Rank stadium home-plate orientation rows to manually verify first.

Prioritization uses current weather sensitivity:
  - How much effective wind-out (mph) changes under +/-20 degree bearing error
  - Converted to log-rate uncertainty via WIND_OUT_COEFF
  - Weighted by source confidence (low > medium > high)

Usage:
  python3 scripts/rank_orientation_verification.py \
    --schedule data/daily/2026-03-15/schedule_weather_audit.csv \
    --weather data/daily/2026-03-15/weather_audit.csv \
    --stadium data/registries/stadium_orientations.csv \
    --top 20 \
    --out output/orientation_priority_top20_2026-03-15.csv
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from weather_park_adjustment import WIND_OUT_COEFF, wind_out_directional


def orientation_confidence(source: str) -> str:
    src = (source or "").strip()
    high = {"manual_satellite", "satellite/known", "osm_polygon_verified"}
    low = {
        "osm_wider_1200m",
        "osm_very_wide_2500m",
        "osm_landuse_1500m",
        "osm_any_baseball_1500m",
        "manual_satellite_old_field",
    }
    if src in high:
        return "high"
    if src in low:
        return "low"
    return "medium"


def confidence_weight(conf: str) -> float:
    if conf == "low":
        return 1.3
    if conf == "medium":
        return 1.0
    return 0.6


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank stadium orientation manual verification priorities.")
    parser.add_argument("--schedule", type=Path, required=True, help="Schedule CSV (must include game_num/home_cid/home_name).")
    parser.add_argument("--weather", type=Path, required=True, help="Weather CSV (must include game_num/home_cid/wind_mph/wind_dir_deg).")
    parser.add_argument("--stadium", type=Path, default=Path("data/registries/stadium_orientations.csv"))
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("output/orientation_priority_top20.csv"))
    args = parser.parse_args()

    schedule = pd.read_csv(args.schedule, dtype=str)
    weather = pd.read_csv(args.weather, dtype=str)
    stadium = pd.read_csv(args.stadium, dtype=str)

    for df in (schedule, weather):
        df["game_num"] = pd.to_numeric(df.get("game_num"), errors="coerce").astype("Int64")
    weather["wind_mph"] = pd.to_numeric(weather.get("wind_mph"), errors="coerce")
    weather["wind_dir_deg"] = pd.to_numeric(weather.get("wind_dir_deg"), errors="coerce")

    stadium["hp_bearing_deg"] = pd.to_numeric(stadium.get("hp_bearing_deg"), errors="coerce")
    stadium["source"] = stadium.get("source", "").fillna("")
    stadium["orientation_confidence"] = stadium["source"].apply(orientation_confidence)

    merged = schedule.merge(
        weather[["game_num", "home_cid", "wind_mph", "wind_dir_deg"]],
        on=["game_num", "home_cid"],
        how="left",
    ).merge(
        stadium[["canonical_id", "venue_name", "hp_bearing_deg", "source", "orientation_confidence", "elevation_ft"]],
        left_on="home_cid",
        right_on="canonical_id",
        how="left",
    )

    rows: list[dict] = []
    for _, r in merged.iterrows():
        try:
            ws = float(r["wind_mph"])
            wd = float(r["wind_dir_deg"])
            bearing = float(r["hp_bearing_deg"])
        except (TypeError, ValueError):
            continue

        base = wind_out_directional(wd, ws, bearing)["wind_out_eff"]
        plus20 = wind_out_directional(wd, ws, bearing + 20.0)["wind_out_eff"]
        minus20 = wind_out_directional(wd, ws, bearing - 20.0)["wind_out_eff"]
        delta20 = max(abs(plus20 - base), abs(minus20 - base))

        conf = str(r.get("orientation_confidence", "medium"))
        unc_log = WIND_OUT_COEFF * float(delta20)
        score = unc_log * confidence_weight(conf)

        rows.append(
            {
                "game_num": int(r["game_num"]) if pd.notna(r["game_num"]) else None,
                "home_cid": str(r.get("home_cid", "")),
                "home_name": str(r.get("home_name", "")),
                "venue_name": str(r.get("venue_name", "")),
                "source": str(r.get("source", "")),
                "orientation_confidence": conf,
                "hp_bearing_deg": round(float(bearing), 1),
                "wind_mph": round(float(ws), 1),
                "wind_dir_deg": round(float(wd), 1),
                "wind_out_eff_mph": round(float(base), 2),
                "delta20_mph": round(float(delta20), 2),
                "wind_lograte_uncertainty_20deg": round(float(unc_log), 4),
                "priority_score": round(float(score), 4),
                "elevation_ft": r.get("elevation_ft", ""),
            }
        )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        print("No rankable rows found.")
        return 0

    conf_rank = {"low": 0, "medium": 1, "high": 2}
    out_df["_conf_rank"] = out_df["orientation_confidence"].map(conf_rank).fillna(1).astype(int)
    out_df = out_df.sort_values(
        ["priority_score", "_conf_rank", "delta20_mph"],
        ascending=[False, True, False],
    ).drop(columns=["_conf_rank"])

    top_df = out_df.head(max(1, args.top)).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    top_df.to_csv(args.out, index=False)

    print(f"Wrote {len(top_df)} rows -> {args.out}")
    print(top_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

