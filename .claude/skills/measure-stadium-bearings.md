---
name: measure-stadium-bearings
description: Measure home plate to center field compass bearings for NCAA stadiums using OpenStreetMap polygon data. Use when stadiums have default 67° bearings that need real measurements.
user_invocable: true
---

# Stadium Bearing Measurement

Measures the compass bearing from home plate (HP) toward center field (CF) for NCAA baseball stadiums using OpenStreetMap (Overpass API) polygon data.

## Why This Matters

The wind-out calculation `wind_speed * cos(wind_toward - hp_bearing)` is the core weather adjustment. A wrong bearing can flip wind from +20 mph OUT to -20 mph IN, swinging predicted totals by 5+ runs. Default 67° bearings MUST be replaced with real measurements.

## How It Works

1. Query Overpass API for baseball stadium/pitch polygons near the stadium's lat/lon
2. Extract polygon geometry (vertices as lat/lon pairs)
3. Find the vertex with the sharpest interior angle — this is home plate (baseball fields have a characteristic ~90° angle at HP)
4. Compute compass bearing from that vertex toward the polygon centroid (roughly center field direction)
5. Update `data/registries/stadium_orientations.csv`

## Scripts

### Primary: `scripts/measure_stadium_bearings.py`

```bash
# Process all stadiums still at default 67°
.venv/bin/python3 scripts/measure_stadium_bearings.py --only-default

# Single stadium
.venv/bin/python3 scripts/measure_stadium_bearings.py --canonical-id BSB_TEXAS_TECH

# Dry run (don't write CSV)
.venv/bin/python3 scripts/measure_stadium_bearings.py --only-default --dry-run --batch 10
```

Uses 800m search radius, 4s rate limit, 3 retries with 8s backoff.

### Retry: `scripts/measure_bearings_retry.py`

For stadiums the primary script missed. Tries 5 strategies:
1. **wider_1200m** — Same queries, 1200m radius
2. **very_wide_2500m** — Baseball pitches only, 2500m radius
3. **relations_1500m** — Multi-polygon relations (large stadium complexes)
4. **any_baseball_1500m** — Any way tagged `sport=baseball`
5. **landuse_1500m** — Recreation grounds / sports centres (checks for baseball-shaped polygon with min angle < 120°)

```bash
.venv/bin/python3 scripts/measure_bearings_retry.py
```

Uses 5s rate limit, 10s retry backoff. Slower but catches many more stadiums.

## Overpass API Rate Limiting

The Overpass API is free but rate-limited. Key settings:
- **Rate limit:** 4-5 seconds between queries minimum
- **Retries:** 3 attempts on HTTP 429 (Too Many Requests) or 504 (Gateway Timeout)
- **Backoff:** `RETRY_BACKOFF * (attempt + 1)` seconds (8-10s base)
- **User-Agent:** Always set to `ncaa-baseball-model/1.0`
- **Long runs:** 314 stadiums at 5s/query ≈ 26 minutes minimum; retries can double this

Run with `nohup` for the full batch:
```bash
nohup .venv/bin/python3 scripts/measure_bearings_retry.py > /tmp/bearings_retry.log 2>&1 &
```

## Validation

After measuring, spot-check suspicious bearings:
- Bearings near 0° or 360° (due north) are unusual but valid (e.g., Oklahoma = 6°)
- Bearings > 300° (NW-facing) exist (e.g., Louisville = 343°, UT Martin = 321°)
- If a bearing seems wrong, verify via satellite imagery (Google Maps satellite view)
- The `source` column tracks provenance: `osm_polygon_estimated`, `osm_wider_1200m`, etc.

## Stadium Orientations CSV Format

```
canonical_id,venue_name,lat,lon,hp_bearing_deg,source
BSB_TEXAS_TECH,Dan Law Field at Rip Griffin Park,33.5886,-101.8767,172,osm_polygon_verified
```

- `hp_bearing_deg`: Compass degrees, 0=N, 90=E, 180=S, 270=W
- `source`: How the bearing was determined
  - `default` — Still at 67° (needs measurement)
  - `osm_polygon_estimated` — From primary OSM script
  - `osm_wider_1200m`, `osm_very_wide_2500m`, etc. — From retry script strategies
  - `osm_polygon_verified` — Confirmed against satellite/user knowledge

## Current Coverage

As of 2026-03-11: **307/314 stadiums** have real OSM-derived bearings. 7 remain at default (no OSM polygon data): Indiana, Old Dominion, Samford, South Dakota St., SFA, North Alabama, Alcorn State. These may require manual satellite measurement.
