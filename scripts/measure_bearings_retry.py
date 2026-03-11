"""
Retry bearing measurement for stadiums still at default 67° using
expanded query strategies:
  1. Original query with wider radius (1200m, 2000m)
  2. Relations (multi-polygon stadiums) not just ways
  3. leisure=pitch sport=baseball with wider radius
  4. Any way with sport=baseball tag nearby
  5. Nominatim name search → OSM polygon around those coords
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

STADIUM_CSV = Path("data/registries/stadium_orientations.csv")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
RATE_LIMIT_SEC = 5.0
MAX_RETRIES = 3
RETRY_BACKOFF = 10


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2)
         - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def overpass_query(query: str, timeout: int = 20) -> dict | None:
    data = urllib.parse.urlencode({"data": query}).encode()
    for attempt in range(MAX_RETRIES):
        req = urllib.request.Request(OVERPASS_URL, data=data,
                                     headers={"User-Agent": "ncaa-baseball-model/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 504) and attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF * (attempt + 1)
                print(f"      Overpass {e.code}, retry {attempt+1}/{MAX_RETRIES} in {wait}s...",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"      Overpass error: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"      Overpass error: {e}", file=sys.stderr)
            return None
    return None


def extract_polygons_from_result(result: dict) -> list[list[tuple[float, float]]]:
    """Extract polygon coordinate lists from Overpass result, handling both ways and relations."""
    if result is None:
        return []
    elements = result.get("elements", [])
    nodes = {e["id"]: (e["lat"], e["lon"]) for e in elements if e["type"] == "node"}
    ways = [e for e in elements if e["type"] == "way"]

    polygons = []
    for w in ways:
        tags = w.get("tags", {})
        sport = tags.get("sport", "").lower()
        # Skip non-baseball sport tags (but allow untagged ways that are part of stadium)
        if sport and "baseball" not in sport and "softball" not in sport:
            continue
        node_ids = w.get("nodes", [])
        coords = [(nodes[nid][0], nodes[nid][1]) for nid in node_ids if nid in nodes]
        if len(coords) >= 4:
            polygons.append(coords)

    return polygons


def estimate_bearing_from_polygon(coords):
    n = len(coords)
    if n < 4:
        return None
    if coords[0] == coords[-1]:
        coords = coords[:-1]
        n = len(coords)
    if n < 4:
        return None

    def angle_at(i):
        p0 = coords[(i - 1) % n]
        p1 = coords[i]
        p2 = coords[(i + 1) % n]
        v1 = (p0[0] - p1[0], p0[1] - p1[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if mag1 < 1e-12 or mag2 < 1e-12:
            return 180.0
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))

    angles = [(angle_at(i), i) for i in range(n)]
    angles.sort()
    hp_angle, hp_idx = angles[0]
    hp_lat, hp_lon = coords[hp_idx]
    cx = sum(c[0] for c in coords) / n
    cy = sum(c[1] for c in coords) / n
    b = bearing_deg(hp_lat, hp_lon, cx, cy)
    return round(b)


# ── Query strategies ──────────────────────────────────────────────────────

def strategy_wider_radius(lat, lon, radius=1200):
    """Same as original but with wider radius."""
    query = f"""
    [out:json][timeout:15];
    (
      way["leisure"="stadium"]["sport"~"baseball"](around:{radius},{lat},{lon});
      way["leisure"="pitch"]["sport"~"baseball"](around:{radius},{lat},{lon});
      way["leisure"="stadium"](around:{radius},{lat},{lon});
      way["building"~"stadium"](around:{radius},{lat},{lon});
    );
    (._;>;);
    out body;
    """
    return extract_polygons_from_result(overpass_query(query))


def strategy_very_wide(lat, lon, radius=2500):
    """Even wider radius, baseball pitches only."""
    query = f"""
    [out:json][timeout:15];
    (
      way["leisure"="pitch"]["sport"~"baseball"](around:{radius},{lat},{lon});
      way["leisure"="pitch"]["sport"~"softball"](around:{radius},{lat},{lon});
    );
    (._;>;);
    out body;
    """
    return extract_polygons_from_result(overpass_query(query))


def strategy_relations(lat, lon, radius=1500):
    """Search for relations (multi-polygon) stadiums."""
    query = f"""
    [out:json][timeout:15];
    (
      relation["leisure"="stadium"](around:{radius},{lat},{lon});
      relation["building"~"stadium"](around:{radius},{lat},{lon});
      relation["sport"~"baseball"](around:{radius},{lat},{lon});
    );
    (._;>;);
    out body;
    """
    result = overpass_query(query)
    if result is None:
        return []
    # Relations are complex — extract ways from them
    return extract_polygons_from_result(result)


def strategy_any_baseball(lat, lon, radius=1500):
    """Any way tagged sport=baseball."""
    query = f"""
    [out:json][timeout:15];
    (
      way["sport"~"baseball"](around:{radius},{lat},{lon});
    );
    (._;>;);
    out body;
    """
    return extract_polygons_from_result(overpass_query(query))


def strategy_landuse(lat, lon, radius=1500):
    """Search landuse=recreation_ground or leisure=sports_centre near coords."""
    query = f"""
    [out:json][timeout:15];
    (
      way["landuse"="recreation_ground"](around:{radius},{lat},{lon});
      way["leisure"="sports_centre"](around:{radius},{lat},{lon});
    );
    (._;>;);
    out body;
    """
    # For these, we need to check if any of the polygons look like baseball fields
    # (fan-shaped with a sharp corner). We'll accept them if sharpest angle < 100°
    result = overpass_query(query)
    if result is None:
        return []
    elements = result.get("elements", [])
    nodes = {e["id"]: (e["lat"], e["lon"]) for e in elements if e["type"] == "node"}
    ways = [e for e in elements if e["type"] == "way"]
    polygons = []
    for w in ways:
        node_ids = w.get("nodes", [])
        coords = [(nodes[nid][0], nodes[nid][1]) for nid in node_ids if nid in nodes]
        if len(coords) >= 4:
            polygons.append(coords)
    return polygons


STRATEGIES = [
    ("wider_1200m", lambda lat, lon: strategy_wider_radius(lat, lon, 1200)),
    ("very_wide_2500m", lambda lat, lon: strategy_very_wide(lat, lon, 2500)),
    ("relations_1500m", strategy_relations),
    ("any_baseball_1500m", strategy_any_baseball),
    ("landuse_1500m", strategy_landuse),
]


def process_stadium(row: dict) -> dict:
    cid = row["canonical_id"]
    lat = float(row["lat"])
    lon = float(row["lon"])
    name = row["venue_name"]

    print(f"\n  {cid}: {name} ({lat:.4f}, {lon:.4f})", file=sys.stderr)

    for strategy_name, strategy_fn in STRATEGIES:
        print(f"    trying {strategy_name}...", file=sys.stderr, end=" ")
        time.sleep(RATE_LIMIT_SEC)

        polygons = strategy_fn(lat, lon)
        if not polygons:
            print("no polygons", file=sys.stderr)
            continue

        # Use largest polygon
        largest = max(polygons, key=len)
        est_bearing = estimate_bearing_from_polygon(largest)
        if est_bearing is not None:
            # Sanity: check sharpest angle is < 120° (otherwise probably not a baseball diamond)
            n = len(largest)
            if largest[0] == largest[-1]:
                largest_clean = largest[:-1]
            else:
                largest_clean = largest
            nc = len(largest_clean)
            if nc >= 4:
                def angle_at(i):
                    p0 = largest_clean[(i - 1) % nc]
                    p1 = largest_clean[i]
                    p2 = largest_clean[(i + 1) % nc]
                    v1 = (p0[0] - p1[0], p0[1] - p1[1])
                    v2 = (p2[0] - p1[0], p2[1] - p1[1])
                    dot = v1[0] * v2[0] + v1[1] * v2[1]
                    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                    if mag1 < 1e-12 or mag2 < 1e-12:
                        return 180.0
                    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                    return math.degrees(math.acos(cos_angle))
                min_angle = min(angle_at(i) for i in range(nc))
                if min_angle > 120:
                    print(f"polygon found ({len(largest)} nodes) but min_angle={min_angle:.0f}° (not baseball-shaped), skipping", file=sys.stderr)
                    continue

            print(f"✓ bearing={est_bearing}° via {strategy_name} ({len(largest)} nodes)", file=sys.stderr)
            row["hp_bearing_deg"] = str(est_bearing)
            row["source"] = f"osm_{strategy_name}"
            return row
        else:
            print(f"polygon ({len(largest)} nodes) but bearing estimation failed", file=sys.stderr)

    print(f"    ✗ all strategies failed", file=sys.stderr)
    return row


def main():
    rows = []
    with open(STADIUM_CSV) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    # Only process stadiums still at default 67°
    to_process = [r for r in rows if r.get("hp_bearing_deg", "67") == "67"]
    print(f"Loaded {len(rows)} stadiums, {len(to_process)} still at default 67°", file=sys.stderr)

    updated = 0
    failed = []
    for row in to_process:
        old_source = row.get("source", "unknown")
        row = process_stadium(row)
        if row.get("source") != old_source:
            updated += 1
        else:
            failed.append(row["canonical_id"])

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Updated: {updated}/{len(to_process)}", file=sys.stderr)
    print(f"Still failed: {len(failed)}", file=sys.stderr)
    if failed:
        for f_id in failed:
            print(f"  {f_id}", file=sys.stderr)

    if updated > 0:
        with open(STADIUM_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {updated} updates to {STADIUM_CSV}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
