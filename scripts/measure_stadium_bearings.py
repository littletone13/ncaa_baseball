"""
Measure stadium home plate → center field bearings from OpenStreetMap polygon data.

Approach:
  1. For each stadium in stadium_orientations.csv, search OSM for nearby baseball
     stadium/pitch polygons using Overpass API.
  2. Download polygon geometry (node coordinates).
  3. Estimate HP→CF bearing from the polygon shape:
     - Baseball stadium polygons are typically asymmetric with HP at one corner
     - The "pointy" end of the polygon is home plate
     - Use PCA + geometric analysis to find the long axis
  4. Output updated CSV with verified bearings.

Usage:
  python3 scripts/measure_stadium_bearings.py
  python3 scripts/measure_stadium_bearings.py --canonical-id BSB_TEXAS_TECH
  python3 scripts/measure_stadium_bearings.py --batch 20  # first 20 stadiums
"""
from __future__ import annotations

import argparse
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
RATE_LIMIT_SEC = 4.0  # be polite to Overpass
MAX_RETRIES = 3
RETRY_BACKOFF = 8  # seconds between retries


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute compass bearing from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2)
         - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def overpass_query(query: str, timeout: int = 15) -> dict | None:
    """Run an Overpass API query with retry logic, return parsed JSON or None."""
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
                print(f"    Overpass {e.code}, retry {attempt+1}/{MAX_RETRIES} in {wait}s...",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"    Overpass error: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"    Overpass error: {e}", file=sys.stderr)
            return None
    return None


def find_baseball_polygons(lat: float, lon: float, radius: int = 800) -> list[list[tuple[float, float]]]:
    """
    Search OSM for baseball stadium/pitch polygons near (lat, lon).
    Returns list of polygons, each polygon is a list of (lat, lon) tuples.
    """
    query = f"""
    [out:json][timeout:12];
    (
      way["leisure"="stadium"]["sport"~"baseball"](around:{radius},{lat},{lon});
      way["leisure"="pitch"]["sport"~"baseball"](around:{radius},{lat},{lon});
      way["leisure"="stadium"](around:{radius},{lat},{lon});
      way["building"~"stadium"](around:{radius},{lat},{lon});
    );
    (._;>;);
    out body;
    """
    result = overpass_query(query)
    if result is None:
        return []

    elements = result.get("elements", [])
    nodes = {e["id"]: (e["lat"], e["lon"]) for e in elements if e["type"] == "node"}
    ways = [e for e in elements if e["type"] == "way"]

    polygons = []
    for w in ways:
        tags = w.get("tags", {})
        # Prefer baseball-tagged ways; skip tennis/soccer/etc
        sport = tags.get("sport", "").lower()
        if sport and "baseball" not in sport and "softball" not in sport:
            continue

        node_ids = w.get("nodes", [])
        coords = [(nodes[nid][0], nodes[nid][1]) for nid in node_ids if nid in nodes]
        if len(coords) >= 4:
            polygons.append(coords)

    return polygons


def estimate_bearing_from_polygon(coords: list[tuple[float, float]]) -> float | None:
    """
    Estimate HP→CF bearing from a stadium polygon.

    Strategy: Baseball stadiums have a characteristic shape — the "fan" opens out
    from home plate. Home plate is at the tightest/sharpest corner of the polygon.

    We find the vertex with the smallest interior angle (sharpest corner = HP),
    then compute the bearing from that vertex toward the centroid of the polygon
    (which is roughly toward center field).
    """
    n = len(coords)
    if n < 4:
        return None

    # Remove duplicate closing point if present
    if coords[0] == coords[-1]:
        coords = coords[:-1]
        n = len(coords)

    if n < 4:
        return None

    # Compute interior angles at each vertex
    def angle_at(i):
        """Interior angle at vertex i (in degrees)."""
        p0 = coords[(i - 1) % n]
        p1 = coords[i]
        p2 = coords[(i + 1) % n]
        # Vectors
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
    angles.sort()  # smallest angle first

    # The sharpest corner is likely home plate
    hp_angle, hp_idx = angles[0]
    hp_lat, hp_lon = coords[hp_idx]

    # Centroid (rough center field direction)
    cx = sum(c[0] for c in coords) / n
    cy = sum(c[1] for c in coords) / n

    # Bearing from HP to centroid
    b = bearing_deg(hp_lat, hp_lon, cx, cy)

    return round(b)


def wikidata_coords(venue_name: str) -> tuple[float | None, float | None]:
    """Try to get coordinates from Wikidata for a venue name."""
    search_url = (
        "https://www.wikidata.org/w/api.php?"
        f"action=wbsearchentities&search={urllib.parse.quote(venue_name)}"
        "&language=en&format=json&limit=3"
    )
    req = urllib.request.Request(search_url,
                                 headers={"User-Agent": "ncaa-baseball-model/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        for result in data.get("search", []):
            qid = result["id"]
            entity_url = (
                f"https://www.wikidata.org/w/api.php?"
                f"action=wbgetentities&ids={qid}&props=claims&format=json"
            )
            req2 = urllib.request.Request(entity_url,
                                          headers={"User-Agent": "ncaa-baseball-model/1.0"})
            with urllib.request.urlopen(req2, timeout=10) as resp2:
                edata = json.loads(resp2.read())
            claims = edata.get("entities", {}).get(qid, {}).get("claims", {})
            if "P625" in claims:
                coord = claims["P625"][0]["mainsnak"]["datavalue"]["value"]
                return coord["latitude"], coord["longitude"]
    except Exception:
        pass
    return None, None


def process_stadium(row: dict) -> dict:
    """
    Process a single stadium: look up polygon, estimate bearing.
    Returns updated row dict.
    """
    cid = row["canonical_id"]
    lat = float(row["lat"])
    lon = float(row["lon"])
    name = row["venue_name"]
    old_bearing = float(row.get("hp_bearing_deg", 67))
    old_source = row.get("source", "unknown")

    print(f"  {cid}: {name} ({lat:.4f}, {lon:.4f})", file=sys.stderr)

    # Try OSM polygon approach
    polygons = find_baseball_polygons(lat, lon)

    if polygons:
        # Use the largest polygon (most likely the main stadium)
        largest = max(polygons, key=len)
        est_bearing = estimate_bearing_from_polygon(largest)
        if est_bearing is not None:
            print(f"    → OSM polygon ({len(largest)} nodes): bearing ≈ {est_bearing}°",
                  file=sys.stderr)
            row["hp_bearing_deg"] = str(est_bearing)
            row["source"] = "osm_polygon_estimated"
            return row
        else:
            print(f"    → OSM polygon found but bearing estimation failed", file=sys.stderr)
    else:
        print(f"    → No OSM polygons found", file=sys.stderr)

    # Keep existing if we can't determine
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure stadium bearings from OSM data.")
    parser.add_argument("--canonical-id", type=str, help="Process single stadium")
    parser.add_argument("--batch", type=int, default=0, help="Process first N stadiums")
    parser.add_argument("--only-default", action="store_true",
                        help="Only process stadiums with default 67° bearing")
    parser.add_argument("--dry-run", action="store_true", help="Don't write output")
    args = parser.parse_args()

    # Load stadium data
    rows = []
    with open(STADIUM_CSV) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} stadiums", file=sys.stderr)

    # Filter
    to_process = []
    for row in rows:
        if args.canonical_id and row["canonical_id"] != args.canonical_id:
            continue
        if args.only_default and float(row.get("hp_bearing_deg", 67)) != 67:
            continue
        to_process.append(row)

    if args.batch > 0:
        to_process = to_process[:args.batch]

    print(f"Processing {len(to_process)} stadiums...", file=sys.stderr)

    updated = 0
    for row in to_process:
        old_bearing = row.get("hp_bearing_deg", "67")
        old_source = row.get("source", "unknown")

        row = process_stadium(row)

        if row.get("source") != old_source:
            updated += 1

        time.sleep(RATE_LIMIT_SEC)

    print(f"\nUpdated {updated}/{len(to_process)} stadiums", file=sys.stderr)

    if not args.dry_run and updated > 0:
        # Write back
        with open(STADIUM_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {STADIUM_CSV}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
