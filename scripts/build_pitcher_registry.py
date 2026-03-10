"""
Build unified pitcher registry from ESPN and NCAA data sources.

Creates:
  1. pitcher_registry.csv  — all unique pitchers with unified IDs
  2. pitcher_appearances.csv — per-game pitcher appearances with IP, stats

Pitcher IDs:
  - ESPN pitchers: "ESPN_{espn_id}"  (e.g., "ESPN_67525")
  - NCAA pitchers: "NCAA_{normalized_name}__{canonical_team_id}"
    (e.g., "NCAA_michael_barnett__BSB_UCLA")

Handles name normalization for NCAA pitchers:
  - "First Last" -> "first_last"
  - "F. Last" -> "f_last"
  - "LAST" -> "last" (single-name entries)
  - Strips Jr./Sr./III etc.

Usage:
  python3 scripts/build_pitcher_registry.py
  python3 scripts/build_pitcher_registry.py --out-dir data/processed
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from ncaa_baseball.phase1 import (
    build_odds_name_to_canonical,
    load_canonical_teams,
    resolve_odds_teams,
)


# ──────────────────────────────────────────────────────────────────────────────
# Name normalization
# ──────────────────────────────────────────────────────────────────────────────

SUFFIX_RE = re.compile(
    r"\s+(Jr\.?|Sr\.?|III|II|IV|V)\s*$", re.IGNORECASE,
)

def normalize_pitcher_name(name: str) -> str:
    """Normalize pitcher name to lowercase key form.

    Examples:
      "Michael Barnett" -> "michael_barnett"
      "F. Last"         -> "f_last"
      "JONES"           -> "jones"
      "De La Cruz Jr."  -> "de_la_cruz"
    """
    name = name.strip()
    if not name:
        return ""
    # Remove accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    # Remove suffixes
    name = SUFFIX_RE.sub("", name).strip()
    # Remove periods from abbreviations
    name = name.replace(".", "")
    # Lowercase, collapse whitespace, replace spaces with underscores
    name = re.sub(r"\s+", "_", name.lower().strip())
    # Remove non-alphanumeric (keep underscores)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name


def make_ncaa_pitcher_id(name: str, team_canonical_id: str) -> str:
    """Build a unified NCAA pitcher ID from name + team."""
    norm = normalize_pitcher_name(name)
    if not norm:
        return ""
    team_clean = team_canonical_id.replace(" ", "_")
    return f"NCAA_{norm}__{team_clean}"


# ──────────────────────────────────────────────────────────────────────────────
# ESPN pitcher loading
# ──────────────────────────────────────────────────────────────────────────────

def load_espn_pitchers(
    espn_dir: Path,
    seasons: list[str],
) -> tuple[dict[str, dict], list[dict]]:
    """Load ESPN pitcher IDs from game starters.

    Returns:
      pitcher_info: {espn_id: {name, teams_seen}}
      appearances: list of {game_id, pitcher_id, team, role, ip, ...}
    """
    pitcher_info: dict[str, dict] = {}
    appearances: list[dict] = []

    for season in seasons:
        path = espn_dir / f"games_{season}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    g = json.loads(line.strip())
                except (json.JSONDecodeError, ValueError):
                    continue

                event_id = g.get("event_id") or g.get("id") or ""
                game_date = (g.get("date") or "")[:10]
                home_name = (g.get("home_team") or {}).get("name", "")
                away_name = (g.get("away_team") or {}).get("name", "")

                starters = g.get("starters") or {}
                for side_key, team_name in [
                    ("home_pitcher", home_name),
                    ("away_pitcher", away_name),
                ]:
                    p = starters.get(side_key)
                    if not p:
                        continue
                    espn_id = str(p.get("espn_id") or p.get("id") or "")
                    name = str(p.get("name") or "").strip()
                    if not espn_id:
                        continue
                    pid = f"ESPN_{espn_id}"
                    if pid not in pitcher_info:
                        pitcher_info[pid] = {"name": name, "teams_seen": set()}
                    pitcher_info[pid]["teams_seen"].add(team_name)
                    appearances.append({
                        "game_id": event_id,
                        "game_date": game_date,
                        "pitcher_id": pid,
                        "pitcher_name": name,
                        "team_name": team_name,
                        "role": "starter",
                        "side": "home" if side_key == "home_pitcher" else "away",
                        "ip": None,  # ESPN starters don't have IP in game data
                        "source": "espn",
                    })
    return pitcher_info, appearances


# ──────────────────────────────────────────────────────────────────────────────
# NCAA pitcher loading
# ──────────────────────────────────────────────────────────────────────────────

def load_ncaa_pitchers(
    ncaa_path: Path,
    canonical: pd.DataFrame,
    name_to_canonical: dict,
) -> tuple[dict[str, dict], list[dict]]:
    """Load ALL pitchers (starters + relievers) from NCAA boxscores.

    Returns:
      pitcher_info: {ncaa_pitcher_id: {name, team_canonical_id, ...}}
      appearances: list of {game_id, pitcher_id, team, role, ip, stats...}
    """
    pitcher_info: dict[str, dict] = {}
    appearances: list[dict] = []

    # Build team name -> canonical_id lookup
    name_map: dict[str, str] = {}
    for _, row in canonical.iterrows():
        tname = (row.get("team_name") or "").strip()
        cid = (row.get("canonical_id") or "").strip()
        if tname and cid:
            name_map[tname.lower()] = cid

    if not ncaa_path.exists():
        return pitcher_info, appearances

    with ncaa_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                g = json.loads(line.strip())
            except (json.JSONDecodeError, ValueError):
                continue

            game_id = str(g.get("game_id", ""))
            game_date = str(g.get("date", ""))[:10]
            pitching = g.get("pitching") or {}

            for side in ["home", "away"]:
                team_name = str(g.get(f"{side}_team", "")).strip()
                # Resolve team canonical ID
                team_cid = name_map.get(team_name.lower(), "")
                if not team_cid:
                    h_t, a_t = resolve_odds_teams(
                        team_name, "", canonical, name_to_canonical,
                    )
                    if h_t:
                        team_cid = h_t[0]

                pitchers = pitching.get(side, [])
                for p in pitchers:
                    name = str(p.get("name", "")).strip()
                    if not name:
                        continue

                    pid = make_ncaa_pitcher_id(name, team_cid or team_name)
                    if not pid:
                        continue

                    is_starter = bool(p.get("starter", False))
                    ip = p.get("ip", 0)

                    if pid not in pitcher_info:
                        pitcher_info[pid] = {
                            "name": name,
                            "team_canonical_id": team_cid,
                            "team_name": team_name,
                        }

                    appearances.append({
                        "game_id": f"NCAA_{game_id}",
                        "game_date": game_date,
                        "pitcher_id": pid,
                        "pitcher_name": name,
                        "team_name": team_name,
                        "team_canonical_id": team_cid,
                        "role": "starter" if is_starter else "reliever",
                        "side": side,
                        "ip": float(ip) if ip else 0.0,
                        "ip_raw": str(p.get("ip_raw", "")),
                        "h": int(p.get("h", 0) or 0),
                        "r": int(p.get("r", 0) or 0),
                        "er": int(p.get("er", 0) or 0),
                        "bb": int(p.get("bb", 0) or 0),
                        "k": int(p.get("k", 0) or 0),
                        "bf": int(p.get("bf", 0) or 0),
                        "source": "ncaa",
                    })

    return pitcher_info, appearances


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build unified pitcher registry from ESPN + NCAA data.",
    )
    parser.add_argument("--espn-dir", type=Path, default=Path("data/raw/espn"))
    parser.add_argument("--ncaa-boxscores", type=Path, default=Path("data/raw/ncaa/boxscores_2026.jsonl"))
    parser.add_argument("--canonical", type=Path, default=Path("data/registries/canonical_teams_2026.csv"))
    parser.add_argument("--seasons", type=str, default="2024,2025,2026")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    canonical = load_canonical_teams(args.canonical)
    name_to_canonical = build_odds_name_to_canonical(canonical)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]

    # Load ESPN pitchers
    print("Loading ESPN pitcher data...")
    espn_info, espn_appearances = load_espn_pitchers(args.espn_dir, seasons)
    print(f"  ESPN: {len(espn_info)} unique pitchers, {len(espn_appearances)} appearances")

    # Load NCAA pitchers (starters + relievers)
    print("Loading NCAA pitcher data (starters + relievers)...")
    ncaa_info, ncaa_appearances = load_ncaa_pitchers(
        args.ncaa_boxscores, canonical, name_to_canonical,
    )
    print(f"  NCAA: {len(ncaa_info)} unique pitchers, {len(ncaa_appearances)} appearances")

    # Count by role
    ncaa_starters = sum(1 for a in ncaa_appearances if a["role"] == "starter")
    ncaa_relievers = sum(1 for a in ncaa_appearances if a["role"] == "reliever")
    print(f"    Starter appearances: {ncaa_starters}")
    print(f"    Reliever appearances: {ncaa_relievers}")

    # Combine into registry
    all_info = {}
    all_info.update(espn_info)
    all_info.update(ncaa_info)

    all_appearances = espn_appearances + ncaa_appearances

    # Build pitcher registry with sequential indices
    # idx 0 = unknown (reserved)
    registry_rows = [{"pitcher_id": "unknown", "pitcher_idx": 0,
                       "pitcher_name": "unknown", "source": "reserved"}]
    for i, (pid, info) in enumerate(sorted(all_info.items()), start=1):
        name = info.get("name", "")
        source = "espn" if pid.startswith("ESPN_") else "ncaa"
        team = info.get("team_canonical_id", "") or ""
        teams_seen = info.get("teams_seen", set())
        if teams_seen:
            team = ", ".join(sorted(teams_seen))
        registry_rows.append({
            "pitcher_id": pid,
            "pitcher_idx": i,
            "pitcher_name": name,
            "team": team,
            "source": source,
        })

    registry_df = pd.DataFrame(registry_rows)

    # Build pitcher_index.csv mapping pitcher_id -> pitcher_idx
    # Stores BOTH unified ID ("ESPN_67525") and bare ESPN ID ("67525")
    # for backward compatibility with old and new run_events formats.
    index_rows = [{"pitcher_espn_id": "unknown", "pitcher_idx": 0}]
    for _, r in registry_df.iterrows():
        pid = r["pitcher_id"]
        if pid == "unknown":
            continue
        # Always store the full unified ID
        index_rows.append({"pitcher_espn_id": pid, "pitcher_idx": r["pitcher_idx"]})
        # For ESPN pitchers, also store the bare numeric ID for backward compat
        if pid.startswith("ESPN_"):
            espn_id = pid.replace("ESPN_", "")
            index_rows.append({"pitcher_espn_id": espn_id, "pitcher_idx": r["pitcher_idx"]})
    index_df = pd.DataFrame(index_rows)

    # Save outputs
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reg_path = args.out_dir / "pitcher_registry.csv"
    registry_df.to_csv(reg_path, index=False)

    idx_path = args.out_dir / "run_event_pitcher_index.csv"
    index_df.to_csv(idx_path, index=False)

    app_path = args.out_dir / "pitcher_appearances.csv"
    app_df = pd.DataFrame(all_appearances)
    app_df.to_csv(app_path, index=False)

    # Summary
    n_espn = sum(1 for _, r in registry_df.iterrows() if r["source"] == "espn")
    n_ncaa = sum(1 for _, r in registry_df.iterrows() if r["source"] == "ncaa")
    print(f"\n=== PITCHER REGISTRY ===")
    print(f"Total pitchers: {len(registry_df) - 1} (+ 1 unknown)")
    print(f"  ESPN: {n_espn}")
    print(f"  NCAA: {n_ncaa}")
    print(f"Total appearances: {len(all_appearances)}")
    print(f"  ESPN starter appearances: {len(espn_appearances)}")
    print(f"  NCAA starter appearances: {ncaa_starters}")
    print(f"  NCAA reliever appearances: {ncaa_relievers}")
    print(f"\nOutputs:")
    print(f"  Registry: {reg_path}")
    print(f"  Pitcher index: {idx_path}")
    print(f"  Appearances: {app_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
