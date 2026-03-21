"""
build_player_registry.py — Build a unified player registry from all available sources.

Merges:
  1. Sidearm rosters (7,718 players from 240 teams, full B/T data)
  2. D1B rotation data (255 pitchers, throws only)
  3. Pitcher table (12,809 pitchers, throws for ~3,177)
  4. D1B batting leaderboards (2,357 batters, stats but no B/T)
  5. D1B pitching leaderboards (~670 pitchers, stats but no B/T)
  6. ESPN boxscores (batters and pitchers from games)
  7. NCAA API boxscores (pitcher appearances)

Output:
  data/processed/player_registry.csv — One row per unique (canonical_id, player_name)
  Columns:
    canonical_id, player_name, position, bats, throws, team_name, conference,
    is_pitcher, is_batter, fip, era, wrc_plus, pitcher_idx, season,
    source (comma-separated list of data sources)

Usage:
  python3 scripts/build_player_registry.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _norm_name(name: str) -> str:
    """Normalize a player name for matching."""
    name = name.strip()
    # Normalize unicode apostrophes
    name = name.replace("\u2019", "'").replace("\u2018", "'")
    # Remove position prefix patterns like "John - P Smith"
    name = re.sub(r"\s*-\s*P\s+", " ", name)
    # Remove extra whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _norm_name_lower(name: str) -> str:
    return _norm_name(name).lower()


def build_player_registry(
    sidearm_csv: Path = Path("data/processed/sidearm_rosters.csv"),
    ncaa_rosters_csv: Path = Path("data/processed/ncaa_rosters.csv"),
    pitcher_table_csv: Path = Path("data/processed/pitcher_table.csv"),
    d1b_rotations_csv: Path = Path("data/processed/d1baseball_rotations.csv"),
    d1b_root: Path = Path("data/raw/d1baseball"),
    d1b_crosswalk_csv: Path = Path("data/registries/d1baseball_crosswalk.csv"),
    canonical_csv: Path = Path("data/registries/canonical_teams_2026.csv"),
    appearances_csv: Path = Path("data/processed/pitcher_appearances.csv"),
    out_csv: Path = Path("data/processed/player_registry.csv"),
) -> pd.DataFrame:
    """Build unified player registry."""

    # Load canonical team info
    canon = pd.read_csv(canonical_csv)
    cid_to_name = dict(zip(canon["canonical_id"], canon["team_name"]))
    cid_to_conf = dict(zip(canon["canonical_id"], canon["conference"]))

    # D1B crosswalk
    d1b_to_cid = {}
    if d1b_crosswalk_csv.exists():
        xw = pd.read_csv(d1b_crosswalk_csv)
        for _, r in xw.iterrows():
            d1b_name = str(r["d1baseball_name"]).strip().replace("\u2019", "'")
            d1b_to_cid[d1b_name.lower()] = str(r["canonical_id"]).strip()

    # ── Master registry: (canonical_id, norm_name_lower) → record ──
    registry: dict[tuple[str, str], dict] = {}

    def _upsert(cid: str, name: str, **fields):
        key = (cid, _norm_name_lower(name))
        if key not in registry:
            registry[key] = {
                "canonical_id": cid,
                "player_name": _norm_name(name),
                "position": "",
                "bats": "",
                "throws": "",
                "is_pitcher": False,
                "is_batter": False,
                "fip": None,
                "era": None,
                "wrc_plus": None,
                "pitcher_idx": 0,
                "season": 2026,
                "sources": set(),
            }
        rec = registry[key]
        # Update non-empty fields (don't overwrite with blanks)
        for k, v in fields.items():
            if k == "sources":
                rec["sources"].update(v) if isinstance(v, set) else rec["sources"].add(v)
            elif k in ("is_pitcher", "is_batter"):
                rec[k] = rec[k] or v
            elif v is not None and v != "" and v != 0:
                # Keep first non-empty value (prioritize earlier sources)
                if rec.get(k) in (None, "", 0, np.nan):
                    rec[k] = v

    # ── 1. Sidearm rosters (highest priority for B/T) ──
    print("Loading sidearm rosters...", file=sys.stderr)
    if sidearm_csv.exists():
        sr = pd.read_csv(sidearm_csv, dtype=str)
        for _, r in sr.iterrows():
            cid = str(r.get("canonical_id", "")).strip()
            name = str(r.get("player_name", "")).strip()
            if not cid or not name:
                continue
            pos = str(r.get("position", "")).strip()
            is_p = "P" in pos.upper() if pos else False
            is_b = not is_p  # Non-pitchers are batters
            _upsert(
                cid, name,
                position=pos,
                bats=str(r.get("bats", "")).strip(),
                throws=str(r.get("throws", "")).strip(),
                is_pitcher=is_p,
                is_batter=is_b,
                sources="sidearm",
            )
        print(f"  {len(sr)} players from sidearm", file=sys.stderr)

    # ── 1b. NCAA roster scrapes (fills gaps from teams Sidearm missed) ──
    print("Loading NCAA roster scrapes...", file=sys.stderr)
    if ncaa_rosters_csv.exists():
        nr = pd.read_csv(ncaa_rosters_csv, dtype=str)
        for _, r in nr.iterrows():
            cid = str(r.get("canonical_id", "")).strip()
            name = str(r.get("player_name", "")).strip()
            if not cid or not name:
                continue
            pos = str(r.get("position", "")).strip()
            is_p = bool(re.search(r'\b(P|Pitcher|RHP|LHP|RHSP|LHSP)\b', pos, re.IGNORECASE)) if pos else False
            is_b = not is_p
            _upsert(
                cid, name,
                position=pos,
                bats=str(r.get("bats", "")).strip(),
                throws=str(r.get("throws", "")).strip(),
                is_pitcher=is_p,
                is_batter=is_b,
                sources="ncaa_roster",
            )
        print(f"  {len(nr)} players from NCAA roster scrapes", file=sys.stderr)

    # ── 2. D1B rotation data ──
    print("Loading D1B rotations...", file=sys.stderr)
    if d1b_rotations_csv.exists():
        rot = pd.read_csv(d1b_rotations_csv, dtype=str)
        for _, r in rot.iterrows():
            cid = str(r.get("canonical_id", "")).strip()
            name = str(r.get("pitcher_name", "")).strip()
            if not cid or not name:
                continue
            hand_raw = str(r.get("hand", "")).strip().upper()
            throws = "L" if hand_raw == "LHP" else "R" if hand_raw == "RHP" else ""
            _upsert(
                cid, name,
                throws=throws,
                is_pitcher=True,
                sources="d1b_rotation",
            )
        print(f"  {len(rot)} pitcher entries from D1B rotations", file=sys.stderr)

    # ── 3. Pitcher table (for pitcher_idx, FIP, ERA, throws) ──
    print("Loading pitcher table...", file=sys.stderr)
    if pitcher_table_csv.exists():
        pt = pd.read_csv(pitcher_table_csv)
        for _, r in pt.iterrows():
            cid = str(r.get("team_canonical_id", "")).strip()
            name = str(r.get("pitcher_name", "")).strip()
            if not cid or not name:
                continue
            throws = str(r.get("throws", "")).strip()
            if throws not in ("L", "R"):
                throws = ""
            pidx = int(r["pitcher_idx"]) if pd.notna(r["pitcher_idx"]) else 0
            fip = float(r["fip"]) if pd.notna(r.get("fip")) else None
            era = float(r["season_era"]) if pd.notna(r.get("season_era")) else None
            _upsert(
                cid, name,
                throws=throws,
                is_pitcher=True,
                pitcher_idx=pidx,
                fip=fip,
                era=era,
                sources="pitcher_table",
            )
        print(f"  {len(pt)} pitchers from pitcher table", file=sys.stderr)

    # ── 4. D1B batting leaderboard ──
    print("Loading D1B batting stats...", file=sys.stderr)
    bat_path = d1b_root / "batting_advanced.tsv"
    if bat_path.exists():
        bat = pd.read_csv(bat_path, sep="\t")
        for _, r in bat.iterrows():
            d1b_team = str(r.get("Team", "")).strip().replace("\u2019", "'")
            cid = d1b_to_cid.get(d1b_team.lower(), "")
            name = str(r.get("Player", "")).strip()
            if not cid or not name:
                continue
            wrc = float(r["wRC+"]) if pd.notna(r.get("wRC+")) else None
            _upsert(
                cid, name,
                is_batter=True,
                wrc_plus=wrc,
                sources="d1b_batting",
            )
        print(f"  {len(bat)} batters from D1B", file=sys.stderr)

    # ── 5. D1B pitching leaderboard ──
    print("Loading D1B pitching stats...", file=sys.stderr)
    pitch_path = d1b_root / "pitching_advanced.tsv"
    if pitch_path.exists():
        pitch = pd.read_csv(pitch_path, sep="\t")
        for _, r in pitch.iterrows():
            d1b_team = str(r.get("Team", "")).strip().replace("\u2019", "'")
            cid = d1b_to_cid.get(d1b_team.lower(), "")
            name = str(r.get("Player", "")).strip()
            if not cid or not name:
                continue
            fip = float(r["FIP"]) if pd.notna(r.get("FIP")) else None
            _upsert(
                cid, name,
                is_pitcher=True,
                fip=fip,
                sources="d1b_pitching",
            )
        print(f"  {len(pitch)} pitchers from D1B pitching", file=sys.stderr)

    # ── 6. D1B batting standard (for position) ──
    bat_std_path = d1b_root / "batting_standard.tsv"
    if bat_std_path.exists():
        bat_std = pd.read_csv(bat_std_path, sep="\t")
        for _, r in bat_std.iterrows():
            d1b_team = str(r.get("Team", "")).strip().replace("\u2019", "'")
            cid = d1b_to_cid.get(d1b_team.lower(), "")
            name = str(r.get("Player", "")).strip()
            pos = str(r.get("POS", "")).strip()
            if not cid or not name:
                continue
            is_p = "P" in pos.upper() if pos else False
            _upsert(
                cid, name,
                position=pos,
                is_pitcher=is_p,
                is_batter=True,
                sources="d1b_batting_std",
            )

    # ── 7. NCAA pitcher appearances ──
    print("Loading NCAA pitcher appearances...", file=sys.stderr)
    if appearances_csv.exists():
        app = pd.read_csv(appearances_csv, dtype=str, nrows=None)
        for _, r in app.iterrows():
            cid = str(r.get("canonical_id", "")).strip()
            name = str(r.get("pitcher_name", "")).strip()
            if not cid or not name:
                continue
            _upsert(
                cid, name,
                is_pitcher=True,
                sources="ncaa_appearances",
            )
        print(f"  {len(app)} appearance records", file=sys.stderr)

    # ── Build DataFrame ──
    rows = []
    for key, rec in registry.items():
        rec["sources"] = ",".join(sorted(rec["sources"]))
        rec["team_name"] = cid_to_name.get(rec["canonical_id"], "")
        rec["conference"] = cid_to_conf.get(rec["canonical_id"], "")
        rows.append(rec)

    df = pd.DataFrame(rows)

    # Sort by team, then name
    df = df.sort_values(["canonical_id", "player_name"]).reset_index(drop=True)

    # Column order
    col_order = [
        "canonical_id", "team_name", "conference", "player_name", "position",
        "bats", "throws", "is_pitcher", "is_batter",
        "pitcher_idx", "fip", "era", "wrc_plus", "season", "sources",
    ]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    # Write
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Player Registry: {len(df)} unique players", file=sys.stderr)
    print(f"  Teams: {df['canonical_id'].nunique()}", file=sys.stderr)
    print(f"  Pitchers: {df['is_pitcher'].sum()}", file=sys.stderr)
    print(f"  Batters: {df['is_batter'].sum()}", file=sys.stderr)
    print(f"  With throws: {(df['throws'].isin(['L','R'])).sum()} ({(df['throws'].isin(['L','R'])).sum()/len(df)*100:.1f}%)", file=sys.stderr)
    print(f"  With bats: {(df['bats'].isin(['L','R','S','B'])).sum()} ({(df['bats'].isin(['L','R','S','B'])).sum()/len(df)*100:.1f}%)", file=sys.stderr)
    print(f"  With FIP: {df['fip'].notna().sum()}", file=sys.stderr)
    print(f"  With wRC+: {df['wrc_plus'].notna().sum()}", file=sys.stderr)
    print(f"  With pitcher_idx > 0: {(df['pitcher_idx'] > 0).sum()}", file=sys.stderr)
    print(f"  → {out_csv}", file=sys.stderr)

    # Per-conference summary
    print(f"\n  Conference coverage (throws):", file=sys.stderr)
    conf_stats = df.groupby("conference").agg(
        n_players=("player_name", "count"),
        n_throws=("throws", lambda x: x.isin(["L", "R"]).sum()),
    )
    conf_stats["pct"] = (conf_stats["n_throws"] / conf_stats["n_players"] * 100).round(1)
    for _, r in conf_stats.sort_values("pct").iterrows():
        print(f"    {r.name:20s} {int(r['n_throws']):>4}/{int(r['n_players']):>4} ({r['pct']:5.1f}%)", file=sys.stderr)

    return df


if __name__ == "__main__":
    build_player_registry()
