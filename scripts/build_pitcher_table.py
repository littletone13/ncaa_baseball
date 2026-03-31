"""
Build a unified pitcher lookup table from all source files.

Merges:
  1. pitcher_appearances.csv          → base roster, n_appearances, IP/ER, last_appearance
  2. run_event_pitcher_index.csv      → Stan model index (pitcher_idx)
  3. pitching_advanced.tsv            → FIP, SIERA
  4. pitching_standard.tsv            → ERA for D1B-qualified pitchers
  5. pitching_batted_ball.tsv         → FB%
  6. d1baseball_rotations.csv         → handedness (RHP/LHP)
  7. d1baseball_crosswalk.csv         → D1B team name → canonical_id
  8. canonical_teams_2026.csv         → espn_name → canonical_id

Output: data/processed/pitcher_table.csv

Usage:
    python3 scripts/build_pitcher_table.py
    python3 scripts/build_pitcher_table.py --out data/processed/pitcher_table.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────

LEAGUE_AVG_FB = 38.9  # D1 average fly-ball % (used as denominator for fb_sensitivity)
ABILITY_STD_EST = 0.08  # Estimated std of pitcher_ability posterior (bumped from 0.05 for more differentiation)
ERA_SLOPE = 0.027       # Linear ERA→ability mapping slope (for rotation-page ERA)
ERA_INTERCEPT = -0.01
ERA_BASELINE = 5.2      # NCAA D1-scale baseline for ERA fallback mapping


# ── Normal inverse CDF approximation ──────────────────────────────────────────

def _norm_ppf(p: float) -> float:
    """Approximate normal inverse CDF (Abramowitz & Stegun 26.2.23)."""
    if p <= 0:
        return -4.0
    if p >= 1:
        return 4.0
    if p == 0.5:
        return 0.0
    if p > 0.5:
        return -_norm_ppf(1 - p)
    t = math.sqrt(-2 * math.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return -(t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t))


def _fip_pctile_to_ability(fip_val: float, fip_arr: np.ndarray) -> float:
    """Map FIP value → ability adjustment via percentile matching.

    LOW FIP = GOOD pitcher = NEGATIVE ability adj (suppresses opponent runs).
    """
    pctile = np.searchsorted(fip_arr, fip_val) / len(fip_arr)
    pctile = float(np.clip(pctile, 0.01, 0.99))
    z = _norm_ppf(pctile)
    return z * ABILITY_STD_EST


# ── String normalization ───────────────────────────────────────────────────────

def _norm_str(s: str) -> str:
    """Normalize curly apostrophes to straight, lowercase, strip."""
    return s.replace("\u2019", "'").replace("\u2018", "'").strip().lower()


def _norm_name(s: str) -> str:
    """Normalize a pitcher name for matching."""
    return _norm_str(s)


def _fb_sensitivity(fb_pct: float | None) -> float:
    """Convert FB% (as a raw number like 38.9) to sensitivity scalar."""
    if fb_pct is None or (isinstance(fb_pct, float) and math.isnan(fb_pct)):
        return 1.0
    return max(0.3, 0.3 + 0.7 * (fb_pct / LEAGUE_AVG_FB))


# ── Load helpers ───────────────────────────────────────────────────────────────

def load_d1b_crosswalk(path: Path) -> dict[str, str]:
    """Return dict of d1baseball_name (normalized) → canonical_id."""
    cw: dict[str, str] = {}
    if not path.exists():
        return cw
    with open(path) as f:
        for row in csv.DictReader(f):
            d1b = _norm_str(row.get("d1baseball_name", ""))
            cid = row.get("canonical_id", "").strip()
            if d1b and cid:
                cw[d1b] = cid
                # Also index with apostrophe variants
                alt = d1b.replace("'", "\u2019")
                if alt != d1b:
                    cw[alt] = cid
    return cw


def load_espn_to_canonical(canonical_path: Path) -> dict[str, str]:
    """Return dict: espn_name (lower) → canonical_id."""
    m: dict[str, str] = {}
    if not canonical_path.exists():
        return m
    df = pd.read_csv(canonical_path, dtype=str)
    for _, row in df.iterrows():
        espn = str(row.get("espn_name", "")).strip()
        cid = str(row.get("canonical_id", "")).strip()
        if espn and cid:
            m[espn.lower()] = cid
    return m


def load_pitcher_index(path: Path) -> dict[str, int]:
    """Return dict: pitcher_espn_id (str, numeric) → Stan model index.

    Indexes BOTH the raw key (e.g. 'ESPN_59632') and the stripped numeric
    form ('59632') so lookups work regardless of which format callers use.
    """
    idx: dict[str, int] = {}
    if not path.exists():
        return idx
    df = pd.read_csv(path, dtype=str)
    for _, row in df.iterrows():
        eid = str(row.get("pitcher_espn_id", "")).strip()
        try:
            pidx = int(row["pitcher_idx"])
        except (ValueError, TypeError):
            pidx = 0
        if eid and eid != "unknown":
            idx[eid] = pidx
            # Strip .0 suffix from float-formatted IDs (e.g. "8135.0" → "8135")
            if eid.endswith(".0"):
                idx[eid[:-2]] = pidx
            # Also index without ESPN_ prefix for numeric lookups
            if eid.startswith("ESPN_"):
                numeric = eid[5:]
                idx[numeric] = pidx
    return idx


def _parse_season_from_path(path: Path) -> int | None:
    parent = path.parent.name
    if parent.isdigit() and len(parent) == 4:
        return int(parent)
    return None


def _collect_season_files(root: Path, basename: str, explicit_path: Path | None = None) -> list[tuple[int | None, Path]]:
    """Collect multi-season files from root and root/YYYY/."""
    seen: set[Path] = set()
    out: list[tuple[int | None, Path]] = []

    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    candidates.append(root / basename)
    for p in sorted(root.glob(f"*/{basename}")):
        candidates.append(p)

    for p in candidates:
        if not p.exists():
            continue
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append((_parse_season_from_path(p), p))
    return out


def _season_weighted_value(records: list[tuple[int, float]], target_season: int, max_back: int = 3, decay: float = 0.65) -> float | None:
    """Recency-weighted value using records up to target season."""
    elig = [(s, v) for (s, v) in records if s <= target_season and s >= (target_season - max_back)]
    if not elig:
        # fallback: nearest season record in either direction
        if not records:
            return None
        nearest = sorted(records, key=lambda x: abs(x[0] - target_season))[0]
        return float(nearest[1])

    num = 0.0
    den = 0.0
    for s, v in elig:
        w = decay ** max(0, target_season - s)
        num += w * float(v)
        den += w
    if den <= 0:
        return None
    return num / den


# ── Main build function ────────────────────────────────────────────────────────

def build_pitcher_table(
    appearances_csv: Path,
    pitcher_index_csv: Path,
    pitching_advanced_tsv: Path,
    pitching_standard_tsv: Path,
    pitching_batted_ball_tsv: Path,
    rotations_csv: Path,
    d1b_crosswalk_csv: Path,
    canonical_csv: Path,
    out_csv: Path,
    d1b_root: Path,
    as_of_season: int | None = None,
) -> pd.DataFrame:

    # ── 1. Build crosswalk lookups ─────────────────────────────────────────
    print("Loading crosswalks...", file=sys.stderr)
    d1b_to_cid = load_d1b_crosswalk(d1b_crosswalk_csv)
    espn_to_cid = load_espn_to_canonical(canonical_csv)
    pitcher_idx_map = load_pitcher_index(pitcher_index_csv)

    print(f"  D1B crosswalk: {len(d1b_to_cid)} teams", file=sys.stderr)
    print(f"  ESPN name→cid: {len(espn_to_cid)} teams", file=sys.stderr)
    print(f"  Pitcher index: {len(pitcher_idx_map)} pitchers with ESPN IDs", file=sys.stderr)

    # ── 2. Load pitcher_appearances.csv ────────────────────────────────────
    print("Loading pitcher appearances...", file=sys.stderr)
    app = pd.read_csv(appearances_csv, dtype=str)
    # Backward compatibility: older extracts used pitcher_espn_id + starter
    # while downstream logic expects pitcher_id + role.
    if "pitcher_id" not in app.columns:
        pid_series = app["pitcher_espn_id"] if "pitcher_espn_id" in app.columns else pd.Series([""] * len(app))
        app["pitcher_id"] = pid_series.apply(
            lambda x: f"ESPN_{str(x).strip()}" if str(x).strip() else ""
        )
    if "role" not in app.columns:
        starter_series = app["starter"] if "starter" in app.columns else pd.Series([""] * len(app))
        app["role"] = starter_series.apply(
            lambda x: "starter" if str(x).strip().lower() in ("true", "1", "yes") else "reliever"
        )
    for col in ["ip", "er"]:
        app[col] = pd.to_numeric(app[col], errors="coerce")
    app["game_date"] = pd.to_datetime(app["game_date"], errors="coerce")
    app["season"] = app["game_date"].dt.year
    if as_of_season is None:
        if app["season"].dropna().empty:
            as_of_season = 2026
        else:
            as_of_season = int(app["season"].dropna().max())

    # Strip ESPN_ prefix from pitcher_id to get numeric ESPN id
    app["pitcher_espn_id"] = app["pitcher_id"].apply(
        lambda x: x.replace("ESPN_", "") if isinstance(x, str) and x.startswith("ESPN_") else ""
    )

    # For ESPN-sourced rows, resolve team_canonical_id from team_name via espn_name
    def resolve_cid_from_team_name(team_name: str) -> str:
        if not isinstance(team_name, str):
            return ""
        return espn_to_cid.get(team_name.strip().lower(), "")

    app["team_canonical_id"] = app.apply(
        lambda r: (
            r["team_canonical_id"]
            if pd.notna(r.get("team_canonical_id", "")) and str(r.get("team_canonical_id", "")).strip()
            else resolve_cid_from_team_name(str(r.get("team_name", "")))
        ),
        axis=1,
    )

    # Aggregate by (pitcher_espn_id, pitcher_name, team_canonical_id, season)
    # Use pitcher_id as the grouping key to handle NCAA_ format IDs
    grp_cols = ["pitcher_id", "pitcher_name", "team_canonical_id", "season"]
    agg = (
        app.groupby(grp_cols, dropna=False)
        .agg(
            n_appearances=("game_date", "count"),
            total_ip=("ip", "sum"),
            total_er=("er", "sum"),
            last_appearance=("game_date", "max"),
            role_raw=("role", lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else ""),
        )
        .reset_index()
    )

    # Map role to SP/RP
    def map_role(r: str) -> str:
        if not isinstance(r, str):
            return ""
        r = r.strip().lower()
        if r in ("starter", "sp"):
            return "SP"
        if r in ("reliever", "rp"):
            return "RP"
        return ""

    agg["role"] = agg["role_raw"].apply(map_role)

    # Season ERA from appearances (9*ER/IP)
    agg["season_ip"] = pd.to_numeric(agg["total_ip"], errors="coerce")
    agg["season_er"] = pd.to_numeric(agg["total_er"], errors="coerce")
    agg["season_era"] = agg.apply(
        lambda r: (r["season_er"] / r["season_ip"] * 9.0)
        if pd.notna(r["season_ip"]) and r["season_ip"] >= 5.0 and pd.notna(r["season_er"])
        else float("nan"),
        axis=1,
    )
    agg["last_appearance"] = agg["last_appearance"].dt.strftime("%Y-%m-%d")

    # Extract pitcher_espn_id (numeric) from pitcher_id column
    agg["pitcher_espn_id"] = agg["pitcher_id"].apply(
        lambda x: x.replace("ESPN_", "") if isinstance(x, str) and x.startswith("ESPN_") else ""
    )

    # Keep most recent season per pitcher (pitcher_id + team combination)
    # Sort by season desc, take max
    agg = agg.sort_values("season", ascending=False)
    agg = agg.drop_duplicates(subset=["pitcher_id", "team_canonical_id"], keep="first")

    # Add pitcher_idx from pitcher index map
    # The index has two ID formats:
    #   ESPN: "ESPN_65678" (with numeric-stripped keys too)
    #   NCAA: "NCAA_john_smith__BSB_UCLA" (name__team format from run_events_expanded)
    # Appearances have:
    #   pitcher_espn_id: "65678.0" (float) or ""
    #   pitcher_id: "NCAA_6541986_John_Smith" (different NCAA format)
    #   pitcher_name + team_canonical_id (can build NCAA format key)
    import unicodedata, re as re_mod

    _SUFFIX_RE_LOCAL = re_mod.compile(r"\s+(Jr\.?|Sr\.?|III|II|IV|V)\s*$", re_mod.IGNORECASE)

    def _make_ncaa_index_key(name: str, team_cid: str) -> str:
        """Build NCAA_name__team key matching run_events_expanded format exactly."""
        if not name or not team_cid:
            return ""
        # Must match _normalize_pitcher_name() in build_run_events_expanded.py exactly:
        n = name.strip()
        n = unicodedata.normalize("NFKD", n)
        n = "".join(c for c in n if not unicodedata.combining(c))
        n = _SUFFIX_RE_LOCAL.sub("", n).strip()
        n = n.replace(".", "")
        n = re_mod.sub(r"\s+", "_", n.lower().strip())
        n = re_mod.sub(r"[^a-z0-9_]", "", n)
        if not n:
            return ""
        team_clean = team_cid.replace(" ", "_")
        return f"NCAA_{n}__{team_clean}"

    def _lookup_pitcher_idx(row):
        eid = row.get("pitcher_espn_id", "")
        # Try ESPN numeric ID first
        if eid and str(eid) not in ("", "nan", "None"):
            s = str(eid)
            if s in pitcher_idx_map:
                return pitcher_idx_map[s]
            if s.endswith(".0"):
                s = s[:-2]
            if s in pitcher_idx_map:
                return pitcher_idx_map[s]
            if s.isdigit() and f"ESPN_{s}" in pitcher_idx_map:
                return pitcher_idx_map[f"ESPN_{s}"]
        # Try NCAA name__team format (full name)
        pname = str(row.get("pitcher_name", ""))
        tcid = str(row.get("team_canonical_id", ""))
        ncaa_key = _make_ncaa_index_key(pname, tcid)
        if ncaa_key and ncaa_key in pitcher_idx_map:
            return pitcher_idx_map[ncaa_key]
        # Try abbreviated forms: first_initial + last_name
        # The run_events backfill often uses "O. Kelly" while pitcher_table has "Owen Kelly"
        parts = pname.strip().split()
        if len(parts) >= 2:
            # Try: first initial + last name (e.g., "Owen Kelly" → "o_kelly")
            initial = parts[0][0].lower() if parts[0] else ""
            lastname = parts[-1]
            if initial:
                abbrev_key = _make_ncaa_index_key(f"{initial}. {lastname}", tcid)
                if abbrev_key and abbrev_key in pitcher_idx_map:
                    return pitcher_idx_map[abbrev_key]
            # Try: last name only (for single-name entries like "Ciampa")
            ln_key = _make_ncaa_index_key(lastname, tcid)
            if ln_key and ln_key in pitcher_idx_map:
                return pitcher_idx_map[ln_key]
        return 0

    agg["pitcher_idx"] = agg.apply(_lookup_pitcher_idx, axis=1)

    print(f"  {len(agg)} unique (pitcher, team) combinations", file=sys.stderr)
    print(f"  With pitcher_idx > 0: {(agg['pitcher_idx'] > 0).sum()}", file=sys.stderr)

    # ── 3. Load D1B advanced stats (FIP, SIERA) ────────────────────────────
    print("Loading D1B advanced stats...", file=sys.stderr)
    # Build FIP/SIERA distributions for percentile mapping
    all_fips: list[float] = []
    all_sieras: list[float] = []
    d1b_adv_data: list[dict] = []  # (cid, name, fip, siera)

    adv_files = _collect_season_files(d1b_root, "pitching_advanced.tsv", pitching_advanced_tsv)
    for season_hint, adv_path in adv_files:
        season_val = season_hint if season_hint is not None else as_of_season
        adv = pd.read_csv(adv_path, sep="\t", dtype=str)
        for _, r in adv.iterrows():
            pname = _norm_name(str(r.get("Player", "")))
            team = _norm_str(str(r.get("Team", "")))
            cid = d1b_to_cid.get(team, "")
            if not pname or not cid:
                continue
            try:
                fip = float(r["FIP"])
                all_fips.append(fip)
            except (ValueError, TypeError):
                fip = None
            try:
                siera = float(r["SIERA"])
                all_sieras.append(siera)
            except (ValueError, TypeError):
                siera = None
            d1b_adv_data.append({"cid": cid, "name": pname, "season": int(season_val), "fip": fip, "siera": siera})

    all_fips_arr = np.array(sorted(all_fips)) if all_fips else np.array([4.5])
    all_sieras_arr = np.array(sorted(all_sieras)) if all_sieras else all_fips_arr
    print(f"  FIP distribution: {len(all_fips)} pitchers, mean={all_fips_arr.mean():.2f}", file=sys.stderr)

    # Build (cid, normalized_name) → {fip, siera} dict
    d1b_adv_map: dict[tuple[str, str], list[dict]] = {}
    for row in d1b_adv_data:
        key = (row["cid"], row["name"])
        d1b_adv_map.setdefault(key, []).append({"season": int(row["season"]), "fip": row["fip"], "siera": row["siera"]})

    # ── 4. Load D1B standard stats (ERA for non-advanced pitchers) ─────────
    print("Loading D1B standard stats...", file=sys.stderr)
    d1b_std_map: dict[tuple[str, str], dict] = {}

    std_files = _collect_season_files(d1b_root, "pitching_standard.tsv", pitching_standard_tsv)
    for season_hint, std_path in std_files:
        season_val = season_hint if season_hint is not None else as_of_season
        std = pd.read_csv(std_path, sep="\t", dtype=str)
        for _, r in std.iterrows():
            pname = _norm_name(str(r.get("Player", "")))
            team = _norm_str(str(r.get("Team", "")))
            cid = d1b_to_cid.get(team, "")
            if not pname or not cid:
                continue
            try:
                era = float(r["ERA"])
                ip = float(r.get("IP", "0") or "0")
            except (ValueError, TypeError):
                continue
            d1b_std_map[(cid, pname)] = {"season": int(season_val), "era_d1b": era, "ip_d1b": ip}

    print(f"  D1B standard: {len(d1b_std_map)} entries", file=sys.stderr)

    # ── 5. Load FB% (batted ball) ──────────────────────────────────────────
    print("Loading batted ball data (FB%)...", file=sys.stderr)
    d1b_fb_map: dict[tuple[str, str], float] = {}  # (cid, name) → FB% as raw number

    bb_files = _collect_season_files(d1b_root, "pitching_batted_ball.tsv", pitching_batted_ball_tsv)
    for season_hint, bb_path in bb_files:
        season_val = season_hint if season_hint is not None else as_of_season
        bb = pd.read_csv(bb_path, sep="\t", dtype=str)
        for _, r in bb.iterrows():
            pname = _norm_name(str(r.get("Player", "")))
            team = _norm_str(str(r.get("Team", "")))
            cid = d1b_to_cid.get(team, "")
            if not pname or not cid:
                continue
            fb_str = str(r.get("FB%", "")).strip().rstrip("%")
            try:
                fb_pct = float(fb_str)
                d1b_fb_map[(cid, pname)] = {"season": int(season_val), "fb_pct": fb_pct}
            except (ValueError, TypeError):
                pass

    print(f"  FB% data: {len(d1b_fb_map)} pitchers", file=sys.stderr)

    # ── 6. Load handedness from d1baseball_rotations.csv ──────────────────
    print("Loading pitcher handedness...", file=sys.stderr)
    rotation_map: dict[tuple[str, str], dict] = {}  # (cid, norm_name) → {hand, era, ip}

    if rotations_csv.exists():
        rot = pd.read_csv(rotations_csv, dtype=str)
        for _, r in rot.iterrows():
            cid = str(r.get("canonical_id", "")).strip()
            pname = _norm_name(str(r.get("pitcher_name", "")))
            hand = str(r.get("hand", "")).strip()
            era_str = str(r.get("era", "")).strip()
            ip_str = str(r.get("ip", "")).strip()
            if not cid or not pname:
                continue
            throws = ""
            if hand == "RHP":
                throws = "R"
            elif hand == "LHP":
                throws = "L"
            try:
                era_rot = float(era_str)
            except (ValueError, TypeError):
                era_rot = None
            try:
                ip_rot = float(ip_str)
            except (ValueError, TypeError):
                ip_rot = None
            # Keep first (most recent) entry for each (cid, name)
            if (cid, pname) not in rotation_map:
                rotation_map[(cid, pname)] = {"throws": throws, "era_rot": era_rot, "ip_rot": ip_rot}

    print(f"  Rotation handedness: {len(rotation_map)} entries", file=sys.stderr)

    # ── 6b. Load handedness from sidearm rosters (fallback) ──────────────
    sidearm_csv = Path("data/processed/sidearm_rosters.csv")
    sidearm_map: dict[tuple[str, str], str] = {}  # (cid, norm_name) → throws
    sidearm_lastname_idx: dict[tuple[str, str], list[tuple[str, str]]] = {}  # (cid, lastname) → [(fullname, throws)]
    if sidearm_csv.exists():
        sr = pd.read_csv(sidearm_csv, dtype=str)
        sr_pitchers = sr[sr["position"].str.contains("P", na=False)]
        for _, r in sr_pitchers.iterrows():
            cid = str(r.get("canonical_id", "")).strip()
            pname = _norm_name(str(r.get("player_name", "")))
            throws = str(r.get("throws", "")).strip()
            if cid and pname and throws in ("L", "R"):
                if (cid, pname) not in sidearm_map:
                    sidearm_map[(cid, pname)] = throws
                # Also build last-name index for abbreviated name matching
                parts = pname.split()
                if parts:
                    lastname = parts[-1]
                    sidearm_lastname_idx.setdefault((cid, lastname), []).append((pname, throws))
        print(f"  Sidearm handedness: {len(sidearm_map)} pitcher entries from {sr_pitchers['canonical_id'].nunique()} teams", file=sys.stderr)
    else:
        print("  Sidearm rosters not found, skipping", file=sys.stderr)

    # ── 6c. Load handedness from player_registry.csv (broadest fallback) ─
    registry_csv = Path("data/processed/player_registry.csv")
    registry_throws_map: dict[tuple[str, str], str] = {}  # (cid, norm_name) → throws
    registry_lastname_idx: dict[tuple[str, str], list[tuple[str, str]]] = {}
    if registry_csv.exists():
        reg = pd.read_csv(registry_csv, dtype=str)
        reg_pitchers = reg[reg["is_pitcher"].astype(str).str.lower().isin(["true", "1"])]
        for _, r in reg_pitchers.iterrows():
            cid = str(r.get("canonical_id", "")).strip()
            pname = _norm_name(str(r.get("player_name", "")))
            throws = str(r.get("throws", "")).strip()
            if cid and pname and throws in ("L", "R"):
                if (cid, pname) not in registry_throws_map:
                    registry_throws_map[(cid, pname)] = throws
                parts = pname.split()
                if parts:
                    lastname = parts[-1]
                    registry_lastname_idx.setdefault((cid, lastname), []).append((pname, throws))
        print(f"  Registry handedness: {len(registry_throws_map)} pitcher entries", file=sys.stderr)
    else:
        print("  Player registry not found, skipping", file=sys.stderr)

    # ── 7. Build last-name index for D1B fuzzy matching ───────────────────
    # D1B pitchers have full names; appearances may use "J. Cheeseman" style.
    # Index by (cid, last_name) for fallback lookups.
    d1b_lastname_idx: dict[tuple[str, str], list[tuple[str, str]]] = {}
    # sources: adv, std, fb
    for key in set(list(d1b_adv_map.keys()) + list(d1b_std_map.keys()) + list(d1b_fb_map.keys())):
        cid, fullname = key
        parts = fullname.split()
        if parts:
            lastname = parts[-1]
            d1b_lastname_idx.setdefault((cid, lastname), []).append((fullname, cid))
        # Also index with hyphen/space collapsed (e.g., "van kempen" → "vankempen")
        collapsed = fullname.replace(" ", "").replace("-", "")
        if collapsed != fullname:
            d1b_lastname_idx.setdefault((cid, collapsed), []).append((fullname, cid))

    def _d1b_lookup_name(cid: str, raw_name: str) -> str | None:
        """Resolve a (possibly abbreviated) pitcher name to D1B full name."""
        norm = _norm_name(raw_name)
        # Exact match first
        if (cid, norm) in d1b_adv_map or (cid, norm) in d1b_std_map or (cid, norm) in d1b_fb_map:
            return norm
        # Last-name lookup for abbreviated names (e.g., "J. Cheeseman")
        parts = norm.split()
        if not parts:
            return None
        lastname = parts[-1]
        candidates = d1b_lastname_idx.get((cid, lastname), [])
        if len(candidates) == 1:
            return candidates[0][0]
        if len(candidates) > 1:
            # Multiple candidates — try first-initial match
            if len(parts) >= 2:
                first_part = parts[0].rstrip(".")
                for fullname, _ in candidates:
                    fname_parts = fullname.split()
                    if fname_parts and fname_parts[0].lower().startswith(first_part):
                        return fullname
            # Return first match as fallback
            return candidates[0][0]
        return None

    def _rotation_lookup_name(cid: str, raw_name: str) -> str | None:
        """Resolve pitcher name to rotation map key."""
        norm = _norm_name(raw_name)
        if (cid, norm) in rotation_map:
            return norm
        # Last-name fallback
        parts = norm.split()
        if not parts:
            return None
        lastname = parts[-1]
        # Search all rotation_map keys for matching (cid, lastname)
        matches = [(k, v) for k, v in rotation_map.items() if k[0] == cid and k[1].split()[-1:] == [lastname]]
        if len(matches) == 1:
            return matches[0][0][1]
        return None

    # ── 8. Assemble final rows ─────────────────────────────────────────────
    print("Assembling pitcher table...", file=sys.stderr)
    rows = []
    for _, r in agg.iterrows():
        pitcher_id = str(r.get("pitcher_id", "")).strip()
        pitcher_name = str(r.get("pitcher_name", "")).strip()
        cid = str(r.get("team_canonical_id", "")).strip()
        season = int(r["season"]) if pd.notna(r.get("season")) else 0
        pitcher_espn_id = str(r.get("pitcher_espn_id", "")).strip()
        pitcher_idx = int(r.get("pitcher_idx", 0))
        role = str(r.get("role", "")).strip()
        n_app = int(r.get("n_appearances", 0))
        last_app = str(r.get("last_appearance", "")).strip()
        season_ip = float(r["season_ip"]) if pd.notna(r.get("season_ip")) else float("nan")
        season_era_app = float(r["season_era"]) if pd.notna(r.get("season_era")) else float("nan")

        # ── Resolve D1B names ──────────────────────────────────────────────
        d1b_name = _d1b_lookup_name(cid, pitcher_name) if cid else None
        rot_name = _rotation_lookup_name(cid, pitcher_name) if cid else None

        # ── FIP, SIERA ─────────────────────────────────────────────────────
        fip = None
        siera = None
        if d1b_name and (cid, d1b_name) in d1b_adv_map:
            adv_entries = d1b_adv_map[(cid, d1b_name)]
            fip_records = [(int(x["season"]), float(x["fip"])) for x in adv_entries if x.get("fip") is not None]
            siera_records = [(int(x["season"]), float(x["siera"])) for x in adv_entries if x.get("siera") is not None]
            fip = _season_weighted_value(fip_records, season)
            siera = _season_weighted_value(siera_records, season)

        # ── D1B standard ERA ───────────────────────────────────────────────
        era_d1b = None
        if d1b_name and (cid, d1b_name) in d1b_std_map:
            std_entry = d1b_std_map[(cid, d1b_name)]
            if std_entry.get("ip_d1b", 0) >= 5.0:
                era_d1b = std_entry.get("era_d1b")

        # ── FB% ────────────────────────────────────────────────────────────
        fb_pct = None
        if d1b_name and (cid, d1b_name) in d1b_fb_map:
            fb_entry = d1b_fb_map[(cid, d1b_name)]
            fb_pct = float(fb_entry["fb_pct"])

        fb_sens = _fb_sensitivity(fb_pct)

        # ── Handedness ─────────────────────────────────────────────────────
        throws = ""
        # Try rotation map first (best source)
        if rot_name and (cid, rot_name) in rotation_map:
            throws = rotation_map[(cid, rot_name)].get("throws", "")
        # Also check direct norm name match
        if not throws:
            norm_pname = _norm_name(pitcher_name)
            if (cid, norm_pname) in rotation_map:
                throws = rotation_map[(cid, norm_pname)].get("throws", "")
        # Fallback to sidearm roster data
        if not throws:
            norm_pname = _norm_name(pitcher_name)
            if (cid, norm_pname) in sidearm_map:
                throws = sidearm_map[(cid, norm_pname)]
            else:
                # Try last-name matching for abbreviated names (e.g., "J. Smith")
                parts = norm_pname.split()
                if parts:
                    lastname = parts[-1]
                    candidates = sidearm_lastname_idx.get((cid, lastname), [])
                    if len(candidates) == 1:
                        throws = candidates[0][1]
                    elif len(candidates) > 1 and len(parts) >= 2:
                        first_part = parts[0].rstrip(".")
                        for fullname, t in candidates:
                            fname_parts = fullname.split()
                            if fname_parts and fname_parts[0].startswith(first_part):
                                throws = t
                                break

        # Fallback to player registry (broadest source — 7K+ with throws)
        if not throws:
            norm_pname = _norm_name(pitcher_name)
            if (cid, norm_pname) in registry_throws_map:
                throws = registry_throws_map[(cid, norm_pname)]
            else:
                parts = norm_pname.split()
                if parts:
                    lastname = parts[-1]
                    candidates = registry_lastname_idx.get((cid, lastname), [])
                    if len(candidates) == 1:
                        throws = candidates[0][1]
                    elif len(candidates) > 1 and len(parts) >= 2:
                        first_part = parts[0].rstrip(".")
                        for fullname, t in candidates:
                            fname_parts = fullname.split()
                            if fname_parts and fname_parts[0].startswith(first_part):
                                throws = t
                                break

        # ERA from rotation map
        era_rot = None
        if rot_name and (cid, rot_name) in rotation_map:
            era_rot = rotation_map[(cid, rot_name)].get("era_rot")
        elif not throws:
            norm_pname = _norm_name(pitcher_name)
            if (cid, norm_pname) in rotation_map:
                era_rot = rotation_map[(cid, norm_pname)].get("era_rot")

        # ── Ability adjustment: FIP percentile chain ───────────────────────
        d1b_ability_adj = 0.0
        d1b_ability_source = ""

        if fip is not None and fip > 0:
            d1b_ability_adj = _fip_pctile_to_ability(fip, all_fips_arr)
            d1b_ability_source = "fip"
        elif siera is not None and siera > 0:
            d1b_ability_adj = _fip_pctile_to_ability(siera, all_sieras_arr)
            d1b_ability_source = "siera"
        elif era_d1b is not None and era_d1b > 0:
            d1b_ability_adj = _fip_pctile_to_ability(era_d1b, all_fips_arr)
            d1b_ability_source = "era_d1b"
        elif era_rot is not None and era_rot > 0:
            # Rotation page ERA: use linear ERA→ability mapping (different method)
            d1b_ability_adj = ERA_SLOPE * math.log(era_rot / ERA_BASELINE) + ERA_INTERCEPT
            d1b_ability_source = "era_rotation"
        elif not math.isnan(season_era_app) and season_era_app > 0:
            d1b_ability_adj = _fip_pctile_to_ability(season_era_app, all_fips_arr)
            d1b_ability_source = "era_appearances"

        rows.append({
            "pitcher_espn_id": pitcher_espn_id if pitcher_espn_id else "",
            "pitcher_idx": pitcher_idx,
            "pitcher_name": pitcher_name,
            "team_canonical_id": cid,
            "season": season,
            "throws": throws,
            "role": role,
            "season_ip": round(season_ip, 1) if not math.isnan(season_ip) else None,
            "season_era": round(season_era_app, 2) if not math.isnan(season_era_app) else None,
            "fip": fip,
            "siera": siera,
            "fb_pct": fb_pct,
            "fb_sensitivity": round(fb_sens, 4),
            "d1b_ability_adj": round(d1b_ability_adj, 6),
            "d1b_ability_source": d1b_ability_source,
            "n_appearances": n_app,
            "last_appearance": last_app if last_app else None,
        })

    df = pd.DataFrame(rows)

    # ── 9. Deduplicate by (pitcher_name_norm, team_canonical_id) ──────────
    # The appearances file may have the same pitcher under two different pitcher_id
    # formats (e.g., "NCAA_belardes__NCAA_614730" vs "NCAA_alec_belardes__NCAA_614730").
    # Prefer: higher pitcher_idx > more appearances > later last_appearance.
    df["_name_norm"] = df["pitcher_name"].str.strip().str.lower()
    # Sort so that best rows come first: pitcher_idx desc, n_appearances desc
    df = df.sort_values(["pitcher_idx", "n_appearances"], ascending=[False, False])
    pre_dedup = len(df)
    df = df.drop_duplicates(subset=["_name_norm", "team_canonical_id"], keep="first")
    df = df.drop(columns=["_name_norm"])
    df = df.reset_index(drop=True)
    if pre_dedup != len(df):
        print(f"  Deduped {pre_dedup - len(df)} duplicate (name, team) rows → {len(df)} rows", file=sys.stderr)

    # ── 10. Summary stats ──────────────────────────────────────────────────
    print(f"\nPitcher table summary:", file=sys.stderr)
    print(f"  Total rows: {len(df)}", file=sys.stderr)
    print(f"  With pitcher_idx > 0: {(df['pitcher_idx'] > 0).sum()}", file=sys.stderr)
    print(f"  With FIP: {df['fip'].notna().sum()}", file=sys.stderr)
    print(f"  With SIERA: {df['siera'].notna().sum()}", file=sys.stderr)
    print(f"  With FB%: {df['fb_pct'].notna().sum()}", file=sys.stderr)
    print(f"  With throws: {(df['throws'] != '').sum()}", file=sys.stderr)
    print(f"  With d1b_ability_adj != 0: {(df['d1b_ability_adj'] != 0).sum()}", file=sys.stderr)
    print(f"  Ability source breakdown:", file=sys.stderr)
    src_counts = df["d1b_ability_source"].value_counts()
    for src, cnt in src_counts.items():
        print(f"    {src or '(none)'}: {cnt}", file=sys.stderr)

    # ── 11. Write output ───────────────────────────────────────────────────
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    col_order = [
        "pitcher_espn_id", "pitcher_idx", "pitcher_name", "team_canonical_id",
        "season", "throws", "role", "season_ip", "season_era",
        "fip", "siera", "fb_pct", "fb_sensitivity",
        "d1b_ability_adj", "d1b_ability_source",
        "n_appearances", "last_appearance",
    ]
    df = df[col_order]
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(df)} rows to {out_csv}", file=sys.stderr)

    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Build unified pitcher lookup table.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/pitcher_table.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--appearances", type=Path, default=Path("data/processed/pitcher_appearances.csv"))
    parser.add_argument("--pitcher-index", type=Path, default=Path("data/processed/run_event_pitcher_index.csv"))
    parser.add_argument("--pitching-advanced", type=Path, default=Path("data/raw/d1baseball/pitching_advanced.tsv"))
    parser.add_argument("--pitching-standard", type=Path, default=Path("data/raw/d1baseball/pitching_standard.tsv"))
    parser.add_argument("--pitching-batted-ball", type=Path, default=Path("data/raw/d1baseball/pitching_batted_ball.tsv"))
    parser.add_argument("--d1b-root", type=Path, default=Path("data/raw/d1baseball"))
    parser.add_argument("--as-of-season", type=int, default=None)
    parser.add_argument("--rotations", type=Path, default=Path("data/processed/d1baseball_rotations.csv"))
    parser.add_argument("--d1b-crosswalk", type=Path, default=Path("data/registries/d1baseball_crosswalk.csv"))
    parser.add_argument("--canonical", type=Path, default=Path("data/registries/canonical_teams_2026.csv"))
    args = parser.parse_args()

    build_pitcher_table(
        appearances_csv=args.appearances,
        pitcher_index_csv=args.pitcher_index,
        pitching_advanced_tsv=args.pitching_advanced,
        pitching_standard_tsv=args.pitching_standard,
        pitching_batted_ball_tsv=args.pitching_batted_ball,
        rotations_csv=args.rotations,
        d1b_crosswalk_csv=args.d1b_crosswalk,
        canonical_csv=args.canonical,
        out_csv=args.out,
        d1b_root=args.d1b_root,
        as_of_season=args.as_of_season,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
