"""
Build EXPANDED run_events table from ALL data sources (Path A step 6.1b).

Combines:
  1. ESPN PBP games (actual at-bat-level run events — the gold standard)
  2. ESPN non-PBP games (have total scores; run events estimated)
  3. NCAA boxscore games with inning-level linescores (inning-level decomposition)
  4. NCAA boxscore games without linescores (game-level bootstrap estimation)

Run event estimation hierarchy (best to worst):
  1. PBP run events: exact counts from play-by-play data
  2. Inning-level decomposition: for each inning in the linescore, decompose runs into
     run events using multinomial weights from PBP event-type probabilities.
     0-run innings = exact zeros; 1-run innings = exact (1,0,0,0); 2+ run innings =
     sampled from weighted compositions.  Far more precise than game-level since ~70%
     of innings score 0 runs and ~20% score exactly 1.
  3. Game-level bootstrap: P(run_k | total_game_runs) from PBP games (fallback)

Pitcher integration:
  - ESPN games: use espn_id (e.g. "ESPN_67525")
  - NCAA games: use normalized name+team ID (e.g. "NCAA_john_smith__BSB_UCLA")
  - All pitchers (starters + relievers) are in the unified pitcher registry
  - The starter pitcher_id is stored for each side; reliever data in pitcher_appearances.csv

Source tags:
  pbp                    — ESPN PBP (exact run events)
  espn_score             — ESPN score-only (game-level bootstrap)
  ncaa_linescore         — NCAA with inning-level decomposition (both sides)
  ncaa_linescore_partial — NCAA with inning-level for one side, bootstrap for other
  ncaa_score             — NCAA without linescores (game-level bootstrap)

Usage:
  python3 scripts/build_run_events_expanded.py
  python3 scripts/build_run_events_expanded.py --out data/processed/run_events_expanded.csv
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from ncaa_baseball.phase1 import (
    build_odds_name_to_canonical,
    load_canonical_teams,
    resolve_odds_teams,
)


# ──────────────────────────────────────────────────────────────────────────────
# Pitcher name normalization (must match build_pitcher_registry.py)
# ──────────────────────────────────────────────────────────────────────────────

_SUFFIX_RE = re.compile(r"\s+(Jr\.?|Sr\.?|III|II|IV|V)\s*$", re.IGNORECASE)

def _normalize_pitcher_name(name: str) -> str:
    name = name.strip()
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = _SUFFIX_RE.sub("", name).strip()
    name = name.replace(".", "")
    name = re.sub(r"\s+", "_", name.lower().strip())
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name

def _make_ncaa_pitcher_id(name: str, team_canonical_id: str) -> str:
    norm = _normalize_pitcher_name(name)
    if not norm:
        return ""
    team_clean = team_canonical_id.replace(" ", "_")
    return f"NCAA_{norm}__{team_clean}"

# ──────────────────────────────────────────────────────────────────────────────
# Empirical conditional distribution of run events given total runs
# ──────────────────────────────────────────────────────────────────────────────

def build_conditional_run_events(
    pbp_df: pd.DataFrame,
) -> dict[int, list[tuple[int, int, int, int]]]:
    """
    From PBP run events, build mapping:  total_runs -> list of (run_1,run_2,run_3,run_4)
    observed for that total.  Used for bootstrap sampling.
    """
    cond: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for _, row in pbp_df.iterrows():
        for side in ("home", "away"):
            r1 = int(row.get(f"{side}_run_1", 0) or 0)
            r2 = int(row.get(f"{side}_run_2", 0) or 0)
            r3 = int(row.get(f"{side}_run_3", 0) or 0)
            r4 = int(row.get(f"{side}_run_4", 0) or 0)
            total = int(row.get(f"{side}_score", 0) or 0)
            if total >= 0:
                cond[total].append((r1, r2, r3, r4))
    return dict(cond)


def sample_run_events(
    total_runs: int,
    cond: dict[int, list[tuple[int, int, int, int]]],
    rng: random.Random,
) -> tuple[int, int, int, int]:
    """
    Sample (run_1, run_2, run_3, run_4) from the empirical conditional distribution
    for the given total_runs.  Falls back to the nearest available total if exact
    match has no data.
    """
    if total_runs in cond and cond[total_runs]:
        return rng.choice(cond[total_runs])
    # Nearest total with data
    available = sorted(cond.keys())
    if not available:
        # Extreme fallback: attribute all runs to run_1 events
        return (total_runs, 0, 0, 0)
    nearest = min(available, key=lambda t: abs(t - total_runs))
    return rng.choice(cond[nearest])


# ──────────────────────────────────────────────────────────────────────────────
# Inning-level run event decomposition (using linescore data)
#
# Instead of sampling P(run_events | total_game_runs), we decompose each inning:
#   0-run inning → (0,0,0,0) exactly
#   1-run inning → (1,0,0,0) exactly
#   k-run inning → sample from multinomial-weighted compositions
# Then sum across innings for game totals.  This is far more precise since
# ~70% of innings score 0 runs and ~20% score exactly 1 run.
# ──────────────────────────────────────────────────────────────────────────────

def compute_event_type_probs(
    pbp_df: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """Compute per-event-type probabilities from PBP data.

    Returns (p_1, p_2, p_3, p_4) where p_k = P(scoring event yields k runs).
    Estimated as: count of run_k events / total scoring events across all PBP games.
    """
    t1 = t2 = t3 = t4 = 0
    for _, row in pbp_df.iterrows():
        for side in ("home", "away"):
            t1 += int(row.get(f"{side}_run_1", 0) or 0)
            t2 += int(row.get(f"{side}_run_2", 0) or 0)
            t3 += int(row.get(f"{side}_run_3", 0) or 0)
            t4 += int(row.get(f"{side}_run_4", 0) or 0)
    total = t1 + t2 + t3 + t4
    if total == 0:
        return (1.0, 0.0, 0.0, 0.0)
    return (t1 / total, t2 / total, t3 / total, t4 / total)


def _enumerate_compositions(k: int) -> list[tuple[int, int, int, int]]:
    """Enumerate all (n1, n2, n3, n4) where n1 + 2*n2 + 3*n3 + 4*n4 = k."""
    result = []
    for n4 in range(k // 4 + 1):
        rem4 = k - 4 * n4
        for n3 in range(rem4 // 3 + 1):
            rem3 = rem4 - 3 * n3
            for n2 in range(rem3 // 2 + 1):
                n1 = rem3 - 2 * n2
                result.append((n1, n2, n3, n4))
    return result


def build_inning_decomp_table(
    event_probs: tuple[float, float, float, float],
    max_k: int = 30,
    event_count_discount: float = 0.35,
) -> dict[int, list[tuple[tuple[int, int, int, int], float]]]:
    """Build decomposition table: inning_runs -> [(composition, prob), ...].

    For each possible single-inning run total k:
      k=0 → (0,0,0,0) deterministically
      k=1 → (1,0,0,0) deterministically
      k≥2 → enumerate all (n1,n2,n3,n4) with n1+2n2+3n3+4n4=k,
             weight each by:
               multinomial(n1..n4 | N, p) × discount^(N-1)
             where N = n1+n2+n3+n4.

    The event_count_discount (β) penalizes compositions requiring more scoring
    events in a single inning, since P(N events in inning) decreases with N.
    Without this, the multinomial systematically over-weights run_1 events.
    β≈0.35 calibrates so game-level event-type fractions match PBP ground truth.
    """
    p = event_probs
    log_p = [math.log(pi) if pi > 0 else float("-inf") for pi in p]
    log_discount = math.log(event_count_discount) if event_count_discount > 0 else float("-inf")

    table: dict[int, list[tuple[tuple[int, int, int, int], float]]] = {
        0: [((0, 0, 0, 0), 1.0)],
        1: [((1, 0, 0, 0), 1.0)],
    }

    for k in range(2, max_k + 1):
        comps = _enumerate_compositions(k)
        weighted: list[tuple[tuple[int, int, int, int], float]] = []
        for c in comps:
            n = sum(c)
            # log-multinomial weight + event count penalty
            log_w = math.lgamma(n + 1)
            # Discount for number of events: β^(N-1)
            if n > 1:
                log_w += (n - 1) * log_discount
            valid = True
            for i in range(4):
                log_w -= math.lgamma(c[i] + 1)
                if c[i] > 0:
                    if log_p[i] == float("-inf"):
                        valid = False
                        break
                    log_w += c[i] * log_p[i]
            if valid:
                weighted.append((c, log_w))

        if not weighted:
            table[k] = [((k, 0, 0, 0), 1.0)]
            continue

        # Convert from log-weights to normalized probabilities
        max_log = max(w for _, w in weighted)
        probs = [(comp, math.exp(lw - max_log)) for comp, lw in weighted]
        total_w = sum(w for _, w in probs)
        table[k] = [(comp, w / total_w) for comp, w in probs]

    return table


def sample_inning_decomp(
    inning_runs: int,
    decomp_table: dict[int, list[tuple[tuple[int, int, int, int], float]]],
    rng: random.Random,
) -> tuple[int, int, int, int]:
    """Sample a (run_1, run_2, run_3, run_4) decomposition for one inning."""
    if inning_runs <= 0:
        return (0, 0, 0, 0)
    if inning_runs == 1:
        return (1, 0, 0, 0)
    entries = decomp_table.get(inning_runs)
    if not entries:
        return (inning_runs, 0, 0, 0)
    # Weighted random choice
    comps, weights = zip(*entries)
    cum = []
    running = 0.0
    for w in weights:
        running += w
        cum.append(running)
    r = rng.random() * running
    for i, c in enumerate(cum):
        if r <= c:
            return comps[i]
    return comps[-1]


def decompose_linescore(
    innings: list[int],
    decomp_table: dict[int, list[tuple[tuple[int, int, int, int], float]]],
    rng: random.Random,
) -> tuple[int, int, int, int]:
    """Decompose linescore (runs per inning) into game-level run event totals.

    For each inning, sample a run event decomposition, then sum across innings.
    """
    r1 = r2 = r3 = r4 = 0
    for inning_runs in innings:
        dr1, dr2, dr3, dr4 = sample_inning_decomp(inning_runs, decomp_table, rng)
        r1 += dr1
        r2 += dr2
        r3 += dr3
        r4 += dr4
    return (r1, r2, r3, r4)


# ──────────────────────────────────────────────────────────────────────────────
# ESPN game loading
# ──────────────────────────────────────────────────────────────────────────────

def load_espn_games(
    espn_dir: Path,
    seasons: list[str],
    canonical: pd.DataFrame,
    name_to_canonical: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Load all ESPN games.  Returns (pbp_rows, score_only_rows).
    pbp_rows have actual run events.  score_only_rows have total scores only.
    """
    pbp_rows: list[dict] = []
    score_rows: list[dict] = []

    for season in seasons:
        path = espn_dir / f"games_{season}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    g = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_id = g.get("event_id") or g.get("id") or ""
                game_date = (g.get("date") or "")[:10]
                try:
                    yr = int(g.get("season") or season)
                except (TypeError, ValueError):
                    yr = int(season) if season.isdigit() else 0

                home = g.get("home_team") or {}
                away = g.get("away_team") or {}
                home_name = (home.get("name") or "").strip()
                away_name = (away.get("name") or "").strip()
                if not home_name or not away_name:
                    continue

                home_t, away_t = resolve_odds_teams(
                    home_name, away_name, canonical, name_to_canonical,
                )
                home_canonical = home_t[0] if home_t else ""
                away_canonical = away_t[0] if away_t else ""

                home_score = _safe_int(g.get("home_score"))
                away_score = _safe_int(g.get("away_score"))
                if home_score is None or away_score is None:
                    continue

                starters = g.get("starters") or {}
                hp = starters.get("home_pitcher") or {}
                ap = starters.get("away_pitcher") or {}
                hp_espn = str(hp.get("espn_id") or hp.get("id") or "")
                ap_espn = str(ap.get("espn_id") or ap.get("id") or "")
                # Unified pitcher IDs: "ESPN_{espn_id}" or empty
                hp_id = f"ESPN_{hp_espn}" if hp_espn else ""
                ap_id = f"ESPN_{ap_espn}" if ap_espn else ""

                # Check for PBP run events
                re = g.get("run_events")
                if re and isinstance(re, dict):
                    home_re = re.get("home") or {}
                    away_re = re.get("away") or {}
                    if home_re and away_re:
                        pbp_rows.append({
                            "event_id": event_id,
                            "game_date": game_date,
                            "season": yr,
                            "home_canonical_id": home_canonical,
                            "away_canonical_id": away_canonical,
                            "home_pitcher_id": hp_id,
                            "away_pitcher_id": ap_id,
                            "home_run_1": _safe_int(home_re.get("run_1")) or 0,
                            "home_run_2": _safe_int(home_re.get("run_2")) or 0,
                            "home_run_3": _safe_int(home_re.get("run_3")) or 0,
                            "home_run_4": _safe_int(home_re.get("run_4")) or 0,
                            "away_run_1": _safe_int(away_re.get("run_1")) or 0,
                            "away_run_2": _safe_int(away_re.get("run_2")) or 0,
                            "away_run_3": _safe_int(away_re.get("run_3")) or 0,
                            "away_run_4": _safe_int(away_re.get("run_4")) or 0,
                            "home_score": home_score,
                            "away_score": away_score,
                            "source": "pbp",
                        })
                        continue

                # Score-only game (no PBP)
                score_rows.append({
                    "event_id": event_id,
                    "game_date": game_date,
                    "season": yr,
                    "home_canonical_id": home_canonical,
                    "away_canonical_id": away_canonical,
                    "home_pitcher_id": hp_id,
                    "away_pitcher_id": ap_id,
                    "home_score": home_score,
                    "away_score": away_score,
                    "source": "espn_score",
                })

    return pbp_rows, score_rows


# ──────────────────────────────────────────────────────────────────────────────
# NCAA boxscore loading
# ──────────────────────────────────────────────────────────────────────────────

def _find_starter(pitchers: list[dict], team_cid: str) -> str:
    """Find the starting pitcher from a list of pitcher dicts.

    Strategy:
      1. Look for the pitcher with starter=True
      2. If multiple starters, pick the one with most IP
      3. If no starters flagged, pick the one with most IP (likely the starter)
    Returns unified NCAA pitcher ID or "".
    """
    if not pitchers:
        return ""

    starters = [p for p in pitchers if p.get("starter")]
    if starters:
        # Pick the one with most IP among starters
        best = max(starters, key=lambda p: float(p.get("ip", 0) or 0))
    else:
        # No starter flag — pick most IP pitcher as heuristic
        best = max(pitchers, key=lambda p: float(p.get("ip", 0) or 0))

    name = str(best.get("name", "")).strip()
    if not name:
        return ""
    return _make_ncaa_pitcher_id(name, team_cid)


def load_ncaa_boxscores(
    ncaa_path: Path,
    canonical: pd.DataFrame,
    name_to_canonical: dict,
    espn_event_ids: set[str],
) -> list[dict]:
    """
    Load NCAA boxscore games, skipping games already captured from ESPN.
    Uses fuzzy team name resolution via resolve_odds_teams.

    Now extracts STARTER pitcher IDs using unified NCAA pitcher ID format,
    and records reliever IP for bullpen quality computation.
    """
    rows: list[dict] = []
    if not ncaa_path.exists():
        return rows

    # Build quick team_name -> canonical_id lookup from canonical teams
    name_map: dict[str, str] = {}
    for _, row in canonical.iterrows():
        tname = (row.get("team_name") or "").strip()
        cid = (row.get("canonical_id") or "").strip()
        if tname and cid:
            name_map[tname.lower()] = cid

    n_with_starter = 0

    with ncaa_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                g = json.loads(line)
            except json.JSONDecodeError:
                continue

            game_id = str(g.get("game_id", ""))
            # Skip if we already have this game from ESPN
            if game_id in espn_event_ids:
                continue

            home_name = str(g.get("home_team", "")).strip()
            away_name = str(g.get("away_team", "")).strip()
            if not home_name or not away_name:
                continue

            home_score = _safe_int(g.get("home_score"))
            away_score = _safe_int(g.get("away_score"))
            if home_score is None or away_score is None:
                continue

            # Resolve team names — try direct lookup first, then fuzzy resolution
            home_cid = name_map.get(home_name.lower(), "")
            away_cid = name_map.get(away_name.lower(), "")
            if not home_cid or not away_cid:
                h_t, a_t = resolve_odds_teams(
                    home_name, away_name, canonical, name_to_canonical,
                )
                if not home_cid and h_t:
                    home_cid = h_t[0]
                if not away_cid and a_t:
                    away_cid = a_t[0]

            game_date = str(g.get("date", ""))[:10]
            pitching = g.get("pitching") or {}

            # Extract starter pitcher IDs using unified NCAA pitcher ID
            home_pitchers = pitching.get("home", [])
            away_pitchers = pitching.get("away", [])

            home_starter_id = _find_starter(home_pitchers, home_cid)
            away_starter_id = _find_starter(away_pitchers, away_cid)

            if home_starter_id or away_starter_id:
                n_with_starter += 1

            # Compute reliever IP totals for each side
            home_reliever_ip = sum(
                float(p.get("ip", 0) or 0)
                for p in home_pitchers if not p.get("starter")
            )
            away_reliever_ip = sum(
                float(p.get("ip", 0) or 0)
                for p in away_pitchers if not p.get("starter")
            )

            rows.append({
                "event_id": f"NCAA_{game_id}",
                "game_date": game_date,
                "season": 2026,
                "home_canonical_id": home_cid,
                "away_canonical_id": away_cid,
                "home_pitcher_id": home_starter_id,
                "away_pitcher_id": away_starter_id,
                "home_score": home_score,
                "away_score": away_score,
                "home_reliever_ip": home_reliever_ip,
                "away_reliever_ip": away_reliever_ip,
                "home_n_pitchers": len(home_pitchers),
                "away_n_pitchers": len(away_pitchers),
                "source": "ncaa_score",
            })

    print(f"  NCAA games with identified starter: {n_with_starter}/{len(rows)}")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _safe_int(v) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build expanded run_events.csv from ALL data sources.",
    )
    parser.add_argument("--espn-dir", type=Path, default=Path("data/raw/espn"))
    parser.add_argument("--ncaa-boxscores", type=Path, default=Path("data/raw/ncaa/boxscores_2026.jsonl"))
    parser.add_argument("--ncaa-linescores", type=Path, default=Path("data/raw/ncaa/linescores_2026.jsonl"))
    parser.add_argument("--canonical", type=Path, default=Path("data/registries/canonical_teams_2026.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/run_events_expanded.csv"))
    parser.add_argument("--seasons", type=str, default="2024,2025,2026")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap sampling")
    args = parser.parse_args()

    canonical = load_canonical_teams(args.canonical)
    name_to_canonical = build_odds_name_to_canonical(canonical)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]

    print("Loading ESPN games (PBP + score-only)...")
    pbp_rows, score_rows = load_espn_games(
        args.espn_dir, seasons, canonical, name_to_canonical,
    )
    print(f"  ESPN PBP games: {len(pbp_rows)}")
    print(f"  ESPN score-only games: {len(score_rows)}")

    # Build empirical conditional distribution from PBP games (game-level fallback)
    pbp_df = pd.DataFrame(pbp_rows)
    cond = build_conditional_run_events(pbp_df) if not pbp_df.empty else {}
    if cond:
        totals = sorted(cond.keys())
        print(f"  Conditional distribution covers totals: {min(totals)}-{max(totals)} "
              f"({sum(len(v) for v in cond.values())} observations)")

    # Build inning-level decomposition table from PBP event-type rates
    decomp_table = {}
    if not pbp_df.empty:
        event_probs = compute_event_type_probs(pbp_df)
        print(f"  Event type probs: p1={event_probs[0]:.3f}, p2={event_probs[1]:.3f}, "
              f"p3={event_probs[2]:.3f}, p4={event_probs[3]:.3f}")
        decomp_table = build_inning_decomp_table(event_probs)
        # Show examples for common inning totals
        for k in (2, 3, 4):
            entries = decomp_table.get(k, [])
            top = sorted(entries, key=lambda x: -x[1])[:3]
            parts = ", ".join(f"({c[0]},{c[1]},{c[2]},{c[3]})={w:.1%}" for c, w in top)
            print(f"    k={k}: {parts}")

    # Load NCAA linescores for inning-level decomposition
    linescores: dict[str, dict] = {}
    if args.ncaa_linescores.exists():
        with args.ncaa_linescores.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ls = json.loads(line)
                except json.JSONDecodeError:
                    continue
                gid = str(ls.get("game_id", ""))
                if gid:
                    linescores[gid] = ls
        print(f"  Loaded {len(linescores)} linescores for inning-level decomposition")

    # Load NCAA boxscores (dedup against ESPN)
    espn_event_ids = set()
    for r in pbp_rows:
        espn_event_ids.add(str(r.get("event_id", "")))
    for r in score_rows:
        espn_event_ids.add(str(r.get("event_id", "")))

    print("Loading NCAA boxscores...")
    ncaa_rows = load_ncaa_boxscores(
        args.ncaa_boxscores, canonical, name_to_canonical, espn_event_ids,
    )
    print(f"  NCAA boxscore games (non-duplicate): {len(ncaa_rows)}")

    # ── Estimate run events for score-only games ──────────────────────────────
    # Strategy:
    #   1. For NCAA games with matching linescores: inning-level decomposition
    #      (per-side: only if sum(innings) == total_score for that side)
    #   2. Fallback: game-level bootstrap from PBP conditional distribution
    rng = random.Random(args.seed)
    n_linescore_full = 0       # both sides from inning-level
    n_linescore_partial = 0    # one side from inning-level, other from bootstrap
    n_bootstrap = 0            # both sides from game-level bootstrap

    for row in score_rows + ncaa_rows:
        event_id = str(row.get("event_id", ""))

        # Look up linescore for NCAA games
        ls = None
        if event_id.startswith("NCAA_") and decomp_table:
            raw_gid = event_id[5:]  # strip "NCAA_" prefix
            ls = linescores.get(raw_gid)

        side_used_ls = {"home": False, "away": False}

        for side in ("home", "away"):
            total = int(row.get(f"{side}_score", 0) or 0)
            used_ls = False

            # Try inning-level decomposition if linescore available
            if ls:
                innings_raw = ls.get(f"{side}_innings", [])
                try:
                    innings_int = [int(x) for x in innings_raw if x is not None]
                except (TypeError, ValueError):
                    innings_int = []

                if innings_int and sum(innings_int) == total:
                    r1, r2, r3, r4 = decompose_linescore(
                        innings_int, decomp_table, rng,
                    )
                    row[f"{side}_run_1"] = r1
                    row[f"{side}_run_2"] = r2
                    row[f"{side}_run_3"] = r3
                    row[f"{side}_run_4"] = r4
                    used_ls = True

            if not used_ls:
                # Game-level bootstrap fallback
                r1, r2, r3, r4 = sample_run_events(total, cond, rng)
                row[f"{side}_run_1"] = r1
                row[f"{side}_run_2"] = r2
                row[f"{side}_run_3"] = r3
                row[f"{side}_run_4"] = r4

            side_used_ls[side] = used_ls

        # Update source tag and counters
        if side_used_ls["home"] and side_used_ls["away"]:
            if row.get("source") == "ncaa_score":
                row["source"] = "ncaa_linescore"
            n_linescore_full += 1
        elif side_used_ls["home"] or side_used_ls["away"]:
            if row.get("source") == "ncaa_score":
                row["source"] = "ncaa_linescore_partial"
            n_linescore_partial += 1
        else:
            n_bootstrap += 1

    n_total_est = n_linescore_full + n_linescore_partial + n_bootstrap
    print(f"  Estimated run events for {n_total_est} games:")
    print(f"    Inning-level (both sides): {n_linescore_full}")
    print(f"    Inning-level (one side):   {n_linescore_partial}")
    print(f"    Game-level bootstrap:      {n_bootstrap}")

    # Combine all rows
    all_rows = pbp_rows + score_rows + ncaa_rows
    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No games found across any source!")
        return 1

    # Sort by date
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values("game_date").reset_index(drop=True)
    df["game_date"] = df["game_date"].dt.strftime("%Y-%m-%d")

    # Filter to games where both teams resolved
    n_before = len(df)
    both_resolved = (df["home_canonical_id"].astype(str).str.strip() != "") & \
                    (df["away_canonical_id"].astype(str).str.strip() != "")
    df_resolved = df[both_resolved].copy()
    n_after = len(df_resolved)

    # Count unique teams
    all_teams = set(df_resolved["home_canonical_id"].unique()) | set(df_resolved["away_canonical_id"].unique())
    all_teams.discard("")

    # Source breakdown
    source_counts = df_resolved["source"].value_counts()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_resolved.to_csv(args.out, index=False)
    print(f"\n=== EXPANDED RUN EVENTS ===")
    print(f"Total games: {n_after} (of {n_before} loaded, {n_before - n_after} dropped unresolved)")
    print(f"Unique teams: {len(all_teams)}")
    for src, cnt in source_counts.items():
        print(f"  {src}: {cnt}")
    print(f"Output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
