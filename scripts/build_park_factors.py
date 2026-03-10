"""
Build park factors from ESPN game JSONL files.

For each venue that hosts non-neutral-site games, calculates:
  - Overall park factor  = (runs/game at venue) / (league-wide runs/game)
  - Per-run-type factors  = same formula applied to run_1 .. run_4 individually
  - Bayesian-shrinkage adjusted factors to regress toward 1.0 for small samples

The home team for each venue is inferred from the most-common home_team across
games at that venue, then resolved to a canonical_id via the team registry.

Outputs:
  data/processed/park_factors.csv

Usage:
  python3 scripts/build_park_factors.py
  python3 scripts/build_park_factors.py --shrinkage 30 --min-games 5 --out data/processed/park_factors.csv
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from ncaa_baseball.phase1 import (
    build_odds_name_to_canonical,
    load_canonical_teams,
    resolve_odds_teams,
)

RUN_EVENT_KEYS = ("run_1", "run_2", "run_3", "run_4")


# ---------------------------------------------------------------------------
# JSONL reader
# ---------------------------------------------------------------------------

def stream_games(espn_dir: Path, seasons: list[str]):
    """Yield parsed game dicts from games_YYYY.jsonl files."""
    for season in seasons:
        path = espn_dir / f"games_{season}.jsonl"
        if not path.exists():
            print(f"  skip (not found): {path}")
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# ---------------------------------------------------------------------------
# Accumulation helpers
# ---------------------------------------------------------------------------

class VenueAccum:
    """Accumulates per-venue totals needed for park factor calculation."""

    __slots__ = (
        "games", "total_runs",
        "re_games",  # games with run_events data
        "re_totals",  # {key: total} for run_1..run_4
        "home_teams",  # Counter of home team names
    )

    def __init__(self):
        self.games: int = 0
        self.total_runs: int = 0
        self.re_games: int = 0
        self.re_totals: dict[str, int] = {k: 0 for k in RUN_EVENT_KEYS}
        self.home_teams: Counter = Counter()


def _safe_int(val) -> int | None:
    """Coerce to int or return None."""
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _venue_key(venue: dict) -> tuple[str, str, str]:
    """Return (name, city, state) tuple used as the venue dict key."""
    return (
        (venue.get("name") or "").strip(),
        (venue.get("city") or "").strip(),
        (venue.get("state") or "").strip(),
    )


def accumulate_venues(espn_dir: Path, seasons: list[str]) -> dict[tuple, VenueAccum]:
    """Read all JSONL files and accumulate per-venue stats."""
    venues: dict[tuple, VenueAccum] = defaultdict(VenueAccum)
    n_total = 0
    n_neutral = 0
    n_no_venue = 0
    n_no_score = 0

    for g in stream_games(espn_dir, seasons):
        n_total += 1

        # Skip neutral-site games
        if g.get("neutral_site"):
            n_neutral += 1
            continue

        venue = g.get("venue")
        if not venue or not (venue.get("name") or "").strip():
            n_no_venue += 1
            continue

        home_score = _safe_int(g.get("home_score"))
        away_score = _safe_int(g.get("away_score"))
        if home_score is None or away_score is None:
            n_no_score += 1
            continue

        key = _venue_key(venue)
        acc = venues[key]
        acc.games += 1
        acc.total_runs += home_score + away_score

        home_team = g.get("home_team") or {}
        home_name = (home_team.get("name") or "").strip()
        if home_name:
            acc.home_teams[home_name] += 1

        # Run-event data (may be null)
        re = g.get("run_events")
        if re and isinstance(re, dict):
            home_re = re.get("home") or {}
            away_re = re.get("away") or {}
            if home_re and away_re:
                acc.re_games += 1
                for rk in RUN_EVENT_KEYS:
                    h = _safe_int(home_re.get(rk))
                    a = _safe_int(away_re.get(rk))
                    acc.re_totals[rk] += (h or 0) + (a or 0)

    print(f"  total lines:    {n_total}")
    print(f"  neutral-site:   {n_neutral} (skipped)")
    print(f"  no venue:       {n_no_venue} (skipped)")
    print(f"  no score:       {n_no_score} (skipped)")
    print(f"  usable games:   {n_total - n_neutral - n_no_venue - n_no_score}")
    print(f"  unique venues:  {len(venues)}")

    return dict(venues)


# ---------------------------------------------------------------------------
# Park-factor computation
# ---------------------------------------------------------------------------

def compute_park_factors(
    venues: dict[tuple, VenueAccum],
    *,
    shrinkage_k: int,
    min_games: int,
) -> pd.DataFrame:
    """
    Compute raw and Bayesian-adjusted park factors.

    Park factor = (runs/game at venue) / (league-wide runs/game).
    Adjusted PF = (n * raw + k * 1.0) / (n + k).
    """
    # League-wide totals (across all venues that pass min_games filter)
    league_runs = 0
    league_games = 0
    league_re_runs: dict[str, int] = {k: 0 for k in RUN_EVENT_KEYS}
    league_re_games = 0

    for acc in venues.values():
        if acc.games < min_games:
            continue
        league_runs += acc.total_runs
        league_games += acc.games
        league_re_games += acc.re_games
        for rk in RUN_EVENT_KEYS:
            league_re_runs[rk] += acc.re_totals[rk]

    if league_games == 0:
        print("No qualifying venues — cannot compute park factors.")
        return pd.DataFrame()

    league_rpg = league_runs / league_games

    # Per-run-type league averages (per game, across games with run_events)
    league_re_rpg: dict[str, float] = {}
    for rk in RUN_EVENT_KEYS:
        league_re_rpg[rk] = league_re_runs[rk] / league_re_games if league_re_games > 0 else 0.0

    print(f"\n  league avg runs/game: {league_rpg:.3f}  ({league_games} games)")
    for rk in RUN_EVENT_KEYS:
        print(f"  league avg {rk}/game:  {league_re_rpg[rk]:.3f}  ({league_re_games} games w/ PBP)")

    # Build rows
    rows: list[dict] = []
    for (vname, vcity, vstate), acc in venues.items():
        if acc.games < min_games:
            continue

        rpg = acc.total_runs / acc.games
        raw_pf = rpg / league_rpg if league_rpg > 0 else 1.0
        n = acc.games
        adj_pf = (n * raw_pf + shrinkage_k * 1.0) / (n + shrinkage_k)

        row: dict = {
            "venue_name": vname,
            "venue_city": vcity,
            "venue_state": vstate,
            "n_games": n,
            "runs_per_game": round(rpg, 4),
            "raw_pf": round(raw_pf, 4),
            "adjusted_pf": round(adj_pf, 4),
        }

        # Dominant home team (most-common home_team.name at this venue)
        if acc.home_teams:
            row["_home_team_name"] = acc.home_teams.most_common(1)[0][0]
        else:
            row["_home_team_name"] = ""

        # Per-run-type park factors
        if acc.re_games >= max(1, min_games // 2):
            for rk in RUN_EVENT_KEYS:
                venue_rpg_rk = acc.re_totals[rk] / acc.re_games
                lg_rpg_rk = league_re_rpg[rk]
                raw_rk = venue_rpg_rk / lg_rpg_rk if lg_rpg_rk > 0 else 1.0
                n_re = acc.re_games
                adj_rk = (n_re * raw_rk + shrinkage_k * 1.0) / (n_re + shrinkage_k)
                row[f"raw_pf_{rk}"] = round(raw_rk, 4)
                row[f"adjusted_pf_{rk}"] = round(adj_rk, 4)
        else:
            for rk in RUN_EVENT_KEYS:
                row[f"raw_pf_{rk}"] = None
                row[f"adjusted_pf_{rk}"] = None

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Team resolution
# ---------------------------------------------------------------------------

def resolve_home_teams(
    df: pd.DataFrame,
    canonical_path: Path,
) -> pd.DataFrame:
    """Map venue home team name to canonical_id using the team registry."""
    if df.empty:
        df["home_team_id"] = []
        df["home_team_name"] = []
        return df

    canonical = load_canonical_teams(canonical_path)
    name_to_canonical = build_odds_name_to_canonical(canonical)

    # Build a quick map: canonical_id -> team_name from registry for display
    cid_to_display: dict[str, str] = {}
    for _, row in canonical.iterrows():
        cid = (row.get("canonical_id") or "").strip()
        tname = (row.get("team_name") or "").strip()
        if cid and tname:
            cid_to_display[cid] = tname

    home_ids = []
    home_names = []
    n_resolved = 0
    for ht_name in df["_home_team_name"]:
        if not ht_name:
            home_ids.append("")
            home_names.append("")
            continue
        # resolve_odds_teams expects a pair; use the same name for both and take home
        ht, _ = resolve_odds_teams(ht_name, ht_name, canonical, name_to_canonical)
        if ht:
            home_ids.append(ht[0])
            home_names.append(cid_to_display.get(ht[0], ht_name))
            n_resolved += 1
        else:
            home_ids.append("")
            home_names.append(ht_name)

    df["home_team_id"] = home_ids
    df["home_team_name"] = home_names
    print(f"\n  home-team resolution: {n_resolved}/{len(df)} venues mapped to canonical_id")

    # Drop internal helper column
    df = df.drop(columns=["_home_team_name"])
    return df


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, n: int = 10) -> None:
    """Print top hitter-friendly and pitcher-friendly parks."""
    if df.empty:
        print("No park factors to display.")
        return

    sorted_df = df.sort_values("adjusted_pf", ascending=False)

    print(f"\n{'='*72}")
    print(f" TOP {n} HITTER-FRIENDLY PARKS (adjusted PF)")
    print(f"{'='*72}")
    print(f"  {'Venue':<40s} {'Team':<20s} {'Games':>5s} {'Raw':>6s} {'Adj':>6s}")
    print(f"  {'-'*40} {'-'*20} {'-'*5} {'-'*6} {'-'*6}")
    for _, row in sorted_df.head(n).iterrows():
        venue = row["venue_name"][:40]
        team = (row.get("home_team_name") or "")[:20]
        print(f"  {venue:<40s} {team:<20s} {row['n_games']:>5d} {row['raw_pf']:>6.3f} {row['adjusted_pf']:>6.3f}")

    print(f"\n{'='*72}")
    print(f" TOP {n} PITCHER-FRIENDLY PARKS (adjusted PF)")
    print(f"{'='*72}")
    print(f"  {'Venue':<40s} {'Team':<20s} {'Games':>5s} {'Raw':>6s} {'Adj':>6s}")
    print(f"  {'-'*40} {'-'*20} {'-'*5} {'-'*6} {'-'*6}")
    for _, row in sorted_df.tail(n).iloc[::-1].iterrows():
        venue = row["venue_name"][:40]
        team = (row.get("home_team_name") or "")[:20]
        print(f"  {venue:<40s} {team:<20s} {row['n_games']:>5d} {row['raw_pf']:>6.3f} {row['adjusted_pf']:>6.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "venue_name", "venue_city", "venue_state",
    "home_team_id", "home_team_name",
    "n_games", "runs_per_game", "raw_pf", "adjusted_pf",
    "raw_pf_run_1", "raw_pf_run_2", "raw_pf_run_3", "raw_pf_run_4",
    "adjusted_pf_run_1", "adjusted_pf_run_2", "adjusted_pf_run_3", "adjusted_pf_run_4",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build park factors from ESPN game JSONL files.",
    )
    parser.add_argument(
        "--espn-dir",
        type=Path,
        default=Path("data/raw/espn"),
        help="Directory containing games_YYYY.jsonl",
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=Path("data/registries/canonical_teams_2026.csv"),
        help="Canonical teams CSV for name resolution",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/park_factors.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default="2024,2025,2026",
        help="Comma-separated seasons to include",
    )
    parser.add_argument(
        "--shrinkage",
        type=int,
        default=20,
        help="Bayesian shrinkage constant k (games). Higher = more regression to mean.",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=5,
        help="Minimum games at a venue to include it in output",
    )
    args = parser.parse_args()

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    print(f"Reading ESPN JSONL from {args.espn_dir} (seasons: {', '.join(seasons)})")

    # 1. Accumulate per-venue stats
    venues = accumulate_venues(args.espn_dir, seasons)
    if not venues:
        print("No game data found.")
        return 1

    # 2. Compute park factors
    print(f"\nComputing park factors (shrinkage k={args.shrinkage}, min_games={args.min_games})...")
    df = compute_park_factors(venues, shrinkage_k=args.shrinkage, min_games=args.min_games)
    if df.empty:
        print("No venues passed the minimum-games filter.")
        return 1

    # 3. Resolve home teams to canonical_id
    print("\nResolving home teams to canonical_id...")
    df = resolve_home_teams(df, args.canonical)

    # 4. Order columns and write
    # Ensure all expected columns exist (fill missing with None)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[OUTPUT_COLUMNS].sort_values("adjusted_pf", ascending=False).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} venue park factors -> {args.out}")

    # 5. Summary
    print_summary(df)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
