#!/usr/bin/env python3
"""Convert NCAA inning-by-inning linescores → run_events format.

Each inning where a team scores N runs becomes a run_N event.
A 3-run inning = one run_3 event (we don't need full PBP to know this).

Only uses games with quality="full" (both teams have real inning data).
Resolves NCAA team names → canonical_id via phase1 resolver.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Bootstrap path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import _bootstrap  # noqa: F401

from integrate_ncaa_boxscores import build_ncaa_team_map, resolve_team_name


def load_linescores(path: Path) -> list[dict]:
    """Load NCAA linescore JSONL."""
    games = []
    with open(path) as f:
        for line in f:
            games.append(json.loads(line))
    return games


def innings_to_run_events(innings: list[int]) -> dict[str, int]:
    """Convert inning-by-inning runs to run_event counts.

    Each inning with N runs (N>0) contributes one run_N event.
    run_4 captures all innings with 4+ runs (capped at 4).
    """
    counts = {"run_1": 0, "run_2": 0, "run_3": 0, "run_4": 0}
    for runs in innings:
        if runs <= 0:
            continue
        key = f"run_{min(runs, 4)}"
        counts[key] += 1
    return counts


def build_run_events(
    linescores_path: Path,
    canonical_csv: Path,
    output_path: Path,
    *,
    quality_filter: str = "full",
) -> pd.DataFrame:
    """Main: convert linescores → run_events CSV."""
    games = load_linescores(linescores_path)
    print(f"Loaded {len(games)} linescore games", file=sys.stderr)

    # Filter to full quality only
    if quality_filter:
        games = [g for g in games if g.get("quality") == quality_filter]
        print(f"  After quality={quality_filter} filter: {len(games)}", file=sys.stderr)

    # Resolve team names using same approach as integrate_ncaa_boxscores
    name_map = build_ncaa_team_map(canonical_csv)
    print(f"  Built {len(name_map)} team name mappings", file=sys.stderr)

    rows = []
    unresolved_home = set()
    unresolved_away = set()
    for g in games:
        h_cid = resolve_team_name(g["home_team"], name_map)
        a_cid = resolve_team_name(g["away_team"], name_map)
        if not h_cid:
            unresolved_home.add(g["home_team"])
            continue
        if not a_cid:
            unresolved_away.add(g["away_team"])
            continue

        # Skip if same team (data error)
        if h_cid == a_cid:
            continue

        h_events = innings_to_run_events(g["home_innings"])
        a_events = innings_to_run_events(g["away_innings"])

        rows.append({
            "event_id": f"ls_{g['game_id']}",
            "game_date": g["date"],
            "season": 2026,
            "home_canonical_id": h_cid,
            "away_canonical_id": a_cid,
            "home_pitcher_id": "",   # no pitcher data from linescores
            "away_pitcher_id": "",
            "home_run_1": h_events["run_1"],
            "home_run_2": h_events["run_2"],
            "home_run_3": h_events["run_3"],
            "home_run_4": h_events["run_4"],
            "away_run_1": a_events["run_1"],
            "away_run_2": a_events["run_2"],
            "away_run_3": a_events["run_3"],
            "away_run_4": a_events["run_4"],
            "home_score": g["home_runs_total"],
            "away_score": g["away_runs_total"],
            "source": "ncaa_linescore",
            "home_reliever_ip": "",
            "away_reliever_ip": "",
            "home_n_pitchers": "",
            "away_n_pitchers": "",
        })

    if unresolved_home:
        print(f"  WARNING: {len(unresolved_home)} unresolved home teams: {sorted(unresolved_home)[:10]}...", file=sys.stderr)
    if unresolved_away:
        print(f"  WARNING: {len(unresolved_away)} unresolved away teams: {sorted(unresolved_away)[:10]}...", file=sys.stderr)

    df = pd.DataFrame(rows)
    print(f"  Generated {len(df)} run_event rows from linescores", file=sys.stderr)

    # Validate: total runs from events should match linescore totals
    df["_check_home"] = df["home_run_1"] + 2*df["home_run_2"] + 3*df["home_run_3"] + 4*df["home_run_4"]
    df["_check_away"] = df["away_run_1"] + 2*df["away_run_2"] + 3*df["away_run_3"] + 4*df["away_run_4"]
    # run_4 captures 4+ runs, so check will undercount big innings
    # Just warn on gross mismatches
    bad = df[abs(df["_check_home"] - df["home_score"]) > 3]
    if len(bad) > 0:
        print(f"  WARNING: {len(bad)} games with >3 run discrepancy (expected for 5+ run innings)", file=sys.stderr)
    df.drop(columns=["_check_home", "_check_away"], inplace=True)

    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}", file=sys.stderr)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert NCAA linescores → run_events")
    parser.add_argument("--linescores", default="data/raw/ncaa/linescores_2026.jsonl")
    parser.add_argument("--canonical", default="data/registries/canonical_teams_2026.csv")
    parser.add_argument("--out", default="data/processed/run_events_from_linescores.csv")
    args = parser.parse_args()

    build_run_events(
        Path(args.linescores),
        Path(args.canonical),
        Path(args.out),
    )
