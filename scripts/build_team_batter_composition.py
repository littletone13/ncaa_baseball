#!/usr/bin/env python3
"""
build_team_batter_composition.py — Compute per-team batter handedness composition.

Reads sidearm_rosters.csv (8K+ players with bats field) and computes
the fraction of RHB/LHB/switch hitters per team. This replaces the
blanket 75% RHB assumption in the platoon model with actual team data.

Output: data/processed/team_batter_handedness.csv
  canonical_id, n_batters, pct_rhb, pct_lhb, pct_switch, effective_rhb_frac

effective_rhb_frac treats switch hitters as 50/50 (they bat from both sides):
  effective_rhb = (n_rhb + 0.5 * n_switch) / n_total

Usage:
  python3 scripts/build_team_batter_composition.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401

# League averages (computed from full sidearm data as fallback)
LEAGUE_AVG_RHB = 0.684
LEAGUE_AVG_LHB = 0.289
LEAGUE_AVG_SWITCH = 0.027
MIN_BATTERS_FOR_TEAM_SPECIFIC = 5


def build_composition(
    sidearm_csv: Path = Path("data/processed/sidearm_rosters.csv"),
    out_csv: Path = Path("data/processed/team_batter_handedness.csv"),
) -> pd.DataFrame:
    """Build per-team batter handedness composition."""

    sr = pd.read_csv(sidearm_csv, dtype=str)
    print(f"Loaded {len(sr)} players from {sidearm_csv}", file=sys.stderr)

    # Filter to non-pitchers only (position players)
    # Exclude: P, LHP, RHP, pitcher, Pitcher
    pitcher_mask = sr["position"].fillna("").str.upper().str.contains(
        r"^P$|^LHP$|^RHP$|PITCHER", regex=True
    )
    batters = sr[~pitcher_mask].copy()
    print(f"  Non-pitchers: {len(batters)}", file=sys.stderr)

    # Filter to batters with known handedness
    batters["bats"] = batters["bats"].fillna("").str.strip().str.upper()
    # Normalize: B = switch, S = switch
    batters["bats"] = batters["bats"].replace({"B": "S"})
    known = batters[batters["bats"].isin(["R", "L", "S"])]
    print(f"  With known bats: {len(known)} ({len(known)/len(batters)*100:.0f}%)", file=sys.stderr)

    # Compute league averages from this data
    league_n = len(known)
    league_rhb = (known["bats"] == "R").sum() / league_n
    league_lhb = (known["bats"] == "L").sum() / league_n
    league_switch = (known["bats"] == "S").sum() / league_n
    print(f"  League avg: RHB={league_rhb:.3f}, LHB={league_lhb:.3f}, Switch={league_switch:.3f}",
          file=sys.stderr)

    # Per-team composition
    rows = []
    for cid, grp in known.groupby("canonical_id"):
        n = len(grp)
        n_rhb = (grp["bats"] == "R").sum()
        n_lhb = (grp["bats"] == "L").sum()
        n_switch = (grp["bats"] == "S").sum()

        if n < MIN_BATTERS_FOR_TEAM_SPECIFIC:
            # Too few batters — use league average
            pct_rhb = league_rhb
            pct_lhb = league_lhb
            pct_switch = league_switch
        else:
            pct_rhb = n_rhb / n
            pct_lhb = n_lhb / n
            pct_switch = n_switch / n

        # Effective RHB fraction: switch hitters split 50/50
        effective_rhb = pct_rhb + 0.5 * pct_switch

        rows.append({
            "canonical_id": cid,
            "n_batters": n,
            "n_rhb": n_rhb,
            "n_lhb": n_lhb,
            "n_switch": n_switch,
            "pct_rhb": round(pct_rhb, 4),
            "pct_lhb": round(pct_lhb, 4),
            "pct_switch": round(pct_switch, 4),
            "effective_rhb_frac": round(effective_rhb, 4),
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("canonical_id")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)

    # Summary stats
    print(f"\nWrote {len(result)} teams → {out_csv}", file=sys.stderr)
    print(f"  effective_rhb_frac range: {result['effective_rhb_frac'].min():.3f} – "
          f"{result['effective_rhb_frac'].max():.3f}", file=sys.stderr)
    print(f"  Mean: {result['effective_rhb_frac'].mean():.3f}, "
          f"Std: {result['effective_rhb_frac'].std():.3f}", file=sys.stderr)

    # Show extremes
    most_rhb = result.nlargest(5, "effective_rhb_frac")
    most_lhb = result.nsmallest(5, "effective_rhb_frac")
    print(f"\n  Most RHB-heavy teams:", file=sys.stderr)
    for _, r in most_rhb.iterrows():
        print(f"    {r['canonical_id']}: {r['effective_rhb_frac']:.1%} RHB ({r['n_batters']} batters)",
              file=sys.stderr)
    print(f"\n  Most LHB-heavy teams:", file=sys.stderr)
    for _, r in most_lhb.iterrows():
        print(f"    {r['canonical_id']}: {r['effective_rhb_frac']:.1%} RHB ({r['n_batters']} batters)",
              file=sys.stderr)

    return result


if __name__ == "__main__":
    build_composition()
