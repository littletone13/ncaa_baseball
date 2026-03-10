"""
Compute data-driven bullpen quality metrics from pitcher appearances.

Uses actual reliever performance (ERA, WHIP, K rate) and workload distribution
to create a composite bullpen quality score per team per season.

After model fitting, this can also incorporate learned pitcher_ability posteriors
to create model-informed bullpen quality adjustments.

Output: data/processed/bullpen_quality.csv with columns:
  - team_canonical_id, season
  - n_relievers, total_ip, total_appearances
  - era, whip, k_per_bf, er_per_bf (raw stats)
  - bullpen_depth_score (z-score composite, higher = better)
  - bullpen_adj (log-scale adjustment for Stan model)

Usage:
  python3 scripts/compute_bullpen_quality.py
  python3 scripts/compute_bullpen_quality.py --posterior data/processed/run_event_posterior.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute bullpen quality metrics from pitcher appearances.",
    )
    parser.add_argument("--appearances", type=Path,
                        default=Path("data/processed/pitcher_appearances.csv"))
    parser.add_argument("--pitcher-index", type=Path,
                        default=Path("data/processed/run_event_pitcher_index.csv"))
    parser.add_argument("--posterior", type=Path, default=None,
                        help="Optional posterior CSV to incorporate learned pitcher abilities")
    parser.add_argument("--meta", type=Path,
                        default=Path("data/processed/run_event_fit_meta.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/processed/bullpen_quality.csv"))
    args = parser.parse_args()

    if not args.appearances.exists():
        print(f"Missing: {args.appearances}")
        return 1

    df = pd.read_csv(args.appearances, dtype=str)

    # Filter to relievers from NCAA (they have actual stats)
    relievers = df[(df["role"] == "reliever") & (df["source"] == "ncaa")].copy()
    print(f"Reliever appearances: {len(relievers)}")

    # Coerce numeric columns
    for col in ["ip", "h", "r", "er", "bb", "k", "bf"]:
        if col in relievers.columns:
            relievers[col] = pd.to_numeric(relievers[col], errors="coerce").fillna(0)

    # Derive season from game_date
    relievers["season"] = relievers["game_date"].str[:4].astype(int)

    # Group by team + season
    team_stats = relievers.groupby(["team_canonical_id", "season"]).agg(
        n_relievers=("pitcher_id", "nunique"),
        total_appearances=("pitcher_id", "count"),
        total_ip=("ip", "sum"),
        total_h=("h", "sum"),
        total_r=("r", "sum"),
        total_er=("er", "sum"),
        total_bb=("bb", "sum"),
        total_k=("k", "sum"),
        total_bf=("bf", "sum"),
    ).reset_index()

    # Compute rate stats (avoid div by zero)
    team_stats["era"] = np.where(
        team_stats["total_ip"] > 0,
        team_stats["total_er"] / team_stats["total_ip"] * 9,
        99.0,
    )
    team_stats["whip"] = np.where(
        team_stats["total_ip"] > 0,
        (team_stats["total_h"] + team_stats["total_bb"]) / team_stats["total_ip"],
        9.99,
    )
    team_stats["k_per_bf"] = np.where(
        team_stats["total_bf"] > 0,
        team_stats["total_k"] / team_stats["total_bf"],
        0.0,
    )
    team_stats["er_per_bf"] = np.where(
        team_stats["total_bf"] > 0,
        team_stats["total_er"] / team_stats["total_bf"],
        0.5,
    )
    team_stats["bb_per_bf"] = np.where(
        team_stats["total_bf"] > 0,
        team_stats["total_bb"] / team_stats["total_bf"],
        0.2,
    )

    # Only compute z-scores for teams with meaningful sample (>= 10 IP)
    mask = team_stats["total_ip"] >= 10
    print(f"Teams with >= 10 relief IP: {mask.sum()} / {len(team_stats)}")

    # Compute z-scores for each component (within season)
    # Note: for ERA, WHIP, er_per_bf, bb_per_bf: lower is better -> negate
    # For k_per_bf: higher is better -> keep as is
    # For n_relievers (depth): more is better -> keep as is, but diminishing returns
    for season in team_stats["season"].unique():
        smask = (team_stats["season"] == season) & mask
        if smask.sum() < 5:
            continue

        subset = team_stats.loc[smask]

        def zscore(col, negate=False):
            mean = subset[col].mean()
            std = subset[col].std()
            if std < 1e-8:
                return pd.Series(0.0, index=subset.index)
            z = (subset[col] - mean) / std
            return -z if negate else z

        z_era = zscore("era", negate=True)  # lower ERA = better
        z_whip = zscore("whip", negate=True)  # lower WHIP = better
        z_k = zscore("k_per_bf")  # higher K rate = better
        z_er = zscore("er_per_bf", negate=True)  # lower ER rate = better
        z_depth = zscore("n_relievers")  # more relievers = better (depth)

        # Composite: weighted average
        # ERA/ER rate dominate, K rate and depth secondary
        composite = 0.30 * z_era + 0.25 * z_er + 0.20 * z_k + 0.15 * z_whip + 0.10 * z_depth
        team_stats.loc[smask, "bullpen_depth_score"] = composite

    # Fill NaN with 0 (league average)
    team_stats["bullpen_depth_score"] = team_stats["bullpen_depth_score"].fillna(0.0)

    # ── Optional: incorporate learned pitcher abilities from posterior ──
    if args.posterior is not None and args.posterior.exists() and args.meta.exists():
        print(f"\nIncorporating learned pitcher abilities from {args.posterior}")
        draws_df = pd.read_csv(args.posterior)
        with open(args.meta) as f:
            meta = json.load(f)

        # Load pitcher index
        pi_df = pd.read_csv(args.pitcher_index, dtype=str)
        pid_to_idx: dict[str, int] = {}
        for _, r in pi_df.iterrows():
            pid = str(r.get("pitcher_espn_id", "")).strip()
            idx = int(r.get("pitcher_idx", 0))
            if pid and pid.lower() != "unknown":
                pid_to_idx[pid] = idx

        # Compute mean posterior pitcher_ability for each pitcher
        n_pitchers = meta["N_pitchers"]
        mean_ability = {}
        for i in range(1, n_pitchers + 1):
            col = f"pitcher_ability[{i}]"
            if col in draws_df.columns:
                mean_ability[i] = draws_df[col].mean()

        # For each team-season, compute IP-weighted mean reliever ability
        for i, row in team_stats.iterrows():
            cid = row["team_canonical_id"]
            season = row["season"]
            team_relievers = relievers[
                (relievers["team_canonical_id"] == cid) &
                (relievers["season"] == season)
            ]
            if len(team_relievers) == 0:
                continue

            total_ip = 0.0
            weighted_ability = 0.0
            for _, r in team_relievers.iterrows():
                pid = str(r.get("pitcher_id", "")).strip()
                ip = float(r.get("ip", 0) or 0)
                if ip <= 0:
                    continue
                idx = pid_to_idx.get(pid, 0)
                if idx > 0 and idx in mean_ability:
                    weighted_ability += ip * mean_ability[idx]
                    total_ip += ip

            if total_ip > 0:
                team_stats.loc[i, "model_bullpen_ability"] = weighted_ability / total_ip

        if "model_bullpen_ability" in team_stats.columns:
            # Combine statistical and model-based bullpen quality
            mb = team_stats["model_bullpen_ability"].fillna(0)
            # pitcher_ability is on log-rate scale (negative = better pitcher = fewer runs)
            # So negative model_bullpen_ability = better bullpen
            # Normalize to z-score scale
            mb_mean = mb[mb != 0].mean() if (mb != 0).any() else 0
            mb_std = mb[mb != 0].std() if (mb != 0).any() else 1
            mb_z = (mb - mb_mean) / max(mb_std, 1e-8)
            # Negate: negative ability (good pitching) -> positive quality score
            team_stats["model_bullpen_z"] = -mb_z

            # Blend: 60% model, 40% statistical
            team_stats["bullpen_depth_score"] = (
                0.6 * team_stats["model_bullpen_z"].fillna(0) +
                0.4 * team_stats["bullpen_depth_score"]
            )
            print(f"Blended model-based ({(mb != 0).sum()} teams) with statistical bullpen scores")

    # Save
    out_cols = [
        "team_canonical_id", "season",
        "n_relievers", "total_appearances", "total_ip",
        "era", "whip", "k_per_bf", "er_per_bf",
        "bullpen_depth_score",
    ]
    if "model_bullpen_ability" in team_stats.columns:
        out_cols.append("model_bullpen_ability")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    team_stats[out_cols].to_csv(args.out, index=False)

    # Summary
    print(f"\n=== BULLPEN QUALITY ===")
    print(f"Teams: {len(team_stats)}")
    for season in sorted(team_stats["season"].unique()):
        s = team_stats[team_stats["season"] == season]
        print(f"  {season}: {len(s)} teams, "
              f"mean ERA={s['era'].mean():.2f}, "
              f"mean depth score={s['bullpen_depth_score'].mean():.3f}")

    # Top/bottom 5
    latest = team_stats[team_stats["season"] == team_stats["season"].max()].copy()
    latest = latest.sort_values("bullpen_depth_score", ascending=False)
    print(f"\nTop 5 bullpens ({latest['season'].iloc[0]}):")
    for _, r in latest.head(5).iterrows():
        print(f"  {r['team_canonical_id']}: score={r['bullpen_depth_score']:.3f}, "
              f"ERA={r['era']:.2f}, {r['n_relievers']} relievers, {r['total_ip']:.0f} IP")
    print(f"Bottom 5 bullpens:")
    for _, r in latest.tail(5).iterrows():
        print(f"  {r['team_canonical_id']}: score={r['bullpen_depth_score']:.3f}, "
              f"ERA={r['era']:.2f}, {r['n_relievers']} relievers, {r['total_ip']:.0f} IP")

    print(f"\nOutput: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
