"""
Build pitcher ratings, expected innings, and bullpen workload from pitching_lines_espn.csv.

Outputs (all in data/processed/):
  - pitcher_ratings.csv: pitcher_espn_id, canonical_id, season, role, n_games, IP, RA9, ERA, avg_IP_per_app
  - team_pitcher_strength.csv: canonical_id, season, sp_ra9, rp_ra9, relief_ip_share, league_ra9
  - bullpen_workload.csv: canonical_id, game_date, ip_last_1d, ip_last_3d, pc_last_1d (relief only, before that date)

Uses only rows with IP > 0 for RA9/ERA. Heavy shrinkage to league mean for small samples (Mack + Peabody).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LEAGUE_RA9_DEFAULT = 5.5
MIN_IP_FOR_RATING = 0.1


def main() -> int:
    parser = argparse.ArgumentParser(description="Build pitcher ratings and bullpen workload.")
    parser.add_argument(
        "--pitching-lines",
        type=Path,
        default=Path("data/processed/pitching_lines_espn.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    if not args.pitching_lines.exists():
        print(f"Missing {args.pitching_lines}")
        return 1

    df = pd.read_csv(args.pitching_lines)
    for col in ("IP", "ER", "R", "PC", "starter"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"])
    df = df[df["IP"].fillna(0) >= MIN_IP_FOR_RATING].copy()
    df["starter"] = df["starter"].fillna(False).astype(bool)

    # ----- Pitcher ratings (per pitcher, per season, SP vs RP) -----
    df["role"] = df["starter"].map({True: "SP", False: "RP"})
    g = df.groupby(["pitcher_espn_id", "canonical_id", "season", "role"], dropna=False)
    agg = g.agg(
        n_games=("event_id", "nunique"),
        IP=("IP", "sum"),
        ER=("ER", "sum"),
        R=("R", "sum"),
    ).reset_index()
    agg["RA9"] = 9 * agg["ER"] / agg["IP"].clip(lower=0.01)
    agg["ERA"] = 9 * agg["ER"] / agg["IP"].clip(lower=0.01)
    # Avg IP per appearance (for SP = expected innings when he starts)
    ip_per_app = df.groupby(["pitcher_espn_id", "canonical_id", "season", "role"], dropna=False)["IP"].mean().reset_index(name="avg_IP_per_app")
    agg = agg.merge(ip_per_app, on=["pitcher_espn_id", "canonical_id", "season", "role"], how="left")

    # Shrinkage to league: blend RA9 with league_ra9 by IP (more IP = less shrink) — Mack/Peabody style
    shrink_ip = 20.0  # after ~20 IP we're halfway to raw RA9
    agg["ra9"] = (agg["IP"] / (agg["IP"] + shrink_ip)) * agg["RA9"] + (shrink_ip / (agg["IP"] + shrink_ip)) * LEAGUE_RA9_DEFAULT
    pitcher_ratings = agg[
        ["pitcher_espn_id", "canonical_id", "season", "role", "n_games", "IP", "ER", "RA9", "ra9", "avg_IP_per_app"]
    ].copy()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pitcher_ratings.to_csv(args.out_dir / "pitcher_ratings.csv", index=False)
    print(f"Wrote pitcher_ratings.csv: {len(pitcher_ratings)} rows")

    # ----- Team SP/RP strength and relief share -----
    sp = df[df["starter"]].groupby(["canonical_id", "season"], dropna=False).agg(sp_ip=("IP", "sum"), sp_er=("ER", "sum")).reset_index()
    rp = df[~df["starter"]].groupby(["canonical_id", "season"], dropna=False).agg(rp_ip=("IP", "sum"), rp_er=("ER", "sum")).reset_index()
    team = sp.merge(rp, on=["canonical_id", "season"], how="outer").fillna(0)
    team["sp_ra9"] = 9 * team["sp_er"] / team["sp_ip"].clip(lower=0.01)
    team["rp_ra9"] = 9 * team["rp_er"] / team["rp_ip"].clip(lower=0.01)
    team["relief_ip_share"] = team["rp_ip"] / (team["sp_ip"] + team["rp_ip"]).clip(lower=0.01)
    team["league_ra9"] = LEAGUE_RA9_DEFAULT
    team = team[["canonical_id", "season", "sp_ra9", "rp_ra9", "relief_ip_share", "league_ra9"]]
    team.to_csv(args.out_dir / "team_pitcher_strength.csv", index=False)
    print(f"Wrote team_pitcher_strength.csv: {len(team)} rows")

    # ----- Bullpen workload: by team and date, relief IP/PC in last 1 and 3 days (before that date) -----
    rp_only = df[~df["starter"]].copy()
    rp_only["_date"] = rp_only["game_date"].dt.normalize()
    all_dates = sorted(rp_only["_date"].unique())
    rows = []
    for canonical_id in rp_only["canonical_id"].dropna().unique():
        if canonical_id == "":
            continue
        team_rp = rp_only[rp_only["canonical_id"] == canonical_id]
        for d in all_dates:
            d_ts = pd.Timestamp(d)
            d_minus_1 = d_ts - pd.Timedelta(days=1)
            d_minus_3 = d_ts - pd.Timedelta(days=3)
            mask_1 = team_rp["_date"] == d_minus_1
            mask_3 = (team_rp["_date"] >= d_minus_3) & (team_rp["_date"] < d_ts)
            ip_last_1d = float(team_rp.loc[mask_1, "IP"].sum())
            ip_last_3d = float(team_rp.loc[mask_3, "IP"].sum())
            pc_last_1d = float(team_rp.loc[mask_1, "PC"].sum())
            rows.append({
                "canonical_id": canonical_id,
                "game_date": d_ts.strftime("%Y-%m-%d"),
                "ip_last_1d": ip_last_1d,
                "ip_last_3d": ip_last_3d,
                "pc_last_1d": pc_last_1d,
            })
    workload = pd.DataFrame(rows)
    if not workload.empty:
        workload.to_csv(args.out_dir / "bullpen_workload.csv", index=False)
        print(f"Wrote bullpen_workload.csv: {len(workload)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
