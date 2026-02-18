from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build a deterministic team registry (id->name) from a scoreboard games CSV."
    )
    p.add_argument("--games-csv", type=Path, required=True, help="Input games CSV (from scrape_ncaa_scoreboard.py)")
    p.add_argument("--out", type=Path, default=None, help="Output CSV path (default: data/registries/scoreboard_teams_<year>.csv)")
    args = p.parse_args()

    games = pd.read_csv(args.games_csv)
    if "academic_year" not in games.columns:
        raise ValueError("games-csv missing academic_year")
    years = sorted({int(x) for x in pd.to_numeric(games["academic_year"], errors="coerce").dropna().astype(int).tolist()})
    academic_year = years[0] if len(years) == 1 else None

    away = games[["away_team_ncaa_id", "away_team_name"]].rename(
        columns={"away_team_ncaa_id": "team_ncaa_id", "away_team_name": "team_name"}
    )
    home = games[["home_team_ncaa_id", "home_team_name"]].rename(
        columns={"home_team_ncaa_id": "team_ncaa_id", "home_team_name": "team_name"}
    )
    teams = pd.concat([away, home], ignore_index=True)
    teams["team_ncaa_id"] = pd.to_numeric(teams["team_ncaa_id"], errors="coerce").astype("Int64")
    teams["team_name"] = teams["team_name"].astype(str).str.strip()

    name_counts = (
        teams.dropna(subset=["team_ncaa_id"])
        .groupby(["team_ncaa_id", "team_name"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    canonical = (
        name_counts.sort_values(["team_ncaa_id", "n", "team_name"], ascending=[True, False, True])
        .drop_duplicates(["team_ncaa_id"])
        .rename(columns={"team_name": "team_name_canonical"})
        .loc[:, ["team_ncaa_id", "team_name_canonical"]]
    )
    variants = (
        name_counts.groupby(["team_ncaa_id"], dropna=False)["team_name"]
        .nunique(dropna=False)
        .reset_index(name="n_name_variants")
    )
    out_df = canonical.merge(variants, on="team_ncaa_id", how="left")
    if academic_year is not None:
        out_df.insert(0, "academic_year", academic_year)

    out = args.out
    if out is None:
        if academic_year is None:
            raise SystemExit("Multiple academic_year values found; provide --out explicitly.")
        out = Path("data/registries") / f"scoreboard_teams_{academic_year}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Wrote {len(out_df)} teams -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

