from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_team_game_table(games: pd.DataFrame) -> pd.DataFrame:
    required = {
        "academic_year",
        "game_date",
        "contest_id",
        "away_team_ncaa_id",
        "away_team_name",
        "home_team_ncaa_id",
        "home_team_name",
        "away_runs",
        "home_runs",
        "neutral_site",
    }
    missing = sorted(required - set(games.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    games = games.copy()
    games["away_runs"] = pd.to_numeric(games["away_runs"], errors="coerce")
    games["home_runs"] = pd.to_numeric(games["home_runs"], errors="coerce")

    away = pd.DataFrame(
        {
            "academic_year": games["academic_year"],
            "game_date": games["game_date"],
            "contest_id": games["contest_id"],
            "team_ncaa_id": games["away_team_ncaa_id"],
            "team_name": games["away_team_name"],
            "opponent_ncaa_id": games["home_team_ncaa_id"],
            "opponent_name": games["home_team_name"],
            "is_home": 0,
            "neutral_site": games["neutral_site"],
            "runs_for": games["away_runs"],
            "runs_against": games["home_runs"],
        }
    )
    home = pd.DataFrame(
        {
            "academic_year": games["academic_year"],
            "game_date": games["game_date"],
            "contest_id": games["contest_id"],
            "team_ncaa_id": games["home_team_ncaa_id"],
            "team_name": games["home_team_name"],
            "opponent_ncaa_id": games["away_team_ncaa_id"],
            "opponent_name": games["away_team_name"],
            "is_home": 1,
            "neutral_site": games["neutral_site"],
            "runs_for": games["home_runs"],
            "runs_against": games["away_runs"],
        }
    )
    out = pd.concat([away, home], ignore_index=True)

    # Mark result only when we have scores.
    out["has_score"] = (~out["runs_for"].isna()) & (~out["runs_against"].isna())
    out["win"] = out["has_score"] & (out["runs_for"] > out["runs_against"])
    out["loss"] = out["has_score"] & (out["runs_for"] < out["runs_against"])
    out["tie"] = out["has_score"] & (out["runs_for"] == out["runs_against"])
    return out


def summarize_team_stats(team_games: pd.DataFrame) -> pd.DataFrame:
    g = team_games.groupby(["academic_year", "team_ncaa_id"], dropna=False)
    out = g.agg(
        games=("contest_id", "count"),
        games_with_score=("has_score", "sum"),
        wins=("win", "sum"),
        losses=("loss", "sum"),
        ties=("tie", "sum"),
        runs_for=("runs_for", "sum"),
        runs_against=("runs_against", "sum"),
    ).reset_index()

    # Choose a canonical display name per (academic_year, team_id) based on most-frequent observed name.
    name_counts = (
        team_games.groupby(["academic_year", "team_ncaa_id", "team_name"], dropna=False)["contest_id"]
        .count()
        .reset_index(name="n")
    )
    canonical = (
        name_counts.sort_values(
            ["academic_year", "team_ncaa_id", "n", "team_name"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["academic_year", "team_ncaa_id"])
        .rename(columns={"team_name": "team_name_canonical"})
        .loc[:, ["academic_year", "team_ncaa_id", "team_name_canonical"]]
    )
    variant_counts = (
        name_counts.groupby(["academic_year", "team_ncaa_id"], dropna=False)["team_name"]
        .nunique(dropna=False)
        .reset_index(name="n_name_variants")
    )
    out = out.merge(canonical, on=["academic_year", "team_ncaa_id"], how="left").merge(
        variant_counts, on=["academic_year", "team_ncaa_id"], how="left"
    )
    out = out.rename(columns={"team_name_canonical": "team_name"})
    out["run_diff"] = out["runs_for"] - out["runs_against"]
    out["rpg_for"] = out["runs_for"] / out["games_with_score"].where(out["games_with_score"] != 0, pd.NA)
    out["rpg_against"] = out["runs_against"] / out["games_with_score"].where(out["games_with_score"] != 0, pd.NA)
    return out.sort_values(["academic_year", "team_name", "team_ncaa_id"]).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Build simple team-level stats from scoreboard game results (runs only).")
    p.add_argument("--games-csv", type=Path, required=True, help="Input games CSV (from scrape_ncaa_scoreboard.py)")
    p.add_argument("--out-team-games", type=Path, default=None, help="Optional output: team-game table CSV")
    p.add_argument("--out-team-stats", type=Path, default=None, help="Output: team-season stats CSV")
    p.add_argument(
        "--out-name-variants",
        type=Path,
        default=None,
        help="Optional output: per-team observed name variants CSV (debug/cleaning)",
    )
    args = p.parse_args()

    games = pd.read_csv(args.games_csv)
    team_games = build_team_game_table(games)

    out_stats = args.out_team_stats or args.games_csv.with_name(
        args.games_csv.name.replace("games_", "team_stats_")
    )
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    summarize_team_stats(team_games).to_csv(out_stats, index=False)
    print(f"Wrote team stats -> {out_stats}")

    if args.out_team_games is not None:
        args.out_team_games.parent.mkdir(parents=True, exist_ok=True)
        team_games.to_csv(args.out_team_games, index=False)
        print(f"Wrote team-games -> {args.out_team_games}")

    if args.out_name_variants is not None:
        variants = (
            team_games.groupby(["academic_year", "team_ncaa_id", "team_name"], dropna=False)["contest_id"]
            .count()
            .reset_index(name="n_games")
            .sort_values(["academic_year", "team_ncaa_id", "n_games", "team_name"], ascending=[True, True, False, True])
        )
        args.out_name_variants.parent.mkdir(parents=True, exist_ok=True)
        variants.to_csv(args.out_name_variants, index=False)
        print(f"Wrote name variants -> {args.out_name_variants}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
