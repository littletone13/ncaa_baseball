from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLS = [
    "academic_year",
    "division",
    "sport_code",
    "game_date",
    "contest_id",
    "away_team_ncaa_id",
    "away_team_name",
    "home_team_ncaa_id",
    "home_team_name",
    "away_runs",
    "home_runs",
    "neutral_site",
    "attendance",
]


def _load_games(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")
    return df


def _load_team_ids(path: Path) -> set[int]:
    df = pd.read_csv(path)
    if "ncaa_teams_id" not in df.columns:
        raise ValueError(f"{path}: expected column ncaa_teams_id")
    return set(int(x) for x in df["ncaa_teams_id"].dropna().astype(int).tolist())


def audit(games: pd.DataFrame, *, known_team_ids: set[int] | None) -> dict[str, object]:
    out: dict[str, object] = {}

    out["n_rows"] = int(len(games))
    out["n_unique_contest_id"] = int(games["contest_id"].nunique(dropna=True))
    out["n_duplicate_contest_id_rows"] = int(out["n_rows"] - out["n_unique_contest_id"])

    # Types / parsing
    bad_dates = 0
    try:
        dt = pd.to_datetime(games["game_date"], format="%Y-%m-%d", errors="coerce")
        bad_dates = int(dt.isna().sum())
    except Exception:
        bad_dates = int(len(games))
    out["n_bad_game_date"] = bad_dates

    same_team = (games["away_team_ncaa_id"] == games["home_team_ncaa_id"]).fillna(False)
    out["n_away_equals_home"] = int(same_team.sum())

    # Scores
    away_runs = pd.to_numeric(games["away_runs"], errors="coerce")
    home_runs = pd.to_numeric(games["home_runs"], errors="coerce")
    out["n_missing_scores"] = int((away_runs.isna() | home_runs.isna()).sum())
    out["n_negative_runs"] = int(((away_runs < 0) | (home_runs < 0)).fillna(False).sum())

    # Names
    out["n_missing_team_names"] = int(
        ((games["away_team_name"].isna()) | (games["away_team_name"].astype(str).str.strip() == "")).sum()
        + ((games["home_team_name"].isna()) | (games["home_team_name"].astype(str).str.strip() == "")).sum()
    )

    # Team coverage
    team_ids = pd.concat([games["away_team_ncaa_id"], games["home_team_ncaa_id"]], ignore_index=True)
    team_ids = pd.to_numeric(team_ids, errors="coerce").dropna().astype(int)
    out["n_unique_team_ids"] = int(team_ids.nunique())

    if known_team_ids is not None:
        unknown = sorted(set(team_ids.tolist()) - known_team_ids)
        out["n_unknown_team_ids_vs_registry"] = int(len(unknown))
        out["unknown_team_ids_vs_registry"] = unknown[:50]
        if len(unknown) > 50:
            out["unknown_team_ids_vs_registry_truncated"] = True
    else:
        out["n_unknown_team_ids_vs_registry"] = None
        out["unknown_team_ids_vs_registry"] = None

    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Audit a scoreboard-derived games CSV for basic integrity issues.")
    p.add_argument("--games-csv", type=Path, required=True, help="Input games CSV (from scrape_ncaa_scoreboard.py)")
    p.add_argument(
        "--team-registry-csv",
        type=Path,
        default=None,
        help=(
            "NCAA D1 team registry CSV for team-id validation. "
            "Default: infer from games' academic_year as data/registries/ncaa_d1_teams_<year>.csv if present."
        ),
    )
    p.add_argument("--json-out", type=Path, default=None, help="Write audit JSON here (optional)")
    args = p.parse_args()

    games = _load_games(args.games_csv)
    inferred_year: int | None = None
    try:
        years = sorted({int(x) for x in pd.to_numeric(games["academic_year"], errors="coerce").dropna().astype(int).tolist()})
        inferred_year = years[0] if len(years) == 1 else None
    except Exception:
        inferred_year = None

    team_registry_csv = args.team_registry_csv
    if team_registry_csv is None and inferred_year is not None:
        candidate = Path("data/registries") / f"ncaa_d1_teams_{inferred_year}.csv"
        if candidate.exists():
            team_registry_csv = candidate
        else:
            team_registry_csv = None

    known_team_ids = None
    if team_registry_csv is not None and team_registry_csv.exists():
        known_team_ids = _load_team_ids(team_registry_csv)

    report = audit(games, known_team_ids=known_team_ids)
    report["games_csv"] = str(args.games_csv)
    report["team_registry_csv"] = str(team_registry_csv) if team_registry_csv is not None and team_registry_csv.exists() else None
    report["inferred_academic_year"] = inferred_year

    blob = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(blob + "\n", encoding="utf-8")
        print(f"Wrote audit -> {args.json_out}")
    else:
        print(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
