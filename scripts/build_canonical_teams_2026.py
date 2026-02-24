"""
Build a single canonical team registry for 2026 from NCAA D1 list + manual crosswalk.

No fuzzy matching: every row comes from ncaa_d1_teams_2026.csv; optional fields
come from name_crosswalk_manual_2026.csv. canonical_id defaults to NCAA_<id>
when not in crosswalk.

Usage:
  python3 scripts/build_canonical_teams_2026.py
  python3 scripts/build_canonical_teams_2026.py --ncaa-csv data/registries/ncaa_d1_teams_2026.csv --crosswalk data/registries/name_crosswalk_manual_2026.csv --out data/registries/canonical_teams_2026.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build canonical_teams_2026.csv from NCAA D1 list and manual crosswalk."
    )
    parser.add_argument(
        "--ncaa-csv",
        type=Path,
        default=Path("data/registries/ncaa_d1_teams_2026.csv"),
        help="NCAA D1 teams CSV (academic_year, ncaa_teams_id, team_name, conference, conference_id, ...)",
    )
    parser.add_argument(
        "--crosswalk",
        type=Path,
        default=Path("data/registries/name_crosswalk_manual_2026.csv"),
        help="Manual crosswalk CSV (ncaa_teams_id, canonical_team_id, odds_api_team_name, ...)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/registries/canonical_teams_2026.csv"),
        help="Output canonical registry CSV",
    )
    args = parser.parse_args()

    ncaa = pd.read_csv(args.ncaa_csv)
    required_ncaa = {"academic_year", "ncaa_teams_id", "team_name", "conference", "conference_id"}
    missing = required_ncaa - set(ncaa.columns)
    if missing:
        raise SystemExit(f"{args.ncaa_csv} missing columns: {sorted(missing)}")

    if ncaa["ncaa_teams_id"].duplicated().any():
        dupes = ncaa.loc[ncaa["ncaa_teams_id"].duplicated(keep=False), "ncaa_teams_id"].unique().tolist()
        raise SystemExit(f"Duplicate ncaa_teams_id in NCAA CSV: {dupes[:20]}")

    crosswalk = pd.read_csv(args.crosswalk)
    if "ncaa_teams_id" not in crosswalk.columns:
        raise SystemExit(f"{args.crosswalk} must have column ncaa_teams_id")
    xw_cols = ["ncaa_teams_id", "canonical_team_id", "odds_api_team_name", "baseballr_team_name", "notes"]
    xw_use = [c for c in xw_cols if c in crosswalk.columns]
    crosswalk = crosswalk[xw_use].drop_duplicates(subset=["ncaa_teams_id"], keep="first")

    merged = ncaa.merge(crosswalk, on="ncaa_teams_id", how="left")

    # canonical_id: from crosswalk or default NCAA_<id>
    merged["canonical_id"] = merged.get("canonical_team_id", pd.Series(dtype=object))
    merged["canonical_id"] = merged["canonical_id"].fillna("").astype(str).str.strip()
    missing_canon = merged["canonical_id"] == ""
    merged.loc[missing_canon, "canonical_id"] = "NCAA_" + merged.loc[missing_canon, "ncaa_teams_id"].astype(str)
    merged = merged.drop(columns=["canonical_team_id"], errors="ignore")

    merged["odds_api_name"] = merged.get("odds_api_team_name", pd.Series("", index=merged.index)).fillna("").astype(str).str.strip()
    merged = merged.drop(columns=["odds_api_team_name"], errors="ignore")

    merged["baseballr_team_name"] = merged.get("baseballr_team_name", pd.Series("", index=merged.index)).fillna("").astype(str).str.strip()
    merged["baseballr_team_id"] = ""
    merged["baseballr_season_id"] = ""
    merged["notes"] = merged.get("notes", pd.Series("", index=merged.index)).fillna("").astype(str).str.strip()

    out_cols = [
        "academic_year",
        "ncaa_teams_id",
        "team_name",
        "conference",
        "conference_id",
        "canonical_id",
        "odds_api_name",
        "baseballr_team_id",
        "baseballr_season_id",
        "baseballr_team_name",
        "notes",
    ]
    out = merged[out_cols].copy()

    # Validate: no duplicate canonical_id (among non-empty)
    canon = out["canonical_id"].astype(str).str.strip()
    canon_counts = canon[canon != ""].value_counts()
    dupes_canon = canon_counts[canon_counts > 1].index.tolist()
    if dupes_canon:
        raise SystemExit(f"Duplicate canonical_id in output (from crosswalk): {dupes_canon[:20]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} teams -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
