from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Initialize a manual/deterministic team-name crosswalk template (no fuzzy matching). "
            "This creates a CSV you can fill in by hand."
        )
    )
    p.add_argument(
        "--ncaa-teams-csv",
        type=Path,
        default=Path("data/registries/ncaa_d1_teams_2026.csv"),
        help="Input NCAA D1 teams registry CSV",
    )
    p.add_argument(
        "--canonical-teams-csv",
        type=Path,
        default=Path("data/registries/teams.csv"),
        help="Canonical (curated) team registry CSV (optional exact-match enrichment)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/registries/name_crosswalk_manual_2026.csv"),
        help="Output template CSV",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists")
    args = p.parse_args()

    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {args.out} (use --overwrite)")

    ncaa_rows = _read_csv(args.ncaa_teams_csv)

    canonical_by_school: dict[str, dict[str, str]] = {}
    if args.canonical_teams_csv.exists():
        for r in _read_csv(args.canonical_teams_csv):
            school = (r.get("school") or "").strip()
            if school and school not in canonical_by_school:
                canonical_by_school[school] = r

    out_fields = [
        "ncaa_teams_id",
        "ncaa_team_name",
        "conference",
        "canonical_team_id",
        "canonical_school",
        "odds_api_team_name",
        "baseballr_team_name",
        "notes",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in ncaa_rows:
            ncaa_id = (r.get("ncaa_teams_id") or "").strip()
            ncaa_name = (r.get("team_name") or "").strip()
            conf = (r.get("conference") or "").strip()

            canonical = canonical_by_school.get(ncaa_name)
            w.writerow(
                {
                    "ncaa_teams_id": ncaa_id,
                    "ncaa_team_name": ncaa_name,
                    "conference": conf,
                    "canonical_team_id": (canonical or {}).get("id", ""),
                    "canonical_school": (canonical or {}).get("school", ""),
                    "odds_api_team_name": "",
                    "baseballr_team_name": "",
                    "notes": "",
                }
            )

    print(f"Wrote manual crosswalk template -> {args.out}")
    print("Rules: fill columns manually; do NOT use fuzzy matching.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

