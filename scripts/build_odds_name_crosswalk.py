from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _extract_team_names(events_payload: object) -> list[str]:
    if not isinstance(events_payload, dict):
        raise ValueError("Expected events payload to be a JSON object.")
    data = events_payload.get("data")
    if not isinstance(data, list):
        raise ValueError("Expected events payload to have a 'data' list.")

    names: list[str] = []
    for ev in data:
        if not isinstance(ev, dict):
            continue
        for k in ("home_team", "away_team"):
            v = ev.get(k)
            if isinstance(v, str) and v.strip():
                names.append(v.strip())
    return sorted(set(names))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic Odds API team-name crosswalk template (no fuzzy matching)."
    )
    parser.add_argument("--events-json", type=Path, required=True, help="Odds API events JSON (historical).")
    parser.add_argument(
        "--teams-csv",
        type=Path,
        default=Path("data/registries/teams.csv"),
        help="Canonical teams registry CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/registries/name_crosswalk_odds_api.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--source", default="odds_api", help="Source label for the crosswalk")
    args = parser.parse_args()

    events = json.loads(args.events_json.read_text(encoding="utf-8"))
    odds_names = _extract_team_names(events)

    teams = pd.read_csv(args.teams_csv)
    required = {"id", "school"}
    missing = sorted(required - set(teams.columns))
    if missing:
        raise SystemExit(f"{args.teams_csv} missing columns: {missing}")

    by_school = { _norm(str(row.school)): str(row.id) for row in teams.itertuples(index=False) }

    rows: list[dict[str, str]] = []
    for name in odds_names:
        norm = _norm(name)
        team_id = by_school.get(norm, "")
        rows.append(
            {
                "source": args.source,
                "source_name": name,
                "team_id": team_id,
                "match_type": "exact_school" if team_id else "",
                "notes": "",
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["source_name"], kind="stable").reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

