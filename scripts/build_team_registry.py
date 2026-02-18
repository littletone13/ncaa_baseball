from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from ncaa_baseball.teams import parse_teams_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical team registry from teams YAML.")
    parser.add_argument("yaml_path", type=Path, help="Path to teams_baseball.yaml")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/registries/teams.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    teams = parse_teams_yaml(args.yaml_path.read_text(encoding="utf-8"))
    df = pd.DataFrame([dataclasses.asdict(t) for t in teams]).sort_values(
        ["conference", "school", "id"], kind="stable"
    )
    df.reset_index(drop=True, inplace=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} teams -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
