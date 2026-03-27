"""
Run or print projection/simulation commands from today's starter table.

Primary integration:
- project_game.py (uses --home-sp-id / --away-sp-id ESPN ids)

Optional integration:
- simulate_run_event_game.py command construction using --home-pitcher / --away-pitcher
  with pitcher_idx from run_event_pitcher_index.csv.

Usage:
  python3 scripts/run_today_with_starters.py --starters-csv data/processed/todays_starters.csv
  python3 scripts/run_today_with_starters.py --mode both --execute
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def parse_extra(s: str) -> list[str]:
    s = (s or "").strip()
    if not s:
        return []
    return shlex.split(s)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run/print today's projection commands using starter table.")
    parser.add_argument("--starters-csv", type=Path, default=Path("data/processed/todays_starters.csv"))
    parser.add_argument("--mode", choices=["project", "simulate", "both"], default="project")
    parser.add_argument("--execute", action="store_true", help="Execute commands (default: print only)")
    parser.add_argument("--project-script", type=Path, default=Path("scripts/project_game.py"))
    parser.add_argument("--simulate-script", type=Path, default=Path("scripts/simulate_run_event_game.py"))
    parser.add_argument("--project-extra", default="", help='Extra args appended to each project command, e.g. "--use-pitchers --season 2026"')
    parser.add_argument("--simulate-extra", default="", help='Extra args appended to each simulate command')
    parser.add_argument("--max-games", type=int, default=0, help="Process only first N rows (0 = all)")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Skip games below this minimum side confidence")
    args = parser.parse_args()

    if not args.starters_csv.exists():
        print(f"Missing starters CSV: {args.starters_csv}")
        return 1

    df = pd.read_csv(args.starters_csv, dtype=str).fillna("")
    required = [
        "game_date",
        "event_id",
        "home_team_name",
        "away_team_name",
        "home_pitcher_espn_id",
        "away_pitcher_espn_id",
        "home_pitcher_idx",
        "away_pitcher_idx",
        "home_starter_confidence",
        "away_starter_confidence",
    ]
    for c in required:
        if c not in df.columns:
            df[c] = ""

    if args.max_games > 0:
        df = df.head(args.max_games)

    project_extra = parse_extra(args.project_extra)
    simulate_extra = parse_extra(args.simulate_extra)

    n_total = 0
    n_ran = 0
    n_fail = 0

    for _, row in df.iterrows():
        try:
            h_conf = float(row.get("home_starter_confidence") or 0.0)
        except ValueError:
            h_conf = 0.0
        try:
            a_conf = float(row.get("away_starter_confidence") or 0.0)
        except ValueError:
            a_conf = 0.0
        if min(h_conf, a_conf) < args.min_confidence:
            continue

        game_date = row.get("game_date", "")
        home_team = row.get("home_team_name", "")
        away_team = row.get("away_team_name", "")
        event_id = row.get("event_id", "")
        home_sp_id = row.get("home_pitcher_espn_id", "").strip()
        away_sp_id = row.get("away_pitcher_espn_id", "").strip()
        home_pidx = row.get("home_pitcher_idx", "").strip() or "0"
        away_pidx = row.get("away_pitcher_idx", "").strip() or "0"

        if args.mode in {"project", "both"}:
            cmd = [
                sys.executable,
                str(args.project_script),
                "--team-a",
                home_team,
                "--team-b",
                away_team,
                "--game-date",
                game_date,
                "--use-pitchers",
            ]
            if home_sp_id:
                cmd += ["--home-sp-id", home_sp_id]
            if away_sp_id:
                cmd += ["--away-sp-id", away_sp_id]
            cmd += project_extra

            n_total += 1
            if args.execute:
                print(f"\n[project] event_id={event_id}  {home_team} vs {away_team}")
                print(shell_join(cmd))
                proc = subprocess.run(cmd, check=False)
                n_ran += 1
                if proc.returncode != 0:
                    n_fail += 1
            else:
                print(shell_join(cmd))

        if args.mode in {"simulate", "both"}:
            cmd = [
                sys.executable,
                str(args.simulate_script),
                "--home-pitcher",
                home_pidx,
                "--away-pitcher",
                away_pidx,
            ]
            cmd += simulate_extra

            n_total += 1
            if args.execute:
                if not args.simulate_script.exists():
                    print(f"\n[simulate] skip missing script: {args.simulate_script}")
                    n_fail += 1
                else:
                    print(f"\n[simulate] event_id={event_id}  {home_team} vs {away_team}")
                    print(shell_join(cmd))
                    proc = subprocess.run(cmd, check=False)
                    n_ran += 1
                    if proc.returncode != 0:
                        n_fail += 1
            else:
                print(shell_join(cmd))

    if args.execute:
        print(f"\nDone. commands_total={n_total} commands_ran={n_ran} failures={n_fail}")
        return 0 if n_fail == 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
