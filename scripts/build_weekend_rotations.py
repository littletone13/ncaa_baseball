"""
Build projected weekend rotations from historical starter patterns.

Analyzes each team's Friday/Saturday/Sunday starter history across 2026 season
to project who will start each day of the upcoming weekend. In college baseball:
  - Friday = ace (#1 starter)
  - Saturday = #2 starter
  - Sunday = #3 starter

Teams typically maintain consistent weekend rotations unless injured or shuffled.
This script identifies each team's most recent rotation pattern and projects forward.

Uses:
  - ESPN boxscore data (starter flags + game dates)
  - NCAA boxscore data (starter flags + game dates)
  - pitcher_appearances.csv for unified pitcher data

Output:
  data/processed/weekend_rotations.csv — projected starters for next weekend
  Columns: canonical_id, team_name, day (fri/sat/sun), pitcher_name, pitcher_id,
           confidence, last_started_date, starts_this_role

Usage:
  python3 scripts/build_weekend_rotations.py
  python3 scripts/build_weekend_rotations.py --weekend 2026-03-13  # specific Friday date
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import _bootstrap  # noqa: F401
from ncaa_baseball.phase1 import (
    build_odds_name_to_canonical,
    load_canonical_teams,
    resolve_odds_teams,
)


def get_day_of_week(d: str) -> int:
    """Return 0=Mon .. 6=Sun for a YYYY-MM-DD date string."""
    return datetime.strptime(d, "%Y-%m-%d").weekday()


def day_label(dow: int) -> str:
    return {4: "fri", 5: "sat", 6: "sun"}.get(dow, f"dow{dow}")


def load_espn_starters(espn_dir: Path, seasons: list[str]) -> list[dict]:
    """Load starters from ESPN JSONL files."""
    rows = []
    for season in seasons:
        path = espn_dir / f"games_{season}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                game = json.loads(line)
                game_date = game.get("date", "")
                if not game_date:
                    continue
                starters = game.get("starters") or {}
                home_team = game.get("home_team", {}).get("name", "")
                away_team = game.get("away_team", {}).get("name", "")

                hp = starters.get("home_pitcher") or {}
                ap = starters.get("away_pitcher") or {}
                if hp.get("name"):
                    rows.append({
                        "team_name": home_team,
                        "pitcher_name": hp["name"],
                        "pitcher_id": f"ESPN_{hp.get('espn_id', '')}",
                        "game_date": game_date,
                        "side": "home",
                    })
                if ap.get("name"):
                    rows.append({
                        "team_name": away_team,
                        "pitcher_name": ap["name"],
                        "pitcher_id": f"ESPN_{ap.get('espn_id', '')}",
                        "game_date": game_date,
                        "side": "away",
                    })
    return rows


def load_ncaa_starters(boxscore_path: Path) -> list[dict]:
    """Load starters from NCAA boxscore JSONL."""
    rows = []
    if not boxscore_path.exists():
        return rows
    with open(boxscore_path) as f:
        for line in f:
            game = json.loads(line)
            game_date = game.get("game_date", "")
            if not game_date:
                continue

            for side in ["home", "away"]:
                pitchers = game.get(f"{side}_pitching", [])
                team_name = game.get(f"{side}_team", "")
                for p in pitchers:
                    if p.get("starter"):
                        rows.append({
                            "team_name": team_name,
                            "pitcher_name": p.get("name", ""),
                            "pitcher_id": f"NCAA_{p.get('name', '')}_{team_name}",
                            "game_date": game_date,
                            "side": side,
                        })
                        break  # only first starter per team
    return rows


def load_appearances_starters(appearances_path: Path) -> list[dict]:
    """Load from unified pitcher_appearances.csv."""
    rows = []
    if not appearances_path.exists():
        return rows
    with open(appearances_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("role") == "starter" or r.get("is_starter") == "True":
                rows.append({
                    "team_canonical": r.get("canonical_id", ""),
                    "team_name": r.get("team_name", ""),
                    "pitcher_name": r.get("pitcher_name", ""),
                    "pitcher_id": r.get("pitcher_id", ""),
                    "game_date": r.get("game_date", ""),
                })
    return rows


def build_team_rotation_history(starters: list[dict]) -> dict:
    """
    Build per-team rotation history: team -> list of (game_date, dow, pitcher_name, pitcher_id).
    Only include weekend games (Fri/Sat/Sun).
    """
    team_history = defaultdict(list)
    for s in starters:
        gd = s.get("game_date", "")
        if not gd:
            continue
        try:
            dow = get_day_of_week(gd)
        except ValueError:
            continue
        if dow not in (4, 5, 6):  # Fri, Sat, Sun only
            continue

        team_key = s.get("team_canonical") or s.get("team_name", "")
        if not team_key:
            continue

        team_history[team_key].append({
            "game_date": gd,
            "dow": dow,
            "day": day_label(dow),
            "pitcher_name": s.get("pitcher_name", ""),
            "pitcher_id": s.get("pitcher_id", ""),
        })

    # Sort each team's history by date
    for team in team_history:
        team_history[team].sort(key=lambda x: x["game_date"])

    return dict(team_history)


def project_rotation(team_history: dict, target_friday: str) -> list[dict]:
    """
    For each team, project Friday/Saturday/Sunday starters based on patterns.

    Strategy:
      1. Look at most recent weekend (last Fri/Sat/Sun starters)
      2. Count how many times each pitcher started on each day-of-week
      3. For each slot (Fri/Sat/Sun), pick pitcher with most starts in that slot,
         weighted toward recency
      4. Confidence = based on consistency (always same guy = high, rotating = low)
    """
    projections = []
    target_dt = datetime.strptime(target_friday, "%Y-%m-%d")

    for team, history in team_history.items():
        if not history:
            continue

        # Group by day-of-week
        day_starters = defaultdict(list)  # day -> [(date, pitcher_name, pitcher_id)]
        for h in history:
            day_starters[h["day"]].append({
                "date": h["game_date"],
                "name": h["pitcher_name"],
                "pid": h["pitcher_id"],
            })

        for day in ["fri", "sat", "sun"]:
            entries = day_starters.get(day, [])
            if not entries:
                continue

            # Count starts per pitcher for this day slot
            pitcher_counts = defaultdict(int)
            pitcher_last_date = {}
            pitcher_id_map = {}
            for e in entries:
                name = e["name"]
                pitcher_counts[name] += 1
                pitcher_last_date[name] = e["date"]
                pitcher_id_map[name] = e["pid"]

            # Weight by recency: most recent start gets 2x weight
            most_recent = entries[-1]
            pitcher_scores = {}
            for name, count in pitcher_counts.items():
                score = count
                if name == most_recent["name"]:
                    score += 1.5  # recency bonus
                pitcher_scores[name] = score

            # Pick the top pitcher
            best = max(pitcher_scores, key=pitcher_scores.get)
            total_starts = sum(pitcher_counts.values())
            consistency = pitcher_counts[best] / total_starts if total_starts else 0

            # Confidence levels
            if pitcher_counts[best] >= 3 and consistency >= 0.7:
                confidence = "high"
            elif pitcher_counts[best] >= 2 and consistency >= 0.5:
                confidence = "medium"
            else:
                confidence = "low"

            projections.append({
                "canonical_id": team,
                "day": day,
                "pitcher_name": best,
                "pitcher_id": pitcher_id_map.get(best, ""),
                "confidence": confidence,
                "starts_this_role": pitcher_counts[best],
                "total_day_starts": total_starts,
                "last_started_date": pitcher_last_date.get(best, ""),
            })

    return projections


def main():
    parser = argparse.ArgumentParser(description="Build projected weekend rotations")
    parser.add_argument("--weekend", type=str, default=None,
                        help="Friday date for target weekend (YYYY-MM-DD). Default: next Friday.")
    parser.add_argument("--appearances", type=Path,
                        default=Path("data/processed/pitcher_appearances.csv"))
    parser.add_argument("--espn-dir", type=Path, default=Path("data/raw/espn"))
    parser.add_argument("--ncaa-boxscores", type=Path,
                        default=Path("data/raw/ncaa/boxscores_2026.jsonl"))
    parser.add_argument("--canonical", type=Path,
                        default=Path("data/registries/canonical_teams_2026.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/processed/weekend_rotations.csv"))
    args = parser.parse_args()

    # Determine target Friday
    if args.weekend:
        target_friday = args.weekend
    else:
        today = date.today()
        days_until_friday = (4 - today.weekday()) % 7
        if days_until_friday == 0 and today.weekday() != 4:
            days_until_friday = 7
        target_friday = (today + timedelta(days=days_until_friday)).isoformat()

    print(f"Projecting weekend rotations for {target_friday} weekend")

    # Load canonical teams for name resolution
    canonical = load_canonical_teams(args.canonical)
    name_to_canonical = build_odds_name_to_canonical(canonical)

    # Load starter data from appearances CSV (primary source)
    print("Loading pitcher appearances...")
    starters = load_appearances_starters(args.appearances)
    print(f"  {len(starters)} starter appearances from appearances CSV")

    # Filter to 2026 season only for rotation projection
    starters_2026 = [s for s in starters if s.get("game_date", "").startswith("2026")]
    print(f"  {len(starters_2026)} starter appearances in 2026 season")
    all_starters = starters_2026

    # Build rotation history (weekend games only)
    team_history = build_team_rotation_history(all_starters)
    print(f"  {len(team_history)} teams with weekend starter history")

    # Project rotations
    projections = project_rotation(team_history, target_friday)

    # Sort by team, then day order
    day_order = {"fri": 0, "sat": 1, "sun": 2}
    projections.sort(key=lambda x: (x["canonical_id"], day_order.get(x["day"], 9)))

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "canonical_id", "day", "pitcher_name", "pitcher_id",
            "confidence", "starts_this_role", "total_day_starts", "last_started_date",
        ])
        writer.writeheader()
        writer.writerows(projections)

    # Summary stats
    high_conf = sum(1 for p in projections if p["confidence"] == "high")
    med_conf = sum(1 for p in projections if p["confidence"] == "medium")
    low_conf = sum(1 for p in projections if p["confidence"] == "low")
    teams_with_full = sum(
        1 for team in team_history
        if sum(1 for p in projections if p["canonical_id"] == team) == 3
    )

    print(f"\n=== WEEKEND ROTATION PROJECTIONS ({target_friday}) ===")
    print(f"  Total projections: {len(projections)}")
    print(f"  Confidence: {high_conf} high, {med_conf} medium, {low_conf} low")
    print(f"  Teams with full Fri/Sat/Sun rotation: {teams_with_full}")
    print(f"  Output: {args.out}")

    # Show some example projections for marquee teams
    marquee = ["Arkansas", "LSU", "Texas", "Florida", "Vanderbilt", "Tennessee",
               "Georgia", "Ole Miss", "Alabama", "Clemson", "Virginia", "Wake Forest",
               "Oregon St.", "Stanford", "UCLA", "Duke", "North Carolina", "Miami"]
    print(f"\n  === MARQUEE TEAM ROTATIONS ===")
    for p in projections:
        team = p["canonical_id"]
        for m in marquee:
            if m.lower() in team.lower():
                print(f"  {team:30s} {p['day'].upper():3s}: {p['pitcher_name']:25s} "
                      f"({p['confidence']}, {p['starts_this_role']} starts)")
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
