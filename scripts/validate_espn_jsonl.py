#!/usr/bin/env python3
"""Validate ESPN games JSONL: counts and spot-check scores/starters."""
import json
import sys
from pathlib import Path

def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/espn/games_2024.jsonl")
    total = 0
    with_pbp = 0
    with_boxscore = 0
    with_starters = 0
    samples = []

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            g = json.loads(line)
            has_pbp = bool(g.get("pbp_available") and (g.get("plays") or []))
            box = g.get("boxscore") or {}
            has_box = bool(box.get("home") or box.get("away"))
            st = g.get("starters") or {}
            has_start = bool(st.get("home_pitcher") or st.get("away_pitcher"))
            if has_pbp:
                with_pbp += 1
            if has_box:
                with_boxscore += 1
            if has_start:
                with_starters += 1
            if len(samples) < 5 and has_pbp:
                samples.append(g)

    print("=== COUNTS ===")
    print(f"Total games: {total}")
    print(f"Games with PBP (plays): {with_pbp}")
    print(f"Games with boxscore: {with_boxscore}")
    print(f"Games with starters: {with_starters}")
    print()
    print("=== SPOT-CHECK (PBP games) ===")
    for i, g in enumerate(samples):
        home = g.get("home_team", {}).get("name", "?")
        away = g.get("away_team", {}).get("name", "?")
        hs, aws = g.get("home_score"), g.get("away_score")
        ls = g.get("line_scores", {})
        home_inning = sum(ls.get("home") or [])
        away_inning = sum(ls.get("away") or [])
        scores_ok = away_inning == aws and home_inning == hs
        print(f"--- Sample {i+1}: {away} @ {home} ({g.get('date')}) ---")
        print(f"  Final score: {aws}-{hs} (away-home)")
        print(f"  Line-score sums: away={away_inning} home={home_inning}  match={scores_ok}")
        st = g.get("starters", {})
        print(f"  Starters: away_pitcher={bool(st.get('away_pitcher'))} home_pitcher={bool(st.get('home_pitcher'))}")
        if st.get("away_pitcher"):
            print(f"    Away: {st['away_pitcher'].get('name', '?')}")
        if st.get("home_pitcher"):
            print(f"    Home: {st['home_pitcher'].get('name', '?')}")
        print()
    print("=== SUMMARY ===")
    print(f"File: {path}")
    print(f"Total: {total} games | PBP: {with_pbp} | Boxscore: {with_boxscore} | Starters: {with_starters}")

if __name__ == "__main__":
    main()
