"""
Merge full scrape (with boxscore) and scoreboard-only 2026 games into one JSONL.
Keeps full record when available; fills in missing games from scoreboard with empty boxscore.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

def main():
    espn_dir = Path("data/raw/espn")
    full_path = espn_dir / "games_2026.jsonl"
    scoreboard_path = espn_dir / "games_2026_scoreboard.jsonl"
    out_path = espn_dir / "games_2026_merged.jsonl"

    by_id = {}
    if full_path.exists():
        with full_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    g = json.loads(line)
                except json.JSONDecodeError:
                    continue
                eid = g.get("event_id") or g.get("id")
                if eid:
                    by_id[str(eid)] = g

    if scoreboard_path.exists():
        with scoreboard_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    g = json.loads(line)
                except json.JSONDecodeError:
                    continue
                eid = g.get("event_id") or g.get("id")
                if not eid:
                    continue
                if str(eid) in by_id:
                    continue
                # Ensure same shape as full record for downstream
                g["boxscore"] = g.get("boxscore") or {}
                g["plays"] = g.get("plays") or []
                g["run_events"] = g.get("run_events")
                g["starters"] = g.get("starters") or {"home_pitcher": None, "away_pitcher": None}
                g["line_scores"] = g.get("line_scores") or {"home": [], "away": []}
                if "event_id" not in g and "id" in g:
                    g["event_id"] = g["id"]
                by_id[str(eid)] = g

    # Write in date order
    games = list(by_id.values())
    games.sort(key=lambda x: (x.get("date") or "", x.get("event_id") or x.get("id") or ""))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for g in games:
            f.write(json.dumps(g) + "\n")

    # Replace games_2026.jsonl so rest of pipeline uses one file
    final = espn_dir / "games_2026.jsonl"
    if out_path.resolve() != final.resolve():
        final.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
        out_path.unlink(missing_ok=True)
        print(f"Merged {len(games)} 2026 games -> {final}")
    else:
        print(f"Merged {len(games)} 2026 games -> {out_path}")
    full_count = sum(1 for g in games if g.get("boxscore") and isinstance(g.get("boxscore"), dict) and len(g.get("boxscore", {})) > 0)
    print(f"  With boxscore: {full_count}, scoreboard-only: {len(games) - full_count}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
