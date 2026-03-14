# Pipeline Refactor: Single-Pass Extract + Unified Tables + Decomposed Predict

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 6 redundant ESPN JSONL parsers with one extract pass, unify pitcher/team data into two lookup tables, and decompose the 820-line predict_day.py god-script into 4 focused modules.

**Architecture:** Single `extract_espn.py` reads all ESPN JSONL once, outputs 4 CSVs. `build_pitcher_table.py` and `build_team_table.py` merge all enrichment sources into pre-computed lookup CSVs. `predict_day.py` is decomposed into schedule resolution, starter resolution, weather resolution, and a pure simulation engine orchestrated by a thin wrapper. A Makefile encodes the full dependency DAG.

**Tech Stack:** Python 3.12+, pandas, numpy, CmdStanPy (fitting only), Open-Meteo API (weather), NCAA/ESPN/Odds APIs (schedule)

---

## File Structure

### New Files

| File | Purpose |
|------|---------|
| `scripts/extract_espn.py` | Single-pass ESPN JSONL extractor → 4 output CSVs |
| `scripts/build_pitcher_table.py` | Unified pitcher lookup (registry + D1B stats + FB% + handedness + ERA) |
| `scripts/build_team_table.py` | Unified team lookup (bullpen quality + wRC+ + conference) |
| `scripts/resolve_schedule.py` | Fetch day's games from NCAA/ESPN APIs → schedule CSV |
| `scripts/resolve_starters.py` | Project starters + enrich with ability estimates → starters CSV |
| `scripts/resolve_weather.py` | Fetch weather per game → weather CSV |
| `scripts/simulate.py` | Pure Monte Carlo engine (no API calls) → predictions CSV |
| `Makefile` | Pipeline DAG with file-level dependencies |
| `tests/test_extract_espn.py` | Tests for extract logic |
| `tests/test_pitcher_table.py` | Tests for pitcher table builder |
| `tests/test_team_table.py` | Tests for team table builder |
| `tests/test_simulate.py` | Tests for simulation engine |

### Modified Files

| File | Change |
|------|--------|
| `scripts/predict_day.py` | Slim to thin orchestrator calling resolve_schedule → resolve_starters → resolve_weather → simulate |
| `scripts/lookup_starters.py` | Refactor to read from `pitcher_table.csv` instead of 5 separate files |
| `scripts/fb_sensitivity.py` | Refactor to read from `pitcher_table.csv` (FB% column) |
| `scripts/platoon_adjustment.py` | Refactor to read from `pitcher_table.csv` (throws column) |

### Deprecated (kept but no longer in main pipeline)

| File | Replacement |
|------|-------------|
| `scripts/build_run_events_from_espn.py` | `extract_espn.py` → `run_events.csv` |
| `scripts/build_pitcher_registry.py` | `extract_espn.py` → `pitcher_appearances.csv` + `build_pitcher_table.py` |
| `scripts/build_games_from_espn.py` | `extract_espn.py` → `games.csv` |
| `scripts/build_pitching_from_espn.py` | `extract_espn.py` → `pitching_lines.csv` |
| `scripts/build_park_factors.py` | `extract_espn.py` → `venue_stats.csv` + existing park factor logic |
| `scripts/build_bullpen_fatigue.py` | `extract_espn.py` → `pitcher_appearances.csv` + `build_team_table.py` |

---

## Chunk 1: Single-Pass ESPN Extractor

### Task 1: Build `extract_espn.py`

**Files:**
- Create: `scripts/extract_espn.py`
- Create: `tests/test_extract_espn.py`
- Reference: `scripts/build_run_events_from_espn.py` (run_events extraction logic)
- Reference: `scripts/build_pitcher_registry.py` (pitcher appearance extraction logic)
- Reference: `scripts/build_pitching_from_espn.py` (pitching line extraction logic)
- Reference: `scripts/build_park_factors.py` (venue aggregation logic)

**Concept:** One pass over each `data/raw/espn/games_{YYYY}.jsonl`. For each game JSON, extract ALL information into normalized rows. Output 4 CSV files that replace what 6 scripts currently produce.

**Output files (all to `data/processed/`):**

1. **`games.csv`** — one row per game (replaces `games_espn.csv`)
   ```
   event_id, game_date, season, home_name, away_name,
   home_canonical_id, away_canonical_id, home_score, away_score,
   winner_home, venue_name, venue_city, venue_state, neutral_site,
   home_pitcher_espn_id, away_pitcher_espn_id,
   home_pitcher_name, away_pitcher_name,
   has_run_events, has_boxscore
   ```

2. **`run_events.csv`** — one row per game with PBP run decomposition (replaces current `run_events.csv`)
   ```
   event_id, game_date, season, home_canonical_id, away_canonical_id,
   home_pitcher_espn_id, away_pitcher_espn_id,
   home_run_1, home_run_2, home_run_3, home_run_4,
   away_run_1, away_run_2, away_run_3, away_run_4,
   home_score, away_score
   ```
   (Schema identical to current — downstream Stan pipeline unchanged)

3. **`pitcher_appearances.csv`** — one row per pitcher per game (replaces current `pitcher_appearances.csv` ESPN portion)
   ```
   event_id, game_date, season, pitcher_espn_id, pitcher_name,
   team_canonical_id, team_name, side (home/away), starter (bool),
   ip, h, r, er, bb, k, hr, pc
   ```

4. **`venue_stats.csv`** — one row per venue (aggregated, replaces inline park factor calc)
   ```
   venue_name, venue_city, venue_state, home_canonical_id,
   n_games, total_home_runs, total_away_runs, rpg,
   home_run_1_avg, home_run_2_avg, home_run_3_avg, home_run_4_avg,
   away_run_1_avg, away_run_2_avg, away_run_3_avg, away_run_4_avg
   ```

5. **`extract_manifest.json`** — metadata
   ```json
   {
     "extracted_at": "2026-03-13T...",
     "seasons": ["2024", "2025", "2026"],
     "n_games": 8234,
     "n_run_event_games": 2051,
     "n_pitcher_appearances": 45000,
     "n_venues": 312,
     "source_files": ["games_2024.jsonl", "games_2025.jsonl", "games_2026.jsonl"]
   }
   ```

- [ ] **Step 1: Write test for game extraction**

  Create `tests/test_extract_espn.py` with a synthetic ESPN JSONL line and verify all 4 output tables are produced correctly from a single game.

  ```python
  """Tests for extract_espn.py single-pass ESPN extractor."""
  import json
  import tempfile
  from pathlib import Path

  import pandas as pd
  import pytest

  # Minimal ESPN game JSON for testing
  SAMPLE_GAME = {
      "event_id": "401234567",
      "id": "401234567",
      "date": "2026-03-10T18:00Z",
      "season": 2026,
      "neutral_site": False,
      "home_team": {"id": "123", "name": "Florida Gators", "abbreviation": "FLA"},
      "away_team": {"id": "456", "name": "Georgia Bulldogs", "abbreviation": "UGA"},
      "home_score": 5,
      "away_score": 3,
      "venue": {"name": "Florida Ballpark", "city": "Gainesville", "state": "FL"},
      "starters": {
          "home_pitcher": {"espn_id": "67525", "name": "Hurston Waldrep"},
          "away_pitcher": {"espn_id": "67890", "name": "Charlie Condon"},
      },
      "run_events": {
          "home": {"run_1": 2, "run_2": 1, "run_3": 1, "run_4": 1},
          "away": {"run_1": 1, "run_2": 1, "run_3": 1, "run_4": 0},
      },
      "boxscore": {
          "FLA": {
              "pitching": [
                  {"espn_id": "67525", "name": "Waldrep", "starter": True,
                   "stats": {"IP": "6.0", "H": "4", "R": "2", "ER": "2",
                             "BB": "1", "K": "8", "HR": "0", "PC": "92"}},
                  {"espn_id": "67526", "name": "Reliever A", "starter": False,
                   "stats": {"IP": "3.0", "H": "1", "R": "1", "ER": "1",
                             "BB": "0", "K": "4", "HR": "0", "PC": "40"}},
              ]
          },
          "UGA": {
              "pitching": [
                  {"espn_id": "67890", "name": "Condon", "starter": True,
                   "stats": {"IP": "5.1", "H": "6", "R": "4", "ER": "4",
                             "BB": "2", "K": "5", "HR": "1", "PC": "88"}},
              ]
          }
      }
  }

  def _write_jsonl(path: Path, games: list[dict]):
      with open(path, "w") as f:
          for g in games:
              f.write(json.dumps(g) + "\n")

  def _make_canonical_csv(path: Path):
      """Minimal canonical_teams_2026.csv with Florida + Georgia."""
      pd.DataFrame([
          {"canonical_id": "BSB_FLORIDA", "ncaa_teams_id": 234,
           "team_name": "Florida", "espn_name": "Florida Gators",
           "odds_api_name": "Florida Gators", "conference": "SEC"},
          {"canonical_id": "BSB_GEORGIA", "ncaa_teams_id": 345,
           "team_name": "Georgia", "espn_name": "Georgia Bulldogs",
           "odds_api_name": "Georgia Bulldogs", "conference": "SEC"},
      ]).to_csv(path, index=False)


  def test_extract_produces_all_tables():
      """Single game should produce rows in all 4 output CSVs."""
      with tempfile.TemporaryDirectory() as tmp:
          tmp = Path(tmp)
          espn_dir = tmp / "espn"
          espn_dir.mkdir()
          _write_jsonl(espn_dir / "games_2026.jsonl", [SAMPLE_GAME])
          canonical = tmp / "canonical.csv"
          _make_canonical_csv(canonical)
          out_dir = tmp / "out"

          from extract_espn import extract_all
          manifest = extract_all(
              espn_dir=espn_dir,
              canonical_csv=canonical,
              out_dir=out_dir,
              seasons=["2026"],
          )

          # Games table
          games = pd.read_csv(out_dir / "games.csv")
          assert len(games) == 1
          assert games.iloc[0]["home_canonical_id"] == "BSB_FLORIDA"
          assert games.iloc[0]["away_canonical_id"] == "BSB_GEORGIA"
          assert games.iloc[0]["winner_home"] == 1

          # Run events
          re = pd.read_csv(out_dir / "run_events.csv")
          assert len(re) == 1
          assert re.iloc[0]["home_run_1"] == 2

          # Pitcher appearances
          pa = pd.read_csv(out_dir / "pitcher_appearances.csv")
          assert len(pa) == 3  # 2 Florida pitchers + 1 Georgia pitcher
          starters = pa[pa["starter"] == True]
          assert len(starters) == 2

          # Venue stats
          vs = pd.read_csv(out_dir / "venue_stats.csv")
          assert len(vs) == 1
          assert vs.iloc[0]["venue_name"] == "Florida Ballpark"
          assert vs.iloc[0]["n_games"] == 1

          # Manifest
          assert manifest["n_games"] == 1
          assert manifest["n_run_event_games"] == 1


  def test_extract_skips_games_without_scores():
      """Games with null scores should appear in games.csv but not affect venue_stats."""
      game_no_score = {**SAMPLE_GAME, "home_score": None, "away_score": None,
                        "event_id": "999", "run_events": None, "boxscore": {}}
      with tempfile.TemporaryDirectory() as tmp:
          tmp = Path(tmp)
          espn_dir = tmp / "espn"
          espn_dir.mkdir()
          _write_jsonl(espn_dir / "games_2026.jsonl", [SAMPLE_GAME, game_no_score])
          canonical = tmp / "canonical.csv"
          _make_canonical_csv(canonical)
          out_dir = tmp / "out"

          from extract_espn import extract_all
          extract_all(espn_dir=espn_dir, canonical_csv=canonical,
                      out_dir=out_dir, seasons=["2026"])

          games = pd.read_csv(out_dir / "games.csv")
          assert len(games) == 2  # both games recorded
          vs = pd.read_csv(out_dir / "venue_stats.csv")
          assert vs.iloc[0]["n_games"] == 1  # only scored game
  ```

- [ ] **Step 2: Run tests — verify they fail (module not found)**

  ```bash
  cd /Users/anthonyeding/ncaa_baseball/ncaa_baseball-1
  .venv/bin/python3 -m pytest tests/test_extract_espn.py -v
  ```
  Expected: `ModuleNotFoundError: No module named 'extract_espn'`

- [ ] **Step 3: Implement `extract_espn.py`**

  Core function `extract_all()` does a single pass over each JSONL file. Key design:
  - Load canonical teams ONCE at the start
  - Build name resolution map ONCE
  - Iterate each line, extract into 4 accumulator lists
  - Write all CSVs at end
  - IP parsing: handle ESPN's `.1` → 1/3, `.2` → 2/3 convention

  ```python
  """
  Single-pass ESPN JSONL extractor.

  Replaces 6 separate scripts that each independently parsed the same
  ESPN game files. One pass → 4 output CSVs + manifest.

  Usage:
    python3 scripts/extract_espn.py
    python3 scripts/extract_espn.py --seasons 2026
    python3 scripts/extract_espn.py --espn-dir data/raw/espn --out-dir data/processed
  """
  from __future__ import annotations

  import argparse
  import json
  import math
  from datetime import datetime, timezone
  from pathlib import Path

  import pandas as pd

  import _bootstrap  # noqa: F401
  from ncaa_baseball.phase1 import (
      build_odds_name_to_canonical,
      load_canonical_teams,
      resolve_odds_teams,
  )


  def _safe_int(v, default: int = 0) -> int | None:
      if v is None:
          return default
      try:
          return int(v)
      except (TypeError, ValueError):
          return default


  def _safe_float(v, default: float | None = None) -> float | None:
      if v is None:
          return default
      try:
          return float(v)
      except (TypeError, ValueError):
          return default


  def _parse_ip(raw) -> float | None:
      """Parse ESPN IP format: '6.1' → 6.333, '5.2' → 5.667."""
      v = _safe_float(raw)
      if v is None:
          return None
      whole = int(v)
      frac = round(v - whole, 1)
      if abs(frac - 0.1) < 0.01:
          return whole + 1 / 3
      elif abs(frac - 0.2) < 0.01:
          return whole + 2 / 3
      return v


  def _resolve_team(name: str, name_to_canonical, canonical_df):
      """Resolve ESPN team name → (canonical_id, ncaa_teams_id) or (None, None)."""
      if not name:
          return None, None
      ht, _ = resolve_odds_teams(name, "", canonical_df, name_to_canonical)
      if ht:
          return ht[0], ht[1]
      return None, None


  def extract_all(
      espn_dir: Path,
      canonical_csv: Path,
      out_dir: Path,
      seasons: list[str] | None = None,
  ) -> dict:
      """
      Single-pass extraction from ESPN JSONL files.

      Returns manifest dict with extraction statistics.
      """
      canonical = load_canonical_teams(canonical_csv)
      name_to_canonical = build_odds_name_to_canonical(canonical)

      if seasons is None:
          seasons = ["2024", "2025", "2026"]

      games_rows = []
      run_event_rows = []
      pitcher_app_rows = []
      venue_agg: dict[str, dict] = {}  # venue_name → accumulator

      for season in seasons:
          path = espn_dir / f"games_{season}.jsonl"
          if not path.exists():
              continue
          with path.open(encoding="utf-8") as f:
              for line in f:
                  line = line.strip()
                  if not line:
                      continue
                  try:
                      g = json.loads(line)
                  except json.JSONDecodeError:
                      continue

                  _process_game(
                      g, season, canonical, name_to_canonical,
                      games_rows, run_event_rows, pitcher_app_rows, venue_agg,
                  )

      # Write outputs
      out_dir.mkdir(parents=True, exist_ok=True)

      games_df = pd.DataFrame(games_rows)
      games_df.to_csv(out_dir / "games.csv", index=False)

      re_df = pd.DataFrame(run_event_rows)
      if re_df.empty:
          re_df = pd.DataFrame(columns=[
              "event_id", "game_date", "season",
              "home_canonical_id", "away_canonical_id",
              "home_pitcher_espn_id", "away_pitcher_espn_id",
              "home_run_1", "home_run_2", "home_run_3", "home_run_4",
              "away_run_1", "away_run_2", "away_run_3", "away_run_4",
              "home_score", "away_score",
          ])
      re_df.to_csv(out_dir / "run_events.csv", index=False)

      pa_df = pd.DataFrame(pitcher_app_rows)
      pa_df.to_csv(out_dir / "pitcher_appearances.csv", index=False)

      venue_rows = list(venue_agg.values())
      vs_df = pd.DataFrame(venue_rows)
      vs_df.to_csv(out_dir / "venue_stats.csv", index=False)

      manifest = {
          "extracted_at": datetime.now(timezone.utc).isoformat(),
          "seasons": seasons,
          "n_games": len(games_rows),
          "n_run_event_games": len(run_event_rows),
          "n_pitcher_appearances": len(pitcher_app_rows),
          "n_venues": len(venue_agg),
          "source_files": [f"games_{s}.jsonl" for s in seasons],
      }
      with open(out_dir / "extract_manifest.json", "w") as f:
          json.dump(manifest, f, indent=2)

      print(f"Extracted: {manifest['n_games']} games, "
            f"{manifest['n_run_event_games']} with run events, "
            f"{manifest['n_pitcher_appearances']} pitcher appearances, "
            f"{manifest['n_venues']} venues")

      return manifest


  def _process_game(g, season, canonical, name_to_canonical,
                    games_rows, run_event_rows, pitcher_app_rows, venue_agg):
      """Process a single ESPN game JSON into all output tables."""
      event_id = str(g.get("event_id") or g.get("id") or "")
      game_date = (g.get("date") or "")[:10]
      try:
          yr = int(g.get("season") or season)
      except (TypeError, ValueError):
          yr = int(season) if str(season).isdigit() else 0

      home = g.get("home_team") or {}
      away = g.get("away_team") or {}
      home_name = (home.get("name") or "").strip()
      away_name = (away.get("name") or "").strip()
      if not home_name or not away_name:
          return

      h_cid, _ = _resolve_team(home_name, name_to_canonical, canonical)
      a_cid, _ = _resolve_team(away_name, name_to_canonical, canonical)
      h_cid = h_cid or ""
      a_cid = a_cid or ""

      home_score = _safe_int(g.get("home_score"), default=None)
      away_score = _safe_int(g.get("away_score"), default=None)
      winner_home = None
      if home_score is not None and away_score is not None:
          winner_home = 1 if home_score > away_score else 0

      venue = g.get("venue") or {}
      venue_name = (venue.get("name") or "").strip()
      venue_city = (venue.get("city") or "").strip()
      venue_state = (venue.get("state") or "").strip()
      neutral = bool(g.get("neutral_site"))

      starters = g.get("starters") or {}
      hp = starters.get("home_pitcher") or {}
      ap = starters.get("away_pitcher") or {}
      hp_id = str(hp.get("espn_id") or hp.get("id") or "")
      ap_id = str(ap.get("espn_id") or ap.get("id") or "")
      hp_name = (hp.get("name") or "").strip()
      ap_name = (ap.get("name") or "").strip()

      re = g.get("run_events")
      has_re = bool(re and isinstance(re, dict) and re.get("home") and re.get("away"))
      has_box = bool(g.get("boxscore"))

      # ── games.csv row ──
      games_rows.append({
          "event_id": event_id,
          "game_date": game_date,
          "season": yr,
          "home_name": home_name,
          "away_name": away_name,
          "home_canonical_id": h_cid,
          "away_canonical_id": a_cid,
          "home_score": home_score,
          "away_score": away_score,
          "winner_home": winner_home,
          "venue_name": venue_name,
          "venue_city": venue_city,
          "venue_state": venue_state,
          "neutral_site": neutral,
          "home_pitcher_espn_id": hp_id,
          "away_pitcher_espn_id": ap_id,
          "home_pitcher_name": hp_name,
          "away_pitcher_name": ap_name,
          "has_run_events": has_re,
          "has_boxscore": has_box,
      })

      # ── run_events.csv row ──
      if has_re:
          home_re = re.get("home") or {}
          away_re = re.get("away") or {}
          run_event_rows.append({
              "event_id": event_id,
              "game_date": game_date,
              "season": yr,
              "home_canonical_id": h_cid,
              "away_canonical_id": a_cid,
              "home_pitcher_espn_id": hp_id,
              "away_pitcher_espn_id": ap_id,
              "home_run_1": _safe_int(home_re.get("run_1")),
              "home_run_2": _safe_int(home_re.get("run_2")),
              "home_run_3": _safe_int(home_re.get("run_3")),
              "home_run_4": _safe_int(home_re.get("run_4")),
              "away_run_1": _safe_int(away_re.get("run_1")),
              "away_run_2": _safe_int(away_re.get("run_2")),
              "away_run_3": _safe_int(away_re.get("run_3")),
              "away_run_4": _safe_int(away_re.get("run_4")),
              "home_score": home_score,
              "away_score": away_score,
          })

      # ── pitcher_appearances.csv rows ──
      boxscore = g.get("boxscore") or {}
      for team_key, bs_data in boxscore.items():
          if not isinstance(bs_data, dict):
              continue
          pitching = bs_data.get("pitching") or []
          # Determine which side this team is
          home_abbr = (home.get("abbreviation") or "").upper()
          away_abbr = (away.get("abbreviation") or "").upper()
          home_id_str = str(home.get("id") or "")
          away_id_str = str(away.get("id") or "")
          if team_key.upper() == home_abbr or team_key == home_id_str:
              side = "home"
              team_cid = h_cid
              team_display = home_name
          elif team_key.upper() == away_abbr or team_key == away_id_str:
              side = "away"
              team_cid = a_cid
              team_display = away_name
          else:
              side = "unknown"
              team_cid = ""
              team_display = team_key

          for p in pitching:
              if not isinstance(p, dict):
                  continue
              stats = p.get("stats") or {}
              pitcher_app_rows.append({
                  "event_id": event_id,
                  "game_date": game_date,
                  "season": yr,
                  "pitcher_espn_id": str(p.get("espn_id") or p.get("id") or ""),
                  "pitcher_name": (p.get("name") or "").strip(),
                  "team_canonical_id": team_cid,
                  "team_name": team_display,
                  "side": side,
                  "starter": bool(p.get("starter")),
                  "ip": _parse_ip(stats.get("IP")),
                  "h": _safe_int(stats.get("H"), default=None),
                  "r": _safe_int(stats.get("R"), default=None),
                  "er": _safe_int(stats.get("ER"), default=None),
                  "bb": _safe_int(stats.get("BB"), default=None),
                  "k": _safe_int(stats.get("K"), default=None),
                  "hr": _safe_int(stats.get("HR"), default=None),
                  "pc": _safe_int(stats.get("PC"), default=None),
              })

      # ── venue_stats accumulator ──
      if (venue_name and not neutral and home_score is not None
              and away_score is not None and has_re):
          home_re = re.get("home") or {}
          away_re = re.get("away") or {}
          if venue_name not in venue_agg:
              venue_agg[venue_name] = {
                  "venue_name": venue_name,
                  "venue_city": venue_city,
                  "venue_state": venue_state,
                  "home_canonical_id": h_cid,
                  "n_games": 0,
                  "total_home_runs": 0,
                  "total_away_runs": 0,
                  **{f"sum_home_run_{k}": 0 for k in range(1, 5)},
                  **{f"sum_away_run_{k}": 0 for k in range(1, 5)},
              }
          v = venue_agg[venue_name]
          v["n_games"] += 1
          v["total_home_runs"] += home_score
          v["total_away_runs"] += away_score
          for k in range(1, 5):
              v[f"sum_home_run_{k}"] += _safe_int(home_re.get(f"run_{k}"))
              v[f"sum_away_run_{k}"] += _safe_int(away_re.get(f"run_{k}"))

      # Finalize venue averages (will be called after loop in caller)
      # Actually we do this at write time — compute rpg and averages from sums


  def _finalize_venue_stats(venue_agg: dict) -> list[dict]:
      """Convert sum accumulators to averages."""
      rows = []
      for v in venue_agg.values():
          n = v["n_games"]
          if n == 0:
              continue
          row = {
              "venue_name": v["venue_name"],
              "venue_city": v["venue_city"],
              "venue_state": v["venue_state"],
              "home_canonical_id": v["home_canonical_id"],
              "n_games": n,
              "total_home_runs": v["total_home_runs"],
              "total_away_runs": v["total_away_runs"],
              "rpg": round((v["total_home_runs"] + v["total_away_runs"]) / n, 3),
          }
          for side in ("home", "away"):
              for k in range(1, 5):
                  row[f"{side}_run_{k}_avg"] = round(
                      v[f"sum_{side}_run_{k}"] / n, 4)
          rows.append(row)
      return rows


  def main() -> int:
      parser = argparse.ArgumentParser(
          description="Single-pass ESPN JSONL extractor → 4 output CSVs.",
      )
      parser.add_argument("--espn-dir", type=Path,
                          default=Path("data/raw/espn"))
      parser.add_argument("--canonical", type=Path,
                          default=Path("data/registries/canonical_teams_2026.csv"))
      parser.add_argument("--out-dir", type=Path,
                          default=Path("data/processed"))
      parser.add_argument("--seasons", type=str, default="2024,2025,2026",
                          help="Comma-separated seasons to extract")
      args = parser.parse_args()

      seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
      extract_all(
          espn_dir=args.espn_dir,
          canonical_csv=args.canonical,
          out_dir=args.out_dir,
          seasons=seasons,
      )
      return 0


  if __name__ == "__main__":
      raise SystemExit(main())
  ```

  **Important:** Update the `extract_all` function to call `_finalize_venue_stats(venue_agg)` instead of writing raw accumulator dicts. Replace the venue write section:
  ```python
  venue_rows = _finalize_venue_stats(venue_agg)
  vs_df = pd.DataFrame(venue_rows)
  vs_df.to_csv(out_dir / "venue_stats.csv", index=False)
  ```

- [ ] **Step 4: Run tests — verify they pass**
  ```bash
  .venv/bin/python3 -m pytest tests/test_extract_espn.py -v
  ```

- [ ] **Step 5: Integration test — run on real data, compare with current outputs**
  ```bash
  # Run new extractor
  .venv/bin/python3 scripts/extract_espn.py --out-dir data/processed/extracted_test

  # Compare run_events row counts
  wc -l data/processed/run_events.csv data/processed/extracted_test/run_events.csv

  # Compare schema
  head -1 data/processed/run_events.csv
  head -1 data/processed/extracted_test/run_events.csv

  # Spot-check: same teams resolved for a few games
  .venv/bin/python3 -c "
  import pandas as pd
  old = pd.read_csv('data/processed/run_events.csv')
  new = pd.read_csv('data/processed/extracted_test/run_events.csv')
  print(f'Old: {len(old)} rows, New: {len(new)} rows')
  # Compare first 5 event_ids
  for eid in old['event_id'].head(5):
      o = old[old['event_id']==str(eid)]
      n = new[new['event_id']==str(eid)]
      if len(n):
          print(f'{eid}: old={o.iloc[0][\"home_canonical_id\"]}, new={n.iloc[0][\"home_canonical_id\"]}')
  "
  ```

- [ ] **Step 6: Commit**
  ```bash
  git add scripts/extract_espn.py tests/test_extract_espn.py
  git commit -m "feat: single-pass ESPN extractor replaces 6 redundant parsers"
  ```

---

## Chunk 2: Unified Pitcher Table

### Task 2: Build `build_pitcher_table.py`

**Files:**
- Create: `scripts/build_pitcher_table.py`
- Create: `tests/test_pitcher_table.py`
- Reference: `scripts/fb_sensitivity.py` (FB% extraction)
- Reference: `scripts/platoon_adjustment.py` (handedness extraction)
- Reference: `scripts/lookup_starters.py` (crosswalk logic, lines 40-120)
- Reference: `scripts/predict_day.py` (D1B FIP percentile mapping, lines 154-410)

**Concept:** One script that produces a single CSV where every pitcher has ONE row with ALL attributes. No more joining 5 files at runtime.

**Output: `data/processed/pitcher_table.csv`**

```
pitcher_id          # Canonical format: ESPN_{id} or NCAA_{name}__{cid}
pitcher_espn_id     # Numeric ESPN ID (for Stan index lookup), empty if NCAA-only
pitcher_idx         # Stan model index (0 = no posterior)
pitcher_name        # Display name
team_canonical_id   # Team
season              # Most recent season
throws              # L, R, or empty (from D1B rotations)
role                # SP, RP, or BOTH (from appearance history)
season_ip           # Total IP this season
season_era          # Season ERA (from appearances, 5+ IP)
fip                 # FIP from D1B pitching_advanced.tsv (or None)
siera               # SIERA from D1B (or None)
fb_pct              # Fly-ball % from D1B batted_ball.tsv (or None)
fb_sensitivity      # Pre-computed: 0.3 + 0.7 * (fb_pct / league_avg)
d1b_ability_adj     # Pre-computed ability adjustment (FIP percentile → z × 0.05)
d1b_ability_source  # fip, siera, era_d1b, era_rotation, era_appearances, or empty
n_appearances       # Count of appearances this season
last_appearance     # YYYY-MM-DD of most recent appearance
weekend_slot        # fri, sat, sun, or empty (projected rotation day)
weekend_confidence  # high, medium, low, or empty
```

**Sources merged:**
1. `pitcher_appearances.csv` (from extract_espn) — base roster, ERA, IP, appearance dates
2. `run_event_pitcher_index.csv` — Stan model pitcher_idx mapping
3. `d1baseball/pitching_advanced.tsv` — FIP, SIERA
4. `d1baseball/pitching_standard.tsv` — ERA for qualified D1B pitchers
5. `d1baseball/pitching_batted_ball.tsv` — FB%
6. `d1baseball_rotations.csv` — handedness (throws L/R), rotation ERA
7. `d1baseball_crosswalk.csv` — D1B team name → canonical_id
8. `weekend_rotations.csv` — projected rotation slots
9. `pitcher_registry.csv` — canonical pitcher_id format

- [ ] **Step 1: Write tests for pitcher table builder**

  ```python
  """Tests for build_pitcher_table.py unified pitcher table."""
  import tempfile
  from pathlib import Path
  import pandas as pd
  import pytest


  def test_pitcher_table_merges_all_sources():
      """Pitcher table should contain columns from all input sources."""
      with tempfile.TemporaryDirectory() as tmp:
          tmp = Path(tmp)
          # Create minimal input files
          # ... (detailed test fixtures for each input file)
          from build_pitcher_table import build_pitcher_table
          result = build_pitcher_table(
              appearances_csv=tmp / "pitcher_appearances.csv",
              pitcher_index_csv=tmp / "pitcher_index.csv",
              d1b_dir=tmp / "d1baseball",
              crosswalk_csv=tmp / "crosswalk.csv",
              rotations_csv=tmp / "rotations.csv",
              weekend_csv=tmp / "weekend.csv",
              registry_csv=tmp / "registry.csv",
              out_csv=tmp / "pitcher_table.csv",
          )
          df = pd.read_csv(tmp / "pitcher_table.csv")
          # Every pitcher should have these columns
          for col in ["pitcher_id", "pitcher_idx", "pitcher_name",
                      "team_canonical_id", "throws", "fb_sensitivity",
                      "d1b_ability_adj", "d1b_ability_source"]:
              assert col in df.columns, f"Missing column: {col}"


  def test_fip_percentile_mapping():
      """FIP → percentile → ability should produce correct sign."""
      from build_pitcher_table import _fip_to_ability
      # Low FIP (good pitcher) → negative ability adj (suppresses runs)
      assert _fip_to_ability(2.5, [2.0, 3.0, 4.0, 5.0, 6.0]) < 0
      # High FIP (bad pitcher) → positive ability adj
      assert _fip_to_ability(6.0, [2.0, 3.0, 4.0, 5.0, 6.0]) > 0


  def test_fb_sensitivity_calculation():
      """FB sensitivity should scale around 1.0 with league avg."""
      from build_pitcher_table import _fb_sensitivity
      # At league average → 1.0
      assert abs(_fb_sensitivity(38.9, 38.9) - 1.0) < 0.01
      # High FB% → > 1.0
      assert _fb_sensitivity(50.0, 38.9) > 1.0
      # Low FB% → < 1.0 but >= 0.3
      assert 0.3 <= _fb_sensitivity(10.0, 38.9) < 1.0
  ```

- [ ] **Step 2: Run tests — verify they fail**

- [ ] **Step 3: Implement `build_pitcher_table.py`**

  Key logic (moved from predict_day.py lines 154-410):
  - Load all D1B TSVs with unicode apostrophe normalization
  - Build FIP distribution, compute percentiles
  - Map each pitcher: FIP → percentile → z-score → ability (sign: low FIP = negative = good)
  - Fallback chain: FIP → SIERA → ERA (D1B standard) → ERA (rotation page) → ERA (appearances 5+ IP)
  - Merge FB% from batted_ball.tsv
  - Merge handedness from d1baseball_rotations.csv
  - Merge weekend slot from weekend_rotations.csv
  - Output one row per pitcher

- [ ] **Step 4: Run tests — verify they pass**

- [ ] **Step 5: Integration test — compare with current predict_day.py D1B ability values**
  ```bash
  # Run builder
  .venv/bin/python3 scripts/build_pitcher_table.py

  # Spot-check a known pitcher's ability
  .venv/bin/python3 -c "
  import pandas as pd
  pt = pd.read_csv('data/processed/pitcher_table.csv')
  # Check a known D1B pitcher
  p = pt[pt['pitcher_name'].str.contains('Harrison', case=False, na=False)]
  print(p[['pitcher_name','team_canonical_id','d1b_ability_adj','d1b_ability_source','fb_sensitivity','throws']])
  "
  ```

- [ ] **Step 6: Commit**

---

## Chunk 3: Unified Team Table

### Task 3: Build `build_team_table.py`

**Files:**
- Create: `scripts/build_team_table.py`
- Create: `tests/test_team_table.py`
- Reference: `scripts/compute_bullpen_quality.py` (bullpen quality logic)
- Reference: `scripts/predict_day.py` (wRC+ adjustment, lines 412-449)

**Output: `data/processed/team_table.csv`**

```
canonical_id        # Universal join key
team_idx            # Stan model team index (0 = not in model)
team_name           # Display name
conference          # SEC, ACC, Big 12, etc.
season              # 2026
bullpen_quality_z   # Composite z-score (from compute_bullpen_quality)
bullpen_adj         # Pre-computed: -score * 0.1 (log-rate)
wrc_plus            # Team mean wRC+ from D1B batting_advanced.tsv
wrc_offense_adj     # Pre-computed: (wRC+ - 100) / sigma * 0.109
n_games             # Games played (from games.csv)
```

**Sources merged:**
1. `canonical_teams_2026.csv` — base team info
2. `run_event_team_index.csv` — Stan model team_idx
3. `bullpen_quality.csv` — bullpen quality z-score
4. `d1baseball/batting_advanced.tsv` — team wRC+
5. `d1baseball_crosswalk.csv` — D1B team name → canonical_id

- [ ] **Step 1: Write tests**
- [ ] **Step 2: Run tests — verify they fail**
- [ ] **Step 3: Implement `build_team_table.py`**
- [ ] **Step 4: Run tests — verify they pass**
- [ ] **Step 5: Integration test**
- [ ] **Step 6: Commit**

---

## Chunk 4: Decompose predict_day.py

### Task 4a: `resolve_schedule.py`

**Files:**
- Create: `scripts/resolve_schedule.py`

**Extracts:** predict_day.py PHASE 7 (lines 484-553) — NCAA/ESPN API calls for game schedule.

**Output: `data/daily/schedule_YYYY-MM-DD.csv`**
```
game_num, home_name, away_name, home_cid, away_cid,
home_team_idx, away_team_idx,
start_utc, start_local_hour, venue_cid
```

**Interface:**
```python
def resolve_schedule(date: str, canonical_csv: Path, team_index_csv: Path,
                     meta_json: Path, odds_jsonl: Path | None = None,
                     out_csv: Path | None = None) -> pd.DataFrame:
```

- [ ] **Step 1: Write tests** (mock NCAA/ESPN API responses)
- [ ] **Step 2: Run tests — verify fail**
- [ ] **Step 3: Implement** (extract API fetching logic from predict_day.py)
- [ ] **Step 4: Run tests — verify pass**
- [ ] **Step 5: Commit**

### Task 4b: `resolve_starters.py`

**Files:**
- Create: `scripts/resolve_starters.py`

**Extracts:** predict_day.py PHASE 8b-8f (lines 592-659) — starter lookup + D1B enrichment.

**Reads:** schedule CSV + `pitcher_table.csv` + `team_table.csv`

**Output: `data/daily/starters_YYYY-MM-DD.csv`**
```
game_num, home_starter, away_starter,
home_starter_idx, away_starter_idx,
hp_throws, ap_throws,
hp_ability_adj, ap_ability_adj,
hp_ability_src, ap_ability_src,
hp_fb_sens, ap_fb_sens,
hp_bp_fb_sens, ap_bp_fb_sens,
home_wrc_adj, away_wrc_adj,
platoon_adj_home, platoon_adj_away
```

**Key simplification:** Instead of rebuilding D1B lookup maps at runtime (200+ lines), just read pre-computed values from `pitcher_table.csv`. The starter resolution still needs the `StarterLookup` class for rotation projection, but enrichment is a simple join.

- [ ] **Step 1: Write tests**
- [ ] **Step 2: Run tests — verify fail**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run tests — verify pass**
- [ ] **Step 5: Commit**

### Task 4c: `resolve_weather.py`

**Files:**
- Create: `scripts/resolve_weather.py`

**Extracts:** predict_day.py PHASE 8g (lines 661-740) — weather fetching + park factor computation.

**Reads:** schedule CSV + `stadium_orientations.csv` + `park_factors.csv` + Open-Meteo API

**Output: `data/daily/weather_YYYY-MM-DD.csv`**
```
game_num, home_cid,
park_factor, wind_adj_raw, non_wind_adj,
wind_out_mph, wind_out_lf, wind_out_cf, wind_out_rf,
temp_f, wind_mph, wind_dir_deg,
weather_mode, elevation_ft
```

- [ ] **Step 1: Write tests** (mock Open-Meteo API)
- [ ] **Step 2: Run tests — verify fail**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run tests — verify pass**
- [ ] **Step 5: Commit**

### Task 4d: `simulate.py` — Pure Monte Carlo Engine

**Files:**
- Create: `scripts/simulate.py`
- Create: `tests/test_simulate.py`

**Extracts:** predict_day.py PHASE 6 + PHASE 8h (lines 451-825) — posterior loading + simulation loop.

**This is the most critical piece.** The simulation engine takes pre-resolved inputs and produces predictions. NO API calls. NO name resolution. NO file discovery. Pure math.

**Interface:**
```python
def simulate_games(
    schedule_df: pd.DataFrame,      # from resolve_schedule
    starters_df: pd.DataFrame,      # from resolve_starters
    weather_df: pd.DataFrame,       # from resolve_weather
    posterior_csv: Path,             # Stan posterior draws
    meta_json: Path,                # {N_teams, N_pitchers, n_draws}
    team_table_csv: Path,           # bullpen quality, etc.
    n_sims: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run Monte Carlo simulation for all games.

    Returns DataFrame with one row per game:
      home, away, home_win_prob, ml_home, ml_away,
      exp_home, exp_away, exp_total, ...
    """
```

**Key property:** Given identical inputs, `simulate_games()` produces identical output. Deterministic. Testable. No side effects.

- [ ] **Step 1: Write tests for simulation engine**

  ```python
  def test_simulate_neutral_game():
      """Two identical teams at neutral park → ~50% win prob."""
      # Set up: both teams idx=1, both pitchers idx=1, no weather
      schedule = pd.DataFrame([{
          "game_num": 0, "home_cid": "A", "away_cid": "B",
          "home_team_idx": 1, "away_team_idx": 1,
      }])
      starters = pd.DataFrame([{
          "game_num": 0,
          "home_starter_idx": 1, "away_starter_idx": 1,
          "hp_ability_adj": 0.0, "ap_ability_adj": 0.0,
          "hp_fb_sens": 1.0, "ap_fb_sens": 1.0,
          "hp_bp_fb_sens": 1.0, "ap_bp_fb_sens": 1.0,
          "home_wrc_adj": 0.0, "away_wrc_adj": 0.0,
          "platoon_adj_home": 0.0, "platoon_adj_away": 0.0,
      }])
      weather = pd.DataFrame([{
          "game_num": 0,
          "park_factor": 0.0, "wind_adj_raw": 0.0, "non_wind_adj": 0.0,
      }])
      # ... load real posterior for indices 1
      result = simulate_games(schedule, starters, weather, ...)
      # Home advantage → slightly above 50%
      assert 0.48 < result.iloc[0]["home_win_prob"] < 0.58


  def test_simulate_deterministic():
      """Same inputs + same seed → same output."""
      r1 = simulate_games(schedule, starters, weather, ..., seed=42)
      r2 = simulate_games(schedule, starters, weather, ..., seed=42)
      assert r1.equals(r2)
  ```

- [ ] **Step 2: Run tests — verify fail**
- [ ] **Step 3: Implement `simulate.py`**

  Move the posterior loading (PHASE 6) and simulation loop (PHASE 8h) from predict_day.py. The simulation loop is ~80 lines of NumPy — it stays almost identical, just receives its inputs as function arguments instead of computing them inline.

- [ ] **Step 4: Run tests — verify pass**
- [ ] **Step 5: Integration test — compare with current predict_day.py output**

  ```bash
  # Run old pipeline
  .venv/bin/python3 scripts/predict_day.py --date 2026-03-14 --N 1000 --seed 42 \
      --out data/processed/pred_old.csv

  # Run new pipeline
  .venv/bin/python3 scripts/resolve_schedule.py --date 2026-03-14
  .venv/bin/python3 scripts/resolve_starters.py --date 2026-03-14
  .venv/bin/python3 scripts/resolve_weather.py --date 2026-03-14
  .venv/bin/python3 scripts/simulate.py --date 2026-03-14 --N 1000 --seed 42 \
      --out data/processed/pred_new.csv

  # Compare
  .venv/bin/python3 -c "
  import pandas as pd
  old = pd.read_csv('data/processed/pred_old.csv')
  new = pd.read_csv('data/processed/pred_new.csv')
  merged = old.merge(new, on=['home','away'], suffixes=('_old','_new'))
  print(f'Games matched: {len(merged)}/{len(old)}')
  diff = abs(merged['home_win_prob_old'] - merged['home_win_prob_new'])
  print(f'Max win prob diff: {diff.max():.4f}')
  print(f'Mean win prob diff: {diff.mean():.4f}')
  "
  ```

- [ ] **Step 6: Commit**

### Task 4e: Slim `predict_day.py` to Orchestrator

**Files:**
- Modify: `scripts/predict_day.py` (replace 820 lines with ~80-line wrapper)

**New predict_day.py:**
```python
"""
Daily prediction pipeline orchestrator.

Calls four focused scripts in sequence:
  1. resolve_schedule.py  — fetch game schedule
  2. resolve_starters.py  — project starters + enrich
  3. resolve_weather.py   — fetch weather + park factors
  4. simulate.py          — Monte Carlo simulation

Usage:
  python3 scripts/predict_day.py --date 2026-03-14
  python3 scripts/predict_day.py --date 2026-03-14 --N 5000 --no-weather
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from resolve_schedule import resolve_schedule
from resolve_starters import resolve_starters
from resolve_weather import resolve_weather
from simulate import simulate_games, format_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily NCAA baseball predictions.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--N", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-weather", action="store_true")
    # File paths (all have sensible defaults)
    parser.add_argument("--posterior", type=Path,
                        default=Path("data/processed/run_event_posterior_2k.csv"))
    parser.add_argument("--meta", type=Path,
                        default=Path("data/processed/run_event_fit_meta.json"))
    parser.add_argument("--pitcher-table", type=Path,
                        default=Path("data/processed/pitcher_table.csv"))
    parser.add_argument("--team-table", type=Path,
                        default=Path("data/processed/team_table.csv"))
    parser.add_argument("--stadium-csv", type=Path,
                        default=Path("data/registries/stadium_orientations.csv"))
    args = parser.parse_args()

    daily_dir = Path(f"data/daily/{args.date}")
    daily_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Schedule
    print("Step 1/4: Resolving schedule...", file=sys.stderr)
    schedule_csv = daily_dir / "schedule.csv"
    schedule = resolve_schedule(
        date=args.date,
        team_table_csv=args.team_table,
        out_csv=schedule_csv,
    )
    print(f"  {len(schedule)} games", file=sys.stderr)

    # Step 2: Starters
    print("Step 2/4: Resolving starters...", file=sys.stderr)
    starters_csv = daily_dir / "starters.csv"
    starters = resolve_starters(
        schedule_csv=schedule_csv,
        pitcher_table_csv=args.pitcher_table,
        team_table_csv=args.team_table,
        date=args.date,
        out_csv=starters_csv,
    )

    # Step 3: Weather
    print("Step 3/4: Fetching weather...", file=sys.stderr)
    weather_csv = daily_dir / "weather.csv"
    if args.no_weather:
        # Write zeros
        weather = schedule[["game_num"]].copy()
        weather["park_factor"] = 0.0
        weather["wind_adj_raw"] = 0.0
        weather["non_wind_adj"] = 0.0
        weather.to_csv(weather_csv, index=False)
    else:
        weather = resolve_weather(
            schedule_csv=schedule_csv,
            stadium_csv=args.stadium_csv,
            date=args.date,
            out_csv=weather_csv,
        )

    # Step 4: Simulate
    print("Step 4/4: Running simulation...", file=sys.stderr)
    predictions = simulate_games(
        schedule_csv=schedule_csv,
        starters_csv=starters_csv,
        weather_csv=weather_csv,
        posterior_csv=args.posterior,
        meta_json=args.meta,
        team_table_csv=args.team_table,
        n_sims=args.N,
        seed=args.seed,
    )

    # Output
    out_csv = args.out or Path(f"data/processed/predictions_{args.date}.csv")
    predictions.to_csv(out_csv, index=False)
    print(f"\nWrote {len(predictions)} predictions → {out_csv}", file=sys.stderr)

    if args.json:
        import json
        print(json.dumps(predictions.to_dict("records"), indent=2))
    else:
        format_predictions(predictions, args.date)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 1: Rename old predict_day.py**
  ```bash
  cp scripts/predict_day.py scripts/predict_day_legacy.py
  ```
- [ ] **Step 2: Write new predict_day.py orchestrator**
- [ ] **Step 3: End-to-end test**
  ```bash
  .venv/bin/python3 scripts/predict_day.py --date 2026-03-14 --N 100
  ```
- [ ] **Step 4: Commit**

---

## Chunk 5: Makefile + Pipeline Verification

### Task 5: Create Makefile

**Files:**
- Create: `Makefile`

```makefile
# NCAA Baseball Prediction Pipeline
# Usage:
#   make extract        # Re-extract from ESPN JSONL (after new scrape)
#   make tables         # Rebuild pitcher + team tables
#   make model          # Refit Stan model
#   make predict        # Run daily predictions (set DATE=YYYY-MM-DD)
#   make all            # Full rebuild from raw data

PYTHON = .venv/bin/python3
DATE ?= $(shell date +%Y-%m-%d)

# ── Layer 1: Extract ──────────────────────────────────────────────
ESPN_JSONL = $(wildcard data/raw/espn/games_*.jsonl)
EXTRACTED = data/processed/extract_manifest.json

$(EXTRACTED): scripts/extract_espn.py $(ESPN_JSONL) data/registries/canonical_teams_2026.csv
	$(PYTHON) scripts/extract_espn.py
	@echo "✓ Extract complete"

extract: $(EXTRACTED)

# ── Layer 2: Indices ──────────────────────────────────────────────
TEAM_INDEX = data/processed/run_event_team_index.csv
PITCHER_INDEX = data/processed/run_event_pitcher_index.csv

$(TEAM_INDEX) $(PITCHER_INDEX): scripts/build_run_event_indices.py data/processed/run_events.csv
	$(PYTHON) scripts/build_run_event_indices.py

indices: $(TEAM_INDEX) $(PITCHER_INDEX)

# ── Layer 2b: Park Factors ────────────────────────────────────────
PARK_FACTORS = data/processed/park_factors.csv

$(PARK_FACTORS): scripts/build_park_factors.py data/processed/venue_stats.csv
	$(PYTHON) scripts/build_park_factors.py

park_factors: $(PARK_FACTORS)

# ── Layer 2c: Bullpen Quality ─────────────────────────────────────
BULLPEN = data/processed/bullpen_quality.csv

$(BULLPEN): scripts/compute_bullpen_quality.py data/processed/pitcher_appearances.csv
	$(PYTHON) scripts/compute_bullpen_quality.py

bullpen: $(BULLPEN)

# ── Layer 2d: Weekend Rotations ───────────────────────────────────
ROTATIONS = data/processed/weekend_rotations.csv

$(ROTATIONS): scripts/build_weekend_rotations.py data/processed/pitcher_appearances.csv
	$(PYTHON) scripts/build_weekend_rotations.py

rotations: $(ROTATIONS)

# ── Layer 3: Unified Tables ───────────────────────────────────────
PITCHER_TABLE = data/processed/pitcher_table.csv

$(PITCHER_TABLE): scripts/build_pitcher_table.py $(PITCHER_INDEX) $(ROTATIONS) \
                   data/raw/d1baseball/pitching_advanced.tsv \
                   data/raw/d1baseball/pitching_batted_ball.tsv \
                   data/processed/d1baseball_rotations.csv
	$(PYTHON) scripts/build_pitcher_table.py
	@echo "✓ Pitcher table built"

TEAM_TABLE = data/processed/team_table.csv

$(TEAM_TABLE): scripts/build_team_table.py $(TEAM_INDEX) $(BULLPEN) \
                data/raw/d1baseball/batting_advanced.tsv
	$(PYTHON) scripts/build_team_table.py
	@echo "✓ Team table built"

tables: $(PITCHER_TABLE) $(TEAM_TABLE)

# ── Layer 4: Model Fit ────────────────────────────────────────────
POSTERIOR = data/processed/run_event_posterior_2k.csv
META = data/processed/run_event_fit_meta.json

$(POSTERIOR) $(META): scripts/fit_run_event_model.py $(TEAM_INDEX) $(PITCHER_INDEX) \
                       $(PARK_FACTORS) $(BULLPEN) stan/ncaa_baseball_run_events.stan
	$(PYTHON) scripts/fit_run_event_model.py
	# Subsample to 2K draws
	head -1 data/processed/run_event_posterior.csv > $(POSTERIOR)
	tail -n +2 data/processed/run_event_posterior.csv | shuf -n 2000 >> $(POSTERIOR)
	@echo "✓ Model fit complete"

model: $(POSTERIOR)

# ── Layer 5: Daily Predict ────────────────────────────────────────
PREDICTIONS = data/processed/predictions_$(DATE).csv

$(PREDICTIONS): scripts/predict_day.py $(POSTERIOR) $(PITCHER_TABLE) $(TEAM_TABLE)
	$(PYTHON) scripts/predict_day.py --date $(DATE) --N 5000 --out $@
	cp $@ ~/Desktop/ncaaBases_projections_$(DATE).csv
	@echo "✓ Predictions for $(DATE)"

predict: $(PREDICTIONS)

# ── Odds ──────────────────────────────────────────────────────────
odds:
	source ~/.zshrc && ODDS_API_KEY="$$THE_ODDS_API_KEY" \
		$(PYTHON) scripts/pull_odds.py --mode current --regions us,us2,eu --markets h2h,totals,spreads

# ── Convenience ───────────────────────────────────────────────────
all: extract indices park_factors bullpen rotations tables model predict

rebuild: extract indices park_factors bullpen rotations tables

daily: predict odds

clean-daily:
	rm -rf data/daily/$(DATE)

.PHONY: extract indices park_factors bullpen rotations tables model predict odds all rebuild daily clean-daily
```

- [ ] **Step 1: Write Makefile**
- [ ] **Step 2: Test `make extract`**
- [ ] **Step 3: Test `make tables`**
- [ ] **Step 4: Test `make predict DATE=2026-03-14`**
- [ ] **Step 5: Test `make daily DATE=2026-03-14`**
- [ ] **Step 6: Commit**

---

## Chunk 6: Cleanup + Documentation

### Task 6: Update CLAUDE.md and deprecate old scripts

- [ ] **Step 1: Move deprecated scripts to `scripts/deprecated/`**
  ```bash
  mkdir -p scripts/deprecated
  mv scripts/build_run_events_from_espn.py scripts/deprecated/
  mv scripts/build_pitcher_registry.py scripts/deprecated/
  mv scripts/build_games_from_espn.py scripts/deprecated/
  mv scripts/build_pitching_from_espn.py scripts/deprecated/
  mv scripts/build_bullpen_fatigue.py scripts/deprecated/
  mv scripts/predict_day_legacy.py scripts/deprecated/
  ```

- [ ] **Step 2: Update CLAUDE.md**
  - Replace pipeline description with Makefile-based workflow
  - Update Critical Files table
  - Add new file descriptions
  - Update Daily Pipeline Commands to use `make predict`

- [ ] **Step 3: Update MEMORY.md**
  - Note pipeline refactor completion
  - Update data flow description

- [ ] **Step 4: Final integration test**
  ```bash
  # Full pipeline from scratch
  make rebuild
  make predict DATE=2026-03-14
  # Verify output matches expectations
  ```

- [ ] **Step 5: Commit all cleanup**

---

## Verification Checklist

After all chunks complete:

- [ ] `make extract` produces identical `run_events.csv` to old pipeline (row counts match, team resolutions match)
- [ ] `pitcher_table.csv` contains all 7591+ pitchers with correct columns
- [ ] `team_table.csv` contains all 308+ teams with correct columns
- [ ] `make predict DATE=2026-03-14` produces predictions within ±0.01 win prob of old pipeline (same seed)
- [ ] No API calls happen inside `simulate.py`
- [ ] `simulate.py` is deterministic (same seed → same output)
- [ ] `data/daily/YYYY-MM-DD/` directory structure works (schedule, starters, weather CSVs)
- [ ] `make daily` runs the full daily workflow (predict + odds)
- [ ] Old deprecated scripts still work if needed (in `scripts/deprecated/`)
- [ ] All tests pass: `pytest tests/ -v`
