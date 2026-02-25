# Probable / announced starters pipeline (NCAA D1)

Goal: produce a repeatable daily starter table for **today's games** that existing scripts can consume:
- `scripts/project_game.py` (`--home-sp-id`, `--away-sp-id`, ESPN pitcher IDs)
- run-event simulation flow (`--home-pitcher`, `--away-pitcher`, pitcher indices from `run_event_pitcher_index.csv`)

---

## 1) Source decision and reliability

### What was evaluated

1. **ESPN API (`scoreboard`, `summary`)**
   - Reliable for schedule and in-progress/final boxscore starters.
   - For most pregame NCAA events, ESPN does **not** expose probable starters in `scoreboard`/`summary`.

2. **The Odds API (`baseball_ncaa`)**
   - Good for odds markets (`h2h`, `spreads`, `totals`).
   - Does **not** provide NCAA probable pitcher fields.

3. **D1Baseball**
   - Public team lineup pages (`/team/<slug>/lineup/`) include starter columns by game.
   - Same-day rows can be used when posted, but coverage/timing is not guaranteed for all games.
   - API exists but is auth-gated; no credentials are committed in this repo.

4. **Inference from internal historical starts**
   - Always available when `data/processed/pitching_lines_espn.csv` exists.
   - Lower confidence than announced starters, but robust fallback.

### Chosen v1 strategy (implemented)

Per-game, per-side source priority:
1. **Manual announced input** (from D1Baseball/team sites/social/beat notes)
2. **D1 same-day lineup scrape** (if row exists with SP)
3. **ESPN summary starter** (for games already started)
4. **Rotation inference** from `pitching_lines_espn.csv`
5. **Unknown** (graceful fallback; pipeline still writes output)

This gives a stable daily process with best-effort announced starters plus a deterministic fallback path.

---

## 2) Implemented scripts

## `scripts/build_todays_starters.py`

Builds `data/processed/todays_starters.csv` for one date.

Key behaviors:
- Pulls today's games from ESPN scoreboard (`event_id`, teams, status, time).
- Resolves teams to canonical IDs (`canonical_teams_2026.csv`).
- Optionally writes a manual template CSV with game rows to fill announced starters.
- Applies source priority (manual -> D1 -> ESPN summary -> inference -> unknown).
- Resolves pitcher names to ESPN IDs using historical starts when possible.
- Adds run-event indices if index files exist:
  - `run_event_pitcher_index.csv` -> `home_pitcher_idx`, `away_pitcher_idx`
  - `run_event_team_index.csv` -> `home_team_idx`, `away_team_idx`

### Main CLI

- `--date YYYY-MM-DD` (default: today)
- `--write-manual-template` (writes `data/raw/starters_manual/manual_starters_<date>.csv`)
- `--manual-template-only` (template and exit)
- `--manual-input` (override manual CSV path)
- `--include-final` (include finals; default excludes finals)
- `--disable-d1-lineups`
- `--disable-espn-summary`
- `--out` (default `data/processed/todays_starters.csv`)

## `scripts/run_today_with_starters.py`

Thin integration wrapper to consume the starter table and build/run downstream commands.

- `--mode project|simulate|both`
- `--execute` to run commands (without it, prints commands)
- `--project-script` (default `scripts/project_game.py`)
- `--simulate-script` (default `scripts/simulate_run_event_game.py`)
- `--project-extra` / `--simulate-extra`
- `--min-confidence`

`project` mode passes:
- `--team-a`, `--team-b`, `--game-date`, `--use-pitchers`
- plus `--home-sp-id`/`--away-sp-id` when available.

`simulate` mode passes:
- `--home-pitcher`/`--away-pitcher` using resolved pitcher indices.

---

## 3) Output schema (`todays_starters.csv`)

Primary keys / identity:
- `game_date`, `event_id`, `home_team_name`, `away_team_name`
- `home_canonical_id`, `away_canonical_id`

Starter fields:
- `home_pitcher_name`, `away_pitcher_name`
- `home_pitcher_espn_id`, `away_pitcher_espn_id`
- `home_pitcher_idx`, `away_pitcher_idx` (if run-event index exists)

Team index fields:
- `home_team_idx`, `away_team_idx` (if run-event team index exists)

Audit / reliability:
- `home_starter_source`, `away_starter_source`
- `home_starter_confidence`, `away_starter_confidence`
- `home_starter_note`, `away_starter_note`
- `status`, `commence_time`, `generated_at_utc`

---

## 4) Daily runbook

1. Create/refresh today's manual template:
   - `python3 scripts/build_todays_starters.py --date YYYY-MM-DD --write-manual-template --manual-template-only`
2. Fill `data/raw/starters_manual/manual_starters_YYYY-MM-DD.csv` with announced names/IDs where available.
3. Build final starter table:
   - `python3 scripts/build_todays_starters.py --date YYYY-MM-DD --write-manual-template`
4. Run projection commands:
   - Print only: `python3 scripts/run_today_with_starters.py --mode project`
   - Execute: `python3 scripts/run_today_with_starters.py --mode project --execute`

If a starter is `TBD`/unknown:
- leave manual fields blank;
- collector falls back to D1 same-day lineup, ESPN summary (if started), then inference;
- output still includes game row with source/confidence so downstream can decide skip vs use.

---

## 5) Maintenance notes

- No secrets are required for this v1 pipeline.
- If a paid/public API with reliable NCAA probable starters is adopted later, add it as a source **ahead of inference** and keep output schema unchanged.
- Keep identity system unified: canonical team IDs + ESPN pitcher IDs + run-event indices.
