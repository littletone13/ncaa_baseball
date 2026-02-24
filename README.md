# NCAA_BASEBALL (data-first)

Goal: build a clean, deterministic dataset pipeline (teams → odds → performance) that we can trust before we model anything.

## Canonical team library (2026)

**Single source of truth for 2026:** `data/registries/canonical_teams_2026.csv`. Built from the full NCAA D1 list plus the manual crosswalk (no fuzzy matching). See [docs/DATA_PLAN_2026.md](docs/DATA_PLAN_2026.md) for schema and roster plan.

```bash
# 1. Refresh NCAA D1 list from stats.ncaa.org (optional; already have 2026)
python3 scripts/scrape_ncaa_d1_team_registry.py --academic-year 2026

# 2. Edit data/registries/name_crosswalk_manual_2026.csv to fill canonical_team_id and odds_api_team_name where needed

# 3. Build the canonical registry
python3 scripts/build_canonical_teams_2026.py
```

Output: `data/registries/canonical_teams_2026.csv` (academic_year, ncaa_teams_id, team_name, conference, canonical_id, odds_api_name, …).

## Phase 1 build (model vs odds)

Resolve Odds API team names to canonical teams, apply **Elo** (if fitted) or prior-only win prob, compare to devigged moneyline. See [docs/MODEL_BUILD_REFINEMENTS.md](docs/MODEL_BUILD_REFINEMENTS.md).

**Full pipeline (ESPN games → Elo → compare to odds):**

```bash
# 1. Canonical teams (above)
python3 scripts/build_canonical_teams_2026.py

# 2. Build game-result table from ESPN JSONL (resolves team names to canonical)
python3 scripts/build_games_from_espn.py \
  --espn-dir data/raw/espn \
  --out data/processed/games_espn.csv

# 3. Fit Elo on resolved games
python3 scripts/fit_phase1_elo.py \
  --games data/processed/games_espn.csv \
  --out data/processed/phase1_team_ratings.csv

# 4. Compare model to odds (uses Elo when phase1_team_ratings.csv exists)
python3 scripts/run_phase1_odds_compare.py \
  --odds-jsonl data/raw/odds/odds_baseball_ncaa_20260221.jsonl \
  --ratings data/processed/phase1_team_ratings.csv \
  --out data/processed/phase1_compare.csv
```

**Outputs:**
- `data/processed/games_espn.csv` — one row per game (date, home/away, scores, canonical ids when resolved).
- `data/processed/phase1_team_ratings.csv` — Elo rating per team (canonical_id, elo_rating, n_games).
- `data/processed/phase1_compare.csv` — model_win_prob_home/away, market_fair_home/away, edge_home/away.

If `phase1_team_ratings.csv` is missing, step 4 uses prior-only (~52% home). Unresolved odds names are printed; add `odds_api_team_name` in `name_crosswalk_manual_2026.csv` and re-run canonical build + steps 2–4 to resolve more.

### Pitcher-aware projection (SP ratings, bullpen health, market blend)

After building games and pitching lines, build pitcher ratings and workload (Mack-style SP/RP + Peabody shrinkage + NoVig blend):

```bash
python3 scripts/build_pitcher_ratings.py
```

Outputs: `data/processed/pitcher_ratings.csv`, `team_pitcher_strength.csv`, `bullpen_workload.csv`.

**Single-game projection with SP and optional market blend:**

```bash
python3 scripts/project_game.py --team-a "Michigan" --team-b "Kansas St" --neutral --use-pitchers --game-date 2026-02-22
# With devigged market prob (e.g. 0.48 home) and n_games for blend (more market weight when low):
python3 scripts/project_game.py --team-a "Michigan" --team-b "Kansas St" --neutral --use-pitchers --market-fair-home 0.48 --n-games 6
```

Optional: `--home-sp-id`, `--away-sp-id` (ESPN pitcher ids) when starters are known. See [docs/MODEL_BUILD_REFINEMENTS.md](docs/MODEL_BUILD_REFINEMENTS.md) §2.5–2.6 (pitcher model, early-season profit).

## Rosters (2026)

2026 (2025–26) rosters use roster year_id **614802** (see `data/registries/ncaa_roster_year_id_lu.csv`).

```bash
pip install beautifulsoup4   # if not already installed
python3 scripts/scrape_rosters_2026.py --canonical data/registries/canonical_teams_2026.csv --out-dir data/raw/rosters --sleep 1
```

Output: `data/raw/rosters/roster_2026_<ncaa_teams_id>.csv` per team (player_id, player_name, and table columns).

**If you get 403:** stats.ncaa.org often blocks server-side requests. Try (1) browser-based scraping (Playwright/Selenium), (2) a different network or VPN, or (3) running from a machine that gets 200. The script is ready once the site allows requests.

## Team registry (curated list, BSB_ IDs)

`teams_baseball.yaml` is the source of truth for the **curated** team list with BSB_ canonical IDs (no fuzzy matching).

```bash
python3 scripts/build_team_registry.py teams_baseball.yaml --out data/registries/teams.csv
```

## Odds API (raw JSON archiving)

Set your key (avoid putting it in code or files you commit):

```bash
export ODDS_API_KEY="..."
```

Fetch historical events list at a timestamp:

```bash
python3 scripts/pull_odds_events.py --sport baseball_ncaa --date "2025-05-18T16:00:00Z"
```

Build a name crosswalk template from that events file (no fuzzy matching; exact school-name matches only):

```bash
python3 scripts/build_odds_name_crosswalk.py --events-json data/raw/odds/historical/events_baseball_ncaa_2025-05-18T160000Z.json
```

Fetch a historical odds snapshot (all games at that timestamp) and archive:

```bash
python3 scripts/pull_odds_snapshot.py --sport baseball_ncaa --date "2025-05-18T16:00:00Z" --markets h2h,spreads,totals --regions us
```

Each pull writes:
- `*.json` = API response body (unaltered)
- `*.meta.json` = request params + fetch time + response headers (credits remaining, etc.)

## R performance pipeline (baseballr)

Install packages:

```bash
Rscript scripts/install_R_deps.R
```

Pull performance data (examples):

```bash
# bulk historical PBP (can be huge)
Rscript scripts/01_scrape_pbp.R --seasons 2021-2025

# build full NCAA D1 team registry (~300) for the current academic year
python3 scripts/scrape_ncaa_d1_team_registry.py --academic-year 2026

# build deterministic crosswalk (canonical school -> baseballr team_id)
Rscript scripts/00_build_baseballr_crosswalk.R --year 2024

# rosters + team player stats for a season
Rscript scripts/02_scrape_rosters.R --season 2024 --sleep 1.0

# NOTE: team player stats endpoints currently fail (NCAA site path + anti-bot changes);
# scripts/03_scrape_player_stats.R will explain the blockage.
Rscript scripts/03_scrape_player_stats.R --season 2024
```

## Data audits / coverage

Scoreboard integrity check:

```bash
python3 scripts/audit_scoreboard_games.py --games-csv data/processed/scoreboard/games_2025.csv --json-out data/processed/reports/audit_games_2025.json
```

Build team-season run stats from scoreboard:

```bash
python3 scripts/build_team_game_stats.py --games-csv data/processed/scoreboard/games_2025.csv --out-team-stats data/processed/scoreboard/team_stats_2025.csv
```

Build a team registry (id→name) directly from scoreboard:

```bash
python3 scripts/build_scoreboard_team_registry.py --games-csv data/processed/scoreboard/games_2025.csv
```

Export rosters to CSV + missing-player-id list:

```bash
Rscript scripts/export_rosters_csv.R 2024
```

Coverage report (teams + scoreboard + rosters + pbp):

```bash
Rscript scripts/build_coverage_report.R
```
