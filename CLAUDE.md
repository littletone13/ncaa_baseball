# NCAA Baseball Prediction Model

## Project Overview
NCAA Division 1 baseball prediction model that simulates games using starting pitcher quality, bullpen strength, park factors, and live weather (wind direction relative to stadium orientation). Outputs daily projections with moneylines, totals, and edges vs market odds.

## Architecture

### Data Flow (Refactored Pipeline)
```
── BUILD PHASE (offline, run via Makefile) ──────────────
games_2026.jsonl → extract_espn.py → games.csv, run_events.csv,
                                       pitcher_appearances.csv, venue_stats.csv

run_events.csv → build_run_event_indices.py → team_index.csv, pitcher_index.csv
              → fit_run_event_model.py → run_event_posterior.csv (Stan, 4 chains)

pitcher_appearances + D1B stats + Stan indices → build_pitcher_table.py → pitcher_table.csv
canonical_teams + Stan indices + bullpen/wRC+ → build_team_table.py → team_table.csv

── PREDICT PHASE (daily, orchestrated by predict_day.py) ──
predict_day.py calls 4 modules in sequence:
  1. resolve_schedule.py  → data/daily/{date}/schedule.csv   (NCAA + ESPN + Odds APIs)
  2. resolve_starters.py  → data/daily/{date}/starters.csv   (StarterLookup + pitcher_table)
  3. resolve_weather.py   → data/daily/{date}/weather.csv    (Open-Meteo + park factors)
  4. simulate.py          → predictions CSV                   (Monte Carlo, 5000 draws)

pull_odds.py → odds JSONL → compare model vs market
```

### Key Directories
- `scripts/` — All pipeline scripts (Python + R)
- `scripts/deprecated/` — Replaced scripts (kept for reference)
- `src/ncaa_baseball/` — Package code (phase1 team resolution)
- `stan/` — Stan model files
- `data/registries/` — Canonical teams, stadium orientations, crosswalks
- `data/processed/` — Predictions, pitcher/team tables, model outputs
- `data/daily/{date}/` — Intermediate pipeline outputs (schedule, starters, weather CSVs)
- `data/raw/odds/` — Odds pulls (JSONL, append-only log)
- `docs/` — Design docs and plans

### Critical Files — Build Phase
| File | Purpose |
|------|---------|
| `scripts/extract_espn.py` | Single-pass ESPN JSONL → games, run_events, appearances, venues |
| `scripts/build_run_event_indices.py` | Build team/pitcher index CSVs from run_events |
| `scripts/fit_run_event_model.py` | Fit Stan model via CmdStanPy (4 chains, 10K iter) |
| `scripts/build_pitcher_table.py` | Unified pitcher lookup (Stan + D1B + handedness + FB%) |
| `scripts/build_team_table.py` | Unified team lookup (Stan + bullpen + wRC+) |
| `data/processed/pitcher_table.csv` | 9K+ pitchers with ability, handedness, FB sensitivity |
| `data/processed/team_table.csv` | 308 teams with team_idx, bullpen_z, wrc_offense_adj |

### Critical Files — Predict Phase
| File | Purpose |
|------|---------|
| `scripts/predict_day.py` | Thin orchestrator calling 4 modules below |
| `scripts/resolve_schedule.py` | NCAA + ESPN + Odds APIs → game schedule |
| `scripts/resolve_starters.py` | StarterLookup + pitcher_table → projected starters |
| `scripts/resolve_weather.py` | Open-Meteo + park factors → per-game weather |
| `scripts/simulate.py` | Pure Monte Carlo engine (no API calls, deterministic) |

### Critical Files — Other
| File | Purpose |
|------|---------|
| `scripts/pull_odds.py` | Odds data fetcher (current + historical) |
| `scripts/weather_park_adjustment.py` | Wind/temp → run scoring adjustment (directional wind model) |
| `scripts/lookup_starters.py` | Starting pitcher inference (StarterLookup class) |
| `scripts/platoon_adjustment.py` | LHP/RHP platoon lookup (currently disabled) |
| `data/registries/canonical_teams_2026.csv` | Team registry with odds_api_name + espn_name mappings |
| `data/registries/stadium_orientations.csv` | Stadium lat/lon + HP→CF bearings |
| `data/registries/d1baseball_crosswalk.csv` | D1Baseball team name → canonical_id (308 entries) |
| `Makefile` | Pipeline DAG (extract → indices → tables → model → predict) |

## Environment Setup
- Python 3.12+ with `.venv/` virtual environment
- Always run scripts with `.venv/bin/python3`
- Odds API key: set `ODDS_API_KEY` env var or in `.env` file
- Weather: Open-Meteo (free, no key needed)
- Stan: CmdStan for model fitting

## Conventions

### Team Resolution
- Every team has a `canonical_id` (e.g., `BSB_TEXAS_TECH`, `NCAA_614704`)
- NCAA API names → canonical via `src/ncaa_baseball/phase1.py`
- Odds API names → canonical via `odds_api_name` column in `canonical_teams_2026.csv`
- ESPN names → canonical via `espn_name` column (full mascot format, e.g. "Arizona State Sun Devils")
- `build_odds_name_to_canonical()` indexes BOTH `odds_api_name` and `espn_name` for exact matching
- **Prefix matching pitfall**: "Florida State" must NOT match "Florida" — `espn_name` exact match prevents this
- When odds/ESPN names don't resolve, add the mapping to `canonical_teams_2026.csv`
- 276/308 D1 teams have ESPN run_event data; 32 small programs (Ivy, NEC, etc.) use wRC+ fallback only

### Pitcher ID Crosswalk (NCAA→ESPN)
- Run events use numeric ESPN IDs (`64890.0`); pitcher appearances scraper assigns `NCAA_` format IDs
- Same pitcher may appear at two registry indices — ESPN (1–1743, has posterior data) and NCAA (1744+, prior-only)
- `lookup_starters.py` has `_build_ncaa_espn_crosswalk()` that maps NCAA_ IDs → ESPN indices via (name, team) matching
- Crosswalk reads `espn_name` column from `canonical_teams_2026.csv` — if this column is missing, crosswalk produces 0 matches
- Only ~285/7591 NCAA pitchers have ESPN counterparts — most pitchers have no learned posterior (prior-dominated)
- Do NOT use fuzzy matching for team names — use explicit `espn_name` mappings only

### D1Baseball Advanced Stats Fallback
- **Pitcher ability (FIP percentile mapping)**: For pitchers with idx=0 (no posterior), map FIP percentile in D1B distribution → z-score × posterior_ability_std (0.05). Fallback chain: FIP → SIERA → ERA (rotation page). ~856 pitchers covered.
- **Team offense (wRC+ adjustment)**: For teams with team_idx=0 (not in Stan model), aggregate team mean wRC+ from `batting_advanced.tsv`, map to att adjustment via `(wRC+ - 100) / wRC_std * att_std`. 32 non-model teams get this adjustment.
- **D1B crosswalk**: `data/registries/d1baseball_crosswalk.csv` maps D1B team names → canonical_id. Handles Unicode apostrophe (U+2019) vs ASCII apostrophe normalization.
- **D1B stats are from pre-scraped leaderboard TSVs** in `data/raw/d1baseball/` — individual player pages are paywalled

### Stadium Bearings (HP→CF)
- `hp_bearing_deg`: Compass bearing from home plate toward center field
- Batter at home plate looks toward CF at this bearing
- Default is 67° (NE, the "Johnnian" convention) — MUST be replaced with real measurements
- Wind-out = `wind_speed * cos(wind_toward - hp_bearing)` — positive = wind carries balls out
- Source field tracks provenance: `osm_polygon_estimated`, `osm_polygon_verified`, `osm_wider_1200m`, etc.

### Odds
- Stored as JSONL in `data/raw/odds/odds_latest.jsonl` (overwritten each pull)
- Append-only log in `data/raw/odds/odds_pull_log.jsonl`
- Market data uses `bookmaker_lines` field (not `bookmakers`)
- Each bookmaker entry uses `bookmaker_key` (not `key`) for the sportsbook identifier
- American odds conversion: negative → `|ml|/(|ml|+100)`, positive → `100/(ml+100)`
- Available markets: `h2h`, `totals`, `spreads` — pass all three to `pull_odds.py --markets`
- Live games have shifted odds — filter by `commence_time` vs now to exclude in-progress games

### Weather & Altitude Model
- **Directional wind**: 5-point arc-weighted sampling across 90° outfield arc
  - LF foul pole (0.10) → LF power alley (0.25) → CF (0.30) → RF power alley (0.25) → RF foul pole (0.10)
  - LF bearing = CF bearing − 45°, RF bearing = CF bearing + 45°
  - `wind_out_directional()` in `weather_park_adjustment.py` computes weighted effective wind
- Wind: ~0.008 log-rate/mph → 10 mph tailwind ≈ +8% runs/game (validated vs Nathan's +4%/5mph)
- Temperature: ~0.002 log-rate/°F above 72°F baseline
- Altitude: 0.25 × (1 − air_density_ratio) → Air Force (6686 ft) = +5.6% runs
- Negative wind_out = wind blowing IN (suppresses scoring)
- Crosswind correctly produces near-zero effective wind (symmetric arc cancels)
- `elevation_ft` column in `stadium_orientations.csv` sourced from Open-Meteo elevation API

## Daily Pipeline Commands

### Full daily run (predictions + odds + compare)
```bash
# 1. Run simulation (orchestrator calls resolve_schedule → resolve_starters → resolve_weather → simulate)
.venv/bin/python3 scripts/predict_day.py --date YYYY-MM-DD --N 5000 --out data/processed/predictions_YYYY-MM-DD.csv

# Or via Makefile
make predict DATE=YYYY-MM-DD

# 2. Pull fresh odds
source ~/.zshrc; ODDS_API_KEY="$THE_ODDS_API_KEY" .venv/bin/python3 scripts/pull_odds.py --mode current --regions us,us2,eu --markets h2h,totals,spreads

# 3. Compare (see odds comparison skill)
```

### Makefile targets
```bash
make rebuild    # Full rebuild: extract → indices → tables → model
make predict    # Daily prediction (DATE=YYYY-MM-DD)
make daily      # Pull odds + predict today
make all        # Full rebuild + predict
```

### Stadium bearing measurement
```bash
# All default-bearing stadiums
.venv/bin/python3 scripts/measure_stadium_bearings.py --only-default

# Retry failures with wider search
.venv/bin/python3 scripts/measure_bearings_retry.py

# Single stadium
.venv/bin/python3 scripts/measure_stadium_bearings.py --canonical-id BSB_TEXAS_TECH
```

### Prediction CSV Columns
- Team names: `home`, `away` (not `home_team`/`away_team`)
- Expected runs: `exp_home`, `exp_away`, `exp_total` (not `mean_*`)
- Win probability: `home_win_prob`, `away_win_prob`; moneylines: `ml_home`, `ml_away`
- Pitcher handedness: `hp_throws`, `ap_throws` (may be blank if unknown)
- Platoon: `platoon_adj_home`, `platoon_adj_away` (currently 0.0, disabled)
- D1B pitcher fallback: `hp_d1b_adj`, `ap_d1b_adj` (FIP/SIERA/ERA ability estimate), `hp_d1b_src`, `ap_d1b_src`
- D1B team offense: `home_wrc_adj`, `away_wrc_adj` (wRC+-based att adjustment for non-model teams)

### Platoon Splits (LHP/RHP)
- `scripts/platoon_adjustment.py` tracks pitcher handedness via D1B rotation data
- Rate adjustment disabled (`DEFAULT_LHP_ADJ = 0.0`) — sign uncertain, double-counts with `pitcher_ability`
- Stage 2 plan: add `platoon_effect` parameter to Stan model to learn sign/magnitude from data
- Batter handedness available on D1Baseball player pages (`BAT/THRW` field, e.g. `R/R`)
- D1Baseball lineup pages (`/team/{slug}/lineup/`) show game-by-game batting orders (free, no login)

### Model Refit Pipeline
When run_events data or team/pitcher indices change, the Stan model must be refit:
```bash
# Via Makefile (recommended — handles all dependencies):
make rebuild

# Or manually:
# 1. Extract from ESPN JSONL (single-pass, replaces 6 old scripts)
.venv/bin/python3 scripts/extract_espn.py

# 2. Rebuild team/pitcher indices
.venv/bin/python3 scripts/build_run_event_indices.py

# 3. Refit Stan model (~10 min on M-series Mac)
.venv/bin/python3 scripts/fit_run_event_model.py

# 4. Rebuild unified lookup tables
.venv/bin/python3 scripts/build_pitcher_table.py
.venv/bin/python3 scripts/build_team_table.py

# 5. Subsample posterior for daily predictions
python3 -c "import pandas as pd; d=pd.read_csv('data/processed/run_event_posterior.csv'); d.sample(2000,random_state=42).to_csv('data/processed/run_event_posterior_2k.csv',index=False)"
```

## Common Pitfalls
- Overpass API rate limits: Use 4-5s between queries, retry on 429/504 with exponential backoff
- `bookmaker_lines` not `bookmakers` in odds JSONL
- Stadium default 67° will produce wrong wind calculations — always verify bearing source
- NCAA API game names don't match odds API names — use canonical_id as join key
- The ODDS_API_KEY is stored in shell env as `$THE_ODDS_API_KEY` (source ~/.zshrc first)
- Pitcher ID formats: run_events uses numeric ESPN IDs, appearances uses `NCAA_` format — crosswalk in `lookup_starters.py` bridges them
- `canonical_teams_2026.csv` columns: `team_name` (short), `odds_api_name` (full mascot for odds), `espn_name` (full mascot for ESPN pitcher registry), `baseballr_team_name`
- Prediction CSV columns: `home`/`away` (not `home_team`), `exp_total` (not `mean_total`), `home_starter_idx`/`away_starter_idx` (not `hp_pitcher_idx`)
- **ESPN team name prefix collision**: "Florida State Seminoles" previously matched "Florida" via prefix matching — fixed by indexing `espn_name` for exact match in `build_odds_name_to_canonical()`
- **D1B apostrophe mismatch**: D1Baseball TSVs use Unicode U+2019 (`'`) while crosswalk uses ASCII `'` — normalized in both directions during lookup
- **Team indices beyond posterior**: After adding teams to team_index, team_idx > N_teams_in_posterior gets clamped to 0 (league avg + wRC+ fallback) until model is refit
