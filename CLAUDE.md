# NCAA Baseball Prediction Model

## Project Overview
NCAA Division 1 baseball prediction model that simulates games using starting pitcher quality, bullpen strength, park factors, and live weather (wind direction relative to stadium orientation). Outputs daily projections with moneylines, totals, and edges vs market odds.

## Architecture

### Data Flow
```
NCAA API / ESPN → game schedule
pitcher_appearances.csv → starter inference (StarterLookup)
stadium_orientations.csv → HP→CF bearing + lat/lon
Open-Meteo API → live wind/temp at game time
the-odds-api → market lines (h2h, totals)
Stan model fit → team/pitcher parameters
────────────────────────────────────
predict_day.py → Monte Carlo simulation (5000 draws) → predictions CSV
pull_odds.py → odds JSONL → compare model vs market
```

### Key Directories
- `scripts/` — All pipeline scripts (Python + R)
- `src/ncaa_baseball/` — Package code (phase1 team resolution)
- `stan/` — Stan model files
- `data/registries/` — Canonical teams, stadium orientations, crosswalks
- `data/raw/odds/` — Odds pulls (JSONL, append-only log)
- `data/processed/` — Predictions, pitcher data, model outputs
- `docs/` — Design docs and plans

### Critical Files
| File | Purpose |
|------|---------|
| `scripts/predict_day.py` | Daily simulation pipeline (main entry point) |
| `scripts/pull_odds.py` | Odds data fetcher (current + historical) |
| `scripts/weather_park_adjustment.py` | Wind/temp → run scoring adjustment |
| `scripts/lookup_starters.py` | Starting pitcher inference |
| `scripts/measure_stadium_bearings.py` | OSM-based HP→CF bearing measurement |
| `scripts/measure_bearings_retry.py` | Retry with wider radii for missed stadiums |
| `data/registries/stadium_orientations.csv` | Stadium lat/lon + HP→CF bearings |
| `scripts/platoon_adjustment.py` | LHP/RHP platoon lookup (PlatoonLookup class, currently disabled) |
| `data/registries/canonical_teams_2026.csv` | Team registry with odds_api_name mappings |

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
- ESPN names → canonical via `espn_name` column (full mascot format, e.g. "Arizona Wildcats")
- When odds/ESPN names don't resolve, add the mapping to `canonical_teams_2026.csv`

### Pitcher ID Crosswalk (NCAA→ESPN)
- Run events use numeric ESPN IDs (`64890.0`); pitcher appearances scraper assigns `NCAA_` format IDs
- Same pitcher may appear at two registry indices — ESPN (1–1743, has posterior data) and NCAA (1744+, prior-only)
- `lookup_starters.py` has `_build_ncaa_espn_crosswalk()` that maps NCAA_ IDs → ESPN indices via (name, team) matching
- Crosswalk reads `espn_name` column from `canonical_teams_2026.csv` — if this column is missing, crosswalk produces 0 matches
- Only ~285/7591 NCAA pitchers have ESPN counterparts — most pitchers have no learned posterior (prior-dominated)
- Do NOT use fuzzy matching for team names — use explicit `espn_name` mappings only

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
- `wind_out_mph` = wind component blowing from HP toward CF
- Wind: ~0.008 log-rate/mph → 10 mph tailwind ≈ +8% runs/game (validated vs Nathan's +4%/5mph)
- Temperature: ~0.002 log-rate/°F above 72°F baseline
- Altitude: 0.25 × (1 − air_density_ratio) → Air Force (6686 ft) = +5.6% runs
- Negative wind_out = wind blowing IN (suppresses scoring)
- `elevation_ft` column in `stadium_orientations.csv` sourced from Open-Meteo elevation API

## Daily Pipeline Commands

### Full daily run (predictions + odds + compare)
```bash
# 1. Run simulation
.venv/bin/python3 scripts/predict_day.py --date YYYY-MM-DD --N 5000 --out data/processed/predictions_YYYY-MM-DD.csv

# 2. Pull fresh odds
source ~/.zshrc; ODDS_API_KEY="$THE_ODDS_API_KEY" .venv/bin/python3 scripts/pull_odds.py --mode current --regions us,us2,eu --markets h2h,totals,spreads

# 3. Compare (see odds comparison skill)
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

### Platoon Splits (LHP/RHP)
- `scripts/platoon_adjustment.py` tracks pitcher handedness via D1B rotation data
- Rate adjustment disabled (`DEFAULT_LHP_ADJ = 0.0`) — sign uncertain, double-counts with `pitcher_ability`
- Stage 2 plan: add `platoon_effect` parameter to Stan model to learn sign/magnitude from data
- Batter handedness available on D1Baseball player pages (`BAT/THRW` field, e.g. `R/R`)
- D1Baseball lineup pages (`/team/{slug}/lineup/`) show game-by-game batting orders (free, no login)

## Common Pitfalls
- Overpass API rate limits: Use 4-5s between queries, retry on 429/504 with exponential backoff
- `bookmaker_lines` not `bookmakers` in odds JSONL
- Stadium default 67° will produce wrong wind calculations — always verify bearing source
- NCAA API game names don't match odds API names — use canonical_id as join key
- The ODDS_API_KEY is stored in shell env as `$THE_ODDS_API_KEY` (source ~/.zshrc first)
- Pitcher ID formats: run_events uses numeric ESPN IDs, appearances uses `NCAA_` format — crosswalk in `lookup_starters.py` bridges them
- `canonical_teams_2026.csv` columns: `team_name` (short), `odds_api_name` (full mascot for odds), `espn_name` (full mascot for ESPN pitcher registry), `baseballr_team_name`
- Prediction CSV columns: `home`/`away` (not `home_team`), `exp_total` (not `mean_total`), `home_starter_idx`/`away_starter_idx` (not `hp_pitcher_idx`)
