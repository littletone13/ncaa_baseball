# NCAA Baseball Prediction Model

## Project Overview
NCAA Division 1 baseball prediction model that simulates games using starting pitcher quality, bullpen strength, park factors, and live weather (wind direction relative to stadium orientation). Outputs daily projections with moneylines, totals, and edges vs market odds.

## Architecture

### Data Flow (Refactored Pipeline)
```
── BUILD PHASE (offline, run via Makefile) ──────────────
games_2026.jsonl → extract_espn.py → games.csv, run_events.csv,
                                       pitcher_appearances.csv, venue_stats.csv

linescores + run_events → merge_run_events.py → unified run_events.csv
                          (backfills pitcher IDs from pitcher_appearances starters)

run_events.csv → build_run_event_indices.py → team_index.csv, pitcher_index.csv
              → fit_run_event_model.py → run_event_posterior.csv (Stan, 4 chains)

pitcher_appearances + D1B stats + Stan indices → build_pitcher_table.py → pitcher_table.csv
canonical_teams + Stan indices + bullpen/wRC+ → build_team_table.py → team_table.csv

── PREDICT PHASE (daily, orchestrated by predict_day.py) ──
predict_day.py calls 7 steps in sequence:
  0. pull_odds.py         → odds_latest.jsonl                 (auto-pull for market anchor)
  1. resolve_schedule.py  → data/daily/{date}/schedule.csv    (NCAA + ESPN + Odds APIs)
  2. resolve_starters.py  → data/daily/{date}/starters.csv    (StarterLookup + pitcher_table + overrides)
  2b. starter QA report   → starter_qa.csv                    (confidence assessment)
  3. resolve_weather.py   → data/daily/{date}/weather.csv     (Open-Meteo + park + humidity + rain + gusts)
  3b. bullpen_fatigue.py  → data/daily/{date}/fatigue.csv     (Rolling 3-day reliever IP + arm availability)
  3c. compute_game_context.py → data/daily/{date}/context.csv (7 layers: rest, day/night, surface, travel, form, catcher, conf)
  4. simulate.py          → predictions CSV                    (Monte Carlo, 5000 draws, bilateral platoon)
  5. calibration report   → calibration CSV + markdown         (market coherence checks)
  6. upload_projections_to_syndicate() → Supabase public.projections (direct, no adapter)

── STARTER CONFIRMATION (pre-game, ~90 min before first pitch) ──
scrape_statbroadcast.py → starter_overrides.csv → re-run predict_day.py
  StatBroadcast pages have confirmed lineups + SP before game time
  Overrides replace projected starters and re-resolve pitcher indices

pull_odds.py → odds JSONL → compare model vs market (ML, totals, spreads)
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
| `scripts/integrate_ncaa_boxscores.py` | Merge NCAA API boxscores into appearances (100% coverage) |
| `scripts/build_linescore_run_events.py` | Convert NCAA inning-by-inning linescores → run_events |
| `data/processed/pitcher_table.csv` | 9K+ pitchers with ability, handedness, FB sensitivity |
| `data/processed/team_table.csv` | 308 teams with team_idx, bullpen_z, wrc_offense_adj, batting_fb_factor |

### Critical Files — Predict Phase
| File | Purpose |
|------|---------|
| `scripts/predict_day.py` | Thin orchestrator calling 5 modules below |
| `scripts/resolve_schedule.py` | NCAA + ESPN + Odds APIs → game schedule |
| `scripts/resolve_starters.py` | StarterLookup + pitcher_table → projected starters |
| `scripts/resolve_weather.py` | Open-Meteo + park factors → per-game weather |
| `scripts/simulate.py` | Pure Monte Carlo engine (no API calls, deterministic) |

### Critical Files — Other
| File | Purpose |
|------|---------|
| `scripts/pull_odds.py` | Odds data fetcher (current + historical, ML + totals + spreads) |
| `scripts/weather_park_adjustment.py` | Wind/temp → run scoring adjustment (directional wind model) |
| `scripts/lookup_starters.py` | Starting pitcher inference (StarterLookup class) |
| `scripts/bullpen_fatigue.py` | Rolling 3-day reliever IP tracker (fatigue_z + fatigue_adj) |
| `scripts/backtest.py` | Systematic backtesting: Brier score, log-loss, total MAE, calibration |
| `scripts/scrape_statbroadcast.py` | Pre-game lineup scraper — confirmed starters from StatBroadcast (Playwright) |
| `scripts/scrape_statbroadcast_pbp.py` | Post-game PBP/run_events with pitcher attribution from StatBroadcast |
| `scripts/scrape_sidearm_boxscores.py` | Sidearm box score scraper — pitcher stats, lineups, umpires (308 teams) |
| `scripts/parse_d1b_rotations_html.py` | D1B weekly rotation article parser → d1baseball_rotations.csv |
| `scripts/compute_game_context.py` | 7-layer context engine (rest, day/night, surface, travel, form, catcher, conf) |
| `scripts/build_team_batter_composition.py` | Per-team batter handedness from sidearm rosters (bilateral platoon) |
| `scripts/backtest_vs_market.py` | Corrected backtest — raw market odds for payout, tests ML + totals + spreads |
| `scripts/integrate_ncaa_boxscores.py` | Merge NCAA boxscores into pitcher_appearances (100% coverage) |
| `scripts/platoon_adjustment.py` | LHP/RHP platoon lookup (bilateral model, DEFAULT_LHP_ADJ = 0.03) |
| `data/registries/canonical_teams_2026.csv` | Team registry with odds_api_name + espn_name mappings |
| `data/registries/stadium_orientations.csv` | Stadium lat/lon + HP→CF bearings |
| `data/registries/d1baseball_crosswalk.csv` | D1Baseball team name → canonical_id (308 entries) |
| `data/registries/sidearm_urls.csv` | 308 team athletic site URLs for Sidearm scraping |
| `data/registries/statbroadcast_names.csv` | StatBroadcast abbreviation → canonical_id aliases |
| `data/registries/twitter_ncaa_baseball.csv` | 194 Twitter/X accounts (147 teams, 26 beat reporters) |
| `data/registries/surface_types.csv` | 38 turf stadiums with brand attribution |
| `data/registries/catcher_quality.csv` | D1B Top 50 catchers with composite quality score |
| `data/processed/team_batter_handedness.csv` | Per-team RHB/LHB/switch composition (207 teams) |
| `data/processed/d1baseball_rotations.csv` | Weekly rotation starters with canonical_id (must have canonical_id column) |
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
- Simulation prices spreads from full margin distribution (5000 MC sims), can price any spread
- Live games have shifted odds — filter by `commence_time` vs now to exclude in-progress games

### Weather & Altitude Model
- **Directional wind**: 5-point arc-weighted sampling across 90° outfield arc
  - LF foul pole (0.10) → LF power alley (0.25) → CF (0.30) → RF power alley (0.25) → RF foul pole (0.10)
  - `wind_out_directional()` in `weather_park_adjustment.py` computes weighted effective wind
  - **Wind gusts**: blends 0.7 × sustained + 0.3 × gusts for effective wind speed
- Wind: ~0.008 log-rate/mph → 10 mph tailwind ≈ +8% runs/game
- Temperature: ~0.002 log-rate/°F above 72°F baseline, **cold floor at 55°F** (no penalty below)
- Altitude: 0.25 × (1 − air_density_ratio) → Air Force (6686 ft) = +5.6% runs
- **Humidity**: fetches relative_humidity_2m from Open-Meteo, density reduction via Magnus formula
- **Rain**: -0.15% per pct above 25% threshold, capped at -10%
- **Dome bypass**: `is_dome` flag zeros all weather adjustments except altitude
- **Altitude double-counting guard**: subtracts expected altitude component from park_factor
- Negative wind_out = wind blowing IN (suppresses scoring)
- `elevation_ft` + `is_dome` columns in `stadium_orientations.csv`

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
- Platoon: `platoon_adj_home`, `platoon_adj_away` (bilateral model, IP-weighted starter + bullpen)
- Confirmed starters: `hp_confirmed`, `ap_confirmed` (1 = StatBroadcast/D1B rotation confirmed)
- Context: `home_context_adj`, `away_context_adj` (combined 7-layer adjustment)
- D1B pitcher fallback: `hp_d1b_adj`, `ap_d1b_adj` (FIP/SIERA/ERA ability estimate), `hp_d1b_src`, `ap_d1b_src`
- D1B team offense: `home_wrc_adj`, `away_wrc_adj` (wRC+-based att adjustment for non-model teams)

### NCAA Boxscore Integration
- `scripts/integrate_ncaa_boxscores.py` merges NCAA API boxscores into pitcher_appearances
- NCAA API (`ncaa-api.henrygd.me`) has 100% D1 coverage (vs ESPN's ~15%)
- Result: 36,484 appearances (19,057 ESPN + 17,427 novel NCAA)
- Deduplication: prefers ESPN data when both sources have same (date, team, pitcher_name, starter)
- NCAA boxscores provide starter flags, IP, hits, runs, ERs, Ks, BBs for all teams
- Used to improve pitcher crosswalk: 59% of games have pitcher_idx > 0 (was 47%)

### Linescore → Run Events Expansion
- `scripts/build_linescore_run_events.py` converts inning-by-inning scores to run_events
- NCAA linescores (2,214 games) provide per-inning run counts for ALL games
- A 3-run inning = one `run_3` event, independent of play-by-play details
- Massively expands Stan training data beyond ESPN PBP-only coverage

### Pitcher ID Backfill (merge_run_events.py)
- `scripts/merge_run_events.py` merges ESPN + linescore run_events, then backfills missing pitcher IDs
- Reads starter rows from `pitcher_appearances.csv`, builds `(date, team)` → pitcher_id lookup
- Uses stable NCAA ID format: `NCAA_{normalized_name}__{canonical_id}` matching `build_pitcher_table.py`
- Fills ~3,800 pitcher slots that would otherwise be empty (linescore events + ESPN score-only games)
- **Critical**: without this step, pitcher index drops from ~3,900 to ~1,700 and most pitchers lose posteriors

### Bilateral Platoon Model
- `DEFAULT_LHP_ADJ = 0.03` (+3% runs vs LHP, applied when pitcher hand known)
- **IP-weighted**: starter platoon × starter_ip_frac + bullpen LHP frac × bullpen_ip_frac
- **Bilateral**: scales by batting team's actual RHB composition (not blanket 75% assumption)
  - `team_platoon = LHP_ADJ × (team_effective_rhb / LEAGUE_AVG_RHB)`
  - Ole Miss (41% RHB) gets -40% weaker platoon vs LHP; Yale (93% RHB) gets +33% stronger
- Per-team batter handedness from `sidearm_rosters.csv` → `team_batter_handedness.csv` (207 teams, 99% coverage)
- Bullpen LHP fraction: Bayesian shrinkage toward 0.30 prior with 5 pseudo-observations
- Parity check: PLATOON_LHP_ADJ validated against platoon_adjustment.py at import time

### 7-Layer Game Context Engine (`compute_game_context.py`)
- **Rest/schedule density**: -2.5% per game beyond 3-in-4-days, +1.2% for 2+ days rest
- **Day/night**: -1.5% afternoon games, +0.8% evening (reads start_local_hour from schedule)
- **Surface**: +1.8% on turf fields (38 stadiums registered in surface_types.csv)
- **Conference strength**: non-conf opponents from weaker conferences get offense penalty
- **Travel distance**: -0.8% per 500mi beyond 500 for away team (haversine from stadium lat/lon)
- **Recent form**: 7-day scoring rate vs season avg, regressed 50% toward mean, ±4% cap
- **Catcher quality**: D1B Top 50 catchers suppress opponent runs (-2.0% for top 10, -1.2% top 25)

### Starter Confirmation Pipeline
- **StatBroadcast** (`scrape_statbroadcast.py`): lineups available ~30-90 min pre-game, Playwright non-headless
- **D1B rotations** (`parse_d1b_rotations_html.py`): weekly press conference starters, must have `canonical_id` column
- **Overrides CSV** (`data/daily/{date}/starter_overrides.csv`): game_num,side,pitcher_name,source
- `resolve_starters.py` reads overrides, clears pitcher_id so pitcher_table re-resolves correct index
- `hp_confirmed`/`ap_confirmed` flags: 1 = from StatBroadcast or D1B rotation, 0 = appearance history guess
- **D1B rotation gotcha**: `d1baseball_rotations.csv` must have `canonical_id` column (not just `team_abbr`)
- **Thu-Sat series**: D1B labels Gm1/2/3 as fri/sat/sun but actual days shift — check which teams played Thursday

### Supabase Integration
- Direct upload via `upload_projections_to_syndicate()` in `load_baseball_to_postgres.py`
- Upserts on `(sport, game_id, model_version)` — model_version = `stan_v2_5000sim`
- Game ID format: `bsb_{YYYYMMDD}_{away_slug}_{home_slug}`
- Data blob includes starters (with hp_confirmed/ap_confirmed), weather, bullpen, rl_cover
- Project ref: `otfybzwvockuwdldfoed`, password in `SUPABASE_DB_PASSWORD` env var or `.env`

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

# 3. Refit Stan model (~15-20 min on M-series Mac)
#    Current: 308 teams, 5907 pitchers, 4709 run_events, 32K draws
.venv/bin/python3 scripts/fit_run_event_model.py

# 4. Rebuild unified lookup tables
.venv/bin/python3 scripts/build_pitcher_table.py
.venv/bin/python3 scripts/build_team_table.py

# 5. Subsample posterior for daily predictions
python3 -c "import pandas as pd; d=pd.read_csv('data/processed/run_event_posterior.csv'); d.sample(2000,random_state=42).to_csv('data/processed/run_event_posterior_2k.csv',index=False)"
```

## Model Calibration & Validation

### Scoring Calibration (SCORING_CALIBRATION)
- Located in `src/ncaa_baseball/model_runtime.py`, currently `0.034`
- Calibrated from **2,618 actual 2026 game outcomes**: actual avg total = 13.02
- Formula: `C = log(actual_avg / model_base)`
- Applied uniformly to all 4 intercepts in the posterior (additive log-rate shift)
- Recalibrate after every Stan refit by running `backtest.py --tune-calibration`

### Home Advantage
- Stan posterior learns HA = 0.115 (log-rate), implying ~53% home win rate for equal teams
- Market-implied home win rate: 52.7% — matches posterior almost exactly
- Raw NCAA home win rate: 62-64% — higher because good teams play more home games (schedule bias)
- **No post-hoc correction needed** — the model correctly separates team quality from HA
- `--ha-target` defaults to 0.0 (use learned posterior as-is)

### Bullpen Fatigue
- `bullpen_fatigue.py` computes rolling 3-day reliever IP per team
- `fatigue_z` = z-score of team's recent reliever usage vs all teams
- `fatigue_adj` = `fatigue_z * FATIGUE_COEFF` (0.015 per z-unit) — only when z > 0
- Applied in simulation: fatigued home bullpen → away team scores more, and vice versa
- Stacks with static `bullpen_quality_z` from team_table

### Batting FB% Wind Interaction
- Teams with high fly-ball rates benefit more from tailwind (and are hurt more by headwind)
- `batting_fb_factor` = team_FB% / league_avg_FB% (range 0.63–1.29)
- Applied: `wind_adj = wind_adj_raw × pitcher_fb_sens × batting_fb_factor`
- Data source: D1Baseball batted ball leaderboard TSVs

### Backtest Framework
- `scripts/backtest_vs_market.py` — corrected backtest using RAW market odds for payout
- Uses best US book price (DK, FD, MGM, Caesars) for ML payout, not devigged fair odds
- Tests ALL markets: ML home, ML away, overs, unders, spreads
- Validates American odds (rejects -99 to +99 as corrupt), defaults to -115 for totals
- Snapshot merge logic: updates fields per snapshot, doesn't overwrite with missing data
- `scripts/backtest.py` — model-vs-actuals calibration (Brier, log-loss, MAE, bias)
- **Backtest results (111 games, corrected)**: Overs 1.0+ edge: 57.3% W, +5.3% ROI at -115 juice
- **Known totals inflation**: model avg ~13 vs market avg ~10.5 on bettable games — pitcher ability compression (70% idx=0)

## Common Pitfalls
- **Do NOT use fuzzy matching** for team names anywhere — use exact match against canonical registry
- **StatBroadcast proof-of-work**: pages require 8+ seconds to load; headless Playwright gets 403'd; use non-headless with `--disable-blink-features=AutomationControlled`
- **D1B rotation `canonical_id` column**: `d1baseball_rotations.csv` MUST have `canonical_id` column, not just `team_abbr` — StarterLookup matches on canonical_id
- **Thu-Sat vs Fri-Sun series**: D1B labels columns Gm1/2/3 as fri/sat/sun regardless of actual start day. Check which teams played Thursday before mapping game numbers.
- **Starter overrides clear pitcher_id**: when override fires, hp_id is set to "" so pitcher_table re-resolves the correct index. Don't just change the name without clearing the ID.
- **Odds pull log American format**: prices are American (-200, +150), NOT decimal. The `price_to_implied()` function auto-detects but totals prices between -99 and +99 are corrupt (usually spread points leaking into price field).
- **Scoring calibration**: `SCORING_CALIBRATION` in `model_runtime.py` must match `SIMULATE_SCORING_CALIBRATION` in `simulate.py` — parity check asserts at import time
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
