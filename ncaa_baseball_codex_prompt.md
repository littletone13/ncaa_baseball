# NCAA D1 College Baseball Simulation Model — Codex Agent Prompt

## Project Overview

Build a production Bayesian sports betting model for NCAA Division 1 college baseball. The model follows Andrew Mack's methodology from "Statistical Sports Models" (Chapter 18: MLB Log-Linear Negative Binomial Model) adapted for the unique challenges of college baseball (~300 D1 teams, ~56 games/team, massive roster turnover via transfer portal, freshmen with no history, aluminum bats, variable park effects).

The project has two parallel data pipelines that converge:
1. **Performance data** — play-by-play, player stats, rosters from `baseballr` R package (free, unlimited)
2. **Market data** — historical odds line movement at 5-minute intervals from The Odds API (quota-limited, paid)

The market data is not just for backtesting — full line movement history is a powerful signal. Tracking how lines move from open to close reveals sharp money, information asymmetry, and market consensus evolution. This becomes a Bayesian prior and calibration benchmark.

---

## Project Structure

```
ncaa-baseball-model/
├── config/
│   └── config.yaml                  # API keys, seasons, parameters, file paths
├── data/
│   ├── raw/
│   │   ├── pbp/                     # Raw play-by-play by season (parquet)
│   │   ├── rosters/                 # Rosters by team-season
│   │   ├── player_stats/            # Batting + pitching stats by team-season
│   │   ├── schedules/               # Schedules with game URLs
│   │   ├── game_logs/               # Player game logs
│   │   └── odds/                    # Raw Odds API JSON responses
│   │       ├── historical/          # 5-min snapshot archives by date
│   │       ├── current/             # Live odds pulls
│   │       └── scores/              # Game results from Odds API
│   ├── processed/
│   │   ├── games.parquet            # Game-level: teams, date, run events, pitchers, result
│   │   ├── run_events.parquet       # Per-game run event counts (1-run, 2-run, 3-run, 4-run)
│   │   ├── rosters_unified.parquet  # Cross-season player tracking with transfer detection
│   │   ├── returning_production.parquet  # % of PA/IP returning per team per season
│   │   ├── park_factors.parquet     # Park effect estimates
│   │   ├── name_crosswalk.parquet   # Team name mapping: baseballr <-> Odds API <-> common
│   │   └── odds/
│   │       ├── closing_lines.parquet    # Closing odds per game (h2h, spreads, totals)
│   │       ├── opening_lines.parquet    # Opening odds per game
│   │       └── line_movement.parquet    # Full 5-min line movement history per game
│   └── features/
│       ├── team_strength_priors.parquet # Pre-season team ratings from returning production
│       └── pitcher_ratings.parquet      # Individual pitcher ability estimates
├── scripts/
│   ├── 01_scrape_pbp.R             # Bulk PBP pull 2021-2025 via baseballr
│   ├── 02_scrape_rosters.R         # Pull rosters all D1 teams, all seasons
│   ├── 03_scrape_player_stats.R    # Batting + pitching stats per team-season
│   ├── 04_scrape_odds_historical.R # Historical odds at 5-min intervals (The Odds API)
│   ├── 05_scrape_odds_live.R       # Current season live odds polling
│   ├── 06_wrangle_games.R          # Parse PBP → run events per game
│   ├── 07_build_rosters.R          # Unified player tracking, transfer detection, returning %
│   ├── 08_build_name_crosswalk.R   # Map team names between data sources
│   ├── 09_build_line_movement.R    # Process odds snapshots → opening/closing/movement
│   ├── 10_fit_model.R              # Stan model fitting with walk-forward validation
│   ├── 11_simulate_games.R         # Monte Carlo game simulation from posterior
│   ├── 12_backtest.R               # Model vs closing lines, calibration analysis
│   ├── 13_find_value.R             # Compare model prices to current odds, flag bets
│   └── 14_daily_pipeline.R         # Daily automation: scrape new games, update model, find value
├── stan/
│   └── ncaa_baseball.stan           # Bayesian log-linear competing distribution model
├── R/
│   ├── utils.R                      # Shared utility functions
│   ├── odds_api.R                   # Odds API wrapper functions
│   ├── devig.R                      # Vig removal (Power method default, Shin for asymmetry)
│   └── simulate.R                   # Simulation engine functions
├── notebooks/
│   ├── eda_run_events.Rmd           # Exploratory: distribution fitting for run events
│   ├── eda_park_effects.Rmd         # Park factor analysis
│   └── eda_line_movement.Rmd        # Line movement patterns and sharp money signals
└── README.md
```

---

## Phase 1: Data Scraping — Performance Data (baseballr)

### 1A: Bulk Play-by-Play (01_scrape_pbp.R)

```r
library(baseballr)
library(arrow)  # parquet I/O

# Bulk pre-scraped PBP — fastest path for historical seasons
for (season in 2021:2024) {
  pbp <- load_ncaa_baseball_pbp(seasons = season)
  write_parquet(pbp, paste0("data/raw/pbp/pbp_", season, ".parquet"))
  cat("Saved", season, ":", nrow(pbp), "plays\n")
}

# 2025 season — must scrape game-by-game since bulk repo won't have it yet
# First get all D1 teams, then schedules, then PBP per game
teams_2025 <- ncaa_teams(division = 1, season = 2025)
# Loop through teams → ncaa_schedule_info() → ncaa_pbp() per game_info_url
# Rate limit: be respectful, add Sys.sleep(2) between calls
```

**Key fields from PBP:**
- `game_date`, `location`, `inning`, `inning_top_bot`, `score`, `batting`, `description`
- The `description` field contains the play narrative — parse this for run outcomes

### 1B: Rosters (02_scrape_rosters.R)

Pull rosters for ALL D1 teams across ALL seasons (2021-2025). This is the foundation for transfer detection.

```r
# For each team_id in ncaa_teams(), for each season:
roster <- ncaa_roster(team_id = team_id, year = season)
```

**Critical — NO FUZZY MATCHING anywhere in this project. Use NCAA player_id as the authoritative key.**

`baseballr` returns a numeric `player_id` from stats.ncaa.org on every roster and stats call. This is a unique, persistent identifier — not a name string. Use it for everything.

```r
# ncaa_roster() returns player_id for every player on a team
roster <- ncaa_roster(team_id = 736, year = 2024)
# Columns include: player_id, player_name, yr, pos, jersey, ...

# ncaa_game_logs() accepts player_id — no name matching needed
career <- ncaa_game_logs(player_id = 2477974, year = 2023, type = "pitching", span = "career")
# Returns: Year | player_id | player_name | Team | GP | ERA | IP | ...
# The TEAM column changes across years — this IS your transfer detection.
# Example: Paul Skenes shows Air Force in 2020-21, then LSU in 2022-23.
```

**The player_id is persistent within a school.** When a player transfers, stats.ncaa.org MAY assign a new player_id at the new school. The `span = "career"` game logs endpoint still links them — it returns all seasons with the Team column showing each school. This is definitive transfer detection, not guesswork.

**DO NOT use fuzzy name matching for player identification.** It WILL break on:
- Common names (dozens of "Jake Smith" across D1)
- Formatting inconsistencies ("J.J." vs "JJ" vs "J. J.")
- Hyphenated names, suffixes (Jr., III), international names
- Players with identical names on different teams

Build the player registry deterministically:

```r
# STEP 1: Pull rosters for ALL D1 teams, ALL seasons → master registry
# Each row: player_id | player_name | team_id | team_name | year | pos | jersey | class_year
# player_id is the primary key within a team-season

# STEP 2: For transfer detection, pull career logs for EVERY player_id
# ncaa_game_logs(player_id, year, type, span = "career")
# If a player's career shows multiple Teams → they transferred
# The old school + old stats are RIGHT THERE in the response, no matching needed

# STEP 3: For PBP → player linking
# PBP descriptions contain player names in text (e.g., "Smith, J. singled to left")
# Join to roster via EXACT match on normalized name within that team-season:
#   - Normalize: lowercase, strip whitespace, standardize "Last, First" format
#   - Join on: normalized_name + team_name + season
#   - This is exact matching within a known, bounded set (25-35 players per roster)
#   - Flag any non-matches for manual review rather than guessing

# STEP 4: Validate — every player in processed data must have a player_id
# If a PBP name can't be matched to a roster player_id, flag it and skip
# Do NOT guess. A 2% miss rate with clean data beats 100% coverage with 5% wrong matches.
```

**For freshmen with no NCAA history:** They simply have no prior-season rows in the career game logs. Their player_id exists only in the current season roster. The model handles them through team-level regression toward the mean, not individual priors.

**Rate limiting note:** Pulling career game logs for every player across all D1 teams is thousands of API calls to stats.ncaa.org. Build in `Sys.sleep(2)` between calls and run as an overnight batch job. Cache aggressively — a player's 2021-2023 career stats won't change.

### 1C: Player Stats (03_scrape_player_stats.R)

```r
# Batting stats
batting <- ncaa_team_player_stats(team_id, year, type = "batting")
# Pitching stats
pitching <- ncaa_team_player_stats(team_id, year, type = "pitching")
# Game logs for granular analysis
game_logs <- ncaa_game_logs(player_id, year, type = "batting")  # or "pitching"
```

### 1D: Park Factors

```r
park <- ncaa_park_factor(team_id, year, type = "batting")
```

Park effects are MORE important in college than MLB due to aluminum bats + wildly variable stadium dimensions + altitude differences. Store and use as fixed offsets in the model.

---

## Phase 2: Data Scraping — Market Data (The Odds API)

### CRITICAL: Line Movement Strategy

**This is a massive competitive advantage.** The Odds API stores historical snapshots at 5-minute intervals (from Sept 2022+). For NCAA baseball, historical data is available from **May 2023**. We want to capture the FULL line movement arc for every game — not just closing lines.

Line movement reveals:
- **Opening line** = bookmaker's initial assessment (often based on limited info for college)
- **Sharp money moves** = sudden line shifts, especially in first few hours after opening
- **Steam moves** = coordinated sharp action across multiple books simultaneously
- **Reverse line movement** = line moves opposite to public betting percentages
- **Closing line** = market's best estimate, incorporating all information

The opening-to-close trajectory is a powerful feature. Games where your model disagrees with the close AND the line moved toward your position = strongest signals.

### 2A: Historical Odds Scraping (04_scrape_odds_historical.R)

**Sport key:** `baseball_ncaa`
**Available markets:** `h2h`, `spreads`, `totals` (NO player props for college baseball)
**Historical data available from:** May 2023
**Snapshot interval:** Every 5 minutes
**Quota cost:** 10 credits per region per market per call

```r
# Wrapper function for The Odds API historical endpoint
# GET /v4/historical/sports/baseball_ncaa/odds
# Parameters: regions=us, markets=h2h,spreads,totals, date=ISO8601, oddsFormat=american

# Strategy for comprehensive line movement capture:
# 1. Use historical EVENTS endpoint first (1 credit) to get event IDs and commence times
# 2. For each game, calculate snapshot timestamps we want:
#    - From ~48h before first pitch to ~1h before first pitch
#    - At 5-minute intervals (or strategically sample: every 30 min far out, every 5 min close to game)
# 3. Use historical ODDS endpoint for bulk snapshots (10 credits × 3 markets × 1 region = 30 per call)
#    The bulk endpoint returns ALL games at a timestamp, which is more efficient than per-event
# 4. Use previous_timestamp / next_timestamp in responses to paginate through time

# QUOTA MATH for full season scrape:
# ~2,800 D1 baseball games per season (2023 + 2024 = ~5,600 games)
# Bulk historical odds returns all games at one timestamp = 30 credits per snapshot
# If we sample every 30 min for 48h before each game day = ~96 snapshots per game day
# ~150 game days per season × 96 snapshots × 30 credits = ~432,000 credits per season
# This is expensive! Smart sampling strategy:
#   - Full 5-min resolution only for final 2 hours before first pitch
#   - 30-min resolution for 2-24h before
#   - 2-hour resolution for 24-48h before
# This reduces to roughly: (24 + 44 + 12) × 30 = ~2,400 credits per game day
# ~150 days × 2,400 = 360,000 credits per season
# Check your plan limits and adjust sampling accordingly.
# ALTERNATIVE: Use bulk historical odds endpoint which returns ALL games at a timestamp
# This amortizes cost when many games share the same day.

# Efficient approach: batch by DATE not by game
# For each date in the season:
#   Pull historical odds at strategic intervals covering that date's games
#   Each pull returns ALL games active at that timestamp
#   Store raw JSON, process later

get_historical_snapshot <- function(sport, date_iso, api_key, 
                                    regions = "us", 
                                    markets = "h2h,spreads,totals",
                                    odds_format = "american") {
  base_url <- "https://api.the-odds-api.com/v4/historical/sports"
  url <- paste0(base_url, "/", sport, "/odds")
  
  resp <- httr::GET(url, query = list(
    apiKey = api_key,
    regions = regions,
    markets = markets,
    oddsFormat = odds_format,
    date = date_iso
  ))
  
  # Track quota from response headers
  remaining <- httr::headers(resp)$`x-requests-remaining`
  used <- httr::headers(resp)$`x-requests-used`
  last_cost <- httr::headers(resp)$`x-requests-last`
  cat("Quota: used=", used, "remaining=", remaining, "last_cost=", last_cost, "\n")
  
  content <- httr::content(resp, as = "parsed")
  
  return(list(
    data = content$data,
    timestamp = content$timestamp,
    previous_timestamp = content$previous_timestamp,
    next_timestamp = content$next_timestamp,
    quota_remaining = remaining
  ))
}

# Smart sampling function for a single game day
scrape_game_day <- function(game_date, api_key) {
  # game_date is a Date object
  # Assume games start around 16:00-23:00 UTC
  # First game typically ~16:00 UTC (noon ET)
  
  first_pitch_approx <- paste0(game_date, "T16:00:00Z")
  
  # Generate snapshot timestamps:
  # Final 2h before first pitch: every 5 min (24 snapshots)
  # 2-12h before: every 30 min (20 snapshots)  
  # 12-48h before: every 2h (18 snapshots)
  # Total: ~62 snapshots per game day × 30 credits = ~1,860 credits
  
  timestamps <- c(
    # 48h to 12h before: every 2 hours
    seq(as.POSIXct(first_pitch_approx, tz="UTC") - hours(48),
        as.POSIXct(first_pitch_approx, tz="UTC") - hours(12),
        by = "2 hours"),
    # 12h to 2h before: every 30 min
    seq(as.POSIXct(first_pitch_approx, tz="UTC") - hours(12),
        as.POSIXct(first_pitch_approx, tz="UTC") - hours(2),
        by = "30 min"),
    # Final 2h: every 5 min
    seq(as.POSIXct(first_pitch_approx, tz="UTC") - hours(2),
        as.POSIXct(first_pitch_approx, tz="UTC") - minutes(5),
        by = "5 min")
  )
  
  results <- list()
  for (ts in timestamps) {
    ts_iso <- format(as.POSIXct(ts, origin="1970-01-01", tz="UTC"), "%Y-%m-%dT%H:%M:%SZ")
    result <- get_historical_snapshot("baseball_ncaa", ts_iso, api_key)
    results[[ts_iso]] <- result
    
    # Respect rate limits
    Sys.sleep(1)
    
    # Check quota
    if (as.numeric(result$quota_remaining) < 100) {
      warning("Low quota! Stopping.")
      break
    }
  }
  
  return(results)
}
```

### 2B: Live Odds Polling (05_scrape_odds_live.R)

For the CURRENT season, poll live odds on a schedule:

```r
# GET /v4/sports/baseball_ncaa/odds?regions=us&markets=h2h,spreads,totals&oddsFormat=american
# Cost: 3 credits per call (1 per market × 1 region)
# Poll every 5-10 min on game days, store with timestamp
# Also pull scores after games complete:
# GET /v4/sports/baseball_ncaa/scores?daysFrom=3
# Cost: 2 credits

# GET /v4/sports/baseball_ncaa/events (FREE - no quota cost)
# Use this to get event list and commence_times without burning quota
```

### 2C: Processing Line Movement (09_build_line_movement.R)

Transform raw snapshots into structured line movement data:

```r
# For each game_id (Odds API event id):
# 1. Extract all snapshots where this game appears
# 2. For each bookmaker × market, build time series:
#    timestamp | bookmaker | market | outcome | price | point
# 3. Identify opening line (first appearance)
# 4. Identify closing line (last snapshot before commence_time)
# 5. Calculate line movement features:
#    - opening_to_close_h2h_shift (implied prob change)
#    - max_line_move (largest single-interval shift)
#    - reverse_line_movement_flag
#    - consensus_direction (did most books move the same way?)
#    - sharp_vs_public_divergence
#    - time_of_biggest_move (early = sharp, late = public)
# 6. Devig all lines using Power method (default) or Shin method
#    - Store no-vig probabilities alongside raw prices
```

---

## Phase 3: Data Wrangling

### 3A: Parse PBP → Run Events (06_wrangle_games.R)

This directly mirrors Mack's Chapter 18 sections 1-4. From play-by-play, extract per-game counts of scoring events:

```r
# For each game:
# - Count how many times the batting team scored exactly 1 run on a play
# - Count how many times the batting team scored exactly 2 runs on a play
# - Count how many times the batting team scored exactly 3 runs on a play
# - Count how many times the batting team scored exactly 4+ runs on a play
# Output columns per game:
#   game_id, date, home_team, away_team, 
#   home_starting_pitcher, away_starting_pitcher,
#   home_run_1, home_run_2, home_run_3, home_run_4,
#   away_run_1, away_run_2, away_run_3, away_run_4,
#   home_true_score, away_true_score

# baseballr PBP has a 'description' text field
# Parse the description to identify:
# - Scoring plays (look for "scored", "homered", "home run", RBI indicators)
# - Number of runs scored on each play
# - Who was batting (maps to batting_team)
# This parsing is the most labor-intensive part — college PBP descriptions
# are less standardized than MLB. Build robust regex patterns and validate
# against known final scores.
```

### 3B: Roster Continuity & Transfer Detection (07_build_rosters.R)

**All transfer detection uses NCAA player_id and career game logs — zero fuzzy matching.**

```r
# STEP 1: Build master player registry from all rosters
# For each team_id × season → ncaa_roster() → store player_id, name, pos, jersey, class_year
# Primary key: (player_id, team_id, season)

# STEP 2: Detect transfers via career game logs
# For each player_id in the current season roster:
#   career <- ncaa_game_logs(player_id, year, type = "batting", span = "career")  
#   (or type = "pitching" for pitchers)
#   If career shows rows with different Team values → TRANSFER detected
#   Extract: old_school, old_stats (PA, BA, OPS, IP, ERA, etc.) directly from career data
#   No name matching needed — the NCAA system already linked the player across schools

# STEP 3: Classify each player on the current roster
# For each player on team X's 2025 roster:
#   a) Check if player_id appears in team X's 2024 roster → RETURNING
#   b) If not, check career game logs for prior Team entries:
#      - Prior team found → TRANSFER_IN (with exact prior stats from career endpoint)
#      - No prior seasons found → FRESHMAN (new to D1)
#   c) For each player on team X's 2024 roster NOT on 2025 roster:
#      - Check if player_id appears on any other team's 2025 roster → TRANSFER_OUT
#      - Not found anywhere → DEPARTED (graduated, drafted, quit)

# STEP 4: Calculate returning production metrics (no name matching involved)
# returning_pa_pct = sum(PA of RETURNING batters from 2024) / team total PA in 2024
# returning_ip_pct = sum(IP of RETURNING pitchers from 2024) / team total IP in 2024
# transfer_in_quality = for each TRANSFER_IN, their career stats from the career endpoint
# freshman_count = count of FRESHMAN players on roster

# STEP 5: Conference strength adjustment for transfers
# For transfer-in players, their career game logs give you their stats at their OLD school
# Adjust using conference-level averages:
#   adj_factor = conference_avg_metric[new_conf] / conference_avg_metric[old_conf]
# Example: a .380 hitter in the Patriot League transferring to the SEC
#   gets adjusted down based on the offensive environment difference
# Use league-wide wRC+ or OPS+ style adjustments when possible

# STEP 6: Handle edge cases
# - JUCO transfers: career logs will show JUCO team name, stats may be limited
#   Treat like freshmen for modeling purposes (heavy regression to mean)
# - Players who sat out a year (redshirt): career shows gap year, still linkable by player_id
# - Mid-season transfers: rare in baseball but check game_date ranges in game logs
```

### 3C: Team Name Crosswalk (08_build_name_crosswalk.R)

The Odds API uses names like "Arizona Wildcats", baseballr uses "Arizona", NCAA uses yet another format. **Build a deterministic static lookup — NOT fuzzy matching.**

```r
# STEP 1: Pull the definitive team list from each source
# Odds API: GET /v4/sports/baseball_ncaa/participants (1 credit, returns ~300 teams)
# baseballr: ncaa_school_id_lu() returns team_id, team_name, conference for all teams
# baseballr: ncaa_teams(division = 1, season = 2025) for current season

# STEP 2: Normalize both lists with a deterministic key
# Create a normalized_key from each name:
#   - lowercase
#   - strip "university", "college", "state", common suffixes
#   - extract the core school identifier
# Example: "Arizona Wildcats" → "arizona"
#          "Arizona" → "arizona"
# This gives you exact matches for ~80% of teams

# STEP 3: For the remaining ~20% that don't auto-match:
# DO NOT fuzzy match. Instead, generate a CSV of unmatched pairs
# and manually verify them ONCE. This is ~60 teams max, takes 30 minutes.
# Store as a static lookup table:
#   odds_api_name | baseballr_team_name | baseballr_team_id | ncaa_school_id | conference

# STEP 4: Version control this crosswalk file
# Conference realignment, new programs, and name changes happen yearly
# Review and update the crosswalk at the start of each season
# It's a small, finite set (~300 D1 baseball programs) — manual maintenance is fine

# STEP 5: Use the crosswalk as an INNER JOIN for all downstream analysis
# If a game appears in Odds API but can't be matched to baseballr → flag it
# If a game appears in baseballr but not Odds API → that's expected (many games are unlined)
# Never silently drop or guess-match unresolvable teams
```

**Why not fuzzy matching for teams:** College baseball has confusing cases that break fuzzy matchers. "Miami" could be Miami (FL) or Miami (OH). "Saint" vs "St." variations. "USC" vs "Southern California". "UConn" vs "Connecticut". A static lookup table with manual verification handles all of these perfectly and never silently mismatches. It's 300 rows — just do it right once.

---

## Phase 4: Stan Model

### Model Specification (stan/ncaa_baseball.stan)

Directly adapted from Mack's Chapter 18 MLB model. The core structure:

**Run event decomposition:**
- 1-run events ~ Negative Binomial (overdispersed: singles, walks, sacrifice flies, errors)
- 2-run events ~ Negative Binomial (overdispersed: doubles, 2-run HRs, multiple singles)
- 3-run events ~ Poisson (rare enough: 3-run HRs, bases-loaded events)
- 4-run events ~ Poisson (very rare: grand slams, big innings)

**Log-linear structure for each run event type:**
```
log(λ_home) = att[home_team] + def[away_team] + pitcher[away_pitcher] + home_advantage + intercept
log(λ_away) = att[away_team] + def[home_team] + pitcher[home_pitcher] + intercept
```

**Key adaptations for college baseball vs MLB:**
- **Tighter priors on team effects:** `normal(0, 0.15)` instead of `normal(0, 0.2)` — fewer games means more shrinkage needed
- **Tighter priors on pitcher effects:** `normal(0, 0.15)` — college starters have far fewer starts
- **Potentially hierarchical pitcher priors:** pitcher_ability ~ normal(team_pitching_mu, sigma_pitcher) to share information within a staff
- **Park effect offset:** Add `park[venue]` as fixed offset from pre-computed park factors, NOT estimated in the model (not enough data per park)
- **Conference hierarchy (optional, Phase 2):** Team abilities drawn from conference-level distributions for additional shrinkage

**Distribution selection:** Run LOO-PSIS comparison (Poisson vs NegBin) for each run type using brms BEFORE committing to the Stan model, exactly as Mack does in section 4 of Chapter 18. College baseball with aluminum bats may have different overdispersion patterns than MLB.

**Priors for college baseball:**
```stan
// Adjust these based on EDA — college scoring is HIGHER than MLB (aluminum bats)
// MLB intercepts from Mack: int_run_1 ~ normal(1, 0.2), int_run_2 ~ normal(1, 0.2)
// College will likely need higher intercepts
int_run_1 ~ normal(1.2, 0.3);    // More 1-run events in college
int_run_2 ~ normal(1.1, 0.3);    // More 2-run events in college  
int_run_3 ~ normal(0.6, 0.2);    // More 3-run events in college
int_run_4 ~ normal(0.3, 0.2);    // Grand slams still rare

// Team effects — tighter than MLB due to smaller samples
att_run_X_raw ~ normal(0, 0.15);
def_run_X_raw ~ normal(0, 0.15);
pitcher_ability_raw ~ normal(0, 0.15);

// Home advantage may be larger in college (travel, crowd, familiarity)
home_advantage ~ normal(0, 0.15);
```

### Walk-Forward Validation (10_fit_model.R)

**CRITICAL: No data leakage.** The model must only use data available before each game date.

```r
# Walk-forward approach:
# 1. Define training windows (e.g., rolling 2 seasons, or current season + prior)
# 2. For each game date in the validation period:
#    a. Filter training data to games BEFORE this date
#    b. Fit Stan model on training data
#    c. Simulate upcoming games
#    d. Record model probabilities vs actual outcomes vs closing lines
# 3. To reduce computation, refit model weekly (not daily) and use latest posterior
#    for daily predictions — Mack's approach from the hockey model discussion

# For early-season predictions (opening weekend!):
# Use 2024 season data as training set
# Apply returning_production adjustment to team priors:
#   team_prior_2025 = returning_pct × team_rating_2024 + (1 - returning_pct) × conference_mean
# This gives you a principled starting point that updates as 2025 data arrives
```

---

## Phase 5: Simulation Engine

### Monte Carlo Game Simulation (11_simulate_games.R)

Directly follows Mack's Chapter 18 sections 7-8:

```r
# For each matchup:
# 1. Draw N posterior samples (default N = 10,000)
# 2. For each sample, simulate run event counts:
#    - home_run_1 ~ NegBin(mu = exp(att + def + pitcher + HA + int), theta)
#    - home_run_2 ~ NegBin(...)
#    - home_run_3 ~ Pois(lambda = exp(...))
#    - home_run_4 ~ Pois(...)
#    - Same for away team (without home advantage)
# 3. Total runs = 1×run_1 + 2×run_2 + 3×run_3 + 4×run_4
# 4. Handle ties with extra innings (Mack's scaling_factor = 1/9 approach)
# 5. Output distributions:
#    - Win probability (home/away)
#    - Moneyline fair prices
#    - Run line probabilities (±1.5)
#    - Over/under probabilities at various totals
#    - Team total distributions
#    - Exact score probabilities (for exotic markets)
#    - First inning run probabilities (RIFI - from run event simulation)
```

---

## Phase 6: Market Comparison & Value Detection

### Backtesting (12_backtest.R)

```r
# For each historical game with both model predictions AND closing lines:
# 1. Convert closing lines to no-vig probabilities (Power method)
# 2. Compare model prob vs market prob
# 3. Calculate theoretical profit at various edge thresholds:
#    - Would we have bet at 2% edge? 3%? 5%? 
# 4. Calibration plots: model predicted prob vs observed frequency
# 5. Brier score, log loss, ROI by confidence bucket
# 6. Line movement features as additional signals:
#    - Does betting WITH sharp money (line moved toward your side) improve ROI?
#    - Does reverse line movement predict outcomes differently?
#    - What's the optimal timing to bet? (opening vs closing vs midpoint)
```

### Live Value Finder (13_find_value.R)

```r
# For each upcoming game:
# 1. Pull current odds from Odds API
# 2. Run simulation for this matchup
# 3. Compare model prices to each bookmaker
# 4. Flag value bets:
#    - model_prob > devigged_market_prob + edge_threshold
#    - Separate flags for h2h, spread, totals
# 5. Cross-reference with line movement:
#    - Has the line moved TOWARD or AWAY from your model's side?
#    - If toward = confirming (sharp money agrees), increase confidence
#    - If away = diverging (sharp money disagrees), flag for review
# 6. Output: bet recommendations with Kelly criterion sizing
```

---

## Phase 7: Line Movement Features (UNIQUE EDGE)

This is where the 5-minute historical snapshots become powerful model inputs, not just validation tools.

### Line Movement as Bayesian Prior

Following Mack's philosophy of incorporating market information:

```r
# After computing model probabilities, update with market prior:
# posterior_prob = w_model × model_prob + w_market × market_prob
# 
# But WHICH market prob? The line movement trajectory tells you:
# - Opening line = bookmaker's model (noisy, often based on limited college info)
# - Closing line = market consensus (sharp money has corrected errors)
# - Line movement SPEED = how quickly sharp money arrived
# - Line movement DIRECTION across books = consensus vs. one-book outlier
#
# Features to extract per game:
# 1. opening_implied_prob (h2h, spread, total) — devigged
# 2. closing_implied_prob (h2h, spread, total) — devigged
# 3. line_move_magnitude = closing - opening (in prob space)
# 4. line_move_direction = sign(closing - opening) relative to home team
# 5. sharp_move_detected = boolean, any single 5-min interval with >2% shift
# 6. sharp_move_time = minutes before first pitch when biggest single shift occurred
# 7. books_consensus = % of bookmakers that moved the same direction
# 8. spread_change = closing_spread - opening_spread (in points)
# 9. total_change = closing_total - opening_total (in runs)
# 10. max_disagreement = max spread across books at any point (market uncertainty)
# 11. pinnacle_vs_soft_divergence = if Pinnacle available, track sharp vs soft book gap
```

### Reverse Line Movement (RLM) Signals

```r
# RLM occurs when the line moves OPPOSITE to where public money is going
# In college baseball, this is especially powerful because:
# - Public tends to bet name brands (SEC, ACC blue bloods)
# - Sharp money knows which mid-majors are actually good
# - RLM on a mid-major = sharps loading up against the public
# 
# We can infer public side from:
# - Which team has the bigger name/higher ranking
# - Direction of line movement at soft books (BetMGM, DraftKings tend more public)
# - Compare to sharp book movement (if available)
```

---

## Reference: Mack's Chapter 18 Architecture (MLB Model)

The following is the exact flow from the reference code (018_chapter_18.R, 656 lines):

1. **Load raw PBP data** — CSV with game_id, date, inning, batting_team, pitching_team, starting_pitcher, play_event, run_outcome, outs
2. **Wrangle into run events** — count_runs() function: per game, count plays where run_outcome = 1, 2, 3, or 4 for each team
3. **Merge into final_data** — game_id, home/away teams, home/away starting pitchers, home/away run_1/2/3/4 counts, true scores
4. **Distribution selection** — brms + LOO-PSIS to compare Poisson vs NegBin for each run type
5. **Stan model** — Log-linear with sum-to-zero team constraints, NegBin for run_1/run_2, Poisson for run_3/run_4
6. **Posterior extraction** — team attack/defense ratings, pitcher abilities, intercepts, dispersion params
7. **Monte Carlo simulation** — 10K draws from posterior, simulate run events, sum to total runs, handle extra innings
8. **Metrics** — win prob, team totals, run lines, game total
9. **Visualization** — histogram overlays of simulated scores

Our college model follows this EXACT architecture with additions for:
- Roster continuity / transfer adjustments (prior to model fitting)
- Park effect offsets (fixed, not estimated)
- Tighter priors (college sample sizes)
- Market data integration (line movement features + Bayesian prior weighting)
- Walk-forward temporal validation (no data leakage)
- Automated daily pipeline

---

## Implementation Priority

**START HERE — Phase 1 tasks for opening weekend:**

1. `01_scrape_pbp.R` — Pull 2021-2024 bulk PBP immediately
2. `06_wrangle_games.R` — Parse into run events, validate against known scores
3. `08_build_name_crosswalk.R` — Build team name mapping (Odds API ↔ baseballr)
4. `02_scrape_rosters.R` — Pull all rosters for transfer detection
5. `07_build_rosters.R` — Calculate returning production percentages for 2025
6. `stan/ncaa_baseball.stan` — Adapt Mack's model with college priors
7. `10_fit_model.R` — Fit on 2024 data, generate 2025 opening-weekend predictions
8. `04_scrape_odds_historical.R` — Begin historical odds scraping (this runs in parallel, quota-gated)
9. `05_scrape_odds_live.R` — Set up live polling for 2025 season

**The historical odds scraping is a long-running background job** that will take days/weeks depending on your API quota tier. Start it immediately and let it run while you build everything else. Prioritize scraping the most recent season (2024) first since that's your primary backtest data, then work backward to 2023.

---

## Environment Setup

```r
# Required R packages
install.packages(c(
  "baseballr",     # NCAA data scraping
  "rstan",         # Bayesian modeling
  "brms",          # Distribution fitting / model comparison
  "loo",           # LOO-PSIS model selection
  "tidyverse",     # Data wrangling
  "lubridate",     # Date/time handling
  "arrow",         # Parquet I/O (fast, columnar storage)
  "httr",          # HTTP requests for Odds API
  "jsonlite",      # JSON parsing
  "pbapply",       # Progress bars
  "ggplot2",       # Visualization
  "ggridges",      # Ridge density plots
  "viridis",       # Color palettes
  "bayesplot",     # MCMC diagnostics
  "parallel"       # Parallel processing
))
```

## Key Constraints

- **NO FUZZY MATCHING — ANYWHERE.** Use NCAA player_id for players, ncaa_school_id/team_id for teams, and a manually-verified static crosswalk for Odds API ↔ baseballr team name mapping. If an exact match can't be made, flag it for manual review — never guess. Fuzzy matching introduces silent errors that cascade through the entire pipeline.
- **Temporal integrity** — NEVER use future data in training. Weekly batched feature computation.
- **Walk-forward validation** — Always. No in-sample backtesting.
- **Start simple** — Get the base model working before adding complexity.
- **Validate against scores** — After parsing PBP into run events, verify reconstructed scores match actual final scores for EVERY game.
- **Name matching** — Build the crosswalk early. Every downstream analysis depends on it.
- **Quota management** — Track Odds API usage carefully. The `x-requests-remaining` header is your friend.
- **Rate limiting** — Both baseballr (NCAA website) and Odds API have rate limits. Build in sleep() calls and retry logic.
