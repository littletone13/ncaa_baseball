# NCAA_BASEBALL (data-first)

Goal: build a clean, deterministic dataset pipeline (teams → odds → performance) that we can trust before we model anything.

## Team registry (canonical IDs)

`teams_baseball.yaml` is the source of truth for canonical team IDs (no fuzzy matching).

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
