# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

NCAA D1 baseball betting model — a Python data pipeline (scraping, Elo fitting, odds comparison, game projection). No web server, no database, no Docker. See `README.md` for full pipeline commands and `.cursorrules` for architecture/conventions.

### Running the environment

- **Python 3.12+** is required. All deps are declared in `pyproject.toml`; install with `pip install -e ".[dev]"`.
- Ensure `~/.local/bin` is on `PATH` (for `pytest` and other console scripts installed by pip with `--user`).
- R is optional (only needed for `baseballr` scripts under `scripts/*.R`). The core Python pipeline works without R.
- No `.env` file is needed to run the core pipeline. `ODDS_API_KEY` is only required for odds-related scripts (`pull_odds*.py`, `run_phase1_odds_compare.py`).

### Pipeline stages

The full pipeline runs in order. Each stage feeds into the next; see `README.md` for detailed flags.

| # | Stage | Command |
|---|---|---|
| 1 | Build canonical teams | `python3 scripts/build_canonical_teams_2026.py` |
| 2 | Scrape ESPN | `python3 scripts/scrape_espn.py --start YYYY-MM-DD --end YYYY-MM-DD --out data/raw/espn/games_YYYY.jsonl` |
| 3 | Build game matrix | `python3 scripts/build_games_from_espn.py --espn-dir data/raw/espn --out data/processed/games_espn.csv --seasons YYYY` |
| 4 | Fit Elo | `python3 scripts/fit_phase1_elo.py --games data/processed/games_espn.csv --out data/processed/phase1_team_ratings.csv` |
| 5 | Build pitching lines | `python3 scripts/build_pitching_from_espn.py --espn-dir data/raw/espn --out data/processed/pitching_lines.csv --seasons YYYY` |
| 6 | Build pitcher ratings | `python3 scripts/build_pitcher_ratings.py --pitching-lines data/processed/pitching_lines.csv` (outputs `pitcher_ratings.csv`, `team_pitcher_strength.csv`, `bullpen_workload.csv`) |
| 7 | Build run events | `python3 scripts/build_run_events_from_espn.py --espn-dir data/raw/espn --out data/processed/run_events.csv --seasons YYYY` (requires PBP data; ~23% of games have it) |
| 8 | Build run event indices | `python3 scripts/build_run_event_indices.py --run-events data/processed/run_events.csv` (team/pitcher indices for future Stan model) |
| 9 | Project a game | `python3 scripts/project_game.py --team-a "Team A" --team-b "Team B" --neutral` (add `--use-pitchers` after step 6) |
| 10 | Compare to odds | `python3 scripts/run_phase1_odds_compare.py --odds-jsonl <file> --ratings data/processed/phase1_team_ratings.csv` (requires `ODDS_API_KEY` for pulling odds) |

### Other key commands

| Task | Command |
|---|---|
| Install deps | `pip install -e ".[dev]"` |
| Run tests | `python3 -m pytest` (note: `tests/` dir may not exist yet; exit code 5 = no tests collected is expected) |
| Build team registry (YAML) | `python3 scripts/build_team_registry.py teams_baseball.yaml` |

### Optional dependencies

- **R + baseballr**: needed only for `scripts/*.R` (PBP bulk download, roster scraping, coverage reports). Install R packages via `Rscript scripts/install_R_deps.R`. The core Python pipeline does not require R.
- **ODDS_API_KEY**: required for `pull_odds*.py` and `run_phase1_odds_compare.py`. Set in `.env` or `export ODDS_API_KEY=...`. Not needed for scraping, Elo fitting, or game projection.
- **CmdStan / cmdstanr**: planned for the Bayesian Stan model (`stan/ncaa_baseball.stan`) but not yet built; no `.stan` files exist yet.

### Gotchas

- The ESPN scraper (`scrape_espn.py`) fetches game summaries with a default 1s sleep between requests. For a single-day scrape this can take ~1–2 minutes for 100+ games. Use `--skip-summary` for scoreboard-only fast runs.
- `build_games_from_espn.py` expects JSONL files named `games_YYYY.jsonl` inside the `--espn-dir`. The scraper output file name must match this pattern.
- Scripts under `scripts/` use `import _bootstrap` to add `src/` to `sys.path`. Always run scripts from the repo root (`python3 scripts/foo.py`), not from inside `scripts/`.
- No linter (ruff, flake8, mypy) is currently configured. The only dev tool is `pytest`.
- `data/raw/` and `data/processed/` are gitignored. Registry files under `data/registries/` are tracked.
