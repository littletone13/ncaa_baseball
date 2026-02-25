# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

NCAA D1 baseball betting model — a Python data pipeline (scraping, Elo fitting, odds comparison, game projection). No web server, no database, no Docker. See `README.md` for full pipeline commands and `.cursorrules` for architecture/conventions.

### Running the environment

- **Python 3.12+** is required. All deps are declared in `pyproject.toml`; install with `pip install -e ".[dev]"`.
- Ensure `~/.local/bin` is on `PATH` (for `pytest` and other console scripts installed by pip with `--user`).
- R is optional (only needed for `baseballr` scripts under `scripts/*.R`). The core Python pipeline works without R.
- No `.env` file is needed to run the core pipeline. `ODDS_API_KEY` is only required for odds-related scripts (`pull_odds*.py`, `run_phase1_odds_compare.py`).

### Key commands

| Task | Command |
|---|---|
| Install deps | `pip install -e ".[dev]"` |
| Run tests | `python3 -m pytest` (note: `tests/` dir may not exist yet; exit code 5 = no tests collected is expected) |
| Build canonical teams | `python3 scripts/build_canonical_teams_2026.py` |
| Build team registry | `python3 scripts/build_team_registry.py teams_baseball.yaml` |
| Scrape ESPN (1 day) | `python3 scripts/scrape_espn.py --start YYYY-MM-DD --end YYYY-MM-DD --out data/raw/espn/games_YYYY.jsonl` |
| Build game matrix | `python3 scripts/build_games_from_espn.py --espn-dir data/raw/espn --out data/processed/games_espn.csv --seasons YYYY` |
| Fit Elo | `python3 scripts/fit_phase1_elo.py --games data/processed/games_espn.csv --out data/processed/phase1_team_ratings.csv` |
| Project a game | `python3 scripts/project_game.py --team-a "Team A" --team-b "Team B" --neutral` |

### Gotchas

- The ESPN scraper (`scrape_espn.py`) fetches game summaries with a default 1s sleep between requests. For a single-day scrape this can take ~1–2 minutes for 100+ games. Use `--skip-summary` for scoreboard-only fast runs.
- `build_games_from_espn.py` expects JSONL files named `games_YYYY.jsonl` inside the `--espn-dir`. The scraper output file name must match this pattern.
- Scripts under `scripts/` use `import _bootstrap` to add `src/` to `sys.path`. Always run scripts from the repo root (`python3 scripts/foo.py`), not from inside `scripts/`.
- No linter (ruff, flake8, mypy) is currently configured. The only dev tool is `pytest`.
- `data/raw/` and `data/processed/` are gitignored. Registry files under `data/registries/` are tracked.
