# NCAA Baseball Prediction Pipeline
#
# Usage:
#   make extract        # Re-extract from ESPN JSONL (after new scrape)
#   make indices        # Rebuild Stan model indices
#   make tables         # Rebuild pitcher + team lookup tables
#   make model          # Refit Stan model (slow, ~10 min)
#   make predict        # Run daily predictions (set DATE=YYYY-MM-DD)
#   make daily          # predict + pull odds
#   make rebuild        # Full rebuild from extract through tables
#   make all            # Full rebuild + model refit + predict
#
# Examples:
#   make predict DATE=2026-03-14
#   make daily DATE=2026-03-14
#   make rebuild
#   make model

PYTHON = .venv/bin/python3
DATE ?= $(shell date +%Y-%m-%d)
N_SIMS ?= 5000
DATABASE_URL ?=

# ── Layer 1: Extract from ESPN JSONL ──────────────────────────────
ESPN_JSONL = $(wildcard data/raw/espn/games_*.jsonl)
MANIFEST = data/processed/extract_manifest.json

$(MANIFEST): scripts/extract_espn.py $(ESPN_JSONL) data/registries/canonical_teams_2026.csv
	$(PYTHON) scripts/extract_espn.py
	@echo "✓ Extract complete"

extract: $(MANIFEST)

# ── Layer 1b: Integrate NCAA boxscores into pitcher_appearances ───
NCAA_BOXSCORES = $(wildcard data/raw/ncaa/boxscores_*.jsonl)

integrate-ncaa: $(MANIFEST)
	@if ls data/raw/ncaa/boxscores_*.jsonl 1>/dev/null 2>&1; then \
		$(PYTHON) scripts/integrate_ncaa_boxscores.py; \
		echo "✓ NCAA boxscores merged into pitcher_appearances"; \
	else \
		echo "⚠ No NCAA boxscores found, skipping"; \
	fi

# ── Layer 1c: Build linescore run_events + merge into run_events ─
NCAA_LINESCORES = $(wildcard data/raw/ncaa/linescores_*.jsonl)

merge-linescores: $(MANIFEST)
	@if ls data/raw/ncaa/linescores_*.jsonl 1>/dev/null 2>&1; then \
		$(PYTHON) scripts/build_linescore_run_events.py; \
		$(PYTHON) scripts/merge_run_events.py; \
		echo "✓ Linescores merged into run_events"; \
	else \
		echo "⚠ No NCAA linescores found, skipping"; \
	fi

# ── Layer 2: Model indices ────────────────────────────────────────
TEAM_INDEX = data/processed/run_event_team_index.csv
PITCHER_INDEX = data/processed/run_event_pitcher_index.csv

$(TEAM_INDEX) $(PITCHER_INDEX): scripts/build_run_event_indices.py data/processed/run_events.csv
	$(PYTHON) scripts/build_run_event_indices.py

indices: $(TEAM_INDEX) $(PITCHER_INDEX)

# ── Layer 2b: Park factors ────────────────────────────────────────
PARK_FACTORS = data/processed/park_factors.csv

$(PARK_FACTORS): scripts/build_park_factors.py $(MANIFEST)
	$(PYTHON) scripts/build_park_factors.py

park-factors: $(PARK_FACTORS)

# ── Layer 2c: Bullpen quality ─────────────────────────────────────
BULLPEN = data/processed/bullpen_quality.csv

$(BULLPEN): scripts/compute_bullpen_quality.py data/processed/pitcher_appearances.csv
	$(PYTHON) scripts/compute_bullpen_quality.py

bullpen: $(BULLPEN)

# ── Layer 2d: Weekend rotations ───────────────────────────────────
ROTATIONS = data/processed/weekend_rotations.csv

$(ROTATIONS): scripts/build_weekend_rotations.py data/processed/pitcher_appearances.csv
	$(PYTHON) scripts/build_weekend_rotations.py

rotations: $(ROTATIONS)

# ── Layer 3: Unified lookup tables ────────────────────────────────
PITCHER_TABLE = data/processed/pitcher_table.csv

$(PITCHER_TABLE): scripts/build_pitcher_table.py $(PITCHER_INDEX) $(ROTATIONS)
	$(PYTHON) scripts/build_pitcher_table.py
	@echo "✓ Pitcher table built"

TEAM_TABLE = data/processed/team_table.csv

$(TEAM_TABLE): scripts/build_team_table.py $(TEAM_INDEX) $(BULLPEN)
	$(PYTHON) scripts/build_team_table.py
	@echo "✓ Team table built"

tables: $(PITCHER_TABLE) $(TEAM_TABLE)

# ── Layer 4: Stan model fit ──────────────────────────────────────
POSTERIOR = data/processed/run_event_posterior_2k.csv
META = data/processed/run_event_fit_meta.json

model: $(TEAM_INDEX) $(PITCHER_INDEX) $(PARK_FACTORS) $(BULLPEN)
	$(PYTHON) scripts/fit_run_event_model.py
	@# Subsample posterior to 2K draws for daily use
	head -1 data/processed/run_event_posterior.csv > $(POSTERIOR)
	tail -n +2 data/processed/run_event_posterior.csv | sort -R | head -2000 >> $(POSTERIOR)
	@echo "✓ Model fit complete (posterior subsampled to 2K draws)"

# ── Layer 5: Daily predictions ────────────────────────────────────
PREDICTIONS = data/processed/predictions_$(DATE).csv

predict: $(PITCHER_TABLE) $(TEAM_TABLE)
	$(PYTHON) scripts/predict_day.py --date $(DATE) --N $(N_SIMS) --out $(PREDICTIONS)
	@echo "✓ Predictions for $(DATE) -> $(PREDICTIONS) + Supabase"

# ── Odds ──────────────────────────────────────────────────────────
odds:
	@source ~/.zshrc 2>/dev/null; \
	ODDS_API_KEY="$${THE_ODDS_API_KEY}" \
	$(PYTHON) scripts/pull_odds.py --mode current --regions us,us2,eu --markets h2h,totals,spreads

# ── Postgres odds warehouse ───────────────────────────────────────
odds-db-bootstrap:
	@if [ -z "$(DATABASE_URL)" ]; then echo "Set DATABASE_URL"; exit 1; fi
	$(PYTHON) scripts/load_odds_to_postgres.py --dsn "$(DATABASE_URL)" --create-schema

odds-db-load:
	@if [ -z "$(DATABASE_URL)" ]; then echo "Set DATABASE_URL"; exit 1; fi
	$(PYTHON) scripts/load_odds_to_postgres.py --dsn "$(DATABASE_URL)"

# ── Convenience targets ──────────────────────────────────────────
rebuild: extract integrate-ncaa merge-linescores indices park-factors bullpen rotations tables
	@echo "✓ Full rebuild complete"

daily: predict odds web-export web-push
	@echo "✓ Daily pipeline complete for $(DATE) — live at hoopsbracketanalysis.shop"

all: rebuild model predict
	@echo "✓ Full pipeline complete"

clean-daily:
	rm -rf data/daily/$(DATE)

# ── Web dashboard (standalone repo at ~/hoopsbracketanalysis) ────
WEB_DIR ?= $(HOME)/hoopsbracketanalysis

web-export:
	$(PYTHON) scripts/export_web_data.py --date $(DATE) --out $(WEB_DIR)/public/data/
	@echo "✓ Web data exported for $(DATE) → $(WEB_DIR)/public/data/"

web-push:
	cd $(WEB_DIR) && git add public/data/ && git diff --cached --quiet || \
		(cd $(WEB_DIR) && git commit -m "data $(DATE)" && git push)
	@echo "✓ Pushed to GitHub (auto-deploys to hoopsbracketanalysis.shop)"

web-deploy: web-export web-push
	@echo "✓ Deployed to hoopsbracketanalysis.shop"

web-dev:
	cd $(WEB_DIR) && npm run dev

# ── Supabase database targets ──────────────────────────────────────────
db-load-all:
	SUPABASE_DB_PASSWORD="$$SUPABASE_DB_PASSWORD" $(PYTHON) scripts/load_baseball_to_postgres.py --all
	@echo "✓ All tables loaded to Supabase"

db-load-day:
	SUPABASE_DB_PASSWORD="$$SUPABASE_DB_PASSWORD" $(PYTHON) scripts/load_baseball_to_postgres.py --table games
	SUPABASE_DB_PASSWORD="$$SUPABASE_DB_PASSWORD" $(PYTHON) scripts/load_baseball_to_postgres.py --table appearances
	SUPABASE_DB_PASSWORD="$$SUPABASE_DB_PASSWORD" $(PYTHON) scripts/load_baseball_to_postgres.py --table predictions --date $(DATE)
	@echo "✓ Daily data loaded to Supabase for $(DATE)"

db-load-predictions:
	SUPABASE_DB_PASSWORD="$$SUPABASE_DB_PASSWORD" $(PYTHON) scripts/load_baseball_to_postgres.py --table predictions --date $(DATE)

.PHONY: extract integrate-ncaa merge-linescores indices park-factors bullpen rotations tables model predict odds odds-db-bootstrap odds-db-load rebuild daily all clean-daily web-export web-push web-deploy web-dev db-load-all db-load-day db-load-predictions
