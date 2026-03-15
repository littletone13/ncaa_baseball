#!/bin/bash
# NCAA Baseball Daily Pipeline
# Full simulation + odds pull + copy to Desktop
# Designed to run via scheduled task at 4 AM CST daily

set -euo pipefail

cd /Users/anthonyeding/ncaa_baseball/ncaa_baseball-1

# Load environment (API key)
source ~/.zshrc

DATE=$(date '+%Y-%m-%d')
LOGFILE="data/logs/daily_pipeline.log"
mkdir -p data/logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

log "=== DAILY PIPELINE START: $DATE ==="

# Step 1: Run simulation
log "Step 1: Running simulation (5000 draws)..."
.venv/bin/python3 scripts/predict_day.py \
    --date "$DATE" \
    --N 5000 \
    --out "data/processed/predictions_${DATE}.csv" \
    >> "$LOGFILE" 2>&1
log "Step 1: Simulation complete"

# Step 2: Copy predictions to Desktop
DESKTOP_FILE="$HOME/Desktop/ncaaBases_projections_${DATE}.csv"
cp "data/processed/predictions_${DATE}.csv" "$DESKTOP_FILE"
log "Step 2: Copied to $DESKTOP_FILE"

# Step 3: Pull fresh odds
log "Step 3: Pulling odds..."
ODDS_API_KEY="$THE_ODDS_API_KEY" .venv/bin/python3 scripts/pull_odds.py \
    --mode current \
    --regions us,us2,eu \
    --markets h2h,totals \
    >> "$LOGFILE" 2>&1
log "Step 3: Odds pull complete"

log "=== DAILY PIPELINE COMPLETE: $DATE ==="
echo "" >> "$LOGFILE"
