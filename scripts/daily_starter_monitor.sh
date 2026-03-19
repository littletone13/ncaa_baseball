#!/bin/bash
# Daily starter monitor — run this in the morning before games start
# It will check Sidearm live stats pages every 10 minutes for confirmed starters
# and update the starters.csv when it finds them.
#
# Usage:
#   ./scripts/daily_starter_monitor.sh 2026-03-19
#   # Or for today:
#   ./scripts/daily_starter_monitor.sh
#
# To run in background:
#   nohup ./scripts/daily_starter_monitor.sh 2026-03-19 > /tmp/starter_monitor.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

DATE="${1:-$(date +%Y-%m-%d)}"
VENV=".venv/bin/python3"

echo "=== Daily Starter Monitor for ${DATE} ==="
echo "Started at $(date)"
echo ""

# Step 1: Generate predictions if not already done
if [ ! -f "data/processed/predictions_${DATE}.csv" ]; then
    echo "Step 1: Running predictions..."
    $VENV scripts/predict_day.py --date "$DATE" --N 5000 --out "data/processed/predictions_${DATE}.csv"
    cp "data/processed/predictions_${DATE}.csv" ~/Desktop/ncaaBases_projections_${DATE}.csv
    echo ""
fi

# Step 2: Start the daemon to monitor starters
echo "Step 2: Starting starter monitor daemon (checks every 10 min)..."
echo "  Will scrape Sidearm 45 min before each game's first pitch"
echo "  Press Ctrl+C to stop"
echo ""

$VENV scripts/scrape_sidearm_starters.py \
    --date "$DATE" \
    --daemon \
    --interval 600 \
    --window 45
