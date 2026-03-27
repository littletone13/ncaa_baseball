#!/bin/bash
# NCAA Baseball Odds Data Collector
# Runs via cron to pull fresh odds at scheduled intervals
# Logs to data/raw/odds/cron_log.txt

cd /Users/anthonyeding/ncaa_baseball/ncaa_baseball-1

# Load API key from environment
source ~/.zshrc

LOGFILE="data/raw/odds/cron_log.txt"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "[$TIMESTAMP] Starting odds pull..." >> "$LOGFILE"

ODDS_API_KEY="$THE_ODDS_API_KEY" .venv/bin/python3 scripts/pull_odds.py \
    --mode current \
    --regions us,us2,eu \
    --markets h2h,spreads,totals \
    >> "$LOGFILE" 2>&1

echo "[$TIMESTAMP] Done" >> "$LOGFILE"
echo "" >> "$LOGFILE"
