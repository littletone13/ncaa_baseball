---
name: daily-predictions
description: Run the full daily NCAA baseball predictions pipeline — simulate games, pull odds, compare model vs market, and copy projections to desktop.
user_invocable: true
---

# Daily Predictions Pipeline

Run predictions for a given date (default: today). This pipeline fetches the day's schedule, resolves starting pitchers, pulls live weather with wind adjustments using real stadium HP→CF bearings, runs 5000-draw Monte Carlo simulation, pulls market odds, and compares model edges.

## Steps

### 1. Run the simulation

```bash
DATE=$(date +%Y-%m-%d)
.venv/bin/python3 scripts/predict_day.py --date $DATE --N 5000 --out data/processed/predictions_$DATE.csv
```

**What it does:**
- Fetches game schedule from NCAA API (ESPN fallback)
- Resolves teams to canonical IDs via `src/ncaa_baseball/phase1.py`
- Infers starting pitchers from `pitcher_appearances.csv` rotation patterns
- Fetches live weather (Open-Meteo) and computes wind-out using stadium HP→CF bearing from `stadium_orientations.csv`
- Runs Monte Carlo simulation (5000 draws from Stan model posterior)
- Outputs CSV with: win probabilities, moneylines, expected totals, wind-out mph

**Check output:** Should show N games, starter coverage %, and weather coverage.

### 2. Filter to upcoming games only (if some games already final)

Check game statuses via NCAA API:
```python
import json
from urllib.request import Request, urlopen
url = f'https://ncaa-api.henrygd.me/scoreboard/baseball/d1/{DATE.replace("-", "/").replace("-", "/")}'
# Parse year/month/day correctly:
parts = DATE.split("-")
url = f'https://ncaa-api.henrygd.me/scoreboard/baseball/d1/{parts[0]}/{parts[1]}/{parts[2]}'
req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
data = json.loads(urlopen(req, timeout=15).read())
# Filter: game['gameState'] == 'pre' means not started
```

Remove final games from predictions CSV, save as `predictions_{DATE}_upcoming.csv`.

### 3. Pull fresh odds

```bash
source ~/.zshrc
ODDS_API_KEY="$THE_ODDS_API_KEY" .venv/bin/python3 scripts/pull_odds.py --mode current --regions us,us2,eu --markets h2h,totals
```

**Output:** Writes to `data/raw/odds/odds_latest.jsonl` (17-20 games typical).

### 4. Compare model vs market

Join predictions and odds using `canonical_id` as the key:
- Load `canonical_teams_2026.csv` to build `odds_api_name → canonical_id` lookup
- Load predictions keyed by `(home_cid, away_cid)`
- Load odds from `odds_latest.jsonl`, resolve team names via the lookup
- **Important:** Odds JSONL uses `bookmaker_lines` field (NOT `bookmakers`)
- Markets are in `bookmaker_lines[].markets[]` with keys `h2h` and `totals`

American odds conversion:
- Negative ML: `prob = |ml| / (|ml| + 100)`
- Positive ML: `prob = 100 / (ml + 100)`
- Prob to ML: `>= 0.5 → -100*p/(1-p)`, `< 0.5 → 100*(1-p)/p`

Display table with: Model ML, Market ML, Edge %, Model Total, Market Total, Total Edge.

### 5. Copy to desktop

```bash
cp data/processed/predictions_${DATE}_upcoming.csv ~/Desktop/ncaaBases_projections_${DATE}.csv
```

### 6. Report key findings

Highlight:
- Games with >5% ML edge (model disagrees with market)
- Games with >2.0 total edge (OVER/UNDER opportunities)
- Notable wind effects (>10 mph wind-out or wind-in)

## Troubleshooting

- **Unmatched odds teams:** Add `odds_api_name` to `data/registries/canonical_teams_2026.csv`
- **Default 67° bearings:** Run `scripts/measure_stadium_bearings.py --canonical-id <CID>` to get real bearing
- **Weather API timeout:** Open-Meteo is free but can be slow; predictions still work with `--no-weather` flag
- **NCAA API down:** Script falls back to ESPN API automatically
