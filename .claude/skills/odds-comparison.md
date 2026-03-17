---
name: odds-comparison
description: Compare model predictions against market odds to identify edges. Pull fresh odds, match to predictions via canonical team IDs, and display ML and total edges.
user_invocable: true
---

# Odds Comparison

Compare model predictions against live market odds to find value (edges where model disagrees with market).

## Prerequisites

- Predictions CSV already generated (see `daily-predictions` skill)
- Odds API key available: `source ~/.zshrc; echo $THE_ODDS_API_KEY`

## Steps

### 1. Pull fresh odds

```bash
source ~/.zshrc
ODDS_API_KEY="$THE_ODDS_API_KEY" .venv/bin/python3 scripts/pull_odds.py --mode current --regions us,us2,eu --markets h2h,totals
```

Output: `data/raw/odds/odds_latest.jsonl` — one JSON object per game, typically 15-20 games.

### 2. Match odds to predictions

**Key:** Join on `canonical_id`, NOT team names. The pipeline:

1. Load `data/registries/canonical_teams_2026.csv` → build `odds_api_name → canonical_id` map
2. Load predictions CSV → key by `(home_cid, away_cid)` tuple
3. Load `data/raw/odds/odds_latest.jsonl` → resolve `home_team`/`away_team` via the map

**Critical data structure note:**
```python
# Odds JSONL structure:
{
  "home_team": "Texas Tech Red Raiders",
  "away_team": "UTSA Roadrunners",
  "bookmaker_lines": [        # NOT "bookmakers"
    {
      "key": "draftkings",
      "markets": [
        {"key": "h2h", "outcomes": [{"name": "Texas Tech Red Raiders", "price": -137}]},
        {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": 17.0}]}
      ]
    }
  ]
}
```

### 3. American odds ↔ probability conversion

```python
def a2p(ml):
    """American odds → implied probability."""
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    return 100 / (ml + 100)

def p2a(p):
    """Probability → American odds."""
    if p >= 0.5:
        return round(-100 * p / (1 - p))
    return round(100 * (1 - p) / p)
```

### 4. Compute edges

- **ML edge** = `model_home_win_prob - market_implied_prob`
- **Total edge** = `model_total - market_total` (positive = model says OVER)

### 5. Display comparison table

```
  Away                @  Home              Model  Mkt   Edge   | ModTot MktTot TotEdge
  --------------------------------------------------------------------------------------
  UTSA                @  Texas Tech        -137   +124  +13.1% |  16.7   17.0   -0.3
```

### 6. Highlight actionable edges

- **ML edges > 5%**: Model significantly disagrees with market on winner
- **Total edges > 2.0**: OVER/UNDER opportunities
- **Wind-driven totals**: Note games with >10 mph wind-out (high scoring) or wind-in (low scoring)

## Troubleshooting

### Unmatched odds teams (0 matches)
Add missing `odds_api_name` entries to `data/registries/canonical_teams_2026.csv`:
```csv
# Find the canonical row, add odds name to column 7
BSB_BYU → odds_api_name = "BYU Cougars"
NCAA_614564 → odds_api_name = "Butler Bulldogs"
```

The odds API uses full mascot names (e.g., "Texas Tech Red Raiders"), while NCAA uses short names ("Texas Tech"). The `odds_api_name` column bridges this gap.

### Games in odds but not in predictions
These are games not in today's NCAA API schedule (late additions, neutral site games). Re-run `predict_day.py` or note them as unmatched.

### Stale odds
Odds change rapidly. Pull within 30 minutes of game time for best accuracy. The `pull_id` timestamp in `odds_pull_log.jsonl` tracks when each pull occurred.
