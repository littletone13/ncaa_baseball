# Data plan: canonical team library and rosters (2026 minimum)

Goal: **one solid, cleaned dataset** before modeling — canonical team library and rosters for 2026 at minimum.

**See also:** [MODEL_BUILD_REFINEMENTS.md](MODEL_BUILD_REFINEMENTS.md) — phased model rollout, score-only / early-season handling, roster turnover (“this season first”), and andrew_mack as reference.

---

## 0. Current-season data (2026)

With high roster turnover, **this season’s data** should drive team and pitcher strength more than prior-year Elo.

| Source | What’s available | Use |
|--------|-------------------|-----|
| **ESPN** (games_2026.jsonl) | 2026 scores, boxscore, starters, pitching lines (IP, ER, PC) | Team RPG, SP/RP strength, pitcher workload (stamina). Primary for in-season strength. |
| **D1Baseball** | **Lineup:** [team/…/lineup/](https://d1baseball.com/team/michigan/lineup/) — batting order and SP by game, “most games” by position. **Stats:** [team/…/stats/](https://d1baseball.com/team/michigan/stats/) — batting (BA, OBP, SLG, OPS, wOBA, wRC+, etc.), pitching (ERA, IP, GS, SV, FIP, WHIP, etc.), team hitting/pitching leaders; year selector (2026, 2025, …). | Lineup/SP for “who’s playing”; stats for team/pitcher strength (ERA, wOBA, FIP) and optional scrape for current-season totals. |

Pipeline: scrape or pull 2026 games (ESPN + optional D1Baseball) → build pitching lines and team stats for **2026 only** (or rolling) → feed into Elo/RPG/SP-RP model with prior from last year, not the other way around.

### 0.1 Data status (2026 pipeline executed)

| Step | Output | 2026 coverage |
|------|--------|----------------|
| Scrape ESPN 2026 | `data/raw/espn/games_2026.jsonl` | All final games from 2026-02-13 through scrape date (merge full + scoreboard). |
| Merge 2026 | Same file | One JSONL: games with boxscore + scoreboard-only; no duplicate event_ids. |
| Build games | `data/processed/games_espn.csv` | 2026 rows with date, teams, scores; canonical resolution for ML/Elo. |
| Pitching lines | `data/processed/pitching_lines_espn.csv` | 2026 season column: lines from 2026 games that have boxscore. |
| Rosters (boxscore) | `data/processed/rosters/rosters_espn_boxscore.csv` | 2026 season: players who appeared in 2026 boxscores, position, roles. |

To refresh: run `scripts/scrape_espn.py` for new dates (then `merge_espn_2026.py` if you used scoreboard + full), then `build_games_from_espn.py`, `build_pitching_from_espn.py`, `build_rosters_from_espn_boxscore.py` with `--seasons` including 2026.

---

## 1. Canonical team library (2026)

### 1.1 Single source of truth

- **Primary key:** `ncaa_teams_id` (integer from stats.ncaa.org). Stable and used across NCAA, scoreboard, and (with crosswalk) Odds API.
- **Output:** One registry file used everywhere: `data/registries/canonical_teams_2026.csv`.

### 1.2 Schema: `canonical_teams_2026.csv`

| Column | Description | Required |
|--------|-------------|----------|
| `academic_year` | 2026 | Yes |
| `ncaa_teams_id` | NCAA team id (stats.ncaa.org) | Yes |
| `team_name` | Display name (NCAA style, e.g. "Florida St.", "NC State") | Yes |
| `conference` | Conference short name | Yes |
| `conference_id` | NCAA conference id | Yes |
| `canonical_id` | Our internal id (e.g. BSB_* or NCAA_*) for downstream code | Yes (filled from crosswalk or NCAA_&lt;id&gt;) |
| `odds_api_name` | Odds API display name when known (e.g. "Georgia Tech Yellow Jackets") | No |
| `baseballr_team_id` | baseballr team_id when known (for roster/PBP) | No |
| `baseballr_season_id` | baseballr season_id when known (e.g. for 2024 = 16580) | No |
| `notes` | Manual override or ambiguity note | No |

All 2026 D1 teams appear exactly once. No fuzzy matching: `canonical_id` and `odds_api_name` come from the manual crosswalk or default to `NCAA_<ncaa_teams_id>`.

### 1.3 Inputs

- **`data/registries/ncaa_d1_teams_2026.csv`** — Full D1 list from `scrape_ncaa_d1_team_registry.py` (academic_year, division, sport_code, conference_id, conference, ncaa_teams_id, team_name). ~308 teams.
- **`data/registries/name_crosswalk_manual_2026.csv`** — Manual mapping: ncaa_teams_id, ncaa_team_name, conference, canonical_team_id, canonical_school, odds_api_team_name, baseballr_team_name, notes. Fill canonical_team_id and odds_api_team_name where we care (e.g. lined games).

### 1.4 How to build

```bash
python3 scripts/build_canonical_teams_2026.py \
  --ncaa-csv data/registries/ncaa_d1_teams_2026.csv \
  --crosswalk data/registries/name_crosswalk_manual_2026.csv \
  --out data/registries/canonical_teams_2026.csv
```

- Join NCAA list to crosswalk on `ncaa_teams_id`.
- For rows with no crosswalk: `canonical_id = NCAA_<ncaa_teams_id>`, `team_name` from NCAA list.
- Validate: no duplicate `ncaa_teams_id`, no duplicate `canonical_id` among non-empty.

### 1.5 Keeping it clean

- **Scoreboard:** When building scoreboard-derived team lists (e.g. `scoreboard_teams_2026.csv`), join to `canonical_teams_2026.csv` on `ncaa_teams_id` so every team has one canonical row.
- **Odds:** Join Odds API events to canonical registry via `odds_api_name` (or a small odds-specific alias table). Fill `name_crosswalk_manual_2026.csv` `odds_api_team_name` for teams that appear in odds.
- **No new fuzzy matching:** Add new teams or name variants only via explicit crosswalk rows or script defaults (e.g. NCAA_ id).

---

## 2. Rosters (2026)

### 2.1 Current state

- **R script `02_scrape_rosters.R`** uses `baseballr::load_ncaa_baseball_teams()` to get team list and **season_id**. That dataset currently goes through **2024** only, so 2026 is not available there.
- **Roster URL (NCAA):** `https://stats.ncaa.org/team/<ncaa_teams_id>/roster/<year_id>`. The last segment is the **roster year_id** (not the same as baseballr’s `season_id`). The roster page’s season dropdown uses these values (e.g. 2025-26 → 614802).
- **Stats.ncaa.org:** As of the coverage report, 2025+ may sit behind Akamai/interstitial; non-browser scraping can fail. So 2026 roster scraping may require browser automation or waiting for NCAA to relax blocking.

### 2.2 Options for 2026 rosters

1. **Roster year_id lookup**  
   - **Done.** The NCAA roster page season dropdown (e.g. on `stats.ncaa.org/team/736/roster/…`) gives year_id per academic season. For **2025-26** (academic year 2026) the value is **614802**. A lookup table is in `data/registries/ncaa_roster_year_id_lu.csv` (academic_year → ncaa_roster_year_id). Use it in roster scraping.

2. **Python roster scraper**  
   - Input: `canonical_teams_2026.csv` (or a list of `ncaa_teams_id`) + roster year_id for 2026 from `data/registries/ncaa_roster_year_id_lu.csv` (**614802** for academic_year 2026).  
   - For each team, GET `https://stats.ncaa.org/team/<ncaa_teams_id>/roster/614802`, parse table (player name, player_id from `/players/` links, position, etc.), write `data/raw/rosters/roster_2026_<ncaa_teams_id>.parquet` (or CSV).  
   - Same schema as R script where possible (player_id, player_name, team_id, year, conference, etc.) so downstream (e.g. export_rosters_csv, coverage report) can use 2026.

3. **ESPN boxscore-derived rosters (in place)**  
   - **Script:** `scripts/build_rosters_from_espn_boxscore.py` — reads `data/raw/espn/games_*.jsonl`, aggregates every player who appears in a game boxscore (batting or pitching) per team per season, resolves team names to canonical, writes `data/processed/rosters/rosters_espn_boxscore.csv`.  
   - **Columns:** `canonical_id`, `team_name`, `season`, `player_name`, `espn_id`, **`position`**, `roles`.  
   - **Position:** From ESPN API when provided (stored by `scrape_espn.py` in each game’s boxscore); when missing or “Unspecified”, inferred from role: **P** (pitching only), **B** (batting only), **P/B** (both). Re-scraping with the updated scraper stores `position` in new game JSONL for future use.

4. **Manual / delayed**  
   - If we can’t get 2026 season_id or scrape reliably, document “rosters_2026 = TBD” and rely on 2024 (or latest available) rosters for returning-production logic until 2026 data is available.

### 2.3 Target roster schema (aligned with existing)

- `year` = 2026  
- `team_id` = ncaa_teams_id (integer)  
- `team_name`, `conference`, `conference_id`, `division`  
- `season_id` (when known)  
- Per player: `player_id` (from NCAA or ESPN), `player_name`, plus any columns we parse (position, jersey, class, etc.).  
- Store under `data/raw/rosters/` so `export_rosters_csv.R` (or a Python equivalent) can export `data/processed/rosters/rosters_2026.csv`.

### 2.4 Immediate next steps for rosters

1. **2026 roster year_id:** **614802** (stored in `data/registries/ncaa_roster_year_id_lu.csv`).  
2. **Add `scripts/scrape_rosters_2026.py`** (or extend a generic scraper) that:  
   - Reads `data/registries/canonical_teams_2026.csv`,  
   - For each `ncaa_teams_id`, fetches roster with 2026 roster year_id (**614802**),  
   - Writes one file per team under `data/raw/rosters/`.  
3. **Script added:** `scripts/scrape_rosters_2026.py` — reads canonical teams and roster year_id **614802**, fetches each team’s roster page, parses the table, writes `data/raw/rosters/roster_2026_<ncaa_teams_id>.csv`. **If NCAA blocks (403):** stats.ncaa.org often blocks server-side requests. Use browser-based scraping (Playwright/Selenium), a different network/VPN, or manual export; the script is ready once requests succeed.

---

## 3. Checklist (2026 minimum)

- [ ] **Canonical team library**
  - [ ] `ncaa_d1_teams_2026.csv` exists and has ~308 D1 teams.
  - [ ] `name_crosswalk_manual_2026.csv` exists; complete `canonical_team_id` and `odds_api_team_name` for all teams that appear in odds/scoreboard (priority).
  - [ ] `scripts/build_canonical_teams_2026.py` runs and produces `canonical_teams_2026.csv` with no duplicate ncaa_teams_id or canonical_id.
  - [ ] Downstream scripts (scoreboard, odds, modeling) use `canonical_teams_2026.csv` as the team source.
- [ ] **Rosters**
  - [x] 2026 roster year_id identified and recorded: **614802** (`ncaa_roster_year_id_lu.csv`).
  - [x] Roster scraper `scripts/scrape_rosters_2026.py` added; writes `data/raw/rosters/roster_2026_<id>.csv`. (Site may return 403 — use browser/VPN if needed.)
  - [ ] At least one export path (e.g. `rosters_2026.csv`) exists once scraping succeeds.

---

## 4. File map

| File | Purpose |
|------|--------|
| `data/registries/ncaa_roster_year_id_lu.csv` | academic_year → ncaa_roster_year_id for roster URL (2026 = **614802**) |
| `data/registries/ncaa_d1_teams_2026.csv` | Full D1 list from NCAA (source) |
| `data/registries/name_crosswalk_manual_2026.csv` | Manual ncaa_teams_id → canonical_id, odds_api_name, etc. |
| `data/registries/canonical_teams_2026.csv` | **Single canonical team registry for 2026** (built) |
| `data/registries/scoreboard_teams_2026.csv` | Teams seen in scoreboard games (join to canonical on ncaa_teams_id) |
| `data/raw/rosters/roster_2026_<id>.csv` | One roster per team from `scrape_rosters_2026.py` (may get 403 from NCAA) |
| `data/processed/rosters/rosters_espn_boxscore.csv` | ESPN boxscore-derived rosters (canonical_id, player_name, position, roles, season) |
| `data/processed/rosters/rosters_2026.csv` | Flattened roster export for 2026 (NCAA or combined) |

---

## 5. Crosswalk completion (name_crosswalk_manual_2026.csv)

- **Purpose:** Map every NCAA team (by `ncaa_teams_id`) to a stable `canonical_team_id` and, where we have it, `odds_api_team_name` so odds and scoreboard data join to the canonical registry.
- **Current state:** The file has one row per 2026 D1 team (~308 rows). Many rows have empty `canonical_team_id` and `odds_api_team_name`; the build script then assigns `canonical_id = NCAA_<ncaa_teams_id>` so every team still has a unique id.
- **Priority fills:**
  1. **Teams that appear in Odds API events** — Fill `odds_api_team_name` with the exact string from the API (e.g. "Georgia Tech Yellow Jackets"). Use a recent odds JSONL or events JSON to list names, then match by school (manually or with a script that suggests matches from `team_name`).
  2. **Teams you care about for modeling** — Optionally set `canonical_team_id` to a BSB_* or other internal id for consistency with `teams_baseball.yaml` or legacy code.
- **No fuzzy matching:** When in doubt, leave blank or add a note; the pipeline will still run with `NCAA_<id>` and empty `odds_api_name` until you fill it.
