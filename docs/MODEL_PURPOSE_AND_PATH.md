# Model purpose and path: are we headed the right way?

This doc states the **main purpose** of the model (from the original codex prompt), what has **actually been built**, and whether the current direction matches that purpose.

---

## 1. Main purpose (from `ncaa_baseball_codex_prompt.md`)

The codex defines the project as:

**Build a production Bayesian sports betting model for NCAA D1 college baseball**, following **Andrew Mack’s Chapter 18 (MLB Log-Linear Negative Binomial Model)** adapted for college baseball.

**Core pieces:**

1. **Run-event model (Mack Ch 18)**  
   - From play-by-play: per-game counts of **run_1**, **run_2**, **run_3**, **run_4** (runs scored on a play).  
   - Log-linear structure: `log(λ) = att[team] + def[opponent] + pitcher[starter] + home_advantage + intercept` for each run type.  
   - **Stan** (or equivalent): fit Negative Binomial / Poisson for each run-event type; team attack/defense and pitcher as parameters.

2. **Simulation**  
   - Draw from the fitted posterior; simulate run-event counts per game; convert to total runs.  
   - Outputs: **win probability**, **moneyline fair prices**, **run line probabilities** (e.g. ±1.5), **over/under at various totals**, team totals.  
   - So **runline and total** come from **simulation**, not from a single win-prob formula.

3. **Market data and line movement**  
   - Odds API: h2h, spreads, totals; **line movement at 5-minute intervals** (open → close).  
   - Line movement is a **signal** (sharp vs public, reverse line movement) and a **calibration benchmark**, not only for backtesting.

4. **Backtest and value**  
   - **Backtest:** offline evaluation only (not part of daily process). Historical games — model prob vs closing line, ROI at edge thresholds, calibration (Brier, log loss). Run periodically or on demand.  
   - **Find value:** compare model to **current** odds; flag value bets; optional Kelly sizing.  
   - **Daily pipeline:** scrape new games, update model (fit or use latest posterior), find value. Backtest is separate from this.

5. **Data**  
   - Performance: **baseballr** (PBP, rosters, player stats, returning production, transfer detection).  
   - The codex assumes NCAA `player_id`, career game logs, and no fuzzy matching for players/teams.

So the **intended path** is: **run events from PBP → Stan (Bayesian run-event model) → simulation → moneyline + runline + total → backtest + find value**, with line movement and a production daily pipeline.

---

## 2. What has actually been built

| Piece | In codex | In repo today |
|-------|----------|----------------|
| **Data: games & teams** | baseballr PBP, NCAA rosters | **ESPN** game JSONL (scores, boxscore, run_events, starters); canonical teams + crosswalk |
| **Data: run events** | Parse PBP → run_1..run_4 | **ESPN** summary gives run_events per game (where PBP available); not yet used in a model |
| **Data: odds** | Odds API, 5-min line movement | Odds API: h2h + spreads + totals **fetched**; **only h2h devigged**; no line-movement pipeline |
| **Model** | Stan run-event (att, def, pitcher) | **Team Elo** + **pitcher RA9** (shrunk) + **bullpen workload**; no Stan, no run-event fit |
| **Outputs** | Win prob, moneyline, **runline**, **total** from simulation | **Moneyline win prob only** (Elo ± pitcher adj ± market blend); no runline/total from model |
| **Simulation** | Monte Carlo from posterior | None |
| **Backtest** | Model vs closing, ROI, calibration | **None** |
| **Find value** | Model vs current odds, flag bets | Only **compare** (model vs devigged h2h in a CSV); no “find value” script or daily pipeline |
| **Line movement** | Open/close, sharp vs public | Not built |

So: we have a **simpler, different stack** — ESPN data, team Elo, pitcher layer, moneyline-only comparison to odds. We do **not** have the Bayesian run-event model, simulation, runline/total outputs, backtest, or line movement that the codex describes as the main purpose.

---

## 3. Are we headed the right path?

**If “right path” = the codex’s main purpose (production Bayesian run-event model → simulation → runline/total → backtest → value):**

- **No.** The repo is not on that path yet. It’s on a “simple model first” path (Elo + pitcher + market blend, moneyline only), with no Stan, no simulation, no runline/total from the model, no backtest, and no line movement.

**If “right path” = get to a usable, bettable model without shortcuts:**

- The **data** (canonical teams, ESPN games, run_events in JSONL, pitching lines, 2026 games) can support **either**:
  - **Path A (codex):** Run-event table → Stan (or PyMC) fit → simulation → moneyline + runline + total → backtest → find value.  
  - **Path B (simplified):** Keep Elo + pitcher; add **backtest** (moneyline); add a **simple** runline/total (e.g. RPG-based or Elo-based run distribution → simulate or approximate); then runline/total devig and compare.

So:

- **Fully aligned with the codex** would mean: implementing the **run-event model** (Stan or equivalent), then **simulation**, then **runline/total**, then **backtest** and **find value**. That’s the path the original prompt assumed.
- **Current work** is useful (data pipeline, Elo, pitcher layer, moneyline compare) but is a **detour** from the codex’s main purpose until we either (1) add run-event model + simulation + backtest, or (2) explicitly choose “simple path” and document it, then still add backtest and runline/total (simpler method) so we’re not stuck in planning.

---

## 4. Recommendation

1. **Decide which path you want**  
   - **Codex path:** Run-event model (Mack Ch 18 style) → Stan (or PyMC) → simulation → moneyline + runline + total → backtest → find value.  
   - **Simple path:** Elo + pitcher + (optional) market blend; add **one backtest** (moneyline vs closing); then add **runline/total** via a simple run distribution (e.g. team RPG + Poisson or NegBin) and simulate or approximate runline/total probs.

2. **If codex path:** Next concrete steps are: (a) build a **run_events** table from ESPN (we already have run_events in game JSONL); (b) specify and fit a **Stan (or PyMC) run-event model**; (c) implement **simulation**; (d) implement **backtest** and **find_value** (or 13_find_value equivalent). Data and docs should then prioritize what the Stan/sim pipeline needs.

3. **If simple path:** Keep current Elo + pitcher stack; add a **single backtest script** (historical games, model win prob vs closing, ROI/CLV); then add **runline/total** (e.g. expected runs from RPG or Elo, then runline/total probs) and **runline/total devig** so we can compare. Document “we use simple path” in the data plan so the codex remains the long-term reference but the repo’s immediate goal is clear.

4. **Either path:** Add one **status doc** or section that says: “Done: X. Next: Y.” so progress is code and one place, not scattered plan text.

---

## 5. One-sentence summary

The **main purpose** in the codex is a **production Bayesian run-event model (Mack Ch 18) with Stan, simulation, runline/total, backtest, and find value**. Right now we have **data + Elo + pitcher + moneyline compare** and no run-event model, no simulation, no runline/total from the model, and no backtest — so we are **not** yet on the codex’s main path; we’re on a simpler path that needs either a turn toward the codex or an explicit “simple path” plus backtest and runline/total to reach your model goals.

---

## 6. Path A checklist: what to do to get back on the codex path

Ordered steps so you can execute one at a time. Each step has a **deliverable** and **dependency**.

### 6.1 Run-events table (data for Stan)

**Goal:** One flat table of games with run-event counts and starter IDs, resolved to canonical team (and optionally pitcher) indices for Stan.

**Inputs:** `data/raw/espn/games_2024.jsonl`, `games_2025.jsonl`, `games_2026.jsonl` (only rows where `run_events` is non-null); `data/registries/canonical_teams_2026.csv`.

**Deliverable:** `data/processed/run_events.csv` (or `.parquet`) with columns:

- `event_id`, `game_date`, `season`
- `home_canonical_id`, `away_canonical_id`
- `home_pitcher_espn_id`, `away_pitcher_espn_id` (from `starters.home_pitcher.espn_id` / `away_pitcher.espn_id`)
- `home_run_1`, `home_run_2`, `home_run_3`, `home_run_4`, `away_run_1` … `away_run_4`
- `home_score`, `away_score`

**Script to add:** `scripts/build_run_events_from_espn.py` — stream ESPN JSONL, keep games with `run_events`, resolve team names to `canonical_id`, output the table. No new scrapes; use existing JSONL.

**Dependency:** None (we already have run_events in JSONL).

---

### 6.2 Index lookups for Stan (team and pitcher IDs)

**Goal:** Stan needs integer indices: `team_id` in 1..N_teams, `pitcher_id` in 1..N_pitchers. We need stable mappings so we can turn `canonical_id` / `pitcher_espn_id` into indices and back.

**Deliverables:**

- `data/processed/run_event_team_index.csv`: `canonical_id`, `team_idx` (1..N).
- `data/processed/run_event_pitcher_index.csv`: `pitcher_espn_id`, `pitcher_idx` (1..M). Use “unknown” or 0 for missing starter if needed.

**Script:** Can live inside the same `build_run_events_from_espn.py` or a small `scripts/build_run_event_indices.py` that reads `run_events.csv`, collects unique teams and pitchers, assigns indices, writes the two CSVs. Run after 6.1.

**Dependency:** 6.1 done.

---

### 6.3 Stan (or PyMC) run-event model

**Goal:** Fit the Mack Ch 18–style model: for each run type (1–4), log-linear rate with attack, defense, pitcher, home advantage. Likelihood: NegBin for run_1/run_2, Poisson for run_3/run_4 (or all NegBin; codex says run LOO-PSIS to choose).

**Deliverables:**

- `stan/ncaa_baseball_run_events.stan` (or equivalent PyMC script in `scripts/`):  
  - Data: games, with `home_team_idx`, `away_team_idx`, `home_pitcher_idx`, `away_pitcher_idx`, and counts `home_run_1`..`away_run_4`.  
  - Parameters: `att_run_X`, `def_run_X`, `pitcher_ability` (per run type or shared), `home_advantage_run_X`, intercepts.  
  - Priors: as in codex (tighter for college, e.g. `normal(0, 0.15)`).
- `scripts/fit_run_event_model.py` (or `.R`): load `run_events.csv` and index CSVs, build Stan data block, call Stan (CmdStanPy / PyStan / RStan), save posterior draws (e.g. `data/processed/run_event_posterior.csv` or `.nc`).

**Dependency:** 6.1 and 6.2 done.

---

### 6.4 Simulation (moneyline, runline, total)

**Goal:** For a given matchup (home_team_idx, away_team_idx, home_pitcher_idx, away_pitcher_idx), draw from the fitted posterior to simulate run-event counts, convert to total runs, repeat N times. Output win prob, runline cover probs, total runs distribution.

**Deliverables:**

- `scripts/simulate_run_event_game.py` (or `src/ncaa_baseball/simulate.py`):  
  - Inputs: posterior (or fitted model), one matchup (team + pitcher indices), N simulations.  
  - For each draw: sample home/away run_1..run_4 from the model’s likelihood (or use posterior predictive), sum to total runs, compare home vs away (win/loss), home margin (runline), total runs.  
  - Outputs: `win_prob_home`, `runline_cover_home_prob` (e.g. home -1.5), `over_prob` at a given total (e.g. 11.5), and optionally full distribution of total runs.

**Dependency:** 6.3 done (posterior or fitted model available).

---

### 6.5 Backtest (model vs closing line) — offline only, not daily

**Goal:** Offline evaluation: historical games that have both (a) run_events (so we could have predicted them with the model) and (b) closing odds. For each game, run simulation (or use cached posterior predictive), get model moneyline (and optionally runline/total), compare to closing line, compute ROI at edge thresholds and basic calibration. **Not** part of the daily pipeline; run on demand or periodically (e.g. weekly or end-of-season).

**Deliverables:**

- `data/processed/closing_lines.csv`: at least `event_id` (or game identifier), `game_date`, `home_canonical_id`, `away_canonical_id`, `closing_fair_home`, `closing_fair_away` (devigged). Optionally runline/total closing when we have devig for those. Source: Odds API historical or existing odds JSONL with a “closing” notion (e.g. last snapshot per event).
- `scripts/backtest_run_event_model.py`:  
  - Load run_events table and closing lines; join on event_id (or date + teams).  
  - For each historical game, get model probs from simulation (or from a pre-computed table of model outputs).  
  - Compare model vs closing; compute ROI at 2%, 3%, 5% edge; optionally Brier/log loss, calibration buckets.  
  - Write `data/processed/backtest_results.csv` and a short summary (e.g. ROI by threshold).

**Dependency:** 6.4 done; closing odds available (may require runline/total devig in `pull_odds.py` or a separate script).

---

### 6.6 Runline and total devig (so we can compare model to market)

**Goal:** Today we only devig h2h. To backtest and find value on runline/total, we need consensus runline and total and their fair probabilities.

**Deliverables:**

- In `scripts/pull_odds.py` (or a shared `devig` module): extend devig to spreads and totals (Power or Shin), same way we do for h2h.  
- Stored in odds JSONL or a processed table: e.g. `consensus_runline` (e.g. home -1.5), `runline_fair_home_cover`, `consensus_total`, `over_fair_prob`.

**Dependency:** None for implementation; needed for 6.5 and 6.7 to compare runline/total.

---

### 6.7 Find value (model vs current odds)

**Goal:** For upcoming games, run simulation to get model moneyline (and runline/total when available); compare to current odds; flag bets where model prob > devigged market + threshold; optional Kelly sizing.

**Deliverables:**

- `scripts/find_value.py`:  
  - Input: list of upcoming matchups (from Odds API or schedule) and current odds.  
  - For each matchup: resolve to canonical + pitcher if known, run simulation (6.4), get model probs.  
  - Compare to devigged current odds; flag if edge > threshold; output a small table (e.g. CSV or print): game, side, model_prob, market_fair, edge, optional stake.

**Dependency:** 6.4 done; current odds and devig (h2h; runline/total once 6.6 done).

---

### 6.8 Line movement (optional but in the codex)

**Goal:** Odds API historical snapshots at 5-min intervals → opening line, closing line, and simple movement features (open-to-close shift, max move). Used as a signal and for calibration.

**Deliverables:**

- Script(s) to fetch or load historical odds at multiple timestamps per event.  
- `data/processed/line_movement.csv`: `event_id`, `open_fair_home`, `close_fair_home`, optional movement features.  
- Integration in backtest or find_value: e.g. “only flag value when line moved toward us” or “report ROI by line-move bucket.”

**Dependency:** Odds API quota and historical endpoint; can be done after 6.5/6.7.

---

### Order of execution (Path A)

1. **6.1** — Build run_events table from ESPN JSONL.  
2. **6.2** — Build team and pitcher index CSVs.  
3. **6.3** — Implement and fit Stan (or PyMC) run-event model; save posterior.  
4. **6.4** — Implement simulation (moneyline, runline, total).  
5. **6.6** — Add runline/total devig (in parallel or before 6.5).  
6. **6.5** — Build closing lines table and backtest script; run backtest **offline** (on demand or periodically; not in the daily pipeline).  
7. **6.7** — Implement find_value script (this *is* part of the daily process: scrape → update model → find value).  
8. **6.8** — Line movement when you’re ready (optional).

The first step that gets you “back on path” is **6.1**: one script that produces `data/processed/run_events.csv` from existing ESPN JSONL. Everything else depends on that or on the model/sim that uses it.
