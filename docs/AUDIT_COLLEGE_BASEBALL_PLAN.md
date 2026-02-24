# Audit: College Baseball Full-Game Plan vs Repo

**Focus:** Full game — moneyline, runline, total runs.  
**Team components:** Fielding, batting, starting pitching, bullpen, fatigue (and pre-built models in `andrew_mack`).

---

## 1. Repo vs plan (high level)

| Area | In repo / local | In plan (`ncaa_baseball_codex_prompt.md`) |
|------|------------------|-------------------------------------------|
| **Odds pipeline** | ✅ Python: `pull_odds.py`, `pull_odds_snapshot.py`, `odds_api*.py` — fetch h2h, spreads, totals; devig h2h only | ✅ h2h, spreads, totals; line movement; devig (Power/Shin) |
| **Run events (Mack)** | ✅ `scrape_espn.py`: PBP → run_1/2/3/4, starters; ESPN boxscore | ✅ PBP → run events; Stan log-linear NegBin/Poisson |
| **Team stats** | ✅ `build_team_game_stats.py`: runs for/against, W-L, RPG (scoreboard only) | ✅ Plus attack/defense/pitcher from Stan; returning production |
| **Stan / simulation** | ❌ No `stan/` or sim scripts in repo | ✅ `stan/ncaa_baseball.stan`, 11_simulate_games.R, 12_backtest.R |
| **R pipeline (baseballr)** | Partial: 00–03 R scripts, `install_R_deps.R`, `export_rosters_csv.R`, `build_coverage_report.R` | ✅ 01–14 full pipeline (PBP, rosters, crosswalk, line movement, fit, simulate, backtest) |
| **Name crosswalk** | ✅ `build_odds_name_crosswalk.py`, manual CSVs in `data/registries/` | ✅ Odds API ↔ baseballr ↔ NCAA; no fuzzy matching |
| **andrew_mack models** | ❌ No `andrew_mack` folder in repo (you said you have pre-built models locally) | N/A — plan is Mack Ch 18 style; your models can plug in here |

---

## 2. Full-game markets: moneyline, runline, total runs

### 2.1 What exists today

- **Odds API:** Fetched with `h2h,spreads,totals` (`CURRENT_MARKETS` in `pull_odds.py`).
- **Raw JSONL:** Each game has `bookmaker_lines[].markets` with:
  - **h2h** → moneyline (e.g. home -870 / away +500).  
  - **spreads** → runline (e.g. home -3.5 / away +3.5, with prices).  
  - **totals** → total runs (e.g. Over/Under 16.5 with prices).
- **Devig:** Only **h2h** is devigged today (`devig_h2h` in `pull_odds.py`). Stored as `consensus_fair_home` / `consensus_fair_away` and per-book `h2h_fair_home` / `h2h_fair_away`.
- **Spreads and totals:** Stored raw in `bookmaker_lines[].markets` but **not** devigged or summarized (no consensus spread/total or fair probabilities).

So: **moneyline** is supported end-to-end (fetch + devig); **runline** and **total runs** are fetched and stored but not yet processed into consensus/fair values or used in any model.

### 2.2 Gaps for full-game focus

1. **Devig spreads and totals** (same multi-book approach as h2h), and store:
   - Consensus runline (e.g. home -3.5) and fair probabilities for home/away cover.
   - Consensus total (e.g. 16.5) and fair Over/Under probabilities.
2. **Opening vs closing:** Plan calls for line movement (opening → close). Current historical pulls can support this; no script yet that builds opening/closing series and movement features.
3. **Model outputs:** Plan (Phase 5–6) expects:
   - Moneyline fair prices (from win prob).
   - Run line probabilities (±1.5 or book runline).
   - Over/under at various totals.  
   None of this exists until Stan (or your andrew_mack models) + simulation exist.

---

## 3. Team components: fielding, batting, SP, bullpen, fatigue

### 3.1 What the current plan (Mack Ch 18 style) gives you

- **Batting:** Encoded as **attack** (offensive) strength per run-event type (run_1, run_2, run_3, run_4) in the log-linear model. No separate “batting rating” scalar; it’s implicit in attack.
- **Defense (team):** **Defense** ratings in the same model — run prevention. This blends **fielding and staff** (SP + bullpen) at team level.
- **Starting pitching:** Explicit **pitcher** effect in the model (starter ability). So you have “starting pitching” in the sense of the named starter.
- **Bullpen:** Not separated in the codex plan; it’s part of team **defense** and possibly a shared pitcher hierarchy later (“team_pitching_mu”).
- **Fielding:** Not a separate dimension; folded into team **defense**.
- **Fatigue:** Not in the plan (no rest days, pitch counts, or back-to-back usage).

So the plan does **not** “rate each team” as separate scalars for fielding, batting, SP, bullpen, fatigue. It gives you attack/defense/pitcher (and optionally park, conference) that you can use to derive moneyline, runline, and total from simulation.

### 3.2 How to add the components you want

To get **ratings** for fielding, batting, starting pitching, bullpen, and fatigue (and use them for full-game markets), you have two paths:

- **A) Stay within Mack-style model**  
  - Keep attack/defense/pitcher.  
  - **Batting** ≈ attack (or a single “batting” summary from attack run-event params).  
  - **Fielding:** Either (i) add a team fielding factor (e.g. from defensive stats or errors) as an extra term, or (ii) interpret “defense minus pitcher” as defense-from-fielding-and-bullpen and try to split later with other data.  
  - **Bullpen:** Add a separate “bullpen” term (e.g. from relief IP or roles) or a hierarchical pitcher model (starter vs reliever).  
  - **Fatigue:** Add covariates (days rest, recent IP, back-to-back) to the model or as a post-model adjustment in simulation.

- **B) Use your pre-built models in `andrew_mack`**  
  - If those models already output ratings (e.g. fielding, batting, SP, bullpen, fatigue), you can:
    - **Replace** the Stan attack/defense/pitcher with your ratings as inputs to a **simulation** step that computes win prob, runline cover prob, and total runs distribution.
    - Or **combine**: use your ratings as priors or features inside the Stan model (e.g. regression on your batting/SP/bullpen/fielding/fatigue).

Important: The repo does **not** contain an `andrew_mack` folder. To integrate:

1. Decide where the folder lives (e.g. `../andrew_mack` or a path in config).
2. Add a small integration layer (script or package) that:
   - Reads your model outputs (ratings and/or predictions).
   - Maps team/player IDs to this repo’s canonical IDs (using `data/registries/` and the odds name crosswalk).
3. Either feed those outputs into a new/updated simulation script (moneyline, runline, total) or into the planned Stan pipeline as priors/features.

---

## 4. Suggested next steps (focused on full game + components)

1. **Runline and total in the odds pipeline**  
   - Add devig for `spreads` and `totals` (and optionally store consensus runline/total and fair probs) in `pull_odds.py` or a small `process_odds.py` so downstream code can compare model vs market for all three markets.

2. **Locate and document `andrew_mack`**  
   - Confirm path (e.g. `~/andrew_mack` or `ncaa_baseball/andrew_mack`).  
   - List which outputs you have (e.g. team batting, fielding, SP, bullpen, fatigue; or game-level win/total/runline probs).  
   - Add a short `docs/andrew_mack_integration.md` (or section in this audit) describing how those outputs map to teams/games and how they’ll feed simulation or Stan.

3. **Simulation from ratings**  
   - Even before Stan is ported, you can build a **simulation** (Python or R) that:  
     - Takes team (and optionally pitcher) ratings — from andrew_mack and/or from `build_team_game_stats.py` (e.g. RPG for/against) as a simple proxy.  
     - Simulates full-game runs (e.g. simple Poisson or distribution from run events).  
     - Outputs moneyline (win prob), runline cover probs, and total runs distribution.  
   - Then compare these to devigged odds (once runline/total devig is in place).

4. **Explicit component ratings (if not in andrew_mack)**  
   - If you want fielding, batting, SP, bullpen, fatigue **inside this repo**:  
     - Define one place (e.g. `data/processed/team_ratings.csv` or a small parquet schema) for: team_id, season, batting, fielding, sp, bullpen, fatigue (or similar).  
     - Populate from andrew_mack if available, or from future Stan/script outputs.  
     - Use this file in simulation and (optionally) in Stan as priors.

5. **Line movement (opening/closing)**  
   - Add a script that, from historical odds JSONL, builds per-game opening and closing lines (h2h, spreads, totals) and simple movement features (e.g. open→close shift). That supports both backtesting and “sharp move” features mentioned in the plan.

---

## 5. File-level summary

| File / area | Purpose | Full-game (ML/RL/Tot) | Components (F/B/SP/BP/Fat) |
|-------------|---------|------------------------|----------------------------|
| `scripts/pull_odds.py` | Fetch odds, devig h2h, write JSONL | ML ✅; RL/Tot stored, not devigged | — |
| `scripts/scrape_espn.py` | ESPN games → run_events, starters, boxscore | Feeds run-event model | Batting/pitching in boxscore; no ratings |
| `scripts/build_team_game_stats.py` | Scoreboard → team W-L, runs for/against, RPG | Input for simple team strength | Aggregate only; no F/SP/BP/fatigue |
| `data/raw/odds/*.jsonl` | Raw odds per game (h2h, spreads, totals) | All three present | — |
| `ncaa_baseball_codex_prompt.md` | Full plan (Mack, Stan, simulation, backtest) | ML/RL/Tot in Phase 5–6 | Attack/defense/pitcher; no explicit F/BP/fatigue |
| `andrew_mack` (your folder) | Your pre-built models | To be wired for ML/RL/Tot | To be wired for F, B, SP, BP, fatigue |

---

## 6. Bottom line

- **Moneyline:** Data and devig are in place; model-side (win prob → fair price) depends on Stan or andrew_mack simulation.  
- **Runline and total runs:** Data is in the JSONL; add devig + consensus and wire into simulation/backtest.  
- **Fielding, batting, SP, bullpen, fatigue:** The written plan only has attack/defense/pitcher (and optional park). To “rate” those five explicitly, use your andrew_mack outputs and/or extend the plan with a ratings schema and (optionally) Stan terms for bullpen, fielding, and fatigue.  
- **andrew_mack:** Not in the repo; document its path and outputs and add a small integration layer so your pre-built models drive moneyline, runline, and total runs (and component ratings if desired).

If you tell me the exact path to `andrew_mack` and what outputs those models produce (e.g. CSV/parquet of team-game ratings or game-level probs), I can draft the integration steps and the runline/total devig changes in code.
