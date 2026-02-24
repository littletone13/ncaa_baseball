# Model build refinements

This doc captures three refinements to the main model plan so the build is more robust: **(1)** phase in a simpler model first, **(2)** plan explicitly for low PBP / score-only games and early-season teams, **(3)** use andrew_mack Excel and R assets as reference and validation, not as the only implementation path.

The full vision remains in `ncaa_baseball_codex_prompt.md`; this is the phased, defensive implementation plan.

---

## 1. Phased model rollout

### Phase 1 — Simple model first (get “model vs market” working)

**Goal:** Produce moneyline (win prob) and optionally a rough total for every matchup we care about, compare to devigged odds, and run a minimal backtest. No run-event model yet.

**Candidate simple models (pick one or combine):**

| Option | Data needed | Outputs | Reference |
|--------|-------------|---------|-----------|
| **Bradley–Terry** | Pairwise results (team A beat team B); can use game-level W/L only | Win probability for A vs B | SSME_1: 1. Bradley Terry Model.xlsx |
| **Elo (or Glicko)** | Game results (winner/loser, optional margin) | Rating per team → win prob from rating diff | SSME_2: Elo NBA/NFL/NHL/Soccer |
| **Team RPG (runs for/against)** | Scoreboard only: runs scored and allowed per team per season (or rolling) | Expected runs per game → simple win prob (e.g. log5 or Pythagorean) and expected total | `build_team_game_stats.py` already gives RPG |

**Phase 1 inputs (all from current repo):**

- `canonical_teams_2026.csv` (and crosswalk so odds team names resolve).
- Game results: scoreboard (contest_id, home/away, runs) and/or ESPN game list with final scores.
- Optional: team-season (or rolling) runs for / runs against from `build_team_game_stats.py` or equivalent.

**Phase 1 outputs:**

- Per matchup: `win_prob_home`, `win_prob_away` (moneyline), optionally `expected_total` (for a simple over/under).
- Comparison to devigged odds (h2h; runline/total once devigged).
- Minimal backtest: e.g. calibration plot, Brier/log loss, or ROI at a fixed edge threshold.

**Success criterion for Phase 1:** We can run “today’s lined games” through the simple model, get probabilities, compare to market, and have at least one historical backtest run (even on a small window). No Stan, no run-event yet.

---

### Phase 2 — Run-event model (Ch 18 style)

**Goal:** Add the full run-event decomposition (run_1/2/3/4, attack/defense/pitcher) and simulation so we have proper runline and total distributions, not just a point estimate.

**Inputs:** Games with run events (from ESPN PBP where available) or from score-only path (see §2). Team and pitcher IDs aligned to canonical registry. Prior season (or rolling) for fitting.

**Outputs:** Win prob, runline cover probs, total runs distribution (and over/under at various lines). Same comparison and backtest as Phase 1, but for all three markets.

**Success criterion:** Stan (or equivalent) fit on historical run-event (or score-only) data; simulation produces moneyline, runline, total; backtest runs for all three markets.

---

### Phase 3 — Line movement and market strength

**Goal:** Use opening/closing line movement and market-strength assessment as inputs or validation, not just closing line comparison.

**Inputs:** Historical odds with multiple snapshots (opening → close), and/or market-strength metrics. Reference: SSME_2 “Assessing Market Strength”, SSME_1 “Backtesting”.

**Outputs:** Line-movement features, optional Bayesian blend (model + market), and backtest that can segment by “sharp move” or market strength.

---

## 2. Low PBP / score-only games and early-season teams

### 2.1 Two data paths

| Path | When | What we have | How we use it |
|------|------|--------------|---------------|
| **Run-event path** | ESPN (or other) PBP available; game has run_1/2/3/4 and starters | Full run-event counts, starting pitchers | Feed into run-event model (Phase 2); same as Mack Ch 18. |
| **Score-only path** | Only final score (e.g. scoreboard, or ESPN without PBP) | `home_runs`, `away_runs`, maybe `home_team`, `away_team` | Use for team strength (RPG, W/L) and for **simple** run distribution (see below). Do **not** fake run-event counts. |

**Rule:** Never invent run_1/2/3/4 from a final score. For score-only games we either (a) use them only for team strength (Elo, Bradley–Terry, RPG), or (b) use a score-only run distribution (e.g. one Poisson per team based on RPG) for simulation only.

### 2.2 Score-only run distribution (for simulation when PBP is missing)

- **Input:** Team A and B seasonal (or rolling) RPG for and RPG against (from scoreboard + `build_team_game_stats.py` or equivalent).
- **Simple approach:** Expected runs for A in this game ≈ function of A’s RPG for and B’s RPG against (e.g. average or Pythagorean-style). Simulate each team’s runs as Poisson (or NegBin with a fixed dispersion) with that lambda; then win prob and total from simulated games.
- **Use case:** Early season, or games with no PBP. Phase 1 can use this for totals/win prob; Phase 2 can use it as a fallback when run-event data are missing for a matchup.

### 2.3 Early-season and low-history teams

- **Shrinkage:** Tighter priors (as in codex prompt) and/or hierarchical team effects (e.g. by conference). New or low-sample teams regress strongly toward conference or league mean.
- **Minimum games:** Below N games (e.g. 5–10), treat team strength as uncertain; use league-average RPG or a prior-only rating for simulation.
- **Explicit “early season” flag:** In backtests and in production, tag games by “days since season start” or “team games played”; report metrics by segment so we can see how the model behaves before enough data accumulates.

### 2.4 Roster turnover: “this season first”

College rosters have **high turnover** (transfers, graduation, new starters). Basing strength only on **prior-season** Elo (or last year’s W/L) is generally **not enough** to beat sides/totals/moneyline, because the team on the field this year is not the same as last year.

**Strategy:**

- **Prioritize this season’s data:** Use 2026 game results (and rolling RPG, SP/RP stats) as the primary signal. Use prior-season Elo (or BTM) as a **prior** or **blend** that shrinks toward league average, not as the main rating.
- **Current-season inputs:**  
  - **ESPN:** 2026 games with boxscore → team runs for/against, SP/RP lines, starter IDs (for stamina).  
  - **D1Baseball:** Per-team lineup and SP usage for the current season (e.g. [Michigan lineup](https://d1baseball.com/team/michigan/lineup/) — games played, batting order by game, SP by game). Use for “who is actually playing” and rotation depth.
- **Implementation:** Fit or update Elo (or RPG-based strength) on **2026 games only** with a strong prior from last year; or use a short rolling window (e.g. last 10 games). Add SP strength, RP strength, and stamina from ESPN pitching lines (and optionally D1Baseball SP usage) so projections reflect current rotation and bullpen, not last year’s roster.

### 2.5 Pitcher-level model (SP ratings, bullpen health, expected innings) — not yet implemented

**Current state:** The Phase 1 model uses **team** Elo only. It does **not** use:
- A **rating per starting pitcher** (no SP Elo, RA9, or FIP by pitcher).
- **Bullpen health** (no rest days, recent IP/pitches, or availability).
- **Expected innings** for today’s starter or relievers from previous lineups/game data.

**Data we have:** `pitching_lines_espn.csv` has per-game, per-pitcher rows (game_date, team, pitcher_espn_id, pitcher_name, starter, IP, ER, R, BB, K, HR, PC). Game JSONL has `starters.home_pitcher` / `away_pitcher` (name, espn_id). So we can derive:
- **SP rating:** Per pitcher (espn_id), from rows with `starter=True`: RA9 or ERA vs league, or a simple pitcher “strength” (e.g. runs allowed per 9). Optionally pitcher-level Elo from game outcomes when that pitcher started.
- **Bullpen health:** Per team (or per reliever): IP or pitches in last 1/3/5 days from `pitching_lines_espn` + game_date. Use as “availability” or “fatigue” (e.g. discount bullpen if heavily used yesterday).
- **Expected innings:** From historical lineups and game data: average IP per start for this season’s starters (by espn_id), and share of team IP from relievers. For a given game, if we know (or assume) today’s starter, use his season-to-date avg IP and the team’s relief share to split expected runs allowed (SP portion vs bullpen portion).

**Implemented:** `scripts/build_pitcher_ratings.py` → `pitcher_ratings.csv`, `team_pitcher_strength.csv`, `bullpen_workload.csv`. `src/ncaa_baseball/pitcher_model.py` → SP rating lookup, expected IP, bullpen workload, pitcher adj to Elo, `blend_with_market()`. `scripts/project_game.py` → `--use-pitchers`, `--game-date`, `--home-sp-id`, `--away-sp-id`, `--market-fair-home`, `--n-games` for Peabody+Mack+NoVig projection.

---

## 2.6 Early season (6–10 games): what to do for better profit

With teams at **6–10 games**, sample size is tiny. Do the following (Rufus Peabody + Andrew Mack + NoVig mindset):

1. **Shrink hard.** We already shrink pitcher RA9 to league (20 IP half-life). Keep team Elo on 2026-only or rolling 10 games and regress new teams strongly to league/prior. Don’t overfit 6-game records.
2. **Blend with the market.** Use `--market-fair-home` (devigged) and `--n-games` in `project_game.py` so that when `n_games` is low we weight the market more (Peabody: closing line value comes from not fighting the market with noisy estimates). Run odds compare with `blend_with_market(..., n_games_played, alpha_max=0.6)` so early season we lean market.
3. **Prioritize spots where we have an edge.** Use **SP ratings** when the announced starter is known (D1Baseball/ESPN): a known good SP vs a TBD or weak SP is a cleaner signal than team Elo on 6 games. Fade totals until 15+ games unless you have a strong SP/bullpen/weather reason.
4. **Devig properly (NoVig).** Use Power or Shin devig for moneylines so “market” in the blend is fair odds, not raw implied prob. Runline/total devig when we use those markets.
5. **Track CLV, not just hit rate.** Peabody: focus on **closing line value** (would the closing line have made our bet +EV?). Early season, aim for spots where our model + SP blend is meaningfully different from the market and the market has moved toward us by close.
6. **Avoid betting totals on tiny samples** unless you have a strong pitcher/park/weather angle; team run environment is unstable at 6–10 games.
7. **Use this season’s data only for SP/RP and workload.** Build pitcher_ratings and bullpen_workload from 2026 games; team Elo can be 2026-only or 2026 with prior from 2025. Don’t weight 2025 team W/L heavily when the roster turned over.

**Summary:** With 6–10 games, lean on **market blend**, **SP/bullpen** when starters are known, **heavy shrinkage**, and **CLV-focused** evaluation. Add runline/total when we have enough data and devig.

---

## 3. Andrew_mack as reference and validation (not the only path)

### 3.1 How we use the R scripts

- **018_chapter_18.R** (and related Ch 18 data) = **design reference** for the run-event model: data wrangling (count_runs, game-level run_1/2/3/4), Stan structure, and simulation flow. We **reimplement** in this repo (Python or R) so we control inputs (canonical teams, NCAA data, our odds). We do **not** run 018 as-is for NCAA production.
- **Other chapters** (e.g. Elo, Bradley–Terry, Poisson) = reference for Phase 1 and for validation logic (e.g. how they compute win prob from ratings).

**Validation:** After we have a Phase 1 or Phase 2 implementation, we can run a **sanity check** on a small shared dataset (e.g. same team strengths and matchup) and compare our win prob or total distribution to Ch 18 or SSME output, where sport/context are comparable.

### 3.2 How we use the Excel workbooks (SSME)

- **Design and methodology:** Use SSME for formulas, backtest layout, and market-strength ideas (see `docs/ANDREW_MACK_SSME_CATALOG.md`). We translate those into our pipeline (scripts in this repo), not run the workbooks in production.
- **Sanity checks:** When we add a component (e.g. Bradley–Terry, Elo, or a simple total), we can manually compare one or two matchups to the corresponding SSME sheet (e.g. Bradley Terry, Backtesting) to ensure we’re in the same ballpark.
- **Backtest comparison:** SSME_1 “7. Backtesting.xlsx” defines a backtest structure; our backtest (Phase 1 and 2) should align conceptually (e.g. same edge thresholds, same notion of “model vs closing line”) so we can compare results or discuss differences.

**We do not:** Depend on the Excel workbooks being open or automated in the repo; we do not commit workbooks or passwords. Implementation lives in this repo; SSME and andrew_mack R are **reference and validation** only.

### 3.3 One-line summary

- **Implement** in the ncaa_baseball repo (canonical data, our odds, our game/score data).
- **Reference** andrew_mack R and SSME for structure, formulas, and backtest design.
- **Validate** by spot checks and backtest structure alignment, not by making andrew_mack the single source of truth for production.

---

## 4. Checklist (refinements)

- [ ] **Phase 1:** Choose simple model (Bradley–Terry, Elo, or RPG-based); implement; wire to canonical teams + odds; run one backtest.
- [ ] **Score-only path:** Document and implement (e.g. RPG → Poisson or NegBin per team) for games without PBP; use in Phase 1 and as fallback in Phase 2.
- [ ] **Early-season:** Define minimum games and shrinkage (or prior) for team strength; add “early season” segment to backtest and reporting.
- [ ] **andrew_mack:** Use 018 and SSME catalog as reference; implement run-event and backtest in repo; add one sanity-check comparison to SSME or Ch 18 where applicable.
