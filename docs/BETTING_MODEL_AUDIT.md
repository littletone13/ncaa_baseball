# NCAA Baseball Betting Model — Skeptical Quant Audit

**Date:** 2026-03-08
**Auditor posture:** Elite quant reviewer, adversarial, looking for reasons this loses money.

---

## 1. Executive Summary

**Is this repo currently good enough to justify real money? No.**

This repository is a well-organized *data collection and plumbing project* that has been mistaken — by its own documentation — for a betting model. There is no betting model here in any meaningful sense. What exists is:

- A team Elo system fit on W/L outcomes (no margin, no run distribution)
- A pitcher RA9 adjustment bolted onto Elo via ad hoc scaling constants
- A multiplicative devig of moneyline odds (not Power, not Shin — just implied/sum)
- A "compare model to market" script that computes edge = model_prob - market_prob
- Zero backtesting
- Zero evaluation of any kind
- Zero stake sizing
- Zero simulation
- No run-distribution model
- No runline or total output
- No Kelly, no bankroll logic, no line shopping, no bet tracking

The repo has good bones for *data infrastructure* (ESPN scraping, canonical team registry, odds fetching, name crosswalk). But the gap between "data pipeline with a toy Elo" and "profitable betting operation" is enormous — roughly 80% of the work remains.

**Bottom line:** Betting real money on this today would be equivalent to betting on coin flips with a 52% prior, paying 4-5% vig, with no idea whether the 52% is even correct because it has never been tested against outcomes. You would almost certainly lose.

---

## 2. Top 10 Highest-Value Issues

### 1. No backtesting exists — period.
The single most damaging gap. There is no script, notebook, or output that evaluates model predictions against actual game outcomes. No Brier score, no log loss, no calibration curve, no ROI simulation, no CLV measurement. Without this, every claim about the model is unfalsifiable. **Severity: fatal.**

### 2. No simulation engine — only point estimates.
The model outputs a single win probability. It cannot produce run distributions, runline probabilities, total probabilities, team totals, or any derivative market price. This means you can only attack the moneyline — the single sharpest, most efficient market in college baseball. You are bringing a knife to a gunfight. **Severity: critical.**

### 3. Elo is fit on W/L only — no margin information used.
`fit_phase1_elo.py` uses binary win/loss outcomes. A team that wins 15-0 gets the same Elo update as a team that wins 2-1 in extras. This throws away the single most informative signal in the data: the margin of victory. MLB Elo models that ignore margin are known to be 2-5% worse in probability accuracy. In college baseball, where talent gaps are wider (SEC vs. SWAC), this is even more damaging. **Severity: high.**

### 4. Devig method is naive multiplicative — not Power or Shin.
`pull_odds.py:devig_h2h()` computes `implied_home / (implied_home + implied_away)`. This is the simplest possible devig (multiplicative/proportional). It systematically misestimates fair probabilities for favorites (overestimates them by 0.5-2%) and underdogs (underestimates). The codex explicitly calls for Power or Shin devig — neither is implemented. When your "edge" is often 2-3%, a 1% devig error can flip the sign of your edge. **Severity: high.**

### 5. Pitcher adjustment uses magic constants with no empirical basis.
`pitcher_model.py` sets `SP_RA9_TO_ELO_SCALE = 15.0` (1 RA9 = 15 Elo points) and `BP_FATIGUE_PER_IP_LAST_1D = 0.08`. These are not fit from data. They are not validated. They are not even sourced from literature. If `SP_RA9_TO_ELO_SCALE` should be 25 or 8 instead of 15, your pitcher adjustment is systematically wrong. The RA9 shrinkage half-life of 20 IP is also unvalidated. **Severity: high.**

### 6. Home advantage is a fixed constant — not estimated or validated.
Home advantage is hardcoded at 30 Elo points (~54% when equal) and 0.08 log-odds (~52%) in different parts of the code. College baseball home advantage varies by conference, venue quality, travel distance, midweek vs. weekend, and competitive level. A blanket 30 Elo points treats a Big 12 Friday night game the same as a Tuesday nonconference game at a neutral site. **Severity: medium-high.**

### 7. No walk-forward or temporal validation — not even a train/test split.
Elo is fit on *all* games in the dataset at once (2024+2025+2026), then used to compare against current odds. There is no temporal separation. The Elo ratings you compare to today's market already saw games that happened after the games you're comparing against. This is textbook lookahead bias for any retrospective analysis. **Severity: high.**

### 8. Consensus fair price is an unweighted average across all books.
`pull_odds.py` averages devigged probabilities across all bookmakers with equal weight. This treats a copy-paste book (BetRivers copying FanDuel) the same as a sharp originator. In college baseball, most US books copy 1-2 market-makers. Your "consensus" is really 5 copies of the same line plus noise from recreational books. This creates a false sense of market agreement and underestimates the true market probability. **Severity: medium.**

### 9. No handling of doubleheaders, 7-inning games, suspended games, or extra innings.
`build_games_from_espn.py` treats every game as identical. College baseball frequently plays doubleheaders (second game is often 7 innings), has mercy-rule truncated games, and has different extra-inning rules than MLB. These affect run distributions, total runs, and game dynamics. A 7-inning game has systematically lower totals — if your model doesn't know the game is 7 innings, it will systematically overshoot totals. **Severity: medium.**

### 10. Run events are only available for ~23% of games — massive selection bias.
The ESPN scraper notes that PBP (and therefore run events) are available for only ~23% of games, "biased toward lined games." This means the run-event data is biased toward nationally televised, high-profile games. If you fit a run-event model on this subset and apply it to unlined or low-profile games, you are extrapolating from a non-representative sample. The teams, parks, and competitive contexts in the 23% are systematically different from the other 77%. **Severity: medium-high (for Phase 2).**

---

## 3. Hidden Assumptions That Could Be False

1. **"Elo ratings from current-season W/L data provide meaningful differentiation between teams."** With 6-15 games, Elo is still dominated by the initialization prior (1500 for all). Two teams that are both 8-2 but against vastly different schedules get similar Elo. SOS adjustment is absent.

2. **"RA9 is a good proxy for pitcher quality in college baseball."** College baseball has extreme sampling noise in RA9 (a starter might have 15-30 IP in a season). RA9 conflates pitcher quality with defensive quality, bullpen support, opponent quality, and park effects. Without opponent adjustment, a pitcher in the Missouri Valley is not comparable to one in the SEC.

3. **"Blending model with market (Peabody-style) captures remaining information."** The blend formula `alpha = alpha_max * (1 - n_games/25)` is a linear ramp with no empirical tuning. If the model is garbage (which it may be), blending 60% garbage with 40% market still produces garbage. The blend only works if the model adds *independent* information beyond what the market already knows.

4. **"The Odds API consensus line is a fair market."** College baseball markets are thin, wide, and soft. Hold is 8-15%. Lines are often posted by one market-maker and copied. "Consensus" of 5 identical lines is still one opinion. Treating this as a calibrated market signal is generous.

5. **"Bullpen workload (IP last 1 day / 3 days) captures fatigue."** Pitch count matters more than IP. A reliever who threw 40 pitches in 1.2 IP is more fatigued than one who threw 10 pitches in 1.0 IP. IP is a crude proxy. Additionally, college coaches manage bullpens very differently from MLB — midweek bullpen games are common and don't follow standard rest patterns.

6. **"All teams start at Elo 1500."** No carryover from prior seasons. A team that was 50-10 last year and returns 80% of production starts the same as a team that was 10-50. This is incorrect for the first 10+ games of every season.

7. **"Home/away designation in ESPN data is always correct."** Neutral-site tournaments, which comprise a meaningful fraction of college baseball games (especially early season), may be coded as home/away by ESPN. The model gives 30 Elo points of home advantage to a "home" team at a neutral site if `neutral_site` isn't properly checked.

---

## 4. Likely Sources of Real Edge (If Developed)

1. **Starting pitcher identification before the market prices it.** College baseball starter announcements are inconsistent. If you can identify the starter (via D1Baseball, team Twitter, pattern recognition from rotation) before the book adjusts, this is a genuine informational edge. The repo has pitcher ratings infrastructure but no mechanism to *obtain starter information early*.

2. **Bullpen exhaustion in weekend series.** Friday-Saturday-Sunday series are the backbone of college baseball. By Sunday, the bullpen of a team that went deep into both previous games is measurably depleted. Books often don't adjust Sunday lines enough for this. The repo tracks `ip_last_3d` which is directionally right but not sharp enough.

3. **Conference strength differentials in nonconference play.** Early-season nonconference games between mismatched conferences (SEC vs. SWAC, ACC vs. Patriot) are where Elo-style models should have the clearest advantage — the market may not fully price the talent gap. But the model needs to actually capture this gap, which requires SOS or conference-level adjustments.

4. **Weaker derivative markets.** College baseball team totals, F5 (first 5 innings), and alt lines are posted at wider hold by fewer books. These are less efficiently priced than the moneyline. If the repo had a simulation engine, these would be the first markets to attack.

5. **Weather and park effects at extreme venues.** Games at high altitude (Air Force, BYU), extreme heat (Arizona, Texas in May-June), or strong wind (coastal stadiums) systematically shift run totals. Books often use static park factors. A model that updates for game-day weather conditions has a legitimate edge on totals.

---

## 5. Likely Illusions of Edge

1. **"Edge" from comparing a toy Elo to a devigged market.** The "edge_home" and "edge_away" fields in the phase1_compare output are the difference between a barely-informed Elo probability and a naively-devigged market probability. This is noise, not signal. You could get similar "edges" from a random number generator.

2. **Pitcher RA9 adjustment appearing to improve predictions.** Since the RA9 scaling constants are not validated, any apparent improvement from adding pitchers is just as likely to be degrading predictions as improving them. Without a held-out evaluation, you cannot distinguish signal from noise.

3. **High "edge" on heavy favorites or underdogs.** Multiplicative devig systematically overestimates favorite probabilities. If your Elo agrees (slightly less) with the market on an 85% favorite, you'll show a negative edge on the favorite and positive edge on the underdog — not because you found value, but because your devig is wrong.

4. **"Market blend" appearing to calibrate the model.** Blending with the market mechanically pulls your probabilities toward the market. This makes your probabilities *look* better calibrated (closer to the true probability) but removes any independent edge. It's camouflage, not improvement.

5. **Early-season shrinkage appearing conservative.** Using league-average RA9 (5.5) with heavy shrinkage makes your early-season pitcher ratings cluster near the mean. This *feels* conservative but actually means you're not providing any pitcher-level information — you're just outputting a prior.

---

## 6. Missing Data/Features Ranked by Expected Value

| Rank | Feature | Expected Value | Difficulty |
|------|---------|---------------|------------|
| 1 | **Margin-of-victory Elo (or Pythagorean/log5)** | High — captures talent gap, not just W/L | Low |
| 2 | **Strength of schedule adjustment** | High — critical for cross-conference comparison | Medium |
| 3 | **Prior-season Elo carryover (with regression)** | High — fixes early-season cold start | Low |
| 4 | **Opponent-adjusted pitcher ratings** | High — current RA9 is unadjusted | Medium |
| 5 | **Park factors (per venue)** | Medium-high — aluminum bats amplify park effects | Medium |
| 6 | **Game-day weather (temp, wind, humidity)** | Medium-high — large effect on totals | Medium |
| 7 | **Lineup strength / batting order** | Medium — who's actually playing today | High |
| 8 | **Travel/rest schedule** | Medium — Friday start vs. Sunday bullpen game | Medium |
| 9 | **Returning production / roster continuity** | Medium — preseason prior quality | High |
| 10 | **Conference-level hierarchical shrinkage** | Medium — better small-sample estimation | Medium |
| 11 | **Pitcher handedness + platoon splits** | Low-medium — less data in college | High |
| 12 | **Defensive metrics (errors, fielding %)** | Low-medium — noisy in college | Medium |
| 13 | **Catcher framing / baserunning** | Low — minimal data availability | Very high |
| 14 | **Umpire effects** | Negligible — not tracked in college | N/A |

---

## 7. Missing Market Opportunities Ranked by Expected Value

| Rank | Market | Why It's Missing | Edge Potential |
|------|--------|-----------------|----------------|
| 1 | **First 5 innings (F5)** | No simulation, no inning-level model | High — isolates SP quality, removes bullpen noise |
| 2 | **Team totals** | No run distribution model | High — can disagree with game total via team-level strength |
| 3 | **Full-game totals** | No run distribution, no devig for totals | High — weather/park/SP-driven, softer market |
| 4 | **Run line (±1.5)** | No margin distribution | Medium-high — requires run distribution, which you don't have |
| 5 | **Alt run lines** | No tail distribution | Medium — mispriced tails in thin markets |
| 6 | **Live/in-game** | No real-time pipeline | Medium-high — pitching changes, weather, bullpen exhaustion |
| 7 | **Series pricing** | No multi-game correlation model | Medium — weekend series have correlated bullpen states |
| 8 | **Same-game parlays** | No correlation model | Low-medium — requires correlated ML+total model |

The repo currently attacks *only* the full-game moneyline — the sharpest, most efficiently priced market in college baseball. Every other market listed above is softer and more likely to contain exploitable inefficiency.

---

## 8. Backtest Flaws and Realism Issues

There is no backtest. This section audits what *would* be wrong if the current code were used for backtesting:

1. **Lookahead bias in Elo.** `fit_phase1_elo.py` fits Elo on all games at once, then the ratings are used for comparison. Any retrospective analysis using these ratings sees the future.

2. **No closing-line distinction.** The odds pipeline fetches snapshots at arbitrary times. There is no concept of "closing line" vs "opening line." Any comparison to "market" is comparison to whatever snapshot happened to be fetched, which may be hours before game time.

3. **Best-of-market pricing illusion.** If you later compare your model to the *best available* line across all books, you'll overestimate achievable edge. Real betting requires getting down at a specific book at a specific time.

4. **No vig in P&L simulation.** The edge calculation subtracts devigged market from model. But you pay vig when you bet. A 2% edge on a -110 line is actually ~-2.5% after vig. No analysis in the repo accounts for this.

5. **No limit/access assumptions.** College baseball is a limited market. Even if you find edge, you may be limited to $50-200 per game at many books. The repo has no mechanism to estimate achievable volume.

6. **No line movement or timing analysis.** The repo cannot distinguish between "edge that exists at open and closes by game time" (CLV leakage) and "edge that persists to close" (genuine mispricing). These are fundamentally different.

7. **No out-of-sample stability analysis.** Without testing season-by-season, you can't know if the model's edge (if any) is stable or is an artifact of one unusual season.

---

## 9. Concrete Upgrades

### Quick Wins (1-2 days each)

1. **Implement Power devig.** Replace multiplicative devig with Power method. This is ~30 lines of code and immediately improves fair price estimation by 0.5-1.5% on favorites.

2. **Add margin-of-victory to Elo.** Modify `fit_phase1_elo.py` to use `mov_factor = log(1 + abs(margin)) * elo_diff_multiplier` in the K-factor. Standard in sports Elo. ~20 lines of code.

3. **Add prior-season Elo carryover.** At season start, carry over last season's Elo with regression toward 1500 (e.g., `new_elo = 0.6 * old_elo + 0.4 * 1500`). Fixes cold-start problem. ~10 lines.

4. **Write a minimal backtest.** For each game in 2024-2025, compute what the model would have predicted using only prior data, compare to outcome. Compute Brier score, log loss, calibration plot, and ROI at 2%/3%/5% edge thresholds. This is the single most important thing you can do.

5. **Add neutral-site handling.** Check `neutral_site` flag in ESPN data and set home advantage to 0 for neutral games. Currently the flag exists in the data but is not used by the Elo fitter or projection.

### Medium-Term (1-2 weeks each)

6. **Build a Poisson simulation engine.** Use team-level expected runs (from adjusted RPG or the run-event model) to simulate game outcomes via Poisson draws. This unlocks runline, total, team total, and F5 markets. Even a naive Poisson model is vastly better than point-estimate Elo for derivative markets.

7. **Implement walk-forward validation.** Refit Elo weekly (not on the whole dataset), predict the next week's games, and accumulate out-of-sample predictions. This gives you an honest assessment of model quality.

8. **Add strength-of-schedule adjustment.** Compute SOS from opponent Elo. Adjust team ratings for schedule difficulty. Critical for cross-conference comparison.

9. **Build closing-line extraction.** From historical odds snapshots, identify the last pre-game snapshot as "closing." Compute CLV (closing line value) for would-be bets. CLV is the gold standard for evaluating betting models.

10. **Opponent-adjust pitcher RA9.** Regress pitcher RA9 against opponent batting strength. A 3.00 RA9 against weak-hitting nonconference teams is very different from 3.00 against SEC lineups.

### Major Architecture Upgrades (weeks-months)

11. **Implement the Mack Ch.18 run-event model.** This is the codex's core vision. Stan or PyMC fit of log-linear NegBin/Poisson on run events, with team attack/defense/pitcher effects. This produces coherent joint distributions over all markets from a single generative model.

12. **Build a full simulation engine on the posterior.** 10K Monte Carlo draws per matchup, producing win prob, run line cover, total distribution, team totals, F5 lines — all from one coherent model.

13. **Implement a proper betting decision layer.** Kelly criterion (fractional), correlation control for same-day bets, exposure limits, line-shopping across books, market-selection filters (only bet when edge > threshold AND market is soft enough).

14. **Build a daily pipeline.** Automated: fetch today's games → update model → compare to current odds → output actionable bets with sizing → track results. Currently all manual.

---

## 10. Final Verdict

### Strongest Parts

- **Data pipeline architecture.** The ESPN scraper, canonical team registry, name crosswalk, and odds fetcher are well-built. The team registry approach (single canonical ID, manual crosswalk, no fuzzy matching) is correct and disciplined.
- **Documentation and self-awareness.** The docs (`MODEL_PURPOSE_AND_PATH.md`, `MODEL_BUILD_REFINEMENTS.md`) honestly identify the gap between aspiration and reality. The codex prompt is a clear, well-structured blueprint.
- **Run-event extraction.** The PBP-to-run-event parsing in `scrape_espn.py` is thoughtfully done (atBatId grouping, Play Result filtering, high-water-mark delta tracking). This is the right data structure for the Mack model.
- **Pitcher infrastructure.** The pitcher ratings, team SP/RP strength, and bullpen workload CSVs are the right *kind* of data products, even if the scaling/shrinkage parameters are unvalidated.

### Weakest Parts

- **No evaluation of any kind.** This cannot be overstated. A model that has never been tested against outcomes is not a model — it is a hypothesis.
- **No run distribution / simulation.** Without this, you cannot price any market other than the moneyline, and your moneyline pricing is a point estimate with no uncertainty.
- **Ad hoc scaling constants.** The SP_RA9_TO_ELO_SCALE, BP_FATIGUE_PER_IP_LAST_1D, shrink_ip, alpha_max, home_advantage constants are all guesses. They need to be estimated from data.
- **Devig quality.** Multiplicative devig is the worst standard method. On college baseball lines (wide hold, lopsided prices), this introduces systematic error.
- **Single-market focus.** Attacking the moneyline only, in a market where sharps and market-makers are already reasonably efficient, is the lowest-probability path to profitability.

### Exact Path from This Repo to a Serious College Baseball Betting Operation

**Phase 0: Prove the current model is not worse than the market (1-2 weeks)**
1. Implement walk-forward Elo with margin-of-victory and prior-season carryover.
2. Implement Power devig.
3. Build a backtest: 2024-2025 seasons, model vs. closing moneyline, Brier score and ROI.
4. If model Brier > closing-line Brier → model is worse than the market → stop betting.
5. If model Brier < closing-line Brier → there may be signal. Measure CLV.

**Phase 1: Build a coherent pricing engine (2-4 weeks)**
1. Fit Poisson (or NegBin) model for team-level expected runs, with SP/RP adjustment.
2. Build Monte Carlo simulation: 10K draws → win prob, run line, total, team total, F5.
3. Validate against 2024-2025 outcomes across all markets.
4. Identify which markets (ML, RL, total, F5) show positive CLV after vig.

**Phase 2: Build the Mack model (4-8 weeks)**
1. Implement Stan/PyMC run-event model with walk-forward fitting.
2. Add park factors, weather, and conference hierarchy.
3. Full simulation engine with all derivative markets.
4. Backtest across 2023-2025 with realistic bet timing and vig.

**Phase 3: Build the betting layer (2-4 weeks)**
1. Kelly/fractional Kelly sizing with correlation control.
2. Line-shopping across 3-5 books.
3. Market-selection filter: only bet where model edge > threshold AND market is soft.
4. Daily automated pipeline: model update → odds comparison → bet output → P&L tracking.
5. Live-betting triggers for pitching changes and weather shifts.

**Phase 4: Scale carefully (ongoing)**
1. Track CLV obsessively — it's the leading indicator of edge.
2. Monitor for account restrictions and adjust book allocation.
3. Expand to derivative markets only when CLV is positive on the base model.
4. Consider expanding to NAIA/D2/D3 if college baseball limits are too tight.

**Honest assessment of timeline:** 3-6 months of focused development to reach Phase 2 with a validated, backtested model that can identify positive-CLV bets in college baseball. The data infrastructure is 60% built. The modeling and evaluation are 5% built. The betting layer is 0% built.

---

*This audit was conducted by examining every Python/R source file, all documentation, the codex prompt, data registries, and the odds pipeline. No external data was consulted beyond what exists in the repository.*
