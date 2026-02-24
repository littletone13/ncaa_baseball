# Andrew Mack — Statistical Sports Models in Excel (SSME) catalog

Reference inventory of the SSME Excel workbooks used alongside the andrew_mack R scripts. **Do not commit the workbooks or passwords to this repo.** Use env vars (e.g. `SSME1_PASSWORD`, `SSME2_PASSWORD`) if any script needs to open them.

**Base path:** `~/Desktop/andrew_mack` (or `/Users/anthonyeding/Desktop/andrew_mack`)

---

## SSME_1_Protected (extension / Vol 1)

| # | Workbook | Path | Relevance for NCAA baseball |
|---|----------|------|-----------------------------|
| 1 | Bradley Terry Model | `SSME_1_Protected/1. Bradley Terry Model.xlsx` | Pairwise matchup ratings; can inform head-to-head / moneyline. |
| 2 | TOOR Model | `SSME_1_Protected/2. TOOR Model.xlsx` | — |
| 3 | GSSD Model | `SSME_1_Protected/3. GSSD Model.xlsx` | — |
| 4 | ZSD Model | `SSME_1_Protected/4. ZSD Model.xlsx` | — |
| 5 | PRP Model | `SSME_1_Protected/5. PRP Model.xlsx` | — |
| 6 | Tutorial | `SSME_1_Protected/6. Tutorial.xlsx` | General intro. |
| 7 | Backtesting | `SSME_1_Protected/7. Backtesting.xlsx` | **High** — backtest methodology for model vs market. |
| 8 | 3Pt Monte Carlo | `SSME_1_Protected/8. 3Pt Monte Carlo.xlsx` | Monte Carlo simulation patterns. |
| 9 | Competing NegBinomial NFL | `SSME_1_Protected/9. Competing NegBinomial NFL.xlsx` | **High** — same family as Mack Ch 18 / NCAA run-event model (competing NegBin). |
| 10 | Bootstrap | `SSME_1_Protected/10. Bootstrap.xlsx` | Resampling / uncertainty. |
| 11 | Competing Poisson EPL | `SSME_1_Protected/11. Competing Poisson EPL.xlsx` | **High** — competing Poisson; structure similar to run-event (lower run counts = more Poisson-like). |

---

## SSME_2_Protected (predictive / Vol 2)

| # | Workbook | Path | Relevance for NCAA baseball |
|---|----------|------|-----------------------------|
| 1 | Assessing Market Strength | `SSME_2_Protected/1. Assessing Market Strength.xlsx` | **High** — market strength / line assessment. |
| 2 | Elo NBA | `SSME_2_Protected/2. Elo NBA.xlsx` | Elo rating pattern; adaptable to team strength. |
| 3 | Elo NFL | `SSME_2_Protected/3. Elo NFL.xlsx` | Same. |
| 4 | Elo NHL | `SSME_2_Protected/4. Elo NHL.xlsx` | Same. |
| 5 | Elo Soccer | `SSME_2_Protected/5. Elo Soccer.xlsx` | Same. |
| 6 | Predictive Stats | `SSME_2_Protected/6. Predictive Stats.xlsx` | **High** — predictive stats framework. |
| 7 | NBA Win Shares | `SSME_2_Protected/7. NBA Win Shares.xlsx` | Win-share style metrics. |
| 8 | NBA BPM | `SSME_2_Protected/8. NBA BPM.xlsx` | — |
| 9 | NHL GPM | `SSME_2_Protected/9. NHL GPM.xlsx` | — |
| 10 | MLB WRC | `SSME_2_Protected/10. MLB WRC.xlsx` | **High** — MLB weighted runs created; batting/offense proxy for baseball. |
| 11 | NHL Team Totals | `SSME_2_Protected/11. NHL Team Totals.xlsx` | **High** — team totals / over-under structure. |
| 12 | EPL Exact Scores | `SSME_2_Protected/12. EPL Exact Scores.xlsx` | Exact score distribution; useful for runline/total. |
| 13 | QB Passing Yards | `SSME_2_Protected/13. QB Passing Yards.xlsx` | — |
| 14 | NBA Rebounds | `SSME_2_Protected/14. NBA Rebounds.xlsx` | — |
| 15 | MLB Pitcher Strikes | `SSME_2_Protected/15. MLB Pitcher Strikes.xlsx` | **High** — MLB pitcher model; translates to college SP/bullpen. |

---

## Suggested priority for NCAA full-game model

1. **9. Competing NegBinomial NFL** (SSME_1) — same model family as run-event; map to moneyline/runline/total.
2. **7. Backtesting** (SSME_1) — how to compare model vs market and measure ROI.
3. **11. Competing Poisson EPL** (SSME_1) — simpler count model; good for totals.
4. **10. MLB WRC** (SSME_2) — batting/offense side for team strength.
5. **15. MLB Pitcher Strikes** (SSME_2) — pitcher side; SP/bullpen.
6. **11. NHL Team Totals** (SSME_2) — team totals / over-under.
7. **1. Assessing Market Strength** (SSME_2) — market strength and line assessment.
8. **6. Predictive Stats** (SSME_2) — predictive stats framework.

---

## Passwords

- SSME_1 workbooks: use env var (e.g. `SSME1_PASSWORD`); do not store in repo.
- SSME_2 workbooks: use env var (e.g. `SSME2_PASSWORD`); do not store in repo.
