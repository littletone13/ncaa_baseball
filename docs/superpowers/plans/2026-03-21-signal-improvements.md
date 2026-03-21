# 5 Signal Improvements Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase model signal-to-noise ratio by fixing wRC+ wiring, adding conference hierarchy and FIP priors to Stan, enabling platoon splits, and adding conference-adjusted team strength.

**Architecture:** Five changes ranging from 1-line config fixes to Stan model restructuring. Changes 1 (wRC+) and 4 (platoon) are simulation-layer only — no refit needed. Changes 2 (conference prior), 3 (FIP prior), and 5 (conference team weighting) modify the Stan model and require a full refit. We batch all Stan changes into a single refit.

**Tech Stack:** Python 3.12+, CmdStanPy, pandas, numpy

---

## Chunk 1: Simulation-Layer Fixes (no refit needed)

### Task 1: Fix wRC+ Wiring — Apply to ALL Teams

**Problem:** `resolve_starters.py` line 295 gates wRC+ behind `if tidx == 0:`, but all 308 teams have tidx > 0 after the last refit. Result: wRC+ adjustment is **never applied** to any prediction. The wrc_offense_adj values exist in team_table (mean=0.013, std=0.109, range -0.28 to +0.41) but are thrown away.

**Solution:** Remove the `tidx == 0` gate. Apply wRC+ to ALL teams as a supplemental offense signal. For teams WITH posteriors, scale the adjustment down (the posterior already captures some offense signal). For teams without posteriors, use full weight.

**Files:**
- Modify: `scripts/resolve_starters.py:278-298`
- Modify: `scripts/simulate.py` (no changes needed — already consumes `home_wrc_adj`/`away_wrc_adj`)

- [ ] **Step 1: Edit resolve_starters.py to apply wRC+ to all teams**

In `scripts/resolve_starters.py`, replace the block at lines 278-298:

```python
    wrc_adj_by_team: dict[str, float] = {}  # canonical_id → adj (only for team_idx=0)
    batting_fb_by_team: dict[str, float] = {}  # canonical_id → FB factor (for wind model)
    team_idx_by_cid: dict[str, int] = {}
    if team_table_csv.exists():
        tt = pd.read_csv(team_table_csv, dtype=str)
        tt["team_idx"] = pd.to_numeric(tt["team_idx"], errors="coerce").fillna(0).astype(int)
        tt["wrc_offense_adj"] = pd.to_numeric(tt["wrc_offense_adj"], errors="coerce").fillna(0.0)
        tt["batting_fb_factor"] = pd.to_numeric(tt.get("batting_fb_factor"), errors="coerce").fillna(1.0)
        for _, row in tt.iterrows():
            cid = str(row.get("canonical_id", "")).strip()
            if not cid:
                continue
            tidx = int(row["team_idx"])
            team_idx_by_cid[cid] = tidx
            # Batting FB factor for wind scaling (all teams)
            batting_fb_by_team[cid] = float(row["batting_fb_factor"])
            # Only apply wRC+ adj to teams NOT in the Stan model
            if tidx == 0:
                adj = float(row["wrc_offense_adj"])
                if adj != 0.0:
                    wrc_adj_by_team[cid] = adj
```

Replace with:

```python
    wrc_adj_by_team: dict[str, float] = {}  # canonical_id → wRC+ offense adj (all teams)
    batting_fb_by_team: dict[str, float] = {}  # canonical_id → FB factor (for wind model)
    team_idx_by_cid: dict[str, int] = {}
    # Scaling factor: teams WITH posteriors get partial wRC+ (posterior already
    # captures some offense); teams WITHOUT posteriors get full wRC+ weight.
    WRC_POSTERIOR_SCALE = 0.5   # half-weight for teams already in posterior
    WRC_NO_POSTERIOR_SCALE = 1.0  # full weight for teams with no posterior
    if team_table_csv.exists():
        tt = pd.read_csv(team_table_csv, dtype=str)
        tt["team_idx"] = pd.to_numeric(tt["team_idx"], errors="coerce").fillna(0).astype(int)
        tt["wrc_offense_adj"] = pd.to_numeric(tt["wrc_offense_adj"], errors="coerce").fillna(0.0)
        tt["batting_fb_factor"] = pd.to_numeric(tt.get("batting_fb_factor"), errors="coerce").fillna(1.0)
        for _, row in tt.iterrows():
            cid = str(row.get("canonical_id", "")).strip()
            if not cid:
                continue
            tidx = int(row["team_idx"])
            team_idx_by_cid[cid] = tidx
            # Batting FB factor for wind scaling (all teams)
            batting_fb_by_team[cid] = float(row["batting_fb_factor"])
            # Apply wRC+ adj to ALL teams, scaled by whether they have a posterior
            adj = float(row["wrc_offense_adj"])
            if adj != 0.0:
                scale = WRC_NO_POSTERIOR_SCALE if tidx == 0 else WRC_POSTERIOR_SCALE
                wrc_adj_by_team[cid] = adj * scale
```

- [ ] **Step 2: Update the comment at line 381**

Change line 381 from:
```python
        # wRC+ offense adjustment (team_table, only for team_idx=0)
```
to:
```python
        # wRC+ offense adjustment (team_table, all teams — scaled by posterior presence)
```

- [ ] **Step 3: Verify simulate.py already consumes wRC+ correctly**

No changes needed — simulate.py lines 293-295 already read `home_wrc_adj`/`away_wrc_adj` and apply them to the log-rate at lines 454/458. The wiring was always correct downstream; the gate was in resolve_starters.

---

### Task 2: Enable Platoon Splits

**Problem:** `DEFAULT_LHP_ADJ = 0.0` in platoon_adjustment.py, and `resolve_starters.py` hardcodes `platoon_adj_home: 0.0, platoon_adj_away: 0.0` at lines 428-429 without even calling PlatoonLookup.

**Solution:** Wire PlatoonLookup into resolve_starters.py so it looks up each starter's handedness and computes platoon adjustments. Set a conservative initial LHP coefficient based on known college baseball splits.

**Files:**
- Modify: `scripts/platoon_adjustment.py:51` (change DEFAULT_LHP_ADJ)
- Modify: `scripts/resolve_starters.py:193-430` (wire in PlatoonLookup)

- [ ] **Step 1: Set LHP adjustment coefficient**

In `scripts/platoon_adjustment.py`, change line 51:
```python
DEFAULT_LHP_ADJ = 0.0
```
to:
```python
DEFAULT_LHP_ADJ = 0.03  # ~3% more runs vs LHP (platoon advantage for RHB-heavy lineups)
```

Rationale: MLB platoon splits show ~5-8% offense boost vs opposite-hand pitchers. College lineups are ~70% RHB. A 3% boost vs LHP is conservative and directionally correct (most batters have platoon advantage vs LHP). This is small enough to avoid double-counting with pitcher_ability.

- [ ] **Step 2: Wire PlatoonLookup into resolve_starters.py**

Add import at top of resolve_starters.py (near other imports):
```python
from platoon_adjustment import PlatoonLookup
```

After the StarterLookup instantiation (~line 315), add:
```python
    # ── Platoon lookup ────────────────────────────────────────────────────────
    platoon = PlatoonLookup()
    print(platoon.summary(), file=sys.stderr)
```

- [ ] **Step 3: Replace hardcoded platoon values**

In the game loop, replace lines 428-429:
```python
            "platoon_adj_home": 0.0,
            "platoon_adj_away": 0.0,
```
with:
```python
            "platoon_adj_home": platoon.platoon_adj(
                platoon.get_hand(a_cid, ap_name) if ap_name else None
            ),
            "platoon_adj_away": platoon.platoon_adj(
                platoon.get_hand(h_cid, hp_name) if hp_name else None
            ),
```

Note the inversion: home team's platoon adjustment depends on the AWAY pitcher's hand (home batters face away pitcher), and vice versa.

---

## Chunk 2: Stan Model Changes (require refit)

### Task 3: Add Conference Strength Prior to Stan Model

**Problem:** All 308 teams share a flat `normal(0, sigma_att)` prior. An SEC team and an NEC team start from the same prior despite SEC going 80% and NEC going 10% in cross-conference play. The model has to learn this gap purely from game data, which is thin for small-conference teams.

**Solution:** Add a conference-level random effect. Each team's att/def prior is centered on its conference mean rather than zero. The conference means are themselves hierarchical (centered on zero with learned scale). This is a standard nested random effect.

**Files:**
- Modify: `stan/ncaa_baseball_run_events.stan`
- Modify: `scripts/fit_run_event_model.py` (pass conference index to Stan)
- Modify: `scripts/build_run_event_indices.py` (build conference index)

- [ ] **Step 1: Build conference index in build_run_event_indices.py**

Add after the team index creation (after line ~58 in build_run_event_indices.py):

```python
    # ── Conference index ──────────────────────────────────────────────────────
    canon_path = Path("data/registries/canonical_teams_2026.csv")
    if canon_path.exists():
        canon = pd.read_csv(canon_path)
        team_conf = dict(zip(canon["canonical_id"], canon["conference"]))
    else:
        team_conf = {}

    # Assign conference indices (1-based, alphabetical)
    conferences = sorted(set(team_conf.values()))
    conf_to_idx = {c: i + 1 for i, c in enumerate(conferences)}

    # Map each team in team_index to its conference index
    team_index["conference"] = team_index["canonical_id"].map(team_conf).fillna("Unknown")
    team_index["conf_idx"] = team_index["conference"].map(conf_to_idx).fillna(0).astype(int)

    # Save conference index
    conf_index = pd.DataFrame([
        {"conference": c, "conf_idx": idx} for c, idx in conf_to_idx.items()
    ])
    conf_index.to_csv(out_dir / "run_event_conf_index.csv", index=False)
    print(f"  Conference index: {len(conf_index)} conferences", file=sys.stderr)
```

Also update the team_index CSV save to include the new columns.

- [ ] **Step 2: Pass conference data to Stan in fit_run_event_model.py**

After building team indices (~line 76), load conference mapping:

```python
    # Conference index for hierarchical prior
    conf_idx_col = team_df["conf_idx"] if "conf_idx" in team_df.columns else None
    if conf_idx_col is not None:
        team_conf_idx = dict(zip(team_df["canonical_id"], team_df["conf_idx"].astype(int)))
        N_conf = int(team_df["conf_idx"].max())
    else:
        team_conf_idx = {}
        N_conf = 0
```

Add to the stan_data dictionary:

```python
    stan_data["N_conf"] = N_conf
    stan_data["team_conf_idx"] = [
        team_conf_idx.get(str(fit_df.iloc[i]["home_canonical_id"]).strip(), 1)
        for i in range(len(fit_df))
    ]  # This is wrong — we need per-TEAM conf idx, not per-game
```

Actually, conference index should be a simple array mapping team_idx → conf_idx:

```python
    # Build team_idx → conf_idx array (1-indexed)
    conf_of_team = [0] * (N_teams + 1)  # 0 = placeholder for idx 0
    for _, r in team_df.iterrows():
        tidx = int(r["team_idx"])
        cidx = int(r.get("conf_idx", 1))
        if 1 <= tidx <= N_teams:
            conf_of_team[tidx] = cidx
    # Fill any gaps with 1 (unknown → first conference)
    for i in range(1, N_teams + 1):
        if conf_of_team[i] == 0:
            conf_of_team[i] = 1

    stan_data["N_conf"] = N_conf
    stan_data["team_conf"] = conf_of_team[1:]  # Stan is 1-indexed, length N_teams
```

Update meta.json to include N_conf.

- [ ] **Step 3: Add conference hierarchy to Stan model**

Modify `stan/ncaa_baseball_run_events.stan`:

**Data block — add:**
```stan
  int<lower=1> N_conf;                          // number of conferences
  array[N_teams] int<lower=1, upper=N_conf> team_conf;  // team → conference mapping
```

**Parameters block — add:**
```stan
  // Conference-level random effects (hierarchical mean for teams)
  real<lower=0.01, upper=0.3> sigma_conf_att;
  real<lower=0.01, upper=0.3> sigma_conf_def;
  vector[N_conf] conf_att_raw;                  // conference offense strength
  vector[N_conf] conf_def_raw;                  // conference defense strength
```

**Transformed parameters — add:**
```stan
  // Conference effects (sum-to-zero)
  vector[N_conf] conf_att = conf_att_raw - mean(conf_att_raw);
  vector[N_conf] conf_def = conf_def_raw - mean(conf_def_raw);
```

**Model block — change team priors from:**
```stan
  att_run_1_raw ~ normal(0, sigma_att);
  def_run_1_raw ~ normal(0, sigma_def);
  // ... (all 8 lines)
```
**to:**
```stan
  // Conference hierarchy priors
  sigma_conf_att ~ normal(0.10, 0.05);
  sigma_conf_def ~ normal(0.05, 0.03);
  conf_att_raw ~ normal(0, sigma_conf_att);
  conf_def_raw ~ normal(0, sigma_conf_def);

  // Team priors centered on conference means
  for (t in 1:N_teams) {
    att_run_1_raw[t] ~ normal(conf_att[team_conf[t]], sigma_att);
    def_run_1_raw[t] ~ normal(conf_def[team_conf[t]], sigma_def);
    att_run_2_raw[t] ~ normal(conf_att[team_conf[t]], sigma_att);
    def_run_2_raw[t] ~ normal(conf_def[team_conf[t]], sigma_def);
    att_run_3_raw[t] ~ normal(conf_att[team_conf[t]], sigma_att);
    def_run_3_raw[t] ~ normal(conf_def[team_conf[t]], sigma_def);
    att_run_4_raw[t] ~ normal(conf_att[team_conf[t]], sigma_att);
    def_run_4_raw[t] ~ normal(conf_def[team_conf[t]], sigma_def);
  }
```

This replaces the vectorized `att_run_X_raw ~ normal(0, sigma_att)` with per-team priors centered on conference means. The conference means are themselves shrunk toward zero.

---

### Task 4: Use FIP as Informative Prior for Pitcher Ability

**Problem:** All 4,874 pitchers share a flat `normal(0, sigma_pitcher)` prior. FIP only explains 3.8% of posterior ability, but that's because the prior is uninformative — FIP information is thrown away during fitting. An informative prior would help the 86% of pitchers that are currently at the prior.

**Solution:** Pass pitcher-level FIP z-scores to Stan as prior means. Instead of `pitcher_ability_raw ~ normal(0, sigma_pitcher)`, use `pitcher_ability_raw ~ normal(fip_prior, sigma_pitcher)`. Pitchers without FIP data get 0 (unchanged).

**Files:**
- Modify: `scripts/fit_run_event_model.py` (compute and pass FIP priors)
- Modify: `stan/ncaa_baseball_run_events.stan` (use FIP priors)

- [ ] **Step 1: Compute FIP prior means in fit_run_event_model.py**

After loading pitcher indices, load FIP data and compute z-scores:

```python
    # ── FIP informative priors for pitcher ability ────────────────────────────
    pt_path = Path("data/processed/pitcher_table.csv")
    fip_prior = [0.0] * (N_pitchers + 1)  # idx 0 = unknown (stays 0)
    if pt_path.exists():
        pt = pd.read_csv(pt_path)
        pt["pitcher_idx"] = pd.to_numeric(pt["pitcher_idx"], errors="coerce").fillna(0).astype(int)
        pt["fip"] = pd.to_numeric(pt["fip"], errors="coerce")

        # Compute FIP z-scores (lower FIP = better = negative ability = suppresses runs)
        valid_fip = pt.loc[pt["fip"].notna(), "fip"].values
        if len(valid_fip) > 10:
            fip_mean = float(np.mean(valid_fip))
            fip_std = float(np.std(valid_fip))
            if fip_std > 0.1:
                for _, row in pt.iterrows():
                    pidx = int(row["pitcher_idx"])
                    fip_val = row["fip"]
                    if pidx >= 1 and pidx <= N_pitchers and pd.notna(fip_val):
                        # Positive z = high FIP = bad pitcher = positive ability (allows more runs)
                        # Scale: FIP z-score × pitcher_ability_std estimate
                        z = (float(fip_val) - fip_mean) / fip_std
                        # Clip to ±2 to avoid extreme priors
                        z = float(np.clip(z, -2.0, 2.0))
                        # Scale by estimated posterior std (0.08)
                        fip_prior[pidx] = z * 0.08

        n_with_prior = sum(1 for x in fip_prior if x != 0.0)
        print(f"  FIP priors: {n_with_prior}/{N_pitchers} pitchers have informative priors",
              file=sys.stderr)

    stan_data["fip_prior"] = fip_prior[1:]  # Stan is 1-indexed, length N_pitchers
```

- [ ] **Step 2: Add FIP prior to Stan model**

**Data block — add:**
```stan
  array[N_pitchers] real fip_prior;             // FIP-derived prior mean (0 = no data)
```

**Model block — change pitcher prior from:**
```stan
  pitcher_ability_raw ~ normal(0, sigma_pitcher);
```
**to:**
```stan
  // Pitcher ability — informative FIP prior where available (0 = uninformative)
  for (p in 1:N_pitchers) {
    pitcher_ability_raw[p] ~ normal(fip_prior[p], sigma_pitcher);
  }
```

This means pitchers WITH FIP data start at a non-zero prior mean (pulled toward their FIP-implied ability), while pitchers without FIP data still start at zero. The posterior will update from data either way, but the prior gives a head start.

---

### Task 5: Conference-Adjusted Team Strength Weighting

**Problem:** Cross-conference results are treated identically to within-conference results. An SEC team beating a non-conference cupcake gets the same weight as beating an ACC rival. This dilutes the signal from meaningful games.

**Solution:** This is already mostly handled by Task 3 (conference hierarchy). The hierarchical conference effect will naturally adjust team strengths based on the caliber of opponents. No additional weighting mechanism is needed — the conference random effect IS the conference strength adjustment.

However, we can add one more refinement: pass conference strength as a covariate for teams at idx=0 in the simulation layer (similar to wRC+).

**Files:**
- Modify: `scripts/build_team_table.py` (add conference strength column)
- Modify: `scripts/resolve_starters.py` (pass conference adjustment for idx=0 teams)

- [ ] **Step 1: Compute conference strength in build_team_table.py**

After computing wRC+ adjustments, add conference strength computation:

```python
def compute_conference_strength(
    games_csv: Path,
    team_conf: dict[str, str],
    as_of_season: int,
) -> dict[str, float]:
    """Compute conference strength from cross-conference win rates.

    Returns dict of conference → strength_adj (log-rate scale).
    Positive = strong conference, negative = weak.
    """
    games = pd.read_csv(games_csv, dtype=str)
    games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
    games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")
    games = games.dropna(subset=["home_score", "away_score"])
    games["home_won"] = games["home_score"] > games["away_score"]

    games["home_conf"] = games["home_canonical_id"].map(team_conf)
    games["away_conf"] = games["away_canonical_id"].map(team_conf)

    # Cross-conference only
    cross = games[games["home_conf"] != games["away_conf"]]

    conf_wins: dict[str, list] = {}
    for _, g in cross.iterrows():
        hc, ac = g["home_conf"], g["away_conf"]
        if pd.notna(hc):
            conf_wins.setdefault(hc, []).append(1 if g["home_won"] else 0)
        if pd.notna(ac):
            conf_wins.setdefault(ac, []).append(0 if g["home_won"] else 1)

    # Convert to strength adjustment
    overall_rate = 0.5  # baseline
    ATT_STD = 0.109  # from posterior
    result = {}
    for conf, outcomes in conf_wins.items():
        if len(outcomes) >= 20:
            win_rate = sum(outcomes) / len(outcomes)
            # Map win rate deviation to log-rate scale
            # 80% win rate → strong positive; 10% → strong negative
            z = (win_rate - overall_rate) / 0.15  # rough std of conference win rates
            result[conf] = float(np.clip(z * ATT_STD * 0.5, -0.15, 0.15))

    return result
```

Add `conf_strength_adj` column to team_table output — this is the team's conference strength on log-rate scale.

- [ ] **Step 2: Wire conference strength into resolve_starters.py**

Similar to wRC+ — load from team_table and pass through to simulate.py. For teams without posteriors (idx=0), add conference strength as an additional adjustment.

This is a secondary signal for non-model teams and is lower priority than the Stan model changes.

---

## Chunk 3: Refit and Validate

### Task 6: Refit Stan Model

**Files:**
- Run: `scripts/build_run_event_indices.py`
- Run: `scripts/fit_run_event_model.py`
- Run: `scripts/build_pitcher_table.py`
- Run: `scripts/build_team_table.py`

- [ ] **Step 1: Rebuild indices (includes new conference index)**
```bash
.venv/bin/python3 scripts/build_run_event_indices.py
```

- [ ] **Step 2: Refit Stan model (~15-20 min)**
```bash
.venv/bin/python3 scripts/fit_run_event_model.py
```

- [ ] **Step 3: Rebuild lookup tables**
```bash
.venv/bin/python3 scripts/build_pitcher_table.py
.venv/bin/python3 scripts/build_team_table.py
```

- [ ] **Step 4: Subsample posterior for daily predictions**
```bash
.venv/bin/python3 -c "import pandas as pd; d=pd.read_csv('data/processed/run_event_posterior.csv'); d.sample(2000,random_state=42).to_csv('data/processed/run_event_posterior_2k.csv',index=False)"
```

- [ ] **Step 5: Recalibrate scoring constant**
```bash
.venv/bin/python3 scripts/backtest.py --tune-calibration
```

### Task 7: Validate Improvements

- [ ] **Step 1: Run backtest on historical predictions**
```bash
.venv/bin/python3 scripts/backtest.py --out data/processed/backtest_post_improvements.csv
```

- [ ] **Step 2: Compare before/after metrics**
Key metrics to check:
- Brier score: should improve from 0.244 (closer to/below 0.24)
- LogLoss: should improve from 0.681
- Total MAE: should improve from 5.1
- Total correlation: should improve from 0.21
- Confident picks (>60%): should improve from 62.8%

- [ ] **Step 3: Re-run Saturday March 21 predictions**
```bash
.venv/bin/python3 scripts/predict_day.py --date 2026-03-21 --N 5000 --out data/processed/predictions_2026-03-21.csv
```

Compare model spreads before/after to verify wRC+ and platoon are now active.
