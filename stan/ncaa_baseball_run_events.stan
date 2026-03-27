// NCAA D1 run-event model — Mack Ch 18 architecture + park factors + bullpen.
// Log-linear rates for run_1..run_4; attack, defense, pitcher, home advantage,
// park effect, and bullpen quality adjustment.
// NegBin for run_1/run_2 (overdispersed), Poisson for run_3/run_4 (rare events).
// Sum-to-zero centering for identifiability.
//
// College adaptations:
//   - pitcher_idx=0 means unknown starter (adds 0 to log-rate)
//   - park_factor is pre-computed (log-scale, 0 = neutral); pass 0 if unknown
//   - bullpen_adj is pre-computed fatigue/quality composite; pass 0 if unknown
//   - Intercept priors calibrated to college run-event base rates (higher than MLB)
//
// v2 (2026-03-21): Conference hierarchy for team priors + FIP informative pitcher priors.
//   - Teams are nested within conferences (30 D1 conferences)
//   - Conference means are hierarchical (centered on zero with learned scale)
//   - Pitcher priors are centered on FIP z-score × ability_std (0 if no FIP data)

data {
  int<lower=1> N_games;
  int<lower=1> N_teams;
  int<lower=1> N_pitchers;               // max pitcher index (1..N_pitchers); 0 = unknown
  int<lower=1> N_conf;                    // number of conferences
  array[N_teams] int<lower=1, upper=N_conf> team_conf;  // team → conference mapping
  array[N_games] int<lower=1, upper=N_teams> home_team_idx;
  array[N_games] int<lower=1, upper=N_teams> away_team_idx;
  array[N_games] int<lower=0, upper=N_pitchers> home_pitcher_idx;  // 0 = unknown
  array[N_games] int<lower=0, upper=N_pitchers> away_pitcher_idx;
  array[N_games] int<lower=0> home_run_1;
  array[N_games] int<lower=0> home_run_2;
  array[N_games] int<lower=0> home_run_3;
  array[N_games] int<lower=0> home_run_4;
  array[N_games] int<lower=0> away_run_1;
  array[N_games] int<lower=0> away_run_2;
  array[N_games] int<lower=0> away_run_3;
  array[N_games] int<lower=0> away_run_4;
  // Park factor: log(adjusted_pf) for the game venue (0 = neutral/unknown)
  array[N_games] real park_factor;
  // Bullpen quality adjustment (pre-computed, log-scale effect on opponent runs)
  // Positive = worse bullpen (opponent scores more), Negative = better bullpen
  array[N_games] real home_bullpen_adj;
  array[N_games] real away_bullpen_adj;
  // FIP-derived prior mean for pitcher ability (0 = no FIP data / uninformative)
  array[N_pitchers] real fip_prior;
}

parameters {
  // Dispersion (NegBin only for run_1 and run_2)
  real<lower=0.001> theta_run_1;
  real<lower=0.001> theta_run_2;

  // Home advantage — unconstrained (allow data to determine sign)
  real home_advantage;

  // Intercepts on log scale — unconstrained (rate = exp(int), can be <1 or >1)
  // College baseball run_1 is common (~3.3/game), run_2..4 are rare (<1/game)
  real int_run_1;
  real int_run_2;
  real int_run_3;
  real int_run_4;

  // Hierarchical scales (learned from data)
  real<lower=0.01, upper=0.6> sigma_att;
  real<lower=0.01, upper=0.6> sigma_def;
  real<lower=0.01, upper=0.4> sigma_pitcher;

  // Conference-level random effects
  real<lower=0.01, upper=0.3> sigma_conf_att;
  real<lower=0.01, upper=0.3> sigma_conf_def;
  vector[N_conf] conf_att_raw;
  vector[N_conf] conf_def_raw;

  // Raw team abilities (per-run-type att/def)
  vector[N_teams] att_run_1_raw;
  vector[N_teams] def_run_1_raw;
  vector[N_teams] att_run_2_raw;
  vector[N_teams] def_run_2_raw;
  vector[N_teams] att_run_3_raw;
  vector[N_teams] def_run_3_raw;
  vector[N_teams] att_run_4_raw;
  vector[N_teams] def_run_4_raw;

  // Raw pitcher ability — single scalar per pitcher (Mack Ch 18)
  vector[N_pitchers] pitcher_ability_raw;

  // Learned coefficients for park and bullpen effects
  real beta_park;                          // effect of log(park_factor) on log-rate
  real beta_bullpen;                       // effect of bullpen_adj on log-rate
}

transformed parameters {
  // Conference effects (sum-to-zero)
  vector[N_conf] conf_att = conf_att_raw - mean(conf_att_raw);
  vector[N_conf] conf_def = conf_def_raw - mean(conf_def_raw);

  // Sum-to-zero centering for identifiability
  vector[N_teams] att_run_1 = att_run_1_raw - mean(att_run_1_raw);
  vector[N_teams] def_run_1 = def_run_1_raw - mean(def_run_1_raw);
  vector[N_teams] att_run_2 = att_run_2_raw - mean(att_run_2_raw);
  vector[N_teams] def_run_2 = def_run_2_raw - mean(def_run_2_raw);
  vector[N_teams] att_run_3 = att_run_3_raw - mean(att_run_3_raw);
  vector[N_teams] def_run_3 = def_run_3_raw - mean(def_run_3_raw);
  vector[N_teams] att_run_4 = att_run_4_raw - mean(att_run_4_raw);
  vector[N_teams] def_run_4 = def_run_4_raw - mean(def_run_4_raw);
  vector[N_pitchers] pitcher_ability = pitcher_ability_raw - mean(pitcher_ability_raw);
}

model {
  // Priors — global parameters
  // Informative HFA prior: ~53-54% home win rate → ~0.05 on log-rate scale
  home_advantage ~ normal(0.05, 0.03);
  int_run_1 ~ normal(1.2, 0.3);          // ~3.3 run_1 events/game -> log(3.3) ~ 1.19
  int_run_2 ~ normal(-0.1, 0.3);         // ~0.88 run_2 events/game -> log(0.88) ~ -0.13
  int_run_3 ~ normal(-1.3, 0.5);         // ~0.26 run_3 events/game -> log(0.26) ~ -1.35
  int_run_4 ~ normal(-2.1, 0.5);         // ~0.12 run_4 events/game -> log(0.12) ~ -2.12
  theta_run_1 ~ gamma(30, 1);
  theta_run_2 ~ gamma(30, 1);

  // Priors — hierarchical scales (learned from data)
  sigma_att ~ normal(0.15, 0.05);
  sigma_def ~ normal(0.15, 0.05);
  sigma_pitcher ~ normal(0.10, 0.03);

  // Conference hierarchy priors
  sigma_conf_att ~ normal(0.10, 0.05);
  sigma_conf_def ~ normal(0.05, 0.03);
  conf_att_raw ~ normal(0, sigma_conf_att);
  conf_def_raw ~ normal(0, sigma_conf_def);

  // Team priors — centered on conference means (nested hierarchy)
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

  // Pitcher ability — informative FIP prior where available (0 = uninformative)
  for (p in 1:N_pitchers) {
    pitcher_ability_raw[p] ~ normal(fip_prior[p], sigma_pitcher);
  }

  // Priors — park and bullpen coefficients
  beta_park ~ normal(1, 0.3);             // ~1 means park_factor translates directly to log-rate
  beta_bullpen ~ normal(0, 0.2);          // effect of bullpen quality on run rates

  // Likelihood
  for (n in 1:N_games) {
    // Pitcher effects (0 = unknown starter -> contributes 0)
    real p_away = (away_pitcher_idx[n] >= 1) ? pitcher_ability[away_pitcher_idx[n]] : 0;
    real p_home = (home_pitcher_idx[n] >= 1) ? pitcher_ability[home_pitcher_idx[n]] : 0;

    // Park and bullpen effects (0 when data unavailable — graceful degradation)
    real park = beta_park * park_factor[n];
    // Bullpen adj: home team's bullpen affects AWAY team's scoring and vice versa
    real bp_h = beta_bullpen * away_bullpen_adj[n];  // away bullpen quality -> affects home scoring
    real bp_a = beta_bullpen * home_bullpen_adj[n];  // home bullpen quality -> affects away scoring

    // Run 1 — NegBin (overdispersed singles are common)
    home_run_1[n] ~ neg_binomial_2_log(
      int_run_1 + att_run_1[home_team_idx[n]] + def_run_1[away_team_idx[n]] + home_advantage + p_away + park + bp_h,
      theta_run_1);
    away_run_1[n] ~ neg_binomial_2_log(
      int_run_1 + att_run_1[away_team_idx[n]] + def_run_1[home_team_idx[n]] + p_home + park + bp_a,
      theta_run_1);

    // Run 2 — NegBin
    home_run_2[n] ~ neg_binomial_2_log(
      int_run_2 + att_run_2[home_team_idx[n]] + def_run_2[away_team_idx[n]] + home_advantage + p_away + park + bp_h,
      theta_run_2);
    away_run_2[n] ~ neg_binomial_2_log(
      int_run_2 + att_run_2[away_team_idx[n]] + def_run_2[home_team_idx[n]] + p_home + park + bp_a,
      theta_run_2);

    // Run 3 — Poisson (rare events, no overdispersion needed)
    home_run_3[n] ~ poisson_log(
      int_run_3 + att_run_3[home_team_idx[n]] + def_run_3[away_team_idx[n]] + home_advantage + p_away + park + bp_h);
    away_run_3[n] ~ poisson_log(
      int_run_3 + att_run_3[away_team_idx[n]] + def_run_3[home_team_idx[n]] + p_home + park + bp_a);

    // Run 4 — Poisson (very rare 4+ run events)
    home_run_4[n] ~ poisson_log(
      int_run_4 + att_run_4[home_team_idx[n]] + def_run_4[away_team_idx[n]] + home_advantage + p_away + park + bp_h);
    away_run_4[n] ~ poisson_log(
      int_run_4 + att_run_4[away_team_idx[n]] + def_run_4[home_team_idx[n]] + p_home + park + bp_a);
  }
}
