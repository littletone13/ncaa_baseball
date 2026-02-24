"""
Pitcher-aware projection: SP ratings, expected innings, bullpen health, and market blend.

Rufus Peabody + Andrew Mack + NoVig: use run-level structure (Mack), respect market when
sample is tiny (Peabody), devig for fair baseline (NoVig). Early season (6–10 games):
heavy shrinkage and higher weight to market.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LEAGUE_RA9_DEFAULT = 5.5
# How much 1 IP of SP above/below league moves win prob (roughly): scale RA9 diff to Elo-like impact
SP_RA9_TO_ELO_SCALE = 15.0  # 1 RA9 better than league ~ +15 Elo equivalent
# Bullpen fatigue: extra runs per IP thrown yesterday (rough)
BP_FATIGUE_PER_IP_LAST_1D = 0.08  # e.g. 3 IP yesterday -> ~0.24 runs penalty


def load_pitcher_ratings(csv_path: Path | str) -> pd.DataFrame:
    """pitcher_ratings.csv: pitcher_espn_id, canonical_id, season, role, ra9, avg_IP_per_app, ..."""
    df = pd.read_csv(Path(csv_path))
    return df


def load_team_pitcher_strength(csv_path: Path | str) -> pd.DataFrame:
    """team_pitcher_strength.csv: canonical_id, season, sp_ra9, rp_ra9, relief_ip_share, league_ra9"""
    return pd.read_csv(Path(csv_path))


def load_bullpen_workload(csv_path: Path | str) -> pd.DataFrame:
    """bullpen_workload.csv: canonical_id, game_date, ip_last_1d, ip_last_3d, pc_last_1d"""
    df = pd.read_csv(Path(csv_path))
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    return df


def get_sp_rating(
    pitcher_espn_id: str | None,
    canonical_id: str,
    season: int,
    ratings: pd.DataFrame,
    team_strength: pd.DataFrame,
    league_ra9: float = LEAGUE_RA9_DEFAULT,
) -> tuple[float, float]:
    """
    Return (ra9, expected_ip) for this starter. If no pitcher or no row, use team SP and 5.0 IP.
    """
    expected_ip_default = 5.0
    if not pitcher_espn_id or not str(pitcher_espn_id).strip():
        row = team_strength[(team_strength["canonical_id"] == canonical_id) & (team_strength["season"] == season)]
        if len(row):
            return (float(row.iloc[0]["sp_ra9"]), expected_ip_default)
        return (league_ra9, expected_ip_default)
    r = ratings[
        (ratings["pitcher_espn_id"].astype(str) == str(pitcher_espn_id))
        & (ratings["canonical_id"] == canonical_id)
        & (ratings["season"] == season)
        & (ratings["role"] == "SP")
    ]
    if len(r):
        return (float(r.iloc[0]["ra9"]), float(r.iloc[0].get("avg_IP_per_app", expected_ip_default)))
    row = team_strength[(team_strength["canonical_id"] == canonical_id) & (team_strength["season"] == season)]
    if len(row):
        return (float(row.iloc[0]["sp_ra9"]), expected_ip_default)
    return (league_ra9, expected_ip_default)


def get_bullpen_workload(
    canonical_id: str,
    game_date: str | pd.Timestamp,
    workload: pd.DataFrame,
) -> tuple[float, float]:
    """Return (ip_last_1d, ip_last_3d) for this team on this date. 0,0 if missing."""
    if workload.empty or "game_date" not in workload.columns:
        return (0.0, 0.0)
    d = pd.Timestamp(game_date).normalize()
    row = workload[(workload["canonical_id"] == canonical_id) & (workload["game_date"] == d)]
    if len(row):
        return (float(row.iloc[0].get("ip_last_1d", 0) or 0), float(row.iloc[0].get("ip_last_3d", 0) or 0))
    return (0.0, 0.0)


def pitcher_adj_to_elo(
    sp_ra9_home: float,
    sp_ra9_away: float,
    expected_ip_home: float,
    expected_ip_away: float,
    ip_last_1d_home: float,
    ip_last_1d_away: float,
    league_ra9: float = LEAGUE_RA9_DEFAULT,
) -> float:
    """
    Additive adjustment to home Elo (positive = better for home). SP impact + bullpen fatigue.
    """
    # SP: better RA9 (lower) is good. (league - sp_ra9) positive = good. Scale by expected IP/9.
    sp_adj = (
        (league_ra9 - sp_ra9_home) * (expected_ip_home / 9.0)
        - (league_ra9 - sp_ra9_away) * (expected_ip_away / 9.0)
    ) * SP_RA9_TO_ELO_SCALE
    # Bullpen fatigue: more IP yesterday = worse
    bp_adj = (ip_last_1d_away - ip_last_1d_home) * BP_FATIGUE_PER_IP_LAST_1D * SP_RA9_TO_ELO_SCALE
    return sp_adj + bp_adj


def win_prob_with_pitchers(
    home_elo: float,
    away_elo: float,
    sp_ra9_home: float,
    sp_ra9_away: float,
    expected_ip_home: float = 5.0,
    expected_ip_away: float = 5.0,
    ip_last_1d_home: float = 0.0,
    ip_last_1d_away: float = 0.0,
    home_advantage_elo: float = 30.0,
    league_ra9: float = LEAGUE_RA9_DEFAULT,
) -> tuple[float, float]:
    """
    Win prob from team Elo + pitcher adjustment (Mack-style SP/RP, Peabody shrinkage already in ra9).
    """
    adj = pitcher_adj_to_elo(
        sp_ra9_home, sp_ra9_away,
        expected_ip_home, expected_ip_away,
        ip_last_1d_home, ip_last_1d_away,
        league_ra9,
    )
    effective_home = home_elo + home_advantage_elo + adj
    exp_home = 1.0 / (1.0 + 10.0 ** ((away_elo - effective_home) / 400.0))
    return (exp_home, 1.0 - exp_home)


def blend_with_market(
    model_win_prob_home: float,
    market_fair_home: float | None,
    n_games_played: int,
    alpha_max: float = 0.6,
    n_games_full_model: int = 25,
) -> tuple[float, float]:
    """
    Peabody-style: blend model with market. When n_games_played is small, weight market more.
    alpha = weight on market; final = (1-alpha)*model + alpha*market.
    """
    if market_fair_home is None or market_fair_home <= 0 or market_fair_home >= 1:
        return (model_win_prob_home, 1.0 - model_win_prob_home)
    alpha = alpha_max * (1.0 - min(n_games_played, n_games_full_model) / n_games_full_model)
    blended = (1.0 - alpha) * model_win_prob_home + alpha * market_fair_home
    blended = max(0.01, min(0.99, blended))
    return (blended, 1.0 - blended)
