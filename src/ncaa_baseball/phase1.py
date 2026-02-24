"""
Phase 1 model: prior-only win probability and odds comparison.

- Resolve Odds API team names to canonical_teams_2026.
- Provide prior-only win prob (home advantage only until we have game data).
- No BTM/Elo fitting yet; that runs when we have game results.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Default home advantage in log-odds (small; ~52% home when strengths equal)
DEFAULT_HOME_ADVANTAGE_LOGIT = 0.08


def load_canonical_teams(csv_path: Path | str) -> pd.DataFrame:
    """Load canonical_teams_2026.csv."""
    path = Path(csv_path)
    df = pd.read_csv(path)
    for col in ("team_name", "odds_api_name", "canonical_id"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
    if "team_name" in df.columns:
        df["team_name"] = df["team_name"].str.replace("&amp;", "&", regex=False)
    return df


def _normalize_for_match(s: str) -> str:
    """Lowercase, collapse spaces, normalize &amp; to & for matching."""
    s = (s or "").replace("&amp;", "&").replace("&#39;", "'")
    return " ".join(s.lower().split())


def build_odds_name_to_canonical(
    canonical: pd.DataFrame,
) -> dict[str, tuple[str, int]]:
    """
    Build mapping: odds_api_display_name -> (canonical_id, ncaa_teams_id).

    Uses (1) exact odds_api_name when set, (2) team_name prefix match
    (e.g. "Georgia Tech" in "Georgia Tech Yellow Jackets").
    """
    out: dict[str, tuple[str, int]] = {}
    for _, row in canonical.iterrows():
        o = (row.get("odds_api_name") or "").strip()
        if o:
            out[_normalize_for_match(o)] = (
                row["canonical_id"], int(row["ncaa_teams_id"]),
            )
    return out


def resolve_odds_teams(
    home_odds_name: str,
    away_odds_name: str,
    canonical: pd.DataFrame,
    name_to_canonical: dict[str, tuple[str, int]] | None = None,
) -> tuple[tuple[str, int] | None, tuple[str, int] | None]:
    """
    Resolve (home_odds_name, away_odds_name) to (canonical_id, ncaa_teams_id) each.

    Returns (home_tuple, away_tuple) or (None, None) when no match.
    Tries exact odds name first, then "odds name starts with team_name" for each canonical row.
    """
    if name_to_canonical is None:
        name_to_canonical = build_odds_name_to_canonical(canonical)

    def resolve_one(odds_name: str) -> tuple[str, int] | None:
        n = _normalize_for_match(odds_name)
        if n in name_to_canonical:
            return name_to_canonical[n]
        # Try: canonical team_name is prefix of odds name; try longest first to avoid "Florida" matching "Florida St"
        rows_sorted = sorted(
            [
                (len((row.get("team_name") or "").strip()), row)
                for _, row in canonical.iterrows()
                if (row.get("team_name") or "").strip()
            ],
            key=lambda x: -x[0],
        )
        for _, row in rows_sorted:
            name = (row.get("team_name") or "").strip()
            norm_name = _normalize_for_match(name)
            if n.startswith(norm_name + " ") or n == norm_name:
                return (row["canonical_id"], int(row["ncaa_teams_id"]))
            # "Kansas St" vs canonical "Kansas St.": allow when canonical starts with input (trailing punctuation)
            if norm_name.startswith(n) and len(n) >= len(norm_name) - 2:
                return (row["canonical_id"], int(row["ncaa_teams_id"]))
        return None

    home_t = resolve_one(home_odds_name)
    away_t = resolve_one(away_odds_name)
    return (home_t, away_t)


def prior_win_prob(
    home_advantage_logit: float = DEFAULT_HOME_ADVANTAGE_LOGIT,
) -> tuple[float, float]:
    """
    Prior-only win probability (no team strengths yet).
    Returns (win_prob_home, win_prob_away). Sum = 1.
    """
    import math
    p_home = 1.0 / (1.0 + math.exp(-home_advantage_logit))
    p_away = 1.0 - p_home
    return (p_home, p_away)


def load_ratings(csv_path: Path | str) -> dict[str, float]:
    """
    Load phase1_team_ratings.csv (canonical_id, elo_rating).
    Returns dict canonical_id -> elo_rating.
    """
    path = Path(csv_path)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "canonical_id" not in df.columns or "elo_rating" not in df.columns:
        return {}
    return dict(zip(df["canonical_id"].astype(str), df["elo_rating"].astype(float)))


def win_prob_from_elo(
    home_rating: float,
    away_rating: float,
    home_advantage_elo: float = 30.0,
) -> tuple[float, float]:
    """
    Win probability from Elo ratings. home_advantage_elo is added to home for expected score.
    Returns (win_prob_home, win_prob_away).
    """
    exp_home = 1.0 / (1.0 + 10.0 ** ((away_rating - (home_rating + home_advantage_elo)) / 400.0))
    return (exp_home, 1.0 - exp_home)


def compare_to_market(
    model_win_prob_home: float,
    model_win_prob_away: float,
    market_fair_home: float | None,
    market_fair_away: float | None,
) -> dict[str, float | None]:
    """
    Compare model to devigged market. Returns edge and implied diff.
    """
    out: dict[str, float | None] = {
        "model_win_prob_home": model_win_prob_home,
        "model_win_prob_away": model_win_prob_away,
        "market_fair_home": market_fair_home,
        "market_fair_away": market_fair_away,
    }
    if market_fair_home is None or market_fair_away is None:
        out["edge_home"] = None
        out["edge_away"] = None
        return out
    out["edge_home"] = model_win_prob_home - market_fair_home
    out["edge_away"] = model_win_prob_away - market_fair_away
    return out
