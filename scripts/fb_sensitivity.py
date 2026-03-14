"""
Fly-ball sensitivity for weather adjustments.

Loads D1Baseball pitching batted-ball data and computes FB% at multiple levels:
  - Individual pitcher FB% (for starters)
  - Team average FB% (for bullpen — represents the staff the starter hands off to)
  - Conference average FB%
  - D1 overall average FB%

Sensitivity scalar: 0.3 + 0.7 × (FB% / league_avg_FB%)
  - Ground-ball pitchers (low FB%): sensitivity < 1.0 → wind matters less
  - Fly-ball pitchers (high FB%): sensitivity > 1.0 → wind matters more
  - Floor of 0.3 ensures even extreme GB pitchers still have some wind exposure

Architecture:
  Wind affects fly balls from the pitcher on the mound. So:
  - Starter's individual FB% → wind sensitivity for starter innings
  - Team avg FB% → wind sensitivity for bullpen innings (bullpen = staff average)
  - These are returned separately so predict_day.py can apply per-pitcher adjustments

Usage:
    from fb_sensitivity import FBSensitivityLookup
    fb = FBSensitivityLookup()

    # Starter-specific sensitivity
    hp_sens = fb.pitcher_sensitivity("BSB_TEXAS_TECH", "Brandon Beckel")

    # Bullpen (team average) sensitivity
    hp_bp_sens = fb.bullpen_sensitivity("BSB_TEXAS_TECH")

    # Raw FB%
    fb_pct = fb.pitcher_fb_pct("BSB_TEXAS_TECH", "Brandon Beckel")

    # Conference averages
    conf_avg = fb.conference_avg_fb  # dict: conference_name → avg FB%
"""
from __future__ import annotations

import csv
from pathlib import Path


# Floor: even extreme ground-ball pitchers get 30% of the wind effect
FLOOR = 0.3
# Scale: remaining 70% is proportional to FB% relative to league average
SCALE = 0.7


def _parse_pct(s: str) -> float | None:
    """Parse '45.2%' → 0.452, return None if invalid."""
    s = s.strip().rstrip("%")
    try:
        return float(s) / 100.0
    except (ValueError, TypeError):
        return None


def _normalize(s: str) -> str:
    """Normalize curly/smart apostrophes to straight."""
    return s.replace("\u2019", "'").replace("\u2018", "'")


class FBSensitivityLookup:
    """Look up fly-ball sensitivity for pitchers and teams."""

    def __init__(
        self,
        batted_ball_tsv: str | Path = "data/raw/d1baseball/pitching_batted_ball.tsv",
        crosswalk_csv: str | Path = "data/registries/d1baseball_crosswalk.csv",
        canonical_csv: str | Path = "data/registries/canonical_teams_2026.csv",
    ):
        self.batted_ball_tsv = Path(batted_ball_tsv)
        self.crosswalk_csv = Path(crosswalk_csv)
        self.canonical_csv = Path(canonical_csv)

        # d1baseball_name → canonical_id
        self._team_crosswalk: dict[str, str] = {}
        # canonical_id → conference name
        self._team_conference: dict[str, str] = {}
        # canonical_id → list of (player_name, fb_pct)
        self._team_pitchers: dict[str, list[tuple[str, float]]] = {}
        # canonical_id → team average FB%
        self._team_avg_fb: dict[str, float] = {}
        # conference → average FB%
        self.conference_avg_fb: dict[str, float] = {}
        # D1 overall average FB%
        self.league_avg_fb: float = 0.0

        self._load()

    def _load(self):
        """Load crosswalk, conference info, and batted-ball data."""
        # Load crosswalk
        if self.crosswalk_csv.exists():
            with open(self.crosswalk_csv) as f:
                for row in csv.DictReader(f):
                    d1b = _normalize(row["d1baseball_name"])
                    self._team_crosswalk[d1b] = row["canonical_id"]

        # Load conference assignments
        if self.canonical_csv.exists():
            with open(self.canonical_csv) as f:
                for row in csv.DictReader(f):
                    self._team_conference[row["canonical_id"]] = row.get("conference", "")

        # Load batted-ball data
        if not self.batted_ball_tsv.exists():
            return

        all_fb = []
        with open(self.batted_ball_tsv) as f:
            for line in f:
                cols = line.strip().split("\t")
                if cols[0] == "Qual." or len(cols) < 7:
                    continue

                player = _normalize(cols[1])
                team_d1b = _normalize(cols[2])
                fb_pct = _parse_pct(cols[6])  # FB% is column index 6

                if fb_pct is None:
                    continue

                cid = self._team_crosswalk.get(team_d1b)
                if cid is None:
                    continue

                if cid not in self._team_pitchers:
                    self._team_pitchers[cid] = []
                self._team_pitchers[cid].append((player, fb_pct))
                all_fb.append(fb_pct)

        # D1 overall average FB%
        if all_fb:
            self.league_avg_fb = sum(all_fb) / len(all_fb)

        # Team averages
        for cid, pitchers in self._team_pitchers.items():
            fb_values = [fb for _, fb in pitchers]
            self._team_avg_fb[cid] = sum(fb_values) / len(fb_values)

        # Conference averages
        conf_fb: dict[str, list[float]] = {}
        for cid, avg_fb in self._team_avg_fb.items():
            conf = self._team_conference.get(cid, "")
            if conf:
                conf_fb.setdefault(conf, []).append(avg_fb)
        for conf, values in conf_fb.items():
            self.conference_avg_fb[conf] = sum(values) / len(values)

    def _fb_to_sensitivity(self, fb_pct: float) -> float:
        """Convert raw FB% to sensitivity scalar."""
        if self.league_avg_fb == 0:
            return 1.0
        return FLOOR + SCALE * (fb_pct / self.league_avg_fb)

    def pitcher_fb_pct(
        self,
        canonical_id: str,
        pitcher_name: str | None = None,
    ) -> float | None:
        """
        Get raw FB% for a specific pitcher.

        Returns None if pitcher not found.
        """
        if pitcher_name and canonical_id in self._team_pitchers:
            pitcher_norm = pitcher_name.strip().lower()
            for name, fb in self._team_pitchers[canonical_id]:
                if name.strip().lower() == pitcher_norm:
                    return fb
        return None

    def pitcher_sensitivity(
        self,
        canonical_id: str,
        pitcher_name: str | None = None,
    ) -> float:
        """
        Get fly-ball sensitivity for a specific starter.

        Looks up the pitcher's individual FB%. Falls back to team average,
        then conference average, then league average (1.0).
        """
        if self.league_avg_fb == 0:
            return 1.0

        # Try specific pitcher match
        fb = self.pitcher_fb_pct(canonical_id, pitcher_name)
        if fb is not None:
            return self._fb_to_sensitivity(fb)

        # Fall back to team average
        if canonical_id in self._team_avg_fb:
            return self._fb_to_sensitivity(self._team_avg_fb[canonical_id])

        # Fall back to conference average
        conf = self._team_conference.get(canonical_id, "")
        if conf and conf in self.conference_avg_fb:
            return self._fb_to_sensitivity(self.conference_avg_fb[conf])

        # No data → league average
        return 1.0

    def bullpen_sensitivity(self, canonical_id: str) -> float:
        """
        Get fly-ball sensitivity for a team's bullpen (team staff average).

        Uses the team's average FB% across all qualified pitchers.
        Falls back to conference average, then league average (1.0).
        """
        if self.league_avg_fb == 0:
            return 1.0

        if canonical_id in self._team_avg_fb:
            return self._fb_to_sensitivity(self._team_avg_fb[canonical_id])

        conf = self._team_conference.get(canonical_id, "")
        if conf and conf in self.conference_avg_fb:
            return self._fb_to_sensitivity(self.conference_avg_fb[conf])

        return 1.0

    def summary(self) -> str:
        """Print summary statistics."""
        lines = [
            f"FB Sensitivity Lookup loaded:",
            f"  Teams with data: {len(self._team_pitchers)}",
            f"  Total pitchers: {sum(len(v) for v in self._team_pitchers.values())}",
            f"  D1 avg FB%: {self.league_avg_fb:.1%}",
            f"  Conferences: {len(self.conference_avg_fb)}",
        ]
        return "\n".join(lines)
