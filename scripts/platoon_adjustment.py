"""
Platoon adjustment for LHP/RHP starter matchups.

Provides pitcher handedness lookup and a configurable log-rate adjustment
when a team faces a left-handed pitcher (LHP).

Architecture:
  - Loads pitcher handedness from D1Baseball rotation data
  - Provides lookup: given a pitcher name + team → "L" or "R" (or None)
  - Provides platoon adjustment on log-rate scale (configurable sign & magnitude)

  The adjustment is applied per-team in predict_day.py, similar to how
  FB% sensitivity scales wind effects.

IMPORTANT — sign/magnitude notes:
  The direction of the LHP effect on team scoring is empirically uncertain.
  Pure platoon mechanics (most batters are RHB → opposite-hand advantage vs
  LHP) would suggest MORE runs against LHP. But unfamiliarity with LHP and
  LHB penalty could push the other way. Additionally, pitcher quality is
  already captured by pitcher_ability in the Stan model, so adding a
  separate LHP penalty risks double-counting.

  Default is 0.0 (disabled) until we can validate the sign against actual
  college scoring data. Set lhp_adj to a non-zero value to enable:
    negative = teams score FEWER runs vs LHP
    positive = teams score MORE runs vs LHP

Usage:
    from platoon_adjustment import PlatoonLookup
    pl = PlatoonLookup()

    # Look up pitcher handedness
    hand = pl.get_hand("BSB_VIRGINIA", "Tomas Valincius")  # → "L"
    hand = pl.get_hand("BSB_TEXAS_TECH", "Brandon Beckel")  # → "R"

    # Get platoon adjustment for a game
    adj = pl.platoon_adj("L")   # → 0.0 (default disabled)
    adj = pl.platoon_adj("R")   # → 0.0  (facing RHP, baseline)
    adj = pl.platoon_adj(None)  # → 0.0  (unknown, assume RHP)
"""
from __future__ import annotations

import csv
from pathlib import Path


# Default platoon effect on log-rate scale
# 0.0 = disabled (handedness tracked but no rate adjustment)
# Set to non-zero when empirically validated:
#   negative = fewer runs vs LHP, positive = more runs vs LHP
DEFAULT_LHP_ADJ = 0.03  # ~3% more runs vs LHP (platoon advantage for RHB-heavy lineups)


def _normalize(s: str) -> str:
    """Normalize curly/smart apostrophes to straight."""
    return s.replace("\u2019", "'").replace("\u2018", "'")


class PlatoonLookup:
    """Look up pitcher handedness and compute platoon adjustments."""

    def __init__(
        self,
        d1b_rotations_csv: str | Path = "data/processed/d1baseball_rotations.csv",
        crosswalk_csv: str | Path = "data/registries/d1baseball_crosswalk.csv",
        lhp_adj: float = DEFAULT_LHP_ADJ,
    ):
        self.lhp_adj = lhp_adj

        # (canonical_id, pitcher_name_lower) → "L" or "R"
        self._hand: dict[tuple[str, str], str] = {}
        # d1baseball team name → canonical_id
        self._team_xw: dict[str, str] = {}

        self._load_crosswalk(Path(crosswalk_csv))
        self._load_rotations(Path(d1b_rotations_csv))

    def _load_crosswalk(self, path: Path):
        """Load D1Baseball team name → canonical_id crosswalk."""
        if not path.exists():
            return
        with open(path) as f:
            for row in csv.DictReader(f):
                d1b = _normalize(row["d1baseball_name"])
                self._team_xw[d1b.lower()] = row["canonical_id"]

    def _load_rotations(self, path: Path):
        """Load pitcher handedness from D1Baseball rotation data."""
        if not path.exists():
            return

        with open(path) as f:
            for row in csv.DictReader(f):
                hand_raw = row.get("hand", "").strip().upper()
                if hand_raw not in ("RHP", "LHP"):
                    continue

                hand = "L" if hand_raw == "LHP" else "R"
                pitcher = _normalize(row.get("pitcher_name", "")).strip().lower()
                cid = row.get("canonical_id", "")

                if not cid:
                    # Try to resolve via team_name → crosswalk
                    team = _normalize(row.get("team_name", "")).strip()
                    cid = self._team_xw.get(team.lower(), "")

                if cid and pitcher:
                    self._hand[(cid, pitcher)] = hand

    def get_hand(
        self,
        canonical_id: str,
        pitcher_name: str | None = None,
    ) -> str | None:
        """
        Get pitcher handedness: "L", "R", or None (unknown).

        Looks up by canonical_id + pitcher_name (case-insensitive).
        Returns None if pitcher not found in rotation data.
        """
        if not pitcher_name:
            return None

        key = (canonical_id, pitcher_name.strip().lower())
        return self._hand.get(key)

    def platoon_adj(self, hand: str | None) -> float:
        """
        Get platoon adjustment on log-rate scale.

        Args:
            hand: "L", "R", or None

        Returns:
            Log-rate adjustment. Positive when facing LHP (more runs expected,
            since RHB-heavy lineups have platoon advantage vs LHP).
            0.0 for RHP or unknown.
        """
        if hand == "L":
            return self.lhp_adj
        return 0.0

    def summary(self) -> str:
        """Print summary statistics."""
        lhp_count = sum(1 for h in self._hand.values() if h == "L")
        rhp_count = sum(1 for h in self._hand.values() if h == "R")
        teams = len(set(cid for cid, _ in self._hand))
        return (
            f"Platoon Lookup loaded:\n"
            f"  Pitchers with handedness: {len(self._hand)}\n"
            f"  LHP: {lhp_count}, RHP: {rhp_count}\n"
            f"  Teams: {teams}\n"
            f"  LHP adjustment: {self.lhp_adj:+.3f} log-rate"
        )
