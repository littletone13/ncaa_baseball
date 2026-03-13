"""
Look up projected starting pitchers for upcoming games.

Priority chain for weekend games (Fri/Sat/Sun):
  0a. D1Baseball expert picks (press conference intel, ~86 major teams)
  0b. Appearance-based weekend rotation projections (all ~308 teams)
  1.  Most-recent same-day-of-week starter from appearances
  2.  Most-rested-pitcher heuristic fallback

For midweek games (Mon/Tue/Wed):
  1. Most-recent midweek starter
  2. Most-rested-pitcher fallback

Usage:
  from lookup_starters import StarterLookup
  sl = StarterLookup("data/processed/pitcher_appearances.csv",
                      "data/processed/pitcher_registry.csv")
  name, pid, pidx = sl.get_starter("BSB_ARKANSAS", "2026-03-09")
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


class StarterLookup:
    """Resolve projected starting pitcher for a team on a given date."""

    def __init__(
        self,
        appearances_csv: Path | str = "data/processed/pitcher_appearances.csv",
        registry_csv: Path | str = "data/processed/pitcher_registry.csv",
        pitcher_index_csv: Path | str = "data/processed/run_event_pitcher_index.csv",
        weekend_rotations_csv: Path | str = "data/processed/weekend_rotations.csv",
        d1baseball_rotations_csv: Path | str = "data/processed/d1baseball_rotations.csv",
    ):
        self.pa = pd.read_csv(appearances_csv, dtype=str)
        self.pa["game_date"] = pd.to_datetime(self.pa["game_date"])

        self.registry = pd.read_csv(registry_csv, dtype=str)
        # pitcher_id -> pitcher_idx (int)
        pi_df = pd.read_csv(pitcher_index_csv, dtype=str)
        self.pid_to_idx: dict[str, int] = {}
        for _, r in pi_df.iterrows():
            pid = str(r.get("pitcher_espn_id", "")).strip()
            idx = int(r.get("pitcher_idx", 0))
            if pid and pid.lower() != "unknown":
                self.pid_to_idx[pid] = idx

        # pitcher_id -> (pitcher_name, pitcher_idx)
        self.pid_to_info: dict[str, tuple[str, int]] = {}
        for _, r in self.registry.iterrows():
            pid = str(r.get("pitcher_id", "")).strip()
            name = str(r.get("pitcher_name", "")).strip()
            pidx_val = int(r.get("pitcher_idx", 0))
            if pid:
                self.pid_to_info[pid] = (name, pidx_val)

        # name (lower) + team_canonical_id -> pitcher_id for fuzzy matching
        self._name_team_to_pid: dict[tuple[str, str], str] = {}
        for _, r in self.registry.iterrows():
            pid = str(r.get("pitcher_id", "")).strip()
            name = str(r.get("pitcher_name", "")).strip().lower()
            team = str(r.get("team", "")).strip()
            if pid and name and name != "unknown":
                self._name_team_to_pid[(name, team)] = pid

        # Filter to starters only
        self.starters = self.pa[self.pa["role"] == "starter"].copy()
        self.starters = self.starters.sort_values("game_date", ascending=False)

        # Weekend rotation projections (from build_weekend_rotations.py)
        self._weekend_rotations: dict[tuple[str, str], dict] = {}
        wr_path = Path(weekend_rotations_csv)
        if wr_path.exists():
            wr_df = pd.read_csv(wr_path, dtype=str)
            for _, r in wr_df.iterrows():
                cid = str(r.get("canonical_id", "")).strip()
                day = str(r.get("day", "")).strip()  # fri/sat/sun
                if cid and day:
                    self._weekend_rotations[(cid, day)] = {
                        "pitcher_name": str(r.get("pitcher_name", "")),
                        "pitcher_id": str(r.get("pitcher_id", "")),
                        "confidence": str(r.get("confidence", "low")),
                        "starts": int(r.get("starts_this_role", 0)),
                    }
            print(f"  Loaded {len(self._weekend_rotations)} weekend rotation projections",
                  file=sys.stderr)

        # D1Baseball expert rotations (from scrape_d1baseball_rotations.py)
        # This is the HIGHEST priority source — press conference intel
        self._d1baseball_rotations: dict[tuple[str, str], dict] = {}
        d1b_path = Path(d1baseball_rotations_csv)
        if d1b_path.exists():
            d1b_df = pd.read_csv(d1b_path, dtype=str)
            for _, r in d1b_df.iterrows():
                cid = str(r.get("canonical_id", "")).strip()
                day = str(r.get("day", "")).strip()  # fri/sat/sun
                pname = str(r.get("pitcher_name", "")).strip()
                if cid and day and pname:
                    self._d1baseball_rotations[(cid, day)] = {
                        "pitcher_name": pname,
                        "hand": str(r.get("hand", "")),
                        "era": str(r.get("era", "")),
                        "source": str(r.get("source", "d1baseball")),
                    }
            print(f"  Loaded {len(self._d1baseball_rotations)} D1Baseball expert rotation picks",
                  file=sys.stderr)

    def get_starter(
        self, team_canonical_id: str, game_date: str
    ) -> tuple[str, str, int]:
        """
        Get projected starter for a team on a given date.

        Returns: (pitcher_name, pitcher_id, pitcher_idx)
        If unknown: ("unknown", "", 0)
        """
        gd = pd.Timestamp(game_date)
        dow = gd.day_of_week  # 0=Mon ... 6=Sun

        # Strategy 0a: D1Baseball expert picks (highest priority — press conference intel)
        dow_to_day = {4: "fri", 5: "sat", 6: "sun"}
        if dow in dow_to_day:
            day_key = dow_to_day[dow]
            d1b = self._d1baseball_rotations.get((team_canonical_id, day_key))
            if d1b:
                pname = d1b["pitcher_name"]
                pidx = self._resolve_by_name(pname, team_canonical_id)
                if pidx > 0:
                    return (pname, f"d1b_{pname}", pidx)

        # Strategy 0b: Appearance-based weekend rotation projections (fallback)
        if dow in dow_to_day:
            day_key = dow_to_day[dow]
            wr = self._weekend_rotations.get((team_canonical_id, day_key))
            if wr and wr["confidence"] in ("high", "medium"):
                pname = wr["pitcher_name"]
                pid = wr["pitcher_id"]
                pidx = self._resolve_idx(pid)
                if pidx == 0:
                    pidx = self._resolve_by_name(pname, team_canonical_id)
                if pidx > 0:
                    return (pname, pid, pidx)

        team_starters = self.starters[
            self.starters["team_canonical_id"] == team_canonical_id
        ].copy()

        if team_starters.empty:
            return ("unknown", "", 0)

        # Strategy 1: For midweek games (Mon/Tue/Wed), look at midweek history
        if dow <= 2:  # Mon, Tue, Wed
            midweek = team_starters[
                team_starters["game_date"].dt.day_of_week.isin([0, 1, 2])
            ]
            if not midweek.empty:
                # Most recent midweek starter
                best = midweek.iloc[0]
                pid = str(best.get("pitcher_id", "")).strip()
                name = str(best.get("pitcher_name", "")).strip()
                pidx = self._resolve_idx(pid)
                if pidx > 0:
                    return (name, pid, pidx)

        # Strategy 2: For weekend games (Fri/Sat/Sun), match rotation slot
        if dow >= 4:  # Fri=4, Sat=5, Sun=6
            same_dow = team_starters[
                team_starters["game_date"].dt.day_of_week == dow
            ]
            if not same_dow.empty:
                best = same_dow.iloc[0]
                pid = str(best.get("pitcher_id", "")).strip()
                name = str(best.get("pitcher_name", "")).strip()
                pidx = self._resolve_idx(pid)
                if pidx > 0:
                    return (name, pid, pidx)

        # Strategy 3: Most-rested pitcher
        # Get last 4 unique starters for this team
        seen_pids = []
        seen_rows = []
        for _, r in team_starters.iterrows():
            pid = str(r.get("pitcher_id", "")).strip()
            if pid not in seen_pids:
                seen_pids.append(pid)
                seen_rows.append(r)
            if len(seen_pids) >= 4:
                break

        # Pick the one who pitched longest ago (most rested)
        if seen_rows:
            most_rested = seen_rows[-1]
            pid = str(most_rested.get("pitcher_id", "")).strip()
            name = str(most_rested.get("pitcher_name", "")).strip()
            pidx = self._resolve_idx(pid)
            if pidx > 0:
                return (name, pid, pidx)

        # Strategy 4: Just use the most recent starter
        best = team_starters.iloc[0]
        pid = str(best.get("pitcher_id", "")).strip()
        name = str(best.get("pitcher_name", "")).strip()
        pidx = self._resolve_idx(pid)
        return (name, pid, pidx)

    def _resolve_by_name(self, pitcher_name: str, team_canonical_id: str) -> int:
        """Try to resolve pitcher_idx by name + team fuzzy matching."""
        name_lower = pitcher_name.strip().lower()
        # Exact name+team match
        pid = self._name_team_to_pid.get((name_lower, team_canonical_id))
        if pid:
            return self._resolve_idx(pid)
        # Try last-name match against registry
        parts = name_lower.split()
        last = parts[-1] if parts else name_lower
        for (n, t), pid in self._name_team_to_pid.items():
            if t == team_canonical_id and (n == last or n.endswith(last)):
                idx = self._resolve_idx(pid)
                if idx > 0:
                    return idx
        # Try last-name match against appearances data (catches pitchers
        # not in registry but who have appeared in games)
        team_apps = self.pa[self.pa["team_canonical_id"] == team_canonical_id]
        if not team_apps.empty:
            for _, r in team_apps.iterrows():
                pname = str(r.get("pitcher_name", "")).strip().lower()
                if pname == last or pname.endswith(last):
                    pid = str(r.get("pitcher_id", "")).strip()
                    idx = self._resolve_idx(pid)
                    if idx > 0:
                        return idx
        return 0

    def _resolve_idx(self, pitcher_id: str) -> int:
        """Map pitcher_id to pitcher_idx."""
        if not pitcher_id:
            return 0
        # Direct lookup
        idx = self.pid_to_idx.get(pitcher_id, 0)
        if idx > 0:
            return idx
        # Try without prefix
        if pitcher_id.startswith("ESPN_"):
            raw = pitcher_id[5:]
            idx = self.pid_to_idx.get(raw, 0)
            if idx > 0:
                return idx
        # Try with prefix
        idx = self.pid_to_idx.get(f"ESPN_{pitcher_id}", 0)
        if idx > 0:
            return idx
        # Lookup from registry
        info = self.pid_to_info.get(pitcher_id)
        if info:
            return info[1]
        return 0

    def get_all_starters_for_date(
        self, matchups: list[tuple[str, str, str, str]], game_date: str
    ) -> list[tuple[str, int, str, int]]:
        """
        For a list of matchups, return projected starters.

        matchups: list of (home_name, away_name, home_cid, away_cid)
        Returns: list of (home_pitcher_name, home_pitcher_idx,
                          away_pitcher_name, away_pitcher_idx)
        """
        results = []
        for h_name, a_name, h_cid, a_cid in matchups:
            hp_name, hp_id, hp_idx = self.get_starter(h_cid, game_date)
            ap_name, ap_id, ap_idx = self.get_starter(a_cid, game_date)
            results.append((hp_name, hp_idx, ap_name, ap_idx))
        return results


def scrape_d1baseball_rotations() -> dict[str, list[dict]]:
    """
    Scrape the latest D1Baseball projected weekend rotations article.

    Returns: dict mapping team abbreviation -> list of pitcher dicts:
        [{"name": "Smith", "day": "Friday", "hand": "R", "url": "/player/..."}]

    Note: This covers ~78 major conference teams for Fri/Sat/Sun only.
    """
    # Find latest article URL
    try:
        cat_url = "https://d1baseball.com/category/weekend-preview/"
        req = Request(cat_url, headers={"User-Agent": "Mozilla/5.0 (Macintosh)"})
        html = urlopen(req, timeout=15).read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"D1Baseball scrape failed: {e}", file=sys.stderr)
        return {}

    # Find the most recent "projected-weekend-rotations" link
    pattern = r'href="(https://d1baseball\.com/weekend-preview/projected-weekend-rotations[^"]*)"'
    matches = re.findall(pattern, html)
    if not matches:
        print("No projected rotations article found on d1baseball.com", file=sys.stderr)
        return {}

    article_url = matches[0]
    print(f"  D1Baseball rotations: {article_url}", file=sys.stderr)

    try:
        req = Request(article_url, headers={"User-Agent": "Mozilla/5.0 (Macintosh)"})
        html = urlopen(req, timeout=15).read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"D1Baseball article fetch failed: {e}", file=sys.stderr)
        return {}

    # Parse starters matrix
    # Look for data-id attributes on team rows
    team_pattern = r'data-id="([^"]+)"'
    starter_pattern = r'starters-matrix__starter[^>]*href="(/player/[^"]+)"[^>]*>([^<]+)<'

    result: dict[str, list[dict]] = {}

    # Split by team rows
    team_blocks = re.split(r'data-id="', html)[1:]  # skip before first
    for block in team_blocks:
        team_id_match = re.match(r'([^"]+)"', block)
        if not team_id_match:
            continue
        team_abbr = team_id_match.group(1)

        # Find all starters in this block (up to next team)
        starters = re.findall(
            r'starters-matrix__starter[^>]*href="(/player/[^"]+)"[^>]*>\s*([^<]+)',
            block[:3000],  # limit to avoid spanning into next team
        )
        pitchers = []
        days = ["Friday", "Saturday", "Sunday"]
        for i, (url, name) in enumerate(starters[:3]):
            name = name.strip().rstrip("*")
            if name.upper() != "TBA":
                pitchers.append({
                    "name": name,
                    "day": days[i] if i < len(days) else "Unknown",
                    "url": url,
                })
        if pitchers:
            result[team_abbr] = pitchers

    print(f"  D1Baseball: {len(result)} teams with rotation data", file=sys.stderr)
    return result
