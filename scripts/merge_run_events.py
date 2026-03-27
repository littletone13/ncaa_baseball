#!/usr/bin/env python3
"""Merge ESPN run_events with linescore-derived run_events.

Reads the ESPN-only run_events.csv (from extract_espn.py) and the
linescore-derived run_events_from_linescores.csv, deduplicates on event_id
(preferring ESPN where both exist), and writes a unified run_events.csv.

Then backfills missing pitcher IDs from pitcher_appearances.csv starters,
so NCAA pitchers get Stan model indices (critical for pitcher coverage).
"""
import re as re_mod
import sys
import unicodedata
from pathlib import Path

import pandas as pd

CORE_COLS = [
    "event_id", "game_date", "season",
    "home_canonical_id", "away_canonical_id",
    "home_pitcher_espn_id", "away_pitcher_espn_id",
    "home_run_1", "home_run_2", "home_run_3", "home_run_4",
    "away_run_1", "away_run_2", "away_run_3", "away_run_4",
    "home_score", "away_score",
]


def main():
    espn_csv = Path("data/processed/run_events.csv")
    ls_csv = Path("data/processed/run_events_from_linescores.csv")
    out_csv = Path("data/processed/run_events.csv")

    espn = pd.read_csv(espn_csv, dtype=str)
    n_before = len(espn)

    if not ls_csv.exists():
        print(f"No linescores file at {ls_csv}, nothing to merge.", file=sys.stderr)
        return

    ls = pd.read_csv(ls_csv, dtype=str)

    # Rename linescore pitcher columns to match ESPN convention
    if "home_pitcher_id" in ls.columns and "home_pitcher_espn_id" not in ls.columns:
        ls = ls.rename(columns={
            "home_pitcher_id": "home_pitcher_espn_id",
            "away_pitcher_id": "away_pitcher_espn_id",
        })

    # Ensure all core columns exist in both
    for c in CORE_COLS:
        if c not in espn.columns:
            espn[c] = ""
        if c not in ls.columns:
            ls[c] = ""

    # Keep only core columns to avoid schema drift
    espn = espn[CORE_COLS]
    ls = ls[CORE_COLS]

    # Concat — ESPN first so dedup keeps ESPN version
    combined = pd.concat([espn, ls], ignore_index=True)
    combined = combined.drop_duplicates(subset=["event_id"], keep="first")

    n_new = len(combined) - n_before

    # ── Backfill pitcher IDs from pitcher_appearances starters ────────────
    appearances_csv = Path("data/processed/pitcher_appearances.csv")
    n_filled = 0
    if appearances_csv.exists():
        pa = pd.read_csv(appearances_csv, low_memory=False)
        starters = pa[pa["starter"] == True].copy()  # noqa: E712

        # Build lookup: (date, team) → stable NCAA pitcher ID
        # Must match _make_ncaa_index_key() in build_pitcher_table.py exactly
        _SUFFIX_RE = re_mod.compile(r"\s+(Jr\.?|Sr\.?|III|II|IV|V)\s*$", re_mod.IGNORECASE)

        def _make_ncaa_id(name: str, team_cid: str) -> str:
            if not name or not team_cid:
                return ""
            n = name.strip()
            n = unicodedata.normalize("NFKD", n)
            n = "".join(c for c in n if not unicodedata.combining(c))
            n = _SUFFIX_RE.sub("", n).strip()
            n = n.replace(".", "")
            n = re_mod.sub(r"\s+", "_", n.lower().strip())
            n = re_mod.sub(r"[^a-z0-9_]", "", n)
            if not n:
                return ""
            team_clean = team_cid.replace(" ", "_")
            return f"NCAA_{n}__{team_clean}"

        starter_lookup: dict[str, str] = {}  # "date|team_cid" → pitcher_id
        for _, row in starters.iterrows():
            date = str(row.get("game_date", ""))[:10]
            team = str(row.get("team_canonical_id", "")).strip()
            if not date or not team:
                continue
            # Prefer ESPN ID if available; otherwise build NCAA format
            eid = str(row.get("pitcher_espn_id", "")).strip()
            if eid and eid not in ("", "nan", "None"):
                pid = eid
            else:
                pname = str(row.get("pitcher_name", ""))
                pid = _make_ncaa_id(pname, team)
            if pid:
                starter_lookup[f"{date}|{team}"] = pid

        # Fill missing pitcher IDs in combined run_events
        for side, cid_col in [("home", "home_canonical_id"), ("away", "away_canonical_id")]:
            pid_col = f"{side}_pitcher_espn_id"
            missing = combined[pid_col].fillna("").astype(str).str.strip().isin(["", "nan"])
            for idx in combined.index[missing]:
                date = str(combined.at[idx, "game_date"])[:10]
                team = str(combined.at[idx, cid_col]).strip()
                key = f"{date}|{team}"
                if key in starter_lookup:
                    combined.at[idx, pid_col] = starter_lookup[key]
                    n_filled += 1

        print(f"Pitcher backfill: {n_filled} slots filled from appearances", file=sys.stderr)

    combined.to_csv(out_csv, index=False)

    print(
        f"Run events: {n_before} ESPN + {len(ls)} linescores "
        f"→ {len(combined)} combined ({n_new} new)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
