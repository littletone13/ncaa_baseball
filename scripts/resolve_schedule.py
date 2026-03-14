"""Resolve the day's game schedule from NCAA API, ESPN API, and Odds API.

Outputs a schedule CSV with team identifiers and start times for use by
downstream pipeline steps (starter lookup, weather fetch, simulation).

Usage:
  python3 scripts/resolve_schedule.py --date 2026-03-14
  python3 scripts/resolve_schedule.py --date 2026-03-14 --out data/daily/2026-03-14/schedule.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

import _bootstrap  # noqa: F401
from ncaa_baseball.phase1 import (
    build_odds_name_to_canonical,
    load_canonical_teams,
    resolve_odds_teams,
)


def _resolve_team(
    name: str,
    team_idx_map: dict[str, int],
    name_to_cid: dict[str, str],
    canonical: pd.DataFrame,
    name_to_canonical: dict,
) -> tuple[str, int]:
    """Resolve an ESPN/NCAA team display name to (canonical_id, team_idx).

    Returns ("", 0) when the name cannot be resolved at all.
    Returns (canonical_id, 0) when the team is known but absent from the
    Stan model's training data.
    """
    name = name.strip()
    if not name:
        return "", 0
    # Direct canonical_id lookup
    if name in team_idx_map:
        return name, team_idx_map[name]
    # By short team_name (lowercase)
    cid = name_to_cid.get(name.lower())
    if cid:
        return cid, team_idx_map.get(cid, 0)
    # Fuzzy via resolve_odds_teams
    h_t, _ = resolve_odds_teams(name, "", canonical, name_to_canonical)
    if h_t:
        cid = h_t[0]
        return cid, team_idx_map.get(cid, 0)
    # Partial substring match against canonical_id
    nl = name.lower()
    for cid, idx in team_idx_map.items():
        if nl in cid.lower():
            return cid, idx
    return "", 0


def resolve_schedule(
    date: str,
    team_table_csv: Path = Path("data/processed/team_table.csv"),
    canonical_csv: Path = Path("data/registries/canonical_teams_2026.csv"),
    odds_jsonl: Path = Path("data/raw/odds/odds_latest.jsonl"),
    meta_json: Path = Path("data/processed/run_event_fit_meta.json"),
    out_csv: Path | None = None,
) -> pd.DataFrame:
    """Fetch and resolve the game schedule for *date* (YYYY-MM-DD).

    Returns a DataFrame with columns:
      game_num, home_name, away_name, home_cid, away_cid,
      home_team_idx, away_team_idx, start_utc, start_local_hour

    Also writes to *out_csv* when provided.
    """
    # ── Load team index ────────────────────────────────────────────────────
    team_df = pd.read_csv(team_table_csv, dtype=str)
    team_idx_map: dict[str, int] = {
        str(r["canonical_id"]).strip(): int(r["team_idx"])
        for _, r in team_df.iterrows()
        if str(r.get("canonical_id", "")).strip()
    }

    # Read N_teams from meta to support the "idx > N_teams → 0" clamp
    N_teams = 0
    if meta_json.exists():
        with open(meta_json) as f:
            meta = json.load(f)
        N_teams = meta.get("N_teams", 0)

    # ── Load canonical teams ───────────────────────────────────────────────
    canonical = load_canonical_teams(canonical_csv)
    name_to_canonical = build_odds_name_to_canonical(canonical)

    # name_to_cid: short team_name (lowercase) → canonical_id
    name_to_cid: dict[str, str] = {}
    for _, row in canonical.iterrows():
        tname = str(row.get("team_name", "")).strip()
        cid = str(row.get("canonical_id", "")).strip()
        if tname and cid:
            name_to_cid[tname.lower()] = cid
            for word in tname.split():
                w = word.lower().strip()
                if len(w) > 3 and w not in name_to_cid:
                    name_to_cid[w] = cid

    # Odds API name → canonical_id
    odds_name_to_cid: dict[str, str] = {}
    for _, row in canonical.iterrows():
        oname = str(row.get("odds_api_name", "")).strip()
        cid = str(row.get("canonical_id", "")).strip()
        if oname and cid:
            odds_name_to_cid[oname] = cid

    # Stadium longitude lookup (for local hour estimation)
    # Load from stadium_orientations.csv; fall back to empty dict
    stadium_lon: dict[str, float] = {}
    stadium_csv = Path("data/registries/stadium_orientations.csv")
    if stadium_csv.exists():
        sdf = pd.read_csv(stadium_csv, dtype=str)
        for _, row in sdf.iterrows():
            cid = str(row.get("canonical_id", "")).strip()
            try:
                lon = float(row.get("lon", "") or "")
                if cid:
                    stadium_lon[cid] = lon
            except (ValueError, TypeError):
                pass

    # ── Fetch game schedule from NCAA API ──────────────────────────────────
    # matchups: list of (home_name, away_name, start_utc_or_None)
    matchups: list[tuple[str, str, str | None]] = []
    dt_parts = date.split("-")
    ncaa_url = (
        f"https://ncaa-api.henrygd.me/scoreboard/baseball/d1"
        f"/{dt_parts[0]}/{dt_parts[1]}/{dt_parts[2]}"
    )
    try:
        req = Request(ncaa_url, headers={"User-Agent": "Mozilla/5.0 (Macintosh)"})
        ncaa_data = json.loads(urlopen(req, timeout=15).read())
        for g in ncaa_data.get("games", []):
            game = g.get("game", {})
            home = game.get("home", {})
            away = game.get("away", {})
            h_names = home.get("names", {})
            a_names = away.get("names", {})
            h_name = (
                h_names.get("full", "").strip()
                or h_names.get("short", "").strip()
                or "?"
            )
            a_name = (
                a_names.get("full", "").strip()
                or a_names.get("short", "").strip()
                or "?"
            )
            if h_name != "?" and a_name != "?":
                matchups.append((h_name, a_name, None))  # NCAA API times unreliable
        print(f"{len(matchups)} games on {date} (NCAA API)", file=sys.stderr)
    except Exception as ex:
        print(f"NCAA API failed ({ex}), falling back to ESPN...", file=sys.stderr)

    # ── Fetch ESPN start times (and schedule fallback) ─────────────────────
    espn_times: dict[tuple[str, str], str] = {}          # (home_name, away_name) → UTC ISO
    espn_times_cid: dict[tuple[str, str], str] = {}       # (home_cid, away_cid) → UTC ISO
    try:
        dt_nodash = date.replace("-", "")
        espn_url = (
            f"https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball"
            f"/scoreboard?dates={dt_nodash}&limit=200"
        )
        req = Request(espn_url, headers={"User-Agent": "Mozilla/5.0 (Macintosh)"})
        data = json.loads(urlopen(req, timeout=15).read())
        for e in data.get("events", []):
            start_utc = e.get("date")
            comps = e.get("competitions", [{}])
            if not comps:
                continue
            competitors = comps[0].get("competitors", [])
            if len(competitors) != 2:
                continue
            home_c = next((x for x in competitors if x.get("homeAway") == "home"), None)
            away_c = next((x for x in competitors if x.get("homeAway") == "away"), None)
            if home_c and away_c:
                h_espn = home_c.get("team", {}).get("displayName", "?")
                a_espn = away_c.get("team", {}).get("displayName", "?")
                if start_utc:
                    espn_times[(h_espn, a_espn)] = start_utc
                    # Resolve to canonical_ids for fuzzy merge
                    h_cid_e, _ = _resolve_team(
                        h_espn, team_idx_map, name_to_cid, canonical, name_to_canonical
                    )
                    a_cid_e, _ = _resolve_team(
                        a_espn, team_idx_map, name_to_cid, canonical, name_to_canonical
                    )
                    if h_cid_e and a_cid_e:
                        espn_times_cid[(h_cid_e, a_cid_e)] = start_utc
                if not matchups:
                    matchups.append((h_espn, a_espn, start_utc))
        if matchups and matchups[0][2] is None:
            print(
                f"  ESPN provided {len(espn_times)} start times, "
                f"{len(espn_times_cid)} resolved to canonical",
                file=sys.stderr,
            )
    except Exception as ex2:
        print(f"  ESPN time fetch failed: {ex2}", file=sys.stderr)

    # Full ESPN fallback: if NCAA API returned nothing
    if not matchups and espn_times:
        for (h, a), t in espn_times.items():
            matchups.append((h, a, t))
        print(f"{len(matchups)} games on {date} (ESPN fallback)", file=sys.stderr)

    # ── Load odds commence times as additional start time source ───────────
    odds_times_cid: dict[tuple[str, str], str] = {}
    if odds_jsonl.exists():
        try:
            with open(odds_jsonl) as fod:
                for line in fod:
                    og = json.loads(line)
                    ct = og.get("commence_time")
                    if not ct:
                        continue
                    hc = odds_name_to_cid.get(og.get("home_team", ""))
                    ac = odds_name_to_cid.get(og.get("away_team", ""))
                    if hc and ac:
                        odds_times_cid[(hc, ac)] = ct
            if odds_times_cid:
                print(
                    f"  Odds API provided {len(odds_times_cid)} commence times",
                    file=sys.stderr,
                )
        except Exception as ex3:
            print(f"  Odds commence time load failed: {ex3}", file=sys.stderr)

    # ── Resolve each game ──────────────────────────────────────────────────
    rows = []
    for game_num, (h_name, a_name, start_utc) in enumerate(matchups):
        h_cid, h_idx = _resolve_team(
            h_name, team_idx_map, name_to_cid, canonical, name_to_canonical
        )
        a_cid, a_idx = _resolve_team(
            a_name, team_idx_map, name_to_cid, canonical, name_to_canonical
        )

        # Clamp team indices to posterior size (new teams → league avg)
        if N_teams and h_idx > N_teams:
            h_idx = 0
        if N_teams and a_idx > N_teams:
            a_idx = 0

        # ── Resolve start time ─────────────────────────────────────────────
        # Priority: NCAA tuple value → ESPN by raw names → ESPN by canonical_id
        #           → Odds API commence time
        if start_utc is None:
            start_utc = espn_times.get((h_name, a_name))
        if start_utc is None and h_cid and a_cid:
            start_utc = espn_times_cid.get((h_cid, a_cid))
        if start_utc is None and h_cid and a_cid:
            start_utc = odds_times_cid.get((h_cid, a_cid))

        # ── Compute local start hour ───────────────────────────────────────
        # Approximate local time from UTC + stadium longitude offset.
        # Every 15° of longitude ≈ 1 hour from UTC.
        # Default to 18 (6pm local) when no time is available.
        start_local_hour: int | None = None
        if start_utc:
            try:
                utc_dt = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
                lon = stadium_lon.get(h_cid) if h_cid else None
                if lon is not None:
                    tz_offset_hr = round(lon / 15.0)
                    local_dt = utc_dt + timedelta(hours=tz_offset_hr)
                    start_local_hour = local_dt.hour
            except Exception:
                pass

        if start_local_hour is None:
            start_local_hour = 18  # 6pm local default

        rows.append(
            {
                "game_num": game_num,
                "home_name": h_name,
                "away_name": a_name,
                "home_cid": h_cid,
                "away_cid": a_cid,
                "home_team_idx": h_idx,
                "away_team_idx": a_idx,
                "start_utc": start_utc or "",
                "start_local_hour": start_local_hour,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "game_num",
            "home_name",
            "away_name",
            "home_cid",
            "away_cid",
            "home_team_idx",
            "away_team_idx",
            "start_utc",
            "start_local_hour",
        ],
    )

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"Schedule written to {out_csv}", file=sys.stderr)

    print(f"{len(df)} games on {date}", file=sys.stderr)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve the day's NCAA baseball game schedule."
    )
    parser.add_argument("--date", type=str, required=True, help="Date (YYYY-MM-DD)")
    parser.add_argument(
        "--team-table",
        type=Path,
        default=Path("data/processed/team_table.csv"),
        help="Team table CSV with canonical_id and team_idx columns",
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=Path("data/registries/canonical_teams_2026.csv"),
        help="Canonical teams registry CSV",
    )
    parser.add_argument(
        "--odds-jsonl",
        type=Path,
        default=Path("data/raw/odds/odds_latest.jsonl"),
        help="Odds API latest JSONL (for commence times)",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("data/processed/run_event_fit_meta.json"),
        help="Stan model fit metadata JSON",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: data/daily/{date}/schedule.csv)",
    )
    args = parser.parse_args()

    out_csv = args.out
    if out_csv is None:
        out_csv = Path(f"data/daily/{args.date}/schedule.csv")

    df = resolve_schedule(
        date=args.date,
        team_table_csv=args.team_table,
        canonical_csv=args.canonical,
        odds_jsonl=args.odds_jsonl,
        meta_json=args.meta,
        out_csv=out_csv,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
