"""
Odds data pipeline for NCAA baseball using the-odds-api.

Fetches current or historical odds, computes devigged fair probabilities from h2h,
and writes one JSONL record per game to data/raw/odds/.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests

from io_utils import safe_stamp, utc_now_iso

BASE_URL = "https://api.the-odds-api.com"
SPORT = "baseball_ncaa"
CURRENT_REGIONS = "uk,us,us2,us_dfs,us_ex,eu,au"
CURRENT_MARKETS = "h2h,spreads,totals"


def load_env() -> None:
    """Load .env from repo root into os.environ (KEY=value, no quotes)."""
    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def american_to_implied(american: int | float) -> float:
    """Convert American odds to implied probability (0-1)."""
    if american > 0:
        return 100.0 / (100.0 + american)
    return abs(american) / (abs(american) + 100.0)


def devig_h2h(
    outcomes: list[dict[str, Any]], home_team: str, away_team: str
) -> tuple[float, float] | None:
    """
    Multiplicative devig for a single book's h2h outcomes.
    Returns (fair_home, fair_away) or None if outcomes cannot be matched.
    """
    by_team: dict[str, float] = {}
    for o in outcomes:
        name = o.get("name")
        price = o.get("price")
        if name is None or price is None:
            continue
        by_team[name] = american_to_implied(price)
    home_impl = by_team.get(home_team)
    away_impl = by_team.get(away_team)
    if home_impl is None or away_impl is None:
        return None
    total = home_impl + away_impl
    if total <= 0:
        return None
    return (home_impl / total, away_impl / total)


def build_game_record(
    event: dict[str, Any],
    bookmaker_lines: list[dict[str, Any]],
    consensus_fair_home: float | None,
    consensus_fair_away: float | None,
    snapshot_ts: str | None = None,
) -> dict[str, Any]:
    """One JSONL record: game id, meta, all bookmaker lines, devigged fair prices."""
    record: dict[str, Any] = {
        "id": event.get("id"),
        "sport_key": event.get("sport_key"),
        "commence_time": event.get("commence_time"),
        "home_team": event.get("home_team"),
        "away_team": event.get("away_team"),
        "bookmaker_lines": bookmaker_lines,
        "consensus_fair_home": consensus_fair_home,
        "consensus_fair_away": consensus_fair_away,
        "fetched_at": utc_now_iso(),
    }
    if snapshot_ts:
        record["snapshot_timestamp"] = snapshot_ts
    return record


def process_events(
    events: list[dict[str, Any]], snapshot_ts: str | None = None
) -> list[dict[str, Any]]:
    """Compute devigged fair from h2h for each event; return list of game records."""
    records = []
    for ev in events:
        home_team = ev.get("home_team") or ""
        away_team = ev.get("away_team") or ""
        bookmaker_lines: list[dict[str, Any]] = []
        fair_homes: list[float] = []
        fair_aways: list[float] = []

        for bm in ev.get("bookmakers") or []:
            bm_key = bm.get("key", "")
            bm_title = bm.get("title", bm_key)
            line: dict[str, Any] = {
                "bookmaker_key": bm_key,
                "bookmaker_title": bm_title,
                "last_update": bm.get("last_update"),
                "markets": bm.get("markets", []),
            }
            bookmaker_lines.append(line)

            for m in bm.get("markets") or []:
                if m.get("key") != "h2h":
                    continue
                outcomes = m.get("outcomes") or []
                devigged = devig_h2h(outcomes, home_team, away_team)
                if devigged is not None:
                    fh, fa = devigged
                    fair_homes.append(fh)
                    fair_aways.append(fa)
                    line["h2h_fair_home"] = fh
                    line["h2h_fair_away"] = fa
                break

        consensus_fair_home = (
            sum(fair_homes) / len(fair_homes) if fair_homes else None
        )
        consensus_fair_away = (
            sum(fair_aways) / len(fair_aways) if fair_aways else None
        )

        records.append(
            build_game_record(
                ev,
                bookmaker_lines,
                consensus_fair_home,
                consensus_fair_away,
                snapshot_ts,
            )
        )
    return records


def print_quota_headers(resp: requests.Response) -> None:
    """Print x-requests-remaining and x-requests-used from response headers."""
    remaining = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    print(f"API quota: x-requests-remaining={remaining}, x-requests-used={used}")


def fetch_current_odds(
    api_key: str,
    sport: str = SPORT,
    regions: str = CURRENT_REGIONS,
    markets: str = CURRENT_MARKETS,
) -> tuple[list[dict[str, Any]], requests.Response]:
    """Fetch current odds for upcoming games. Returns (events, response)."""
    url = f"{BASE_URL.rstrip('/')}/v4/sports/{sport}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        data = data.get("data", []) if isinstance(data, dict) else []
    return data, resp


def fetch_historical_odds(
    api_key: str,
    date_iso: str,
    sport: str = SPORT,
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
) -> tuple[list[dict[str, Any]], str | None, requests.Response]:
    """
    Fetch historical odds snapshot. Costs 10 credits per region per market.
    Returns (events, snapshot_timestamp, response).
    """
    url = f"{BASE_URL.rstrip('/')}/v4/historical/sports/{sport}/odds"
    params = {
        "apiKey": api_key,
        "date": date_iso,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    body = resp.json()
    data = body.get("data", []) if isinstance(body, dict) else []
    ts = body.get("timestamp") if isinstance(body, dict) else None
    return data, ts, resp


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Append one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser(
        description="Fetch NCAA baseball odds (current or historical), devig h2h, write JSONL."
    )
    parser.add_argument(
        "--mode",
        choices=["current", "historical"],
        default="current",
        help="Current upcoming odds or historical snapshot",
    )
    parser.add_argument(
        "--sport",
        default=SPORT,
        help="Sport key (default: baseball_ncaa)",
    )
    parser.add_argument(
        "--date",
        help='Historical snapshot date ISO8601, e.g. "2025-05-18T16:00:00Z" (required for historical)',
    )
    parser.add_argument(
        "--regions",
        default=CURRENT_REGIONS,
        help=f"Comma-separated regions (current default: {CURRENT_REGIONS})",
    )
    parser.add_argument(
        "--markets",
        default=CURRENT_MARKETS,
        help="Comma-separated markets",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path (default: data/raw/odds/odds_<sport>_<date>.jsonl)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ODDS_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing ODDS_API_KEY. Set in .env or environment.")

    if args.mode == "historical":
        if not args.date:
            raise SystemExit("Historical mode requires --date (ISO8601).")
        events, snapshot_ts, resp = fetch_historical_odds(
            api_key, args.date, sport=args.sport, regions=args.regions, markets=args.markets
        )
        print_quota_headers(resp)
        stamp = safe_stamp(args.date)
    else:
        events, resp = fetch_current_odds(
            api_key, sport=args.sport, regions=args.regions, markets=args.markets
        )
        print_quota_headers(resp)
        stamp = utc_now_iso()[:10].replace("-", "")  # YYYYMMDD for today

    records = process_events(events, snapshot_ts=(snapshot_ts if args.mode == "historical" else None))

    out = args.out or Path("data/raw/odds") / f"odds_{args.sport}_{stamp}.jsonl"
    write_jsonl(out, records)

    print(f"Wrote {len(records)} games to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
