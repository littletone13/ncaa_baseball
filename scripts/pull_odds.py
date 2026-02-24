"""
Odds data pipeline for NCAA baseball using the-odds-api.

Fetches current or historical odds, computes devigged fair probabilities from h2h,
and writes one JSONL record per game to data/raw/odds/.

Historical (one snapshot per day to limit calls):
  - Cost: 10 × regions × markets per snapshot (e.g. us+us2, h2h+spreads+totals = 60/call).
  - Use --from / --to to pull one snapshot per day in that range (default snapshot time 18:00 UTC).
  - Single day: --mode historical --date YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# CST = UTC-6 (no DST in winter; used for snapshot-times-cst)
CST = timezone(timedelta(hours=-6))

import requests

from io_utils import safe_stamp, utc_now_iso

BASE_URL = "https://api.the-odds-api.com"
SPORT = "baseball_ncaa"
CURRENT_REGIONS = "uk,us,us2,us_dfs,us_ex,eu,au"
CURRENT_MARKETS = "h2h,spreads,totals"
# One snapshot per day at this UTC time (keeps historical call count = 1 per day)
HISTORICAL_SNAPSHOT_TIME_UTC = "18:00:00"


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


def decimal_to_implied(decimal: float) -> float:
    """Convert decimal odds to implied probability (0-1)."""
    if decimal <= 0:
        return 0.0
    return 1.0 / decimal


def price_to_implied(price: int | float, odds_format: str = "american") -> float:
    """Convert price to implied probability. odds_format: american | decimal."""
    if odds_format == "decimal" or (isinstance(price, float) and 0 < price < 100 and price != round(price)):
        return decimal_to_implied(price)
    return american_to_implied(price)


def devig_h2h(
    outcomes: list[dict[str, Any]],
    home_team: str,
    away_team: str,
    odds_format: str = "american",
) -> tuple[float, float] | None:
    """
    Multiplicative devig for a single book's h2h outcomes.
    Returns (fair_home, fair_away) or None if outcomes cannot be matched.
    Supports american and decimal odds.
    """
    by_team: dict[str, float] = {}
    for o in outcomes:
        name = o.get("name")
        price = o.get("price")
        if name is None or price is None:
            continue
        by_team[name] = price_to_implied(price, odds_format)
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
                devigged = devig_h2h(outcomes, home_team, away_team, odds_format="american")
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
        help='Historical snapshot date ISO8601 or YYYY-MM-DD (used with --mode historical if no range)',
    )
    parser.add_argument(
        "--from",
        dest="from_date",
        metavar="FROM",
        help="Start of date range (YYYY-MM-DD). With --to, pull one snapshot per day. Saves calls.",
    )
    parser.add_argument(
        "--to",
        dest="to_date",
        metavar="TO",
        help="End of date range (YYYY-MM-DD). Use with --from.",
    )
    parser.add_argument(
        "--snapshot-times-cst",
        metavar="TIMES",
        help="Comma-separated times in CST for historical range, e.g. 8:00,9:00,9:30,14:00,19:00,20:00. Multiple snapshots per day.",
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
        if args.from_date and args.to_date:
            from_d = date.fromisoformat(args.from_date)
            to_d = date.fromisoformat(args.to_date)
            if from_d > to_d:
                raise SystemExit("--from must be <= --to")

            # Parse optional CST times (e.g. "8:00,9:00,9:30,14:00,19:00,20:00")
            if args.snapshot_times_cst:
                times_cst: list[tuple[int, int]] = []
                for part in args.snapshot_times_cst.split(","):
                    part = part.strip()
                    if ":" in part:
                        h, m = part.split(":", 1)
                        times_cst.append((int(h.strip()), int(m.strip())))
                    else:
                        times_cst.append((int(part), 0))
                snapshot_times_per_day = [
                    (h, m) for h, m in times_cst
                ]
            else:
                # Single time per day: 18:00 UTC
                snapshot_times_per_day = [(18, 0)]  # UTC hour, minute

            all_records: list[dict[str, Any]] = []
            day = from_d
            while day <= to_d:
                if args.snapshot_times_cst:
                    for hour_cst, minute_cst in snapshot_times_per_day:
                        dt_cst = datetime(
                            day.year, day.month, day.day,
                            hour_cst, minute_cst, 0, tzinfo=CST,
                        )
                        dt_utc = dt_cst.astimezone(timezone.utc)
                        snapshot_iso = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                        events, snapshot_ts, resp = fetch_historical_odds(
                            api_key,
                            snapshot_iso,
                            sport=args.sport,
                            regions=args.regions,
                            markets=args.markets,
                        )
                        print_quota_headers(resp)
                        recs = process_events(events, snapshot_ts=snapshot_ts)
                        all_records.extend(recs)
                        print(f"  {day} {hour_cst}:{minute_cst:02d} CST -> {snapshot_ts or snapshot_iso}: {len(recs)} games")
                else:
                    snapshot_iso = f"{day.isoformat()}T{HISTORICAL_SNAPSHOT_TIME_UTC}Z"
                    events, snapshot_ts, resp = fetch_historical_odds(
                        api_key,
                        snapshot_iso,
                        sport=args.sport,
                        regions=args.regions,
                        markets=args.markets,
                    )
                    print_quota_headers(resp)
                    recs = process_events(events, snapshot_ts=snapshot_ts)
                    all_records.extend(recs)
                    print(f"  {day}: {len(recs)} games")
                day += timedelta(days=1)
            records = all_records
            stamp = f"{safe_stamp(args.from_date)}_to_{safe_stamp(args.to_date)}"
        elif args.date:
            date_str = args.date.strip()
            if len(date_str) == 10 and date_str[4] == "-":
                snapshot_iso = f"{date_str}T{HISTORICAL_SNAPSHOT_TIME_UTC}Z"
            else:
                snapshot_iso = date_str
            events, snapshot_ts, resp = fetch_historical_odds(
                api_key, snapshot_iso, sport=args.sport, regions=args.regions, markets=args.markets
            )
            print_quota_headers(resp)
            records = process_events(events, snapshot_ts=snapshot_ts)
            stamp = safe_stamp(snapshot_iso)
        else:
            raise SystemExit("Historical mode needs --date or both --from and --to.")
    else:
        events, resp = fetch_current_odds(
            api_key, sport=args.sport, regions=args.regions, markets=args.markets
        )
        print_quota_headers(resp)
        records = process_events(events, snapshot_ts=None)
        stamp = utc_now_iso()[:10].replace("-", "")

    out = args.out or Path("data/raw/odds") / f"odds_{args.sport}_{stamp}.jsonl"
    write_jsonl(out, records)

    print(f"Wrote {len(records)} games to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
