"""
CLI to hit any The Odds API v4 endpoint.

Usage:
  python scripts/odds_api.py sports [--all]
  python scripts/odds_api.py odds --sport baseball_ncaa [--regions us,us2] [--markets h2h,spreads,totals]
  python scripts/odds_api.py scores --sport baseball_ncaa [--days-from 1]
  python scripts/odds_api.py events --sport baseball_ncaa
  python scripts/odds_api.py event-odds --sport baseball_ncaa --event-id <id> [--regions us] [--markets h2h]
  python scripts/odds_api.py event-markets --sport baseball_ncaa --event-id <id> [--regions us]
  python scripts/odds_api.py participants --sport baseball_ncaa
  python scripts/odds_api.py historical-odds --sport baseball_ncaa --date 2026-02-18T18:00:00Z [--regions us] [--markets h2h]
  python scripts/odds_api.py historical-events --sport baseball_ncaa --date 2026-02-18T18:00:00Z
  python scripts/odds_api.py historical-event-odds --sport baseball_ncaa --event-id <id> --date 2026-02-18T18:00:00Z [--regions us] [--markets h2h]

Set ODDS_API_KEY in .env or environment. Use --out FILE to write JSON to file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Load .env from repo root
def _load_env() -> None:
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


from odds_api_client import (
    get_event_markets,
    get_event_odds,
    get_events,
    get_historical_event_odds,
    get_historical_events,
    get_historical_odds,
    get_odds,
    get_participants,
    get_scores,
    get_sports,
    print_quota,
)


def _api_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        print("Error: ODDS_API_KEY not set. Use .env or environment.", file=sys.stderr)
        sys.exit(1)
    return key


def _output(data: object, out: Path | None) -> None:
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(json.dumps(data, indent=2, sort_keys=True))


def cmd_sports(args: argparse.Namespace) -> int:
    resp, data = get_sports(_api_key(), all_sports=args.all)
    print_quota(resp)
    if resp.status_code != 200:
        print(resp.text, file=sys.stderr)
        return 1
    _output(data or [], args.out)
    return 0


def cmd_odds(args: argparse.Namespace) -> int:
    resp, data = get_odds(
        _api_key(),
        sport=args.sport,
        regions=args.regions,
        markets=args.markets or None,
        odds_format=args.odds_format,
        event_ids=args.event_ids or None,
        bookmakers=args.bookmakers or None,
        commence_time_from=args.commence_time_from or None,
        commence_time_to=args.commence_time_to or None,
    )
    print_quota(resp)
    if resp.status_code != 200:
        print(resp.text, file=sys.stderr)
        return 1
    _output(data if data is not None else [], args.out)
    return 0


def cmd_scores(args: argparse.Namespace) -> int:
    resp, data = get_scores(
        _api_key(),
        sport=args.sport,
        days_from=args.days_from,
        event_ids=args.event_ids or None,
    )
    print_quota(resp)
    if resp.status_code != 200:
        print(resp.text, file=sys.stderr)
        return 1
    _output(data if data is not None else [], args.out)
    return 0


def cmd_events(args: argparse.Namespace) -> int:
    resp, data = get_events(
        _api_key(),
        sport=args.sport,
        event_ids=args.event_ids or None,
        commence_time_from=args.commence_time_from or None,
        commence_time_to=args.commence_time_to or None,
    )
    print_quota(resp)
    if resp.status_code != 200:
        print(resp.text, file=sys.stderr)
        return 1
    _output(data if data is not None else [], args.out)
    return 0


def cmd_event_odds(args: argparse.Namespace) -> int:
    resp, data = get_event_odds(
        _api_key(),
        sport=args.sport,
        event_id=args.event_id,
        regions=args.regions,
        markets=args.markets,
        odds_format=args.odds_format,
    )
    print_quota(resp)
    if resp.status_code != 200:
        print(resp.text, file=sys.stderr)
        return 1
    _output(data if data is not None else {}, args.out)
    return 0


def cmd_event_markets(args: argparse.Namespace) -> int:
    resp, data = get_event_markets(
        _api_key(),
        sport=args.sport,
        event_id=args.event_id,
        regions=args.regions,
        bookmakers=args.bookmakers or None,
    )
    print_quota(resp)
    if resp.status_code != 200:
        print(resp.text, file=sys.stderr)
        return 1
    _output(data if data is not None else {}, args.out)
    return 0


def cmd_participants(args: argparse.Namespace) -> int:
    resp, data = get_participants(_api_key(), sport=args.sport)
    print_quota(resp)
    if resp.status_code != 200:
        print(resp.text, file=sys.stderr)
        return 1
    _output(data if data is not None else [], args.out)
    return 0


def cmd_historical_odds(args: argparse.Namespace) -> int:
    resp, data = get_historical_odds(
        _api_key(),
        sport=args.sport,
        date=args.date,
        regions=args.regions,
        markets=args.markets,
        odds_format=args.odds_format,
        event_ids=args.event_ids or None,
    )
    print_quota(resp)
    if resp.status_code != 200:
        print(resp.text, file=sys.stderr)
        return 1
    _output(data if data is not None else {}, args.out)
    return 0


def cmd_historical_events(args: argparse.Namespace) -> int:
    resp, data = get_historical_events(
        _api_key(),
        sport=args.sport,
        date=args.date,
        event_ids=args.event_ids or None,
    )
    print_quota(resp)
    if resp.status_code != 200:
        print(resp.text, file=sys.stderr)
        return 1
    _output(data if data is not None else {}, args.out)
    return 0


def cmd_historical_event_odds(args: argparse.Namespace) -> int:
    resp, data = get_historical_event_odds(
        _api_key(),
        sport=args.sport,
        event_id=args.event_id,
        date=args.date,
        regions=args.regions,
        markets=args.markets,
        odds_format=args.odds_format,
    )
    print_quota(resp)
    if resp.status_code != 200:
        print(resp.text, file=sys.stderr)
        return 1
    _output(data if data is not None else {}, args.out)
    return 0


def _add_common(
    p: argparse.ArgumentParser,
    sport: bool = False,
    out: bool = True,
    regions: bool = False,
    markets: bool = False,
) -> None:
    if sport:
        p.add_argument("--sport", default="baseball_ncaa", help="Sport key")
    if out:
        p.add_argument("--out", type=Path, default=None, help="Write JSON to file")
    if regions:
        p.add_argument("--regions", default="us", help="Comma-separated regions")
    if markets:
        p.add_argument("--markets", default="h2h", help="Comma-separated markets")
    p.add_argument("--odds-format", default="american", choices=["american", "decimal"])
    p.add_argument("--event-ids", default=None, help="Comma-separated event ids")
    p.add_argument("--commence-time-from", default=None, help="ISO8601")
    p.add_argument("--commence-time-to", default=None, help="ISO8601")


def main() -> int:
    _load_env()
    parser = argparse.ArgumentParser(description="The Odds API v4 – hit any endpoint.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # sports
    p = sub.add_parser("sports", help="GET /v4/sports")
    p.add_argument("--all", action="store_true", help="Include out-of-season sports")
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(run=cmd_sports)

    # odds
    p = sub.add_parser("odds", help="GET /v4/sports/{sport}/odds")
    _add_common(p, sport=True, regions=True, markets=True)
    p.add_argument("--bookmakers", default=None, help="Comma-separated bookmaker keys")
    p.set_defaults(run=cmd_odds)

    # scores
    p = sub.add_parser("scores", help="GET /v4/sports/{sport}/scores")
    _add_common(p, sport=True)
    p.add_argument("--days-from", type=int, default=None, choices=[1, 2, 3], help="Include completed games from last N days")
    p.set_defaults(run=cmd_scores)

    # events
    p = sub.add_parser("events", help="GET /v4/sports/{sport}/events")
    _add_common(p, sport=True)
    p.set_defaults(run=cmd_events)

    # event-odds
    p = sub.add_parser("event-odds", help="GET /v4/sports/{sport}/events/{eventId}/odds")
    _add_common(p, sport=True, regions=True, markets=True)
    p.add_argument("--event-id", required=True, help="Event id")
    p.set_defaults(run=cmd_event_odds)

    # event-markets
    p = sub.add_parser("event-markets", help="GET /v4/sports/{sport}/events/{eventId}/markets")
    _add_common(p, sport=True, regions=True, markets=False)
    p.add_argument("--event-id", required=True, help="Event id")
    p.add_argument("--bookmakers", default=None, help="Comma-separated bookmaker keys")
    p.set_defaults(run=cmd_event_markets)

    # participants
    p = sub.add_parser("participants", help="GET /v4/sports/{sport}/participants")
    _add_common(p, sport=True)
    p.set_defaults(run=cmd_participants)

    # historical-odds
    p = sub.add_parser("historical-odds", help="GET /v4/historical/sports/{sport}/odds")
    _add_common(p, sport=True, regions=True, markets=True)
    p.add_argument("--date", required=True, help="ISO8601 snapshot time")
    p.set_defaults(run=cmd_historical_odds)

    # historical-events
    p = sub.add_parser("historical-events", help="GET /v4/historical/sports/{sport}/events")
    _add_common(p, sport=True)
    p.add_argument("--date", required=True, help="ISO8601 snapshot time")
    p.set_defaults(run=cmd_historical_events)

    # historical-event-odds
    p = sub.add_parser("historical-event-odds", help="GET /v4/historical/sports/{sport}/events/{eventId}/odds")
    _add_common(p, sport=True, regions=True, markets=True)
    p.add_argument("--event-id", required=True, help="Event id")
    p.add_argument("--date", required=True, help="ISO8601 snapshot time")
    p.set_defaults(run=cmd_historical_event_odds)

    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
