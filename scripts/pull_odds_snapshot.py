from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests

from io_utils import safe_stamp, utc_now_iso, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch The Odds API historical odds snapshot (raw JSON).")
    parser.add_argument("--sport", default="baseball_ncaa", help="Sport key (e.g., baseball_ncaa)")
    parser.add_argument(
        "--date",
        required=True,
        help='Historical timestamp ISO8601, e.g. "2025-05-18T16:00:00Z"',
    )
    parser.add_argument("--regions", default="us", help="Comma-separated regions (default: us)")
    parser.add_argument("--markets", default="h2h,spreads,totals", help="Comma-separated markets")
    parser.add_argument("--odds-format", default="american", choices=["american", "decimal"], help="Odds format")
    parser.add_argument("--date-format", default="iso", choices=["iso", "unix"], help="Date format in response")
    parser.add_argument("--base-url", default="https://api.the-odds-api.com", help="API base URL")
    parser.add_argument("--out", type=Path, default=None, help="Optional output path for body JSON")
    args = parser.parse_args()

    api_key = os.environ.get("ODDS_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing ODDS_API_KEY env var.")

    url = f"{args.base_url.rstrip('/')}/v4/historical/sports/{args.sport}/odds"
    params = {
        "apiKey": api_key,
        "date": args.date,
        "regions": args.regions,
        "markets": args.markets,
        "oddsFormat": args.odds_format,
        "dateFormat": args.date_format,
    }
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()

    body = resp.json()
    out = args.out or (
        Path("data/raw/odds/historical")
        / f"odds_{args.sport}_{safe_stamp(args.date)}_{args.regions}_{args.markets.replace(',', '-')}.json"
    )
    meta = out.with_suffix(".meta.json")

    write_json(out, body)
    write_json(
        meta,
        {
            "fetched_at": utc_now_iso(),
            "request": {"url": url, "params": {k: v for k, v in params.items() if k != "apiKey"}},
            "response_headers": dict(resp.headers),
            "status_code": resp.status_code,
        },
    )

    n = len(body.get("data", [])) if isinstance(body, dict) and isinstance(body.get("data"), list) else None
    print(f"Wrote {out}" + (f" ({n} events)" if n is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
