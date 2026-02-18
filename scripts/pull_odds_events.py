from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests

from io_utils import safe_stamp, utc_now_iso, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch The Odds API historical events list (raw JSON).")
    parser.add_argument("--sport", default="baseball_ncaa", help="Sport key (e.g., baseball_ncaa)")
    parser.add_argument(
        "--date",
        required=True,
        help='Historical timestamp ISO8601, e.g. "2025-05-18T16:00:00Z"',
    )
    parser.add_argument("--base-url", default="https://api.the-odds-api.com", help="API base URL")
    parser.add_argument("--out", type=Path, default=None, help="Optional output path for body JSON")
    args = parser.parse_args()

    api_key = os.environ.get("ODDS_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing ODDS_API_KEY env var.")

    url = f"{args.base_url.rstrip('/')}/v4/historical/sports/{args.sport}/events"
    params = {"apiKey": api_key, "date": args.date}
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()

    body = resp.json()
    out = args.out or (Path("data/raw/odds/historical") / f"events_{args.sport}_{safe_stamp(args.date)}.json")
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
