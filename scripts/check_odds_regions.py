"""One-off: check which regions return bookmakers for NCAA baseball."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# load .env
repo = Path(__file__).resolve().parent.parent
env_path = repo / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

import requests

BASE = "https://api.the-odds-api.com"
SPORT = "baseball_ncaa"


def main() -> int:
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        print("Missing ODDS_API_KEY", file=sys.stderr)
        return 1
    for region in ["uk", "eu", "au"]:
        r = requests.get(
            f"{BASE}/v4/sports/{SPORT}/odds",
            params={"apiKey": key, "regions": region, "markets": "h2h", "oddsFormat": "american"},
            timeout=30,
        )
        print(f"Region {region}: status={r.status_code}, remaining={r.headers.get('x-requests-remaining')}")
        if r.status_code != 200:
            print(r.text[:400])
            continue
        data = r.json()
        events = data if isinstance(data, list) else data.get("data", [])
        books = set()
        for ev in events:
            for b in ev.get("bookmakers", []):
                books.add(b.get("key", ""))
        print(f"  Events: {len(events)}, Bookmakers: {sorted(books) or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
