#!/usr/bin/env python3
"""
scan_twitter_starters.py — Scan Twitter/X for starting pitcher confirmations.

Pulls recent tweets from team accounts and beat reporters, detects lineup
card photos using Claude Vision API, and extracts starting pitcher names.

Teams post lineup cards as IMAGES — this script downloads those images and
uses Claude's vision model to read the starting pitcher from the photo.

Env vars required:
  TWITTER_BEARER_TOKEN  — Twitter API v2 Bearer token
  ANTHROPIC_API_KEY     — For Claude vision (lineup card image reading)

Usage:
  python3 scripts/scan_twitter_starters.py --date 2026-03-31
  python3 scripts/scan_twitter_starters.py --date 2026-03-31 --teams BSB_TEXAS,BSB_LSU
  python3 scripts/scan_twitter_starters.py --date 2026-03-31 --dry-run
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests

import _bootstrap  # noqa: F401

# ── Constants ────────────────────────────────────────────────────────────────

LINEUP_KEYWORDS = [
    "lineup", "starting lineup", "game day", "gameday", "first pitch",
    "on the mound", "gets the start", "takes the bump", "probable starter",
    "starting pitcher", "gets the ball", "takes the hill", "sp:",
    "pitching matchup", "tonight's starter", "today's starter",
]

PITCHER_TEXT_PATTERNS = [
    r"(?:SP|Starting Pitcher|On the mound)[:\s]+(?:(?:RHP|LHP)\s+)?([A-Z][a-z]+ [A-Z][a-zA-Z'-]+)",
    r"([A-Z][a-z]+ [A-Z][a-zA-Z'-]+)\s+(?:gets the start|on the mound|takes the bump|gets the ball)",
    r"(?:RHP|LHP)\s+([A-Z][a-z]+ [A-Z][a-zA-Z'-]+)\s+(?:vs|@|at|starts?)",
]


def _load_env():
    """Load .env file."""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def _load_twitter_registry() -> dict[str, list[dict]]:
    """Load Twitter handles grouped by canonical_id."""
    path = Path("data/registries/twitter_ncaa_baseball.csv")
    if not path.exists():
        return {}

    by_team: dict[str, list[dict]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("canonical_id", "").startswith("#"):
                continue
            cid = row.get("canonical_id", "").strip()
            handle = row.get("handle", "").strip().lstrip("@")
            if not handle:
                continue
            entry = {
                "handle": handle,
                "account_type": row.get("account_type", ""),
                "posts_lineup_photos": row.get("posts_lineup_photos", "no") == "yes",
            }
            if cid:
                by_team.setdefault(cid, []).append(entry)
    return by_team


# ── Twitter API ──────────────────────────────────────────────────────────────

def search_twitter_v2(
    bearer_token: str,
    query: str,
    max_results: int = 20,
) -> list[dict]:
    """
    Search recent tweets using Twitter API v2.
    Returns list of tweet dicts with text, media, author info.
    """
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    params = {
        "query": query,
        "max_results": min(max_results, 100),
        "tweet.fields": "created_at,author_id,attachments,entities",
        "expansions": "attachments.media_keys,author_id",
        "media.fields": "url,preview_image_url,type",
        "user.fields": "username",
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 429:
            print("  Twitter rate limited, waiting 15s...", file=sys.stderr)
            time.sleep(15)
            resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Twitter API error: {e}", file=sys.stderr)
        return []

    # Build media lookup
    media_map = {}
    for m in data.get("includes", {}).get("media", []):
        media_map[m["media_key"]] = m

    # Build user lookup
    user_map = {}
    for u in data.get("includes", {}).get("users", []):
        user_map[u["id"]] = u.get("username", "")

    tweets = []
    for tweet in data.get("data", []):
        t = {
            "id": tweet["id"],
            "text": tweet.get("text", ""),
            "created_at": tweet.get("created_at", ""),
            "author": user_map.get(tweet.get("author_id", ""), ""),
            "images": [],
        }
        # Attach media URLs
        media_keys = tweet.get("attachments", {}).get("media_keys", [])
        for mk in media_keys:
            m = media_map.get(mk, {})
            if m.get("type") == "photo" and m.get("url"):
                t["images"].append(m["url"])
        tweets.append(t)

    return tweets


def search_team_starters(
    bearer_token: str,
    handle: str,
    team_name: str,
    date: str,
) -> list[dict]:
    """Search a team's recent tweets for lineup/starter announcements."""
    # Search FROM this account for lineup-related content
    keyword_or = " OR ".join(f'"{kw}"' for kw in LINEUP_KEYWORDS[:6])
    query = f"from:{handle} ({keyword_or})"

    tweets = search_twitter_v2(bearer_token, query, max_results=10)

    # Also search for the team name + starter keywords from anyone
    if not tweets:
        query2 = f'"{team_name}" ("starting pitcher" OR "gets the start" OR "lineup")'
        tweets = search_twitter_v2(bearer_token, query2, max_results=10)

    return tweets


# ── Image Analysis (Claude Vision) ──────────────────────────────────────────

def download_image(url: str) -> bytes | None:
    """Download an image from URL."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print(f"  Image download failed: {e}", file=sys.stderr)
        return None


def extract_starter_from_image(
    image_bytes: bytes,
    team_name: str,
    api_key: str,
) -> dict | None:
    """
    Use Claude Vision to read a lineup card photo and extract the starting pitcher.
    Returns {"pitcher_name": str, "throws": str, "confidence": str} or None.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Determine media type
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        media_type = "image/png"
    elif image_bytes[:3] == b'\xff\xd8\xff':
        media_type = "image/jpeg"
    elif image_bytes[:4] == b'RIFF':
        media_type = "image/webp"
    else:
        media_type = "image/jpeg"  # default

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"This is a baseball lineup card or game day graphic for {team_name}. "
                            "Extract ONLY the starting pitcher's name and handedness (RHP/LHP). "
                            "Respond in exactly this format:\n"
                            "PITCHER: First Last\n"
                            "THROWS: R or L\n"
                            "If you cannot determine the starting pitcher, respond: PITCHER: UNKNOWN"
                        ),
                    },
                ],
            }],
        )
        text = message.content[0].text.strip()

        pitcher = None
        throws = None
        for line in text.split("\n"):
            if line.startswith("PITCHER:"):
                pitcher = line.replace("PITCHER:", "").strip()
                if pitcher.upper() == "UNKNOWN":
                    return None
            elif line.startswith("THROWS:"):
                throws = line.replace("THROWS:", "").strip()

        if pitcher:
            return {
                "pitcher_name": pitcher,
                "throws": throws or "",
                "confidence": "high",
                "source": "twitter_image_ocr",
            }
    except Exception as e:
        print(f"  Vision API error: {e}", file=sys.stderr)

    return None


def extract_starter_from_text(tweet_text: str) -> dict | None:
    """Extract starter name from tweet text using regex."""
    for pattern in PITCHER_TEXT_PATTERNS:
        match = re.search(pattern, tweet_text)
        if match:
            name = match.group(1).strip()
            if len(name) > 4 and name.lower() not in ("the game", "head coach", "first pitch"):
                throws = ""
                if "LHP" in tweet_text[:tweet_text.find(name) + len(name) + 5]:
                    throws = "L"
                elif "RHP" in tweet_text[:tweet_text.find(name) + len(name) + 5]:
                    throws = "R"
                return {
                    "pitcher_name": name,
                    "throws": throws,
                    "confidence": "medium",
                    "source": "twitter_text",
                }
    return None


# ── Main Pipeline ────────────────────────────────────────────────────────────

def scan_starters(
    date: str,
    team_ids: list[str] | None = None,
    schedule_csv: Path | None = None,
    dry_run: bool = False,
) -> list[dict]:
    """
    Scan Twitter for starter confirmations for games on a given date.
    Returns list of confirmed starters.
    """
    _load_env()
    bearer = os.environ.get("TWITTER_BEARER_TOKEN", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not bearer:
        print("WARNING: No TWITTER_BEARER_TOKEN — skipping Twitter search.", file=sys.stderr)
        print("  Get one at https://developer.twitter.com/en/portal/projects-and-apps", file=sys.stderr)
        return []

    if not anthropic_key:
        print("WARNING: No ANTHROPIC_API_KEY — will skip lineup photo analysis.", file=sys.stderr)

    # Load team registry
    twitter_reg = _load_twitter_registry()

    # Load team name mapping
    canon = Path("data/registries/canonical_teams_2026.csv")
    cid_to_name = {}
    if canon.exists():
        import pandas as pd
        c = pd.read_csv(canon, dtype=str)
        for _, r in c.iterrows():
            cid_to_name[str(r.get("canonical_id", ""))] = str(r.get("team_name", ""))

    # Determine which teams to scan
    if team_ids:
        scan_teams = team_ids
    elif schedule_csv and schedule_csv.exists():
        import pandas as pd
        sched = pd.read_csv(schedule_csv, dtype=str)
        scan_teams = list(set(
            sched.get("home_cid", sched.get("home_canonical_id", pd.Series())).tolist() +
            sched.get("away_cid", sched.get("away_canonical_id", pd.Series())).tolist()
        ))
    else:
        scan_teams = list(twitter_reg.keys())

    scan_teams = [t for t in scan_teams if t and str(t).strip()]
    print(f"Scanning {len(scan_teams)} teams for starter confirmations...", file=sys.stderr)

    results = []
    for cid in sorted(scan_teams):
        team_name = cid_to_name.get(cid, cid)
        accounts = twitter_reg.get(cid, [])

        # Prioritize team_official accounts (they post the lineup photos)
        team_accounts = [a for a in accounts if a["account_type"] == "team_official"]
        reporter_accounts = [a for a in accounts if a["account_type"] in ("beat_reporter", "fan_media")]

        if dry_run:
            print(f"  {cid}: {len(team_accounts)} team accts, {len(reporter_accounts)} reporters", file=sys.stderr)
            continue

        found = False
        for acct in team_accounts + reporter_accounts:
            handle = acct["handle"]
            tweets = search_team_starters(bearer, handle, team_name, date)

            if not tweets:
                continue

            for tweet in tweets:
                # Check images first (lineup cards)
                if tweet["images"] and anthropic_key and acct.get("posts_lineup_photos"):
                    for img_url in tweet["images"]:
                        img_data = download_image(img_url)
                        if img_data:
                            result = extract_starter_from_image(img_data, team_name, anthropic_key)
                            if result:
                                result["canonical_id"] = cid
                                result["team_name"] = team_name
                                result["tweet_id"] = tweet["id"]
                                result["tweet_author"] = tweet["author"]
                                result["tweet_text"] = tweet["text"][:200]
                                result["image_url"] = img_url
                                results.append(result)
                                print(f"  {cid}: {result['pitcher_name']} ({result['throws']}) via {handle} image",
                                      file=sys.stderr)
                                found = True
                                break
                    if found:
                        break

                # Check text
                text_result = extract_starter_from_text(tweet["text"])
                if text_result:
                    text_result["canonical_id"] = cid
                    text_result["team_name"] = team_name
                    text_result["tweet_id"] = tweet["id"]
                    text_result["tweet_author"] = tweet["author"]
                    text_result["tweet_text"] = tweet["text"][:200]
                    text_result["image_url"] = ""
                    results.append(text_result)
                    print(f"  {cid}: {text_result['pitcher_name']} via {handle} text",
                          file=sys.stderr)
                    found = True
                    break

            if found:
                break

            time.sleep(0.5)  # rate limit

        if not found and not dry_run:
            # No confirmation found for this team
            pass

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan Twitter for starting pitcher confirmations.")
    parser.add_argument("--date", required=True, help="Game date YYYY-MM-DD")
    parser.add_argument("--teams", help="Comma-separated canonical IDs to scan (default: all in schedule)")
    parser.add_argument("--schedule", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    daily_dir = Path(f"data/daily/{args.date}")
    schedule_csv = args.schedule or daily_dir / "schedule.csv"
    out_csv = args.out or daily_dir / "twitter_starters.csv"

    team_ids = args.teams.split(",") if args.teams else None

    results = scan_starters(
        date=args.date,
        team_ids=team_ids,
        schedule_csv=schedule_csv,
        dry_run=args.dry_run,
    )

    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\nWrote {len(df)} confirmed starters → {out_csv}", file=sys.stderr)

        # Summary
        high = sum(1 for r in results if r["confidence"] == "high")
        med = sum(1 for r in results if r["confidence"] == "medium")
        print(f"  Image OCR (high confidence): {high}", file=sys.stderr)
        print(f"  Text extraction (medium): {med}", file=sys.stderr)
    else:
        print("No starters confirmed from Twitter.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
