"""
The Odds API v4 client: call every available endpoint.

Endpoints:
  - GET /v4/sports
  - GET /v4/sports/{sport}/odds
  - GET /v4/sports/{sport}/scores
  - GET /v4/sports/{sport}/events
  - GET /v4/sports/{sport}/events/{eventId}/odds
  - GET /v4/sports/{sport}/events/{eventId}/markets
  - GET /v4/sports/{sport}/participants
  - GET /v4/historical/sports/{sport}/odds
  - GET /v4/historical/sports/{sport}/events
  - GET /v4/historical/sports/{sport}/events/{eventId}/odds

All functions take api_key as first argument and return (response, data).
data is response.json() (or None on non-JSON). Check response.status_code and
response.headers (x-requests-remaining, x-requests-used, x-requests-last).
"""
from __future__ import annotations

from typing import Any

import requests

BASE_URL = "https://api.the-odds-api.com"
DEFAULT_TIMEOUT = 60


def _get(
    api_key: str,
    path: str,
    params: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[requests.Response, Any]:
    params = dict(params or {})
    params.setdefault("apiKey", api_key)
    url = f"{BASE_URL.rstrip('/')}{path}"
    resp = requests.get(url, params=params, timeout=timeout)
    try:
        data = resp.json()
    except Exception:
        data = None
    return resp, data


def get_sports(
    api_key: str,
    all_sports: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[requests.Response, Any]:
    """GET /v4/sports. No quota cost."""
    params = {} if not all_sports else {"all": "true"}
    return _get(api_key, "/v4/sports/", params=params, timeout=timeout)


def get_odds(
    api_key: str,
    sport: str,
    regions: str = "us",
    markets: str | None = None,
    odds_format: str = "american",
    date_format: str = "iso",
    event_ids: str | None = None,
    bookmakers: str | None = None,
    commence_time_from: str | None = None,
    commence_time_to: str | None = None,
    include_links: bool = False,
    include_sids: bool = False,
    include_bet_limits: bool = False,
    include_rotation_numbers: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[requests.Response, Any]:
    """GET /v4/sports/{sport}/odds. Cost: markets × regions."""
    params = {
        "regions": regions,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "includeLinks": str(include_links).lower(),
        "includeSids": str(include_sids).lower(),
        "includeBetLimits": str(include_bet_limits).lower(),
        "includeRotationNumbers": str(include_rotation_numbers).lower(),
    }
    if markets:
        params["markets"] = markets
    if event_ids:
        params["eventIds"] = event_ids
    if bookmakers:
        params["bookmakers"] = bookmakers
    if commence_time_from:
        params["commenceTimeFrom"] = commence_time_from
    if commence_time_to:
        params["commenceTimeTo"] = commence_time_to
    return _get(api_key, f"/v4/sports/{sport}/odds/", params=params, timeout=timeout)


def get_scores(
    api_key: str,
    sport: str,
    days_from: int | None = None,
    date_format: str = "iso",
    event_ids: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[requests.Response, Any]:
    """GET /v4/sports/{sport}/scores. Cost: 1 (or 2 if daysFrom set)."""
    params = {"dateFormat": date_format}
    if days_from is not None and 1 <= days_from <= 3:
        params["daysFrom"] = str(days_from)
    if event_ids:
        params["eventIds"] = event_ids
    return _get(api_key, f"/v4/sports/{sport}/scores/", params=params, timeout=timeout)


def get_events(
    api_key: str,
    sport: str,
    date_format: str = "iso",
    event_ids: str | None = None,
    commence_time_from: str | None = None,
    commence_time_to: str | None = None,
    include_rotation_numbers: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[requests.Response, Any]:
    """GET /v4/sports/{sport}/events. No quota cost."""
    params = {
        "dateFormat": date_format,
        "includeRotationNumbers": str(include_rotation_numbers).lower(),
    }
    if event_ids:
        params["eventIds"] = event_ids
    if commence_time_from:
        params["commenceTimeFrom"] = commence_time_from
    if commence_time_to:
        params["commenceTimeTo"] = commence_time_to
    return _get(api_key, f"/v4/sports/{sport}/events/", params=params, timeout=timeout)


def get_event_odds(
    api_key: str,
    sport: str,
    event_id: str,
    regions: str = "us",
    markets: str = "h2h",
    odds_format: str = "american",
    date_format: str = "iso",
    include_multipliers: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[requests.Response, Any]:
    """GET /v4/sports/{sport}/events/{eventId}/odds. Cost: markets × regions."""
    params = {
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "includeMultipliers": str(include_multipliers).lower(),
    }
    return _get(
        api_key,
        f"/v4/sports/{sport}/events/{event_id}/odds/",
        params=params,
        timeout=timeout,
    )


def get_event_markets(
    api_key: str,
    sport: str,
    event_id: str,
    regions: str = "us",
    bookmakers: str | None = None,
    date_format: str = "iso",
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[requests.Response, Any]:
    """GET /v4/sports/{sport}/events/{eventId}/markets. Cost: 1."""
    params = {"regions": regions, "dateFormat": date_format}
    if bookmakers:
        params["bookmakers"] = bookmakers
    return _get(
        api_key,
        f"/v4/sports/{sport}/events/{event_id}/markets/",
        params=params,
        timeout=timeout,
    )


def get_participants(
    api_key: str,
    sport: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[requests.Response, Any]:
    """GET /v4/sports/{sport}/participants. Cost: 1."""
    return _get(api_key, f"/v4/sports/{sport}/participants/", timeout=timeout)


def get_historical_odds(
    api_key: str,
    sport: str,
    date: str,
    regions: str = "us",
    markets: str = "h2h",
    odds_format: str = "american",
    date_format: str = "iso",
    event_ids: str | None = None,
    bookmakers: str | None = None,
    timeout: int = 120,
) -> tuple[requests.Response, Any]:
    """GET /v4/historical/sports/{sport}/odds. Cost: 10 × regions × markets."""
    params = {
        "date": date,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    if event_ids:
        params["eventIds"] = event_ids
    if bookmakers:
        params["bookmakers"] = bookmakers
    return _get(
        api_key,
        f"/v4/historical/sports/{sport}/odds/",
        params=params,
        timeout=timeout,
    )


def get_historical_events(
    api_key: str,
    sport: str,
    date: str,
    date_format: str = "iso",
    event_ids: str | None = None,
    commence_time_from: str | None = None,
    commence_time_to: str | None = None,
    include_rotation_numbers: bool = False,
    timeout: int = 60,
) -> tuple[requests.Response, Any]:
    """GET /v4/historical/sports/{sport}/events. Cost: 1 (or 0 if no events)."""
    params = {"date": date, "dateFormat": date_format}
    if event_ids:
        params["eventIds"] = event_ids
    if commence_time_from:
        params["commenceTimeFrom"] = commence_time_from
    if commence_time_to:
        params["commenceTimeTo"] = commence_time_to
    params["includeRotationNumbers"] = str(include_rotation_numbers).lower()
    return _get(
        api_key,
        f"/v4/historical/sports/{sport}/events/",
        params=params,
        timeout=timeout,
    )


def get_historical_event_odds(
    api_key: str,
    sport: str,
    event_id: str,
    date: str,
    regions: str = "us",
    markets: str = "h2h",
    odds_format: str = "american",
    date_format: str = "iso",
    include_multipliers: bool = False,
    timeout: int = 120,
) -> tuple[requests.Response, Any]:
    """GET /v4/historical/sports/{sport}/events/{eventId}/odds. Cost: 10 × markets × regions."""
    params = {
        "date": date,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "includeMultipliers": str(include_multipliers).lower(),
    }
    return _get(
        api_key,
        f"/v4/historical/sports/{sport}/events/{event_id}/odds/",
        params=params,
        timeout=timeout,
    )


def print_quota(resp: requests.Response) -> None:
    """Print x-requests-remaining, x-requests-used, x-requests-last."""
    r = resp.headers.get("x-requests-remaining")
    u = resp.headers.get("x-requests-used")
    l = resp.headers.get("x-requests-last")
    print(f"Quota: remaining={r}, used={u}, last={l}")
