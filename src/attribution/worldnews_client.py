"""
World News API client — fetches news articles filtered by location and publish date.

Mirrors src/retrieval/gee_client.py's shape: module-level load_dotenv() as an import-time
side effect, and an EnvironmentError with an actionable message if the API key is missing.
"""

import os
from datetime import date, timedelta
from pathlib import Path as _Path

import requests
from dotenv import load_dotenv

load_dotenv(_Path(__file__).resolve().parent.parent.parent / ".env")

_SEARCH_URL = "https://api.worldnewsapi.com/search-news"
_REQUEST_TIMEOUT_S = 30  # without one, a hung connection would stall the whole pipeline run
_DEFAULT_KEYWORDS = "wildfire OR fire OR burning OR conflict OR attack OR military OR shelling OR war"
_PRE_EVENT_DAYS = 7
_POST_EVENT_DAYS = 14
# World News geotags articles to a place, so this radius is a hard filter on an already-sparse
# index: over rural areas it is the usual reason a search returns nothing. Overridable per call
# (and from the web form) so "no coverage within 10 km" can be distinguished from "no coverage".
DEFAULT_RADIUS_KM = 10


def earliest_publish_date(event_date: str) -> str:
    """Start of the publish-date filter for an event — exposed so callers can report the window
    they actually searched rather than restating the constants."""
    return (date.fromisoformat(event_date) - timedelta(days=_PRE_EVENT_DAYS)).isoformat()


def latest_publish_date(event_date: str) -> str:
    """End of the publish-date filter for an event. See earliest_publish_date."""
    return (date.fromisoformat(event_date) + timedelta(days=_POST_EVENT_DAYS)).isoformat()


def search_articles(
    lat: float,
    lon: float,
    event_date: str,
    keywords: str = _DEFAULT_KEYWORDS,
    radius_km: int = DEFAULT_RADIUS_KM,
    number: int = 5,
    offset: int = 0,
) -> list[dict]:
    """
    Search for news articles near (lat, lon) published within a window around event_date:
    from _PRE_EVENT_DAYS before to _POST_EVENT_DAYS after, covering both early reporting
    that predates on-the-ground confirmation and slower follow-up coverage.

    Args:
        lat, lon: coordinates to center the location filter on.
        event_date: "YYYY-MM-DD" — the search window is
            [event_date - _PRE_EVENT_DAYS, event_date + _POST_EVENT_DAYS].
        keywords: free-text search terms (World News API's `text` param).
        radius_km: location-filter radius in kilometres.
        number: max number of articles to return (World News API's `number` param).
        offset: how many matching articles to skip before returning results (World News
            API's `offset` param) — pass a running total of already-fetched articles to
            page through additional results.

    Returns:
        A list of article dicts as returned by the API's "news" field, each including
        (among other fields) "title", "text" (full article body), "url", "publish_date".

    Raises:
        EnvironmentError: if WORLD_NEWS_API_KEY isn't set.
        requests.HTTPError: if the API request fails (bad key, rate limit, etc.).
    """
    api_key = os.environ.get("WORLD_NEWS_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "WORLD_NEWS_API_KEY is not set. Add it to your .env file:\n"
            "  WORLD_NEWS_API_KEY=your-worldnewsapi-key\n"
            "Get one at https://worldnewsapi.com"
        )

    params = {
        "language": "en",
        "earliest-publish-date": earliest_publish_date(event_date),
        "latest-publish-date": latest_publish_date(event_date),
        "location-filter": f"{lat},{lon},{radius_km}",
        "text": keywords,
        "number": number,
        "offset": offset,
    }
    resp = requests.get(_SEARCH_URL, params=params, headers={"x-api-key": api_key}, timeout=_REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    return data.get("news", [])
