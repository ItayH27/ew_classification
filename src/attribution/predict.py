"""
Stage-4 public entry point: attribution assessment.

Searches the World News API for articles near (lat, lon) published on/after event_date,
then asks Gemini whether those articles evidence conflict or military activity at that
location/date (general conflict evidence, or more specifically conflict-caused
environmental damage). See CLAUDE.md — the design doc also calls for cross-referencing
UCDP GED conflict records, which is not implemented here; this covers only the
media-evidence half of stage 4.
"""

from __future__ import annotations

import requests

from .dedup import deduplicate_articles
from .gemini_client import classify_environmental_warfare
from .worldnews_client import (
    DEFAULT_RADIUS_KM as DEFAULT_SEARCH_RADIUS_KM,
    earliest_publish_date,
    latest_publish_date,
    search_articles,
)

_TARGET_UNIQUE_ARTICLES = 5
_MAX_FETCH_ATTEMPTS = 5  # safety cap: at most 25 articles fetched per attribution call


def _describe_search_failure(e: requests.RequestException) -> str:
    """One human-readable sentence for the `note`, naming the likely cause."""
    status = e.response.status_code if e.response is not None else None
    if status == 402:
        return "World News API daily quota exceeded (402)."
    if status in (401, 403):
        return f"World News API rejected the API key ({status}) — check WORLD_NEWS_API_KEY."
    if status is not None and status >= 500:
        return f"World News API server error ({status}), likely a temporary outage on their side."
    if status is not None:
        return f"World News API request failed ({status})."
    if isinstance(e, requests.Timeout):
        return "World News API request timed out."
    return "Could not reach the World News API."


def skipped_attribution(mode: str, note: str) -> dict:
    """An "attribution was not attempted" block, shaped exactly like a real one so callers and
    renderers need no special case. `note` says why."""
    return {
        "status": "skipped",
        "mode": mode,
        "score": None,
        "num_articles": 0,
        "num_duplicates_removed": 0,
        "yes_list": [],
        "no_list": [],
        "note": note,
    }


def assess_attribution(
    lat: float | None,
    lon: float | None,
    event_date: str | None,
    keywords: str | None = None,
    mode: str = "yes_no",
    domain_prompt: str | None = None,
    output_contract: str | None = None,
    radius_km: int | None = None,
) -> dict:
    """
    Classifies each article found for (lat, lon, event_date) independently via Gemini, then
    aggregates into `score`/`yes_list`/`no_list` so the evidence is inspectable, not just a
    single opaque verdict.

    Args:
        lat, lon, event_date: same values passed to fetch_reference_imagery(). May be
            None in local-file pipeline mode, where there's no location/date to search on.
        keywords: overrides worldnews_client's default search keywords when given.
        mode: "yes_no" (default) or "structured" — see gemini_client's module docstring.
        domain_prompt: overrides gemini_client's default domain definition when given. Works
            the same in both modes.
        output_contract: overrides gemini_client's default output contract when given, but
            only takes effect in "structured" mode (ignored in "yes_no" mode, whose output
            contract is fixed — see classify_environmental_warfare).
        radius_km: overrides the World News geographic filter radius (default
            worldnews_client.DEFAULT_RADIUS_KM). The usual reason a rural location returns no
            articles at all: coverage exists, but nothing is geotagged that close to the point.

    In "yes_no" mode, `yes_list`/`no_list` entries are bucketed by the model's plain "yes"/"no"
    verdict, each `{"title": str, "url": str}`. In "structured" mode, entries carry the full
    per-article JSON verdict alongside title/url, and are bucketed into `yes_list` iff
    `environmental_destruction == "yes"` AND `deliberate == "yes"` (a strict reading matching
    this project's definition of environmental warfare as deliberate destruction, not incidental
    damage near fighting) — articles with `deliberate == "unclear"` land in `no_list`, still
    visible with all their fields for manual review.

    Near-duplicate articles (the same underlying wire-service story republished by
    different outlets under a different title/URL) are collapsed via
    `dedup.deduplicate_articles` before classification, so `num_articles`/`score` reflect
    distinct stories rather than double/triple-counting syndicated coverage. Fetching
    paginates through the World News API (via `offset`) until `_TARGET_UNIQUE_ARTICLES`
    unique articles are collected, the API runs out of results, or `_MAX_FETCH_ATTEMPTS`
    is hit — so `num_articles` can end up below, at, or above the target depending on how
    duplicates land across pages.

    Returns:
        {"status": "skipped" | "ok" | "unavailable", "mode": str, "score": float | None, "num_articles": int,
         "num_duplicates_removed": int,
         "yes_list": [{"title": str, "url": str, ...}, ...], "no_list": [{"title": str, "url": str, ...}, ...],
         "note": str}
        `score` is None when there are no articles to score (skipped, or none found).
        `num_articles` is how many (deduplicated) articles the score/yes_list/no_list are based on.
        `num_duplicates_removed` is how many near-duplicate articles were collapsed.

    Raises:
        EnvironmentError: if WORLD_NEWS_API_KEY or GEMINI_API_KEY isn't set.
        ValueError: if mode isn't "yes_no" or "structured".

    World News API failures (quota 402, auth errors, 5xx outages, timeouts, connection errors)
    never raise: pagination stops and whatever articles were already collected are scored, with
    the cause appended to `note`. If none were collected, `status` is "unavailable" and
    `score` is None, so the rest of the pipeline's results still reach the caller.
    """
    if lat is None or lon is None or event_date is None:
        return skipped_attribution(
            mode, "Attribution requires lat, lon, and event_date; unavailable in local-file mode."
        )

    raw_articles: list[dict] = []
    articles: list[dict] = []
    num_duplicates_removed = 0
    offset = 0
    search_failure: str | None = None

    search_kwargs: dict = {}
    if keywords is not None:
        search_kwargs["keywords"] = keywords
    if radius_km is not None:
        search_kwargs["radius_km"] = radius_km
    effective_radius_km = radius_km if radius_km is not None else DEFAULT_SEARCH_RADIUS_KM
    for _ in range(_MAX_FETCH_ATTEMPTS):
        try:
            batch = search_articles(
                lat, lon, event_date, offset=offset, number=_TARGET_UNIQUE_ARTICLES, **search_kwargs
            )
        except requests.RequestException as e:
            # Any World News failure — 402 (daily quota exhausted: Free/Reporter plans return
            # 402 instead of billing overage), a 5xx outage, a timeout, a dropped connection —
            # stops paginating and uses whatever was already fetched, rather than failing the
            # whole pipeline run and losing the imagery/segmentation/severity results with it.
            search_failure = _describe_search_failure(e)
            break
        if not batch:
            break
        raw_articles.extend(batch)
        offset += len(batch)
        articles, num_duplicates_removed = deduplicate_articles(raw_articles)
        if len(articles) >= _TARGET_UNIQUE_ARTICLES or len(batch) < _TARGET_UNIQUE_ARTICLES:
            break

    if not raw_articles:
        if search_failure:
            note = f"News search unavailable: {search_failure} Attribution was not assessed."
        else:
            # Name the filters that returned nothing. A geotagged, English-only search over a
            # sparsely-reported rural area routinely finds zero articles even when the incident
            # is real and well documented elsewhere — that is an absence of indexed coverage, not
            # evidence against the event, and it reads very differently from a broken key.
            note = (
                f"The news search ran and matched nothing: no English-language articles indexed "
                f"within {effective_radius_km} km of this point between "
                f"{earliest_publish_date(event_date)} and {latest_publish_date(event_date)}. "
                "This is an absence of coverage, not evidence that no incident occurred — widen "
                "the radius or broaden the keywords to search further out."
            )
        return {
            **skipped_attribution(mode, note),
            "status": "unavailable" if search_failure else "ok",
        }

    yes_list = []
    no_list = []
    for article in articles:
        verdict = classify_environmental_warfare(
            article, lat, lon, event_date, mode=mode, domain_prompt=domain_prompt, output_contract=output_contract
        )
        if mode == "yes_no":
            entry = {"title": article.get("title"), "url": article.get("url")}
            is_supporting = verdict == "yes"
        else:
            entry = {"title": article.get("title"), "url": article.get("url"), **verdict}
            is_supporting = verdict.get("environmental_destruction") == "yes" and verdict.get("deliberate") == "yes"
        (yes_list if is_supporting else no_list).append(entry)

    num_articles = len(articles)
    score = 100.0 * len(yes_list) / num_articles
    if mode == "structured":
        note = (
            f"{score:.0f}% ({len(yes_list)} of {num_articles}) news article(s) describe deliberate "
            "environmental destruction near this location and date."
        )
    else:
        note = (
            f"{score:.0f}% ({len(yes_list)} of {num_articles}) news article(s) report conflict "
            "or military activity near this location and date."
        )
    if num_duplicates_removed:
        note += f" ({num_duplicates_removed} duplicate/syndicated article(s) collapsed before classification.)"
    if search_failure:
        note += f" (Stopped searching for more articles — {search_failure} Using what was already fetched.)"
    return {
        "status": "ok",
        "mode": mode,
        "score": score,
        "num_articles": num_articles,
        "num_duplicates_removed": num_duplicates_removed,
        "yes_list": yes_list,
        "no_list": no_list,
        "note": note,
    }
