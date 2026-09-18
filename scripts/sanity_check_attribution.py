"""
End-to-end sanity check for the attribution stage (World News API + Gemini).

Runs, in order:
    1. World News search   — search_articles() for a known damaged location/date
    2. Gemini classify     — classify_environmental_warfare() on synthetic yes/no articles
    3. Full assess_attribution — the public entry point, both GEE-mode and local-mode
    4. Dedup                — deduplicate_articles() on synthetic near-duplicate articles
    5. Gemini retry         — _generate_with_retry() on synthetic 429/5xx errors
    6. Structured classify  — classify_environmental_warfare(mode="structured")
    7. Structured assess_attribution — assess_attribution(mode="structured")

Requires WORLD_NEWS_API_KEY and GEMINI_API_KEY set in .env (see env.example).

Usage:
    python scripts/sanity_check_attribution.py            # run all steps in order
    python scripts/sanity_check_attribution.py --step 2    # run only step 2
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Known burned/damaged area: Kherson region, Ukraine — large agricultural fires
# (same location used by scripts/visualize_retrieval.py). EVENT_DATE must stay within the
# last 30 days: the World News API's free/starter plan rejects an earliest-publish-date
# further back than that (see search_articles's Raises docstring).
LAT, LON = 48.9001, 37.7249
EVENT_DATE = "2026-07-15"


def step1_worldnews_search() -> None:
    print("\n=== Step 1: World News API search ===")
    from src.attribution.worldnews_client import search_articles

    articles = search_articles(LAT, LON, EVENT_DATE)
    print(f"Got {len(articles)} article(s)")
    assert len(articles) <= 5, f"expected at most 5 articles, got {len(articles)}"
    for a in articles:
        assert a.get("title"), "article missing title"
        assert a.get("text"), "article missing text"
        assert a.get("url"), "article missing url"
        print(f"  - {a['title']!r} ({a.get('publish_date')})")
    print("Step 1: OK")


def step2_gemini_classify() -> None:
    print("\n=== Step 2: Gemini classification (synthetic articles) ===")
    from src.attribution.gemini_client import classify_environmental_warfare

    yes_article = {
        "title": "Wildfires ravage farmland amid shelling near Kherson",
        "text": (
            "Local officials said artillery strikes ignited large wildfires that burned "
            "thousands of hectares of wheat fields and forest near the front line this week, "
            "in what residents describe as deliberate destruction of agricultural land."
        ),
        "publish_date": EVENT_DATE,
    }
    no_article = {
        "title": "City council approves new park budget",
        "text": "The city council voted unanimously to increase funding for public parks and playgrounds.",
        "publish_date": EVENT_DATE,
    }

    yes_verdict = classify_environmental_warfare(yes_article, LAT, LON, EVENT_DATE)
    no_verdict = classify_environmental_warfare(no_article, LAT, LON, EVENT_DATE)
    print(f"yes_article -> {yes_verdict!r}, no_article -> {no_verdict!r}")
    assert yes_verdict == "yes", f"expected 'yes', got {yes_verdict!r}"
    assert no_verdict == "no", f"expected 'no', got {no_verdict!r}"
    print("Step 2: OK")


def step3_full_assess_attribution() -> None:
    print("\n=== Step 3: Full assess_attribution() ===")
    from src.attribution.predict import assess_attribution
    from src.attribution.worldnews_client import search_articles

    skipped = assess_attribution(None, None, None)
    print("local-mode result:", skipped)
    assert skipped["status"] == "skipped"
    assert skipped["score"] is None
    assert skipped["num_articles"] == 0
    assert skipped["yes_list"] == [] and skipped["no_list"] == []

    result = assess_attribution(LAT, LON, EVENT_DATE)
    print("GEE-mode result:", result)
    assert result["status"] == "ok"
    # assess_attribution() now paginates the World News API (via offset) until it collects
    # 5 unique (post-dedup) articles or exhausts the source, so num_articles/duplicates can
    # span more than one 5-article page — no fixed upper bound to assert here anymore.
    assert result["num_duplicates_removed"] >= 0
    total = len(result["yes_list"]) + len(result["no_list"])
    assert total == result["num_articles"], f"yes_list+no_list ({total}) != num_articles ({result['num_articles']})"
    if total == 0:
        assert result["score"] is None
    else:
        assert 0.0 <= result["score"] <= 100.0
        assert abs(result["score"] - 100.0 * len(result["yes_list"]) / total) < 1e-6
    for a in result["yes_list"] + result["no_list"]:
        assert a.get("title") and a.get("url"), f"entry missing title/url: {a}"
    yes_urls = {a["url"] for a in result["yes_list"]}
    no_urls = {a["url"] for a in result["no_list"]}
    assert yes_urls.isdisjoint(no_urls)
    print("Step 3: OK")


def step4_dedup_near_duplicates() -> None:
    print("\n=== Step 4: deduplicate_articles() on synthetic near-duplicates ===")
    from src.attribution.dedup import deduplicate_articles

    wire_body = (
        "Local officials said artillery strikes ignited large wildfires that burned "
        "thousands of hectares of wheat fields and forest near the front line this week, "
        "in what residents describe as deliberate destruction of agricultural land. "
        "Emergency crews have been deployed to the region to contain the blaze, officials "
        "said, as the fires continue to threaten nearby villages and farmland."
    )
    article_a = {
        "title": "Wildfires ravage farmland amid shelling near Kherson",
        "text": wire_body,
        "url": "https://outlet-a.example/story1",
    }
    article_b = {  # same wire copy, lightly reworded by a different outlet
        "title": "Massive blazes tear through Kherson farmland after shelling",
        "text": wire_body.replace("Local officials", "Regional authorities").replace("this week", "on Tuesday"),
        "url": "https://outlet-b.example/story2",
    }
    distinct_article = {
        "title": "City council approves new park budget",
        "text": (
            "The city council voted unanimously to increase funding for public parks and "
            "playgrounds, part of a broader plan to revitalize green spaces across residential "
            "neighborhoods."
        ),
        "url": "https://outlet-c.example/story3",
    }

    deduped, num_removed = deduplicate_articles([article_a, article_b, distinct_article])
    print(f"deduped -> {len(deduped)} article(s), num_duplicates_removed={num_removed}")
    assert num_removed == 1, f"expected 1 duplicate collapsed, got {num_removed}"
    assert len(deduped) == 2, f"expected 2 surviving articles, got {len(deduped)}"
    deduped_urls = {a["url"] for a in deduped}
    assert article_a["url"] in deduped_urls, "representative of the near-duplicate cluster should survive"
    assert distinct_article["url"] in deduped_urls, "clearly distinct article should survive"

    # Edge cases
    empty_deduped, empty_removed = deduplicate_articles([])
    assert empty_deduped == [] and empty_removed == 0
    single_deduped, single_removed = deduplicate_articles([article_a])
    assert single_deduped == [article_a] and single_removed == 0
    all_dupes = [article_a, article_b, {**article_a, "url": "https://outlet-d.example/story4"}]
    all_deduped, all_removed = deduplicate_articles(all_dupes)
    assert len(all_deduped) == 1 and all_removed == 2, "fully mutual duplicates should collapse to one"

    print("Step 4: OK")


def step5_gemini_retry() -> None:
    print("\n=== Step 5: _generate_with_retry() on synthetic 429/5xx errors ===")
    from unittest.mock import MagicMock

    from google.genai import errors as genai_errors

    from src.attribution import gemini_client

    real_sleep = gemini_client.time.sleep
    gemini_client.time.sleep = lambda _seconds: None
    try:
        # Case 1: two retryable 429s then success -> retries, then returns the response.
        success_response = MagicMock(text="yes")
        flaky_client = MagicMock()
        flaky_client.models.generate_content.side_effect = [
            genai_errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}),
            genai_errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE"}}),
            success_response,
        ]
        result = gemini_client._generate_with_retry(flaky_client, model="x", contents="y")
        assert result is success_response, "expected the eventual successful response"
        assert flaky_client.models.generate_content.call_count == 3, (
            f"expected 3 calls (2 failures + 1 success), got {flaky_client.models.generate_content.call_count}"
        )
        print("  retry-then-succeed: OK")

        # Case 2: non-retryable 400 -> propagates immediately, no retry.
        bad_request_client = MagicMock()
        bad_request_client.models.generate_content.side_effect = genai_errors.ClientError(
            400, {"error": {"code": 400, "status": "INVALID_ARGUMENT"}}
        )
        try:
            gemini_client._generate_with_retry(bad_request_client, model="x", contents="y")
            raise AssertionError("expected ClientError(400) to propagate")
        except genai_errors.ClientError as e:
            assert e.code == 400
        assert bad_request_client.models.generate_content.call_count == 1, (
            "non-retryable error should not be retried"
        )
        print("  non-retryable fails fast: OK")

        # Case 3: always-429 -> exhausts retries, still raises (doesn't swallow the error).
        always_429_client = MagicMock()
        always_429_client.models.generate_content.side_effect = genai_errors.ClientError(
            429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}
        )
        try:
            gemini_client._generate_with_retry(always_429_client, model="x", contents="y")
            raise AssertionError("expected ClientError(429) to propagate after exhausting retries")
        except genai_errors.ClientError as e:
            assert e.code == 429
        assert always_429_client.models.generate_content.call_count == gemini_client._MAX_RETRY_ATTEMPTS, (
            f"expected exactly {gemini_client._MAX_RETRY_ATTEMPTS} calls, "
            f"got {always_429_client.models.generate_content.call_count}"
        )
        print("  exhausted retries still raises: OK")
    finally:
        gemini_client.time.sleep = real_sleep

    print("Step 5: OK")


def step6_structured_classify() -> None:
    print("\n=== Step 6: Gemini structured classification (synthetic articles) ===")
    from src.attribution.gemini_client import classify_environmental_warfare

    deliberate_article = {
        "title": "Troops set fire to farmland during offensive near Kherson",
        "text": (
            "Soldiers deliberately set fire to wheat fields and forest during an offensive "
            "near the front line this week, in what residents describe as a scorched-earth "
            "tactic to deny cover and resources to the opposing side. 'They torched the "
            "fields on purpose as they advanced,' one farmer said."
        ),
        "publish_date": EVENT_DATE,
    }
    wildfire_article = {
        "title": "Wildfire spreads near Kherson amid dry conditions",
        "text": (
            "A wildfire broke out near Kherson this week amid unusually dry conditions, "
            "burning several hundred hectares of grassland. No connection to military "
            "activity has been reported by local officials."
        ),
        "publish_date": EVENT_DATE,
    }
    unrelated_article = {
        "title": "City council approves new park budget",
        "text": "The city council voted unanimously to increase funding for public parks and playgrounds.",
        "publish_date": EVENT_DATE,
    }

    for label, article in [
        ("deliberate_article", deliberate_article),
        ("wildfire_article", wildfire_article),
        ("unrelated_article", unrelated_article),
    ]:
        verdict = classify_environmental_warfare(article, LAT, LON, EVENT_DATE, mode="structured")
        print(f"{label} -> {verdict}")
        for key in ("conflict_activity", "environmental_destruction", "deliberate", "actor_claimed", "evidence_quote", "confidence"):
            assert key in verdict, f"{label}: missing key {key!r} in {verdict}"
        assert verdict["conflict_activity"] in {"yes", "no"}, f"{label}: bad conflict_activity {verdict['conflict_activity']!r}"
        assert verdict["environmental_destruction"] in {"yes", "no"}, (
            f"{label}: bad environmental_destruction {verdict['environmental_destruction']!r}"
        )
        # When conflict_activity/environmental_destruction are "no", the spec's "leave the
        # other fields at their default/empty values" instruction is satisfied by the model
        # with "" rather than the literal string "unclear" — both are valid here.
        assert verdict["deliberate"] in {"yes", "no", "unclear", ""}, f"{label}: bad deliberate {verdict['deliberate']!r}"
        assert verdict["confidence"] in {"high", "medium", "low", ""}, f"{label}: bad confidence {verdict['confidence']!r}"

    print("Step 6: OK")


def step7_structured_assess_attribution() -> None:
    print("\n=== Step 7: Full assess_attribution(mode=\"structured\") ===")
    from src.attribution.predict import assess_attribution

    result = assess_attribution(LAT, LON, EVENT_DATE, mode="structured")
    print("structured-mode result:", result)
    assert result["status"] == "ok"
    assert result["mode"] == "structured"
    total = len(result["yes_list"]) + len(result["no_list"])
    assert total == result["num_articles"], f"yes_list+no_list ({total}) != num_articles ({result['num_articles']})"
    if total == 0:
        assert result["score"] is None
    else:
        assert 0.0 <= result["score"] <= 100.0
    for a in result["yes_list"] + result["no_list"]:
        assert a.get("title") and a.get("url"), f"entry missing title/url: {a}"
        for key in ("conflict_activity", "environmental_destruction", "deliberate", "confidence"):
            assert key in a, f"structured entry missing key {key!r}: {a}"
    for a in result["yes_list"]:
        assert a["environmental_destruction"] == "yes" and a["deliberate"] == "yes", (
            f"yes_list entry doesn't satisfy the supporting rule: {a}"
        )
    yes_urls = {a["url"] for a in result["yes_list"]}
    no_urls = {a["url"] for a in result["no_list"]}
    assert yes_urls.isdisjoint(no_urls)
    print("Step 7: OK")


STEPS = {
    1: step1_worldnews_search,
    2: step2_gemini_classify,
    3: step3_full_assess_attribution,
    4: step4_dedup_near_duplicates,
    5: step5_gemini_retry,
    6: step6_structured_classify,
    7: step7_structured_assess_attribution,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one or all steps of the attribution stage sanity check.")
    p.add_argument(
        "--step", type=int, choices=sorted(STEPS), default=None,
        help=f"Run only this step ({min(STEPS)}-{max(STEPS)}). Omit to run all steps in order.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.step is not None:
        STEPS[args.step]()
    else:
        for step_fn in STEPS.values():
            step_fn()
        print("\nAll sanity checks passed.")
