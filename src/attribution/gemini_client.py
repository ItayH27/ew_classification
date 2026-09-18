"""
Gemini client — classifies whether a single retrieved news article evidences an environmental
warfare incident at a given location/date. Called once per article by
src/attribution/predict.py's assess_attribution(), which aggregates the per-article
verdicts into a score.

Two classification modes, selected by the `mode` argument to classify_environmental_warfare():
- "yes_no" (default): a single "yes"/"no" verdict on general conflict/military-activity
  evidence, optionally including environmental damage. Domain prompt is overridable; output
  contract is fixed.
- "structured": a per-article JSON verdict (conflict_activity, environmental_destruction,
  deliberate, actor_claimed, evidence_quote, confidence) distinguishing deliberate/
  conflict-directed environmental destruction from incidental damage near fighting. Both the
  domain prompt and the output contract (the JSON field descriptions) are overridable.

Mirrors src/retrieval/gee_client.py's shape: module-level load_dotenv() as an import-time
side effect, and an EnvironmentError with an actionable message if the API key is missing.
"""

import json
import os
import random
import time
from pathlib import Path as _Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

load_dotenv(_Path(__file__).resolve().parent.parent.parent / ".env")

_MODEL = "gemini-3.1-flash-lite"

_MAX_RETRY_ATTEMPTS = 5
_BASE_RETRY_DELAY_S = 2.0
_MAX_RETRY_DELAY_S = 30.0

_YES_NO_DOMAIN_PROMPT = (
    "You are a conflict classifier. You will be given a location (latitude/longitude), a date, "
    "and the text of a single news article retrieved by searching for that location and time "
    "period. Determine whether the article provides evidence of conflict or military activity "
    "at or near that location around that date - either general evidence of fighting, military "
    "operations, or war, or more specifically, conflict-related destruction, burning, or damage "
    "to agricultural land, forests, or other ecosystems (e.g. wildfires linked to military "
    "action, deliberate burning of crops or forests, environmental damage caused by attacks or "
    "fighting). The article must specifically connect the conflict, military activity, or "
    "environmental damage to that location - a mention of the broader war or conflict in the "
    "country or region, without tying it to that location, does not count."
)

# Fixed, non-overridable in yes/no mode: classify_environmental_warfare's response parsing
# below depends on the model actually following this response-format contract, so it's always
# appended after the (possibly user-supplied) domain prompt rather than left for callers to
# override — a custom contract that omits or contradicts it would otherwise produce
# unparseable output instead of just classifying poorly.
_YES_NO_OUTPUT_CONTRACT = (
    "Respond with exactly one word, lowercase, and nothing else: \"yes\" if the article "
    "supports such an incident at that location and date, or \"no\" if it does not, is "
    "unrelated, or the evidence is insufficient."
)

_STRUCTURED_DOMAIN_PROMPT = (
    "You will be given a location (latitude/longitude), a date, and the text of one news "
    "article retrieved by searching for that location and time period.\n\n"
    "Environmental warfare here means deliberate or clearly conflict-directed destruction, "
    "burning, or damage to agricultural land, forests, or other vegetation — for example, "
    "crops or forest set on fire during a raid or offensive, farmland burned as a stated or "
    "apparent tactic, or vegetation destroyed by shelling/airstrikes in the course of fighting. "
    "It does NOT include: wildfires with no stated conflict connection, ordinary seasonal "
    "agricultural burning, damage described only to buildings/infrastructure with no "
    "vegetation mentioned, or \"scorched earth\" used as a figure of speech rather than a "
    "literal description.\n\n"
    "Read the article and judge it against the JSON object fields below. If the article is "
    "unrelated to this location/date or the evidence is insufficient, set conflict_activity "
    "and environmental_destruction to \"no\" and leave the other fields at their "
    "default/empty values."
)

# Overridable in structured mode: just the JSON object's field formatting (names/types/meaning)
# — nothing else. Everything about what counts as an incident, how to read the article, and how
# to default unrelated/insufficient-evidence articles lives in _STRUCTURED_DOMAIN_PROMPT above;
# the "respond with only JSON" framing lives in _STRUCTURED_FORMAT_INSTRUCTION below (fixed, so
# editing this contract can't break parseability).
_STRUCTURED_OUTPUT_CONTRACT = (
    "{\n"
    '  "conflict_activity": "yes" | "no",\n'
    '  "environmental_destruction": "yes" | "no",\n'
    '  "deliberate": "yes" | "no" | "unclear",\n'
    '  "actor_claimed": "<name or \'not stated\'>",\n'
    '  "evidence_quote": "<the exact sentence the judgment is based on, or empty string>",\n'
    '  "confidence": "high" | "medium" | "low"\n'
    "}"
)

# Fixed, non-overridable: same rationale as _YES_NO_OUTPUT_CONTRACT — response parsing depends
# on the model emitting nothing but the JSON object, so this framing is never left to a
# (possibly user-edited) _STRUCTURED_OUTPUT_CONTRACT to restate correctly.
_STRUCTURED_FORMAT_INSTRUCTION = (
    "Respond with exactly one JSON object, nothing else — no markdown code fences, no "
    "explanatory text before or after it."
)

_MODE_DEFAULTS = {
    "yes_no": (_YES_NO_DOMAIN_PROMPT, _YES_NO_OUTPUT_CONTRACT),
    "structured": (_STRUCTURED_DOMAIN_PROMPT, _STRUCTURED_OUTPUT_CONTRACT),
}


def get_mode_defaults(mode: str) -> tuple[str, str]:
    """Returns (default_domain_prompt, default_output_contract) for the given mode."""
    try:
        return _MODE_DEFAULTS[mode]
    except KeyError:
        raise ValueError(f"Unknown attribution mode {mode!r}; expected one of {sorted(_MODE_DEFAULTS)}.")

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Add it to your .env file:\n"
                "  GEMINI_API_KEY=your-gemini-api-key\n"
                "Get one at https://aistudio.google.com/apikey"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, genai_errors.ServerError):
        return True
    if isinstance(error, genai_errors.ClientError):
        return error.code == 429
    return False


def _generate_with_retry(client: genai.Client, **kwargs):
    """
    Calls client.models.generate_content(**kwargs), retrying with exponential backoff and
    jitter on 429 RESOURCE_EXHAUSTED and 5xx server errors — both are typically transient
    (rate limit/quota or momentary server overload) and tend to clear within seconds.
    Any other error, or the final attempt, propagates immediately.
    """
    delay = _BASE_RETRY_DELAY_S
    for attempt in range(1, _MAX_RETRY_ATTEMPTS + 1):
        try:
            return client.models.generate_content(**kwargs)
        except (genai_errors.ClientError, genai_errors.ServerError) as e:
            if not _is_retryable(e) or attempt == _MAX_RETRY_ATTEMPTS:
                raise
            sleep_s = min(delay, _MAX_RETRY_DELAY_S) + random.uniform(0, 1)
            print(
                f"Gemini {e.status or e.code} (attempt {attempt}/{_MAX_RETRY_ATTEMPTS}), "
                f"retrying in {sleep_s:.1f}s..."
            )
            time.sleep(sleep_s)
            delay *= 2


def _build_contents(article: dict, lat: float, lon: float, event_date: str) -> str:
    return (
        f"Location: {lat}, {lon}\nDate: {event_date}\n\n"
        f"Title: {article.get('title', '')}\n"
        f"Published: {article.get('publish_date', '')}\n"
        f"{article.get('text', '')}"
    )


_STRUCTURED_PARSE_FALLBACK = {
    "conflict_activity": "no",
    "environmental_destruction": "no",
    "deliberate": "unclear",
    "actor_claimed": "not stated",
    "evidence_quote": "",
    "confidence": "low",
}


def classify_environmental_warfare(
    article: dict,
    lat: float,
    lon: float,
    event_date: str,
    mode: str = "yes_no",
    domain_prompt: str | None = None,
    output_contract: str | None = None,
) -> str | dict:
    """
    Args:
        article: a single article dict as returned by worldnews_client.search_articles
            (with at least "title" and "text").
        lat, lon, event_date: same values assess_attribution() received.
        mode: "yes_no" (default) or "structured" — see module docstring constants.
        domain_prompt: overrides the mode's default domain definition (get_mode_defaults(mode)[0])
            when given — i.e. what counts as an incident, how to read the article, and (in
            structured mode) how to default unrelated/insufficient-evidence articles. Works the
            same in both modes.
        output_contract: overrides the mode's default output contract when given, but only takes
            effect in "structured" mode — yes/no mode's output contract is fixed
            (_YES_NO_OUTPUT_CONTRACT) since response parsing depends on the model actually
            answering with exactly "yes"/"no", so an output_contract argument is ignored when
            mode="yes_no". In structured mode, output_contract is just the JSON object's field
            formatting (names/types/meaning) — nothing else; the "respond with only JSON" framing
            is a separate fixed instruction (_STRUCTURED_FORMAT_INSTRUCTION) always appended after it.

    Returns:
        mode="yes_no": "yes" or "no". Falls back to "no" if the model doesn't follow the
            one-word contract.
        mode="structured": a dict parsed from the model's JSON response. Falls back to
            _STRUCTURED_PARSE_FALLBACK (plus a "parse_error" key) if the response isn't valid
            JSON. Note a custom output_contract using field names other than the default schema's
            will still parse fine, but assess_attribution's supporting-rule fields
            (environmental_destruction/deliberate) won't be present on it — such an article is
            treated as not-supporting (its raw fields are still shown, just not counted).

    Raises:
        EnvironmentError: if GEMINI_API_KEY isn't set.
        ValueError: if mode isn't "yes_no" or "structured".

    429 RESOURCE_EXHAUSTED and 5xx server errors are retried with exponential backoff (see
    _generate_with_retry) before propagating; other errors propagate immediately.
    """
    default_domain_prompt, default_output_contract = get_mode_defaults(mode)
    resolved_domain_prompt = domain_prompt if domain_prompt is not None else default_domain_prompt

    client = _get_client()
    contents = _build_contents(article, lat, lon, event_date)

    if mode == "yes_no":
        system_instruction = f"{resolved_domain_prompt}\n\n{_YES_NO_OUTPUT_CONTRACT}"
        config = types.GenerateContentConfig(system_instruction=system_instruction, temperature=0)
    else:
        resolved_output_contract = output_contract if output_contract is not None else default_output_contract
        system_instruction = (
            f"{resolved_domain_prompt}\n\n{resolved_output_contract}\n\n{_STRUCTURED_FORMAT_INSTRUCTION}"
        )
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0,
            response_mime_type="application/json",
        )

    response = _generate_with_retry(client, model=_MODEL, contents=contents, config=config)

    if mode == "yes_no":
        verdict = (response.text or "").strip().lower()
        return verdict if verdict in {"yes", "no"} else "no"

    try:
        return json.loads(response.text or "")
    except json.JSONDecodeError as e:
        return {**_STRUCTURED_PARSE_FALLBACK, "parse_error": str(e)}
