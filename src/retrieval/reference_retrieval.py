"""
Public entry point for the reference retrieval step.

Given geographic coordinates and an event date, fetches pre-event (up to 30 days
before) and post-event (up to 7 days after) Sentinel-2 imagery together with NDVI
and NBR spectral indices derived from the same scene.
"""

from datetime import datetime, timedelta

from .gee_client import authenticate, get_sentinel_patch, _make_aoi

_INITIALIZED = False

PRE_EVENT_DAYS = 30
POST_EVENT_DAYS = 21


def _ensure_init() -> None:
    global _INITIALIZED
    if not _INITIALIZED:
        authenticate()
        _INITIALIZED = True


def fetch_reference_imagery(lat: float, lon: float, event_date: str) -> dict:
    """
    Fetch pre-event and post-event Sentinel-2 imagery with spectral indices.

    Args:
        lat:        Latitude of the incident location.
        lon:        Longitude of the incident location.
        event_date: ISO-8601 date string (YYYY-MM-DD) of the suspected incident.

    Returns:
        {
            'pre': {
                'date': str,            # actual acquisition date of the scene
                'rgb':  ndarray(H,W,3), # Sentinel-2 B4/B3/B2 reflectance
                'ndvi': ndarray(H,W),   # Sentinel-2 NDVI  [-1, 1]
                'nbr':  ndarray(H,W),   # Sentinel-2 NBR   [-1, 1]
            },
            'post': { ...same structure... }
        }

    Raises:
        ValueError: If no usable satellite scene is found within either search window.
    """
    _ensure_init()

    date = datetime.strptime(event_date, "%Y-%m-%d")
    pre_start = (date - timedelta(days=PRE_EVENT_DAYS)).strftime("%Y-%m-%d")
    pre_end = (date - timedelta(days=1)).strftime("%Y-%m-%d")
    post_start = event_date
    post_end = (date + timedelta(days=POST_EVENT_DAYS)).strftime("%Y-%m-%d")

    aoi = _make_aoi(lat, lon)

    pre = get_sentinel_patch(aoi, pre_start, pre_end)
    post = get_sentinel_patch(aoi, post_start, post_end)

    return {"pre": pre, "post": post}
