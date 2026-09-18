"""
ESA WorldCover land-cover filter for negative-sample point selection only —
never used by the event-driven retrieval path (a real damage report can be
anywhere, vegetated or not; this filter exists purely to keep bulk-sourced
"benign" training patches on vegetated land, where an absent burn scar is a
meaningful negative rather than a trivial one like open water or urban).
"""

import os
from pathlib import Path as _Path

import ee
from dotenv import load_dotenv

load_dotenv(_Path(__file__).resolve().parent.parent.parent / ".env")

# ESA WorldCover v200 class codes: Tree cover, Shrubland, Grassland, Cropland.
_VEGETATED_CLASSES = [10, 20, 30, 40]
_CHECK_RADIUS_M = 500

_INITIALIZED = False


def _ensure_init() -> None:
    global _INITIALIZED
    if not _INITIALIZED:
        from .gee_client import authenticate
        authenticate()
        _INITIALIZED = True


def is_vegetated_point(lat: float, lon: float) -> bool:
    """
    True if the majority land-cover class within _CHECK_RADIUS_M of (lat, lon)
    is one of _VEGETATED_CLASSES.

    Note: ee.Image.remap() renames its output band to "remapped", not the
    source band name — the result must be re-read under that key, not
    "Map" (WorldCover's native band name).
    """
    _ensure_init()

    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(_CHECK_RADIUS_M)

    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    is_vegetated = worldcover.remap(_VEGETATED_CLASSES, [1] * len(_VEGETATED_CLASSES), 0)

    data = is_vegetated.reduceRegion(
        reducer=ee.Reducer.mode(), geometry=region, scale=10, maxPixels=1e8
    ).getInfo()

    return bool(data["remapped"])
