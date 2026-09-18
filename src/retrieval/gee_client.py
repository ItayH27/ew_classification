"""
GEE client for fetching Sentinel-2 imagery clipped to an AOI.

Sentinel-2 SR bands used:
  B2=Blue, B3=Green, B4=Red (all 10m)
  B8=NIR (10m), B11=SWIR1 (20m), B12=SWIR2 (20m)
  NDVI = (B8 - B4) / (B8 + B4)
  NBR  = (B8 - B12) / (B8 + B12)

The "bands" output key returns all 6 raw bands in HLS order
(B2, B3, B4, B8, B11, B12), scaled to reflectance [0, ~1] by /10000.
"""

import os
from datetime import datetime, timezone

import numpy as np
import ee
from pathlib import Path as _Path
from dotenv import load_dotenv

load_dotenv(_Path(__file__).resolve().parent.parent.parent / ".env")

PATCH_SIZE = 256          # output pixels per side
AOI_RADIUS_M = 10_000     # metres around the input coordinate
# 2 * 10_000m / 256px ≈ 78m/px — keeps sampleRectangle well under its 262144-pixel limit
SCALE_M = int(2 * AOI_RADIUS_M / PATCH_SIZE)

# Sentinel-2 SCL classes to mask out
_S2_MASKED_SCL = [0, 1, 2, 3, 8, 9, 10, 11]  # no-data, saturated, shadow, cloud


def authenticate() -> None:
    """Initialise GEE using a service account key (file or inline JSON) or interactive credentials."""
    key_file = os.environ.get("GEE_KEY_FILE")
    # Inline key contents, for hosts that only offer string secrets (e.g. the HF Space).
    key_json = os.environ.get("GEE_KEY_JSON")
    service_account = os.environ.get("GEE_SERVICE_ACCOUNT")
    project = os.environ.get("GEE_PROJECT")

    if key_json and service_account:
        credentials = ee.ServiceAccountCredentials(service_account, key_data=key_json)
        ee.Initialize(credentials, project=project)
    elif key_file and service_account:
        credentials = ee.ServiceAccountCredentials(service_account, key_file)
        ee.Initialize(credentials, project=project)
    else:
        if not project:
            raise EnvironmentError(
                "GEE_PROJECT is not set. Add it to your .env file:\n"
                "  GEE_PROJECT=your-gcp-project-id\n"
                "Find your project ID at https://console.cloud.google.com"
            )
        ee.Initialize(project=project)


def _make_aoi(lat: float, lon: float) -> ee.Geometry:
    point = ee.Geometry.Point([lon, lat])
    return point.buffer(AOI_RADIUS_M)


def _mask_sentinel_clouds(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    mask = scl.remap(_S2_MASKED_SCL, [0] * len(_S2_MASKED_SCL), 1)
    return image.updateMask(mask)


def _image_to_array(image: ee.Image, bands: list[str], aoi: ee.Geometry) -> np.ndarray:
    """Download a clipped GEE image as a numpy array (H, W, C)."""
    data = (
        image.select(bands)
        .reproject(crs="EPSG:4326", scale=SCALE_M)
        .sampleRectangle(region=aoi, defaultValue=0, properties=[])
    ).getInfo()

    arrays = [np.array(data["properties"][b], dtype=np.float32) for b in bands]
    out = np.stack(arrays, axis=-1)  # (H, W, C)

    # Centre-crop to PATCH_SIZE × PATCH_SIZE
    h, w = out.shape[:2]
    top = max(0, (h - PATCH_SIZE) // 2)
    left = max(0, (w - PATCH_SIZE) // 2)
    return out[top: top + PATCH_SIZE, left: left + PATCH_SIZE]


def get_sentinel_patch(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    max_cloud_pct: int = 80,
    bands_only: bool = False,
) -> dict:
    """
    Fetch the least-cloudy Sentinel-2 SR scene in [start_date, end_date].

    Args:
        max_cloud_pct: Upper bound on CLOUDY_PIXEL_PERCENTAGE for candidate scenes.
        bands_only:    Skip computing rgb/ndvi/nbr (3 fewer getInfo() round-trips) —
                       for callers that only need the raw 6-band array, e.g. bulk
                       negative-sample sourcing.

    Returns:
        {
            "date":  str,                # representative (least-cloudy) scene's acquisition date
            "scenes": [                  # every acquisition contributing to the mosaic, so a
                {"date": str, "cloud_pct": float},   # reader can see what "the image" actually is
                ...
            ],
            "rgb":   ndarray (H, W, 3),  # B4/B3/B2, raw reflectance [0, ~3000]  (omitted if bands_only)
            "ndvi":  ndarray (H, W),     # [-1, 1]                               (omitted if bands_only)
            "nbr":   ndarray (H, W),     # [-1, 1]                               (omitted if bands_only)
            "bands": ndarray (H, W, 6),  # B2,B3,B4,B8,B11,B12 / 10000 — HLS order
        }
    Raises:
        ValueError if the collection is empty after cloud masking.
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .map(_mask_sentinel_clouds)
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    # One round-trip for every scene's date and cloud cover. `mosaic()` below composites ALL of
    # them, not just the representative one, so reporting a single "acquisition date" would
    # understate what the returned pixels are — a mosaic can straddle the whole search window,
    # which matters when the window brackets an event date.
    manifest: dict = ee.Dictionary({
        "dates": collection.aggregate_array("system:time_start"),
        "clouds": collection.aggregate_array("CLOUDY_PIXEL_PERCENTAGE"),
    }).getInfo()

    timestamps = manifest["dates"]
    if not timestamps:
        raise ValueError(
            f"No Sentinel-2 scenes found for AOI between {start_date} and {end_date}."
        )

    scenes = [
        {
            "date": datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
            "cloud_pct": round(float(cloud), 1),
        }
        for ms, cloud in zip(timestamps, manifest["clouds"])
    ]

    # Mosaic all scenes (sorted least-cloudy first) so adjacent tiles fill each
    # other's gaps. Use the date of the least-cloudy scene as the representative date.
    image = collection.mosaic()
    date_str: str = scenes[0]["date"]

    # 6 raw bands in HLS order: Blue, Green, Red, Narrow NIR, SWIR1, SWIR2
    # Scaled by /10000 to match HLS reflectance range for segmentation model input.
    raw_arr = _image_to_array(image, ["B2", "B3", "B4", "B8", "B11", "B12"], aoi)

    if bands_only:
        return {"date": date_str, "scenes": scenes, "bands": raw_arr / 10_000.0}

    rgb_arr = _image_to_array(image, ["B4", "B3", "B2"], aoi)

    ndvi_img = image.normalizedDifference(["B8", "B4"]).rename("ndvi")
    nbr_img = image.normalizedDifference(["B8", "B12"]).rename("nbr")
    indices = ndvi_img.addBands(nbr_img)
    idx_arr = _image_to_array(indices, ["ndvi", "nbr"], aoi)

    return {
        "date":  date_str,
        "scenes": scenes,
        "rgb":   rgb_arr,
        "ndvi":  idx_arr[..., 0],
        "nbr":   idx_arr[..., 1],
        "bands": raw_arr / 10_000.0,  # (256, 256, 6) float32, range [0, ~1]
    }
