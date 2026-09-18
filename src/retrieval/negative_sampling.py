"""
Bulk-sources presumed-benign Sentinel-2 patches for training (not inference):
samples random points within NEGATIVE_SAMPLE_REGIONS, filters to vegetated
land via land_cover.is_vegetated_point, fetches bands via GEE, and writes
accepted patches to disk in BurnScarDataset's on-disk format (<id>_merged.tif
+ <id>.mask.tif, the latter all-zero by construction) under a directory kept
separate from the pristine HLS download so the two never mix.

NEGATIVE_SAMPLE_REGIONS is deliberately a handful of explicit vegetated
bounding boxes spread across continents/biomes — rather than true global
random sampling — so the pool stays reviewable and reproducible while still
covering enough biome/sensor diversity that the model can't learn an easy
"which country is this" shortcut for "no burn scar" instead of a real damage
signal.

Land-cover filtering only — not cross-checked against a burn-perimeter
database, so review a batch's preview PNGs (via
scripts/label_severity_samples.py --auto_label_negatives) before trusting it
at scale.
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from .gee_client import _make_aoi, get_sentinel_patch
from .land_cover import is_vegetated_point

# (min_lon, min_lat, max_lon, max_lat) — a curated spread of vegetated,
# non-conflict farmland/forest/savanna regions across continents, so the
# negative pool isn't dominated by one biome or one sensor-calibration region.
NEGATIVE_SAMPLE_REGIONS = [
    {"name": "iowa_farmland",          "bbox": (-94.5, 41.5, -93.0, 42.3)},   # US Corn Belt
    {"name": "pampas_argentina",       "bbox": (-62.0, -35.0, -60.0, -33.5)}, # South American grassland/farmland
    {"name": "loire_valley_france",    "bbox": (0.5, 47.2, 2.0, 48.0)},       # Western European farmland
    {"name": "kenya_highlands",        "bbox": (35.5, -1.5, 37.0, -0.2)},     # East African highland farmland
    {"name": "mekong_delta_vietnam",   "bbox": (105.5, 9.8, 106.8, 10.8)},    # Southeast Asian rice paddies
    {"name": "punjab_india",           "bbox": (74.5, 30.0, 76.0, 31.0)},     # South Asian farmland
    {"name": "victoria_australia",     "bbox": (143.0, -37.0, 144.5, -36.0)}, # Australian farmland
    {"name": "pacific_nw_forest_usa",  "bbox": (-123.5, 45.5, -122.0, 46.5)}, # Temperate conifer forest
]

# Single fixed window — Northern-Hemisphere growing season. Southern-Hemisphere
# regions above (Argentina, Australia) land in their autumn/winter instead, so
# fields there may show more bare soil / less green canopy than a true local
# growing-season sample would — still valid "no burn scar" ground truth, just
# less phenologically matched. No event to anchor to, so there's no single
# per-hemisphere window that's obviously more "correct."
_DATE_RANGE = ("2023-04-01", "2023-09-30")
_MAX_CLOUD_PCT = 20        # stricter than event-retrieval's default of 80 — no anchor date to justify tolerating cloud
_MAX_INVALID_FRACTION = 0.02  # reject patches with >2% nodata-filled pixels


def _sample_random_point(rng: random.Random, regions: list[dict]) -> tuple[float, float, str]:
    region = rng.choice(regions)
    min_lon, min_lat, max_lon, max_lat = region["bbox"]
    lat = rng.uniform(min_lat, max_lat)
    lon = rng.uniform(min_lon, max_lon)
    return lat, lon, region["name"]


def _invalid_pixel_fraction(bands: np.ndarray) -> float:
    """Fraction of pixels that are exactly all-zero across every band —
    sampleRectangle's defaultValue=0 fill for out-of-bounds/nodata."""
    all_zero = np.all(bands == 0, axis=-1)
    return float(all_zero.mean())


def _write_patch(bands: np.ndarray, lat: float, lon: float, out_path: str) -> None:
    """Write (H, W, 6) float32 reflectance as a 6-band GeoTIFF, same on-disk
    contract BurnScarDataset reads (band count/dtype/shape matter to the
    downstream readers; exact geo-referencing does not, since nothing
    reprojects these patches)."""
    h, w = bands.shape[:2]
    transform = from_origin(lon, lat, 0.0001, 0.0001)
    with rasterio.open(
        out_path, "w", driver="GTiff", height=h, width=w, count=6,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        for b in range(6):
            dst.write(bands[..., b], b + 1)


def _write_zero_mask(shape: tuple[int, int], lat: float, lon: float, out_path: str) -> None:
    h, w = shape
    transform = from_origin(lon, lat, 0.0001, 0.0001)
    with rasterio.open(
        out_path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype="uint8", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(np.zeros((h, w), dtype=np.uint8), 1)


def generate_negative_samples(
    n_target: int,
    out_dir: str,
    val_fraction: float = 0.15,
    seed: int = 42,
    max_attempts: int | None = None,
) -> dict:
    """
    Rejection-samples points from NEGATIVE_SAMPLE_REGIONS, filters via
    is_vegetated_point, fetches via get_sentinel_patch(..., bands_only=True,
    max_cloud_pct=_MAX_CLOUD_PCT), rejects patches with too much nodata, and
    writes accepted patches to
        <out_dir>/{training,validation}/neg_<seq>_<region>_merged.tif
        <out_dir>/{training,validation}/neg_<seq>_<region>.mask.tif
    split by val_fraction (assigned deterministically by running count, so the
    split ratio comes out exact). Also appends a row to <out_dir>/manifest.csv
    (id,split,lat,lon,region,date) for provenance — not consumed by training
    code, purely an audit trail.

    Never raises on a single point's failure (e.g. ValueError from
    get_sentinel_patch when no scene is found); logs and continues.
    """
    rng = random.Random(seed)
    max_attempts = max_attempts if max_attempts is not None else n_target * 15

    train_dir = Path(out_dir) / "training"
    val_dir = Path(out_dir) / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(out_dir) / "manifest.csv"
    manifest_exists = manifest_path.exists()
    manifest_file = open(manifest_path, "a", newline="")
    manifest_writer = csv.writer(manifest_file)
    if not manifest_exists:
        manifest_writer.writerow(["id", "split", "lat", "lon", "region", "date"])

    n_written = 0
    n_val_written = 0
    n_rejected_vegetation = 0
    n_rejected_cloud_or_nodata = 0
    n_attempts = 0

    try:
        while n_written < n_target and n_attempts < max_attempts:
            n_attempts += 1
            lat, lon, region_name = _sample_random_point(rng, NEGATIVE_SAMPLE_REGIONS)

            if not is_vegetated_point(lat, lon):
                n_rejected_vegetation += 1
                continue

            try:
                aoi = _make_aoi(lat, lon)
                result = get_sentinel_patch(
                    aoi, *_DATE_RANGE, max_cloud_pct=_MAX_CLOUD_PCT, bands_only=True,
                )
            except ValueError:
                n_rejected_cloud_or_nodata += 1
                continue

            bands = result["bands"]
            if _invalid_pixel_fraction(bands) > _MAX_INVALID_FRACTION:
                n_rejected_cloud_or_nodata += 1
                continue

            # Assign to validation whenever doing so keeps the running ratio closer
            # to val_fraction, so the final split comes out exact rather than
            # depending on n_target's divisibility.
            is_val = (n_val_written + 1) / (n_written + 1) <= val_fraction
            split = "validation" if is_val else "training"
            split_dir = val_dir if is_val else train_dir
            sample_id = f"neg_{n_written:04d}_{region_name}"

            _write_patch(bands, lat, lon, str(split_dir / f"{sample_id}_merged.tif"))
            _write_zero_mask(bands.shape[:2], lat, lon, str(split_dir / f"{sample_id}.mask.tif"))
            manifest_writer.writerow([sample_id, split, lat, lon, region_name, result["date"]])

            n_written += 1
            if is_val:
                n_val_written += 1
    finally:
        manifest_file.close()

    return {
        "n_written": n_written,
        "n_rejected_vegetation": n_rejected_vegetation,
        "n_rejected_cloud_or_nodata": n_rejected_cloud_or_nodata,
        "n_attempts": n_attempts,
    }
