"""
Local-file alternative to gee_client.py: reads a scene directly from a local HLS
Burn Scar *_merged.tif instead of fetching from GEE, for offline demos/testing
without GEE credentials or network access.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import rasterio

# Root of the HLS Burn Scar Scenes dataset (src/segmentation/dataset.py's download location) --
# only samples under here have a real sibling ground-truth mask; an arbitrary local tif (e.g. a
# web upload landing in a temp dir) should never be treated as if it did.
HLS_BURN_SCAR_ROOT = Path("/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/dataset")

# HLS (Harmonized Landsat Sentinel-2) ground sample distance. Fixed by the product, so it is a
# constant rather than read per-file -- the dataset's tifs carry no reliable geotransform (see
# negative_sampling._write_patch, which fabricates one). Used only to turn a mask pixel count
# into an area; nothing reprojects these scenes.
HLS_PIXEL_SIZE_M = 30.0


def is_hls_burn_scar_sample(tif_path: str) -> bool:
    """True if tif_path lives under the HLS Burn Scar dataset directory, i.e. it's a labeled
    training/validation sample with a real sibling ground-truth mask."""
    try:
        return Path(tif_path).resolve().is_relative_to(HLS_BURN_SCAR_ROOT.resolve())
    except OSError:
        return False


def read_local_scene(tif_path: str) -> dict:
    """
    Read a local HLS Burn Scar *_merged.tif (6-band, HLS order: B2,B3,B4,B8,B11,B12) and
    derive the same rgb/ndvi/nbr/bands fields that fetch_reference_imagery returns.
    """
    with rasterio.open(tif_path) as src:
        bands = src.read().astype(np.float32)  # (6, H, W), already reflectance
    bands = np.transpose(bands, (1, 2, 0))  # (H, W, 6)

    blue, green, red, nir, swir1, swir2 = (bands[..., i] for i in range(6))
    rgb = np.stack([red, green, blue], axis=-1)
    ndvi = (nir - red) / (nir + red + 1e-9)
    nbr = (nir - swir2) / (nir + swir2 + 1e-9)

    return {"rgb": rgb, "ndvi": ndvi, "nbr": nbr, "bands": bands}


def read_local_mask(tif_path: str) -> np.ndarray | None:
    """Read the ground-truth mask sibling of a *_merged.tif, if tif_path is a known HLS Burn
    Scar dataset sample (see is_hls_burn_scar_sample) and the sibling exists."""
    if not is_hls_burn_scar_sample(tif_path):
        return None
    mask_path = tif_path.replace("_merged.tif", ".mask.tif")
    if not os.path.exists(mask_path):
        return None
    with rasterio.open(mask_path) as src:
        arr = src.read(1).astype(np.float32)
    return np.clip(arr, 0.0, 1.0)  # -1 (nodata/cloud) -> background
