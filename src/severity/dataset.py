"""
Lightweight HLS image loader for severity classification.

Independent of BurnScarDataset (segmentation) by design — reads raw,
non-normalized reflectance directly via rasterio so NDVI/NBR are computed
on true reflectance values, and exposes stable sample IDs for joining
against data/severity_labels.csv.
"""

from __future__ import annotations

from glob import glob
from pathlib import Path

import numpy as np
import rasterio

_DATA_ROOT = Path("/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/dataset")
_TRAIN_DIR = _DATA_ROOT / "training"
_NEGATIVE_TRAIN_DIR = _DATA_ROOT / "negative" / "training"


def sample_id_from_path(merged_path: str) -> str:
    """'<...>_merged.tif' -> '<...>' (stable id used as CSV key)."""
    return Path(merged_path).name.replace("_merged.tif", "")


def list_train_image_paths() -> list[str]:
    """Sorted list of all training *_merged.tif paths (540 total)."""
    paths = sorted(glob(str(_TRAIN_DIR / "*_merged.tif")))
    if not paths:
        raise RuntimeError(
            f"No *_merged.tif under {_TRAIN_DIR}. Run BurnScarDataset(split='train') "
            "once first to trigger the HF download/extract."
        )
    return paths


def mask_path_for(merged_path: str) -> str:
    return merged_path.replace("_merged.tif", ".mask.tif")


def read_raw_bands(merged_path: str) -> np.ndarray:
    """Returns (H, W, 6) float32 raw reflectance, HLS band order."""
    with rasterio.open(merged_path) as src:
        arr = src.read().astype(np.float32)  # (6, H, W)
    return np.moveaxis(arr, 0, -1)  # (H, W, 6)


def read_mask(mask_path: str) -> np.ndarray:
    """Returns (H, W) uint8 binary mask, -1/nodata clipped to 0."""
    with rasterio.open(mask_path) as src:
        arr = src.read(1)
    return np.clip(arr, 0, 1).astype(np.uint8)


def list_burn_positive_train_samples() -> list[tuple[str, str, str]]:
    """
    Returns [(sample_id, merged_path, mask_path), ...] for every training
    image where mask.sum() > 0. This is the pool the labeling tool samples
    200 from.
    """
    out = []
    for merged_path in list_train_image_paths():
        mp = mask_path_for(merged_path)
        mask = read_mask(mp)
        if mask.sum() > 0:
            out.append((sample_id_from_path(merged_path), merged_path, mp))
    return out


def list_known_negative_train_samples() -> list[tuple[str, str, str]]:
    """
    Returns [(sample_id, merged_path, mask_path), ...] from
    dataset/negative/training/ (see src/retrieval/negative_sampling.py).

    Unlike list_train_image_paths(), does not raise if the directory is
    missing/empty — GEE negative sampling is optional infra that may not
    have been run yet.
    """
    if not _NEGATIVE_TRAIN_DIR.is_dir():
        return []
    out = []
    for merged_path in sorted(glob(str(_NEGATIVE_TRAIN_DIR / "*_merged.tif"))):
        mp = mask_path_for(merged_path)
        out.append((sample_id_from_path(merged_path), merged_path, mp))
    return out


def list_all_train_image_paths() -> list[str]:
    """Union of HLS training images and GEE-sourced negatives, for
    train.py's sample_id -> path lookup (label CSV rows may reference
    either pool)."""
    negative_paths = [mp for _, mp, _ in list_known_negative_train_samples()]
    return list_train_image_paths() + negative_paths
