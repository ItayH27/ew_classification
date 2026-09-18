"""
Chip-ID bookkeeping for the segmentation-model comparison.

Prithvi-EO-2.0-300M-BurnScars was fine-tuned on a custom train/val/test partition of the same
804-chip pool that this repo's BurnScarDataset splits into training/ (540) + validation/ (264).
264 of our "validation" chips overlap heavily with Prithvi's own splits/train.txt — evaluating it
on our validation folder as-is would be unfair. build_clean_chip_ids() derives the subset of our
validation chips Prithvi never saw during its own training, so the comparison script can report
both the full split and this leak-free subset.

Deliberately imports no torch/terratorch, so this module (and the chip-listing half of a
comparison run) works even before compare_seg/requirements.txt is installed.
"""

from __future__ import annotations

import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from huggingface_hub import hf_hub_download

from src.segmentation.dataset import BurnScarDataset

PRITHVI_REPO_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"

CHIP_ID_RE = re.compile(r"HLS\.[A-Z0-9]+\.(.+)\.\d+_merged\.tif$")


def extract_chip_id(merged_path: str) -> str:
    basename = os.path.basename(merged_path)
    match = CHIP_ID_RE.search(basename)
    if not match:
        raise ValueError(f"Could not extract a chip ID from filename: {basename!r}")
    return match.group(1)


def list_validation_chips(cache_dir: str | None = None) -> list[tuple[str, str, str]]:
    """
    Returns a list of (chip_id, merged_image_path, mask_path) for every chip in the
    HLS Burn Scar Scenes "validation" split (264 chips), reusing BurnScarDataset's existing
    download/extract/glob logic rather than reimplementing it.
    """
    ds = BurnScarDataset(split="validation", augment=False, cache_dir=cache_dir)
    chips = []
    for img_path in ds._image_paths:
        mask_path = img_path.replace("_merged.tif", ".mask.tif")
        chips.append((extract_chip_id(img_path), img_path, mask_path))
    return chips


def list_negative_chips(negative_dir: str) -> list[tuple[str, str, str]]:
    """
    Returns (chip_id, merged_image_path, mask_path) for every GEE-sourced negative chip in
    <negative_dir>/validation/ (see src/retrieval/negative_sampling.py) — same on-disk contract
    as HLS chips (6-band float32 _merged.tif + all-zero uint8 .mask.tif), but not part of the
    HLS/IBM dataset pool, so it has no "clean subset" concept.
    """
    validation_dir = os.path.join(negative_dir, "validation")
    chips = []
    for img_path in sorted(glob.glob(os.path.join(validation_dir, "*_merged.tif"))):
        chip_id = os.path.basename(img_path)[: -len("_merged.tif")]
        mask_path = img_path.replace("_merged.tif", ".mask.tif")
        chips.append((chip_id, img_path, mask_path))
    return chips


def download_ibm_splits(cache_dir: str | None = None) -> dict[str, set[str]]:
    """
    Downloads Prithvi-BurnScars' own splits/{train,val,test}.txt from its HF model repo and
    returns each as a set of chip IDs.
    """
    splits = {}
    for name in ("train", "val", "test"):
        path = hf_hub_download(
            repo_id=PRITHVI_REPO_ID,
            filename=f"splits/{name}.txt",
            repo_type="model",
            local_dir=cache_dir,
        )
        with open(path) as f:
            splits[name] = {line.strip() for line in f if line.strip()}
    return splits


def build_clean_chip_ids(
    our_chip_ids: list[str] | set[str], ibm_splits: dict[str, set[str]]
) -> set[str]:
    """
    Returns the subset of our_chip_ids that Prithvi did NOT see during its own training —
    i.e. present in IBM's val or test split, and absent from IBM's train split.
    """
    our_ids = set(our_chip_ids)
    ibm_all = ibm_splits["train"] | ibm_splits["val"] | ibm_splits["test"]
    unmatched = our_ids - ibm_all
    if unmatched:
        print(
            f"WARNING: {len(unmatched)} of our chip IDs were not found in any IBM split "
            f"(possible chip-ID parsing mismatch): {sorted(unmatched)[:5]}..."
        )

    clean = (ibm_splits["val"] | ibm_splits["test"]) - ibm_splits["train"]
    return our_ids & clean
