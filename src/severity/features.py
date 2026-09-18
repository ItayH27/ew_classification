"""
NDVI/NBR scalar feature extraction — shared by training (HLS GeoTIFFs) and
inference (GEE Sentinel-2 imagery via fetch_reference_imagery).

Both paths must reduce to the same 5 scalars using the same formulas so the
classifier trained on HLS generalizes to GEE Sentinel-2 inputs:
    NDVI = (NIR - Red) / (NIR + Red)
    NBR  = (NIR - SWIR2) / (NIR + SWIR2)

Whole-image and masked-region means are both kept (rather than masked-only)
because an earlier mask-only iteration regressed accuracy/macro-F1 — masking
alone discards the burn-extent signal the whole-image mean was implicitly
carrying (a small severe burn and a large severe burn can share near-identical
masked-region NDVI/NBR but very different whole-image means). Burned-area
fraction is added as an explicit extent feature.
"""

from __future__ import annotations

import numpy as np

# HLS band order: 0=Blue, 1=Green, 2=Red, 3=NarrowNIR, 4=SWIR1, 5=SWIR2
_RED, _NIR, _SWIR2 = 2, 3, 5
_EPS = 1e-9


def compute_ndvi(bands: np.ndarray) -> np.ndarray:
    """bands: (H, W, 6) raw reflectance, HLS order. Returns (H, W) NDVI."""
    red, nir = bands[..., _RED], bands[..., _NIR]
    return (nir - red) / (nir + red + _EPS)


def compute_nbr(bands: np.ndarray) -> np.ndarray:
    """bands: (H, W, 6) raw reflectance, HLS order. Returns (H, W) NBR."""
    nir, swir2 = bands[..., _NIR], bands[..., _SWIR2]
    return (nir - swir2) / (nir + swir2 + _EPS)


def _masked_mean(arr: np.ndarray, mask: np.ndarray | None) -> float:
    """Mean over mask==1 pixels if mask is given and non-empty, else whole-array mean."""
    if mask is not None and mask.sum() > 0:
        return float(np.nanmean(arr[mask == 1]))
    return float(np.nanmean(arr))


def _five_features(
    ndvi: np.ndarray, nbr: np.ndarray, mask: np.ndarray | None, known_negative: bool = False
) -> tuple[float, float, float, float, float]:
    whole_ndvi, whole_nbr = float(np.nanmean(ndvi)), float(np.nanmean(nbr))
    if mask is not None and mask.sum() > 0:
        masked_ndvi = _masked_mean(ndvi, mask)
        masked_nbr = _masked_mean(nbr, mask)
        burned_fraction = float(mask.mean())
    else:
        # No mask (or empty prediction): treat the whole frame as the region of
        # interest rather than emitting an arbitrary zero/NaN the scaler never saw.
        # burned_fraction: 0.0 only when the caller has confirmed this is a true
        # no-damage sample (known_negative); otherwise this is the ambiguous case
        # (e.g. an empty predicted mask at inference isn't necessarily "no damage"),
        # so it keeps the conservative 1.0 fallback.
        #
        # That 1.0 is a MODEL-INPUT convention, not a measurement, and must never be
        # reported to a user as burned area — doing so is what made empty-mask scenes
        # display "100% burned, maximal severity". src/pipeline.py now measures extent
        # straight off the mask and skips this classifier entirely when the mask is
        # empty, so the inference path no longer reaches this branch; training still
        # does, via known_negative.
        masked_ndvi, masked_nbr = whole_ndvi, whole_nbr
        burned_fraction = 0.0 if known_negative else 1.0
    return whole_ndvi, whole_nbr, masked_ndvi, masked_nbr, burned_fraction


def extract_features_from_bands(
    bands: np.ndarray, mask: np.ndarray | None = None, known_negative: bool = False
) -> tuple[float, float, float, float, float]:
    """
    (whole_ndvi, whole_nbr, masked_ndvi, masked_nbr, burned_fraction) from a raw
    (H, W, 6) HLS band array.

    mask: optional (H, W) binary burn mask. When given and non-empty, the masked_*
    features average over mask==1 pixels only (standard practice for burn-severity
    indices like dNBR/CBI) and burned_fraction is mask.mean(); otherwise masked_*
    fall back to the whole-image mean and burned_fraction defaults to 1.0 — unless
    known_negative=True (a confirmed no-damage sample, e.g. GEE-sourced), in which
    case burned_fraction is 0.0. known_negative is never inferred from mask alone;
    the caller must pass it explicitly.
    """
    return _five_features(compute_ndvi(bands), compute_nbr(bands), mask, known_negative)


def extract_features_from_indices(
    ndvi: np.ndarray, nbr: np.ndarray, mask: np.ndarray | None = None, known_negative: bool = False
) -> tuple[float, float, float, float, float]:
    """
    (whole_ndvi, whole_nbr, masked_ndvi, masked_nbr, burned_fraction) from
    precomputed (H, W) index arrays (e.g. GEE output).

    mask, known_negative: see extract_features_from_bands.
    """
    return _five_features(ndvi, nbr, mask, known_negative)
