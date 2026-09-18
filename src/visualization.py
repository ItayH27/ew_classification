"""
Shared display/rendering helpers used by the CLI and web pipeline entry points
(scripts/run_pipeline.py, web/app.py) — array normalization, mask overlays, and
PNG encoding for HTTP responses.
"""

from __future__ import annotations

import base64
from io import BytesIO

import matplotlib.image
import numpy as np


def normalize_rgb(arr: np.ndarray) -> np.ndarray:
    """Stretch a raw reflectance (H,W,3) array to [0,1] for display."""
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    clipped = np.clip(arr, lo, hi)
    return (clipped - lo) / (hi - lo + 1e-9)


def overlay_burn_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.5,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a translucent color over the display-stretched RGB where mask > threshold."""
    base = normalize_rgb(rgb)
    binary = mask > threshold
    overlay = base.copy()
    for c in range(3):
        overlay[..., c] = np.where(binary, (1 - alpha) * base[..., c] + alpha * color[c], base[..., c])
    return overlay


def array_to_png_base64(
    arr: np.ndarray,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> str:
    """
    Encode a (H,W) or (H,W,3) array already in displayable range (e.g. output of
    normalize_rgb/overlay_burn_mask, or a raw index array with vmin/vmax set) as a
    base64 PNG data string, for inline embedding in a JSON API response.

    Uses matplotlib.image.imsave (stateless array-in/bytes-out) rather than pyplot,
    whose global figure state isn't safe to call concurrently from request handlers.
    """
    buf = BytesIO()
    matplotlib.image.imsave(buf, arr, cmap=cmap, vmin=vmin, vmax=vmax, format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")
