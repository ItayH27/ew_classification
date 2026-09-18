"""
Inference entry point for the segmentation model: given a raw HLS band array,
returns a predicted binary burn mask.

Deliberately independent of src.severity — any segmentation→severity hookup
(e.g. passing this mask into classify_severity_from_retrieval) happens at the
call site, not via an import here, so the two stages stay independently
testable.
"""

from __future__ import annotations

import os

import numpy as np
import torch

try:
    import spaces  # only installed on a Hugging Face ZeroGPU Space
except ImportError:
    spaces = None

from .dataset import HLS_BAND_MEANS, HLS_BAND_STDS
from .model import BurnScarSegmenter

# EW_CHECKPOINT_ROOT lets a deployment (e.g. the HF Space, deploy/app.py) point at downloaded weights.
_CHECKPOINT_ROOT = os.environ.get(
    "EW_CHECKPOINT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "checkpoints")
)
_DEFAULT_CHECKPOINT = os.path.join(_CHECKPOINT_ROOT, "burn_scar", "best_model")

_model: BurnScarSegmenter | None = None


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_model(checkpoint_dir: str = _DEFAULT_CHECKPOINT) -> BurnScarSegmenter:
    global _model
    if _model is None:
        _model = BurnScarSegmenter.load_pretrained(checkpoint_dir)
        # On ZeroGPU, keep the weights on CPU here: an import-time .to("cuda") is only emulated
        # there, and its restore inside the GPU call came back broken for this UNet (some layers
        # left on CPU, then an all-zero mask). _forward places the model once a real GPU exists.
        if spaces is None:
            _model.to(_get_device())
        _model.eval()
    return _model


def predict_mask(
    bands: np.ndarray,
    checkpoint_dir: str = _DEFAULT_CHECKPOINT,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Args:
        bands: (H, W, 6) raw reflectance, HLS band order — e.g.
            src.severity.dataset.read_raw_bands(...) or
            fetch_reference_imagery(...)["post"]["bands"].
        checkpoint_dir: directory containing a diffusers-format BurnScarSegmenter checkpoint.
        threshold: sigmoid threshold for the binary mask (0.5 matches SegmentationMetrics' default).

    Returns:
        (H, W) uint8 binary mask {0, 1}.
    """
    return _forward(_get_model(checkpoint_dir), bands, threshold)


def _forward(model: BurnScarSegmenter, bands: np.ndarray, threshold: float) -> np.ndarray:
    # No-op normally (already placed in _get_model). On ZeroGPU this is where the model first
    # reaches the GPU, since one is only attached inside this spaces.GPU-wrapped call.
    device = _get_device()
    model.to(device)

    means = torch.tensor(HLS_BAND_MEANS, dtype=torch.float32).view(6, 1, 1).to(device)
    stds = torch.tensor(HLS_BAND_STDS, dtype=torch.float32).view(6, 1, 1).to(device)

    x = torch.from_numpy(bands).permute(2, 0, 1).float().to(device)  # (6, H, W)
    x = (x - means) / stds
    x = x.unsqueeze(0)  # (1, 6, H, W)

    with torch.no_grad():
        logits = model(x)  # (1, 1, H, W)
        mask = (torch.sigmoid(logits) > threshold).squeeze(0).squeeze(0)  # (H, W)

    return mask.cpu().numpy().astype(np.uint8)


# On a ZeroGPU HF Space (deploy/app.py), only the forward pass holds a GPU slice — not the
# GEE/news network calls around it in src/pipeline.py. `spaces` is absent elsewhere, so this
# is a no-op locally and on the cluster.
if spaces is not None:
    _forward = spaces.GPU(duration=30)(_forward)
