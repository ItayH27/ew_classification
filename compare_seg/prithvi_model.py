"""
Loading and inference wrapper for ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars.

The only file in compare_seg/ that imports terratorch, so split_utils.py and the chip-listing
half of compare.py remain usable before compare_seg/requirements.txt is installed.
"""

from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download

PRITHVI_REPO_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"
PRITHVI_CONFIG_FILENAME = "burn_scars_config.yaml"
PRITHVI_CHECKPOINT_FILENAME = "Prithvi_EO_V2_300M_BurnScars.pt"

# BurnScars-fine-tuning-specific normalization stats (HLS band order: Blue, Green, Red,
# NarrowNIR, SWIR1, SWIR2) — NOT the same values as src.segmentation.dataset.HLS_BAND_MEANS/STDS.
PRITHVI_MEANS = [
    0.033349706741586264, 0.05701185520536176, 0.05889748132001316,
    0.2323245113436119, 0.1972854853760658, 0.11944914225186566,
]
PRITHVI_STDS = [
    0.02269135568823774, 0.026807560223070237, 0.04004109844362779,
    0.07791732423672691, 0.08708738838140137, 0.07241979477437814,
]


def download_prithvi_assets(cache_dir: str | None = None) -> tuple[str, str]:
    """Downloads (and caches) the BurnScars config yaml + ~1.3GB fine-tuned checkpoint."""
    config_path = hf_hub_download(
        repo_id=PRITHVI_REPO_ID, filename=PRITHVI_CONFIG_FILENAME,
        repo_type="model", local_dir=cache_dir,
    )
    checkpoint_path = hf_hub_download(
        repo_id=PRITHVI_REPO_ID, filename=PRITHVI_CHECKPOINT_FILENAME,
        repo_type="model", local_dir=cache_dir,
    )
    return config_path, checkpoint_path


def load_prithvi_model(device: torch.device, cache_dir: str | None = None):
    """Returns a terratorch LightningInferenceModel with its underlying model on `device`, eval mode."""
    from terratorch.cli_tools import LightningInferenceModel

    config_path, checkpoint_path = download_prithvi_assets(cache_dir)
    lightning_model = LightningInferenceModel.from_config(config_path, checkpoint_path)
    lightning_model.model.to(device)
    lightning_model.model.eval()
    return lightning_model


def predict_prithvi(
    lightning_model, image_raw: torch.Tensor, device: torch.device, eval_size: int = 512
) -> torch.Tensor:
    """
    Args:
        lightning_model: as returned by load_prithvi_model()
        image_raw: (6, H, W) raw unnormalized reflectance, HLS band order (CPU tensor)
        eval_size: resolution Prithvi is fed at (its own fine-tuning resolution). HLS chips are
            already native 512, so this is a no-op for them; GEE-sourced negative chips are native
            256, so they're upsampled to eval_size before inference and the prediction is
            downsampled back to native resolution — mirroring how our model is evaluated near its
            own trained resolution in compare.py's _predict_ours.
    Returns:
        (1, 1, H, W) float tensor with values in {0., 1.}, at image_raw's native H, W.
    """
    native_h, native_w = image_raw.shape[-2:]
    means = torch.tensor(PRITHVI_MEANS, dtype=torch.float32).view(6, 1, 1).to(device)
    stds = torch.tensor(PRITHVI_STDS, dtype=torch.float32).view(6, 1, 1).to(device)

    x = image_raw.to(device).unsqueeze(0)  # (1, 6, H, W)
    if (native_h, native_w) != (eval_size, eval_size):
        x = torch.nn.functional.interpolate(
            x, size=(eval_size, eval_size), mode="bilinear", align_corners=False
        )
    x = (x - means) / stds

    with torch.no_grad():
        out = lightning_model.model(x)
        logits = out.output  # (1, 2, eval_size, eval_size)
        preds = torch.argmax(logits, dim=1, keepdim=True).float()  # (1, 1, eval_size, eval_size)

    if (native_h, native_w) != (eval_size, eval_size):
        preds = torch.nn.functional.interpolate(preds, size=(native_h, native_w), mode="nearest")
    return preds


def count_params(lightning_model) -> int:
    return sum(p.numel() for p in lightning_model.model.parameters())
