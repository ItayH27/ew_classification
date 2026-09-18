"""
Visualize pre/post reference imagery for a known conflict-damaged location, or run the
segmentation model on a local HLS Burn Scar dataset scene instead of fetching from GEE.

Usage:
    python scripts/visualize_retrieval.py
    python scripts/visualize_retrieval.py --local_tif dataset/validation/<id>_merged.tif

Outputs:
    scripts/slurm_files/retrieval_visualization.png (GEE mode) — 2×4 grid: RGB | NDVI | NBR |
    Burn mask overlay, for pre and post event.
    scripts/slurm_files/<id>_segmentation.png (--local_tif mode) — 1×5 grid: RGB | NDVI | NBR |
    predicted mask overlay | ground-truth mask overlay (if a sibling .mask.tif exists).
    Not shown on screen (headless-safe for Slurm batch jobs); stats are printed instead.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import rasterio
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from src.retrieval.reference_retrieval import fetch_reference_imagery
from src.segmentation.model import BurnScarSegmenter
from src.segmentation.dataset import HLS_BAND_MEANS, HLS_BAND_STDS

# Known burned/damaged area: Kherson region, Ukraine — large agricultural fires
LAT, LON = 48.9001, 37.7249
EVENT_DATE = "2024-07-19"

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "burn_scar", "best_model"
)
MASK_THRESHOLD = 0.5

# Absolute path so this reliably lands in the right place when run as a Slurm batch job.
OUTPUT_PATH = "/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/scripts/slurm_files/retrieval_visualization.png"


def normalize_rgb(arr: np.ndarray) -> np.ndarray:
    """Stretch a raw reflectance (H,W,3) array to [0,1] for display."""
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    clipped = np.clip(arr, lo, hi)
    return (clipped - lo) / (hi - lo + 1e-9)


def load_segmenter(checkpoint_dir: str = CHECKPOINT_DIR):
    """Load the trained BurnScarSegmenter onto the best available device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BurnScarSegmenter.load_pretrained(checkpoint_dir).to(device)
    model.eval()
    return model, device


def predict_burn_mask(model, device, bands: np.ndarray) -> np.ndarray:
    """
    Run the segmentation model on a 6-band HLS-order patch.

    Args:
        bands: (H, W, 6) float32 reflectance, HLS band order (as returned by
               fetch_reference_imagery's "bands" key) — this is the model's
               actual input, not the 3-band RGB used for display.
    Returns:
        (H, W) float32 burn probability in [0, 1]
    """
    means = torch.tensor(HLS_BAND_MEANS, dtype=torch.float32).view(6, 1, 1)
    stds = torch.tensor(HLS_BAND_STDS, dtype=torch.float32).view(6, 1, 1)

    x = torch.from_numpy(bands).permute(2, 0, 1).float()  # (6, H, W)
    x = (x - means) / stds
    x = x.unsqueeze(0).to(device)  # (1, 6, H, W)

    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return prob


def overlay_burn_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    threshold: float = MASK_THRESHOLD,
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
    """Read the ground-truth mask sibling of a *_merged.tif, if present."""
    mask_path = tif_path.replace("_merged.tif", ".mask.tif")
    if not os.path.exists(mask_path):
        return None
    with rasterio.open(mask_path) as src:
        arr = src.read(1).astype(np.float32)
    return np.clip(arr, 0.0, 1.0)  # -1 (nodata/cloud) -> background


def run_local_tif(tif_path: str) -> None:
    scene_id = os.path.basename(tif_path).replace("_merged.tif", "")
    print(f"Reading local scene {tif_path} …")
    scene = read_local_scene(tif_path)
    gt_mask = read_local_mask(tif_path)

    print(f"Loading segmentation model from {CHECKPOINT_DIR} …")
    model, device = load_segmenter()
    pred_mask = predict_burn_mask(model, device, scene["bands"])

    n_cols = 5 if gt_mask is not None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    index_cmap = "RdYlGn"
    index_norm = mcolors.Normalize(vmin=-1, vmax=1)

    axes[0].imshow(normalize_rgb(scene["rgb"]))
    axes[0].set_title("RGB (HLS)", fontsize=10)

    im_ndvi = axes[1].imshow(scene["ndvi"], cmap=index_cmap, norm=index_norm)
    axes[1].set_title("NDVI", fontsize=10)
    plt.colorbar(im_ndvi, ax=axes[1], fraction=0.046, pad=0.04)

    im_nbr = axes[2].imshow(scene["nbr"], cmap=index_cmap, norm=index_norm)
    axes[2].set_title("NBR", fontsize=10)
    plt.colorbar(im_nbr, ax=axes[2], fraction=0.046, pad=0.04)

    pred_overlay = overlay_burn_mask(scene["rgb"], pred_mask)
    pred_pct = (pred_mask > MASK_THRESHOLD).mean() * 100
    axes[3].imshow(pred_overlay)
    axes[3].set_title(f"Predicted Burn Mask\n({pred_pct:.1f}% burned)", fontsize=10)

    if gt_mask is not None:
        gt_overlay = overlay_burn_mask(scene["rgb"], gt_mask, threshold=0.5, color=(0.0, 0.6, 1.0))
        gt_pct = gt_mask.mean() * 100
        axes[4].imshow(gt_overlay)
        axes[4].set_title(f"Ground Truth Mask\n({gt_pct:.1f}% burned)", fontsize=10)

    for ax in axes:
        ax.axis("off")

    plt.suptitle(f"Segmentation — {scene_id}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(
        os.path.dirname(OUTPUT_PATH), f"{scene_id}_segmentation.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    print(f"\nPredicted burn area : {pred_pct:.1f}%")
    if gt_mask is not None:
        gt_pct = gt_mask.mean() * 100
        iou = (
            ((pred_mask > MASK_THRESHOLD) & (gt_mask > 0.5)).sum()
            / max(((pred_mask > MASK_THRESHOLD) | (gt_mask > 0.5)).sum(), 1)
        )
        print(f"Ground-truth burn area : {gt_pct:.1f}%")
        print(f"IoU vs. ground truth   : {iou:.3f}")


def main() -> None:
    print(f"Fetching imagery for ({LAT}, {LON}) around {EVENT_DATE} …")
    result = fetch_reference_imagery(LAT, LON, EVENT_DATE)

    print(f"Loading segmentation model from {CHECKPOINT_DIR} …")
    model, device = load_segmenter()

    post_mask = predict_burn_mask(model, device, result["post"]["bands"])

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    row_labels = ["Pre-event", "Post-event"]
    col_titles = ["RGB (Sentinel-2)", "NDVI (Sentinel-2)", "NBR (Sentinel-2)", "Burn Mask (Segmentation)"]

    index_cmap = "RdYlGn"
    index_norm = mcolors.Normalize(vmin=-1, vmax=1)

    for row, period in enumerate(["pre", "post"]):
        data = result[period]
        label = f"{row_labels[row]}\n({data['date']})"

        # RGB
        axes[row][0].imshow(normalize_rgb(data["rgb"]))
        axes[row][0].set_title(f"{col_titles[0]}\n{label}", fontsize=10)

        # NDVI
        im_ndvi = axes[row][1].imshow(data["ndvi"], cmap=index_cmap, norm=index_norm)
        axes[row][1].set_title(f"{col_titles[1]}\n{label}", fontsize=10)
        plt.colorbar(im_ndvi, ax=axes[row][1], fraction=0.046, pad=0.04)

        # NBR
        im_nbr = axes[row][2].imshow(data["nbr"], cmap=index_cmap, norm=index_norm)
        axes[row][2].set_title(f"{col_titles[2]}\n{label}", fontsize=10)
        plt.colorbar(im_nbr, ax=axes[row][2], fraction=0.046, pad=0.04)

        # Segmentation mask overlaid on RGB — post-event only
        if period == "post":
            overlay = overlay_burn_mask(data["rgb"], post_mask)
            burn_pct = (post_mask > MASK_THRESHOLD).mean() * 100
            axes[row][3].imshow(overlay)
            axes[row][3].set_title(f"{col_titles[3]}\n{label}  ({burn_pct:.1f}% burned)", fontsize=10)

    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle(
        f"Reference Imagery — ({LAT}, {LON})  |  event: {EVENT_DATE}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")

    pre_ndvi_mean = result["pre"]["ndvi"].mean()
    post_ndvi_mean = result["post"]["ndvi"].mean()
    print(f"\nPre-event  NDVI mean : {pre_ndvi_mean:.3f}")
    print(f"Post-event NDVI mean : {post_ndvi_mean:.3f}")
    delta_ndvi = pre_ndvi_mean - post_ndvi_mean
    if delta_ndvi > 0.05:
        print(f"NDVI drop of {delta_ndvi:.3f} — consistent with vegetation damage.")
    else:
        print(f"NDVI difference is small ({delta_ndvi:.3f}) — check cloud cover or coordinates.")

    pre_nbr_mean = result["pre"]["nbr"].mean()
    post_nbr_mean = result["post"]["nbr"].mean()
    dnbr = pre_nbr_mean - post_nbr_mean
    print(f"\nPre-event  NBR mean  : {pre_nbr_mean:.3f}")
    print(f"Post-event NBR mean  : {post_nbr_mean:.3f}")
    print(f"dNBR (pre − post)    : {dnbr:.3f}")
    if dnbr > 0.1:
        print(f"dNBR of {dnbr:.3f} — consistent with burn/vegetation loss.")
    else:
        print(f"dNBR is low ({dnbr:.3f}) — no strong burn signal detected.")

    post_burn_pct = (post_mask > MASK_THRESHOLD).mean() * 100
    print(f"\nPost-event predicted burn area : {post_burn_pct:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_tif",
        type=str,
        default=None,
        help="Path to a local HLS *_merged.tif (e.g. dataset/validation/<id>_merged.tif). "
        "Bypasses GEE and runs segmentation directly on that scene.",
    )
    args = parser.parse_args()

    if args.local_tif:
        run_local_tif(args.local_tif)
    else:
        main()
