"""
Interactive tool to manually assign severity labels (0=no damage, 1=minimal ..
3=maximal) to a fixed, reproducible sample of 200 HLS burn-positive training
images, plus an optional pool of GEE-sourced negative (no-damage) samples.
Resumable: re-running skips ids already present in data/severity_labels.csv.

Usage:
    python scripts/label_severity_samples.py
    python scripts/label_severity_samples.py --n_samples 200 --seed 42
    # Also label some GEE-sourced negatives (see scripts/fetch_negative_samples.py)
    python scripts/label_severity_samples.py --n_negative_samples 50
    # Negatives are 0=no damage by construction — write them without prompting
    python scripts/label_severity_samples.py --n_negative_samples 50 --auto_label_negatives
"""

import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np

from src.severity.dataset import (
    list_burn_positive_train_samples,
    list_known_negative_train_samples,
    read_raw_bands,
    read_mask,
)
from src.severity.features import compute_ndvi, compute_nbr

LABELS_CSV = "/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/data/severity_labels.csv"
PREVIEW_PNG = "/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/data/_severity_preview.png"


def _load_existing_ids(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        return {row["sample_id"] for row in csv.DictReader(f)}


def _append_label(csv_path: str, sample_id: str, severity: int) -> None:
    file_exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["sample_id", "severity"])
        writer.writerow([sample_id, severity])


def _normalize_rgb(arr: np.ndarray) -> np.ndarray:
    """Same 2-98 percentile stretch as visualize_retrieval.py."""
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    clipped = np.clip(arr, lo, hi)
    return (clipped - lo) / (hi - lo + 1e-9)


def _show_sample(sample_id: str, bands: np.ndarray, mask: np.ndarray) -> None:
    ndvi = compute_ndvi(bands)
    nbr = compute_nbr(bands)
    rgb = bands[..., [2, 1, 0]]  # Red, Green, Blue -> RGB order

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(_normalize_rgb(rgb))
    axes[0].set_title("RGB")
    axes[1].imshow(_normalize_rgb(rgb))
    axes[1].imshow(mask, cmap="Reds", alpha=0.4)
    axes[1].set_title("Mask overlay")
    axes[2].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[2].set_title(f"NDVI (mean={ndvi.mean():.3f})")
    axes[3].imshow(nbr, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[3].set_title(f"NBR (mean={nbr.mean():.3f})")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(sample_id)
    plt.tight_layout()
    # No GUI backend available in this environment (remote/SSH session), so
    # save to a fixed path instead of plt.show() — open it once in the VSCode
    # image viewer and it will refresh in place on each subsequent sample.
    os.makedirs(os.path.dirname(PREVIEW_PNG), exist_ok=True)
    fig.savefig(PREVIEW_PNG, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _label_interactive(sample_id: str) -> str:
    """Prompt for a severity label. Returns 'quit', 'skip', or 'labeled'."""
    while True:
        raw = input(f"[{sample_id}] severity (0=no damage, 1=minimal .. 3=maximal, 's'=skip, 'q'=quit): ").strip()
        if raw.lower() == "q":
            return "quit"
        if raw.lower() == "s":
            return "skip"
        if raw in {"0", "1", "2", "3"}:
            _append_label(LABELS_CSV, sample_id, int(raw))
            return "labeled"
        print("Invalid input.")


def main(
    n_samples: int = 200,
    seed: int = 42,
    n_negative_samples: int = 0,
    auto_label_negatives: bool = False,
) -> None:
    pool = list_burn_positive_train_samples()  # [(id, merged_path, mask_path), ...]
    print(f"Burn-positive pool size: {len(pool)}")

    rng = random.Random(seed)
    pool_sorted = sorted(pool, key=lambda t: t[0])  # deterministic order before sampling
    selected = rng.sample(pool_sorted, k=min(n_samples, len(pool_sorted)))

    neg_selected = []
    if n_negative_samples > 0:
        neg_pool = list_known_negative_train_samples()
        if not neg_pool:
            print(
                "No negative samples found under dataset/negative/training — "
                "run scripts/fetch_negative_samples.py first. Skipping negatives."
            )
        else:
            neg_pool_sorted = sorted(neg_pool, key=lambda t: t[0])
            neg_selected = rng.sample(neg_pool_sorted, k=min(n_negative_samples, len(neg_pool_sorted)))

    already = _load_existing_ids(LABELS_CSV)
    print(f"Already labeled: {len(already)} / {len(selected) + len(neg_selected)}")
    print(f"Each sample's plot is saved to: {PREVIEW_PNG}")
    print("Open that file once in your editor's image viewer — it will refresh on each new sample.\n")

    for sample_id, merged_path, mask_path in selected:
        if sample_id in already:
            continue
        bands = read_raw_bands(merged_path)
        mask = read_mask(mask_path)
        _show_sample(sample_id, bands, mask)

        if _label_interactive(sample_id) == "quit":
            print("Exiting; progress saved.")
            return

    for sample_id, merged_path, mask_path in neg_selected:
        if sample_id in already:
            continue
        bands = read_raw_bands(merged_path)
        mask = read_mask(mask_path)
        # Still rendered for a spot-check — land-cover filtering alone can't
        # rule out an undocumented fire near the sampled point.
        _show_sample(sample_id, bands, mask)

        if auto_label_negatives:
            _append_label(LABELS_CSV, sample_id, 0)
            print(f"[{sample_id}] auto-labeled 0 (known negative)")
            continue

        if _label_interactive(sample_id) == "quit":
            print("Exiting; progress saved.")
            return

    print("Labeling complete for this session.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manually label severity for HLS burn-positive samples.")
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_negative_samples", type=int, default=0,
                    help="Also sample this many GEE-sourced negatives to label (default: 0)")
    p.add_argument("--auto_label_negatives", action="store_true",
                    help="Skip the interactive prompt for the negative pool and write "
                         "severity=0 directly (ground truth is certain by construction — "
                         "land-cover-filtered point sampling — but preview PNGs are still "
                         "rendered for spot-checking, since land-cover filtering alone "
                         "can't rule out an undocumented fire)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        n_samples=args.n_samples,
        seed=args.seed,
        n_negative_samples=args.n_negative_samples,
        auto_label_negatives=args.auto_label_negatives,
    )
