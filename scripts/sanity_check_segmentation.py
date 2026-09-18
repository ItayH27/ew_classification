"""
End-to-end sanity check for the burn-scar segmentation pipeline.

Runs, in order:
    1. Dataset inspection      — BurnScarDataset shapes/dtypes (downloads ~1.1 GB on first run),
                                  plus negative_dir integration against a synthetic negative pool
    2. Model forward pass      — BurnScarSegmenter output shape/param count
    3. Loss and metrics        — CombinedLoss + SegmentationMetrics on a synthetic batch,
                                  including the all-negative-batch fnr=0.0 regression check
    4. Short training run      — 2 epochs, batch 4, output to /tmp/seg_test
    5. Checkpoint reload       — reload best_model and run inference
    6. predict_mask()          — run the production checkpoint's predict_mask() on a real
                                  training image (used by src.severity.predict for masked features)
    7. Negative-mixing training — short training run with a synthetic negative_dir and
                                  negative_fraction, exercising the WeightedRandomSampler path
                                  without needing real GEE data

Usage:
    python scripts/sanity_check_segmentation.py            # run all steps in order
    python scripts/sanity_check_segmentation.py --step 3    # run only step 3
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader

TRAIN_OUTPUT_DIR = "/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/checkpoints/sanity_test"


def _make_synthetic_negative_dir(tmp_root: str) -> str:
    """Writes a tiny negative-sample pool in BurnScarDataset's on-disk format
    (2 training + 1 validation patch), standing in for real GEE-sourced data
    which isn't fetchable in this environment."""
    neg_dir = os.path.join(tmp_root, "sanity_negatives")
    rng = np.random.default_rng(0)

    def _write(split_dir: str, idx: int) -> None:
        os.makedirs(split_dir, exist_ok=True)
        bands = rng.uniform(0, 0.3, size=(256, 256, 6)).astype(np.float32)
        with rasterio.open(
            os.path.join(split_dir, f"synth_neg_{idx}_merged.tif"),
            "w", driver="GTiff", height=256, width=256, count=6, dtype="float32",
        ) as dst:
            for b in range(6):
                dst.write(bands[..., b], b + 1)
        with rasterio.open(
            os.path.join(split_dir, f"synth_neg_{idx}.mask.tif"),
            "w", driver="GTiff", height=256, width=256, count=1, dtype="uint8",
        ) as dst:
            dst.write(np.zeros((256, 256), dtype=np.uint8), 1)

    _write(os.path.join(neg_dir, "training"), 0)
    _write(os.path.join(neg_dir, "training"), 1)
    _write(os.path.join(neg_dir, "validation"), 0)
    return neg_dir


def step1_dataset() -> None:
    print("\n=== Step 1: Dataset inspection ===")
    from src.segmentation.dataset import BurnScarDataset

    ds = BurnScarDataset("train")
    print("Train size:", len(ds))
    assert len(ds) == 540, f"expected 540 train samples, got {len(ds)}"

    img, mask = ds[0]
    print("image shape:", img.shape)
    print("image dtype:", img.dtype)
    print("mask  shape:", mask.shape)
    print("mask unique:", mask.unique())
    assert img.shape == (6, 256, 256), f"unexpected image shape {img.shape}"
    assert img.dtype == torch.float32, f"unexpected image dtype {img.dtype}"
    assert mask.shape == (1, 256, 256), f"unexpected mask shape {mask.shape}"
    assert set(mask.unique().tolist()) <= {0.0, 1.0}, f"unexpected mask values {mask.unique()}"

    dl = DataLoader(ds, batch_size=2, num_workers=0)
    imgs, masks = next(iter(dl))
    print("batch image:", imgs.shape)
    print("batch mask: ", masks.shape)
    assert imgs.shape == (2, 6, 256, 256), f"unexpected batch image shape {imgs.shape}"
    assert masks.shape == (2, 1, 256, 256), f"unexpected batch mask shape {masks.shape}"

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        neg_dir = _make_synthetic_negative_dir(tmp)
        ds_neg = BurnScarDataset("train", negative_dir=neg_dir)
        print("Train size with negatives:", len(ds_neg))
        assert len(ds_neg) == 542, f"expected 540 + 2 synthetic negatives, got {len(ds_neg)}"
        assert ds_neg.is_negative == [False] * 540 + [True, True], "is_negative provenance mismatch"
        _, neg_mask = ds_neg[-1]
        assert neg_mask.sum() == 0, "synthetic negative mask should be all-zero"
    print("Step 1: OK")


def step2_model_forward() -> None:
    print("\n=== Step 2: Model forward pass ===")
    from src.segmentation.model import BurnScarSegmenter

    m = BurnScarSegmenter()
    params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"Parameters: {params:.1f}M")  # expect ~56.6M

    assert 56.0 <= params <= 57.0, f"expected ~56.6M params, got {params:.1f}M"

    x = torch.randn(2, 6, 256, 256)
    y = m(x)
    print("Output shape:", y.shape)  # expect (2, 1, 256, 256)
    assert y.shape == (2, 1, 256, 256)
    print("Step 2: OK")


def step3_loss_and_metrics() -> None:
    print("\n=== Step 3: Loss and metrics sanity check ===")
    from src.segmentation.losses import CombinedLoss
    from src.segmentation.metrics import SegmentationMetrics

    loss_fn = CombinedLoss()
    metrics = SegmentationMetrics()

    logits = torch.zeros(4, 1, 256, 256)
    targets = torch.zeros(4, 1, 256, 256)
    targets[:, :, 64:128, 64:128] = 1.0  # inject a burn region

    loss = loss_fn(logits, targets)
    print("Loss:", loss.item())
    assert loss.item() > 0, f"expected positive loss, got {loss.item()}"

    metrics.update(logits, targets)
    result = metrics.compute()
    print(result)  # iou/dice will be low (logits=0)
    assert 0.0 <= result["iou"] < 1.0, f"unexpected iou {result['iou']}"
    assert 0.0 <= result["dice"] < 1.0, f"unexpected dice {result['dice']}"

    # Regression check for the _batch_fnr fix: a sample with no positive pixel in
    # the target has nothing to miss, so it must not count as fnr=1.0 ("missed
    # everything") — this matters once negative-origin samples are mixed into
    # every batch at negative_fraction.
    metrics.reset()
    all_negative_targets = torch.zeros(4, 1, 256, 256)
    metrics.update(torch.zeros(4, 1, 256, 256), all_negative_targets)
    neg_result = metrics.compute()
    print("all-negative batch:", neg_result)
    assert neg_result["fnr"] == 0.0, f"expected fnr=0.0 for an all-negative batch, got {neg_result['fnr']}"
    print("Step 3: OK")


def step4_short_training_run() -> None:
    print("\n=== Step 4: Short training run (2 epochs) ===")
    import contextlib
    import io
    import re

    from src.segmentation.train import train

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train(
            batch_size=4,
            epochs=2,
            output_dir=TRAIN_OUTPUT_DIR,
            num_workers=0,
        )
    log = buf.getvalue()
    print(log)

    train_losses = [float(v) for v in re.findall(r"train_loss=([\d.]+)", log)]
    assert len(train_losses) == 2, f"expected 2 epoch lines, got {train_losses}"
    assert train_losses[1] < train_losses[0], (
        f"train loss did not decrease: epoch1={train_losses[0]}, epoch2={train_losses[1]} "
        "(try increasing --pos_weight if burn scars are sparse in sampled batches)"
    )
    assert os.path.isdir(f"{TRAIN_OUTPUT_DIR}/best_model"), "checkpoint directory not found"
    print("Step 4: OK")


def step5_checkpoint_reload() -> None:
    print("\n=== Step 5: Checkpoint reload ===")
    from src.segmentation.model import BurnScarSegmenter

    checkpoint_dir = f"{TRAIN_OUTPUT_DIR}/best_model"
    assert os.path.isdir(checkpoint_dir), (
        f"no checkpoint at {checkpoint_dir} — run --step 4 first to produce one"
    )
    m = BurnScarSegmenter.load_pretrained(checkpoint_dir)
    print("Reload OK")
    y = m(torch.randn(1, 6, 256, 256))
    print("Inference shape:", y.shape)
    assert y.shape == (1, 1, 256, 256), f"unexpected inference shape {y.shape}"
    print("Step 5: OK")


def step6_predict_mask() -> None:
    print("\n=== Step 6: predict_mask() on a real training image ===")
    import numpy as np

    from src.segmentation.predict import predict_mask
    from src.severity.dataset import list_burn_positive_train_samples, read_mask, read_raw_bands

    # Reset module-level cache so this step doesn't depend on prior steps having run in-process.
    import src.segmentation.predict as predict_mod
    predict_mod._model = None

    pool = list_burn_positive_train_samples()
    sample_id, merged_path, mask_path = pool[0]
    bands = read_raw_bands(merged_path)  # native 512x512 res — UNet is fully convolutional
    gt_mask = read_mask(mask_path)

    pred = predict_mask(bands)
    print(f"sample_id={sample_id} predicted burned pixels={int(pred.sum())} "
          f"ground-truth burned pixels={int(gt_mask.sum())} out of {pred.size}")
    assert pred.shape == gt_mask.shape, f"shape mismatch: {pred.shape} vs {gt_mask.shape}"
    assert set(np.unique(pred).tolist()) <= {0, 1}, f"unexpected mask values {np.unique(pred)}"
    print("Step 6: OK")


def step7_negative_mixing_training() -> None:
    print("\n=== Step 7: Training run with synthetic negative-dir mixing ===")
    import contextlib
    import io
    import re
    import tempfile

    from src.segmentation.train import train

    with tempfile.TemporaryDirectory() as tmp:
        neg_dir = _make_synthetic_negative_dir(tmp)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            train(
                batch_size=4,
                epochs=1,
                output_dir=f"{TRAIN_OUTPUT_DIR}_negmix",
                num_workers=0,
                negative_dir=neg_dir,
                negative_fraction=0.3,
            )
        log = buf.getvalue()
        print(log)

        train_losses = [float(v) for v in re.findall(r"train_loss=([\d.]+)", log)]
        assert len(train_losses) == 1, f"expected 1 epoch line, got {train_losses}"
    print("Step 7: OK")


STEPS = {
    1: step1_dataset,
    2: step2_model_forward,
    3: step3_loss_and_metrics,
    4: step4_short_training_run,
    5: step5_checkpoint_reload,
    6: step6_predict_mask,
    7: step7_negative_mixing_training,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one or all steps of the segmentation pipeline sanity check."
    )
    p.add_argument(
        "--step", type=int, choices=sorted(STEPS), default=None,
        help="Run only this step (1-7). Omit to run all steps in order.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.step is not None:
        STEPS[args.step]()
    else:
        for step_fn in STEPS.values():
            step_fn()
        print("\nAll sanity checks passed.")