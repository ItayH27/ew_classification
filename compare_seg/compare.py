"""
Compares our BurnScarSegmenter against ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars on two,
separate axes:

1. Burn-scar sensitivity (IoU/Dice/FPR/FNR) on the HLS Burn Scar Scenes validation split.
   Ground-truth masks are at native 512x512 resolution. Prithvi runs natively at 512x512 (matching
   the raw chip size exactly). Our model was only ever trained/validated at 256x256 — running it at
   native 512 is out-of-distribution and empirically collapses its accuracy — so it's evaluated by
   downsampling the input to 256, then upsampling its prediction back to 512 (nearest) to share a
   pixel grid with the ground truth and with Prithvi's output. Nodata/cloud pixels (mask == -1) are
   excluded from IoU/Dice/FPR/FNR for both models, matching how Prithvi itself was trained/evaluated.

   Since ~67% of our validation chips overlap with Prithvi's own training split (see
   split_utils.py), results are reported for both the full validation split and a leak-free subset,
   computed from the same inference pass so both numbers come from identical measurements.

2. No-damage specificity (false-positive rate only) on GEE-sourced negative samples
   (dataset/negative/validation/, all-zero ground-truth masks). IoU/Dice/FNR are undefined/
   degenerate on all-negative targets, so this pass reports FPR alone, in a separate table — not
   blended into the IoU comparison above, since the two measure different things (and Prithvi,
   never trained on negatives, shouldn't be penalized on a metric it wasn't optimized for). These
   chips are native 256x256; our model evaluates at that resolution natively (no resize needed),
   while Prithvi is upsampled to its own 512x512 training resolution for inference, with its
   prediction downsampled back to native 256 for comparison (see prithvi_model.predict_prithvi).

Usage:
    python compare.py                        # full comparison, both models, both axes
    python compare.py --models ours --limit 5   # quick smoke test, no terratorch needed
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import rasterio
import torch

from src.segmentation.dataset import HLS_BAND_MEANS, HLS_BAND_STDS, BurnScarDataset
from src.segmentation.metrics import _batch_dice, _batch_fnr, _batch_fpr, _batch_iou
from src.segmentation.model import BurnScarSegmenter

from split_utils import (
    build_clean_chip_ids,
    download_ibm_splits,
    list_negative_chips,
    list_validation_chips,
)

_DEFAULT_CHECKPOINT = "/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/checkpoints/burn_scar/best_model"
_DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
_DEFAULT_NEGATIVES_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "negative")

# BurnScarSegmenter was only ever trained/validated at this resolution (BurnScarDataset's default
# target_size). Although it's fully convolutional and *can* run at native 512x512 without shape
# errors, doing so is out-of-distribution and tanks its accuracy (empirically verified: IoU on the
# same chips drops from ~0.85-0.99 at 256 to near-zero on several chips at native 512). So it's
# evaluated at this resolution, with predictions upsampled (nearest) back to native size to share
# a pixel grid with the ground-truth mask and with Prithvi's native-512 predictions.
_OURS_EVAL_SIZE = 256


def _get_device(preference: str = "auto") -> torch.device:
    if preference == "cuda":
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _read_raw_mask(path: str) -> torch.Tensor:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W), values in {-1, 0, 1}


def _read_chip(merged_path: str, mask_path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        image_raw:  (6, H, W) raw unnormalized reflectance
        target:     (1, H, W) float in {0., 1.}
        valid_mask: (1, H, W) float in {0., 1.} — 0 where the ground-truth mask is nodata (-1)
    """
    image_raw = BurnScarDataset._read_image(merged_path)
    mask_raw = _read_raw_mask(mask_path)
    valid_mask = (mask_raw >= 0).float()
    target = torch.clamp(mask_raw, min=0.0)
    return image_raw, target, valid_mask


def _predict_ours(
    model: BurnScarSegmenter, image_raw: torch.Tensor, device: torch.device, threshold: float
) -> torch.Tensor:
    """Returns a (1, 1, H, W) prediction at image_raw's native H, W (see _OURS_EVAL_SIZE note above)."""
    native_h, native_w = image_raw.shape[-2:]
    means = torch.tensor(HLS_BAND_MEANS, dtype=torch.float32).view(6, 1, 1).to(device)
    stds = torch.tensor(HLS_BAND_STDS, dtype=torch.float32).view(6, 1, 1).to(device)

    x = image_raw.to(device).unsqueeze(0)  # (1, 6, H, W)
    x = torch.nn.functional.interpolate(
        x, size=(_OURS_EVAL_SIZE, _OURS_EVAL_SIZE), mode="bilinear", align_corners=False
    )
    x = (x - means) / stds

    with torch.no_grad():
        logits = model(x)
        preds = (torch.sigmoid(logits) > threshold).float()  # (1, 1, _OURS_EVAL_SIZE, _OURS_EVAL_SIZE)

    preds = torch.nn.functional.interpolate(preds, size=(native_h, native_w), mode="nearest")
    return preds


def run_eval(model_name, predict_fn, chip_list, device) -> list[dict]:
    rows = []
    for chip_id, merged_path, mask_path in chip_list:
        image_raw, target, valid_mask = _read_chip(merged_path, mask_path)
        target_b = target.unsqueeze(0).to(device)      # (1, 1, H, W)
        valid_b = valid_mask.unsqueeze(0).to(device)    # (1, 1, H, W)

        start = time.perf_counter()
        preds = predict_fn(image_raw)
        latency = time.perf_counter() - start

        fnr_mean, fnr_n_valid = _batch_fnr(preds, target_b, valid_b)

        rows.append({
            "chip_id": chip_id,
            "model": model_name,
            "iou": _batch_iou(preds, target_b, valid_b),
            "dice": _batch_dice(preds, target_b, valid_b),
            "fpr": _batch_fpr(preds, target_b, valid_b),
            "fnr": fnr_mean,
            "fnr_valid": fnr_n_valid > 0,
            "latency_s": latency,
        })
    return rows


def aggregate(rows: list[dict], chip_ids: set[str] | None = None) -> dict:
    if chip_ids is not None:
        rows = [r for r in rows if r["chip_id"] in chip_ids]
    if not rows:
        return {"n_chips": 0, "iou": None, "dice": None, "fpr": None, "fnr": None,
                "avg_latency_s": None, "total_time_s": 0.0}
    # fnr excludes chips with no positive pixel in the target (nothing to miss ≠ missed
    # everything) — same rationale as SegmentationMetrics.compute() in src/segmentation/metrics.py.
    fnr_rows = [r for r in rows if r["fnr_valid"]]
    return {
        "n_chips": len(rows),
        "iou": statistics.mean(r["iou"] for r in rows),
        "dice": statistics.mean(r["dice"] for r in rows),
        "fpr": statistics.mean(r["fpr"] for r in rows),
        "fnr": statistics.mean(r["fnr"] for r in fnr_rows) if fnr_rows else None,
        "avg_latency_s": statistics.mean(r["latency_s"] for r in rows),
        "total_time_s": sum(r["latency_s"] for r in rows),
    }


def _print_summary(model_name: str, n_params: int, full: dict, clean: dict) -> None:
    print(f"\n=== {model_name} ({n_params / 1e6:.1f}M params) ===")
    for label, agg in (("full", full), ("clean", clean)):
        fnr_str = f"{agg['fnr']:.4f}" if agg["fnr"] is not None else "n/a"
        print(
            f"  [{label:5s}] n={agg['n_chips']:3d} | "
            f"iou={agg['iou']:.4f} | dice={agg['dice']:.4f} | "
            f"fpr={agg['fpr']:.4f} | fnr={fnr_str} | "
            f"avg_latency={agg['avg_latency_s']:.4f}s"
        )


def _print_negative_summary(model_name: str, n_params: int, agg: dict) -> None:
    print(f"\n=== {model_name} ({n_params / 1e6:.1f}M params) — negative samples, no-damage specificity ===")
    print(f"  n={agg['n_chips']:3d} | fpr={agg['fpr']:.4f} | avg_latency={agg['avg_latency_s']:.4f}s")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare BurnScarSegmenter vs. Prithvi-EO-2.0-300M-BurnScars on the HLS validation split."
    )
    p.add_argument("--checkpoint_dir", type=str, default=_DEFAULT_CHECKPOINT,
                    help=f"Our model's checkpoint directory (default: {_DEFAULT_CHECKPOINT})")
    p.add_argument("--models", type=str, choices=["both", "ours", "prithvi"], default="both",
                    help="Which model(s) to evaluate (default: both)")
    p.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto",
                    help="Device to run inference on (default: auto)")
    p.add_argument("--threshold", type=float, default=0.5,
                    help="Sigmoid threshold for our model (default: 0.5)")
    p.add_argument("--limit", type=int, default=None,
                    help="Cap the number of validation chips evaluated, for smoke-testing (default: all 264)")
    p.add_argument("--output_dir", type=str, default=_DEFAULT_OUTPUT_DIR,
                    help=f"Where to write summary.json/per_chip.csv (default: {_DEFAULT_OUTPUT_DIR})")
    p.add_argument("--cache_dir", type=str, default=None,
                    help="Optional cache_dir override for HF Hub downloads (dataset + Prithvi assets)")
    p.add_argument("--negatives_dir", type=str, default=_DEFAULT_NEGATIVES_DIR,
                    help=f"GEE-sourced negative samples dir, expects a validation/ subfolder "
                         f"(default: {_DEFAULT_NEGATIVES_DIR})")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = _get_device(args.device)
    print(f"Device: {device}")

    print("Listing validation chips...")
    chip_list = list_validation_chips(args.cache_dir)
    if args.limit is not None:
        chip_list = chip_list[: args.limit]
    print(f"Evaluating on {len(chip_list)} chips.")

    print("Downloading IBM's Prithvi-BurnScars splits to derive the leak-free subset...")
    ibm_splits = download_ibm_splits(args.cache_dir)
    clean_ids = build_clean_chip_ids([c[0] for c in chip_list], ibm_splits)
    print(f"Leak-free subset: {len(clean_ids)} chips (unseen by Prithvi during its own training).")

    print("\nListing negative (no-damage) validation chips...")
    neg_chip_list = list_negative_chips(args.negatives_dir)
    if args.limit is not None:
        neg_chip_list = neg_chip_list[: args.limit]
    if not neg_chip_list:
        print(f"No negative chips found under {args.negatives_dir}/validation — "
              f"skipping negative-sample (specificity) comparison.")
    else:
        print(f"Evaluating on {len(neg_chip_list)} negative chips.")

    all_rows: list[dict] = []
    all_negative_rows: list[dict] = []
    results: dict = {
        "run_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv[1:],
            "checkpoint_dir": args.checkpoint_dir,
            "n_full": len(chip_list),
            "n_clean": len(clean_ids),
            "n_negatives": len(neg_chip_list),
            "negatives_dir": args.negatives_dir,
        }
    }

    if args.models in ("both", "ours"):
        print("\nLoading our model...")
        model = BurnScarSegmenter.load_pretrained(args.checkpoint_dir).to(device)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())

        def predict_fn(image_raw, _model=model, _device=device, _thr=args.threshold):
            return _predict_ours(_model, image_raw, _device, _thr)

        rows = run_eval("ours", predict_fn, chip_list, device)
        all_rows.extend(rows)
        full, clean = aggregate(rows), aggregate(rows, clean_ids)
        _print_summary("ours", n_params, full, clean)
        results["ours"] = {"n_params": n_params, "full": full, "clean": clean}

        if neg_chip_list:
            neg_rows = run_eval("ours", predict_fn, neg_chip_list, device)
            all_negative_rows.extend(neg_rows)
            neg_agg = aggregate(neg_rows)
            _print_negative_summary("ours", n_params, neg_agg)
            results["ours"]["negatives"] = {
                "n_chips": neg_agg["n_chips"], "fpr": neg_agg["fpr"],
                "avg_latency_s": neg_agg["avg_latency_s"],
            }

    if args.models in ("both", "prithvi"):
        from prithvi_model import count_params, load_prithvi_model, predict_prithvi

        print("\nLoading Prithvi-EO-2.0-300M-BurnScars (downloads ~1.3GB on first run)...")
        lightning_model = load_prithvi_model(device, args.cache_dir)
        n_params = count_params(lightning_model)

        def predict_fn(image_raw, _lm=lightning_model, _device=device):
            return predict_prithvi(_lm, image_raw, _device)

        rows = run_eval("prithvi", predict_fn, chip_list, device)
        all_rows.extend(rows)
        full, clean = aggregate(rows), aggregate(rows, clean_ids)
        _print_summary("prithvi", n_params, full, clean)
        results["prithvi"] = {"n_params": n_params, "full": full, "clean": clean}

        if neg_chip_list:
            neg_rows = run_eval("prithvi", predict_fn, neg_chip_list, device)
            all_negative_rows.extend(neg_rows)
            neg_agg = aggregate(neg_rows)
            _print_negative_summary("prithvi", n_params, neg_agg)
            results["prithvi"]["negatives"] = {
                "n_chips": neg_agg["n_chips"], "fpr": neg_agg["fpr"],
                "avg_latency_s": neg_agg["avg_latency_s"],
            }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {summary_path}")

    per_chip_path = os.path.join(args.output_dir, "per_chip.csv")
    with open(per_chip_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["chip_id", "model", "in_clean_subset", "iou", "dice", "fpr", "fnr", "latency_s"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow({**row, "in_clean_subset": row["chip_id"] in clean_ids})
    print(f"Wrote {per_chip_path}")

    if all_negative_rows:
        # Deliberately its own file (not appended to per_chip.csv) and only carries fpr/latency_s
        # — iou/dice/fnr are degenerate on these all-negative targets, so surfacing them here would
        # blend a meaningless number in next to the real specificity signal.
        negative_per_chip_path = os.path.join(args.output_dir, "negative_per_chip.csv")
        with open(negative_per_chip_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["chip_id", "model", "fpr", "latency_s"])
            writer.writeheader()
            for row in all_negative_rows:
                writer.writerow({k: row[k] for k in ("chip_id", "model", "fpr", "latency_s")})
        print(f"Wrote {negative_per_chip_path}")


if __name__ == "__main__":
    main()
