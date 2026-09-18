"""
Measure how much the training seed alone moves final segmentation metrics.

Holds the production loss/data config fixed (see scripts/slurm_files/train_segmentation.slurm:
alpha=0.3, pos_weight=7, batch_size=2, accumulation_steps=4, negative_dir + negative_fraction=0.2)
and re-runs training once per seed, at a reduced epoch count to keep this cheap (mirrors
scripts/sweep_segmentation_hparams.py's cheap-ranking budget). Writes one row per seed to
summary.csv immediately after that seed finishes, so results survive a mid-sweep preemption on
the studentkillable partition, then prints mean/stdev/min/max across seeds for each metric.

Usage:
    python scripts/seed_sensitivity_segmentation.py
    python scripts/seed_sensitivity_segmentation.py --epochs 30 --seeds 1 2 3
"""

import argparse
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.segmentation.train import train

NEGATIVE_DIR = "/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/dataset/negative"

SEEDS = [7, 42, 67, 69, 420]

METRIC_KEYS = ["best_val_iou", "best_val_dice", "best_val_fpr", "best_val_fnr"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure segmentation training's sensitivity to the random seed."
    )
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                   help=f"Seeds to compare (default: {SEEDS})")
    p.add_argument("--epochs", type=int, default=15,
                   help="Epochs per seed run (default: 15)")
    p.add_argument("--out_dir", type=str,
                   default="/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/checkpoints/segmentation_seed_sensitivity",
                   help="Root directory for per-seed checkpoints/summary")
    p.add_argument("--alpha", type=float, default=0.3,
                   help="Weight for the BCE term vs. Dice in CombinedLoss (default: 0.3, production value)")
    p.add_argument("--pos_weight", type=float, default=7.0,
                   help="BCE positive-class weight (default: 7, production value)")
    p.add_argument("--batch_size", type=int, default=2,
                   help="Samples per gradient step (default: 2, production value)")
    p.add_argument("--accumulation_steps", type=int, default=4,
                   help="Gradient accumulation steps (default: 4, production value)")
    p.add_argument("--negative_dir", type=str, default=NEGATIVE_DIR,
                   help="Directory of GEE-sourced negative samples (default: production dataset/negative)")
    p.add_argument("--negative_fraction", type=float, default=0.2,
                   help="Target fraction of negative-origin samples per training batch (default: 0.2)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "summary.csv")

    print(
        f"Fixed config: alpha={args.alpha}, pos_weight={args.pos_weight}, "
        f"batch_size={args.batch_size}, accumulation_steps={args.accumulation_steps}, "
        f"negative_dir={args.negative_dir}, negative_fraction={args.negative_fraction}, "
        f"epochs={args.epochs}"
    )
    print(f"Seeds: {args.seeds}")

    fieldnames = ["seed", "epochs"] + METRIC_KEYS + ["elapsed_seconds"]
    with open(summary_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    rows = []
    for seed in args.seeds:
        seed_dir = os.path.join(args.out_dir, f"seed_{seed}")
        print(f"\n=== Seed {seed} ===")

        start = time.perf_counter()
        train(
            alpha=args.alpha,
            pos_weight=args.pos_weight,
            epochs=args.epochs,
            output_dir=seed_dir,
            negative_dir=args.negative_dir,
            negative_fraction=args.negative_fraction,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            seed=seed,
        )
        elapsed = time.perf_counter() - start

        state = torch.load(os.path.join(seed_dir, "training_state.pt"), weights_only=False)
        row = {
            "seed": seed,
            "epochs": args.epochs,
            "best_val_iou": state["best_val_iou"],
            "best_val_dice": state["best_val_dice"],
            "best_val_fpr": state["best_val_fpr"],
            "best_val_fnr": state["best_val_fnr"],
            "elapsed_seconds": round(elapsed),
        }
        rows.append(row)
        with open(summary_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        print(f"--> seed={seed}: val_iou={row['best_val_iou']:.4f} "
              f"val_dice={row['best_val_dice']:.4f} val_fpr={row['best_val_fpr']:.4f} "
              f"val_fnr={row['best_val_fnr']:.4f} ({elapsed:.0f}s)")

    print(f"\nSweep complete. Summary: {summary_path}")
    with open(summary_path) as f:
        print(f.read())

    print("Seed sensitivity (across {} seeds):".format(len(rows)))
    for key in METRIC_KEYS:
        values = [row[key] for row in rows]
        mean = statistics.mean(values)
        stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
        print(
            f"  {key}: mean={mean:.4f}  stdev={stdev:.4f}  "
            f"min={min(values):.4f}  max={max(values):.4f}  range={max(values) - min(values):.4f}"
        )


if __name__ == "__main__":
    main()
