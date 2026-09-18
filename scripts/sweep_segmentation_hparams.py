"""
Small hyperparameter sweep over CombinedLoss's alpha/pos_weight, under the
production negative-mixing regime (negative_dir + negative_fraction=0.2,
batch_size=2/accumulation_steps=4).

Runs each config for a reduced epoch count to cheaply rank candidates before
committing a full-length confirmation run (via scripts/train_segmentation.py)
for the winner. Writes one row per config to summary.csv immediately after
that config finishes, so results survive a mid-sweep preemption on the
studentkillable partition.

Usage:
    python scripts/sweep_segmentation_hparams.py
    python scripts/sweep_segmentation_hparams.py --epochs 15 --out_dir checkpoints/segmentation_sweep
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.segmentation.train import train

NEGATIVE_DIR = "/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/dataset/negative"

# (label, alpha, pos_weight) — see plan doc for per-config rationale.
CONFIGS = [
    ("A_a0.30_pw7",  0.30, 7.0),
    ("B_a0.30_pw14", 0.30, 14.0),
    ("C_a0.30_pw20", 0.30, 20.0),
    ("D_a0.50_pw10", 0.50, 10.0),
    ("E_a0.15_pw10", 0.15, 10.0),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep alpha/pos_weight for burn-scar segmentation.")
    p.add_argument("--epochs", type=int, default=15,
                   help="Epochs per config in the search phase (default: 15)")
    p.add_argument("--out_dir", type=str,
                   default="/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/checkpoints/segmentation_sweep",
                   help="Root directory for sweep checkpoints/summary")
    p.add_argument("--seed", type=int, default=42,
                   help="Shared seed across all configs, isolating the loss-hyperparameter effect")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "summary.csv")

    fieldnames = ["label", "alpha", "pos_weight", "epochs",
                  "best_val_iou", "best_val_dice", "best_val_fpr", "best_val_fnr",
                  "elapsed_seconds"]
    with open(summary_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for label, alpha, pos_weight in CONFIGS:
        config_dir = os.path.join(args.out_dir, label)
        print(f"\n=== Config {label}: alpha={alpha}, pos_weight={pos_weight} ===")

        start = time.perf_counter()
        train(
            alpha=alpha,
            pos_weight=pos_weight,
            epochs=args.epochs,
            output_dir=config_dir,
            negative_dir=NEGATIVE_DIR,
            negative_fraction=0.2,
            batch_size=2,
            accumulation_steps=4,
            seed=args.seed,
        )
        elapsed = time.perf_counter() - start

        state = torch.load(os.path.join(config_dir, "training_state.pt"), weights_only=False)
        row = {
            "label": label,
            "alpha": alpha,
            "pos_weight": pos_weight,
            "epochs": args.epochs,
            "best_val_iou": state["best_val_iou"],
            "best_val_dice": state["best_val_dice"],
            "best_val_fpr": state["best_val_fpr"],
            "best_val_fnr": state["best_val_fnr"],
            "elapsed_seconds": round(elapsed),
        }
        with open(summary_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        print(f"--> {label}: val_iou={row['best_val_iou']:.4f} "
              f"val_dice={row['best_val_dice']:.4f} val_fpr={row['best_val_fpr']:.4f} "
              f"val_fnr={row['best_val_fnr']:.4f} ({elapsed:.0f}s)")

    print(f"\nSweep complete. Summary: {summary_path}")
    with open(summary_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
