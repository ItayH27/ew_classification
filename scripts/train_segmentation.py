"""
CLI entry point for training the burn-scar segmentation model.

Usage:
    python scripts/train_segmentation.py [options]

Examples:
    # Default training run (30 epochs, batch 8, lr 1e-4)
    python scripts/train_segmentation.py

    # Smaller GPU: effective batch 8 via accumulation
    python scripts/train_segmentation.py --batch_size 2 --accumulation_steps 4

    # Custom output directory
    python scripts/train_segmentation.py --epochs 50 --output_dir checkpoints/run1

    # Dice-heavy loss with a lighter positive-class weight
    python scripts/train_segmentation.py --alpha 0.3 --pos_weight 7

    # Mix in GEE-sourced negatives (see scripts/fetch_negative_samples.py)
    python scripts/train_segmentation.py --negative_dir dataset/negative --negative_fraction 0.2

    # Reproducible run
    python scripts/train_segmentation.py --seed 42
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.segmentation.train import train


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune UNet2DModel on the HLS Burn Scar Scenes dataset."
    )
    p.add_argument("--batch_size",         type=int,   default=8,
                   help="Samples per gradient step (default: 8)")
    p.add_argument("--lr",                 type=float, default=1e-4,
                   help="AdamW learning rate (default: 1e-4)")
    p.add_argument("--epochs",             type=int,   default=30,
                   help="Number of training epochs (default: 30)")
    p.add_argument("--output_dir",         type=str,   default="/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/checkpoints/burn_scar",
                   help="Directory for checkpoints (default: <project>/checkpoints/burn_scar)")
    p.add_argument("--accumulation_steps", type=int,   default=1,
                   help="Gradient accumulation steps; effective batch = batch_size × steps (default: 1)")
    p.add_argument("--pos_weight",         type=float, default=10.0,
                   help="BCE positive-class weight to counter class imbalance (default: 10)")
    p.add_argument("--alpha",              type=float, default=0.5,
                   help="Weight for the BCE term vs. Dice in CombinedLoss (default: 0.5)")
    p.add_argument("--num_workers",        type=int,   default=4,
                   help="DataLoader worker processes (default: 4)")
    p.add_argument("--negative_dir",       type=str,   default=None,
                   help="Directory of GEE-sourced negative samples (see "
                        "scripts/fetch_negative_samples.py); omit to train on HLS "
                        "burn-positive data only (default)")
    p.add_argument("--negative_fraction",  type=float, default=0.0,
                   help="Target fraction of negative-origin samples per training batch "
                        "when --negative_dir is set (default: 0.0)")
    p.add_argument("--seed",               type=int,   default=None,
                   help="Random seed for reproducible runs (default: unseeded)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        output_dir=args.output_dir,
        accumulation_steps=args.accumulation_steps,
        pos_weight=args.pos_weight,
        alpha=args.alpha,
        num_workers=args.num_workers,
        negative_dir=args.negative_dir,
        negative_fraction=args.negative_fraction,
        seed=args.seed,
    )
