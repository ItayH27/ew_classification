"""
CLI entry point for training the severity classifier.

Usage:
    python scripts/train_severity.py [options]

Examples:
    # Default training run (5-fold CV, then refit on all labeled samples)
    python scripts/train_severity.py

    # Custom labels file / output directory
    python scripts/train_severity.py --labels_csv data/severity_labels.csv --output_dir checkpoints/run1
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.severity.train import train


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the severity classifier (LogisticRegression over mean NDVI/NBR)."
    )
    p.add_argument("--labels_csv", type=str,
                   default="/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/data/severity_labels.csv",
                   help="Path to the manually-labeled severity CSV (default: <project>/data/severity_labels.csv)")
    p.add_argument("--output_dir", type=str,
                   default="/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/checkpoints/severity",
                   help="Directory for the saved model (default: <project>/checkpoints/severity)")
    p.add_argument("--n_splits", type=int, default=5,
                   help="Number of stratified CV folds (default: 5, auto-reduced if a class is smaller)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for CV shuffling (default: 42)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        seed=args.seed,
    )
