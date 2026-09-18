"""
Bulk-sample negative (no-damage) Sentinel-2 patches from GEE and write them
to disk in BurnScarDataset's on-disk format, under a directory kept separate
from the pristine HLS burn-scar download.

Usage:
    python scripts/fetch_negative_samples.py --n_target 150 --out_dir dataset/negative
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retrieval.negative_sampling import generate_negative_samples

DEFAULT_OUT_DIR = "/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/dataset/negative"


def main(n_target: int, out_dir: str, val_fraction: float, seed: int) -> None:
    result = generate_negative_samples(
        n_target=n_target, out_dir=out_dir, val_fraction=val_fraction, seed=seed,
    )
    print(f"Written: {result['n_written']} / {n_target} target")
    print(f"Attempts: {result['n_attempts']}")
    print(f"Rejected (non-vegetated): {result['n_rejected_vegetation']}")
    print(f"Rejected (cloud/nodata): {result['n_rejected_cloud_or_nodata']}")
    print(f"Written under: {out_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk-sample negative (no-damage) patches from GEE.")
    p.add_argument("--n_target", type=int, default=150)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(n_target=args.n_target, out_dir=args.out_dir, val_fraction=args.val_fraction, seed=args.seed)
