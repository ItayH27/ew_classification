"""
End-to-end sanity check for the severity classification pipeline.

Runs, in order:
    1. Dataset inspection      — burn-positive sample pool + feature extraction on a real image,
                                  plus the known_negative feature contract and the negative-pool
                                  listing's graceful-empty behavior
    2. Model roundtrip         — SeverityClassifier fit/predict on synthetic 4-class data
    3. Training run            — build_dataset + cross_validate + save on data/severity_labels.csv
                                  (requires at least a handful of real labels; see
                                  scripts/label_severity_samples.py)
    4. Checkpoint reload       — reload the saved model and run classify_severity()

Usage:
    python scripts/sanity_check_severity.py            # run all steps in order
    python scripts/sanity_check_severity.py --step 2    # run only step 2
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

LABELS_CSV = "/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/data/severity_labels.csv"
TRAIN_OUTPUT_DIR = "/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/checkpoints/sanity_test_severity"
MIN_LABELS_FOR_CV = 5  # StratifiedKFold needs at least this many per the smallest class


def step1_dataset() -> None:
    print("\n=== Step 1: Dataset inspection ===")
    from src.severity.dataset import list_burn_positive_train_samples, read_mask, read_raw_bands
    from src.severity.features import extract_features_from_bands

    pool = list_burn_positive_train_samples()
    print("Burn-positive pool size:", len(pool))
    assert len(pool) > 0, "expected at least one burn-positive training sample"
    assert len(pool) <= 540, f"pool should be a subset of the 540 training samples, got {len(pool)}"

    sample_id, merged_path, mask_path = pool[0]
    bands = read_raw_bands(merged_path)
    mask = read_mask(mask_path)
    print("bands shape:", bands.shape)
    assert bands.shape[-1] == 6, f"expected 6-band image, got shape {bands.shape}"

    whole_ndvi, whole_nbr, masked_ndvi, masked_nbr, burned_fraction = extract_features_from_bands(bands, mask)
    print(f"sample_id={sample_id} burned_fraction={burned_fraction:.3f} "
          f"whole=({whole_ndvi:.3f}, {whole_nbr:.3f}) masked=({masked_ndvi:.3f}, {masked_nbr:.3f})")
    for name, val in [("whole_ndvi", whole_ndvi), ("masked_ndvi", masked_ndvi)]:
        assert -1.0 <= val <= 1.0, f"{name} out of range: {val}"
    for name, val in [("whole_nbr", whole_nbr), ("masked_nbr", masked_nbr)]:
        assert -1.0 <= val <= 1.0, f"{name} out of range: {val}"
    assert 0.0 <= burned_fraction <= 1.0, f"burned_fraction out of range: {burned_fraction}"
    if burned_fraction < 1.0:
        assert (masked_ndvi, masked_nbr) != (whole_ndvi, whole_nbr), "masked mean should differ from whole-image mean"

    from src.severity.dataset import list_known_negative_train_samples

    neg_pool = list_known_negative_train_samples()
    print("Known-negative pool size:", len(neg_pool))
    assert isinstance(neg_pool, list), "list_known_negative_train_samples must not raise when the dir is absent"

    known_negative_feats = extract_features_from_bands(bands, mask=None, known_negative=True)
    ambiguous_feats = extract_features_from_bands(bands, mask=None, known_negative=False)
    print("known_negative burned_fraction:", known_negative_feats[-1])
    print("ambiguous burned_fraction:", ambiguous_feats[-1])
    assert known_negative_feats[-1] == 0.0, "confirmed negative should report burned_fraction=0.0"
    assert ambiguous_feats[-1] == 1.0, "ambiguous empty-mask case should keep the conservative burned_fraction=1.0"
    print("Step 1: OK")


def step2_model_roundtrip() -> None:
    print("\n=== Step 2: Model fit/predict roundtrip (synthetic data) ===")
    from src.severity.model import SeverityClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    y = np.tile([0, 1, 2, 3], 10)

    model = SeverityClassifier().fit(X, y)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    print("preds shape:", preds.shape, "proba shape:", proba.shape)
    assert preds.shape == (40,)
    assert proba.shape == (40, 4)
    assert set(preds.tolist()) <= {0, 1, 2, 3}
    print("Step 2: OK")


def step3_training_run() -> None:
    print("\n=== Step 3: Training run on data/severity_labels.csv ===")
    if not os.path.exists(LABELS_CSV):
        print(f"No labels file at {LABELS_CSV} yet — run scripts/label_severity_samples.py first. Skipping.")
        return

    import csv as _csv
    with open(LABELS_CSV, newline="") as f:
        rows = list(_csv.DictReader(f))
    counts: dict[int, int] = {}
    for row in rows:
        counts[int(row["severity"])] = counts.get(int(row["severity"]), 0) + 1
    if not counts or min(counts.values()) < MIN_LABELS_FOR_CV:
        print(f"Not enough labels per class yet for CV (need >= {MIN_LABELS_FOR_CV} per class): "
              f"{counts}. Skipping.")
        return

    from src.severity.train import train

    train(labels_csv=LABELS_CSV, output_dir=TRAIN_OUTPUT_DIR, n_splits=5, seed=42)
    assert os.path.isfile(os.path.join(TRAIN_OUTPUT_DIR, "model.joblib")), "checkpoint not found"
    print("Step 3: OK")


def step4_checkpoint_reload() -> None:
    print("\n=== Step 4: Checkpoint reload + classify_severity() ===")
    checkpoint_path = os.path.join(TRAIN_OUTPUT_DIR, "model.joblib")
    if not os.path.isfile(checkpoint_path):
        print(f"No checkpoint at {checkpoint_path} — run --step 3 first (requires real labels). Skipping.")
        return

    from src.severity.predict import classify_severity

    # Reset module-level cache so this step doesn't depend on step 3 having run in-process.
    import src.severity.predict as predict_mod
    predict_mod._model = None

    ndvi = np.full((32, 32), 0.3, dtype=np.float32)
    nbr = np.full((32, 32), 0.2, dtype=np.float32)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1  # synthetic burn region; classify_severity_from_retrieval's
    # segmentation-integration path (src.segmentation.predict.predict_mask) isn't
    # covered here since it needs real GEE-retrieved imagery, not synthetic arrays.
    result = classify_severity(ndvi, nbr, mask=mask, checkpoint_dir=TRAIN_OUTPUT_DIR)
    print("classify_severity result:", result)
    assert result["severity"] in {0, 1, 2, 3}, f"unexpected severity {result['severity']}"
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6
    print("Step 4: OK")


STEPS = {
    1: step1_dataset,
    2: step2_model_roundtrip,
    3: step3_training_run,
    4: step4_checkpoint_reload,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one or all steps of the severity classification pipeline sanity check."
    )
    p.add_argument(
        "--step", type=int, choices=sorted(STEPS), default=None,
        help="Run only this step (1-4). Omit to run all steps in order.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.step is not None:
        STEPS[args.step]()
    else:
        for step_fn in STEPS.values():
            step_fn()
        print("\nAll sanity checks passed (or skipped where labels are missing).")
