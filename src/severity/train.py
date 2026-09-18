"""
Train the severity classifier: stratified k-fold CV for evaluation metrics,
then refit on all labeled samples for the final saved model.
"""

from __future__ import annotations

import csv
import time
from datetime import timedelta

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold

from .dataset import list_all_train_image_paths, mask_path_for, read_mask, read_raw_bands, sample_id_from_path
from .features import extract_features_from_bands
from .model import SeverityClassifier

LABELS_CSV = "/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/data/severity_labels.csv"


def _load_labels(csv_path: str) -> dict[str, int]:
    with open(csv_path, newline="") as f:
        return {row["sample_id"]: int(row["severity"]) for row in csv.DictReader(f)}


def build_dataset(labels_csv: str = LABELS_CSV) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Join labels CSV with HLS + GEE-negative images; returns (X (N,5), y (N,), ids)."""
    labels = _load_labels(labels_csv)
    id_to_path = {sample_id_from_path(p): p for p in list_all_train_image_paths()}

    missing = set(labels) - set(id_to_path)
    if missing:
        raise ValueError(
            f"{len(missing)} labeled sample_ids not found among training images: "
            f"{sorted(missing)[:5]}..."
        )

    ids, X, y = [], [], []
    for sample_id, severity in labels.items():
        merged_path = id_to_path[sample_id]
        bands = read_raw_bands(merged_path)
        mask = read_mask(mask_path_for(merged_path))
        ids.append(sample_id)
        # mask.sum()==0 is a safe proxy for "confirmed negative" here specifically
        # because every labeled row is either drawn from
        # list_burn_positive_train_samples() (mask.sum() > 0 by construction) or
        # the GEE-negative pool (all-zero mask by construction) — a no-op for any
        # real positive mask.
        X.append(extract_features_from_bands(bands, mask, known_negative=(mask.sum() == 0)))
        y.append(severity)
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64), ids


def cross_validate(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42) -> dict:
    """Stratified k-fold CV. Reduces n_splits automatically if the smallest
    class has fewer than n_splits members (StratifiedKFold requirement)."""
    class_counts = {c: int((y == c).sum()) for c in np.unique(y)}
    min_class_count = min(class_counts.values())
    k = min(n_splits, min_class_count)
    if k < n_splits:
        print(f"Warning: smallest class has {min_class_count} samples; reducing k from {n_splits} to {k}.")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    accs, f1s, maes = [], [], []
    all_true, all_pred = [], []

    for train_idx, test_idx in skf.split(X, y):
        model = SeverityClassifier()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])

        accs.append(accuracy_score(y[test_idx], preds))
        f1s.append(f1_score(y[test_idx], preds, average="macro"))
        maes.append(mean_absolute_error(y[test_idx], preds))
        all_true.extend(y[test_idx])
        all_pred.extend(preds)

    class_labels = sorted(np.unique(y))
    cm = confusion_matrix(all_true, all_pred, labels=class_labels)

    total = cm.sum()
    fpr_per_class, fnr_per_class = {}, {}
    for i, label in enumerate(class_labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        fpr_per_class[int(label)] = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0
        fnr_per_class[int(label)] = float(fn) / float(fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "mae_mean": float(np.mean(maes)),
        "confusion_matrix": cm,
        "fpr_per_class": fpr_per_class,
        "fnr_per_class": fnr_per_class,
        "fpr_mean": float(np.mean(list(fpr_per_class.values()))),
        "fnr_mean": float(np.mean(list(fnr_per_class.values()))),
        "k_used": k,
    }


def train(
    labels_csv: str = LABELS_CSV,
    output_dir: str = "/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/checkpoints/severity",
    n_splits: int = 5,
    seed: int = 42,
) -> None:
    start_time = time.perf_counter()
    X, y, ids = build_dataset(labels_csv)
    print(f"Loaded {len(ids)} labeled samples. Class distribution: "
          f"{ {int(c): int((y == c).sum()) for c in np.unique(y)} }")

    cv_results = cross_validate(X, y, n_splits=n_splits, seed=seed)
    print(f"CV (k={cv_results['k_used']}) accuracy: "
          f"{cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}")
    print(f"CV macro-F1: {cv_results['macro_f1_mean']:.3f} ± {cv_results['macro_f1_std']:.3f}")
    print(f"CV MAE (ordinal): {cv_results['mae_mean']:.3f}")
    print(f"Confusion matrix:\n{cv_results['confusion_matrix']}")
    print(f"CV FPR per class: {cv_results['fpr_per_class']} (mean {cv_results['fpr_mean']:.3f})")
    print(f"CV FNR per class: {cv_results['fnr_per_class']} (mean {cv_results['fnr_mean']:.3f})")

    print("Refitting final model on all labeled samples...")
    final_model = SeverityClassifier().fit(X, y)
    final_model.save_pretrained(output_dir)
    print(f"Saved final model to {output_dir}/model.joblib")

    elapsed = timedelta(seconds=round(time.perf_counter() - start_time))
    print(f"Total training time: {elapsed}")
