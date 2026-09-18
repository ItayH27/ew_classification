"""
Thin wrapper: StandardScaler + multinomial LogisticRegression for
severity classification over 5 features (whole-image mean NDVI/NBR,
masked-region mean NDVI/NBR, burned-area fraction — see src/severity/features.py).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class SeverityClassifier:
    """
    Severity classes: 0 (no damage) .. 3 (maximal), stored/predicted as
    integers matching data/severity_labels.csv values directly (no
    0-indexing remap — sklearn handles arbitrary integer class labels).
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        # No multi_class= kwarg: recent sklearn (>=1.5) auto-selects
        # multinomial loss for >2 classes with lbfgs; passing multi_class
        # explicitly is deprecated. class_weight="balanced" counteracts the
        # 76/80/44 class imbalance, which was otherwise starving the minority
        # "severe" class of recall (FNR 0.73 in the unweighted baseline).
        self.clf = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SeverityClassifier":
        """X: (N, 5) [whole_ndvi, whole_nbr, masked_ndvi, masked_nbr, burned_fraction].
        y: (N,) ints in {0,1,2,3}."""
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.clf.predict(Xs)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.clf.predict_proba(Xs)

    def save_pretrained(self, output_dir: str) -> None:
        """Save scaler+classifier as a single joblib bundle."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "clf": self.clf}, out / "model.joblib")

    @classmethod
    def load_pretrained(cls, checkpoint_dir: str) -> "SeverityClassifier":
        bundle = joblib.load(Path(checkpoint_dir) / "model.joblib")
        instance = cls.__new__(cls)
        instance.scaler = bundle["scaler"]
        instance.clf = bundle["clf"]
        return instance
