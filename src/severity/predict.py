"""
Stage-3 public entry point: severity classification given post-event NDVI/NBR.

v1 uses only post["ndvi"]/post["nbr"] from fetch_reference_imagery — pre-event
data is intentionally unused here because the HLS training data (hls_burn_scars)
only provides single post-event scenes with masks, so training features are
necessarily post-event-only; using pre-event deltas at inference time would
create a train/inference feature mismatch. Revisit if/when a pre/post-labeled
training set becomes available.
"""

from __future__ import annotations

import os

import numpy as np

from .features import extract_features_from_indices
from .model import SeverityClassifier

# EW_CHECKPOINT_ROOT lets a deployment (e.g. the HF Space, deploy/app.py) point at downloaded weights.
_CHECKPOINT_ROOT = os.environ.get(
    "EW_CHECKPOINT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "checkpoints")
)
_DEFAULT_CHECKPOINT = os.path.join(_CHECKPOINT_ROOT, "severity")

_model: SeverityClassifier | None = None


def _get_model(checkpoint_dir: str = _DEFAULT_CHECKPOINT) -> SeverityClassifier:
    global _model
    if _model is None:
        _model = SeverityClassifier.load_pretrained(checkpoint_dir)
    return _model


def classify_severity(
    post_ndvi: np.ndarray,
    post_nbr: np.ndarray,
    mask: np.ndarray | None = None,
    checkpoint_dir: str = _DEFAULT_CHECKPOINT,
) -> dict:
    """
    Args:
        post_ndvi, post_nbr: (H, W) arrays, e.g. from
            fetch_reference_imagery(...)["post"]["ndvi"/"nbr"].
        mask: optional (H, W) binary burn mask restricting the NDVI/NBR means to the
            burned region (matches how the model is trained — see src/severity/train.py).
            Falls back to the whole-image mean if omitted or empty. The caller is
            responsible for producing this, typically via
            src.segmentation.predict.predict_mask() run on the same post-event image.
        checkpoint_dir: directory containing model.joblib.

    Returns:
        {"severity": int (0-3), "probabilities": {0: p0, 1: p1, 2: p2, 3: p3}}
    """
    model = _get_model(checkpoint_dir)
    features = extract_features_from_indices(post_ndvi, post_nbr, mask)
    X = np.array([features])

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    classes = model.clf.classes_
    probabilities = {int(c): float(p) for c, p in zip(classes, proba)}

    return {"severity": pred, "probabilities": probabilities}


def classify_severity_from_retrieval(
    retrieval_result: dict,
    mask: np.ndarray | None = None,
    checkpoint_dir: str = _DEFAULT_CHECKPOINT,
) -> dict:
    """Convenience wrapper taking fetch_reference_imagery()'s full return dict.
    Uses retrieval_result['post'] only — 'pre' is unused in v1 (see module docstring).
    See classify_severity() for the `mask` param."""
    post = retrieval_result["post"]
    return classify_severity(post["ndvi"], post["nbr"], mask=mask, checkpoint_dir=checkpoint_dir)
