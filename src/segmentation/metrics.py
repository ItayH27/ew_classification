"""
Segmentation metrics — IoU (Jaccard) and Dice score.

SegmentationMetrics accumulates per-batch statistics and returns
epoch-level averages via .compute(). Reset between epochs with .reset().

fnr is accumulated as a running weighted average over samples that actually
have a positive pixel in the target, rather than a mean of per-batch means —
a sample with no positive pixels has nothing to miss, so it's excluded from
both the fnr sum and its denominator (otherwise it would silently score as
fnr=1.0, "missed everything", which corrupts the metric once negative-origin
samples are mixed into training). iou/dice/fpr stay on the plain per-batch
mean — they're well-defined and still meaningful (fpr in particular still
penalizes false positives) on an all-negative sample.
"""

import torch


class SegmentationMetrics:
    """
    Stateful metric accumulator for binary segmentation.

    Usage:
        metrics = SegmentationMetrics()
        for logits, targets in val_loader:
            metrics.update(logits, targets)
        results = metrics.compute()   # {"iou": float, "dice": float, "fpr": float, "fnr": float}
        metrics.reset()
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._iou_sum   = 0.0
        self._dice_sum  = 0.0
        self._fpr_sum   = 0.0
        self._fnr_sum   = 0.0
        self._fnr_valid_count = 0
        self._n_batches = 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        """
        Args:
            logits:     (B, 1, H, W) raw scores (pre-sigmoid)
            targets:    (B, 1, H, W) float in {0., 1.}
            valid_mask: (B, 1, H, W) float in {0., 1.}, optional. Pixels where this is
                        0 (e.g. nodata/cloud) are excluded from all four metrics.
        """
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > self.threshold).float()
            self._iou_sum  += _batch_iou(preds, targets, valid_mask)
            self._dice_sum += _batch_dice(preds, targets, valid_mask)
            self._fpr_sum  += _batch_fpr(preds, targets, valid_mask)
            fnr_batch_mean, n_valid = _batch_fnr(preds, targets, valid_mask)
            self._fnr_sum += fnr_batch_mean * n_valid
            self._fnr_valid_count += n_valid
            self._n_batches += 1

    def compute(self) -> dict[str, float]:
        n = max(self._n_batches, 1)
        return {
            "iou":  self._iou_sum  / n,
            "dice": self._dice_sum / n,
            "fpr":  self._fpr_sum  / n,
            "fnr":  self._fnr_sum  / max(self._fnr_valid_count, 1),
        }


def _flatten_with_mask(
    preds: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = preds.view(preds.size(0), -1)
    t = targets.view(targets.size(0), -1)
    m = valid_mask.view(valid_mask.size(0), -1) if valid_mask is not None else torch.ones_like(p)
    return p, t, m


def _batch_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    smooth: float = 1e-6,
) -> float:
    p, t, m = _flatten_with_mask(preds, targets, valid_mask)
    tp = (p * t * m).sum(dim=1)
    fp = (p * (1 - t) * m).sum(dim=1)
    fn = ((1 - p) * t * m).sum(dim=1)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    return iou.mean().item()


def _batch_dice(
    preds: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    smooth: float = 1e-6,
) -> float:
    p, t, m = _flatten_with_mask(preds, targets, valid_mask)
    tp = (p * t * m).sum(dim=1)
    fp = (p * (1 - t) * m).sum(dim=1)
    fn = ((1 - p) * t * m).sum(dim=1)
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    return dice.mean().item()


def _batch_fpr(
    preds: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    smooth: float = 1e-6,
) -> float:
    p, t, m = _flatten_with_mask(preds, targets, valid_mask)
    fp = (p * (1 - t) * m).sum(dim=1)
    tn = ((1 - p) * (1 - t) * m).sum(dim=1)
    fpr = (fp + smooth) / (fp + tn + smooth)
    return fpr.mean().item()


def _batch_fnr(
    preds: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    smooth: float = 1e-6,
) -> tuple[float, int]:
    """Returns (mean_fnr, n_valid_samples), excluding samples with no positive
    pixel in the target — nothing to miss ≠ missed everything."""
    p, t, m = _flatten_with_mask(preds, targets, valid_mask)
    has_positive = t.sum(dim=1) > 0
    if not bool(has_positive.any()):
        return 0.0, 0
    fn = ((1 - p) * t * m).sum(dim=1)
    tp = (p * t * m).sum(dim=1)
    fnr = (fn + smooth) / (fn + tp + smooth)
    return fnr[has_positive].mean().item(), int(has_positive.sum().item())
