"""
Combined BCE + Dice loss for binary burn-scar segmentation.

BCE handles per-pixel gradient signal; Dice is region-based and
class-imbalance-resistant (burn scars are typically 2–5% of pixels).
pos_weight in the BCE term further amplifies gradients on scar pixels.
"""

import torch
import torch.nn as nn


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Soft Dice loss computed over the batch.

    Args:
        logits:  (B, 1, H, W) raw scores (pre-sigmoid)
        targets: (B, 1, H, W) float mask, values in {0., 1.}
    Returns:
        scalar loss in [0, 1]
    """
    probs = torch.sigmoid(logits)
    p = probs.view(probs.size(0), -1)
    t = targets.view(targets.size(0), -1)
    intersection = (p * t).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (p.sum(dim=1) + t.sum(dim=1) + smooth)
    return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    alpha * BCE-with-logits + (1 - alpha) * Dice.

    Args:
        alpha:      weight for the BCE term (0–1). Default 0.5.
        pos_weight: scalar upweight for scar pixels in BCE.
                    Set to approx background/scar pixel ratio (~10–20).
    """

    def __init__(self, alpha: float = 0.5, pos_weight: float = 10.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.register_buffer("_pos_weight", torch.tensor([pos_weight]))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, 1, H, W)
            targets: (B, 1, H, W) float in {0., 1.}
        Returns:
            scalar loss
        """
        bce_val  = self.bce(logits, targets)
        dice_val = dice_loss(logits, targets)
        return self.alpha * bce_val + (1.0 - self.alpha) * dice_val
