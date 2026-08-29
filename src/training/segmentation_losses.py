"""Task-aware segmentation losses (paper B.0.1 segmentation objective).

The defining task loss is binary cross-entropy on raw logits
(``BCEWithLogitsLoss``). A soft Dice term and a small reconstruction-L1
regularizer are optional and controlled by config weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def bce_with_logits_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy on raw logits vs. pseudo-mask. Both ``[B,1,H,W]``."""
    return F.binary_cross_entropy_with_logits(logits, target)


def soft_dice_loss(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss on probabilities in ``[0,1]``. Returns ``1 - dice``."""
    probs_flat = probs.reshape(probs.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    intersection = (probs_flat * target_flat).sum(dim=1)
    denom = probs_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return (1.0 - dice).mean()


@dataclass
class TaskAwareLossWeights:
    """Weights for the combined task-aware loss."""

    seg_bce_weight: float = 1.0
    seg_dice_weight: float = 0.0
    reconstruction_l1_weight: float = 0.0

    @classmethod
    def from_config(cls, training_cfg: dict) -> "TaskAwareLossWeights":
        # Accept both the explicit weights and the historical alias
        # ``segmentation_bce_weight`` so that config is actually read.
        bce = training_cfg.get(
            "seg_bce_weight", training_cfg.get("segmentation_bce_weight", 1.0)
        )
        return cls(
            seg_bce_weight=float(bce),
            seg_dice_weight=float(training_cfg.get("seg_dice_weight", 0.0)),
            reconstruction_l1_weight=float(
                training_cfg.get("reconstruction_l1_weight", 0.0)
            ),
        )


def task_aware_segmentation_loss(
    outputs: dict[str, torch.Tensor],
    target_mask: torch.Tensor,
    weights: TaskAwareLossWeights,
    *,
    specimen: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine BCE-with-logits + optional soft Dice + optional recon L1.

    Args:
        outputs: forward dict containing ``seg_logits``, ``seg_prob`` and
            (if recon regularization is used) ``x_recon``.
        target_mask: pseudo-ground-truth mask ``[B,1,H,W]`` in ``{0,1}``.
        weights: loss weights.
        specimen: ground-truth image, required only when
            ``reconstruction_l1_weight > 0``.

    Returns:
        ``(total_loss, components)`` where ``components`` holds scalar floats.
    """
    components: dict[str, float] = {}
    total = outputs["seg_logits"].new_zeros(())

    bce = bce_with_logits_loss(outputs["seg_logits"], target_mask)
    components["bce"] = float(bce.item())
    if weights.seg_bce_weight != 0.0:
        total = total + weights.seg_bce_weight * bce

    if weights.seg_dice_weight != 0.0:
        dice = soft_dice_loss(outputs["seg_prob"], target_mask)
        components["dice"] = float(dice.item())
        total = total + weights.seg_dice_weight * dice

    if weights.reconstruction_l1_weight != 0.0:
        if specimen is None:
            raise ValueError("reconstruction_l1_weight > 0 requires `specimen`")
        recon = F.l1_loss(outputs["x_recon"], specimen)
        components["recon_l1"] = float(recon.item())
        total = total + weights.reconstruction_l1_weight * recon

    components["total"] = float(total.item())
    return total, components
