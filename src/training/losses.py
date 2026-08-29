"""Training loss functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 reconstruction loss. Inputs: [B, 1, H, W]."""
    return F.l1_loss(prediction, target)


def segmentation_loss_bce(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy for segmentation probabilities vs pseudo-mask [B,1,H,W]."""
    return F.binary_cross_entropy(prediction, target)
