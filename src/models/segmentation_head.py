"""Small convolutional segmentation head (paper supplement B.0.1 step 2).

The head consumes the reconstructed image and predicts raw segmentation
**logits** (no activation), so training can use ``BCEWithLogitsLoss`` for
numerical stability. ``seg_prob`` is obtained downstream via ``sigmoid``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class SegmentationHeadConfig:
    """Configuration for the convolutional segmentation head.

    ``hidden_channels`` lists the output channel count of every conv block in
    order; the final value must be ``1`` (single-channel logits). Every block
    except the last is ``Conv2d -> ReLU``; the last block is a bare ``Conv2d``
    producing raw logits.
    """

    in_channels: int = 1
    hidden_channels: list[int] = field(default_factory=lambda: [16, 16, 1])
    kernel_size: int = 3
    padding: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "SegmentationHeadConfig":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


class SegmentationHead(nn.Module):
    """Map a reconstruction ``[B,1,H,W]`` to segmentation logits ``[B,1,H,W]``."""

    def __init__(self, config: SegmentationHeadConfig) -> None:
        super().__init__()
        self.config = config
        if len(config.hidden_channels) < 1:
            raise ValueError("hidden_channels must contain at least one value")
        if config.hidden_channels[-1] != 1:
            raise ValueError("Last hidden channel count must be 1 (single-channel logits)")

        sizes = [config.in_channels, *config.hidden_channels]
        num_blocks = len(config.hidden_channels)
        blocks: list[nn.Module] = []
        for idx in range(num_blocks):
            in_ch, out_ch = sizes[idx], sizes[idx + 1]
            blocks.append(
                nn.Conv2d(in_ch, out_ch, config.kernel_size, padding=config.padding)
            )
            # No activation on the final block: it emits raw logits.
            if idx < num_blocks - 1:
                blocks.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*blocks)

    @classmethod
    def from_dict(cls, data: dict) -> "SegmentationHead":
        return cls(SegmentationHeadConfig.from_dict(data))

    def forward(self, x_recon: torch.Tensor) -> torch.Tensor:
        """Return raw segmentation logits ``[B,1,H,W]`` (apply sigmoid for probs)."""
        return self.net(x_recon)
