"""Convolutional reconstruction network for the inverse model."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ReconCNNConfig:
    """Configuration for the reconstruction CNN."""

    in_channels: int = 8
    hidden_channels: list[int] = field(default_factory=lambda: [64, 64, 32, 32, 16, 1])
    kernel_size: int = 3
    padding: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "ReconCNNConfig":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


class ReconCNN(nn.Module):
    """Map per-pattern upsampled features to a single-channel reconstruction.

    Architecture (6 blocks):
        Blocks 1-5: Conv2d -> ReLU -> BatchNorm2d
        Block 6: Conv2d -> Sigmoid

    Input shape: [B, T, H, W]
    Output shape: [B, 1, H, W]
    """

    def __init__(self, config: ReconCNNConfig) -> None:
        super().__init__()
        self.config = config

        if len(config.hidden_channels) != 6:
            raise ValueError("hidden_channels must contain exactly 6 values (one per block)")
        if config.hidden_channels[-1] != 1:
            raise ValueError("Last hidden channel count must be 1")

        channel_sizes = [config.in_channels, *config.hidden_channels]
        blocks: list[nn.Module] = []

        for block_idx in range(6):
            in_ch = channel_sizes[block_idx]
            out_ch = channel_sizes[block_idx + 1]
            blocks.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=config.kernel_size,
                    padding=config.padding,
                )
            )
            if block_idx < 5:
                blocks.append(nn.ReLU(inplace=True))
                blocks.append(nn.BatchNorm2d(out_ch))
            else:
                blocks.append(nn.Sigmoid())

        self.net = nn.Sequential(*blocks)

    @classmethod
    def from_dict(cls, data: dict) -> "ReconCNN":
        return cls(ReconCNNConfig.from_dict(data))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Reconstruct the specimen image.

        Args:
            features: Upsampled measurements with shape [B, T, H, W].

        Returns:
            reconstruction: [B, 1, H, W] in [0, 1].
        """
        if features.ndim != 4:
            raise ValueError(f"features must have shape [B, T, H, W], got {tuple(features.shape)}")
        return self.net(features)
