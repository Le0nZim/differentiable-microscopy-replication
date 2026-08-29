"""Inverse model combining upsampling and reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .locality_upsampling import LocalityUpsampling, LocalityUpsamplingConfig
from .recon_cnn import ReconCNN, ReconCNNConfig


@dataclass
class InverseModelConfig:
    """Configuration for the full inverse model."""

    upsampling: LocalityUpsamplingConfig
    reconstruction: ReconCNNConfig

    @classmethod
    def from_dict(cls, data: dict) -> "InverseModelConfig":
        return cls(
            upsampling=LocalityUpsamplingConfig.from_dict(data["upsampling"]),
            reconstruction=ReconCNNConfig.from_dict(data["reconstruction"]),
        )


class InverseModel(nn.Module):
    """Map compressed detector measurements to a reconstructed image."""

    def __init__(
        self,
        config: InverseModelConfig,
        height_down: int,
        width_down: int,
    ) -> None:
        super().__init__()
        self.upsampling = LocalityUpsampling(config.upsampling, height_down, width_down)
        self.reconstruction = ReconCNN(config.reconstruction)

    @classmethod
    def from_dict(cls, data: dict, height_down: int, width_down: int) -> "InverseModel":
        return cls(InverseModelConfig.from_dict(data), height_down, width_down)

    def forward(self, y_down: torch.Tensor) -> torch.Tensor:
        """Reconstruct specimen image from compressed measurements.

        Args:
            y_down: [B, T, H_down, W_down]

        Returns:
            x_recon: [B, 1, H, W]
        """
        y_up = self.upsampling(y_down)
        return self.reconstruction(y_up)
