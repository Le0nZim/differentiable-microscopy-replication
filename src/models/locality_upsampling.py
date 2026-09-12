"""Locality-aware upsampling for the inverse model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

UpsamplingMode = Literal["locality_aware", "transpose_conv", "original_locality", "original_transpose"]


@dataclass
class LocalityUpsamplingConfig:
    """Configuration for upsampling compressed detector measurements."""

    mode: UpsamplingMode = "locality_aware"
    downscale_factor: int = 8
    num_patterns: int = 8
    use_mixing_cnn: bool = False
    mixing_hidden_channels: int = 16

    @classmethod
    def from_dict(cls, data: dict) -> "LocalityUpsamplingConfig":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


class LocalityAwareUpsampling(nn.Module):
    """Project each detector pixel into an n x n patch and tile to full resolution.

    For detector location (i, j) and pattern t:
        patch = y_down[b, t, i, j] * weights[t, i, j, :, :]
    Tiled output shape: [B, T, H, W] where H = H_down * n and W = W_down * n.
    """

    def __init__(
        self,
        num_patterns: int,
        height_down: int,
        width_down: int,
        downscale_factor: int,
    ) -> None:
        super().__init__()
        self.num_patterns = num_patterns
        self.height_down = height_down
        self.width_down = width_down
        self.downscale_factor = downscale_factor

        # weights: [T, H_down, W_down, n, n]
        patch_size = downscale_factor
        self.weights = nn.Parameter(
            torch.randn(num_patterns, height_down, width_down, patch_size, patch_size) * 0.01
        )

    def forward(self, y_down: torch.Tensor) -> torch.Tensor:
        """Upsample compressed measurements.

        Args:
            y_down: [B, T, H_down, W_down]

        Returns:
            y_up: [B, T, H, W]
        """
        if y_down.ndim != 4:
            raise ValueError(f"y_down must have shape [B, T, H_down, W_down], got {tuple(y_down.shape)}")

        batch_size, num_patterns, height_down, width_down = y_down.shape
        down = self.downscale_factor

        if num_patterns != self.num_patterns:
            raise ValueError(
                f"Expected T={self.num_patterns}, got T={num_patterns}"
            )
        if (height_down, width_down) != (self.height_down, self.width_down):
            raise ValueError(
                "y_down spatial size does not match module initialization: "
                f"expected {(self.height_down, self.width_down)}, got {(height_down, width_down)}"
            )

        # y_down: [B, T, H_down, W_down, 1, 1]
        # weights: [T, H_down, W_down, n, n]
        # patches: [B, T, H_down, W_down, n, n]
        patches = y_down[..., None, None] * self.weights.unsqueeze(0)

        # Tile patches into [B, T, H, W].
        y_up = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        y_up = y_up.view(batch_size, num_patterns, height_down * down, width_down * down)
        return y_up


class MixingCNN(nn.Module):
    """Optional light CNN applied after locality-aware tiling (paper Fig. 2)."""

    def __init__(self, num_patterns: int, hidden_channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_patterns, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_channels, num_patterns, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Refine tiled features. Shape preserved: [B, T, H, W]."""
        return self.net(features)


class LocalityUpsampling(nn.Module):
    """Upsampling block with locality-aware or transpose-convolution modes."""

    def __init__(self, config: LocalityUpsamplingConfig, height_down: int, width_down: int) -> None:
        super().__init__()
        self.config = config
        self.mode = config.mode
        self.downscale_factor = config.downscale_factor
        self.num_patterns = config.num_patterns

        if config.mode == "locality_aware":
            self.upsampler: nn.Module = LocalityAwareUpsampling(
                num_patterns=config.num_patterns,
                height_down=height_down,
                width_down=width_down,
                downscale_factor=config.downscale_factor,
            )
            self.mixing_cnn = (
                MixingCNN(config.num_patterns, config.mixing_hidden_channels)
                if config.use_mixing_cnn
                else None
            )
        elif config.mode == "transpose_conv":
            self.upsampler = TransposeConvUpsampling(
                num_patterns=config.num_patterns,
                downscale_factor=config.downscale_factor,
            )
            self.mixing_cnn = None
        elif config.mode == "original_locality":
            self.upsampler = OriginalLocalityUpsampling(config.num_patterns, height_down,
                                                       width_down, config.downscale_factor)
            self.mixing_cnn = None
        elif config.mode == "original_transpose":
            self.upsampler = OriginalTransposeUpsampling(config.num_patterns, config.downscale_factor)
            self.mixing_cnn = None
        else:
            raise ValueError(f"Unsupported upsampling mode: {config.mode}")

    @classmethod
    def from_dict(cls, data: dict, height_down: int, width_down: int) -> "LocalityUpsampling":
        return cls(LocalityUpsamplingConfig.from_dict(data), height_down, width_down)

    def forward(self, y_down: torch.Tensor) -> torch.Tensor:
        """Upsample detector measurements to full resolution per pattern.

        Args:
            y_down: [B, T, H_down, W_down]

        Returns:
            y_up: [B, T, H, W]
        """
        y_up = self.upsampler(y_down)
        if self.mixing_cnn is not None:
            y_up = self.mixing_cnn(y_up)
        return y_up


class TransposeConvUpsampling(nn.Module):
    """Transpose-convolution baseline for upsampling ablations."""

    def __init__(self, num_patterns: int, downscale_factor: int) -> None:
        super().__init__()
        self.num_patterns = num_patterns
        self.downscale_factor = downscale_factor
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=num_patterns,
            out_channels=num_patterns,
            kernel_size=downscale_factor,
            stride=downscale_factor,
            bias=True,
        )

    def forward(self, y_down: torch.Tensor) -> torch.Tensor:
        """Upsample via transpose convolution.

        Args:
            y_down: [B, T, H_down, W_down]

        Returns:
            y_up: [B, T, H, W]
        """
        return self.conv_transpose(y_down)


class OriginalLocalityUpsampling(nn.Module):
    """Independent implementation of upstream custom_v2 plus its channel lift.

    Each detector site's T-vector maps to ONE d*d patch, with a site-dependent
    bias, followed by Conv-BN(1 -> max(1,T//2) -> T). The historical rewrite's
    `locality_aware` preserves T separate patches and is a different network.
    """

    def __init__(self, num_patterns, height_down, width_down, downscale_factor):
        super().__init__()
        self.shape = (num_patterns, height_down, width_down)
        self.downscale_factor = downscale_factor
        self.weights = nn.Parameter(torch.empty(height_down * width_down, num_patterns,
                                                 downscale_factor**2))
        self.bias = nn.Parameter(torch.zeros(height_down * width_down, downscale_factor**2))
        nn.init.xavier_normal_(self.weights)
        hidden = max(1, num_patterns // 2)
        self.channel_lift = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1), nn.BatchNorm2d(hidden),
            nn.Conv2d(hidden, num_patterns, 3, padding=1), nn.BatchNorm2d(num_patterns),
        )

    def project(self, measurements):
        if measurements.ndim != 4 or tuple(measurements.shape[1:]) != self.shape:
            raise ValueError(f"Expected [B,{self.shape}], got {tuple(measurements.shape)}")
        b, t, h, w = measurements.shape
        d = self.downscale_factor
        sites = measurements.flatten(2).transpose(1, 2)
        patches = torch.einsum("blt,ltp->blp", sites, self.weights) + self.bias
        return patches.reshape(b, h, w, d, d).permute(0, 1, 3, 2, 4).reshape(b, 1, h*d, w*d)

    def forward(self, measurements):
        return self.channel_lift(self.project(measurements))


class OriginalTransposeUpsampling(nn.Module):
    """Upstream repeated 2x ConvTranspose-ReLU-BN baseline, evaluated correctly."""

    def __init__(self, num_patterns, downscale_factor):
        super().__init__()
        if downscale_factor < 1 or downscale_factor & (downscale_factor - 1):
            raise ValueError("Original transpose baseline requires a power-of-two downscale")
        blocks = []
        for _ in range(downscale_factor.bit_length() - 1):
            blocks.extend([nn.ConvTranspose2d(num_patterns, num_patterns, 4, stride=2, padding=1),
                           nn.ReLU(), nn.BatchNorm2d(num_patterns)])
        self.net = nn.Sequential(*blocks)

    def forward(self, measurements):
        return self.net(measurements)
