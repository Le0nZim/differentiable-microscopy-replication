"""Physics-based forward model for compressive fluorescence microscopy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ForwardModelConfig:
    """Configuration for the optical forward model."""

    downscale_factor: int = 8
    use_impulse_psfs: bool = True
    ex_psf_kernel_size: int = 3
    em_psf_kernel_size: int = 3

    @classmethod
    def from_dict(cls, data: dict) -> "ForwardModelConfig":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


def sum_pool_nxn(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Optical demagnification via sum pooling.

    sumpool_{n x n}(X) = avg_pool2d(X, n, n) * n^2

    Args:
        x: Tensor of shape [B, C, H, W].
        kernel_size: Pooling factor n.

    Returns:
        Tensor of shape [B, C, H/n, W/n].
    """
    if kernel_size == 1:
        return x
    pooled = F.avg_pool2d(x, kernel_size=kernel_size, stride=kernel_size)
    return pooled * (kernel_size**2)


def _make_impulse_kernel(kernel_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    kernel = torch.zeros(1, 1, kernel_size, kernel_size, device=device, dtype=dtype)
    center = kernel_size // 2
    kernel[..., center, center] = 1.0
    return kernel


def _conv2d_same(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Depthwise same-sized 2D convolution. x: [B, C, H, W], kernel: [1, 1, k, k]."""
    channels = x.shape[1]
    weight = kernel.expand(channels, 1, -1, -1)
    padding = kernel.shape[-1] // 2
    return F.conv2d(x, weight, padding=padding, groups=channels)


class ForwardModel(nn.Module):
    """Differentiable forward model from specimen image to demagnified measurements."""

    def __init__(self, config: ForwardModelConfig) -> None:
        super().__init__()
        self.config = config
        self.downscale_factor = config.downscale_factor

        self.register_buffer("ex_psf", torch.zeros(1))
        self.register_buffer("em_psf", torch.zeros(1))
        self._psf_initialized = False

    @classmethod
    def from_dict(cls, data: dict) -> "ForwardModel":
        return cls(ForwardModelConfig.from_dict(data))

    def _ensure_psfs(self, sample: torch.Tensor) -> None:
        if self._psf_initialized:
            return

        device = sample.device
        dtype = sample.dtype
        if self.config.use_impulse_psfs:
            ex_kernel = _make_impulse_kernel(3, device, dtype)
            em_kernel = _make_impulse_kernel(3, device, dtype)
        else:
            ex_kernel = torch.ones(
                1, 1, self.config.ex_psf_kernel_size, self.config.ex_psf_kernel_size,
                device=device, dtype=dtype,
            )
            ex_kernel = ex_kernel / ex_kernel.sum()
            em_kernel = torch.ones(
                1, 1, self.config.em_psf_kernel_size, self.config.em_psf_kernel_size,
                device=device, dtype=dtype,
            )
            em_kernel = em_kernel / em_kernel.sum()

        self.ex_psf = ex_kernel
        self.em_psf = em_kernel
        self._psf_initialized = True

    def _apply_ex_psf(self, patterns: torch.Tensor) -> torch.Tensor:
        """Convolve excitation PSF with patterns. patterns: [T, 1, H, W]."""
        num_patterns = patterns.shape[0]
        patterns_batched = patterns.reshape(1, num_patterns, patterns.shape[-2], patterns.shape[-1])
        convolved = _conv2d_same(patterns_batched, self.ex_psf)
        return convolved.reshape(num_patterns, 1, patterns.shape[-2], patterns.shape[-1])

    def _encode_pattern(
        self,
        specimen: torch.Tensor,
        pattern: torch.Tensor,
    ) -> torch.Tensor:
        """Compute alpha_t = emPSF * ((exPSF * H_t) * X).

        Args:
            specimen: [B, 1, H, W]
            pattern: [1, 1, H, W]

        Returns:
            alpha_t: [B, 1, H, W]
        """
        ex_pattern = self._apply_ex_psf(pattern)
        encoded = ex_pattern * specimen
        alpha = _conv2d_same(encoded, self.em_psf)
        return alpha

    def forward(
        self,
        specimen: torch.Tensor,
        patterns: torch.Tensor,
    ) -> torch.Tensor:
        """Run the forward model.

        Args:
            specimen: Ground-truth image X with shape [B, 1, H, W].
            patterns: Excitation patterns H_t with shape [T, 1, H, W].

        Returns:
            alpha_down: Demagnified measurements with shape [B, T, H/n, W/n].
        """
        self._ensure_psfs(specimen)

        if specimen.ndim != 4 or specimen.shape[1] != 1:
            raise ValueError(f"specimen must have shape [B, 1, H, W], got {tuple(specimen.shape)}")
        if patterns.ndim != 4 or patterns.shape[1] != 1:
            raise ValueError(f"patterns must have shape [T, 1, H, W], got {tuple(patterns.shape)}")
        if specimen.shape[-2:] != patterns.shape[-2:]:
            raise ValueError("specimen and patterns must share spatial dimensions")

        batch_size = specimen.shape[0]
        num_patterns = patterns.shape[0]
        height, width = specimen.shape[-2:]
        down = self.downscale_factor

        if height % down != 0 or width % down != 0:
            raise ValueError(
                f"H ({height}) and W ({width}) must be divisible by downscale_factor ({down})"
            )

        alpha_list = []
        for pattern_idx in range(num_patterns):
            pattern = patterns[pattern_idx : pattern_idx + 1]
            alpha_t = self._encode_pattern(specimen, pattern)
            alpha_list.append(alpha_t)

        # [B, T, H, W]
        alpha = torch.cat(alpha_list, dim=1)

        # Reshape to [B*T, 1, H, W] for pooling, then back to [B, T, H/n, W/n]
        alpha_flat = alpha.reshape(batch_size * num_patterns, 1, height, width)
        alpha_down_flat = sum_pool_nxn(alpha_flat, down)
        height_down = height // down
        width_down = width // down
        return alpha_down_flat.reshape(batch_size, num_patterns, height_down, width_down)
