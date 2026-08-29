"""Evaluation metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error. Inputs: [B, 1, H, W]."""
    return torch.mean((prediction - target) ** 2)


def psnr(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Peak signal-to-noise ratio."""
    error = mse(prediction, target)
    return 10.0 * torch.log10(torch.tensor(data_range**2, device=error.device) / error.clamp_min(1e-12))


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    window = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Differentiable SSIM averaged over the batch. Inputs: [B, 1, H, W]."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("prediction and target must have shape [B, 1, H, W]")

    channels = prediction.shape[1]
    window = _gaussian_window(window_size, sigma, channels, prediction.device, prediction.dtype)
    padding = window_size // 2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_pred = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu_target = F.conv2d(target, window, padding=padding, groups=channels)

    mu_pred_sq = mu_pred * mu_pred
    mu_target_sq = mu_target * mu_target
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(prediction * prediction, window, padding=padding, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_target_sq
    sigma_pred_target = F.conv2d(prediction * target, window, padding=padding, groups=channels) - mu_pred_target

    numerator = (2.0 * mu_pred_target + c1) * (2.0 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    ssim_map = numerator / denominator.clamp_min(1e-12)
    return ssim_map.mean()
