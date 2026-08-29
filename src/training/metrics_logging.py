"""Per-step training metrics and gradient norms."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from evaluation.metrics import mse, ssim
from models.microscope import DifferentiableMicroscope


def _grad_norm(parameters: list[nn.Parameter]) -> float:
    squared = 0.0
    for param in parameters:
        if param.grad is not None:
            squared += float(param.grad.detach().data.norm(2).item() ** 2)
    return squared**0.5


def pattern_stats(patterns: torch.Tensor) -> dict[str, float]:
    """Summarize illumination patterns H_t. patterns: [T, 1, H, W]."""
    values = patterns.detach()
    near_binary = ((values < 0.1) | (values > 0.9)).float().mean()
    return {
        "H_t_min": float(values.min().item()),
        "H_t_max": float(values.max().item()),
        "H_t_mean": float(values.mean().item()),
        "H_t_binary_fraction": float(near_binary.item()),
    }


def detector_stats(y_down: torch.Tensor) -> dict[str, float]:
    """Summarize detector measurements y_down: [B, T, H, W]."""
    values = y_down.detach()
    return {
        "detector_min": float(values.min().item()),
        "detector_max": float(values.max().item()),
        "detector_mean": float(values.mean().item()),
    }


@torch.no_grad()
def batch_reconstruction_metrics(
    model: DifferentiableMicroscope,
    specimen: torch.Tensor,
    device: torch.device,
    *,
    apply_noise: bool,
    sigmoid_m: float,
) -> dict[str, float]:
    model.eval()
    outputs = model(specimen, sigmoid_m=sigmoid_m, apply_noise=apply_noise)
    metrics = {
        "mse": float(mse(outputs["x_recon"], specimen).item()),
        "ssim": float(ssim(outputs["x_recon"], specimen).item()),
    }
    metrics.update(pattern_stats(outputs["patterns"]))
    metrics.update(detector_stats(outputs["y_down"]))
    model.train()
    return metrics


def collect_step_metrics(
    model: DifferentiableMicroscope,
    outputs: dict[str, torch.Tensor],
    loss: torch.Tensor,
) -> dict[str, Any]:
    """Collect metrics immediately after backward()."""
    illumination_params = model.illumination_parameters()
    inverse_params = model.inverse_parameters()

    metrics: dict[str, Any] = {
        "loss": float(loss.item()),
        "grad_norm_W": _grad_norm(illumination_params) if illumination_params else 0.0,
        "grad_norm_inverse": _grad_norm(inverse_params),
    }
    metrics.update(pattern_stats(outputs["patterns"]))
    metrics.update(detector_stats(outputs["y_down"]))
    return metrics
