"""Evaluate illumination patterns under soft, sharpened, and thresholded variants."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader

from evaluation.metrics import mse, ssim
from evaluation.pattern_inspection import save_pattern_inspection
from models.microscope import DifferentiableMicroscope
from training.pattern_tracking import capture_detector_snapshot, capture_pattern_snapshot
from utils.logging import save_measurement_grid

PatternVariant = Literal["soft", "sharpened", "thresholded"]


@torch.no_grad()
def forward_with_pattern_variant(
    model: DifferentiableMicroscope,
    specimen: torch.Tensor,
    *,
    training_m: float,
    pattern_variant: PatternVariant = "soft",
    sharpen_m: float = 10.0,
    apply_noise: bool = False,
) -> dict[str, torch.Tensor]:
    """Run forward pass with soft, sharpened, or thresholded excitation patterns."""
    if pattern_variant == "soft":
        patterns = model.pattern_generator(sigmoid_m=training_m)
    elif pattern_variant == "sharpened":
        patterns = model.pattern_generator(sigmoid_m=sharpen_m)
    elif pattern_variant == "thresholded":
        soft = model.pattern_generator(sigmoid_m=training_m)
        patterns = (soft > 0.5).float()
    else:
        raise ValueError(f"Unsupported pattern variant: {pattern_variant}")

    alpha_down = model.forward_model(specimen, patterns)
    y_down = model.detector_noise(alpha_down, apply_noise=apply_noise)
    x_recon = model.inverse_model(y_down)
    return {
        "x_recon": x_recon,
        "patterns": patterns,
        "alpha_down": alpha_down,
        "y_down": y_down,
    }


@torch.no_grad()
def evaluate_pattern_variant(
    model: DifferentiableMicroscope,
    dataloader: DataLoader,
    device: torch.device,
    *,
    training_m: float,
    pattern_variant: PatternVariant = "soft",
    sharpen_m: float = 10.0,
    apply_noise: bool = False,
) -> tuple[float, float]:
    model.eval()
    total_mse = 0.0
    total_ssim = 0.0
    count = 0
    for batch in dataloader:
        specimen = batch.to(device)
        outputs = forward_with_pattern_variant(
            model,
            specimen,
            training_m=training_m,
            pattern_variant=pattern_variant,
            sharpen_m=sharpen_m,
            apply_noise=apply_noise,
        )
        total_mse += float(mse(outputs["x_recon"], specimen).item())
        total_ssim += float(ssim(outputs["x_recon"], specimen).item())
        count += 1
    return total_mse / max(count, 1), total_ssim / max(count, 1)


def pattern_stats_from_tensor(patterns: torch.Tensor) -> dict[str, float]:
    values = patterns.detach().cpu()
    return {
        "H_t_min": float(values.min().item()),
        "H_t_max": float(values.max().item()),
        "H_t_mean": float(values.mean().item()),
        "H_t_binary_fraction": float(((values < 0.05) | (values > 0.95)).float().mean().item()),
    }


@torch.no_grad()
def evaluate_all_pattern_variants(
    model: DifferentiableMicroscope,
    dataloaders: dict[str, DataLoader],
    device: torch.device,
    *,
    training_m: float,
    sharpen_m: float = 10.0,
    apply_noise: bool = False,
    reference_specimen: torch.Tensor | None = None,
    initial_patterns: torch.Tensor | None = None,
    initial_y_down: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Evaluate soft, sharpened, and thresholded patterns on val/test splits."""
    results: dict[str, Any] = {"training_m": training_m, "sharpen_m": sharpen_m, "variants": {}}

    for variant in ("soft", "sharpened", "thresholded"):
        variant_entry: dict[str, Any] = {"splits": {}}
        for split_name, loader in dataloaders.items():
            eval_mse, eval_ssim = evaluate_pattern_variant(
                model,
                loader,
                device,
                training_m=training_m,
                pattern_variant=variant,
                sharpen_m=sharpen_m,
                apply_noise=apply_noise,
            )
            variant_entry["splits"][split_name] = {"mse": eval_mse, "ssim": eval_ssim}
        results["variants"][variant] = variant_entry

    if reference_specimen is not None:
        soft_patterns, _ = capture_pattern_snapshot(model, sigmoid_m=training_m)
        sharpened_patterns = model.pattern_generator(sigmoid_m=sharpen_m).detach().cpu()
        thresholded_patterns = (soft_patterns > 0.5).float()
        results["pattern_stats"] = {
            "soft": pattern_stats_from_tensor(soft_patterns),
            "sharpened": pattern_stats_from_tensor(sharpened_patterns),
            "thresholded": pattern_stats_from_tensor(thresholded_patterns),
        }
        if initial_patterns is not None:
            results["pattern_delta_soft"] = float((soft_patterns - initial_patterns).abs().mean().item())
        final_y_down = capture_detector_snapshot(
            model,
            reference_specimen,
            sigmoid_m=training_m,
            apply_noise=apply_noise,
        )
        if initial_y_down is not None:
            results["detector_delta"] = float((final_y_down - initial_y_down).abs().mean().item())

    return results


@torch.no_grad()
def save_variant_artifacts(
    model: DifferentiableMicroscope,
    specimen: torch.Tensor,
    output_dir: Path,
    *,
    training_m: float,
    sharpen_m: float = 10.0,
    apply_noise: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for variant in ("soft", "sharpened", "thresholded"):
        outputs = forward_with_pattern_variant(
            model,
            specimen,
            training_m=training_m,
            pattern_variant=variant,
            sharpen_m=sharpen_m,
            apply_noise=apply_noise,
        )
        prefix = f"H_t_{variant}"
        save_pattern_inspection(outputs["patterns"], output_dir, sigmoid_m=training_m, prefix=prefix)
        save_measurement_grid(
            outputs["x_recon"],
            figures_dir / f"reconstruction_{variant}.png",
        )
