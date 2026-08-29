"""Phased training for sigmoid-m pattern hardening diagnostics."""

from __future__ import annotations

import csv
import itertools
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from evaluation.metrics import mse, ssim
from models.microscope import DifferentiableMicroscope
from training.losses import reconstruction_loss_l1
from training.metrics_logging import batch_reconstruction_metrics, collect_step_metrics
from training.pattern_tracking import PatternSnapshot, capture_detector_snapshot, capture_pattern_snapshot, finalize_pattern_snapshot


def train_fixed_m_phase(
    model: DifferentiableMicroscope,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict[str, Any],
    run_dir: Path,
    *,
    sigmoid_m: float,
    max_steps: int,
    phase_name: str,
    step_offset: int = 0,
    pattern_snapshot: PatternSnapshot | None = None,
    reference_specimen: torch.Tensor | None = None,
    append_log: bool = False,
    freeze_illumination: bool = False,
) -> tuple[list[dict[str, Any]], PatternSnapshot, dict[str, float], int]:
    """Train for max_steps at a fixed sigmoid sharpness m."""
    from training.train_reconstruction import save_checkpoint

    training_cfg = config["training"]
    log_every = int(training_cfg.get("log_every", 250))
    apply_noise = config["detector_noise"].get("apply_noise", True)

    history: list[dict[str, Any]] = []
    step_log_path = run_dir / "metrics" / "step_log.csv"
    step_log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "step",
        "phase",
        "epoch",
        "loss",
        "train_mse",
        "train_ssim",
        "val_mse",
        "val_ssim",
        "H_t_min",
        "H_t_max",
        "H_t_mean",
        "H_t_binary_fraction",
        "grad_norm_W",
        "grad_norm_inverse",
        "detector_min",
        "detector_max",
        "detector_mean",
        "sigmoid_m",
    ]

    if pattern_snapshot is None:
        initial_patterns, initial_w = capture_pattern_snapshot(model, sigmoid_m=sigmoid_m)
        reference_specimen = next(iter(val_loader)).to(device)
        initial_y_down = capture_detector_snapshot(
            model,
            reference_specimen,
            sigmoid_m=sigmoid_m,
            apply_noise=apply_noise,
        )
        pattern_snapshot = PatternSnapshot(
            initial_patterns=initial_patterns,
            final_patterns=None,
            initial_w=initial_w,
            final_w=None,
            initial_y_down=initial_y_down,
            final_y_down=None,
        )
    elif reference_specimen is None:
        reference_specimen = next(iter(val_loader)).to(device)

    if freeze_illumination:
        model.set_illumination_trainable(False)
    elif model.pattern_generator.patterns_are_learnable():
        model.set_illumination_trainable(True)

    train_iter = itertools.cycle(train_loader)
    best_val_mse = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_step = step_offset

    log_mode = "a" if append_log else "w"
    with step_log_path.open(log_mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not append_log:
            writer.writeheader()

        for local_step in range(1, max_steps + 1):
            global_step = step_offset + local_step
            epoch = (local_step - 1) // max(len(train_loader), 1) + 1

            specimen = next(train_iter).to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(specimen, sigmoid_m=sigmoid_m, apply_noise=apply_noise)
            loss = reconstruction_loss_l1(outputs["x_recon"], specimen)
            loss.backward()
            step_metrics = collect_step_metrics(model, outputs, loss)
            grad_clip = training_cfg.get("gradient_clip_norm")
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            pattern_snapshot.max_grad_norm_w = max(
                pattern_snapshot.max_grad_norm_w,
                float(step_metrics["grad_norm_W"]),
            )
            optimizer.step()

            if local_step % log_every != 0 and local_step != max_steps:
                continue

            train_metrics = batch_reconstruction_metrics(
                model,
                specimen,
                device,
                apply_noise=apply_noise,
                sigmoid_m=sigmoid_m,
            )
            val_mse, val_ssim = _evaluate_loader(model, val_loader, device, apply_noise, sigmoid_m)
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_state_dict = deepcopy(model.state_dict())
                best_step = global_step

            row = {
                "step": global_step,
                "phase": phase_name,
                "epoch": epoch,
                "loss": step_metrics["loss"],
                "train_mse": train_metrics["mse"],
                "train_ssim": train_metrics["ssim"],
                "val_mse": val_mse,
                "val_ssim": val_ssim,
                "H_t_min": step_metrics["H_t_min"],
                "H_t_max": step_metrics["H_t_max"],
                "H_t_mean": step_metrics["H_t_mean"],
                "H_t_binary_fraction": step_metrics["H_t_binary_fraction"],
                "grad_norm_W": step_metrics["grad_norm_W"],
                "grad_norm_inverse": step_metrics["grad_norm_inverse"],
                "detector_min": step_metrics["detector_min"],
                "detector_max": step_metrics["detector_max"],
                "detector_mean": step_metrics["detector_mean"],
                "sigmoid_m": sigmoid_m,
            }
            writer.writerow(row)
            handle.flush()
            history.append(row)

            print(
                f"[{phase_name}] step={global_step} m={sigmoid_m} loss={row['loss']:.5f} "
                f"val_mse={row['val_mse']:.5f} bin_frac={row['H_t_binary_fraction']:.3f}",
                flush=True,
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    finalize_pattern_snapshot(
        pattern_snapshot,
        model,
        sigmoid_m=sigmoid_m,
        reference_specimen=reference_specimen,
        apply_noise=apply_noise,
    )

    phase_dir = run_dir / "checkpoints" / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(phase_dir / "best.pt", model, optimizer, int(sigmoid_m), config)

    pattern_metrics = pattern_snapshot.to_dict()
    pattern_metrics["best_val_mse"] = best_val_mse
    pattern_metrics["best_step"] = best_step
    pattern_metrics["sigmoid_m"] = sigmoid_m
    pattern_metrics["phase"] = phase_name
    if pattern_snapshot.final_patterns is not None:
        final_h = pattern_snapshot.final_patterns
        pattern_metrics["H_t_min"] = float(final_h.min().item())
        pattern_metrics["H_t_max"] = float(final_h.max().item())
        pattern_metrics["H_t_mean"] = float(final_h.mean().item())
        pattern_metrics["H_t_binary_fraction"] = float(
            ((final_h < 0.05) | (final_h > 0.95)).float().mean().item()
        )

    with (run_dir / "metrics" / f"{phase_name}_pattern_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(pattern_metrics, handle, indent=2)

    return history, pattern_snapshot, pattern_metrics, step_offset + max_steps


@torch.no_grad()
def _evaluate_loader(
    model: DifferentiableMicroscope,
    dataloader: DataLoader,
    device: torch.device,
    apply_noise: bool,
    sigmoid_m: float,
) -> tuple[float, float]:
    model.eval()
    total_mse = 0.0
    total_ssim = 0.0
    count = 0
    for batch in dataloader:
        specimen = batch.to(device)
        outputs = model(specimen, sigmoid_m=sigmoid_m, apply_noise=apply_noise)
        total_mse += float(mse(outputs["x_recon"], specimen).item())
        total_ssim += float(ssim(outputs["x_recon"], specimen).item())
        count += 1
    model.train()
    return total_mse / max(count, 1), total_ssim / max(count, 1)
