"""Step-based training loop with dense metric logging."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any

from copy import deepcopy

import torch
from torch.utils.data import DataLoader

from evaluation.metrics import mse, ssim
from evaluation.pattern_inspection import save_pattern_inspection
from models.microscope import DifferentiableMicroscope
from models.pattern_generator import SigmoidSchedule
from training.losses import reconstruction_loss_l1
from training.metrics_logging import batch_reconstruction_metrics, collect_step_metrics
from training.pattern_tracking import (
    PatternSnapshot,
    capture_detector_snapshot,
    capture_pattern_snapshot,
    finalize_pattern_snapshot,
)
from training.schedulers import configure_training_stage
from utils.logging import save_measurement_grid, save_patterns


def _should_learn_patterns(config: dict[str, Any], model: DifferentiableMicroscope) -> bool:
    training_cfg = config["training"]
    if "learn_patterns" in training_cfg:
        return bool(training_cfg["learn_patterns"])
    return model.pattern_generator.patterns_are_learnable()


def _apply_pattern_freeze(model: DifferentiableMicroscope, config: dict[str, Any], epoch: int, schedule: SigmoidSchedule) -> float:
    if not _should_learn_patterns(config, model):
        model.set_illumination_trainable(False)
        return schedule.get_m()
    force_freeze = not model.pattern_generator.patterns_are_learnable()
    return configure_training_stage(model, schedule, epoch, force_freeze_patterns=force_freeze)


def train_steps(
    model: DifferentiableMicroscope,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict[str, Any],
    run_dir: Path,
    schedule: SigmoidSchedule,
) -> tuple[list[dict[str, Any]], PatternSnapshot, dict[str, float]]:
    training_cfg = config["training"]
    max_steps = int(training_cfg["max_steps"])
    log_every = int(training_cfg.get("log_every", 25))
    apply_noise = config["detector_noise"].get("apply_noise", True)

    history: list[dict[str, Any]] = []
    step_log_path = run_dir / "metrics" / "step_log.csv"
    step_log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "step",
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

    step = 0
    epoch = 1
    steps_per_epoch = max(len(train_loader), 1)
    fixed_sigmoid_m = training_cfg.get("fixed_sigmoid_m")
    sigmoid_m = (
        float(fixed_sigmoid_m)
        if fixed_sigmoid_m is not None
        else _apply_pattern_freeze(model, config, epoch, schedule)
    )
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
    if not _should_learn_patterns(config, model):
        model.set_illumination_trainable(False)
    train_iter = itertools.cycle(train_loader)
    best_val_mse = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_step = 0

    with step_log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        while step < max_steps:
            step += 1
            new_epoch = (step - 1) // steps_per_epoch + 1
            if new_epoch != epoch:
                epoch = new_epoch
                if fixed_sigmoid_m is None:
                    sigmoid_m = _apply_pattern_freeze(model, config, epoch, schedule)

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

            if step % log_every != 0 and step != max_steps:
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
                best_step = step

            row = {
                "step": step,
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
                f"step={step} loss={row['loss']:.5f} val_mse={row['val_mse']:.5f} "
                f"val_ssim={row['val_ssim']:.4f} grad_W={row['grad_norm_W']:.4e} "
                f"H_t=[{row['H_t_min']:.3f},{row['H_t_max']:.3f}]",
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
    pattern_metrics = pattern_snapshot.to_dict()
    pattern_metrics["best_val_mse"] = best_val_mse
    pattern_metrics["best_step"] = best_step
    if pattern_snapshot.final_patterns is not None:
        final_h = pattern_snapshot.final_patterns
        pattern_metrics["H_t_min"] = float(final_h.min().item())
        pattern_metrics["H_t_max"] = float(final_h.max().item())
        pattern_metrics["H_t_mean"] = float(final_h.mean().item())
        pattern_metrics["H_t_binary_fraction"] = float(
            ((final_h < 0.05) | (final_h > 0.95)).float().mean().item()
        )

    with (run_dir / "metrics" / "pattern_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(pattern_metrics, handle, indent=2)

    with (run_dir / "metrics" / "step_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    _save_step_run_artifacts(
        model,
        val_loader,
        device,
        run_dir,
        apply_noise,
        sigmoid_m,
        pattern_snapshot,
    )
    return history, pattern_snapshot, pattern_metrics


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


@torch.no_grad()
def _save_step_run_artifacts(
    model: DifferentiableMicroscope,
    val_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
    apply_noise: bool,
    sigmoid_m: float,
    pattern_snapshot: PatternSnapshot,
) -> None:
    model.eval()
    batch = next(iter(val_loader)).to(device)
    outputs = model(batch, sigmoid_m=sigmoid_m, apply_noise=apply_noise)

    save_patterns(outputs["patterns"], run_dir)
    save_pattern_inspection(outputs["patterns"], run_dir, sigmoid_m=sigmoid_m, prefix="H_t_final")
    if pattern_snapshot.initial_patterns is not None:
        save_pattern_inspection(
            pattern_snapshot.initial_patterns.to(outputs["patterns"].device),
            run_dir,
            sigmoid_m=sigmoid_m,
            prefix="H_t_initial",
        )

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_measurement_grid(batch, figures_dir / "ground_truth.png")
    save_measurement_grid(outputs["x_recon"], figures_dir / "reconstruction.png")
    save_measurement_grid(outputs["y_down"], figures_dir / "measurements.png", nrow=4)
