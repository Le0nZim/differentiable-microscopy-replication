"""Reconstruction evaluation and artifact export."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from evaluation.metrics import mse, ssim
from models.microscope import DifferentiableMicroscope
from training.dataloaders import build_dataloader
from utils.device import device_from_config, resolve_device
from utils.experiment_config import load_experiment_config
from utils.logging import append_results_row, save_measurement_grid, save_patterns


@dataclass
class EvaluationResult:
    mse: float
    ssim: float
    figure_path: str


@torch.no_grad()
def evaluate_reconstruction(
    model: DifferentiableMicroscope,
    dataloader: DataLoader,
    device: torch.device,
    *,
    apply_noise: bool | None = None,
    sigmoid_m: float | None = None,
) -> tuple[float, float]:
    """Compute mean MSE and SSIM over a dataloader."""
    model.eval()
    total_mse = 0.0
    total_ssim = 0.0
    count = 0

    for batch in dataloader:
        specimen = batch.to(device) if torch.is_tensor(batch) else batch[0].to(device)
        outputs = model(specimen, sigmoid_m=sigmoid_m, apply_noise=apply_noise)
        total_mse += float(mse(outputs["x_recon"], specimen).item())
        total_ssim += float(ssim(outputs["x_recon"], specimen).item())
        count += 1

    if count == 0:
        raise ValueError("Cannot evaluate an empty dataloader")
    return total_mse / count, total_ssim / count


@torch.no_grad()
def save_evaluation_artifacts(
    model: DifferentiableMicroscope,
    dataloader: DataLoader,
    output_dir: str | Path,
    device: torch.device,
    *,
    max_batches: int = 1,
    apply_noise: bool | None = None,
) -> Path:
    """Save example reconstructions, measurements, and learned patterns."""
    model.eval()
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_path = None
    for batch_idx, batch in enumerate(dataloader):
        specimen = batch.to(device) if torch.is_tensor(batch) else batch[0].to(device)
        outputs = model(specimen, apply_noise=apply_noise)

        save_patterns(outputs["patterns"], output_dir)
        save_measurement_grid(specimen, figures_dir / f"ground_truth_batch{batch_idx}.png")
        save_measurement_grid(outputs["x_recon"], figures_dir / f"reconstruction_batch{batch_idx}.png")
        save_measurement_grid(outputs["y_down"], figures_dir / f"measurements_batch{batch_idx}.png", nrow=4)
        torch.save(outputs["y_down"].detach().cpu(), figures_dir / f"measurements_batch{batch_idx}.pt")
        figure_path = figures_dir / f"reconstruction_batch{batch_idx}.png"

        if batch_idx + 1 >= max_batches:
            break

    if figure_path is None:
        raise ValueError("Cannot save artifacts from an empty dataloader")
    return figure_path


def log_evaluation_result(
    results_csv: str | Path,
    config: dict[str, Any],
    run_id: str,
    metrics: EvaluationResult,
    checkpoint_path: str,
    split: str,
) -> None:
    dataset_cfg = config["dataset"]
    pattern_cfg = config["pattern_generator"]
    forward_cfg = config["forward_model"]
    noise_cfg = config["detector_noise"]
    upsampling_cfg = config["inverse_model"]["upsampling"]
    downscale = forward_cfg["downscale_factor"]
    num_patterns = pattern_cfg["num_patterns"]

    append_results_row(
        results_csv,
        {
            "run_id": run_id,
            "dataset": dataset_cfg.get("name", ""),
            "pattern_mode": pattern_cfg["mode"],
            "downscale_factor": downscale,
            "num_patterns": num_patterns,
            "compression": (downscale**2) / num_patterns,
            "upsampling_type": upsampling_cfg["mode"],
            "frequency_domain_optimization": pattern_cfg["mode"] == "learnable_frequency",
            "noise_mode": noise_cfg["mode"],
            "photon_count": noise_cfg.get("photon_count", ""),
            "sigma_read": noise_cfg.get("sigma_read", ""),
            "loss": config["training"].get("loss", "l1"),
            "MSE": metrics.mse,
            "SSIM": metrics.ssim,
            "checkpoint_path": checkpoint_path,
            "figure_path": metrics.figure_path,
            "notes": split,
        },
    )


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    *,
    split: str = "test",
    output_dir: str | Path | None = None,
) -> EvaluationResult:
    """Load a checkpoint and evaluate it on a dataset split."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = device_from_config(config)
    print(f"Using device: {device}", flush=True)

    model = DifferentiableMicroscope.from_run_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataloader = build_dataloader(config, split)
    apply_noise = config["detector_noise"].get("apply_noise", True)
    eval_mse, eval_ssim = evaluate_reconstruction(model, dataloader, device, apply_noise=apply_noise)

    figure_path = ""
    if output_dir is not None:
        figure_path = str(
            save_evaluation_artifacts(
                model,
                dataloader,
                output_dir,
                device,
                apply_noise=apply_noise,
            )
        )

    return EvaluationResult(mse=eval_mse, ssim=eval_ssim, figure_path=figure_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained reconstruction checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:1 or gpu1")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.device:
        config.setdefault("experiment", {})
        config["experiment"]["device"] = args.device
    result = evaluate_checkpoint(
        args.checkpoint,
        config,
        split=args.split,
        output_dir=args.output_dir,
    )

    if args.results_csv:
        log_evaluation_result(
            args.results_csv,
            config,
            config["experiment"]["run_id"],
            result,
            args.checkpoint,
            split=args.split,
        )

    print(json.dumps({"mse": result.mse, "ssim": result.ssim, "figure_path": result.figure_path}, indent=2))


if __name__ == "__main__":
    main()
