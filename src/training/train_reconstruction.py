"""Train the differentiable microscope for image reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from evaluation.eval_reconstruction import (
    EvaluationResult,
    evaluate_reconstruction,
    log_evaluation_result,
    save_evaluation_artifacts,
)
from models.microscope import DifferentiableMicroscope
from models.pattern_generator import PatternGenerator, SigmoidSchedule
from training.dataloaders import build_dataloader
from training.losses import reconstruction_loss_l1
from training.schedulers import configure_training_stage
from training.step_trainer import train_steps
from utils.device import device_from_config
from utils.experiment_config import load_experiment_config
from utils.logging import (
    copy_assumptions,
    ensure_run_directory,
    save_run_config,
    save_run_metadata,
)
from utils.reproducibility import get_git_commit_hash, set_seed


def build_optimizer(model: DifferentiableMicroscope, config: dict[str, Any]) -> torch.optim.Optimizer:
    training_cfg = config["training"]
    illumination_lr = training_cfg["illumination_lr"]
    inverse_lr = training_cfg["inverse_lr"]

    param_groups = []
    illumination_params = model.illumination_parameters()
    inverse_params = model.inverse_parameters()

    if illumination_params:
        param_groups.append({"params": illumination_params, "lr": illumination_lr})
    if inverse_params:
        param_groups.append({"params": inverse_params, "lr": inverse_lr})

    return torch.optim.Adam(param_groups)


def train_one_epoch(
    model: DifferentiableMicroscope,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    schedule: SigmoidSchedule,
    config: dict[str, Any],
) -> float:
    model.train()
    force_freeze = not model.pattern_generator.patterns_are_learnable()
    sigmoid_m = configure_training_stage(model, schedule, epoch, force_freeze_patterns=force_freeze)

    total_loss = 0.0
    count = 0
    apply_noise = config["detector_noise"].get("apply_noise", True)

    for batch in dataloader:
        specimen = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(specimen, sigmoid_m=sigmoid_m, apply_noise=apply_noise)
        loss = reconstruction_loss_l1(outputs["x_recon"], specimen)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        count += 1

    return total_loss / max(count, 1)


def save_checkpoint(
    path: Path,
    model: DifferentiableMicroscope,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def apply_pattern_init_checkpoint(model: DifferentiableMicroscope, config: dict[str, Any]) -> None:
    """Load paired-init W from a saved checkpoint if configured."""
    init_path = config.get("pattern_generator", {}).get("init_checkpoint")
    if not init_path:
        return
    payload = PatternGenerator.load_frequency_checkpoint(init_path)
    model.pattern_generator.load_frequency_weights(payload["W"])


def train(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    experiment_cfg = config["experiment"]
    training_cfg = config["training"]
    seed = experiment_cfg["seed"]
    set_seed(seed)

    run_dir = ensure_run_directory(output_dir)
    save_run_config(config, run_dir)
    copy_assumptions(run_dir)
    save_run_metadata(
        run_dir,
        {
            "seed": seed,
            "git_commit": get_git_commit_hash(),
            "run_id": experiment_cfg["run_id"],
        },
    )

    device = device_from_config(config)
    print(f"Using device: {device}", flush=True)
    model = DifferentiableMicroscope.from_run_config(config).to(device)
    apply_pattern_init_checkpoint(model, config)
    optimizer = build_optimizer(model, config)

    schedule = SigmoidSchedule.from_dict(config["sigmoid_schedule"])
    train_loader = build_dataloader(config, "train")
    val_loader = build_dataloader(config, "val")
    test_loader = build_dataloader(config, "test")

    if training_cfg.get("max_steps"):
        history, pattern_snapshot, pattern_metrics = train_steps(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            config,
            run_dir,
            schedule,
        )
        checkpoint_path = run_dir / "checkpoints" / "best.pt"
        save_checkpoint(checkpoint_path, model, optimizer, schedule._m, config)
        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, schedule._m, config)

        test_mse, test_ssim = evaluate_reconstruction(
            model,
            test_loader,
            device,
            apply_noise=config["detector_noise"].get("apply_noise", True),
            sigmoid_m=training_cfg.get("fixed_sigmoid_m"),
        )
        results_csv = Path(experiment_cfg.get("results_csv", "experiments/content_aware/results.csv"))
        figure_path = run_dir / "figures" / "reconstruction.png"
        eval_result = EvaluationResult(mse=test_mse, ssim=test_ssim, figure_path=str(figure_path))
        log_evaluation_result(
            results_csv,
            config,
            experiment_cfg["run_id"],
            eval_result,
            str(checkpoint_path),
            split="test",
        )
        summary = {
            "run_dir": str(run_dir),
            "test_mse": test_mse,
            "test_ssim": test_ssim,
            "checkpoint_path": str(checkpoint_path),
            "best_val_mse": pattern_metrics.get("best_val_mse"),
            "best_step": pattern_metrics.get("best_step"),
            "final_val_mse": history[-1]["val_mse"] if history else None,
            "final_loss": history[-1]["loss"] if history else None,
            "pattern_delta": pattern_metrics.get("pattern_delta"),
            "w_delta": pattern_metrics.get("w_delta"),
            "detector_delta": pattern_metrics.get("detector_delta"),
            "max_grad_norm_W": pattern_metrics.get("max_grad_norm_W"),
            "H_t_min": pattern_metrics.get("H_t_min"),
            "H_t_max": pattern_metrics.get("H_t_max"),
            "H_t_mean": pattern_metrics.get("H_t_mean"),
            "H_t_binary_fraction": pattern_metrics.get("H_t_binary_fraction"),
        }
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        with (run_dir / "metrics" / "run_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary

    history: list[dict[str, float]] = []
    best_val_mse = float("inf")
    checkpoint_path = run_dir / "checkpoints" / "last.pt"

    for epoch in range(1, training_cfg["num_epochs"] + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            schedule,
            config,
        )
        val_mse, val_ssim = evaluate_reconstruction(
            model,
            val_loader,
            device,
            apply_noise=config["detector_noise"].get("apply_noise", True),
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mse": val_mse,
                "val_ssim": val_ssim,
            }
        )
        save_checkpoint(checkpoint_path, model, optimizer, epoch, config)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, epoch, config)

    with (run_dir / "metrics" / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    best_path = run_dir / "checkpoints" / "best.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        checkpoint_path = best_path
    else:
        checkpoint_path = run_dir / "checkpoints" / "last.pt"

    figure_path = save_evaluation_artifacts(
        model,
        test_loader,
        run_dir,
        device,
        apply_noise=config["detector_noise"].get("apply_noise", True),
    )
    test_mse, test_ssim = evaluate_reconstruction(
        model,
        test_loader,
        device,
        apply_noise=config["detector_noise"].get("apply_noise", True),
        sigmoid_m=training_cfg.get("fixed_sigmoid_m"),
    )

    results_csv = Path(experiment_cfg.get("results_csv", "experiments/content_aware/results.csv"))
    eval_result = EvaluationResult(mse=test_mse, ssim=test_ssim, figure_path=str(figure_path))
    log_evaluation_result(
        results_csv,
        config,
        experiment_cfg["run_id"],
        eval_result,
        str(checkpoint_path),
        split="test",
    )

    return {
        "run_dir": str(run_dir),
        "test_mse": test_mse,
        "test_ssim": test_ssim,
        "checkpoint_path": str(checkpoint_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train differentiable microscopy reconstruction model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", default=None, help="Override experiment output directory")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:1 or gpu1")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.device:
        config.setdefault("experiment", {})
        config["experiment"]["device"] = args.device
    output_dir = args.output_dir or config["experiment"]["output_dir"]
    summary = train(config, output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
