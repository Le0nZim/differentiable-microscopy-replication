"""Full staged training with inverse warmup, joint soft, and m hardening."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.eval_reconstruction import EvaluationResult, log_evaluation_result
from evaluation.pattern_hardening import evaluate_all_pattern_variants, save_variant_artifacts
from models.microscope import DifferentiableMicroscope
from models.pattern_generator import SigmoidSchedule
from training.dataloaders import build_dataloader
from training.hardening_trainer import train_fixed_m_phase
from training.pattern_tracking import PatternSnapshot
from training.train_reconstruction import apply_pattern_init_checkpoint, build_optimizer, save_checkpoint
from utils.device import device_from_config
from utils.logging import copy_assumptions, ensure_run_directory, save_run_config, save_run_metadata
from utils.reproducibility import get_git_commit_hash, set_seed


def _staged_phases(training_cfg: dict[str, Any]) -> list[tuple[str, float, int, bool]]:
    """Return (phase_name, sigmoid_m, max_steps, freeze_illumination) tuples."""
    staged = training_cfg.get("staged_hardening", {})
    inverse_steps = int(staged.get("inverse_warmup_steps", 3000))
    joint_steps = int(staged.get("joint_soft_steps", 7350))
    harden_steps = int(staged.get("harden_steps_per_m", 1500))
    harden_values = [float(v) for v in staged.get("harden_m_values", [2, 4, 8])]

    phases: list[tuple[str, float, int, bool]] = [
        ("inverse_warmup_m1", 1.0, inverse_steps, True),
        ("joint_soft_m1", 1.0, joint_steps, False),
    ]
    for m in harden_values:
        phases.append((f"harden_m{int(m)}", m, harden_steps, False))
    return phases


def train_staged_hardening(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Run inverse warmup → joint soft → m hardening for learnable illumination."""
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
            "training_mode": "staged_hardening",
        },
    )

    device = device_from_config(config)
    print(f"Using device: {device}", flush=True)
    model = DifferentiableMicroscope.from_run_config(config).to(device)
    apply_pattern_init_checkpoint(model, config)
    optimizer = build_optimizer(model, config)

    train_loader = build_dataloader(config, "train")
    val_loader = build_dataloader(config, "val")
    test_loader = build_dataloader(config, "test")
    apply_noise = config["detector_noise"].get("apply_noise", False)

    sharpen_m = float(training_cfg.get("sharpen_eval_m", 10.0))
    phases = _staged_phases(training_cfg)

    all_history: list[dict] = []
    phase_summaries: list[dict] = []
    step_offset = 0
    pattern_snapshot: PatternSnapshot | None = None
    reference_specimen = None
    final_training_m = 1.0

    for phase_name, sigmoid_m, max_steps, freeze_illum in phases:
        print(f"\n=== {experiment_cfg['run_id']}: {phase_name} (m={sigmoid_m}, steps={max_steps}) ===", flush=True)
        history, pattern_snapshot, pattern_metrics, step_offset = train_fixed_m_phase(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            config,
            run_dir,
            sigmoid_m=sigmoid_m,
            max_steps=max_steps,
            phase_name=phase_name,
            step_offset=step_offset,
            pattern_snapshot=pattern_snapshot,
            reference_specimen=reference_specimen,
            append_log=step_offset > 0,
            freeze_illumination=freeze_illum,
        )
        all_history.extend(history)
        if reference_specimen is None:
            reference_specimen = next(iter(val_loader)).to(device)
        final_training_m = sigmoid_m

        eval_results = evaluate_all_pattern_variants(
            model,
            {"val": val_loader, "test": test_loader},
            device,
            training_m=sigmoid_m,
            sharpen_m=sharpen_m,
            apply_noise=apply_noise,
            reference_specimen=reference_specimen,
            initial_patterns=pattern_snapshot.initial_patterns if pattern_snapshot else None,
            initial_y_down=pattern_snapshot.initial_y_down if pattern_snapshot else None,
        )
        eval_dir = run_dir / "eval" / phase_name
        save_variant_artifacts(
            model,
            reference_specimen,
            eval_dir,
            training_m=sigmoid_m,
            sharpen_m=sharpen_m,
            apply_noise=apply_noise,
        )
        with (eval_dir / "variant_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(eval_results, handle, indent=2)

        phase_summaries.append(
            {
                "phase": phase_name,
                "training_m": sigmoid_m,
                "steps": max_steps,
                "best_val_mse": pattern_metrics.get("best_val_mse"),
                "best_step": pattern_metrics.get("best_step"),
                "pattern_delta": pattern_metrics.get("pattern_delta"),
                "detector_delta": eval_results.get("detector_delta"),
                "soft_test_mse": eval_results["variants"]["soft"]["splits"]["test"]["mse"],
                "soft_test_ssim": eval_results["variants"]["soft"]["splits"]["test"]["ssim"],
                "sharpened_test_mse": eval_results["variants"]["sharpened"]["splits"]["test"]["mse"],
                "thresholded_test_mse": eval_results["variants"]["thresholded"]["splits"]["test"]["mse"],
                "H_t_binary_fraction": pattern_metrics.get("H_t_binary_fraction"),
            }
        )

    schedule = SigmoidSchedule.from_dict(config["sigmoid_schedule"])
    _ = schedule
    save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, int(final_training_m), config)
    save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, int(final_training_m), config)

    with (run_dir / "metrics" / "step_history.json").open("w", encoding="utf-8") as handle:
        json.dump(all_history, handle, indent=2)
    with (run_dir / "metrics" / "phase_summaries.json").open("w", encoding="utf-8") as handle:
        json.dump(phase_summaries, handle, indent=2)

    final_eval = evaluate_all_pattern_variants(
        model,
        {"val": val_loader, "test": test_loader},
        device,
        training_m=final_training_m,
        sharpen_m=sharpen_m,
        apply_noise=apply_noise,
        reference_specimen=reference_specimen,
        initial_patterns=pattern_snapshot.initial_patterns if pattern_snapshot else None,
        initial_y_down=pattern_snapshot.initial_y_down if pattern_snapshot else None,
    )
    save_variant_artifacts(
        model,
        reference_specimen,
        run_dir / "eval" / "final",
        training_m=final_training_m,
        sharpen_m=sharpen_m,
        apply_noise=apply_noise,
    )

    soft_test = final_eval["variants"]["soft"]["splits"]["test"]
    sharp_test = final_eval["variants"]["sharpened"]["splits"]["test"]
    thresh_test = final_eval["variants"]["thresholded"]["splits"]["test"]

    results_csv = Path(experiment_cfg.get("results_csv", "experiments/content_aware/results.csv"))
    eval_result = EvaluationResult(
        mse=soft_test["mse"],
        ssim=soft_test["ssim"],
        figure_path=str(run_dir / "eval" / "final" / "figures" / "reconstruction_soft.png"),
    )
    log_evaluation_result(
        results_csv,
        config,
        experiment_cfg["run_id"],
        eval_result,
        str(run_dir / "checkpoints" / "best.pt"),
        split="test_soft",
    )

    summary = {
        "run_dir": str(run_dir),
        "training_mode": "staged_hardening",
        "final_training_m": final_training_m,
        "test_mse": soft_test["mse"],
        "test_ssim": soft_test["ssim"],
        "sharpened_test_mse": sharp_test["mse"],
        "sharpened_test_ssim": sharp_test["ssim"],
        "thresholded_test_mse": thresh_test["mse"],
        "thresholded_test_ssim": thresh_test["ssim"],
        "best_val_mse": phase_summaries[-1]["best_val_mse"] if phase_summaries else None,
        "pattern_delta": pattern_snapshot.pattern_delta() if pattern_snapshot else None,
        "detector_delta": final_eval.get("detector_delta"),
        "H_t_binary_fraction": phase_summaries[-1].get("H_t_binary_fraction") if phase_summaries else None,
        "phase_summaries": phase_summaries,
        "checkpoint_path": str(run_dir / "checkpoints" / "best.pt"),
    }
    with (run_dir / "metrics" / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
