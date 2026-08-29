#!/usr/bin/env python3
"""BBBC022 substitute task-aware segmentation (paper B.0.1; AM-2 / RR-3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.train_task_aware_segmentation import train_task_aware_segmentation
from utils.device import resolve_device
from utils.experiment_config import load_experiment_config

OUT = ROOT / "experiments/task_aware_segmentation/bbbc022_segmentation_taskaware"
POSTHOC = ROOT / "experiments/task_aware_segmentation/bbbc022_segmentation"
BBBC022_META = ROOT / "experiments/ablations"
COMPRESSIONS = [("x64", 16, 4), ("x256", 32, 4), ("x1024", 64, 4)]


def run_taskaware_experiments(device: str, seed: int) -> list[dict]:
    results = []
    for comp_name, d, t in COMPRESSIONS:
        for pattern, learnable in [("random_fixed", False), ("learnable_frequency", True)]:
            run_id = f"seg_{comp_name}_{pattern}"
            config = load_experiment_config(ROOT / "configs/base_bbbc022_substitute.yaml")
            if (BBBC022_META / "preprocessing_report.json").exists():
                config["dataset"]["preprocessing_mode"] = json.loads(
                    (BBBC022_META / "preprocessing_report.json").read_text()
                )["chosen_official_mode"]
            config["dataset"]["return_mask"] = True
            config["forward_model"]["downscale_factor"] = d
            config["pattern_generator"]["mode"] = pattern
            config["pattern_generator"]["num_patterns"] = t
            config["inverse_model"]["upsampling"]["downscale_factor"] = d
            config["inverse_model"]["upsampling"]["num_patterns"] = t
            config["inverse_model"]["upsampling"]["mode"] = "locality_aware"
            config["inverse_model"]["reconstruction"]["in_channels"] = t
            config["experiment"]["seed"] = seed
            config["dataset"]["seed"] = seed
            config["pattern_generator"]["seed"] = seed
            config["experiment"]["device"] = device
            config["experiment"]["run_id"] = run_id
            config["training"]["learn_patterns"] = learnable
            config["training"]["task_aware"] = {
                "content_aware_checkpoint": str(
                    POSTHOC / f"{run_id}_seed{seed}" / "checkpoints" / "best.pt"
                ),
                "seg_head_steps": 1000,
                "finetune_steps": 2000,
                "eval_sigmoid_m": 10.0,
            }
            config["segmentation_head"] = {
                "in_channels": 1,
                "hidden_channels": [16, 16, 1],
                "kernel_size": 3,
                "padding": 1,
            }

            out_dir = OUT / f"{run_id}_seed{seed}"
            if (out_dir / "metrics" / "run_summary.json").exists():
                print(f"Skipping existing {out_dir.name}", flush=True)
                summary = json.loads((out_dir / "metrics" / "run_summary.json").read_text())
            else:
                print(f"\n========== task-aware {run_id} seed={seed} ==========", flush=True)
                summary = train_task_aware_segmentation(config, out_dir)

            row = {
                "run_id": run_id,
                "compression": comp_name,
                "pattern": pattern,
                "dice_mean": summary["test_dice"],
                "iou_mean": summary["test_iou"],
                "seed": seed,
                "training_mode": "task_aware_bce",
            }
            results.append(row)

    posthoc = json.loads((POSTHOC / "aggregate_summary.json").read_text())["results"]
    comparison = []
    for row in results:
        match = next(r for r in posthoc if r["run_id"] == row["run_id"])
        comparison.append(
            {
                "run_id": row["run_id"],
                "posthoc_dice": match["dice_mean"],
                "taskaware_dice": row["dice_mean"],
                "taskaware_beats_posthoc": row["dice_mean"] > match["dice_mean"],
            }
        )

    payload = {
        "label": "BBBC022 substitute task-aware segmentation (paper B.0.1 proxy; not U2OS)",
        "results": results,
        "comparison_to_posthoc": comparison,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "aggregate_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = [
        "# Task-aware segmentation report (BBBC022 substitute)\n\n",
        "Paper B.0.1: content-aware checkpoint → seg-head warmup → end-to-end BCE finetune.\n\n",
        "| compression | pattern | post-hoc Dice | task-aware Dice | task-aware wins? |\n",
        "|---|---|---:|---:|---|\n",
    ]
    for item in comparison:
        comp = item["run_id"].replace("seg_", "").rsplit("_", 1)
        md.append(
            f"| {comp[0]} | {comp[1]} | {item['posthoc_dice']:.4f} | "
            f"{item['taskaware_dice']:.4f} | {'yes' if item['taskaware_beats_posthoc'] else 'no'} |\n"
        )
    (OUT / "taskaware_vs_posthoc.md").write_text("".join(md), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="BBBC022 task-aware segmentation (RR-3)")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    resolve_device(args.device)
    run_taskaware_experiments(args.device, args.seed)


if __name__ == "__main__":
    main()
