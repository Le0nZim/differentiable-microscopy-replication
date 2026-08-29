#!/usr/bin/env python3
"""Figure 4 task-aware segmentation — CALIBRATED-preprocessing re-run (BBBC022 proxy).

Identical three-stage regime to scripts/fig4_seg_fix_train.py (paper §5.3 / B.0.1)
for the {x64, x256, x1024} x {fixed pseudo-random, learnable} matrix, with ONE
deliberate change: the dataset preprocessing is ``bbbc022_calibrated`` instead of
``paper_strict``. Because the preprocessing changed, Stage 1 (content-aware
pretrain) is RETRAINED from scratch on calibrated data (``stage1_mode=train``)
using the identical Stage-1 hyperparameters — it is NOT reused from the frozen
paper_strict am2 checkpoints. Writes into
``experiments/figure4_bbbc022_segmentation_calibrated_v1/runs/``.

Usage:
  python scripts/fig4_seg_calibrated_train.py --device cuda:0                 # all 6 cells
  python scripts/fig4_seg_calibrated_train.py --device cuda:1 --cells x64 x256  # subset
  python scripts/fig4_seg_calibrated_train.py --device cuda:0 --smoke         # fast validation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.train_task_aware_segmentation import train_task_aware_segmentation  # noqa: E402
from utils.device import resolve_device  # noqa: E402
from utils.experiment_config import compression_ratio, load_experiment_config  # noqa: E402

CONFIG_PATH = ROOT / "experiments/figure4_bbbc022_segmentation_calibrated_v1/configs/fig4_seg_calibrated_base.yaml"
OUT = ROOT / "experiments/figure4_bbbc022_segmentation_calibrated_v1/runs"
COMPRESSIONS = [("x64", 16), ("x256", 32), ("x1024", 64)]
VARIANTS = [("random_fixed", False), ("learnable_frequency", True)]


def _build_config(comp_name: str, downscale: int, pattern: str, learnable: bool, seed: int,
                  device: str, smoke: bool) -> dict:
    config = load_experiment_config(CONFIG_PATH)
    num_patterns = config["pattern_generator"]["num_patterns"]
    config["forward_model"]["downscale_factor"] = downscale
    config["inverse_model"]["upsampling"]["downscale_factor"] = downscale
    config["inverse_model"]["upsampling"]["num_patterns"] = num_patterns
    config["inverse_model"]["reconstruction"]["in_channels"] = num_patterns
    config["pattern_generator"]["mode"] = pattern
    config["training"]["learn_patterns"] = learnable
    # Retrain Stage 1 from scratch on calibrated data (no am2 reuse).
    config["training"]["task_aware"]["stage1_mode"] = "train"
    config["training"]["task_aware"]["content_aware_checkpoint"] = None
    config["experiment"]["seed"] = seed
    config["dataset"]["seed"] = seed
    config["pattern_generator"]["seed"] = seed
    config["experiment"]["device"] = device
    config["experiment"]["compression"] = compression_ratio(downscale, num_patterns)
    run_id = f"taskaware_{comp_name}_{pattern}"
    config["experiment"]["run_id"] = run_id
    config["experiment"]["output_dir"] = str(OUT / f"{run_id}_seed{seed}")
    if smoke:
        s1 = config["training"]["task_aware"]["stage1"]
        s1["learnable"] = {"inverse_warmup_steps": 20, "joint_soft_steps": 20,
                            "harden_m_values": [2, 4], "harden_steps_per_m": 10}
        s1["fixed_steps"] = 30
        config["training"]["task_aware"]["seg_head_steps"] = 20
        config["training"]["task_aware"]["finetune_steps"] = 20
        config["training"]["log_every"] = 10
    return config


def run(device: str, seed: int, cells: list[str] | None, smoke: bool) -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for comp_name, downscale in COMPRESSIONS:
        for pattern, learnable in VARIANTS:
            run_id = f"taskaware_{comp_name}_{pattern}"
            if cells and run_id not in cells and comp_name not in cells:
                continue
            out_dir = OUT / f"{run_id}_seed{seed}"
            summary_path = out_dir / "metrics" / "run_summary.json"
            if summary_path.exists() and not smoke:
                print(f"Skipping existing {out_dir.name}", flush=True)
                summary = json.loads(summary_path.read_text())
            else:
                print(f"\n========== fig4-calibrated {run_id} seed={seed} device={device} ==========", flush=True)
                config = _build_config(comp_name, downscale, pattern, learnable, seed, device, smoke)
                summary = train_task_aware_segmentation(config, out_dir)
            results.append({
                "run_id": run_id, "compression": comp_name, "pattern": pattern, "learnable": learnable,
                "test_dice": summary["test_dice"], "test_iou": summary["test_iou"],
                "test_dice_at_0p5": summary.get("test_dice_at_0p5"),
                "selected_threshold": summary.get("selected_threshold"),
                "stage2_val_dice": summary.get("stage2_val_dice"),
                "illumination_pattern_delta_l2": summary.get("illumination_pattern_delta_l2"),
                "seed": seed,
            })
    if not smoke:
        (OUT.parent / "aggregate_partial.json").write_text(
            json.dumps({"results": results, "seed": seed}, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fig4 calibrated-preprocessing task-aware segmentation runner")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cells", nargs="*", default=None,
                        help="Optional subset: run_ids or compression labels (e.g. x64 taskaware_x256_learnable_frequency)")
    parser.add_argument("--smoke", action="store_true", help="Tiny step counts to validate the pipeline end-to-end")
    args = parser.parse_args()
    resolve_device(args.device)
    run(args.device, args.seed, args.cells, args.smoke)


if __name__ == "__main__":
    main()
