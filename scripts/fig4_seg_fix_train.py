#!/usr/bin/env python3
"""Figure 4 task-aware segmentation — isolated verified re-run (BBBC022 proxy).

Re-runs the faithful three-stage procedure (paper §5.3 / B.0.1) for the
{x64, x256, x1024} x {fixed pseudo-random, learnable} matrix into the isolated
directory ``experiments/figure4_bbbc022_segmentation_fix_v1/runs/``. Stage 1
(content-aware pretrain) is REUSED from the frozen ``am2_task_aware_full``
checkpoints (``stage1_mode=load``) so only Stage 2 (frozen seg-head) and Stage 3
(end-to-end task-aware finetune) are recomputed. Aggregates Dice/IoU, records
the Stage-2 (content-aware) vs. final (task-aware) attribution, and writes a
report. Additive only — the frozen am2 run is not modified.
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

CONFIG_PATH = ROOT / "experiments/figure4_bbbc022_segmentation_fix_v1/configs/fig4_seg_fix_base.yaml"
OUT = ROOT / "experiments/figure4_bbbc022_segmentation_fix_v1/runs"
AM2 = ROOT / "experiments/task_aware_segmentation/am2_task_aware_full"
# (label, downscale_factor); num_patterns T = 4 -> compression = d^2 / T.
COMPRESSIONS = [("x64", 16), ("x256", 32), ("x1024", 64)]
VARIANTS = [("random_fixed", False), ("learnable_frequency", True)]


def _stage1_ckpt(comp_name: str, pattern: str, seed: int) -> Path:
    return AM2 / f"taskaware_{comp_name}_{pattern}_seed{seed}" / "stage1_content_aware/checkpoints/best.pt"


def _build_config(comp_name: str, downscale: int, pattern: str, learnable: bool, seed: int, device: str) -> dict:
    config = load_experiment_config(CONFIG_PATH)
    num_patterns = config["pattern_generator"]["num_patterns"]
    config["forward_model"]["downscale_factor"] = downscale
    config["inverse_model"]["upsampling"]["downscale_factor"] = downscale
    config["inverse_model"]["upsampling"]["num_patterns"] = num_patterns
    config["inverse_model"]["reconstruction"]["in_channels"] = num_patterns
    config["pattern_generator"]["mode"] = pattern
    config["training"]["learn_patterns"] = learnable
    config["training"]["task_aware"]["stage1_mode"] = "load"
    ckpt = _stage1_ckpt(comp_name, pattern, seed)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing frozen Stage-1 content-aware checkpoint: {ckpt}")
    config["training"]["task_aware"]["content_aware_checkpoint"] = str(ckpt)
    config["experiment"]["seed"] = seed
    config["dataset"]["seed"] = seed
    config["pattern_generator"]["seed"] = seed
    config["experiment"]["device"] = device
    config["experiment"]["compression"] = compression_ratio(downscale, num_patterns)
    run_id = f"taskaware_{comp_name}_{pattern}"
    config["experiment"]["run_id"] = run_id
    config["experiment"]["output_dir"] = str(OUT / f"{run_id}_seed{seed}")
    return config


def run(device: str, seed: int, cells: list[str] | None) -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for comp_name, downscale in COMPRESSIONS:
        for pattern, learnable in VARIANTS:
            run_id = f"taskaware_{comp_name}_{pattern}"
            if cells and run_id not in cells and comp_name not in cells:
                continue
            out_dir = OUT / f"{run_id}_seed{seed}"
            summary_path = out_dir / "metrics" / "run_summary.json"
            if summary_path.exists():
                print(f"Skipping existing {out_dir.name}", flush=True)
                summary = json.loads(summary_path.read_text())
            else:
                print(f"\n========== fig4-fix {run_id} seed={seed} ==========", flush=True)
                config = _build_config(comp_name, downscale, pattern, learnable, seed, device)
                summary = train_task_aware_segmentation(config, out_dir)
            results.append({
                "run_id": run_id,
                "compression": comp_name,
                "pattern": pattern,
                "learnable": learnable,
                "test_dice": summary["test_dice"],
                "test_iou": summary["test_iou"],
                "test_dice_at_0p5": summary.get("test_dice_at_0p5"),
                "selected_threshold": summary.get("selected_threshold"),
                "stage2_val_dice": summary.get("stage2_val_dice"),
                "val_dice": summary.get("val_dice"),
                "illumination_grad_norm_stage3_max": summary.get("illumination_grad_norm_stage3_max"),
                "inverse_grad_norm_stage3_max": summary.get("inverse_grad_norm_stage3_max"),
                "seg_head_grad_norm_stage2_max": summary.get("seg_head_grad_norm_stage2_max"),
                "illumination_pattern_delta_l2": summary.get("illumination_pattern_delta_l2"),
                "illumination_pattern_delta_relative": summary.get("illumination_pattern_delta_relative"),
                "seed": seed,
            })
    _aggregate(results, seed)
    return results


def _aggregate(results: list[dict], seed: int) -> None:
    if not results:
        print("[aggregate] no results yet; skipping summary", flush=True)
        return
    by_key = {(r["compression"], r["pattern"]): r for r in results}
    have_full_matrix = all((c, p) in by_key for c, _ in COMPRESSIONS for p, _ in VARIANTS)

    payload = {
        "label": "Fig4 task-aware segmentation fix (BBBC022 proxy; Stage-1 reused frozen)",
        "procedure": "Stage1 content-aware (REUSED frozen) -> Stage2 frozen seg-head -> Stage3 end-to-end task-aware finetune",
        "seed": seed,
        "results": results,
    }
    (OUT.parent / "aggregate_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not have_full_matrix:
        print("[aggregate] partial matrix; wrote aggregate_summary.json only", flush=True)
        return

    metrics_summary = {
        "label": payload["label"],
        "procedure": payload["procedure"],
        "seed": seed,
        "compressions": {},
    }
    learnable_beats_fixed = {}
    for comp_name, _ in COMPRESSIONS:
        cell = {}
        for pattern, learnable in VARIANTS:
            row = by_key[(comp_name, pattern)]
            cell["learnable" if learnable else "fixed"] = {
                "test_dice": row["test_dice"],
                "test_iou": row["test_iou"],
                "test_dice_at_0p5": row["test_dice_at_0p5"],
                "selected_threshold": row["selected_threshold"],
                "stage2_val_dice": row["stage2_val_dice"],
            }
        metrics_summary["compressions"][comp_name] = cell
        learnable_beats_fixed[comp_name] = cell["learnable"]["test_dice"] > cell["fixed"]["test_dice"]

    learnable_rows = [r for r in results if r["learnable"]]
    illum_updated = all(
        (r["illumination_pattern_delta_l2"] or 0) > 0 and (r["illumination_grad_norm_stage3_max"] or 0) > 0
        for r in learnable_rows
    )
    metrics_summary["learnable_beats_fixed"] = learnable_beats_fixed
    metrics_summary["stage3_illumination_evidence"] = {
        r["run_id"]: {
            "illumination_grad_norm_stage3_max": r["illumination_grad_norm_stage3_max"],
            "illumination_pattern_delta_l2": r["illumination_pattern_delta_l2"],
            "illumination_pattern_delta_relative": r["illumination_pattern_delta_relative"],
            "inverse_grad_norm_stage3_max": r["inverse_grad_norm_stage3_max"],
            "seg_head_grad_norm_stage2_max": r["seg_head_grad_norm_stage2_max"],
        }
        for r in learnable_rows
    }
    metrics_summary["acceptance_criteria"] = {
        "learnable_beats_fixed_all_compressions": bool(all(learnable_beats_fixed.values())),
        "learnable_illumination_updated_by_seg_loss_in_stage3": bool(illum_updated),
        "outputs_for_x64_x256_x1024": True,
    }
    metrics_summary["status"] = (
        "FIG4_FIX_PASS / RESULTS_PROXY_BBBC022"
        if all(metrics_summary["acceptance_criteria"].values())
        else "FIG4_FIX_REVIEW / RESULTS_PROXY_BBBC022"
    )
    (OUT.parent / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
    _write_report(results)


def _write_report(results: list[dict]) -> None:
    by_key = {(r["compression"], r["pattern"]): r for r in results}
    lines = [
        "# Fig4 task-aware segmentation fix — verified re-run tables (BBBC022 proxy)\n\n",
        "Pseudo-GT masks: **TrackMate** detector (raw MIP>506, 4-connectivity, "
        "Douglas–Peucker ε0.5). Stage 1 content-aware pretrain **reused frozen** from "
        "am2_task_aware_full; Stage 2 (frozen seg-head) + Stage 3 (end-to-end task-aware "
        "finetune) recomputed in this isolated directory. NOT exact U2OS reproduction.\n\n",
        "## Dice / IoU (test, val-selected threshold)\n\n",
        "| compression | illumination | Dice | IoU | Dice@0.5 | thr | stage2 (content-aware) val Dice |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for comp_name, _ in COMPRESSIONS:
        for pattern, learnable in VARIANTS:
            row = by_key[(comp_name, pattern)]
            mode = "learnable" if learnable else "fixed"
            s2 = f"{row['stage2_val_dice']:.4f}" if row.get("stage2_val_dice") is not None else "n/a"
            lines.append(
                f"| {comp_name} | {mode} | {row['test_dice']:.4f} | {row['test_iou']:.4f} | "
                f"{(row['test_dice_at_0p5'] or 0):.4f} | {row['selected_threshold']} | {s2} |\n"
            )
    lines.append("\n## Learnable vs. fixed (test Dice)\n\n")
    lines.append("| compression | fixed | learnable | Δ | learnable wins? |\n|---|---:|---:|---:|---|\n")
    for comp_name, _ in COMPRESSIONS:
        fx = by_key[(comp_name, "random_fixed")]
        ln = by_key[(comp_name, "learnable_frequency")]
        wins = "yes" if ln["test_dice"] > fx["test_dice"] else "no"
        lines.append(f"| {comp_name} | {fx['test_dice']:.4f} | {ln['test_dice']:.4f} | {ln['test_dice'] - fx['test_dice']:+.4f} | {wins} |\n")
    lines.append("\n## Stage-3 gradient evidence (illumination receives the seg-task gradient)\n\n")
    lines.append(
        "| run | seg-head gn (stage2) | inverse gn (stage3) | illumination gn (stage3) | illum pattern Δ (L2) | rel |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    for comp_name, _ in COMPRESSIONS:
        for pattern, _ in VARIANTS:
            row = by_key[(comp_name, pattern)]
            delta = row.get("illumination_pattern_delta_l2")
            rel = row.get("illumination_pattern_delta_relative")
            delta_s = f"{delta:.3e}" if delta is not None else "n/a"
            rel_s = f"{rel:.1%}" if rel is not None else "n/a"
            lines.append(
                f"| {row['run_id']} | {(row['seg_head_grad_norm_stage2_max'] or 0):.3e} | "
                f"{(row['inverse_grad_norm_stage3_max'] or 0):.3e} | {(row['illumination_grad_norm_stage3_max'] or 0):.3e} | {delta_s} | {rel_s} |\n"
            )
    (OUT.parent / "report_tables.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fig4 task-aware segmentation isolated fix runner")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cells", nargs="*", default=None,
                        help="Optional subset: run_ids or compression labels (e.g. x64 taskaware_x256_learnable_frequency)")
    args = parser.parse_args()
    resolve_device(args.device)
    run(args.device, args.seed, args.cells)


if __name__ == "__main__":
    main()
