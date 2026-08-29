#!/usr/bin/env python3
"""Focused preprocessing ablation on BBBC022 Hoechst (paper Fig. 3 + Fig. 4).

Compares four preprocessing modes (aggressive_current / minimal_percentile /
per_image_minmax_no_clip / trainset_global_percentile) on:

* **Fig. 3** content-aware reconstruction @ x16 (downscale 8, T=4)
* **Fig. 4** segmentation @ x64 (downscale 16, T=4)

Everything except the preprocessing mode is held fixed (split, architecture,
seed, optimizer, LR, batch size, steps, patch size, augmentations). The shared
well-disjoint split is loaded from ``configs/split.json``; mode D global
percentiles are fit ONCE on the train split and injected. Fig. 4 uses a single
canonical pseudo-GT (mode B) reused across all modes.

Run order (each gated):
    --phase smoke         tiny subset, all modes, both experiments (fast sanity)
    --phase fig3 --budget short|full
    --phase fig4 --budget short|full
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets.bbbc022_preproc_ablation import ALL_MODES, fit_trainset_global_percentiles  # noqa: E402
from datasets.bbbc022_split import load_split  # noqa: E402
from evaluation.metrics import mse as mse_metric  # noqa: E402
from evaluation.metrics import psnr as psnr_metric  # noqa: E402
from evaluation.metrics import ssim as ssim_metric  # noqa: E402
from models.microscope import DifferentiableMicroscope  # noqa: E402
from models.task_aware_microscope import TaskAwareMicroscope  # noqa: E402
from training.dataloaders import build_dataloader  # noqa: E402
from training.staged_hardening_train import train_staged_hardening  # noqa: E402
from training.train_reconstruction import train as train_reconstruction  # noqa: E402
from training.train_task_aware_segmentation import train_task_aware_segmentation  # noqa: E402
from utils.device import resolve_device  # noqa: E402
from utils.experiment_config import load_experiment_config, sync_derived_config_fields  # noqa: E402

OUT = ROOT / "results/preprocessing_ablation_bbbc022_hoechst"
SPLIT_PATH = OUT / "configs/split.json"
CANONICAL_MASK_MODE = "minimal_percentile"

# (illumination label, pattern mode, learn_patterns, use_staged_hardening)
ILLUM_VARIANTS = [
    ("learnable", "learnable_frequency", True, True),
    ("fixed", "random_fixed", False, False),
]

BUDGETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "subset": 8, "batch": 4,
        "recon_staged": {"inverse_warmup_steps": 30, "joint_soft_steps": 30, "harden_m_values": [2, 4, 8], "harden_steps_per_m": 10},
        "recon_fixed_steps": 60,
        "seg_stage1_fixed": 60, "seg_stage1_learn": {"inverse_warmup_steps": 30, "joint_soft_steps": 30, "harden_m_values": [2, 4, 8], "harden_steps_per_m": 10},
        "seg_head_steps": 30, "finetune_steps": 30,
    },
    "short": {
        "subset": 64, "batch": 16,
        "recon_staged": {"inverse_warmup_steps": 400, "joint_soft_steps": 800, "harden_m_values": [2, 4, 8], "harden_steps_per_m": 200},
        "recon_fixed_steps": 1200,
        "seg_stage1_fixed": 1200, "seg_stage1_learn": {"inverse_warmup_steps": 400, "joint_soft_steps": 800, "harden_m_values": [2, 4, 8], "harden_steps_per_m": 200},
        "seg_head_steps": 400, "finetune_steps": 600,
    },
    "full": {
        "subset": None, "batch": 32,
        "recon_staged": {"inverse_warmup_steps": 1500, "joint_soft_steps": 4000, "harden_m_values": [2, 4, 8], "harden_steps_per_m": 1000},
        "recon_fixed_steps": 4000,
        "seg_stage1_fixed": 4000, "seg_stage1_learn": {"inverse_warmup_steps": 1500, "joint_soft_steps": 3500, "harden_m_values": [2, 4, 8], "harden_steps_per_m": 800},
        "seg_head_steps": 1200, "finetune_steps": 2500,
    },
}


# --------------------------------------------------------------------------- #
# config builders
# --------------------------------------------------------------------------- #
def _global_percentiles() -> tuple[float, float]:
    split = load_split(SPLIT_PATH, ROOT)
    return fit_trainset_global_percentiles(split["train"], q_low=0.001, q_high=0.999, seed=42)


def _apply_dataset_block(config: dict, mode: str, *, return_mask: bool, budget: dict, globals_lo_hi: tuple[float, float]) -> None:
    lo, hi = globals_lo_hi
    ds = config["dataset"]
    ds["name"] = "bbbc022_preproc_ablation"
    ds["split_path"] = str(SPLIT_PATH)
    ds["repo_root"] = str(ROOT)
    ds["preproc_mode"] = mode
    ds["downscale_factor_aggressive"] = 1.0  # native resolution (study decision)
    ds["q_low"] = 0.001
    ds["q_high"] = 0.999
    ds["global_low"] = lo
    ds["global_high"] = hi
    ds["patch_size"] = 256
    ds["image_size"] = 256
    ds["return_mask"] = return_mask
    ds["canonical_mask_mode"] = CANONICAL_MASK_MODE
    ds["mask_threshold"] = 0.3
    ds["mask_closing_kernel"] = 10
    ds["train_random_crops"] = True
    ds["random_flips"] = True
    if budget["subset"] is not None:
        ds["max_train_samples"] = budget["subset"]
        ds["max_val_samples"] = min(budget["subset"], 21)
        ds["max_test_samples"] = min(budget["subset"], 21)


def build_fig3_config(mode: str, illum: tuple, budget: dict, globals_lo_hi: tuple[float, float], device: str) -> dict:
    _, pattern, learn, staged = illum
    config = load_experiment_config(ROOT / "configs/_shared/base_bbbc022_substitute.yaml")
    config["experiment"]["device"] = device
    config["experiment"]["run_id"] = f"fig3_{mode}_{illum[0]}"
    config["experiment"]["seed"] = 42
    config["dataset"]["seed"] = 42
    config["pattern_generator"]["seed"] = 42
    # x16: downscale 8, T=4 (base config already set; enforce explicitly).
    config["forward_model"]["downscale_factor"] = 8
    config["pattern_generator"]["num_patterns"] = 4
    config["pattern_generator"]["mode"] = pattern
    config["inverse_model"]["upsampling"]["mode"] = "locality_aware"
    config["inverse_model"]["upsampling"]["downscale_factor"] = 8
    config["inverse_model"]["upsampling"]["num_patterns"] = 4
    config["inverse_model"]["reconstruction"]["in_channels"] = 4
    config["training"]["learn_patterns"] = learn
    config["training"]["use_staged_hardening"] = staged
    config["training"]["batch_size"] = budget["batch"]
    if staged:
        config["training"].pop("max_steps", None)
        config["training"]["staged_hardening"] = budget["recon_staged"]
    else:
        config["training"]["max_steps"] = budget["recon_fixed_steps"]
        config["training"].pop("staged_hardening", None)
    _apply_dataset_block(config, mode, return_mask=False, budget=budget, globals_lo_hi=globals_lo_hi)
    return sync_derived_config_fields(config)


def build_fig4_config(mode: str, illum: tuple, budget: dict, globals_lo_hi: tuple[float, float], device: str) -> dict:
    _, pattern, learn, _ = illum
    config = load_experiment_config(ROOT / "configs/figure04_segmentation/stage1_frozen.yaml")
    config["experiment"]["device"] = device
    config["experiment"]["run_id"] = f"fig4_{mode}_{illum[0]}"
    config["experiment"]["seed"] = 42
    config["dataset"]["seed"] = 42
    config["pattern_generator"]["seed"] = 42
    # x64: downscale 16, T=4.
    config["forward_model"]["downscale_factor"] = 16
    config["pattern_generator"]["num_patterns"] = 4
    config["pattern_generator"]["mode"] = pattern
    config["inverse_model"]["upsampling"]["mode"] = "locality_aware"
    config["inverse_model"]["upsampling"]["downscale_factor"] = 16
    config["inverse_model"]["upsampling"]["num_patterns"] = 4
    config["inverse_model"]["reconstruction"]["in_channels"] = 4
    config["training"]["learn_patterns"] = learn
    config["training"]["batch_size"] = budget["batch"]
    task = config["training"].setdefault("task_aware", {})
    task["stage1_mode"] = "train"
    task["seg_head_steps"] = budget["seg_head_steps"]
    task["finetune_steps"] = budget["finetune_steps"]
    task.setdefault("stage1", {})
    task["stage1"]["fixed_steps"] = budget["seg_stage1_fixed"]
    task["stage1"]["learnable"] = budget["seg_stage1_learn"]
    _apply_dataset_block(config, mode, return_mask=True, budget=budget, globals_lo_hi=globals_lo_hi)
    return sync_derived_config_fields(config)


# --------------------------------------------------------------------------- #
# extended evaluation + panels
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extended_recon_eval(config: dict, checkpoint_path: Path, device: torch.device, *, panel_path: Path) -> dict:
    """Reload best checkpoint; compute MSE/MAE/SSIM/PSNR + foreground-weighted MSE; save panel."""
    eval_cfg = copy.deepcopy(config)
    eval_cfg["dataset"]["return_mask"] = True  # need canonical fg mask for weighting
    eval_cfg["dataset"].pop("max_test_samples", None)
    model = DifferentiableMicroscope.from_run_config(eval_cfg).to(device)
    image_size = int(eval_cfg["dataset"]["image_size"])
    model(torch.zeros(1, 1, image_size, image_size, device=device), sigmoid_m=10.0, apply_noise=False)  # lazily init PSF buffers
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    loader = build_dataloader(eval_cfg, "test")
    apply_noise = config["detector_noise"].get("apply_noise", False)
    sigmoid_m = config["training"].get("fixed_sigmoid_m")

    tot = {"mse": 0.0, "mae": 0.0, "ssim": 0.0, "psnr": 0.0, "fg_mse": 0.0, "n": 0}
    first = None
    for specimen, mask in loader:
        specimen = specimen.to(device)
        mask = mask.to(device)
        out = model(specimen, sigmoid_m=sigmoid_m, apply_noise=apply_noise)
        recon = out["x_recon"]
        tot["mse"] += float(mse_metric(recon, specimen)) * specimen.shape[0]
        tot["mae"] += float(torch.mean(torch.abs(recon - specimen))) * specimen.shape[0]
        tot["ssim"] += float(ssim_metric(recon, specimen)) * specimen.shape[0]
        tot["psnr"] += float(psnr_metric(recon, specimen)) * specimen.shape[0]
        fg = mask > 0.5
        if fg.any():
            tot["fg_mse"] += float(((recon - specimen) ** 2)[fg].mean()) * specimen.shape[0]
        tot["n"] += specimen.shape[0]
        if first is None:
            first = (specimen.cpu(), recon.cpu(), out["y_down"].cpu())

    n = max(tot["n"], 1)
    metrics = {k: tot[k] / n for k in ("mse", "mae", "ssim", "psnr", "fg_mse")}

    if first is not None:
        specimen, recon, ydown = first
        k = min(4, specimen.shape[0])
        fig, axes = plt.subplots(k, 4, figsize=(12, 3 * k))
        if k == 1:
            axes = axes.reshape(1, -1)
        cols = ["measurement y (ch0)", "target (GT)", "reconstruction", "|error|"]
        for i in range(k):
            err = (recon[i, 0] - specimen[i, 0]).abs()
            panels = [ydown[i, 0], specimen[i, 0], recon[i, 0], err]
            for j, img in enumerate(panels):
                if j == 0:  # low-res measurement: robust per-image display range
                    vmin, vmax = float(img.min()), float(img.max())
                    vmax = vmax if vmax > vmin else vmin + 1e-6
                elif j < 3:
                    vmin, vmax = 0.0, 1.0
                else:
                    vmin, vmax = 0.0, float(err.max().clamp_min(1e-6))
                axes[i, j].imshow(img.numpy(), cmap="gray" if j < 3 else "magma", vmin=vmin, vmax=vmax)
                axes[i, j].axis("off")
                if i == 0:
                    axes[i, j].set_title(cols[j], fontsize=10)
        fig.tight_layout()
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(panel_path, dpi=120)
        plt.close(fig)
    return metrics


@torch.no_grad()
def extended_seg_eval(config: dict, run_dir: Path, threshold: float, device: torch.device, *, panel_path: Path) -> dict:
    """Reload stage-3 checkpoint; compute precision/recall/F1/IoU/Dice; save overlay + FP/FN panel."""
    model = TaskAwareMicroscope.from_run_config(config).to(device)
    image_size = int(config["dataset"]["image_size"])
    model(torch.zeros(1, 1, image_size, image_size, device=device), sigmoid_m=10.0, apply_noise=False)  # lazily init buffers
    payload = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    loader = build_dataloader(config, "test")
    apply_noise = config["detector_noise"].get("apply_noise", False)
    eval_m = float(config["training"].get("task_aware", {}).get("eval_sigmoid_m", 10.0))

    tp = fp = fn = inter = union = dice_sum = nseen = 0.0
    first = None
    for specimen, mask in loader:
        specimen = specimen.to(device)
        mask = mask.to(device)
        out = model(specimen, sigmoid_m=eval_m, apply_noise=apply_noise)
        pred = (out["seg_prob"] > threshold).float()
        tp += float((pred * mask).sum())
        fp += float((pred * (1 - mask)).sum())
        fn += float(((1 - pred) * mask).sum())
        for i in range(pred.shape[0]):
            pi, mi = pred[i], mask[i]
            inter_i = float((pi * mi).sum())
            union_i = float(pi.sum() + mi.sum() - inter_i)
            inter += inter_i
            union += union_i
            dice_sum += (2 * inter_i + 1e-6) / (float(pi.sum() + mi.sum()) + 1e-6)
            nseen += 1
        if first is None:
            first = (specimen.cpu(), mask.cpu(), out["seg_prob"].cpu(), pred.cpu())

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = inter / (union + 1e-6)
    dice = dice_sum / max(nseen, 1)
    metrics = {"precision": precision, "recall": recall, "f1": f1, "iou": iou, "dice": dice, "threshold": threshold}

    if first is not None:
        specimen, mask, prob, pred = first
        k = min(4, specimen.shape[0])
        fig, axes = plt.subplots(k, 5, figsize=(15, 3 * k))
        if k == 1:
            axes = axes.reshape(1, -1)
        cols = ["input", "pseudo-GT", "pred mask", "overlay", "FP(red)/FN(blue)"]
        for i in range(k):
            gt, pr = mask[i, 0], pred[i, 0]
            overlay = torch.stack([pr, gt, torch.zeros_like(pr)], dim=-1)  # pred=R, gt=G
            fpfn = torch.zeros(gt.shape[0], gt.shape[1], 3)
            fpfn[..., 0] = ((pr == 1) & (gt == 0)).float()  # FP red
            fpfn[..., 2] = ((pr == 0) & (gt == 1)).float()  # FN blue
            imgs = [specimen[i, 0], gt, pr, overlay, fpfn]
            for j, img in enumerate(imgs):
                if img.ndim == 2:
                    axes[i, j].imshow(img.numpy(), cmap="gray", vmin=0, vmax=1)
                else:
                    axes[i, j].imshow(img.numpy())
                axes[i, j].axis("off")
                if i == 0:
                    axes[i, j].set_title(cols[j], fontsize=10)
        fig.tight_layout()
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(panel_path, dpi=120)
        plt.close(fig)
    return metrics


# --------------------------------------------------------------------------- #
# phase drivers
# --------------------------------------------------------------------------- #
def run_fig3(modes: list[str], budget_name: str, device: str) -> None:
    budget = BUDGETS[budget_name]
    dev = resolve_device(device)
    globals_lo_hi = _global_percentiles()
    exp_dir = OUT / "fig3_reconstruction" / budget_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in modes:
        for illum in ILLUM_VARIANTS:
            illum_label = illum[0]
            run_id = f"fig3_{mode}_{illum_label}"
            run_dir = exp_dir / run_id
            print(f"\n===== {run_id} (budget={budget_name}) =====", flush=True)
            config = build_fig3_config(mode, illum, budget, globals_lo_hi, device)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "used_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
            t0 = time.time()
            if illum[3]:
                summary = train_staged_hardening(config, run_dir)
            else:
                summary = train_reconstruction(config, run_dir)
            ckpt = Path(summary.get("checkpoint_path", run_dir / "checkpoints" / "best.pt"))
            ext = extended_recon_eval(config, ckpt, dev, panel_path=run_dir / "panel_recon.png")
            elapsed = time.time() - t0
            row = {"experiment": "fig3", "mode": mode, "illum": illum_label, **ext, "seconds": round(elapsed, 1), "run_dir": str(run_dir)}
            rows.append(row)
            (run_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            print(f"[{run_id}] MSE={ext['mse']:.5f} SSIM={ext['ssim']:.4f} PSNR={ext['psnr']:.2f} MAE={ext['mae']:.5f} fgMSE={ext['fg_mse']:.5f} ({elapsed:.0f}s)", flush=True)
    _write_csv(exp_dir / "results.csv", rows, ["experiment", "mode", "illum", "mse", "mae", "ssim", "psnr", "fg_mse", "seconds", "run_dir"])
    (exp_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nFig3 results -> {exp_dir / 'results.csv'}", flush=True)


def run_fig4(modes: list[str], budget_name: str, device: str) -> None:
    budget = BUDGETS[budget_name]
    dev = resolve_device(device)
    globals_lo_hi = _global_percentiles()
    exp_dir = OUT / "fig4_segmentation" / budget_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in modes:
        for illum in ILLUM_VARIANTS:
            illum_label = illum[0]
            run_id = f"fig4_{mode}_{illum_label}"
            run_dir = exp_dir / run_id
            print(f"\n===== {run_id} (budget={budget_name}) =====", flush=True)
            config = build_fig4_config(mode, illum, budget, globals_lo_hi, device)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "used_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
            t0 = time.time()
            summary = train_task_aware_segmentation(config, run_dir)
            ext = extended_seg_eval(config, run_dir, float(summary["selected_threshold"]), dev, panel_path=run_dir / "panel_seg.png")
            elapsed = time.time() - t0
            row = {
                "experiment": "fig4", "mode": mode, "illum": illum_label,
                "dice": summary["test_dice"], "iou": summary["test_iou"], "bce": summary.get("test_bce"),
                "precision": ext["precision"], "recall": ext["recall"], "f1": ext["f1"],
                "threshold": ext["threshold"], "seconds": round(elapsed, 1), "run_dir": str(run_dir),
            }
            rows.append(row)
            (run_dir / "metrics_extended.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            print(f"[{run_id}] Dice={row['dice']:.4f} IoU={row['iou']:.4f} P={ext['precision']:.4f} R={ext['recall']:.4f} F1={ext['f1']:.4f} ({elapsed:.0f}s)", flush=True)
    _write_csv(exp_dir / "results.csv", rows, ["experiment", "mode", "illum", "dice", "iou", "f1", "precision", "recall", "bce", "threshold", "seconds", "run_dir"])
    (exp_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nFig4 results -> {exp_dir / 'results.csv'}", flush=True)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="BBBC022 preprocessing ablation (Fig3 + Fig4)")
    parser.add_argument("--phase", choices=["smoke", "fig3", "fig4"], required=True)
    parser.add_argument("--budget", choices=["smoke", "short", "full"], default="short")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--modes", nargs="+", default=list(ALL_MODES))
    args = parser.parse_args()

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Missing split: {SPLIT_PATH}. Run scripts/qc_bbbc022_preprocessing.py first.")

    if args.phase == "smoke":
        run_fig3(args.modes, "smoke", args.device)
        run_fig4(args.modes, "smoke", args.device)
    elif args.phase == "fig3":
        run_fig3(args.modes, args.budget, args.device)
    elif args.phase == "fig4":
        run_fig4(args.modes, args.budget, args.device)


if __name__ == "__main__":
    main()
