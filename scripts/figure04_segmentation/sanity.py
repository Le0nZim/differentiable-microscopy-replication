#!/usr/bin/env python3
"""Sanity-gate suite for the Figure 4 task-aware segmentation fix.

Runs a set of cheap, inspectable checks that must pass BEFORE the (short)
verified re-run of the task-aware matrix, mirroring the discipline used for the
Figure 3 SwinIR fix. Gates:

  1. mask_correctness  - TrackMate-style pseudo-GT mask reflects the RAW-intensity
     threshold (raw MIP > mask_raw_threshold): mask is binary, foreground fraction
     is non-degenerate, and IoU vs. the plain raw>threshold detection is high
     (contour smoothing / hole-fill cause only small deviations).
  2. no_leakage_split  - train/val/test wells are disjoint (no image leakage).
  3. distribution      - GT images normalized to [0,1]; mask fg fraction per split.
  4. mask_not_in_input - the segmentation forward consumes ONLY the specimen;
     the mask is never fed to the model (structural + perturbation check).
  5. degenerate_baselines - Dice/IoU for all-zeros / all-ones / perfect masks,
     plus the historical post-hoc (x_recon>0.3) baseline on a frozen Stage-1
     content-aware microscope (sanity vs. the old diagnostic numbers).
  6. tiny_overfit      - a frozen Stage-1 microscope + trainable seg head can
     overfit a handful of samples (train Dice -> high), proving the
     head/loss/pairing pipeline learns.

Outputs: sanity/sanity_results.json, sanity/SANITY_REPORT.md,
sanity/mask_examples.png. Read-only w.r.t. all frozen runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets.bbbc022_hoechst import (  # noqa: E402
    BBBC022HoechstConfig,
    BBBC022HoechstDataset,
    make_trackmate_mask,
    parse_well_site,
)
from models.task_aware_microscope import TaskAwareMicroscope  # noqa: E402
from training.dataloaders import build_dataset  # noqa: E402
from training.segmentation_losses import (  # noqa: E402
    TaskAwareLossWeights,
    task_aware_segmentation_loss,
)
from training.train_task_aware_segmentation import (  # noqa: E402
    _dice,
    _iou,
    _load_microscope_checkpoint,
)
from utils.device import resolve_device  # noqa: E402
from utils.experiment_config import load_experiment_config  # noqa: E402

CONFIG_PATH = ROOT / "configs/figure04_segmentation/task_aware.yaml"
OUT_DIR = ROOT / "experiments/figure04_segmentation/task_aware/sanity"
# Reuse the frozen x64-learnable Stage-1 content-aware checkpoint (downscale 16).
STAGE1_CKPT = (
    ROOT
    / "experiments/figure04_segmentation/stage1_frozen"
    / "taskaware_x64_learnable_frequency_seed42/stage1_content_aware/checkpoints/best.pt"
)


def _cfg() -> dict:
    return load_experiment_config(CONFIG_PATH)


def _dataset(split: str, *, return_mask: bool = True) -> BBBC022HoechstDataset:
    cfg = _cfg()
    cfg["dataset"]["return_mask"] = return_mask
    ds_cfg = BBBC022HoechstConfig.from_dict(cfg["dataset"])
    return BBBC022HoechstDataset(ds_cfg, split=split)


# --------------------------------------------------------------------------- #
def gate_mask_correctness() -> dict:
    """TrackMate pseudo-GT reflects the RAW-intensity threshold detection.

    The dataset builds masks by thresholding the raw MIP intensity at
    ``mask_raw_threshold`` (>), 4-connecting, and simplifying/filling each region
    contour. We independently recompute the plain ``raw > threshold`` detection on
    the SAME (deterministic center) test crop and require a high IoU: the
    contour smoothing (~2 px) + Douglas-Peucker (0.5) + hole-fill only perturb
    the boundary, so IoU should be high but < 1. Also require binary + a
    non-degenerate foreground fraction.
    """
    cfg = _cfg()["dataset"]
    thr = float(cfg.get("mask_raw_threshold", 506.0))
    ps = int(cfg["patch_size"])
    ds = _dataset("test", return_mask=True)  # test => deterministic center crops
    ok_binary, fg_fractions, ious = True, [], []
    n_check = min(8, len(ds))
    for idx in range(n_check):
        _patch, mask = ds[idx]
        m = mask[0]
        uniq = torch.unique(m)
        if not torch.all((uniq == 0) | (uniq == 1)):
            ok_binary = False
        fg_fractions.append(float(m.mean().item()))
        raw_full = ds._load_raw_mip(ds.paths[idx])  # [H, W], raw MIP (no norm)
        height, width = raw_full.shape
        top = max(0, (height - ps) // 2)
        left = max(0, (width - ps) // 2)
        raw_crop = raw_full[top : top + ps, left : left + ps]
        thr_mask = (raw_crop > thr).float()
        fg = m > 0.5
        tm = thr_mask > 0.5
        inter = float((fg & tm).sum())
        union = float((fg | tm).sum())
        ious.append(inter / union if union > 0 else 1.0)
    mean_fg = sum(fg_fractions) / len(fg_fractions)
    mean_iou = sum(ious) / len(ious)
    non_degenerate = 0.01 < mean_fg < 0.95
    reflects_threshold = mean_iou > 0.7
    return {
        "passed": bool(ok_binary and non_degenerate and reflects_threshold),
        "mask_mode": cfg.get("mask_mode", "trackmate"),
        "mask_is_binary": ok_binary,
        "mean_foreground_fraction": round(mean_fg, 4),
        "foreground_non_degenerate": non_degenerate,
        "mask_reflects_raw_threshold_iou": round(mean_iou, 4),
        "iou_gate": 0.7,
        "raw_threshold": thr,
        "smooth_interval": float(cfg.get("mask_smooth_interval", 2.0)),
        "dp_epsilon": float(cfg.get("mask_dp_epsilon", 0.5)),
        "num_checked": n_check,
    }


def gate_no_leakage_split() -> dict:
    cfg = _cfg()
    cfg["dataset"]["return_mask"] = False  # only paths needed; skip mask precompute
    ds_cfg = BBBC022HoechstConfig.from_dict(cfg["dataset"])
    splits = {}
    for split in ("train", "val", "test"):
        ds = BBBC022HoechstDataset(ds_cfg, split=split)
        wells = {(parse_well_site(p)[0] or p.stem) for p in ds.paths}
        splits[split] = wells
    tr, va, te = splits["train"], splits["val"], splits["test"]
    overlaps = {
        "train_val": sorted(tr & va),
        "train_test": sorted(tr & te),
        "val_test": sorted(va & te),
    }
    disjoint = all(len(v) == 0 for v in overlaps.values())
    return {
        "passed": bool(disjoint),
        "n_train_wells": len(tr),
        "n_val_wells": len(va),
        "n_test_wells": len(te),
        "overlaps": overlaps,
        "split_by_well": bool(cfg["dataset"].get("split_by_well", False)),
    }


def gate_distribution() -> dict:
    stats = {}
    ok = True
    for split in ("train", "val", "test"):
        ds = _dataset(split, return_mask=True)
        imgs, fgs = [], []
        for idx in range(len(ds)):
            patch, mask = ds[idx]
            imgs.append(patch)
            fgs.append(float(mask.mean().item()))
        stacked = torch.stack(imgs)
        lo, hi = float(stacked.min()), float(stacked.max())
        if lo < -1e-4 or hi > 1.0 + 1e-4:
            ok = False
        stats[split] = {
            "n": len(ds),
            "img_min": round(lo, 4),
            "img_max": round(hi, 4),
            "img_mean": round(float(stacked.mean()), 4),
            "mask_fg_mean": round(sum(fgs) / len(fgs), 4),
        }
    return {"passed": bool(ok), "per_split": stats}


def gate_mask_not_in_input(device: torch.device) -> dict:
    """The seg forward must depend on the specimen only, never the mask."""
    cfg = _cfg()
    model = TaskAwareMicroscope.from_run_config(cfg).to(device)
    _load_microscope_checkpoint(model, STAGE1_CKPT, device, int(cfg["dataset"]["image_size"]))
    model.eval()
    ds = _dataset("test", return_mask=True)
    patch, mask = ds[0]
    x = patch.unsqueeze(0).to(device)
    with torch.no_grad():
        out_a = model(x, sigmoid_m=10.0, apply_noise=False)["seg_logits"]
        out_b = model(x, sigmoid_m=10.0, apply_noise=False)["seg_logits"]
    forward_takes_mask = "mask" in TaskAwareMicroscope.forward.__code__.co_varnames
    deterministic = bool(torch.allclose(out_a, out_b, atol=1e-6))
    # Perturbing the *specimen* must change the seg output (it actually uses input).
    with torch.no_grad():
        out_c = model(x + 0.25, sigmoid_m=10.0, apply_noise=False)["seg_logits"]
    responds_to_input = not bool(torch.allclose(out_a, out_c, atol=1e-4))
    return {
        "passed": bool((not forward_takes_mask) and deterministic and responds_to_input),
        "forward_signature_excludes_mask": not forward_takes_mask,
        "forward_deterministic_wo_mask": deterministic,
        "seg_responds_to_specimen_perturbation": responds_to_input,
    }


def gate_degenerate_baselines(device: torch.device) -> dict:
    cfg = _cfg()
    ds = _dataset("test", return_mask=True)
    masks = torch.stack([ds[i][1] for i in range(len(ds))])  # [N,1,H,W]
    zeros = torch.zeros_like(masks)
    ones = torch.ones_like(masks)
    dice_zero = sum(_dice(zeros[i], masks[i]) for i in range(len(ds))) / len(ds)
    dice_one = sum(_dice(ones[i], masks[i]) for i in range(len(ds))) / len(ds)
    dice_perfect = sum(_dice(masks[i], masks[i]) for i in range(len(ds))) / len(ds)

    # Historical post-hoc baseline: (x_recon > 0.3) from the frozen Stage-1 model.
    model = TaskAwareMicroscope.from_run_config(cfg).to(device)
    _load_microscope_checkpoint(model, STAGE1_CKPT, device, int(cfg["dataset"]["image_size"]))
    model.eval()
    posthoc_dices = []
    with torch.no_grad():
        for i in range(len(ds)):
            patch, mask = ds[i]
            recon = model.microscope(patch.unsqueeze(0).to(device), sigmoid_m=10.0, apply_noise=False)["x_recon"]
            pred = (recon[0].cpu() > 0.3).float()
            posthoc_dices.append(_dice(pred, mask))
    posthoc = sum(posthoc_dices) / len(posthoc_dices)

    checks = {
        "dice_all_zeros_low": dice_zero < 0.1,
        "dice_all_ones_reasonable": 0.05 < dice_one < 0.9,
        "dice_perfect_is_one": abs(dice_perfect - 1.0) < 1e-3,
        "posthoc_plausible": 0.3 < posthoc < 1.0,
    }
    return {
        "passed": bool(all(checks.values())),
        "dice_all_zeros": round(dice_zero, 4),
        "dice_all_ones": round(dice_one, 4),
        "dice_perfect": round(dice_perfect, 4),
        "posthoc_recon_gt_0p3_dice_x64_learnable": round(posthoc, 4),
        "checks": checks,
    }


def gate_tiny_overfit(device: torch.device, steps: int) -> dict:
    cfg = _cfg()
    model = TaskAwareMicroscope.from_run_config(cfg).to(device)
    _load_microscope_checkpoint(model, STAGE1_CKPT, device, int(cfg["dataset"]["image_size"]))
    # Freeze microscope; only the seg head learns (Stage-2-like, tiny subset).
    model.set_microscope_trainable(False)
    model.set_segmentation_trainable(True)
    model.microscope.eval()
    model.segmentation_head.train()

    ds = _dataset("train", return_mask=True)
    n = min(4, len(ds))
    xs = torch.stack([ds[i][0] for i in range(n)]).to(device)
    ys = torch.stack([ds[i][1] for i in range(n)]).to(device)

    weights = TaskAwareLossWeights(seg_bce_weight=1.0, seg_dice_weight=0.5, reconstruction_l1_weight=0.0)
    opt = torch.optim.Adam(model.segmentation_parameters(), lr=1e-2)
    best_dice, curve = 0.0, []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        out = model(xs, sigmoid_m=10.0, apply_noise=False)
        loss, comps = task_aware_segmentation_loss(out, ys, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.segmentation_parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            pred = (out["seg_prob"] > 0.5).float()
            d = sum(_dice(pred[i], ys[i]) for i in range(n)) / n
        best_dice = max(best_dice, d)
        if step == 1 or step % 50 == 0 or step == steps:
            curve.append({"step": step, "loss": round(float(comps["total"]), 5), "train_dice": round(d, 4)})
    return {
        "passed": bool(best_dice > 0.90),
        "best_train_dice": round(best_dice, 4),
        "num_samples": n,
        "steps": steps,
        "curve": curve,
    }


def save_mask_examples() -> str:
    cfg = _cfg()["dataset"]
    thr = float(cfg.get("mask_raw_threshold", 506.0))
    ps = int(cfg["patch_size"])
    ds = _dataset("test", return_mask=True)
    n = min(4, len(ds))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i in range(n):
        patch, mask = ds[i]
        raw_full = ds._load_raw_mip(ds.paths[i])
        height, width = raw_full.shape
        top = max(0, (height - ps) // 2)
        left = max(0, (width - ps) // 2)
        raw_crop = raw_full[top : top + ps, left : left + ps].numpy()
        thr_mask = (raw_crop > thr).astype("float32")
        axes[i, 0].imshow(patch[0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title("GT image (normalized)" if i == 0 else "")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(thr_mask, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(f"raw MIP > {thr:g}" if i == 0 else "")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(mask[0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title("TrackMate pseudo-GT (4-conn, DP 0.5)" if i == 0 else "")
        axes[i, 2].axis("off")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "mask_examples.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Figure 4 task-aware segmentation sanity gates")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overfit-steps", type=int, default=300)
    args = ap.parse_args()
    device = resolve_device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[sanity] device={device}  stage1_ckpt={STAGE1_CKPT.name}", flush=True)

    results = {}
    results["mask_correctness"] = gate_mask_correctness()
    print("[sanity] mask_correctness:", results["mask_correctness"]["passed"], flush=True)
    results["no_leakage_split"] = gate_no_leakage_split()
    print("[sanity] no_leakage_split:", results["no_leakage_split"]["passed"], flush=True)
    results["distribution"] = gate_distribution()
    print("[sanity] distribution:", results["distribution"]["passed"], flush=True)
    results["mask_not_in_input"] = gate_mask_not_in_input(device)
    print("[sanity] mask_not_in_input:", results["mask_not_in_input"]["passed"], flush=True)
    results["degenerate_baselines"] = gate_degenerate_baselines(device)
    print("[sanity] degenerate_baselines:", results["degenerate_baselines"]["passed"], flush=True)
    results["tiny_overfit"] = gate_tiny_overfit(device, args.overfit_steps)
    print("[sanity] tiny_overfit:", results["tiny_overfit"]["passed"],
          "best_train_dice=", results["tiny_overfit"]["best_train_dice"], flush=True)

    mask_png = save_mask_examples()
    all_pass = all(g["passed"] for g in results.values())
    payload = {"all_passed": bool(all_pass), "gates": results, "mask_examples_png": mask_png}
    (OUT_DIR / "sanity_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Markdown summary
    lines = ["# Figure 4 task-aware segmentation — sanity gates\n\n"]
    lines.append(f"**Overall: {'PASS ✅' if all_pass else 'FAIL ❌'}**\n\n")
    lines.append("| gate | passed | key numbers |\n|---|:--:|---|\n")
    key = {
        "mask_correctness": lambda g: f"{g['mask_mode']} raw>{g['raw_threshold']:g}: fg={g['mean_foreground_fraction']}, binary={g['mask_is_binary']}, IoU_vs_raw_thr={g['mask_reflects_raw_threshold_iou']}",
        "no_leakage_split": lambda g: f"wells {g['n_train_wells']}/{g['n_val_wells']}/{g['n_test_wells']}, overlaps={sum(len(v) for v in g['overlaps'].values())}",
        "distribution": lambda g: ", ".join(f"{s}:[{v['img_min']},{v['img_max']}] fg={v['mask_fg_mean']}" for s, v in g["per_split"].items()),
        "mask_not_in_input": lambda g: f"forward_excludes_mask={g['forward_signature_excludes_mask']}, responds_to_input={g['seg_responds_to_specimen_perturbation']}",
        "degenerate_baselines": lambda g: f"zeros={g['dice_all_zeros']}, ones={g['dice_all_ones']}, perfect={g['dice_perfect']}, posthoc={g['posthoc_recon_gt_0p3_dice_x64_learnable']}",
        "tiny_overfit": lambda g: f"best_train_dice={g['best_train_dice']} ({g['num_samples']} samples, {g['steps']} steps)",
    }
    for name, g in results.items():
        lines.append(f"| {name} | {'✅' if g['passed'] else '❌'} | {key[name](g)} |\n")
    lines.append(f"\nMask examples: `{Path(mask_png).name}`\n")
    (OUT_DIR / "SANITY_REPORT.md").write_text("".join(lines), encoding="utf-8")

    print(f"\n[sanity] ALL_PASSED={all_pass}", flush=True)
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
