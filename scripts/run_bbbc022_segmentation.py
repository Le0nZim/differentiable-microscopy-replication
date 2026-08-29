#!/usr/bin/env python3
"""BBBC022 substitute segmentation-aware compression (pseudo-mask QC + training)."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets.bbbc022_hoechst import BBBC022HoechstConfig, BBBC022HoechstDataset
from models.microscope import DifferentiableMicroscope
from training.staged_hardening_train import train_staged_hardening
from training.train_reconstruction import train
from utils.device import device_from_config, resolve_device
from utils.experiment_config import load_experiment_config
from utils.reproducibility import set_seed

OUT = ROOT / "experiments/task_aware_segmentation/bbbc022_segmentation"
BBBC022_META = ROOT / "experiments/ablations"
COMPRESSIONS = [("x64", 16, 4), ("x256", 32, 4), ("x1024", 64, 4)]


def mask_qc() -> dict:
    qc_dir = OUT / "mask_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    config = BBBC022HoechstConfig(
        data_root="data/substitute_data",
        preprocessing_mode=json.loads((BBBC022_META / "preprocessing_report.json").read_text())["chosen_official_mode"]
        if (BBBC022_META / "preprocessing_report.json").exists()
        else "bbbc022_calibrated",
        return_mask=True,
    )
    ds = BBBC022HoechstDataset(config, "val")
    stats = []
    for i in range(min(21, len(ds))):
        img, mask = ds[i]
        frac = float(mask.mean())
        stats.append(frac)
    summary = {
        "threshold": 0.3,
        "closing_kernel": 10,
        "mask_foreground_fraction_mean": sum(stats) / len(stats) if stats else 0,
        "mask_foreground_fraction_min": min(stats) if stats else 0,
        "degenerate": any(f < 0.001 or f > 0.99 for f in stats),
        "note": "Paper-mimic threshold 0.3; BBBC022 Hoechst nuclei may differ from paper DAPI",
    }
    (qc_dir / "mask_qc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    return float((2 * inter + eps) / (pred.sum() + target.sum() + eps))


def _iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return float((inter + eps) / (union + eps))


def _load_trained_model(
    out_dir: Path,
    fallback_config: dict,
) -> tuple[DifferentiableMicroscope, dict, torch.device]:
    ckpt_path = out_dir / "checkpoints/best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_config_path = out_dir / "config.yaml"
    if saved_config_path.exists():
        eval_config = load_experiment_config(saved_config_path)
    else:
        eval_config = ckpt.get("config") or fallback_config
    eval_device = device_from_config(eval_config)
    model = DifferentiableMicroscope.from_run_config(eval_config).to(eval_device)
    image_size = int(eval_config["dataset"]["image_size"])
    with torch.no_grad():
        dummy = torch.zeros(1, 1, image_size, image_size, device=eval_device)
        model(dummy, sigmoid_m=10.0, apply_noise=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, eval_config, eval_device


def run_segmentation_experiments(device: str, seed: int) -> list[dict]:
    qc = mask_qc()
    if qc.get("degenerate"):
        (OUT / "SEGMENTATION_BLOCKED.md").write_text(
            "# Segmentation blocked\n\nThreshold 0.3 produced degenerate masks on BBBC022 substitute data.\n",
            encoding="utf-8",
        )
        return []
    results = []
    for comp_name, d, t in COMPRESSIONS:
        for pattern, learnable, staged in [
            ("random_fixed", False, False),
            ("learnable_frequency", True, True),
        ]:
            config = load_experiment_config(ROOT / "configs/base_bbbc022_substitute.yaml")
            if (BBBC022_META / "preprocessing_report.json").exists():
                config["dataset"]["preprocessing_mode"] = json.loads(
                    (BBBC022_META / "preprocessing_report.json").read_text()
                )["chosen_official_mode"]
            config["dataset"]["return_mask"] = False
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
            config["training"]["learn_patterns"] = learnable
            config["training"]["use_staged_hardening"] = staged
            config["training"]["segmentation_bce_weight"] = 1.0
            run_id = f"seg_{comp_name}_{pattern}"
            config["experiment"]["run_id"] = run_id
            out_dir = OUT / f"{run_id}_seed{seed}"
            # Use standard reconstruction training; evaluate mask overlap post-hoc
            if not (out_dir / "metrics/run_summary.json").exists():
                if staged:
                    train_staged_hardening(config, str(out_dir))
                else:
                    train(config, str(out_dir))
            # Post-hoc mask metrics from reconstruction threshold
            set_seed(seed)
            model, eval_config, eval_device = _load_trained_model(out_dir, config)
            test_ds = BBBC022HoechstDataset(
                BBBC022HoechstConfig.from_dict({**eval_config["dataset"], "return_mask": True}),
                "test",
            )
            loader = DataLoader(test_ds, batch_size=4, shuffle=False)
            dices, ious = [], []
            with torch.no_grad():
                for batch in loader:
                    img, mask = batch
                    img = img.to(eval_device)
                    mask = mask.to(eval_device)
                    out = model(img, sigmoid_m=10.0, apply_noise=False)
                    pred_mask = (out["x_recon"] > 0.3).float()
                    for b in range(img.shape[0]):
                        dices.append(_dice(pred_mask[b], mask[b]))
                        ious.append(_iou(pred_mask[b], mask[b]))
            row = {
                "run_id": run_id,
                "compression": comp_name,
                "pattern": pattern,
                "dice_mean": sum(dices) / len(dices),
                "iou_mean": sum(ious) / len(ious),
                "seed": seed,
            }
            results.append(row)
    (OUT / "aggregate_summary.json").write_text(json.dumps({"results": results, "qc": qc}, indent=2), encoding="utf-8")
    (OUT / "figure4_style_report.md").write_text(
        "# Fig. 4-style segmentation report (BBBC022 substitute)\n\n"
        "Post-hoc mask overlap from reconstruction threshold 0.3. Not comparable to paper Fig. 4 numerically.\n",
        encoding="utf-8",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    resolve_device(args.device)
    run_segmentation_experiments(args.device, args.seed)


if __name__ == "__main__":
    main()
