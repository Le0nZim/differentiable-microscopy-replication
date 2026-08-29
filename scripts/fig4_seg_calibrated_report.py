#!/usr/bin/env python3
"""Figure 4 calibrated-preprocessing report helpers (BBBC022 proxy).

Mirrors scripts/fig4_seg_fix_report.py but points at
``experiments/figure4_bbbc022_segmentation_calibrated_v1`` and uses the calibrated
preprocessing config. Provides the same interface the SVG builder consumes
(_test_examples, _load_model, _summary, ROW_ORDER, EVAL_M, DOWNSCALE, METRICS,
fig4run, BBBC022HoechstConfig, BBBC022HoechstDataset) plus write_csv().
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_spec = importlib.util.spec_from_file_location("fig4calrun", ROOT / "scripts/fig4_seg_calibrated_train.py")
fig4run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig4run)

from datasets.bbbc022_hoechst import BBBC022HoechstConfig, BBBC022HoechstDataset  # noqa: E402
from models.task_aware_microscope import TaskAwareMicroscope  # noqa: E402
from utils.experiment_config import load_experiment_config  # noqa: E402

EXP = ROOT / "experiments/figure4_bbbc022_segmentation_calibrated_v1"
RUNS = EXP / "runs"
METRICS = EXP / "metrics"
CONFIG_PATH = fig4run.CONFIG_PATH
EVAL_M = 10.0
ROW_ORDER = [
    ("x64", "random_fixed", "C1  x64 pseudo-random"),
    ("x64", "learnable_frequency", "C2  x64 learnable"),
    ("x256", "random_fixed", "D1  x256 pseudo-random"),
    ("x256", "learnable_frequency", "D2  x256 learnable"),
    ("x1024", "random_fixed", "E1  x1024 pseudo-random"),
    ("x1024", "learnable_frequency", "E2  x1024 learnable"),
]
DOWNSCALE = {"x64": 16, "x256": 32, "x1024": 64}


def _run_dir(comp: str, pattern: str, seed: int) -> Path:
    return RUNS / f"taskaware_{comp}_{pattern}_seed{seed}"


def _summary(comp: str, pattern: str, seed: int) -> dict | None:
    p = _run_dir(comp, pattern, seed) / "metrics/run_summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def _build_config(comp: str, pattern: str, learnable: bool, seed: int, device: torch.device) -> dict:
    cfg = load_experiment_config(CONFIG_PATH)
    num_patterns = cfg["pattern_generator"]["num_patterns"]
    cfg["forward_model"]["downscale_factor"] = DOWNSCALE[comp]
    cfg["inverse_model"]["upsampling"]["downscale_factor"] = DOWNSCALE[comp]
    cfg["inverse_model"]["upsampling"]["num_patterns"] = num_patterns
    cfg["inverse_model"]["reconstruction"]["in_channels"] = num_patterns
    cfg["pattern_generator"]["mode"] = pattern
    cfg["training"]["learn_patterns"] = learnable
    cfg["experiment"]["seed"] = seed
    cfg["dataset"]["seed"] = seed
    cfg["pattern_generator"]["seed"] = seed
    cfg["experiment"]["device"] = str(device)
    return cfg


def _load_model(comp: str, pattern: str, learnable: bool, seed: int, device: torch.device) -> TaskAwareMicroscope:
    cfg = _build_config(comp, pattern, learnable, seed, device)
    model = TaskAwareMicroscope.from_run_config(cfg).to(device)
    with torch.no_grad():
        model(torch.zeros(1, 1, int(cfg["dataset"]["image_size"]), int(cfg["dataset"]["image_size"]), device=device),
              sigmoid_m=EVAL_M, apply_noise=False)
    payload = torch.load(_run_dir(comp, pattern, seed) / "checkpoints/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _test_examples(seed: int, k: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    cfg = load_experiment_config(CONFIG_PATH)
    cfg["dataset"]["return_mask"] = True
    cfg["dataset"]["seed"] = seed
    ds = BBBC022HoechstDataset(BBBC022HoechstConfig.from_dict(cfg["dataset"]), split="test")
    k = min(k, len(ds))
    imgs, masks = [], []
    for i in range(k):
        patch, mask = ds[i]
        imgs.append(patch)
        masks.append(mask)
    return imgs, masks


def write_csv(seed: int = 42) -> Path:
    METRICS.mkdir(parents=True, exist_ok=True)
    out = METRICS / "fig4_metrics.csv"
    fields = ["compression", "downscale", "compression_ratio", "illumination",
              "test_dice", "test_iou", "test_dice_at_0p5", "selected_threshold", "stage2_content_aware_val_dice"]
    rows = []
    for comp in ["x64", "x256", "x1024"]:
        for pattern in ["random_fixed", "learnable_frequency"]:
            s = _summary(comp, pattern, seed)
            if s is None:
                continue
            rows.append({
                "compression": comp,
                "downscale": DOWNSCALE[comp],
                "compression_ratio": DOWNSCALE[comp] ** 2 // 4,
                "illumination": "learnable" if pattern == "learnable_frequency" else "pseudo_random",
                "test_dice": round(s["test_dice"], 4),
                "test_iou": round(s["test_iou"], 4),
                "test_dice_at_0p5": round(s.get("test_dice_at_0p5") or 0, 4),
                "selected_threshold": s.get("selected_threshold"),
                "stage2_content_aware_val_dice": round(s["stage2_val_dice"], 4) if s.get("stage2_val_dice") is not None else "",
            })
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] wrote {out} ({len(rows)} rows)", flush=True)
    return out


if __name__ == "__main__":
    write_csv(42)
