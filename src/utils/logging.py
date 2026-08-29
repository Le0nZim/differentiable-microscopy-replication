"""Run logging and artifact saving."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from torchvision.utils import save_image


RESULTS_COLUMNS = [
    "run_id",
    "dataset",
    "pattern_mode",
    "downscale_factor",
    "num_patterns",
    "compression",
    "upsampling_type",
    "frequency_domain_optimization",
    "noise_mode",
    "photon_count",
    "sigma_read",
    "loss",
    "MSE",
    "SSIM",
    "checkpoint_path",
    "figure_path",
    "notes",
]


def ensure_run_directory(output_dir: str | Path) -> Path:
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "metrics").mkdir(exist_ok=True)
    (run_dir / "learned_patterns").mkdir(exist_ok=True)
    return run_dir


def save_run_config(config: dict[str, Any], output_dir: str | Path) -> None:
    run_dir = Path(output_dir)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def save_run_metadata(output_dir: str | Path, metadata: dict[str, Any]) -> None:
    with (Path(output_dir) / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def append_results_row(results_csv: str | Path, row: dict[str, Any]) -> None:
    path = Path(results_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in RESULTS_COLUMNS})


def save_patterns(patterns: torch.Tensor, output_dir: str | Path, prefix: str = "patterns") -> Path:
    """Save illumination patterns. patterns: [T, 1, H, W]."""
    out_dir = Path(output_dir) / "learned_patterns"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = out_dir / f"{prefix}.png"
    save_image(patterns, grid_path, nrow=max(1, int(patterns.shape[0] ** 0.5)))
    torch.save(patterns.detach().cpu(), out_dir / f"{prefix}.pt")

    binarized = (patterns > 0.5).float()
    save_image(binarized, out_dir / f"{prefix}_binarized.png", nrow=max(1, int(patterns.shape[0] ** 0.5)))
    torch.save(binarized.detach().cpu(), out_dir / f"{prefix}_binarized.pt")
    return grid_path


def save_measurement_grid(tensor: torch.Tensor, path: str | Path, nrow: int = 4) -> None:
    """Save a batch slice of measurements or reconstructions as an image grid."""
    values = tensor.detach().cpu().float()
    if values.ndim == 4 and values.shape[1] not in {1, 3}:
        # y_down: [B, T, H, W] -> visualize first batch, one channel per pattern.
        values = values[0].unsqueeze(1)
    values = values - values.min()
    max_val = values.max().clamp_min(1e-8)
    values = values / max_val
    save_image(values, path, nrow=nrow)


def copy_assumptions(output_dir: str | Path, assumptions_path: str | Path = "ASSUMPTIONS.md") -> None:
    source = Path(assumptions_path)
    if source.exists():
        shutil.copy2(source, Path(output_dir) / "ASSUMPTIONS.md")
