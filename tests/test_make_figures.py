"""Tests for results figure generation."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from evaluation.make_figures import make_figures


def test_make_figures_from_results_csv(tmp_path: Path):
    results_csv = tmp_path / "results.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "pattern_mode",
                "MSE",
                "SSIM",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {"run_id": "a", "pattern_mode": "random_fixed", "MSE": "0.01", "SSIM": "0.8"}
        )
        writer.writerow(
            {
                "run_id": "b",
                "pattern_mode": "learnable_frequency",
                "MSE": "0.005",
                "SSIM": "0.85",
            }
        )

    output_dir = tmp_path / "figures"
    outputs = make_figures(results_csv, output_dir)
    assert Path(outputs["mse_by_run"]).exists()
    assert Path(outputs["ssim_by_pattern_mode"]).exists()
