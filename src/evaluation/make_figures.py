"""Generate summary figures from experiment results.csv files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(results_csv: str | Path) -> list[dict[str, str]]:
    path = Path(results_csv)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: str) -> float:
    return float(value) if value not in ("", None) else float("nan")


def plot_metric_by_run(
    rows: list[dict[str, str]],
    metric: str,
    output_path: str | Path,
    *,
    group_key: str = "run_id",
) -> Path:
    labels = [row[group_key] for row in rows]
    values = [_to_float(row[metric]) for row in rows]

    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))
    axis.bar(labels, values)
    axis.set_ylabel(metric)
    axis.set_title(f"{metric} by experiment")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_metric_by_pattern_mode(
    rows: list[dict[str, str]],
    metric: str,
    output_path: str | Path,
) -> Path:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["pattern_mode"]].append(_to_float(row[metric]))

    labels = sorted(grouped)
    values = [sum(grouped[label]) / len(grouped[label]) for label in labels]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(labels, values)
    axis.set_ylabel(f"mean {metric}")
    axis.set_title(f"Mean {metric} by pattern mode")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def make_figures(results_csv: str | Path, output_dir: str | Path) -> dict[str, str]:
    rows = load_results(results_csv)
    if not rows:
        raise ValueError(f"No rows found in {results_csv}")

    output_dir = Path(output_dir)
    outputs = {
        "mse_by_run": str(plot_metric_by_run(rows, "MSE", output_dir / "mse_by_run.png")),
        "ssim_by_run": str(plot_metric_by_run(rows, "SSIM", output_dir / "ssim_by_run.png")),
        "mse_by_pattern_mode": str(
            plot_metric_by_pattern_mode(rows, "MSE", output_dir / "mse_by_pattern_mode.png")
        ),
        "ssim_by_pattern_mode": str(
            plot_metric_by_pattern_mode(rows, "SSIM", output_dir / "ssim_by_pattern_mode.png")
        ),
    }
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create summary plots from results.csv")
    parser.add_argument("--results-csv", default="experiments/content_aware/results.csv")
    parser.add_argument("--output-dir", default="results/figures/summary")
    args = parser.parse_args()

    outputs = make_figures(args.results_csv, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
