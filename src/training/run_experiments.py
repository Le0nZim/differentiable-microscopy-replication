"""Run one or more experiments from standalone configs or matrix YAML files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.train_reconstruction import train
from utils.experiment_config import expand_experiment_matrix, load_experiment_config


def run_matrix(matrix_path: str | Path, *, dry_run: bool = False, device: str | None = None) -> list[dict]:
    summaries: list[dict] = []
    for config in expand_experiment_matrix(matrix_path):
        run_id = config["experiment"]["run_id"]
        output_dir = config["experiment"]["output_dir"]
        if device is not None:
            config.setdefault("experiment", {})
            config["experiment"]["device"] = device
        if dry_run:
            summaries.append(
                {
                    "run_id": run_id,
                    "output_dir": output_dir,
                    "compression": config["experiment"].get("compression"),
                    "pattern_mode": config["pattern_generator"]["mode"],
                    "upsampling": config["inverse_model"]["upsampling"]["mode"],
                }
            )
            continue
        print(f"Starting experiment: {run_id}")
        summary = train(config, output_dir)
        summary["run_id"] = run_id
        summaries.append(summary)
        print(json.dumps(summary, indent=2))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run differentiable microscopy experiment matrix")
    parser.add_argument("--matrix", help="Path to experiment matrix YAML")
    parser.add_argument("--config", help="Path to a single experiment YAML config")
    parser.add_argument("--output-dir", help="Override output directory for single-config runs")
    parser.add_argument("--dry-run", action="store_true", help="Print expanded configs without training")
    parser.add_argument("--device", default=None, help="Device for all runs, e.g. cuda:1 or gpu1")
    args = parser.parse_args()

    if args.matrix:
        summaries = run_matrix(args.matrix, dry_run=args.dry_run, device=args.device)
        print(json.dumps(summaries, indent=2))
        return

    if args.config:
        config = load_experiment_config(args.config)
        if args.output_dir:
            config["experiment"]["output_dir"] = args.output_dir
        if args.dry_run:
            print(json.dumps(config, indent=2, default=str))
            return
        summary = train(config, config["experiment"]["output_dir"])
        print(json.dumps(summary, indent=2))
        return

    parser.error("Provide either --matrix or --config")


if __name__ == "__main__":
    main()
