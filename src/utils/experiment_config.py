"""Experiment configuration helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from utils.config import load_yaml_config
from utils.config_normalize import normalize_run_config


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override values into a base config dictionary."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def compression_ratio(downscale_factor: int, num_patterns: int) -> float:
    """Compute compression = downscale_factor^2 / number_of_patterns."""
    if num_patterns <= 0:
        raise ValueError("num_patterns must be positive")
    return float(downscale_factor**2) / float(num_patterns)


def sync_derived_config_fields(config: dict[str, Any]) -> dict[str, Any]:
    """Keep dependent fields aligned across the experiment config."""
    config = copy.deepcopy(config)
    image_size = config["dataset"]["image_size"]
    downscale = config["forward_model"]["downscale_factor"]
    num_patterns = config["pattern_generator"]["num_patterns"]

    config["pattern_generator"]["height"] = image_size
    config["pattern_generator"]["width"] = image_size

    upsampling = config["inverse_model"]["upsampling"]
    upsampling["downscale_factor"] = downscale
    upsampling["num_patterns"] = num_patterns

    reconstruction = config["inverse_model"]["reconstruction"]
    reconstruction["in_channels"] = num_patterns

    if "compression" not in config.get("experiment", {}):
        config.setdefault("experiment", {})
    config["experiment"]["compression"] = compression_ratio(downscale, num_patterns)
    return config


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load a standalone experiment config with derived fields synchronized."""
    config = normalize_run_config(load_yaml_config(path))
    return sync_derived_config_fields(config)


def _resolve_config_path(matrix_path: Path, relative: str) -> Path:
    candidates = [
        matrix_path.parent / relative,
        matrix_path.parent.parent / relative,
        Path(relative),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve config path {relative!r} from {matrix_path}")


def expand_experiment_matrix(matrix_path: str | Path) -> list[dict[str, Any]]:
    """Expand a matrix YAML file into a list of runnable experiment configs."""
    matrix_file = Path(matrix_path)
    matrix = load_yaml_config(matrix_file)
    base_config = load_yaml_config(_resolve_config_path(matrix_file, matrix["base_config"]))

    results_csv = matrix.get("results_csv", "experiments/content_aware/results.csv")
    experiments: list[dict[str, Any]] = []

    for entry in matrix["experiments"]:
        run_id = entry["run_id"]
        overrides = entry.get("overrides", {})
        config = normalize_run_config(deep_merge(base_config, overrides))
        config.setdefault("experiment", {})
        config["experiment"]["run_id"] = run_id
        config["experiment"]["output_dir"] = entry.get("output_dir", f"experiments/{run_id}")
        config["experiment"]["results_csv"] = results_csv
        if "notes" in entry:
            config["experiment"]["notes"] = entry["notes"]
        experiments.append(sync_derived_config_fields(config))

    return experiments


def build_noise_sweep_experiments(
    base_config: dict[str, Any],
    *,
    photon_counts: list[float],
    sigma_reads: list[float],
    pattern_modes: list[str],
    results_csv: str = "experiments/noise_robustness/noise_table/results.csv",
) -> list[dict[str, Any]]:
    """Build a full noise-robustness experiment cartesian product."""
    experiments: list[dict[str, Any]] = []
    for pattern_mode in pattern_modes:
        for photon_count in photon_counts:
            for sigma_read in sigma_reads:
                run_id = (
                    f"patchmnist_noise_{pattern_mode}_pc{int(photon_count)}_sr{sigma_read}"
                ).replace(".", "p")
                overrides = {
                    "pattern_generator": {"mode": pattern_mode},
                    "detector_noise": {
                        "mode": "differentiable_poisson_plus_read",
                        "photon_count": photon_count,
                        "sigma_read": sigma_read,
                        "apply_noise": True,
                    },
                }
                config = deep_merge(base_config, overrides)
                config.setdefault("experiment", {})
                config["experiment"]["run_id"] = run_id
                config["experiment"]["output_dir"] = f"experiments/{run_id}"
                config["experiment"]["results_csv"] = results_csv
                config["experiment"]["notes"] = "noise_robustness"
                experiments.append(sync_derived_config_fields(config))
    return experiments
