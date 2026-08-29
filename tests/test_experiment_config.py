"""Tests for experiment configuration utilities."""

from __future__ import annotations

from pathlib import Path

from utils.experiment_config import (
    build_noise_sweep_experiments,
    compression_ratio,
    deep_merge,
    expand_experiment_matrix,
    load_experiment_config,
)


def test_compression_ratio_matches_spec_formula():
    assert compression_ratio(8, 8) == 8.0
    assert compression_ratio(32, 16) == 64.0


def test_deep_merge_overrides_nested_fields():
    base = {"training": {"batch_size": 32, "num_epochs": 40}, "dataset": {"image_size": 256}}
    overrides = {"training": {"num_epochs": 2}}
    merged = deep_merge(base, overrides)
    assert merged["training"]["batch_size"] == 32
    assert merged["training"]["num_epochs"] == 2
    assert merged["dataset"]["image_size"] == 256


def test_expand_debug_patchmnist_matrix():
    matrix_path = Path("configs/matrices/patchmnist_x8_matrix_debug.yaml")
    experiments = expand_experiment_matrix(matrix_path)
    assert len(experiments) == 2
    assert experiments[0]["experiment"]["run_id"].startswith("smoke_patchmnist")
    assert experiments[0]["inverse_model"]["reconstruction"]["in_channels"] == 4
    assert experiments[0]["experiment"]["compression"] == 16.0


def test_load_experiment_config_syncs_derived_fields():
    config = load_experiment_config("configs/base_patchmnist.yaml")
    assert config["pattern_generator"]["height"] == config["dataset"]["image_size"]
    assert config["inverse_model"]["reconstruction"]["in_channels"] == config["pattern_generator"]["num_patterns"]


def test_build_noise_sweep_experiments_count():
    base = load_experiment_config("configs/base_patchmnist.yaml")
    experiments = build_noise_sweep_experiments(
        base,
        photon_counts=[10.0, 10000.0],
        sigma_reads=[0.0, 6.0],
        pattern_modes=["random_fixed", "learnable_frequency"],
    )
    assert len(experiments) == 8
