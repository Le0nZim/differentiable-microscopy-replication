"""AM-5 guards: PatchMNIST content-aware configs must use paper pc=10000 noise.

Proves the AM-5 fix is implementation-aligned and that the frozen noise-free
configs are left untouched. AM-5 = PatchMNIST content-aware run aligned to the
paper's photon_count=10000 (Fig. 3 / Fig. S3) using the AM-1 (RR-1 v3)
paper-aligned normalized differentiable Poisson path.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from models.microscope import DifferentiableMicroscope
from utils.config import load_yaml_config
from utils.experiment_config import expand_experiment_matrix, load_experiment_config

ROOT = Path(__file__).resolve().parents[1]

AM5_BASE_CONFIG = "configs/paper_aligned_patchmnist_contentaware_pc10000.yaml"
AM5_MATRICES = [
    "configs/matrices/am5_patchmnist_pc10000_learnable_matrix.yaml",
    "configs/matrices/am5_patchmnist_pc10000_fixed_matrix.yaml",
]

FROZEN_BASE_CONFIG = "configs/paper_aligned_patchmnist_x8.yaml"
FROZEN_LEARNABLE_MATRIX = "configs/matrices/paper_aligned_lr1_learnable_matrix.yaml"
FROZEN_FIXED_MATRIX = "configs/paper_aligned_patchmnist_fixed_baselines.yaml"

PAPER_POISSON_MODES = {"differentiable_poisson", "differentiable_poisson_plus_read"}


def _all_am5_configs() -> list[dict]:
    configs: list[dict] = []
    for matrix in AM5_MATRICES:
        configs.extend(expand_experiment_matrix(matrix))
    return configs


# --- Requirement 1: AM-5 configs do NOT use noise_free -----------------------

def test_am5_base_config_not_noise_free():
    cfg = load_experiment_config(AM5_BASE_CONFIG)
    assert cfg["detector_noise"]["mode"] != "noise_free"
    assert cfg["detector_noise"]["mode"] in PAPER_POISSON_MODES


def test_am5_matrix_configs_not_noise_free():
    configs = _all_am5_configs()
    assert len(configs) == 8, "AM-5 matrix should expand to 8 content-aware variants"
    for cfg in configs:
        assert cfg["detector_noise"]["mode"] != "noise_free", cfg["experiment"]["run_id"]
        assert cfg["detector_noise"]["mode"] in PAPER_POISSON_MODES


# --- Requirement 2: apply_noise is true --------------------------------------

def test_am5_apply_noise_true_everywhere():
    assert load_experiment_config(AM5_BASE_CONFIG)["detector_noise"]["apply_noise"] is True
    for cfg in _all_am5_configs():
        assert cfg["detector_noise"]["apply_noise"] is True, cfg["experiment"]["run_id"]


# --- Requirement 3: photon_count is exactly 10000 ----------------------------

def test_am5_photon_count_is_10000():
    assert float(load_experiment_config(AM5_BASE_CONFIG)["detector_noise"]["photon_count"]) == 10000.0
    for cfg in _all_am5_configs():
        assert float(cfg["detector_noise"]["photon_count"]) == 10000.0, cfg["experiment"]["run_id"]


def test_am5_read_noise_is_zero():
    # AM-5 is the 10000-photon content-aware setting, NOT a Table-1 read-noise experiment.
    assert float(load_experiment_config(AM5_BASE_CONFIG)["detector_noise"]["sigma_read"]) == 0.0
    for cfg in _all_am5_configs():
        assert float(cfg["detector_noise"]["sigma_read"]) == 0.0, cfg["experiment"]["run_id"]


# --- Requirement 4: AM-1 paper-aligned path (paper_v3), not legacy/v2 --------

def test_am5_uses_paper_v3_normalization_in_config():
    assert load_experiment_config(AM5_BASE_CONFIG)["detector_noise"]["noise_normalization"] == "paper_v3"
    for cfg in _all_am5_configs():
        assert cfg["detector_noise"]["noise_normalization"] == "paper_v3", cfg["experiment"]["run_id"]


def test_built_microscope_resolves_paper_v3_path():
    for cfg in _all_am5_configs():
        model = DifferentiableMicroscope.from_run_config(cfg)
        assert model.detector_noise.config.noise_normalization == "paper_v3"
        # downscale_factor must be populated for the noise model (auto-filled from forward_model).
        assert model.detector_noise.config.downscale_factor == cfg["forward_model"]["downscale_factor"]


def test_paper_v3_path_is_not_legacy_or_v2_behaviourally():
    """At sigma_read=0 the paper_v3 mean must be alpha_down + gamma/k (alpha_divisor=1).

    The superseded AM-1 v2 ("paper") path divides alpha by d², so its mean would be
    alpha_down/d² + gamma/k. This test pins paper_v3 (alpha_divisor=1) and rejects v2.
    """
    cfg = load_experiment_config(AM5_BASE_CONFIG)
    model = DifferentiableMicroscope.from_run_config(cfg)
    noise = model.detector_noise
    k = float(cfg["detector_noise"]["photon_count"])
    gamma = float(cfg["detector_noise"]["gamma"])

    alpha_down = torch.full((1, 8, 4, 4), 0.5)
    zeros = torch.zeros_like(alpha_down)
    out = noise(alpha_down, apply_noise=True, poisson_noise=zeros, read_noise=zeros)

    expected_v3_mean = alpha_down + gamma / k          # alpha_divisor = 1 (paper_v3)
    rejected_v2_mean = alpha_down / (8 ** 2) + gamma / k  # alpha_divisor = d² (v2)

    assert torch.allclose(out, expected_v3_mean, atol=1e-6)
    assert not torch.allclose(out, rejected_v2_mean, atol=1e-4)


def test_runner_noise_guard_rejects_noise_free_and_accepts_am5():
    spec = importlib.util.spec_from_file_location(
        "run_am5_patchmnist_pc10000", ROOT / "scripts" / "run_am5_patchmnist_pc10000.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Accepts a paper-aligned AM-5 config.
    module.assert_paper_aligned_noise(load_experiment_config(AM5_BASE_CONFIG))

    # Rejects a noise-free config (the old frozen setting).
    bad = load_experiment_config(FROZEN_BASE_CONFIG)
    with pytest.raises(ValueError):
        module.assert_paper_aligned_noise(bad)

    # Rejects legacy normalization even if Poisson noise is on.
    legacy = copy.deepcopy(load_experiment_config(AM5_BASE_CONFIG))
    legacy["detector_noise"]["noise_normalization"] = "legacy"
    with pytest.raises(ValueError):
        module.assert_paper_aligned_noise(legacy)


# --- Requirement 5: frozen noise-free configs are NOT modified ---------------

def test_frozen_base_config_still_noise_free():
    cfg = load_yaml_config(FROZEN_BASE_CONFIG)
    assert cfg["detector_noise"]["mode"] == "noise_free"
    assert cfg["detector_noise"]["apply_noise"] is False


def test_frozen_matrices_still_point_at_noise_free_base():
    for matrix in (FROZEN_LEARNABLE_MATRIX, FROZEN_FIXED_MATRIX):
        meta = load_yaml_config(matrix)
        assert meta["base_config"] == "paper_aligned_patchmnist_x8.yaml"
        for cfg in expand_experiment_matrix(matrix):
            assert cfg["detector_noise"]["mode"] == "noise_free"
            assert cfg["detector_noise"]["apply_noise"] is False


def test_frozen_saved_run_config_still_noise_free():
    saved = ROOT / (
        "experiments/content_aware/lr1_full/"
        "patchmnist_x8_learnable_locality_seed42/config.yaml"
    )
    if not saved.exists():
        pytest.skip("frozen saved run config not present")
    cfg = load_yaml_config(saved)
    assert cfg["detector_noise"]["mode"] == "noise_free"
    assert cfg["detector_noise"]["apply_noise"] is False


# --- AM-5 changes ONLY the noise: matrix + recipe identical to frozen --------

def test_am5_matrix_run_ids_match_frozen_matrices():
    am5_learnable = {e["experiment"]["run_id"] for e in expand_experiment_matrix(AM5_MATRICES[0])}
    frozen_learnable = {e["experiment"]["run_id"] for e in expand_experiment_matrix(FROZEN_LEARNABLE_MATRIX)}
    assert am5_learnable == frozen_learnable

    am5_fixed = {e["experiment"]["run_id"] for e in expand_experiment_matrix(AM5_MATRICES[1])}
    frozen_fixed = {e["experiment"]["run_id"] for e in expand_experiment_matrix(FROZEN_FIXED_MATRIX)}
    assert am5_fixed == frozen_fixed


def test_am5_base_identical_to_frozen_except_noise_and_results_csv():
    am5 = load_experiment_config(AM5_BASE_CONFIG)
    frozen = load_experiment_config(FROZEN_BASE_CONFIG)
    for cfg in (am5, frozen):
        cfg.pop("detector_noise", None)
        cfg["experiment"].pop("results_csv", None)
    assert am5 == frozen, "AM-5 base must match the frozen base except detector_noise + results_csv"
