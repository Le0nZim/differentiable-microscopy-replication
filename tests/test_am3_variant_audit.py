"""AM-3 machine-checkable audit tests.

Phase 1 (Table-3 variant wiring) and Phase 2 (locality-aware upsampling)
correctness, gradient flow and parameter-count checks. These tests prove the
A/B/C/D ablation is wired exactly as the paper specifies and that the proposed
locality block is implemented faithfully (so any result-level mismatch is *not*
a wiring/implementation bug).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from evaluation.variant_audit import (
    TABLE3_EXPECTED,
    audit_microscope,
    check_variant,
    describe_config,
)
from models.locality_upsampling import LocalityAwareUpsampling
from models.microscope import DifferentiableMicroscope
from utils.experiment_config import expand_experiment_matrix

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/table03_ablation/bbbc022_x16_matrix.yaml"


def _variant_letter(run_id: str) -> str:
    # run ids look like bbbc022_x16_A_random_transpose
    for letter in ("A", "B", "C", "D"):
        if f"_x16_{letter}_" in run_id:
            return letter
    raise AssertionError(f"Could not infer variant letter from {run_id}")


def _configs_by_variant() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cfg in expand_experiment_matrix(MATRIX):
        out[_variant_letter(cfg["experiment"]["run_id"])] = cfg
    return out


# --------------------------------------------------------------------------- #
# Phase 1: variant wiring                                                      #
# --------------------------------------------------------------------------- #


def test_matrix_has_all_four_variants():
    configs = _configs_by_variant()
    assert set(configs) == {"A", "B", "C", "D"}


@pytest.mark.parametrize("letter", ["A", "B", "C", "D"])
def test_variant_wiring_matches_paper(letter):
    cfg = _configs_by_variant()[letter]
    model = DifferentiableMicroscope.from_run_config(cfg)
    descriptor = audit_microscope(model, cfg)
    problems = check_variant(letter, descriptor)
    assert not problems, "; ".join(problems)


def test_all_variants_share_compression_T_and_downscale():
    for letter, cfg in _configs_by_variant().items():
        d = describe_config(cfg)
        assert d["downscale_factor"] == 8, f"{letter} downscale != 8"
        assert d["num_patterns"] == 4, f"{letter} T != 4"
        assert d["compression"] == pytest.approx(16.0), f"{letter} compression != x16"


def test_no_upsampling_leakage():
    """A,B must be transpose; C,D must be locality (no cross-contamination)."""
    configs = _configs_by_variant()
    models = {k: DifferentiableMicroscope.from_run_config(v) for k, v in configs.items()}
    audits = {k: audit_microscope(models[k], configs[k]) for k in configs}
    assert audits["A"]["actual_upsampling_module"] == "transpose_conv"
    assert audits["B"]["actual_upsampling_module"] == "transpose_conv"
    assert audits["C"]["actual_upsampling_module"] == "locality_aware"
    assert audits["D"]["actual_upsampling_module"] == "locality_aware"


def test_no_illumination_leakage():
    """A frozen (0 trainable illum params); B,C,D have trainable illumination."""
    configs = _configs_by_variant()
    for letter, cfg in configs.items():
        model = DifferentiableMicroscope.from_run_config(cfg)
        audit = audit_microscope(model, cfg)
        if letter == "A":
            assert audit["num_illumination_trainable_params"] == 0
        else:
            assert audit["num_illumination_trainable_params"] > 0


def test_frequency_domain_only_in_ABC_not_D():
    configs = _configs_by_variant()
    flags = {}
    for letter, cfg in configs.items():
        flags[letter] = describe_config(cfg)["frequency_domain_optimization"]
    # B,C learn W in Fourier space; D learns spatial tau (no freq); A is fixed.
    assert flags["B"] is True
    assert flags["C"] is True
    assert flags["D"] is False
    assert flags["A"] is False  # fixed => no W being trained


def test_same_reconstruction_backbone_across_variants():
    configs = _configs_by_variant()
    recon_param_counts = set()
    for cfg in configs.values():
        model = DifferentiableMicroscope.from_run_config(cfg)
        audit = audit_microscope(model, cfg)
        recon_param_counts.add(audit["num_reconstruction_params"])
    assert len(recon_param_counts) == 1, f"recon backbone differs across variants: {recon_param_counts}"


def test_pattern_generator_parameterization_per_variant():
    configs = _configs_by_variant()
    a = DifferentiableMicroscope.from_run_config(configs["A"]).pattern_generator
    b = DifferentiableMicroscope.from_run_config(configs["B"]).pattern_generator
    c = DifferentiableMicroscope.from_run_config(configs["C"]).pattern_generator
    d = DifferentiableMicroscope.from_run_config(configs["D"]).pattern_generator
    # A: fixed buffer, no learnable W/tau.
    assert a.W is None and a.tau is None
    # B,C: frequency W (complex), no spatial tau.
    assert b.W is not None and b.tau is None
    assert c.W is not None and c.tau is None
    # D: spatial tau, no frequency W.
    assert d.W is None and d.tau is not None


# --------------------------------------------------------------------------- #
# Phase 2: locality-aware upsampling implementation                           #
# --------------------------------------------------------------------------- #


def _table3_locality() -> LocalityAwareUpsampling:
    """The exact Table-3 detector geometry: 256x256 image, downscale 8 => 32x32."""
    return LocalityAwareUpsampling(
        num_patterns=4, height_down=32, width_down=32, downscale_factor=8
    )


def test_locality_param_count_is_per_location():
    block = _table3_locality()
    expected = 4 * 32 * 32 * 8 * 8  # T * H_down * W_down * n * n
    actual = sum(p.numel() for p in block.parameters())
    assert actual == expected == 262144
    # The weight tensor must be location-specific (one matrix per detector pixel),
    # not a single shared kernel.
    assert block.weights.shape == (4, 32, 32, 8, 8)


def test_single_detector_pixel_activates_only_its_patch():
    block = _table3_locality()
    # deterministic non-trivial weights so a change is visible
    torch.manual_seed(0)
    block.weights.data.normal_()

    y0 = torch.zeros(1, 4, 32, 32)
    out0 = block(y0)
    assert torch.count_nonzero(out0) == 0  # zero input => zero output (no bias)

    y1 = y0.clone()
    y1[0, 2, 5, 7] = 1.0  # one detector pixel, pattern channel 2, location (5,7)
    out1 = block(y1)

    diff = (out1 - out0).abs()
    changed = diff > 0
    # Only the 8x8 block at rows 40:48, cols 56:64 of channel 2 may change.
    region = torch.zeros_like(changed)
    region[0, 2, 40:48, 56:64] = True
    assert torch.equal(changed, changed & region), "activation leaked outside the (5,7) patch"
    assert changed.any(), "the corresponding patch did not change"


def test_different_locations_use_different_weights():
    block = _table3_locality()
    torch.manual_seed(1)
    block.weights.data.normal_()
    w_a = block.weights[2, 5, 7]
    w_b = block.weights[2, 6, 7]
    w_c = block.weights[2, 5, 8]
    assert not torch.allclose(w_a, w_b)
    assert not torch.allclose(w_a, w_c)


def test_gradient_nonzero_only_for_activated_location():
    block = _table3_locality()
    block.weights.data.normal_()
    y = torch.zeros(1, 4, 32, 32)
    y[0, 1, 10, 20] = 2.0
    out = block(y)
    out.sum().backward()
    grad = block.weights.grad
    assert grad is not None
    # Activated location must have nonzero grad.
    assert grad[1, 10, 20].abs().sum() > 0
    # A non-activated location (zero input there) must have zero grad.
    assert torch.count_nonzero(grad[1, 0, 0]) == 0
    assert torch.count_nonzero(grad[0, 10, 20]) == 0  # different channel


def test_gradients_flow_to_all_locations_with_dense_input():
    block = _table3_locality()
    block.weights.data.normal_()
    y = torch.rand(2, 4, 32, 32) + 0.1  # strictly positive everywhere
    out = block(y)
    out.sum().backward()
    grad = block.weights.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    # every (t,i,j) location should receive gradient (input positive everywhere)
    per_loc = grad.abs().sum(dim=(-1, -2))  # [T, H_down, W_down]
    assert (per_loc > 0).all()


def test_layout_is_consistent_for_non_square_grid():
    """Use H_down != W_down to catch any height/width transpose bug."""
    block = LocalityAwareUpsampling(num_patterns=1, height_down=2, width_down=3, downscale_factor=2)
    block.weights.data.zero_()
    # Make the patch at location (i=1, j=2) an identity-ish marker.
    block.weights.data[0, 1, 2] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.zeros(1, 1, 2, 3)
    y[0, 0, 1, 2] = 1.0
    out = block(y)  # [1,1,4,6]
    assert out.shape == (1, 1, 4, 6)
    expected_patch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    # location (1,2) -> rows 2:4, cols 4:6
    assert torch.allclose(out[0, 0, 2:4, 4:6], expected_patch)
    # everything else zero
    out[0, 0, 2:4, 4:6] = 0.0
    assert torch.count_nonzero(out) == 0
