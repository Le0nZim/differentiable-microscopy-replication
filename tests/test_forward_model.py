"""Tests for the optical forward model."""

from __future__ import annotations

import pytest
import torch

from models.forward_model import ForwardModel, ForwardModelConfig, sum_pool_nxn
from utils.config import load_yaml_config


@pytest.fixture
def yaml_config():
    return load_yaml_config("configs/_shared/base_patchmnist.yaml")


def test_sum_pool_equals_block_sum():
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    pooled = sum_pool_nxn(x, kernel_size=2)

    block_00 = x[0, 0, 0:2, 0:2].sum()
    block_01 = x[0, 0, 0:2, 2:4].sum()
    block_10 = x[0, 0, 2:4, 0:2].sum()
    block_11 = x[0, 0, 2:4, 2:4].sum()

    expected = torch.tensor([[block_00, block_01], [block_10, block_11]])
    assert torch.allclose(pooled[0, 0], expected)


def test_forward_model_output_shape():
    batch_size, num_patterns = 2, 4
    height, width, down = 32, 32, 8

    forward = ForwardModel(ForwardModelConfig(downscale_factor=down))
    specimen = torch.rand(batch_size, 1, height, width)
    patterns = torch.rand(num_patterns, 1, height, width)

    alpha_down = forward(specimen, patterns)
    assert alpha_down.shape == (batch_size, num_patterns, height // down, width // down)


def test_forward_model_is_deterministic_with_impulse_psfs():
    forward = ForwardModel(ForwardModelConfig(downscale_factor=4, use_impulse_psfs=True))
    specimen = torch.rand(1, 1, 16, 16)
    patterns = torch.rand(3, 1, 16, 16)

    out_a = forward(specimen, patterns)
    out_b = forward(specimen, patterns)
    assert torch.allclose(out_a, out_b)


def test_impulse_forward_equals_pattern_weighted_sum_pool():
    down = 4
    forward = ForwardModel(ForwardModelConfig(downscale_factor=down, use_impulse_psfs=True))

    specimen = torch.ones(1, 1, 8, 8)
    patterns = torch.zeros(1, 1, 8, 8)
    patterns[0, 0, 0:4, 0:4] = 1.0

    alpha_down = forward(specimen, patterns)
    expected = torch.zeros(1, 1, 2, 2)
    expected[0, 0, 0, 0] = 16.0
    assert torch.allclose(alpha_down, expected)


def test_forward_model_from_yaml(yaml_config):
    forward = ForwardModel.from_dict(yaml_config["forward_model"])
    down = yaml_config["forward_model"]["downscale_factor"]
    specimen = torch.rand(1, 1, 64, 64)
    patterns = torch.rand(8, 1, 64, 64)
    alpha_down = forward(specimen, patterns)
    assert alpha_down.shape == (1, 8, 64 // down, 64 // down)
