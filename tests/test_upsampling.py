"""Tests for locality-aware and transpose-convolution upsampling."""

from __future__ import annotations

import pytest
import torch

from models.locality_upsampling import (
    LocalityAwareUpsampling,
    LocalityUpsampling,
    LocalityUpsamplingConfig,
    TransposeConvUpsampling,
)
from utils.experiment_config import load_experiment_config


@pytest.fixture
def yaml_config():
    return load_experiment_config("configs/_shared/base_patchmnist.yaml")


def test_locality_aware_output_shape():
    batch_size, num_patterns = 2, 4
    height_down, width_down, down = 8, 8, 4

    upsampler = LocalityAwareUpsampling(
        num_patterns=num_patterns,
        height_down=height_down,
        width_down=width_down,
        downscale_factor=down,
    )
    y_down = torch.rand(batch_size, num_patterns, height_down, width_down)
    y_up = upsampler(y_down)

    assert y_up.shape == (batch_size, num_patterns, height_down * down, width_down * down)


def test_locality_aware_gradient_flows_to_weights():
    upsampler = LocalityAwareUpsampling(
        num_patterns=2,
        height_down=4,
        width_down=4,
        downscale_factor=2,
    )
    y_down = torch.rand(1, 2, 4, 4, requires_grad=True)
    y_up = upsampler(y_down)
    y_up.sum().backward()

    assert upsampler.weights.grad is not None
    assert torch.isfinite(upsampler.weights.grad).all()


def test_locality_aware_tiles_scalar_measurement_into_patch():
    down = 2
    upsampler = LocalityAwareUpsampling(
        num_patterns=1,
        height_down=2,
        width_down=2,
        downscale_factor=down,
    )
    upsampler.weights.data.fill_(1.0)

    y_down = torch.zeros(1, 1, 2, 2)
    y_down[0, 0, 0, 0] = 3.0

    y_up = upsampler(y_down)
    expected = torch.zeros(1, 1, 4, 4)
    expected[0, 0, 0:2, 0:2] = 3.0
    assert torch.allclose(y_up, expected)


def test_transpose_conv_output_shape():
    batch_size, num_patterns = 1, 3
    height_down, width_down, down = 5, 5, 4

    upsampler = TransposeConvUpsampling(num_patterns=num_patterns, downscale_factor=down)
    y_down = torch.rand(batch_size, num_patterns, height_down, width_down)
    y_up = upsampler(y_down)

    assert y_up.shape == (batch_size, num_patterns, height_down * down, width_down * down)


def test_locality_upsampling_wrapper_modes():
    height_down, width_down, down = 8, 8, 8

    for mode in ("locality_aware", "transpose_conv"):
        config = LocalityUpsamplingConfig(
            mode=mode,
            downscale_factor=down,
            num_patterns=8,
        )
        model = LocalityUpsampling(config, height_down=height_down, width_down=width_down)
        y_down = torch.rand(2, 8, height_down, width_down)
        y_up = model(y_down)
        assert y_up.shape == (2, 8, height_down * down, width_down * down)


def test_locality_upsampling_from_yaml(yaml_config):
    height_down, width_down = 8, 8
    model = LocalityUpsampling.from_dict(yaml_config["inverse_model"]["upsampling"], height_down, width_down)
    down = yaml_config["inverse_model"]["upsampling"]["downscale_factor"]
    num_patterns = yaml_config["inverse_model"]["upsampling"]["num_patterns"]
    y_down = torch.rand(1, num_patterns, height_down, width_down)
    y_up = model(y_down)
    assert y_up.shape == (1, num_patterns, height_down * down, width_down * down)
