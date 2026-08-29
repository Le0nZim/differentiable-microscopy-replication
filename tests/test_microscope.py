"""Tests for the full differentiable microscope pipeline."""

from __future__ import annotations

import torch

from models.microscope import DifferentiableMicroscope
from utils.config import load_yaml_config


def test_microscope_end_to_end_shapes():
    config = load_yaml_config("configs/_shared/base_patchmnist.yaml")
    model = DifferentiableMicroscope.from_run_config(config)

    batch_size = 2
    image_size = config["dataset"]["image_size"]
    num_patterns = config["pattern_generator"]["num_patterns"]
    downscale = config["forward_model"]["downscale_factor"]

    specimen = torch.rand(batch_size, 1, image_size, image_size)
    outputs = model(specimen, apply_noise=False)

    height_down = image_size // downscale
    assert outputs["x_recon"].shape == (batch_size, 1, image_size, image_size)
    assert outputs["patterns"].shape == (num_patterns, 1, image_size, image_size)
    assert outputs["alpha_down"].shape == (batch_size, num_patterns, height_down, height_down)
    assert outputs["y_down"].shape == (batch_size, num_patterns, height_down, height_down)


def test_microscope_gradient_flows_to_illumination_parameters():
    config = load_yaml_config("configs/_shared/base_patchmnist.yaml")
    model = DifferentiableMicroscope.from_run_config(config)
    model.set_illumination_trainable(True)

    specimen = torch.rand(1, 1, config["dataset"]["image_size"], config["dataset"]["image_size"])
    outputs = model(specimen, apply_noise=False)
    outputs["x_recon"].sum().backward()

    illumination_params = model.illumination_parameters()
    assert illumination_params
    assert any(param.grad is not None and torch.isfinite(param.grad).all() for param in illumination_params)
