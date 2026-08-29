"""Tests for the reconstruction CNN."""

from __future__ import annotations

import pytest
import torch

from models.inverse_model import InverseModel
from models.recon_cnn import ReconCNN, ReconCNNConfig
from utils.experiment_config import load_experiment_config


@pytest.fixture
def yaml_config():
    return load_experiment_config("configs/base_patchmnist.yaml")


def test_recon_cnn_output_shape():
    batch_size, num_patterns = 2, 8
    height, width = 64, 64

    model = ReconCNN(ReconCNNConfig(in_channels=num_patterns))
    features = torch.rand(batch_size, num_patterns, height, width)
    reconstruction = model(features)

    assert reconstruction.shape == (batch_size, 1, height, width)


def test_recon_cnn_output_in_unit_interval():
    model = ReconCNN(ReconCNNConfig(in_channels=4, hidden_channels=[8, 8, 8, 8, 8, 1]))
    model.eval()
    features = torch.rand(2, 4, 32, 32)
    reconstruction = model(features)

    assert reconstruction.min() >= 0.0
    assert reconstruction.max() <= 1.0


def test_recon_cnn_gradient_flow():
    model = ReconCNN(ReconCNNConfig(in_channels=3, hidden_channels=[8, 8, 8, 8, 8, 1]))
    features = torch.rand(2, 3, 16, 16, requires_grad=True)
    reconstruction = model(features)
    reconstruction.sum().backward()

    assert features.grad is not None
    assert any(param.grad is not None for param in model.parameters())


def test_recon_cnn_from_yaml(yaml_config):
    model = ReconCNN.from_dict(yaml_config["inverse_model"]["reconstruction"])
    features = torch.rand(1, 8, 32, 32)
    reconstruction = model(features)
    assert reconstruction.shape == (1, 1, 32, 32)


def test_inverse_model_end_to_end_shape(yaml_config):
    height_down, width_down = 8, 8
    down = yaml_config["inverse_model"]["upsampling"]["downscale_factor"]
    num_patterns = yaml_config["inverse_model"]["upsampling"]["num_patterns"]

    model = InverseModel.from_dict(yaml_config["inverse_model"], height_down=height_down, width_down=width_down)
    model.eval()
    y_down = torch.rand(2, num_patterns, height_down, width_down)
    x_recon = model(y_down)

    assert x_recon.shape == (2, 1, height_down * down, width_down * down)
