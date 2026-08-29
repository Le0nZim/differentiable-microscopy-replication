"""Tests for excitation pattern generation."""

from __future__ import annotations

import pytest
import torch

from models.pattern_generator import (
    PatternGenerator,
    PatternGeneratorConfig,
    SigmoidSchedule,
)
from utils.experiment_config import load_experiment_config


@pytest.fixture
def yaml_config():
    return load_experiment_config("configs/_shared/base_patchmnist.yaml")


def test_learnable_frequency_shape_and_range():
    config = PatternGeneratorConfig(num_patterns=4, height=32, width=32, seed=0)
    generator = PatternGenerator(config)
    patterns = generator()

    assert patterns.shape == (4, 1, 32, 32)
    assert patterns.min() >= 0.0
    assert patterns.max() <= 1.0


def test_learnable_frequency_gradient_flows_to_w():
    config = PatternGeneratorConfig(num_patterns=2, height=16, width=16, seed=1)
    generator = PatternGenerator(config)
    patterns = generator()
    # patterns: [T, 1, H, W]
    loss = patterns.sum()
    loss.backward()

    assert generator.W is not None
    assert generator.W.grad is not None
    assert torch.isfinite(generator.W.grad).all()


def test_frequency_checkpoint_roundtrip(tmp_path):
    config = PatternGeneratorConfig(num_patterns=2, height=16, width=16, seed=7)
    generator = PatternGenerator(config)
    payload = generator.export_frequency_checkpoint(sigmoid_m=1.0)
    path = tmp_path / "pattern_init.pt"
    torch.save(payload, path)

    other = PatternGenerator(config)
    other.W.data.normal_()
    loaded = PatternGenerator.load_frequency_checkpoint(path)
    other.load_frequency_weights(loaded["W"])

    assert torch.allclose(generator.W, other.W)
    assert torch.allclose(generator(sigmoid_m=1.0), other(sigmoid_m=1.0))


def test_sigmoid_schedule_stage_a_and_m_updates():
    schedule = SigmoidSchedule(epoch_baseline=20, epoch_cutoff=35, epoch_step=5, m_init=1.0)

    assert schedule.should_freeze_patterns(epoch=10) is True
    assert schedule.get_m() == 1.0

    schedule.step(epoch=21)
    assert schedule.should_freeze_patterns(epoch=21) is False
    assert schedule.get_m() == 1.0

    schedule.step(epoch=40)
    assert schedule.get_m() == 2.0

    schedule.step(epoch=41)
    assert schedule.get_m() == 1.0


def test_fixed_pattern_modes_have_expected_shapes():
    height, width, num_patterns = 16, 16, 3

    for mode in ("random_fixed", "uniform_all_ones", "hadamard_fixed"):
        config = PatternGeneratorConfig(
            mode=mode,
            num_patterns=num_patterns,
            height=height,
            width=width,
            seed=7,
        )
        patterns = PatternGenerator(config)()
        assert patterns.shape == (num_patterns, 1, height, width)
        assert patterns.min() >= 0.0
        assert patterns.max() <= 1.0


def test_uniform_patterns_are_all_ones():
    config = PatternGeneratorConfig(
        mode="uniform_all_ones",
        num_patterns=2,
        height=8,
        width=8,
    )
    patterns = PatternGenerator(config)()
    assert torch.allclose(patterns, torch.ones_like(patterns))


def test_custom_sigmoid_m_changes_output():
    config = PatternGeneratorConfig(num_patterns=2, height=16, width=16, sigmoid_m=1.0)
    generator = PatternGenerator(config)
    soft = generator(sigmoid_m=1.0)
    sharp = generator(sigmoid_m=20.0)
    assert not torch.allclose(soft, sharp)


def test_pattern_generator_from_yaml(yaml_config):
    generator = PatternGenerator.from_dict(yaml_config["pattern_generator"])
    patterns = generator()
    height = yaml_config["pattern_generator"]["height"]
    width = yaml_config["pattern_generator"]["width"]
    num_patterns = yaml_config["pattern_generator"]["num_patterns"]
    assert patterns.shape == (num_patterns, 1, height, width)


def test_exp1_scale_fixed_patterns_shape_and_range():
    """Paper-scale PatchMNIST: 256x256, T=8, x8 compression."""
    height, width, num_patterns = 256, 256, 8
    downscale = 8
    from utils.experiment_config import compression_ratio

    assert compression_ratio(downscale, num_patterns) == 8.0

    for mode in ("uniform_all_ones", "hadamard_fixed", "random_fixed"):
        config = PatternGeneratorConfig(
            mode=mode,
            num_patterns=num_patterns,
            height=height,
            width=width,
            seed=42,
        )
        patterns = PatternGenerator(config)()
        assert patterns.shape == (num_patterns, 1, height, width)
        assert patterns.min() >= 0.0
        assert patterns.max() <= 1.0


def test_hadamard_patterns_are_deterministic():
    config = PatternGeneratorConfig(
        mode="hadamard_fixed",
        num_patterns=8,
        height=32,
        width=32,
        seed=0,
    )
    a = PatternGenerator(config)()
    b = PatternGenerator(config)()
    assert torch.allclose(a, b)


def test_hadamard_patterns_differ_across_indices():
    config = PatternGeneratorConfig(
        mode="hadamard_fixed",
        num_patterns=8,
        height=32,
        width=32,
    )
    patterns = PatternGenerator(config)()
    assert not torch.allclose(patterns[0], patterns[1])
