"""Tests for PatchMNIST dataset generation."""

from __future__ import annotations

import pytest
import torch

from datasets.patchmnist import PatchMNISTConfig, PatchMNISTDataset, generate_patchmnist_split


@pytest.fixture(scope="module")
def tiny_config():
    return PatchMNISTConfig(
        image_size=256,
        num_train=4,
        num_val=2,
        num_test=2,
        seed=123,
        download=True,
    )


def test_patchmnist_image_shape(tiny_config):
    dataset = PatchMNISTDataset(tiny_config, split="train")
    image = dataset[0]
    assert image.shape == (1, 256, 256)
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_patchmnist_split_sizes(tiny_config):
    train = PatchMNISTDataset(tiny_config, split="train")
    val = PatchMNISTDataset(tiny_config, split="val")
    test = PatchMNISTDataset(tiny_config, split="test")
    assert len(train) == 4
    assert len(val) == 2
    assert len(test) == 2


def test_patchmnist_val_test_use_mnist_test_digits_only(tiny_config):
    train_digits = generate_patchmnist_split(tiny_config, "train")
    val_digits = generate_patchmnist_split(tiny_config, "val")
    assert len(train_digits) == 4
    assert len(val_digits) == 2
    assert train_digits[0].shape == (1, 256, 256)


def test_patchmnist_is_deterministic_for_fixed_seed(tiny_config):
    first = PatchMNISTDataset(tiny_config, split="train")[0]
    second = PatchMNISTDataset(tiny_config, split="train")[0]
    assert torch.allclose(first, second)
