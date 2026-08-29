"""Tests for evaluation metrics."""

from __future__ import annotations

import torch

from evaluation.metrics import mse, psnr, ssim


def test_mse_identical_images_is_zero():
    image = torch.rand(2, 1, 32, 32)
    assert torch.allclose(mse(image, image), torch.tensor(0.0))


def test_ssim_identical_images_is_one():
    image = torch.rand(2, 1, 32, 32)
    value = ssim(image, image)
    assert value > 0.99


def test_psnr_identical_images_is_large():
    image = torch.rand(1, 1, 32, 32)
    value = psnr(image, image)
    assert value > 40.0
