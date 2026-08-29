"""Tests for U2OS preprocessing utilities."""

from __future__ import annotations

import torch

from datasets.u2os import U2OSPreprocessor


def test_u2os_preprocessing_chain_shape_and_range():
    stack = torch.arange(60 * 16 * 16, dtype=torch.float32).reshape(60, 16, 16)
    image = U2OSPreprocessor.preprocess_stack(stack, bias=0.0, clip_max=500.0, downscale_factor=2.0)

    assert image.shape[0] == 1
    assert image.shape[1] == 8
    assert image.shape[2] == 8
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_u2os_maximum_intensity_projection():
    stack = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 1.0], [2.0, 9.0]],
        ]
    )
    projected = U2OSPreprocessor.maximum_intensity_projection(stack)
    expected = torch.tensor([[5.0, 2.0], [3.0, 9.0]])
    assert torch.allclose(projected, expected)
