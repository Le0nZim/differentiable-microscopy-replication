"""Tests for device resolution."""

from __future__ import annotations

import pytest
import torch

from utils.device import resolve_device


def test_resolve_cpu():
    assert resolve_device("cpu").type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_resolve_cuda_index():
    device = resolve_device("cuda:0")
    assert device.type == "cuda"
    assert device.index == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_resolve_gpu_alias():
    device = resolve_device("gpu0")
    assert device.type == "cuda"
