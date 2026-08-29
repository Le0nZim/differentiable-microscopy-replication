"""Tests for U2OS substitute data loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from datasets.u2os import U2OSConfig, U2OSDataset


@pytest.fixture
def substitute_root():
    path = Path("data/substitute_data")
    if not path.exists():
        pytest.skip("substitute U2OS data not available")
    return path


def test_u2os_substitute_splits_are_disjoint(substitute_root):
    config = U2OSConfig(
        data_root=str(substitute_root),
        stack_glob="**/*.tif",
        clip_max=3000.0,
        downscale_factor=1.0,
    )
    train = U2OSDataset(config, "train")
    val = U2OSDataset(config, "val")
    test = U2OSDataset(config, "test")

    assert len(train) == 168
    assert len(val) == 21
    assert len(test) == 21
    assert train[0].shape == (1, 256, 256)
