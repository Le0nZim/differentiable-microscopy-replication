"""Tests for BBBC022 Hoechst substitute dataset."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from datasets.bbbc022_hoechst import (
    BBBC022HoechstConfig,
    BBBC022HoechstDataset,
    assign_split_paths,
    discover_image_paths,
    make_pseudo_mask,
    select_hoechst_paths,
)


@pytest.fixture
def substitute_root():
    path = Path("data/substitute_data")
    if not path.exists():
        pytest.skip("BBBC022 substitute data not available")
    return path


def test_hoechst_paths_discovered(substitute_root):
    paths = select_hoechst_paths(discover_image_paths(substitute_root, "**/*.tif"))
    assert len(paths) >= 210


def test_well_aware_splits_disjoint(substitute_root):
    paths = select_hoechst_paths(discover_image_paths(substitute_root, "**/*.tif"))
    config = BBBC022HoechstConfig(data_root=str(substitute_root), split_by_well=True)
    splits = assign_split_paths(paths, config)
    train_wells = {p.name.split("_")[1] for p in splits["train"]}
    val_wells = {p.name.split("_")[1] for p in splits["val"]}
    test_wells = {p.name.split("_")[1] for p in splits["test"]}
    assert train_wells.isdisjoint(val_wells)
    assert train_wells.isdisjoint(test_wells)
    assert val_wells.isdisjoint(test_wells)


def test_dataset_tensors_in_unit_interval(substitute_root):
    config = BBBC022HoechstConfig(
        data_root=str(substitute_root),
        preprocessing_mode="bbbc022_calibrated",
        downscale_factor=1.0,
    )
    train = BBBC022HoechstDataset(config, "train")
    val = BBBC022HoechstDataset(config, "val")
    test = BBBC022HoechstDataset(config, "test")
    assert len(train) == 168 and len(val) == 21 and len(test) == 21
    for ds in (train, val, test):
        x = ds[0]
        assert x.shape == (1, 256, 256)
        assert torch.isfinite(x).all()
        assert x.min() >= 0.0 and x.max() <= 1.0


def test_val_test_crops_deterministic(substitute_root):
    config = BBBC022HoechstConfig(data_root=str(substitute_root), preprocessing_mode="raw_normalized")
    val = BBBC022HoechstDataset(config, "val")
    assert torch.allclose(val[0], val[0])


def test_pseudo_mask_finite():
    x = torch.rand(1, 64, 64)
    m = make_pseudo_mask(x, 0.3, 10)
    assert m.shape == x.shape
    assert torch.isfinite(m).all()
