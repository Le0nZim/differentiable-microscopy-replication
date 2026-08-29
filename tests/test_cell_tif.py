"""Tests for cell.tif sanity-check dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from datasets.cell_tif import CellTifConfig, CellTifDataset, load_cell_tif_image, preprocess_cell_tif_image


@pytest.fixture
def cell_tif_path():
    path = Path("data/cell.tif")
    if not path.exists():
        pytest.skip("data/cell.tif not available")
    return path


def test_load_cell_tif_shape(cell_tif_path):
    image = load_cell_tif_image(cell_tif_path)
    assert image.ndim == 3
    assert image.shape[0] == 1
    assert image.shape[1] == image.shape[2]


def test_preprocess_cell_tif_range(cell_tif_path):
    raw = load_cell_tif_image(cell_tif_path)
    processed = preprocess_cell_tif_image(
        raw,
        bias=134.28,
        clip_max=500.0,
        downscale_factor=63.0 / 20.0,
    )
    assert processed.ndim == 3
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_cell_tif_dataset_patch_shape(cell_tif_path):
    dataset = CellTifDataset(
        CellTifConfig(
            image_path=str(cell_tif_path),
            patch_size=64,
            num_patches=4,
        )
    )
    patch = dataset[0]
    assert patch.shape == (1, 64, 64)
