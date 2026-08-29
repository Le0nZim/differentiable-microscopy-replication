"""Tests for MCF7 channel-2 dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHANNEL_DIR = ROOT / "data/mcf7_bbbc021/channel2_selected"


@pytest.mark.skipif(not CHANNEL_DIR.exists() or not any(CHANNEL_DIR.glob("*.tif")), reason="MCF7 not extracted")
def test_mcf7_dataset_len():
    from datasets.mcf7_channel2 import MCF7Channel2Config, MCF7Channel2Dataset

    cfg = MCF7Channel2Config(num_train=8, num_val=4, num_test=4)
    train = MCF7Channel2Dataset(cfg, "train")
    assert len(train) == 8
    sample = train[0]
    assert sample.shape == (1, cfg.patch_size, cfg.patch_size)
    assert sample.isfinite().all()


@pytest.mark.skipif(not CHANNEL_DIR.exists() or not any(CHANNEL_DIR.glob("*.tif")), reason="MCF7 not extracted")
def test_mcf7_no_nan():
    from datasets.mcf7_channel2 import MCF7Channel2Config, MCF7Channel2Dataset

    ds = MCF7Channel2Dataset(MCF7Channel2Config(num_train=4, num_val=2, num_test=2), "val")
    for i in range(len(ds)):
        x = ds[i]
        assert x.min() >= 0 and x.max() <= 1
