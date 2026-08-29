"""SwinIR dataset adapter tests."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SET5 = ROOT / "data/sr/test/Set5/HR"


@pytest.mark.skipif(not SET5.exists() or not any(SET5.glob("*.png")), reason="Set5 not extracted")
def test_sr_dataset_adapter():
    from baselines.swinir.dataset_adapter import SRImageFolderDataset

    ds = SRImageFolderDataset(SET5, patch_size=32, max_images=2)
    assert len(ds) >= 1
    x = ds[0]
    assert x.shape == (1, 32, 32)
