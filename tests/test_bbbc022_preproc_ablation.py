"""Tests for the BBBC022 preprocessing-ablation modes (additive; non-destructive)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from datasets.bbbc022_preproc_ablation import (
    ALL_MODES,
    PAPER_DOWNSCALE,
    PreprocParams,
    fit_trainset_global_percentiles,
    preprocess,
)

H, W = 520, 696


def _fake_raw(seed: int = 0) -> torch.Tensor:
    """A uint16-like 2D Hoechst field: low background + a few bright nuclei."""
    g = torch.Generator().manual_seed(seed)
    img = 130 + 8 * torch.randn(H, W, generator=g).abs()
    # a few bright blobs well above the clip ceiling
    for _ in range(5):
        cy = int(torch.randint(20, H - 20, (1,), generator=g))
        cx = int(torch.randint(20, W - 20, (1,), generator=g))
        img[cy - 10 : cy + 10, cx - 10 : cx + 10] += 1500.0
    return img.round().clamp(0, 4095)


def _params() -> PreprocParams:
    return PreprocParams(global_low=140.0, global_high=1400.0)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_output_shape_dtype_finite_range(mode):
    raw = _fake_raw()
    out = preprocess(raw, mode, _params())
    assert out.ndim == 3 and out.shape[0] == 1, f"{mode}: expected [1,H,W], got {tuple(out.shape)}"
    assert out.dtype == torch.float32, f"{mode}: expected float32, got {out.dtype}"
    assert torch.isfinite(out).all(), f"{mode}: non-finite values"
    assert float(out.min()) >= 0.0 - 1e-6 and float(out.max()) <= 1.0 + 1e-6, f"{mode}: out of [0,1]"


def test_minimal_modes_do_not_downscale():
    """Modes B/C/D must preserve native spatial size (no full-image downscaling)."""
    raw = _fake_raw()
    for mode in ("minimal_percentile", "per_image_minmax_no_clip", "trainset_global_percentile"):
        out = preprocess(raw, mode, _params())
        assert out.shape[-2:] == (H, W), f"{mode} unexpectedly changed size to {tuple(out.shape[-2:])}"


def test_aggressive_mode_downscales_by_63_over_20():
    raw = _fake_raw()
    out = preprocess(raw, "aggressive_current", _params())
    expected_h = int(H * (1.0 / PAPER_DOWNSCALE))
    expected_w = int(W * (1.0 / PAPER_DOWNSCALE))
    assert abs(out.shape[-2] - expected_h) <= 1 and abs(out.shape[-1] - expected_w) <= 1, (
        f"aggressive size {tuple(out.shape[-2:])} != ~{(expected_h, expected_w)}"
    )
    assert out.shape[-2] < H and out.shape[-1] < W


def test_aggressive_clip_saturates_bright_signal():
    """Bright blobs above bias+clip should be flattened to the max (==1.0) in mode A."""
    raw = _fake_raw()
    out = preprocess(raw, "aggressive_current", _params()).squeeze(0)
    assert float((out >= 0.999).float().mean()) > 0.0


def test_minimal_percentile_clips_outliers_less_than_aggressive():
    raw = _fake_raw()
    a = preprocess(raw, "aggressive_current", _params())
    b = preprocess(raw, "minimal_percentile", _params())
    # mode A saturates more pixels than the gentle percentile clip of mode B
    assert float((a >= 0.999).float().mean()) >= float((b >= 0.999).float().mean())


def test_trainset_global_requires_fit():
    raw = _fake_raw()
    with pytest.raises(ValueError):
        preprocess(raw, "trainset_global_percentile", PreprocParams(global_low=None, global_high=None))


def test_principle_matched_modes_keep_native_size_and_range():
    """Analog U2OS recipes estimated on this camera must not 63/20-downscale."""
    raw = _fake_raw()
    params = PreprocParams(q_low=0.01, q_high=0.999, clip_max=500.0)
    for mode in ("principle_calibrated", "principle_matched_window"):
        out = preprocess(raw, mode, params)
        assert out.shape[-2:] == (H, W), f"{mode} changed size to {tuple(out.shape[-2:])}"
        assert float(out.min()) >= 0.0 - 1e-6 and float(out.max()) <= 1.0 + 1e-6
        a = preprocess(raw, mode, params)
        assert torch.allclose(out, a)


def test_matched_window_saturates_more_than_calibrated():
    """A 500-count residual window should flatten more nuclei than p99.9 clipping."""
    raw = _fake_raw()
    params = PreprocParams(q_low=0.01, q_high=0.999, clip_max=500.0)
    window = preprocess(raw, "principle_matched_window", params)
    calib = preprocess(raw, "principle_calibrated", params)
    assert float((window >= 0.999).float().mean()) >= float((calib >= 0.999).float().mean())


def test_modes_are_deterministic():
    raw = _fake_raw()
    for mode in ALL_MODES:
        a = preprocess(raw, mode, _params())
        b = preprocess(raw, mode, _params())
        assert torch.allclose(a, b)


def test_fit_trainset_global_percentiles_on_real_data_if_present():
    data_root = Path("data/substitute_data")
    if not data_root.exists():
        pytest.skip("BBBC022 substitute data not available")
    from datasets.bbbc022_hoechst import discover_image_paths, select_hoechst_paths

    paths = select_hoechst_paths(discover_image_paths(data_root, "**/*.tif"))[:6]
    lo, hi = fit_trainset_global_percentiles(paths, q_low=0.001, q_high=0.999, seed=42)
    assert hi > lo and lo >= 0.0


# --------------------------------------------------------------------------- #
# dataset wrapper tests (require real data + a saved split)
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = REPO_ROOT / "results/preprocessing_ablation_bbbc022_hoechst/configs/split.json"


def _need_dataset():
    if not (REPO_ROOT / "data/substitute_data").exists():
        pytest.skip("BBBC022 substitute data not available")
    if not SPLIT_PATH.exists():
        pytest.skip("split.json not built yet (run qc_bbbc022_preprocessing.py)")


def _ds_cfg(mode: str, return_mask: bool) -> dict:
    return {
        "data_root": "data/substitute_data",
        "split_path": str(SPLIT_PATH),
        "repo_root": str(REPO_ROOT),
        "preproc_mode": mode,
        "downscale_factor_aggressive": 1.0,
        "patch_size": 256,
        "seed": 42,
        "return_mask": return_mask,
        "canonical_mask_mode": "minimal_percentile",
    }


@pytest.mark.parametrize("mode", ["aggressive_current", "minimal_percentile", "per_image_minmax_no_clip"])
def test_dataset_patch_shape_dtype_range(mode):
    from datasets.bbbc022_preproc_ablation import PreprocAblationDataset

    _need_dataset()
    ds = PreprocAblationDataset.from_dict(_ds_cfg(mode, return_mask=False), split="val")
    x = ds[0]
    assert x.shape == (1, 256, 256)
    assert x.dtype == torch.float32
    assert torch.isfinite(x).all()
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0 + 1e-6


def test_dataset_mask_shape_and_binary():
    from datasets.bbbc022_preproc_ablation import PreprocAblationDataset

    _need_dataset()
    ds = PreprocAblationDataset.from_dict(_ds_cfg("aggressive_current", return_mask=True), split="val")
    x, m = ds[0]
    assert x.shape == (1, 256, 256) and m.shape == (1, 256, 256)
    assert torch.isfinite(m).all()
    assert set(torch.unique(m).tolist()).issubset({0.0, 1.0})


def test_canonical_mask_identical_across_input_modes():
    """Same canonical (mode B) target regardless of the input preprocessing mode."""
    from datasets.bbbc022_preproc_ablation import PreprocAblationDataset

    _need_dataset()
    ds_a = PreprocAblationDataset.from_dict(_ds_cfg("aggressive_current", return_mask=True), split="val")
    ds_c = PreprocAblationDataset.from_dict(_ds_cfg("per_image_minmax_no_clip", return_mask=True), split="val")
    _, m_a = ds_a[3]
    _, m_c = ds_c[3]
    assert torch.equal(m_a, m_c), "canonical masks must match across input modes"
