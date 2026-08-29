"""Preprocessing-ablation variants for BBBC022 Hoechst 33342 images.

This module is **purely additive**: it does NOT modify or replace the existing
official preprocessing path in ``datasets.bbbc022_hoechst`` (``preprocess_image``
with modes ``paper_strict`` / ``bbbc022_calibrated`` / ``raw_normalized``). It is
used by the focused preprocessing ablation study comparing four modes on the
content-aware reconstruction (paper Fig. 3) and segmentation (paper Fig. 4)
experiments.

Modes (selectable via config / CLI):

* ``aggressive_current`` (A) -- the paper's U2OS-style aggressive pipeline:
  optional MIP -> subtract fixed camera bias (134.28) -> hard clip to [0, 500]
  -> per-image min-max -> downscale by ``63/20`` (area). This reproduces the
  pipeline described for U2OS in paper Sec. 5.1.
* ``minimal_percentile`` (B) -- BBBC022-appropriate, less aggressive: assume 2D,
  float32, robust per-image percentile clip (q_low=0.001, q_high=0.999),
  normalize to [0, 1]. No fixed bias subtraction, no fixed clip at 500, no
  full-image downscaling.
* ``per_image_minmax_no_clip`` (C) -- float32, per-image min-max to [0, 1]. No
  bias, no clip, no downscaling.
* ``trainset_global_percentile`` (D) -- estimate low/high percentiles from the
  *training* images only, then apply those fixed values to train/val/test (no
  per-image leakage from val/test). No bias, no clip, no downscaling.

Spatial handling (cropping / flips / padding) is performed downstream by the
dataset wrapper and is identical across modes, except for the explicit
``63/20`` downscaling that is part of mode A only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets.bbbc022_hoechst import (
    load_tiff,
    make_pseudo_mask,
    maximum_intensity_projection,
    pad_for_crop,
)

PreprocMode = Literal[
    "aggressive_current",
    "minimal_percentile",
    "per_image_minmax_no_clip",
    "trainset_global_percentile",
    # Principle-matched analogs of the paper U2OS recipe (bias/clip estimated
    # from THIS camera rather than copying 134.28 / 500).
    "principle_calibrated",
    "principle_matched_window",
]

ALL_MODES: tuple[PreprocMode, ...] = (
    "aggressive_current",
    "minimal_percentile",
    "per_image_minmax_no_clip",
    "trainset_global_percentile",
)

# Paper U2OS constants (Sec. 5.1).
PAPER_BIAS = 134.28
PAPER_CLIP_MAX = 500.0
PAPER_DOWNSCALE = 63.0 / 20.0


@dataclass
class PreprocParams:
    """Parameters shared by the preprocessing modes.

    ``global_low`` / ``global_high`` are only used by ``trainset_global_percentile``
    and must be fit on the training split via :func:`fit_trainset_global_percentiles`.
    """

    bias: float = PAPER_BIAS
    clip_max: float = PAPER_CLIP_MAX
    downscale_factor_aggressive: float = PAPER_DOWNSCALE
    q_low: float = 0.001
    q_high: float = 0.999
    global_low: Optional[float] = None
    global_high: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict) -> "PreprocParams":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


def _to_2d_float(image: torch.Tensor) -> torch.Tensor:
    """MIP over Z if a stack (no-op for 2D fields); cast to float32, shape [H, W]."""
    return maximum_intensity_projection(image).float()


def _minmax(img: torch.Tensor) -> torch.Tensor:
    mn, mx = img.min(), img.max()
    if mx > mn:
        return (img - mn) / (mx - mn)
    return torch.zeros_like(img)


def _quantile(img: torch.Tensor, q: float) -> float:
    """Robust quantile that tolerates very large tensors (subsamples if needed)."""
    flat = img.flatten()
    cap = 2_000_000
    if flat.numel() > cap:
        idx = torch.linspace(0, flat.numel() - 1, cap, dtype=torch.long, device=flat.device)
        flat = flat[idx]
    return float(torch.quantile(flat, q))


def preprocess_aggressive_current(image: torch.Tensor, params: PreprocParams) -> torch.Tensor:
    """Mode A: paper U2OS pipeline (bias 134.28, clip 500, min-max, downscale 63/20)."""
    img = _to_2d_float(image)
    img = img - params.bias
    img = torch.clamp(img, min=0.0, max=params.clip_max)
    img = _minmax(img).unsqueeze(0)  # [1, H, W]
    factor = params.downscale_factor_aggressive
    if factor and factor != 1.0:
        img = F.interpolate(img.unsqueeze(0), scale_factor=1.0 / factor, mode="area").squeeze(0)
    return img


def preprocess_minimal_percentile(image: torch.Tensor, params: PreprocParams) -> torch.Tensor:
    """Mode B: per-image robust percentile clip + normalize; no bias/clip/downscale."""
    img = _to_2d_float(image)
    lo = _quantile(img, params.q_low)
    hi = _quantile(img, params.q_high)
    img = torch.clamp(img, min=lo, max=hi)
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = torch.zeros_like(img)
    return img.unsqueeze(0)


def preprocess_per_image_minmax_no_clip(image: torch.Tensor, params: PreprocParams) -> torch.Tensor:
    """Mode C: per-image min-max only; no bias/clip/downscale."""
    img = _to_2d_float(image)
    return _minmax(img).unsqueeze(0)


def preprocess_principle_calibrated(image: torch.Tensor, params: PreprocParams) -> torch.Tensor:
    """Per-image analog of camera-bias + outlier-clip, estimated from this field.

    Paper: subtract a *camera* dark offset, then clip outliers, then min-max.
    Here the dark offset is the per-image ``q_low`` quantile (default p0.1/p1)
    and the outlier clip is the per-image ``q_high`` quantile of the
    background-subtracted field (default p99.9). No 63/20 downscale.
    """
    img = _to_2d_float(image)
    bg = _quantile(img, params.q_low)
    img = img - bg
    hi = _quantile(img, params.q_high)
    img = torch.clamp(img, min=0.0, max=max(hi, 0.0))
    return _minmax(img).unsqueeze(0)


def preprocess_principle_matched_window(image: torch.Tensor, params: PreprocParams) -> torch.Tensor:
    """Keep the paper's 500-count signal window, but estimate bias from this field.

    Paper clamps to ``[0, 500]`` *after* subtracting 134.28. This subtracts the
    per-image ``q_low`` background and still clamps to ``[0, clip_max]`` (500).
    Tests whether the damage is the copied 134.28 number or the 500-count cap.
    """
    img = _to_2d_float(image)
    bg = _quantile(img, params.q_low)
    img = img - bg
    img = torch.clamp(img, min=0.0, max=params.clip_max)
    return _minmax(img).unsqueeze(0)


def preprocess_trainset_global_percentile(image: torch.Tensor, params: PreprocParams) -> torch.Tensor:
    """Mode D: apply train-set-fit global percentiles to all splits; no bias/clip/downscale."""
    if params.global_low is None or params.global_high is None:
        raise ValueError(
            "trainset_global_percentile requires global_low/global_high; "
            "fit them on the train split via fit_trainset_global_percentiles()."
        )
    img = _to_2d_float(image)
    lo, hi = float(params.global_low), float(params.global_high)
    img = torch.clamp(img, min=lo, max=hi)
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = torch.zeros_like(img)
    return img.unsqueeze(0)


_DISPATCH = {
    "aggressive_current": preprocess_aggressive_current,
    "minimal_percentile": preprocess_minimal_percentile,
    "per_image_minmax_no_clip": preprocess_per_image_minmax_no_clip,
    "trainset_global_percentile": preprocess_trainset_global_percentile,
    "principle_calibrated": preprocess_principle_calibrated,
    "principle_matched_window": preprocess_principle_matched_window,
}


def preprocess(image: torch.Tensor, mode: PreprocMode, params: PreprocParams) -> torch.Tensor:
    """Dispatch to the requested preprocessing mode. Returns ``[1, H, W]`` in [0, 1]."""
    if mode not in _DISPATCH:
        raise ValueError(f"Unknown preprocessing mode: {mode!r}. Valid: {tuple(_DISPATCH)}")
    return _DISPATCH[mode](image, params)


def mode_description(mode: PreprocMode, params: PreprocParams) -> str:
    """Human-readable one-line definition of a mode for reports."""
    if mode == "aggressive_current":
        return (
            f"MIP(if stack) -> subtract bias {params.bias} -> clip [0, {params.clip_max}] "
            f"-> per-image min-max -> downscale x{params.downscale_factor_aggressive:.4g} (area)"
        )
    if mode == "minimal_percentile":
        return (
            f"float32 -> per-image clip [q{params.q_low}, q{params.q_high}] -> normalize to [0,1] "
            f"(no bias, no fixed clip, no downscaling)"
        )
    if mode == "per_image_minmax_no_clip":
        return "float32 -> per-image min-max to [0,1] (no bias, no clip, no downscaling)"
    if mode == "trainset_global_percentile":
        lo = "unfit" if params.global_low is None else f"{params.global_low:.2f}"
        hi = "unfit" if params.global_high is None else f"{params.global_high:.2f}"
        return (
            f"train-fit global percentiles [q{params.q_low}->{lo}, q{params.q_high}->{hi}] "
            f"applied to all splits (no bias, no clip, no downscaling, no val/test leakage)"
        )
    if mode == "principle_calibrated":
        return (
            f"per-image analog of camera-bias+outlier-clip: subtract q{params.q_low}, "
            f"clip q{params.q_high} of residual, min-max (no 63/20 downscale)"
        )
    if mode == "principle_matched_window":
        return (
            f"per-image q{params.q_low} background subtract, then paper 500-count "
            f"window clip [0, {params.clip_max}], min-max (no 63/20 downscale)"
        )
    return str(mode)


def fit_trainset_global_percentiles(
    train_paths: list[Path],
    *,
    q_low: float,
    q_high: float,
    max_pixels_per_image: int = 30_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Estimate global low/high percentiles from TRAIN images only.

    Pools a deterministic random subsample of raw (MIP, float) pixels from each
    training image and computes the requested percentiles on the pooled pool.
    No bias subtraction or clipping is applied before estimation, and no
    validation/test pixels are used (prevents leakage).
    """
    from datasets.bbbc022_hoechst import load_tiff

    generator = torch.Generator().manual_seed(seed)
    pooled: list[torch.Tensor] = []
    for path in train_paths:
        img = _to_2d_float(load_tiff(path)).flatten()
        n = img.numel()
        if n > max_pixels_per_image:
            idx = torch.randint(0, n, (max_pixels_per_image,), generator=generator)
            img = img[idx]
        pooled.append(img)
    pool = torch.cat(pooled)
    cap = 5_000_000
    if pool.numel() > cap:
        idx = torch.randint(0, pool.numel(), (cap,), generator=generator)
        pool = pool[idx]
    lo = float(torch.quantile(pool, q_low))
    hi = float(torch.quantile(pool, q_high))
    return lo, hi


SplitName = Literal["train", "val", "test"]


@dataclass
class PreprocAblationConfig:
    """Config for the preprocessing-ablation dataset.

    All preprocessing modes operate at native resolution (mode A uses
    ``downscale_factor_aggressive=1.0`` per the study decision), so the input
    image and the canonical mask image share spatial size and identical crops.
    """

    data_root: str = "data/substitute_data"
    split_path: str = ""
    repo_root: str = ""
    preproc_mode: PreprocMode = "minimal_percentile"
    bias: float = PAPER_BIAS
    clip_max: float = PAPER_CLIP_MAX
    downscale_factor_aggressive: float = 1.0  # native res for training (study decision)
    q_low: float = 0.001
    q_high: float = 0.999
    global_low: Optional[float] = None
    global_high: Optional[float] = None
    patch_size: int = 256
    train_random_crops: bool = True
    random_flips: bool = True
    # When True, training crops/flips are drawn from the (seeded) global RNG so
    # they vary every epoch (true data augmentation). When False (default) the
    # legacy behaviour is kept: crops/flips are seeded per-index and therefore
    # identical every epoch (one fixed patch per image). The legacy default is
    # preserved so other experiments (segmentation, preprocessing ablation) are
    # untouched; the Fig.3 content-aware training opts in to per-epoch crops.
    epoch_varying_train_crops: bool = False
    seed: int = 42
    return_mask: bool = False
    canonical_mask_mode: PreprocMode = "minimal_percentile"
    mask_threshold: float = 0.3
    mask_closing_kernel: int = 10

    @classmethod
    def from_dict(cls, data: dict) -> "PreprocAblationConfig":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})

    def params(self) -> PreprocParams:
        return PreprocParams(
            bias=self.bias,
            clip_max=self.clip_max,
            downscale_factor_aggressive=self.downscale_factor_aggressive,
            q_low=self.q_low,
            q_high=self.q_high,
            global_low=self.global_low,
            global_high=self.global_high,
        )

    def canonical_params(self) -> PreprocParams:
        # Canonical mask uses a per-image mode (B/C) -> no globals required.
        return PreprocParams(q_low=self.q_low, q_high=self.q_high)


class PreprocAblationDataset(Dataset):
    """BBBC022 dataset whose preprocessing is selectable per mode for the ablation.

    The segmentation pseudo-GT mask (``return_mask=True``) is derived from a single
    *canonical* preprocessing (default ``minimal_percentile``) of the same raw
    image, cropped/flipped identically to the model input, so all modes train and
    are evaluated against the exact same targets.
    """

    def __init__(self, config: PreprocAblationConfig, split: SplitName = "train") -> None:
        from datasets.bbbc022_split import load_split

        self.config = config
        self.split = split
        repo_root = Path(config.repo_root) if config.repo_root else Path.cwd()
        split_paths = load_split(Path(config.split_path), repo_root)
        self.paths = split_paths[split]

        if config.preproc_mode == "trainset_global_percentile" and (
            config.global_low is None or config.global_high is None
        ):
            raise ValueError(
                "trainset_global_percentile requires global_low/global_high in config; "
                "fit them on the train split first and inject them."
            )

        params = config.params()
        self.images = [pad_for_crop(preprocess(load_tiff(p), config.preproc_mode, params), config.patch_size) for p in self.paths]

        self.canon_images: list[torch.Tensor] | None = None
        if config.return_mask:
            cparams = config.canonical_params()
            self.canon_images = [
                pad_for_crop(preprocess(load_tiff(p), config.canonical_mask_mode, cparams), config.patch_size)
                for p in self.paths
            ]
            for img, canon in zip(self.images, self.canon_images):
                if img.shape[-2:] != canon.shape[-2:]:
                    raise ValueError(
                        f"input/canonical size mismatch {img.shape[-2:]} vs {canon.shape[-2:]}; "
                        "all modes must be native resolution for mask alignment."
                    )

    @classmethod
    def from_dict(cls, data: dict, split: SplitName) -> "PreprocAblationDataset":
        return cls(PreprocAblationConfig.from_dict(data), split=split)

    def __len__(self) -> int:
        return len(self.images)

    def _crop_coords(
        self, height: int, width: int, generator: torch.Generator | None
    ) -> tuple[int, int]:
        patch = self.config.patch_size
        if self.split == "train" and self.config.train_random_crops:
            top = int(torch.randint(0, height - patch + 1, (1,), generator=generator).item())
            left = int(torch.randint(0, width - patch + 1, (1,), generator=generator).item())
        else:
            top = max(0, (height - patch) // 2)
            left = max(0, (width - patch) // 2)
        return top, left

    def _sample_generator(self, index: int) -> torch.Generator | None:
        """Return the RNG used for train-time crops/flips.

        ``None`` selects the (seeded) global RNG so crops vary every epoch (true
        augmentation); used for the train split when ``epoch_varying_train_crops``
        is set. Otherwise a per-index deterministic generator reproduces the
        legacy fixed-patch behaviour (one identical crop per image every epoch).
        """
        if self.split == "train" and self.config.epoch_varying_train_crops:
            return None
        generator = torch.Generator()
        generator.manual_seed(self.config.seed + index + {"train": 0, "val": 10_000, "test": 20_000}[self.split])
        return generator

    def __getitem__(self, index: int):
        image = self.images[index]
        _, height, width = image.shape
        patch = self.config.patch_size
        generator = self._sample_generator(index)

        top, left = self._crop_coords(height, width, generator)
        crop = image[:, top : top + patch, left : left + patch]

        flip_h = self.split == "train" and self.config.random_flips and torch.rand((), generator=generator).item() > 0.5
        flip_v = self.split == "train" and self.config.random_flips and torch.rand((), generator=generator).item() > 0.5
        if flip_h:
            crop = torch.flip(crop, dims=[-1])
        if flip_v:
            crop = torch.flip(crop, dims=[-2])

        if not self.config.return_mask:
            return crop

        assert self.canon_images is not None
        canon_crop = self.canon_images[index][:, top : top + patch, left : left + patch]
        if flip_h:
            canon_crop = torch.flip(canon_crop, dims=[-1])
        if flip_v:
            canon_crop = torch.flip(canon_crop, dims=[-2])
        mask = make_pseudo_mask(canon_crop, self.config.mask_threshold, self.config.mask_closing_kernel)
        return crop, mask
