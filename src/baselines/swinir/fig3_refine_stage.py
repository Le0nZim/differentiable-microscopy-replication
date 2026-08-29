"""Paper-faithful Figure-3 SwinIR *refinement* stage (BBBC022 U2OS-substitute).

Implements the paper §5.6 U2OS/Fig-3 procedure faithfully:

    frozen content-aware base model  ->  SwinIR (upscale=1)  ->  refined 256x256

The base model (forward microscope + illumination Ht + locality-aware upsampling +
reconstruction CNN) is loaded from the frozen ``bbbc022_content_aware_v2`` runs and
kept entirely frozen (``requires_grad=False`` *and* ``eval()`` so the recon-CNN's
BatchNorm running stats never move). Only the SwinIR refines ``x_base -> x_gt``.

Key faithfulness points (see TABLE2_FIG7_VS_FIG3_SWINIR_DIFF.md):
  * SwinIR uses the SAME ``build_swinir_from_config`` and (for ``paper_faithful``)
    the SAME SwinIR-M capacity + loss stack (pixel+perceptual+adversarial) as the
    validated Table-2/Fig-7 codepath.
  * ``upscale=1`` — SwinIR is an image-to-image restorer, not a spatial upscaler.
  * Trained on 64x64 crops of the frozen 256x256 reconstruction (paper trains
    Table-2 SwinIR "for 64x64 image patches"); evaluated at full 256x256 (SwinIR
    recomputes window masks per input size, so it is size-agnostic).
  * Input/target are the *paired* [0,1] tensors the base was trained on — no
    independent per-image/per-method min-max renormalisation anywhere.

This module is deliberately additive and does NOT modify any frozen run or the
existing ``train_fig03_swinir_columns.py`` path.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from baselines.swinir.losses import build_loss_stack, pixel_loss
from baselines.swinir.refinement_model import OfflineSwinIRRefinement
from evaluation.metrics import mse as mse_metric
from evaluation.metrics import psnr as psnr_metric
from evaluation.metrics import ssim as ssim_metric
from models.microscope import DifferentiableMicroscope
from training.dataloaders import build_dataset
from utils.experiment_config import load_experiment_config

REPO_ROOT = Path(__file__).resolve().parents[3]

ALL_COMPS = ["x16", "x64", "x256", "x1024"]
PATTERNS = ["random_fixed", "learnable_frequency"]
# Eval/forward sigmoid sharpness per pattern (matches how the base results.csv was
# computed: learnable uses sharpen_eval_m=10; fixed patterns ignore m).
EVAL_M = {"random_fixed": None, "learnable_frequency": 10.0}

VALID_LOSS_MODES = ("l1_only", "l1_ssim", "paper_pixel_perceptual_gan")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_stage_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _validate_stage_config(cfg)
    return cfg


def _validate_stage_config(cfg: dict[str, Any]) -> None:
    for key in ("name", "swinir", "train", "loss"):
        if key not in cfg:
            raise ValueError(f"stage config missing top-level '{key}'")
    sw = cfg["swinir"]
    if int(sw.get("upscale", 1)) != 1:
        raise ValueError("Fig-3 SwinIR must use upscale=1 (image-to-image refinement)")
    tr = cfg["train"]
    micro = int(tr["micro_batch_size"])
    accum = int(tr["grad_accum"])
    eff = int(tr["effective_batch_size"])
    if micro * accum != eff:
        raise ValueError(f"micro_batch_size*grad_accum ({micro}*{accum}) != effective_batch_size ({eff})")
    ps = int(tr["train_patch_size"])
    ws = int(sw.get("window_size", 8))
    if ps % ws != 0:
        raise ValueError(f"train_patch_size {ps} not divisible by window_size {ws}")
    mode = cfg["loss"]["mode"]
    if mode not in VALID_LOSS_MODES:
        raise ValueError(f"loss.mode must be one of {VALID_LOSS_MODES}, got {mode!r}")


def swinir_cfg_from_stage(cfg: dict[str, Any]) -> dict[str, Any]:
    sw = dict(cfg["swinir"])
    sw.setdefault("upscale", 1)
    sw.setdefault("in_chans", 1)
    sw.setdefault("window_size", 8)
    sw.setdefault("mlp_ratio", 2)
    sw.setdefault("resi_connection", "1conv")
    sw.setdefault("img_range", 1.0)
    sw.setdefault("upsampler", "")
    # SwinIR img_size drives the (unused unless ape) pos table + the fast-path
    # window-mask; set it to the training patch size so the long loop hits the
    # fast path (eval at 256 recomputes masks, which is exact + cheap).
    sw["img_size"] = int(cfg["train"]["train_patch_size"])
    return sw


def apply_identity_init(swinir: nn.Module) -> None:
    """Zero the final residual conv so the refiner starts at identity (output=input).

    For ``upsampler=''`` SwinIR computes ``out = x + conv_last(res)`` (a global
    residual on the input). Zeroing ``conv_last`` makes the untrained refiner an
    exact identity, i.e. it starts at the *base reconstruction's* quality and can
    only improve it. This is the standard init for a residual refinement stage; it
    does not restrict the reachable solution (full capacity is retained), it only
    fixes the starting point — which massively improves sample-efficiency for a
    frozen-base-then-refine setup. Documented as a Fig-3-refinement design choice.
    """
    if hasattr(swinir, "conv_last") and isinstance(swinir.conv_last, nn.Conv2d):
        nn.init.zeros_(swinir.conv_last.weight)
        if swinir.conv_last.bias is not None:
            nn.init.zeros_(swinir.conv_last.bias)


def build_refiner(swinir_cfg: dict[str, Any], device: torch.device, *,
                  identity_init: bool = True, seed: int | None = None):
    """Build the OfflineSwinIRRefinement (direct, image-to-image) refiner."""
    if seed is not None:
        torch.manual_seed(seed)
    refiner = OfflineSwinIRRefinement(swinir_cfg, {"mode": "direct"}).to(device)
    if identity_init:
        apply_identity_init(refiner.swinir)
    return refiner


# ---------------------------------------------------------------------------
# Frozen base model
# ---------------------------------------------------------------------------
def base_run_dir(base_exp_root: Path, comp: str, pattern: str) -> Path:
    return base_exp_root / f"bbbc022_{comp}_{pattern}_seed42"


def load_frozen_base(base_exp_root: Path, comp: str, pattern: str, device: torch.device):
    """Load a frozen content-aware base microscope + its run config.

    Returns (model, run_cfg, eval_m). The model is fully frozen and in eval mode.
    """
    rd = base_run_dir(base_exp_root, comp, pattern)
    run_cfg = load_experiment_config(rd / "config.yaml")
    run_cfg["experiment"]["device"] = str(device)
    image_size = int(run_cfg["dataset"]["image_size"])
    eval_m = EVAL_M[pattern]
    model = DifferentiableMicroscope.from_run_config(run_cfg).to(device)
    # Warmup forward to lazily build PSF buffers so the state layout is final.
    model(torch.zeros(1, 1, image_size, image_size, device=device), sigmoid_m=eval_m or 10.0, apply_noise=False)
    payload = torch.load(rd / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, run_cfg, eval_m


def base_is_frozen(model: DifferentiableMicroscope) -> bool:
    return all(not p.requires_grad for p in model.parameters()) and not model.training


# ---------------------------------------------------------------------------
# Frozen-base cache: (x_base_256, gt_256) paired tensors
# ---------------------------------------------------------------------------
@dataclass
class PairCache:
    """Paired (frozen-base reconstruction, ground truth) tensors, all 256x256."""

    x_base: torch.Tensor  # [N,1,S,S] float16 on CPU (SwinIR input)
    gt: torch.Tensor      # [N,1,S,S] float16 on CPU (target)
    image_index: torch.Tensor  # [N] source image index (leakage auditing)
    split: str
    crop_size: int

    def __len__(self) -> int:
        return self.x_base.shape[0]


def _crop_positions(h: int, w: int, size: int, gen: torch.Generator | None) -> tuple[int, int]:
    if gen is None:  # center
        return max(0, (h - size) // 2), max(0, (w - size) // 2)
    top = int(torch.randint(0, h - size + 1, (1,), generator=gen).item())
    left = int(torch.randint(0, w - size + 1, (1,), generator=gen).item())
    return top, left


@torch.no_grad()
def build_pair_cache(
    base_model: DifferentiableMicroscope,
    run_cfg: dict[str, Any],
    split: str,
    device: torch.device,
    eval_m: float | None,
    *,
    crop_size: int = 256,
    crops_per_train_image: int = 4,
    base_batch: int = 8,
    seed: int = 42,
) -> PairCache:
    """Run the frozen base over deterministic crops and cache (x_base, gt) pairs.

    train: ``crops_per_train_image`` seeded-random ``crop_size`` crops per image.
    val/test: a single deterministic *center* crop per image (matches how the base
    metrics in ``swinir_results.csv`` were computed → exact base/base+SwinIR parity).
    """
    ds = build_dataset(run_cfg, split)
    images = ds.images  # list of [1,H,W] preprocessed (already [0,1]) tensors
    xb_list: list[torch.Tensor] = []
    gt_list: list[torch.Tensor] = []
    idx_list: list[int] = []
    crops: list[torch.Tensor] = []
    crop_src: list[int] = []
    n_crops = crops_per_train_image if split == "train" else 1
    for img_idx, img in enumerate(images):
        _, h, w = img.shape
        for k in range(n_crops):
            if split == "train":
                gen = torch.Generator().manual_seed(seed * 100003 + img_idx * 131 + k)
            else:
                gen = None
            top, left = _crop_positions(h, w, crop_size, gen)
            crops.append(img[:, top : top + crop_size, left : left + crop_size])
            crop_src.append(img_idx)
    # Batched base forward
    for i in range(0, len(crops), base_batch):
        chunk = torch.stack(crops[i : i + base_batch]).to(device)
        out = base_model(chunk, sigmoid_m=eval_m, apply_noise=False)
        xb = out["x_recon"].detach().float().cpu().to(torch.float16)
        gt = chunk.detach().float().cpu().to(torch.float16)
        xb_list.append(xb)
        gt_list.append(gt)
        idx_list.extend(crop_src[i : i + base_batch])
    return PairCache(
        x_base=torch.cat(xb_list, dim=0),
        gt=torch.cat(gt_list, dim=0),
        image_index=torch.tensor(idx_list, dtype=torch.long),
        split=split,
        crop_size=crop_size,
    )


def sample_patch_batch(
    cache: PairCache,
    batch: int,
    patch: int,
    device: torch.device,
    gen: torch.Generator,
    *,
    random_flips: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a paired 64x64 (input, target) batch from the 256x256 cache.

    Identical crop location + flips for input and target (no spatial mismatch,
    no independent normalisation)."""
    n = len(cache)
    idx = torch.randint(0, n, (batch,), generator=gen)
    S = cache.crop_size
    xs, ts = [], []
    for b in idx.tolist():
        top = int(torch.randint(0, S - patch + 1, (1,), generator=gen).item())
        left = int(torch.randint(0, S - patch + 1, (1,), generator=gen).item())
        x = cache.x_base[b : b + 1, :, top : top + patch, left : left + patch]
        t = cache.gt[b : b + 1, :, top : top + patch, left : left + patch]
        if random_flips:
            if torch.rand((), generator=gen).item() > 0.5:
                x, t = torch.flip(x, dims=[-1]), torch.flip(t, dims=[-1])
            if torch.rand((), generator=gen).item() > 0.5:
                x, t = torch.flip(x, dims=[-2]), torch.flip(t, dims=[-2])
        xs.append(x)
        ts.append(t)
    x = torch.cat(xs, 0).to(device, torch.float32)
    t = torch.cat(ts, 0).to(device, torch.float32)
    return x, t


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - ssim_metric(pred.clamp(0, 1), target)


def make_loss(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Build the loss stack for the requested mode.

    ``paper_pixel_perceptual_gan`` reuses the validated Table-2/Fig-7 loss stack
    (VGG19 perceptual + spectral-norm discriminator). It fails LOUDLY if the VGG
    weights are unavailable (no silent fallback to L1)."""
    mode = cfg["loss"]["mode"]
    lc = cfg["loss"]
    out: dict[str, Any] = {"mode": mode}
    if mode == "l1_only":
        return out
    if mode == "l1_ssim":
        out["l1_weight"] = float(lc.get("l1_weight", 1.0))
        out["ssim_weight"] = float(lc.get("ssim_weight", 1.0))
        return out
    # paper_pixel_perceptual_gan
    if not bool(lc.get("vgg_weights_available", True)):
        raise RuntimeError("paper_pixel_perceptual_gan requires VGG weights; set vgg_weights_available")
    stack = build_loss_stack(
        {
            "pixel_weight": float(lc.get("pixel_weight", 1.0)),
            "pixel_kind": lc.get("pixel_kind", "l1"),
            "perceptual_weight": float(lc.get("perceptual_weight", 1.0)),
            "gan_weight": float(lc.get("gan_weight", 0.1)),
            "gan_type": lc.get("gan_type", "vanilla"),
            "vgg_weights_available": True,
            "in_chans": 1,
        },
        device,
    )
    if "perceptual" not in stack:
        raise RuntimeError("perceptual loss failed to build (VGG missing?) — refusing silent fallback")
    if "discriminator" not in stack:
        raise RuntimeError("discriminator failed to build — refusing silent fallback")
    # Sanity: force the VGG forward once so a missing-weights failure is loud NOW.
    with torch.no_grad():
        probe = torch.rand(1, 1, 64, 64, device=device)
        _ = stack["perceptual"](probe, probe)
    out["stack"] = stack
    return out


def compute_generator_loss(
    loss: dict[str, Any],
    rec: torch.Tensor,
    gt: torch.Tensor,
    disc: nn.Module | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    mode = loss["mode"]
    comps: dict[str, float] = {}
    if mode == "l1_only":
        l = F.l1_loss(rec, gt)
        comps["l1"] = float(l.item())
        return l, comps
    if mode == "l1_ssim":
        l1 = loss["l1_weight"] * F.l1_loss(rec, gt)
        ls = loss["ssim_weight"] * ssim_loss(rec, gt)
        comps["l1"] = float(l1.item())
        comps["ssim"] = float(ls.item())
        return l1 + ls, comps
    # paper_pixel_perceptual_gan
    stack = loss["stack"]
    g = stack["pixel_weight"] * pixel_loss(rec, gt, stack["pixel_kind"])
    comps["pixel"] = float(g.item())
    if "perceptual" in stack:
        gp = stack["perceptual_weight"] * stack["perceptual"](rec, gt)
        g = g + gp
        comps["perceptual"] = float(gp.item())
    if disc is not None:
        ga = stack["gan_weight"] * stack["gan_loss"](disc(rec), True)
        g = g + ga
        comps["adv"] = float(ga.item())
    return g, comps


# ---------------------------------------------------------------------------
# Evaluation (full 256x256, exactly matching the base eval protocol)
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_cache(refiner: OfflineSwinIRRefinement, cache: PairCache, device: torch.device, eval_batch: int = 16) -> dict[str, float]:
    refiner.eval()
    base_mse = base_ssim = base_psnr = 0.0
    ref_mse = ref_ssim = ref_psnr = 0.0
    n = 0
    for i in range(0, len(cache), eval_batch):
        xb = cache.x_base[i : i + eval_batch].to(device, torch.float32)
        gt = cache.gt[i : i + eval_batch].to(device, torch.float32)
        rec = refiner(xb).clamp(0, 1)
        xbc = xb.clamp(0, 1)
        for j in range(xb.shape[0]):
            t = gt[j : j + 1]
            base_mse += float(mse_metric(xbc[j : j + 1], t).item())
            ref_mse += float(mse_metric(rec[j : j + 1], t).item())
            base_ssim += float(ssim_metric(xbc[j : j + 1], t).item())
            ref_ssim += float(ssim_metric(rec[j : j + 1], t).item())
            base_psnr += float(psnr_metric(xbc[j : j + 1], t).item())
            ref_psnr += float(psnr_metric(rec[j : j + 1], t).item())
            n += 1
    m = max(1, n)
    return {
        "n": n,
        "base_mse": base_mse / m, "base_ssim": base_ssim / m, "base_psnr": base_psnr / m,
        "ref_mse": ref_mse / m, "ref_ssim": ref_ssim / m, "ref_psnr": ref_psnr / m,
    }


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def lr_at(step: int, *, base_lr: float, steps: int, warmup: int, min_frac: float) -> float:
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_frac + (1.0 - min_frac) * cos)
