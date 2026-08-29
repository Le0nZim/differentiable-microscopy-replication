"""AM-4 paper-faithful SwinIR Table-2 pipeline helpers.

This module is the reusable, test-covered core for the AM-4 rerun
(`scripts/table02_swinir_sr/run.py` and `scripts/table02_swinir_sr/audit_fairness.py`).

Design goals (vs. the frozen `swinir_table2_full` run):
- SwinIR-M capacity (embed_dim 180) driven entirely by config.
- Effective batch size 32 via gradient accumulation (micro_batch * grad_accum).
- bf16 autocast (no GradScaler — the illumination weights are complex, which
  GradScaler cannot unscale).
- Fair, deterministic evaluation over ALL non-overlapping 64x64 tiles of ALL
  test images (no center-bias, no cap) for non-smoke runs.
- LI and w/o-LI share an identical inverse architecture / losses / optimizer
  settings / data / eval; the only intended difference is the illumination mode.

Nothing here mutates the frozen `table2_pipeline.SwinIRTable2Model`; it is reused
unchanged so the forward physics stays identical to AM-1/2/3.
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from baselines.swinir.table2_pipeline import SwinIRTable2Model
from evaluation.metrics import psnr as psnr_metric
from evaluation.metrics import ssim as ssim_metric

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Config loading / validation
# ---------------------------------------------------------------------------
def load_am4_config(path: str | Path) -> dict[str, Any]:
    """Load + lightly validate an AM-4 Table-2 config."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    for key in ("experiment", "data", "model", "training", "eval"):
        if key not in cfg:
            raise ValueError(f"AM-4 config missing top-level '{key}' section")
    tr = cfg["training"]
    eff = int(tr["effective_batch_size"])
    micro = int(tr["micro_batch_size"])
    accum = int(tr["grad_accum"])
    if micro * accum != eff:
        raise ValueError(
            f"micro_batch_size*grad_accum ({micro}*{accum}={micro*accum}) "
            f"!= effective_batch_size ({eff})"
        )
    if "deviations_from_paper" not in cfg:
        raise ValueError("AM-4 config must include an explicit 'deviations_from_paper' list")
    sw = cfg["model"]["swinir"]
    if sw.get("upscale", 1) != 1:
        raise ValueError("paper sets SwinIR upscale=1 (no upscaling); refuse mismatch")
    # x16 compression sanity: downscale^2 / num_patterns == compression
    m = cfg["model"]
    d = int(m["downscale_factor"])
    t = int(m["num_patterns"])
    comp = int(m["compression"])
    if d * d // t != comp:
        raise ValueError(f"compression mismatch: d^2/T = {d*d//t} != {comp}")


def resolve_grad_accum(cfg: dict[str, Any]) -> tuple[int, int, int]:
    """Return (micro_batch, grad_accum, effective_batch)."""
    tr = cfg["training"]
    return int(tr["micro_batch_size"]), int(tr["grad_accum"]), int(tr["effective_batch_size"])


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_model_config(cfg: dict[str, Any], *, learnable: bool) -> dict[str, Any]:
    """Translate an AM-4 config + condition into a SwinIRTable2Model config dict."""
    m = cfg["model"]
    seed = int(cfg["experiment"]["seed"])
    tr = cfg["training"]
    return {
        "image_size": int(m["image_size"]),
        "pattern_generator": {
            "mode": "learnable_frequency" if learnable else "random_fixed",
            "num_patterns": int(m["num_patterns"]),
            "sigmoid_m": float(tr.get("train_sigmoid_m", 8.0)),
            "random_fixed_m": float(tr.get("random_fixed_m", 10.0)),
            "seed": seed,
        },
        "forward_model": {
            "downscale_factor": int(m["downscale_factor"]),
            "use_impulse_psfs": True,
        },
        "detector_noise": {"mode": "noise_free", "apply_noise": False},
        "inverse_model": {
            "upsampling": {
                "mode": "locality_aware",
                "downscale_factor": int(m["downscale_factor"]),
                "num_patterns": int(m["num_patterns"]),
            }
        },
        "swinir": dict(m["swinir"]),
    }


def build_model(cfg: dict[str, Any], *, learnable: bool, seed: int | None = None) -> SwinIRTable2Model:
    """Build a SwinIRTable2Model for a condition, with a deterministic init seed."""
    if seed is None:
        seed = int(cfg["experiment"]["seed"])
    torch.manual_seed(seed)
    model = SwinIRTable2Model(build_model_config(cfg, learnable=learnable))
    # Lazily initialise PSF buffers so param/state layout is final.
    ps = int(cfg["data"]["patch_size"])
    model(torch.zeros(1, 1, ps, ps))
    return model


def model_arch_summary(model: SwinIRTable2Model) -> dict[str, Any]:
    """Architecture + parameter-count summary used by tests and audits."""
    swinir = model.swinir
    illum = model.illumination_parameters()
    return {
        "embed_dim": int(swinir.embed_dim) if hasattr(swinir, "embed_dim") else None,
        "num_layers": len(swinir.layers) if hasattr(swinir, "layers") else None,
        "window_size": int(swinir.window_size) if hasattr(swinir, "window_size") else None,
        "upscale": int(swinir.upscale) if hasattr(swinir, "upscale") else None,
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "swinir_params": int(sum(p.numel() for p in swinir.parameters())),
        "upsampling_params": int(sum(p.numel() for p in model.upsampling.parameters())),
        "fuse_params": int(sum(p.numel() for p in model.fuse.parameters())),
        "illumination_params": int(sum(p.numel() for p in illum)),
        "illumination_learnable": model.pattern_generator.patterns_are_learnable(),
    }


# ---------------------------------------------------------------------------
# Deterministic tiling + fair evaluation
# ---------------------------------------------------------------------------
def tile_coords(
    n_h: int,
    n_w: int,
    *,
    selection: str = "all",
    max_tiles: int | None = None,
) -> list[tuple[int, int]]:
    """Deterministic list of (ti, tj) tile indices.

    selection="all": row-major over every non-overlapping tile (fair).
    selection="center_biased": nearest-to-center first (budget proxy ONLY).
    """
    coords = [(ti, tj) for ti in range(n_h) for tj in range(n_w)]
    if selection == "center_biased":
        ci, cj = (n_h - 1) / 2.0, (n_w - 1) / 2.0
        coords.sort(key=lambda c: ((c[0] - ci) ** 2 + (c[1] - cj) ** 2, c[0], c[1]))
    elif selection != "all":
        raise ValueError(f"unknown tile selection: {selection}")
    if max_tiles is not None:
        coords = coords[:max_tiles]
    return coords


def _load_gray(path: Path) -> torch.Tensor:
    from torchvision.io import read_image

    img = read_image(str(path)).float() / 255.0
    if img.shape[0] == 3:
        img = img.mean(dim=0, keepdim=True)
    elif img.shape[0] == 4:
        img = img[:3].mean(dim=0, keepdim=True)
    return img


def list_test_images(root: Path, *, max_images: int | None = None) -> list[Path]:
    paths = sorted(root.glob("*.png")) + sorted(root.glob("*.jpg"))
    if max_images is not None:
        paths = paths[:max_images]
    return paths


@torch.no_grad()
def eval_dataset_fair(
    model: SwinIRTable2Model,
    root: Path,
    *,
    patch_size: int,
    device: torch.device,
    learnable: bool,
    eval_sigmoid_m: float,
    selection: str = "all",
    max_tiles_per_image: int | None = None,
    max_images: int | None = None,
    eval_batch: int = 64,
    amp_dtype: torch.dtype | None = None,
    compute_stitched: bool = True,
) -> dict[str, Any]:
    """Evaluate PSNR/SSIM over deterministic tiles of every test image.

    Returns per-tile-averaged PSNR/SSIM (primary) and optionally stitched
    full-image PSNR/SSIM (secondary cross-check).
    """
    model.eval()
    paths = list_test_images(root, max_images=max_images)
    psnr_sum = ssim_sum = 0.0
    n_tiles = 0
    n_images = 0
    stitched_psnr_sum = stitched_ssim_sum = 0.0
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None and device.type == "cuda"
        else _nullcontext()
    )
    for p in paths:
        img = _load_gray(p)
        _, h, w = img.shape
        n_h, n_w = h // patch_size, w // patch_size
        if n_h == 0 or n_w == 0:
            continue
        img = img[:, : n_h * patch_size, : n_w * patch_size]
        coords = tile_coords(n_h, n_w, selection=selection, max_tiles=max_tiles_per_image)
        tiles = [
            img[:, ti * patch_size : (ti + 1) * patch_size, tj * patch_size : (tj + 1) * patch_size]
            for ti, tj in coords
        ]
        batch = torch.stack(tiles).to(device)
        canvas = torch.zeros_like(img, device=device) if compute_stitched else None
        for i in range(0, batch.shape[0], eval_batch):
            chunk = batch[i : i + eval_batch]
            with autocast_ctx:
                out = model(chunk, sigmoid_m=eval_sigmoid_m if learnable else None, apply_noise=False)
            rec = out["x_recon"].float().clamp(0, 1)
            for j in range(chunk.shape[0]):
                tgt = chunk[j : j + 1].float()
                psnr_sum += float(psnr_metric(rec[j : j + 1], tgt).item())
                ssim_sum += float(ssim_metric(rec[j : j + 1], tgt).item())
                n_tiles += 1
                if canvas is not None:
                    ti, tj = coords[i + j]
                    canvas[
                        :, ti * patch_size : (ti + 1) * patch_size, tj * patch_size : (tj + 1) * patch_size
                    ] = rec[j]
        if canvas is not None:
            gt_full = img.to(device).unsqueeze(0)
            rec_full = canvas.unsqueeze(0)
            stitched_psnr_sum += float(psnr_metric(rec_full, gt_full).item())
            stitched_ssim_sum += float(ssim_metric(rec_full, gt_full).item())
        n_images += 1
    out = {
        "psnr": psnr_sum / max(1, n_tiles),
        "ssim": ssim_sum / max(1, n_tiles),
        "tiles": n_tiles,
        "images": n_images,
        "selection": selection,
        "max_tiles_per_image": max_tiles_per_image,
    }
    if compute_stitched and n_images > 0:
        out["stitched_psnr"] = stitched_psnr_sum / n_images
        out["stitched_ssim"] = stitched_ssim_sum / n_images
    return out


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# Optimizer construction (fairness-critical: identical for shared params)
# ---------------------------------------------------------------------------
def build_optimizers(
    model: SwinIRTable2Model,
    cfg: dict[str, Any],
    *,
    learnable: bool,
    discriminator: nn.Module | None = None,
) -> dict[str, Any]:
    """Build the generator (+ optional discriminator) optimizers.

    The shared-parameter group (upsampling+fuse+SwinIR) uses identical settings
    regardless of condition. The illumination group (illum_lr) is present only
    for the learnable condition (w/o-LI has no Ht parameters by construction).
    """
    tr = cfg["training"]
    betas = tuple(tr.get("betas", [0.9, 0.99]))
    groups = [{"params": model.swinir_parameters(), "lr": float(tr["swinir_lr"]), "name": "inverse"}]
    if learnable:
        groups.append(
            {"params": model.illumination_parameters(), "lr": float(tr["illumination_lr"]), "name": "illumination"}
        )
    opt_g = torch.optim.Adam(groups, betas=betas)
    out = {"opt_g": opt_g, "betas": betas}
    if discriminator is not None:
        out["opt_d"] = torch.optim.Adam(discriminator.parameters(), lr=float(tr["disc_lr"]), betas=betas)
    return out


def optimizer_group_summary(opt: torch.optim.Optimizer) -> list[dict[str, Any]]:
    """Summarise per-group lr / betas for fairness comparison."""
    summary = []
    for g in opt.param_groups:
        summary.append(
            {
                "name": g.get("name", "?"),
                "lr": float(g["lr"]),
                "betas": list(g.get("betas", ())),
                "num_params": int(sum(p.numel() for p in g["params"])),
            }
        )
    return summary


def amp_dtype_from_cfg(cfg: dict[str, Any]) -> torch.dtype | None:
    name = str(cfg["training"].get("amp_dtype", "none")).lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    return None


# ---------------------------------------------------------------------------
# Train/val data (shared identically by both conditions)
# ---------------------------------------------------------------------------
class PathListSRDataset(torch.utils.data.Dataset):
    """Grayscale SR patch dataset over an explicit list of image paths."""

    def __init__(
        self,
        paths: list[Path],
        *,
        patch_size: int = 64,
        grayscale: bool = True,
        random_crops: bool = True,
        seed: int = 42,
    ) -> None:
        if not paths:
            raise FileNotFoundError("PathListSRDataset received an empty path list")
        self.paths = list(paths)
        self.patch_size = patch_size
        self.grayscale = grayscale
        self.random_crops = random_crops
        self.seed = seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        img = _load_gray(self.paths[index]) if self.grayscale else _load_any(self.paths[index])
        _, h, w = img.shape
        ps = self.patch_size
        gen = None if self.random_crops else torch.Generator().manual_seed(self.seed + index)
        if h >= ps and w >= ps:
            top = int(torch.randint(0, h - ps + 1, (1,), generator=gen).item())
            left = int(torch.randint(0, w - ps + 1, (1,), generator=gen).item())
            img = img[:, top : top + ps, left : left + ps]
        else:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(ps, ps), mode="bilinear", align_corners=False
            ).squeeze(0)
        return img


def _load_any(path: Path) -> torch.Tensor:
    from torchvision.io import read_image

    return read_image(str(path)).float() / 255.0


_LR_SUFFIX = re.compile(r"x[234]\.(png|jpe?g)$", re.IGNORECASE)


def _is_hr_filename(path: Path) -> bool:
    """True if the filename is not a bicubic x2/x3/x4 sibling."""
    return _LR_SUFFIX.search(path.name) is None


def _scene_id(path: Path) -> str:
    """Stable scene key: Flickr 6-digit ID, DIV2K 4-digit ID, else filename."""
    name = path.stem
    flickr = re.match(r"^(\d{6})(?:x[234])?$", name)
    if flickr:
        return f"flickr:{flickr.group(1)}"
    div2k = re.match(r"^(\d{4})$", name)
    if div2k:
        return f"div2k:{div2k.group(1)}"
    return f"file:{path.name}"


def gather_train_paths(cfg: dict[str, Any]) -> list[Path]:
    """Deterministic, sorted list of all training image paths across roots."""
    paths: list[Path] = []
    hr_only = bool(cfg["data"].get("hr_only", False))
    for rel in cfg["data"]["train_roots"]:
        root = REPO_ROOT / rel
        found = sorted(root.glob("*.png")) + sorted(root.glob("*.jpg"))
        if hr_only:
            found = [p for p in found if _is_hr_filename(p)]
        paths.extend(found)
    paths = sorted(paths)
    max_train = cfg["data"].get("max_train_images")
    if max_train is not None:
        paths = paths[: int(max_train)]
    return paths


def _split_train_val_by_scene(paths: list[Path], cfg: dict[str, Any]) -> tuple[list[Path], list[Path]]:
    """Hold out entire scenes (all scales of a Flickr/DIV2K ID) for val."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[_scene_id(path)].append(path)
    scene_ids = sorted(groups)
    val_fraction = float(cfg["data"].get("val_fraction", 0.0))
    seed = int(cfg["experiment"]["seed"])
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(scene_ids), generator=gen).tolist()
    n_val = int(round(len(scene_ids) * val_fraction))
    n_val = max(0, min(n_val, len(scene_ids) - 1))
    val_ids = {scene_ids[i] for i in perm[:n_val]}
    train_paths: list[Path] = []
    val_paths: list[Path] = []
    for sid in scene_ids:
        dest = val_paths if sid in val_ids else train_paths
        dest.extend(sorted(groups[sid]))
    train_paths = sorted(train_paths)
    val_paths = sorted(val_paths)
    if not val_paths:
        val_paths = train_paths[:1]
    return train_paths, val_paths


def split_train_val(cfg: dict[str, Any]) -> tuple[list[Path], list[Path]]:
    """Deterministically split train paths into (train, val).

    Independent of model RNG so both conditions see identical splits/order.
    When ``data.split_by_scene`` is true, every scale of a Flickr/DIV2K ID
    stays on the same side of the split.
    """
    paths = gather_train_paths(cfg)
    if bool(cfg["data"].get("split_by_scene", False)):
        return _split_train_val_by_scene(paths, cfg)
    val_fraction = float(cfg["data"].get("val_fraction", 0.0))
    seed = int(cfg["experiment"]["seed"])
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(paths), generator=gen).tolist()
    n_val = int(round(len(paths) * val_fraction))
    n_val = max(0, min(n_val, len(paths) - 1))
    val_idx = set(perm[:n_val])
    train_paths = [paths[i] for i in range(len(paths)) if i not in val_idx]
    val_paths = [paths[i] for i in range(len(paths)) if i in val_idx]
    if not val_paths:  # tiny configs: reuse one train image so val never crashes
        val_paths = train_paths[:1]
    return train_paths, val_paths
