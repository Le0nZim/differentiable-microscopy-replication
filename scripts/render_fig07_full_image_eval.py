#!/usr/bin/env python3
"""Figure-7 FULL-IMAGE qualitative outputs for the AM-4 SwinIR Table-2 run.

For each benchmark dataset (Set5, Set14, BSD100, Urban100, Manga109) this:
  1. Loads a representative full-resolution test image (first image, sorted).
  2. GENUINELY ATTEMPTS direct full-image inference (single forward pass on the
     whole HxW image). This is expected to fail because the trained model's
     illumination patterns (H_t) and locality-aware upsampling are built for a
     fixed 64x64 input (see configs/swinir/am4_table2_full.yaml line 100 and
     models/locality_upsampling.py / models/forward_model.py). The exact error
     is captured and recorded in metadata.
  3. Falls back to TILED full-image inference: reflection-pad the image up to a
     multiple of 64, run every non-overlapping 64x64 tile through the model,
     stitch the reconstructions back together, and crop to the original size.
     This yields a true full-resolution, full-image reconstruction.
  4. If tiling itself fails (e.g. an image smaller than one tile), falls back to
     the largest centered multiple-of-64 crop and records fallback_used=True.

Outputs (full_image_eval/):
  <dataset>_GT.png / <dataset>_woLI.png / <dataset>_withLI.png   (full-res)
  fig07_full_image_qualitative.png                              (combined)
  metadata.json                                                 (provenance)

IMPORTANT: This is OUR reconstruction pipeline rendered at full resolution. It
is NOT claimed to be the exact procedure used to render the paper's Figure 7
(the paper does not document its crop/stitch/upscale choices in this repo).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --- repo wiring -----------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
REPL = ROOT
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.swinir import am4_table2 as A          # noqa: E402
from evaluation.metrics import psnr as psnr_metric    # noqa: E402
from evaluation.metrics import ssim as ssim_metric    # noqa: E402

CONFIG = ROOT / "configs/swinir/am4_table2_full_fixed.yaml"
RUN    = ROOT / "experiments/swinir_or_highres/am4_swinir_table2_resolution_fixed/full"
OUT    = RUN / "full_image_eval"

DATASETS = ["Set5", "Set14", "BSD100", "Urban100", "Manga109"]

CONDITIONS = [
    # (key, learnable, checkpoint, out-suffix, column title)
    ("swinir_wo_li",   False, RUN / "swinir_wo_li/checkpoints/best.pt",   "woLI",   "SwinIR w/o LI"),
    ("swinir_with_li", True,  RUN / "swinir_with_li/checkpoints/best.pt", "withLI", "SwinIR with LI"),
]


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        best_idx, best_free = 0, -1
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_idx, best_free = i, free
        return torch.device(f"cuda:{best_idx}")
    return torch.device("cpu")


def load_model(cfg: dict[str, Any], *, learnable: bool, ckpt: Path, device: torch.device):
    model = A.build_model(cfg, learnable=learnable)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


@torch.no_grad()
def try_direct_full_image(model, img: torch.Tensor, device, learnable, eval_m, amp_dtype):
    """Attempt a single forward pass on the WHOLE image. Returns (ok, recon|None, err)."""
    try:
        x = img.unsqueeze(0).to(device)
        ctx = (torch.autocast(device.type, dtype=amp_dtype)
               if amp_dtype is not None and device.type == "cuda" else _null())
        with ctx:
            out = model(x, sigmoid_m=eval_m if learnable else None, apply_noise=False)
        return True, out["x_recon"].float().clamp(0, 1).squeeze(0).cpu(), None
    except Exception as exc:  # noqa: BLE001 — we want the exact reason recorded
        return False, None, f"{type(exc).__name__}: {exc}"


@torch.no_grad()
def tiled_full_image(model, img: torch.Tensor, device, learnable, eval_m, amp_dtype,
                     *, tile: int = 64, eval_batch: int = 64) -> tuple[torch.Tensor, dict]:
    """Reflection-pad to a multiple of `tile`, run every 64x64 tile, stitch, crop back."""
    _, h, w = img.shape
    pad_h = (-h) % tile
    pad_w = (-w) % tile
    # reflect padding keeps edge content natural; falls back to replicate if too large
    mode = "reflect" if (pad_h < h and pad_w < w) else "replicate"
    padded = F.pad(img.unsqueeze(0), (0, pad_w, 0, pad_h), mode=mode).squeeze(0)
    _, ph, pw = padded.shape
    n_h, n_w = ph // tile, pw // tile

    coords = [(ti, tj) for ti in range(n_h) for tj in range(n_w)]
    tiles = torch.stack([
        padded[:, ti * tile:(ti + 1) * tile, tj * tile:(tj + 1) * tile] for ti, tj in coords
    ]).to(device)

    canvas = torch.zeros(1, ph, pw, device=device)
    ctx = (torch.autocast(device.type, dtype=amp_dtype)
           if amp_dtype is not None and device.type == "cuda" else _null())
    for i in range(0, tiles.shape[0], eval_batch):
        chunk = tiles[i:i + eval_batch]
        with ctx:
            out = model(chunk, sigmoid_m=eval_m if learnable else None, apply_noise=False)
        rec = out["x_recon"].float().clamp(0, 1)
        for j in range(chunk.shape[0]):
            ti, tj = coords[i + j]
            canvas[:, ti * tile:(ti + 1) * tile, tj * tile:(tj + 1) * tile] = rec[j]

    recon = canvas[:, :h, :w].cpu()
    info = {"tile_size": tile, "padded_size": [ph, pw],
            "n_tiles": {"rows": n_h, "cols": n_w, "total": len(coords)},
            "pad_mode": mode, "pad_h": pad_h, "pad_w": pad_w}
    return recon, info


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def save_gray(t: torch.Tensor, path: Path) -> None:
    arr = (t.squeeze(0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def metric_pair(recon: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    r = recon.unsqueeze(0)
    g = gt.unsqueeze(0)
    return {"psnr": round(float(psnr_metric(r, g).item()), 4),
            "ssim": round(float(ssim_metric(r, g).item()), 4)}


def build_combined_figure(meta: dict[str, Any]) -> None:
    """Compose a paper-ready GT / w/o LI / with LI grid (rows = datasets)."""
    datasets = list(meta["datasets"].keys())
    col_titles = ["Ground Truth", "SwinIR w/o LI", "SwinIR with LI"]
    suffixes   = ["GT", "woLI", "withLI"]
    n = len(datasets)

    fig, axes = plt.subplots(n, 3, figsize=(9.5, 3.1 * n))
    if n == 1:
        axes = axes.reshape(1, 3)

    for r, ds in enumerate(datasets):
        entry = meta["datasets"][ds]
        for c, (suf, title) in enumerate(zip(suffixes, col_titles)):
            ax = axes[r, c]
            img = np.array(Image.open(OUT / f"{ds}_{suf}.png").convert("L"))
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
            if r == 0:
                ax.set_title(title, fontsize=13, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{ds}\n({entry['source_file']})", fontsize=10)
            else:
                key = "swinir_wo_li" if c == 1 else "swinir_with_li"
                m = entry["conditions"][key]["metrics_vs_gt"]
                ax.set_xlabel(f"PSNR {m['psnr']:.2f} / SSIM {m['ssim']:.3f}", fontsize=9)

    fig.suptitle("Figure 7 (full-image, tiled 64×64 reconstruction) — SwinIR ×16 compression",
                 fontsize=14, fontweight="bold", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path = OUT / "fig07_full_image_qualitative.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined figure → {out_path}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = A.load_am4_config(CONFIG)
    device = pick_device()
    amp_dtype = A.amp_dtype_from_cfg(cfg)
    eval_m = float(cfg["training"].get("eval_sigmoid_m", 8.0))
    print(f"device={device} amp_dtype={amp_dtype}", flush=True)

    models = {key: load_model(cfg, learnable=learn, ckpt=ck, device=device)
              for key, learn, ck, _, _ in CONDITIONS}

    meta: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(CONFIG.relative_to(ROOT)),
        "run": str(RUN.relative_to(ROOT)),
        "device": str(device),
        "checkpoint": "best.pt (best-val) per condition",
        "note": ("OUR reconstruction pipeline at full resolution; NOT a verified "
                 "reproduction of the paper's exact Figure-7 rendering procedure."),
        "datasets": {},
    }

    for ds in DATASETS:
        rel = cfg["data"]["test_roots"][ds]
        paths = A.list_test_images(REPL / rel)
        if not paths:
            print(f"[skip] {ds}: no images", flush=True)
            continue
        src = paths[0]
        gt = A._load_gray(src)                  # [1,H,W] in [0,1]
        _, H, W = gt.shape

        entry: dict[str, Any] = {
            "source_file": src.name,
            "source_relpath": str(src.relative_to(REPL)),
            "original_size_hw": [H, W],
            "conditions": {},
        }

        # GT (full resolution)
        save_gray(gt, OUT / f"{ds}_GT.png")

        for key, learnable, _ck, suffix, _title in CONDITIONS:
            model = models[key]
            ok, _direct, err = try_direct_full_image(model, gt, device, learnable, eval_m, amp_dtype)
            cond_meta: dict[str, Any] = {
                "direct_full_image": {"attempted": True, "succeeded": ok, "error": err},
            }

            if ok:
                recon, target = _direct, gt
                cond_meta["method"] = "direct_full_image"
                cond_meta["fallback_used"] = False
            else:
                try:
                    recon, tinfo = tiled_full_image(model, gt, device, learnable, eval_m, amp_dtype)
                    target = gt
                    cond_meta["method"] = "tiled_stitched_64x64"
                    cond_meta["fallback_used"] = False
                    cond_meta.update(tinfo)
                except Exception as exc:  # noqa: BLE001 — last-resort crop fallback
                    side_h = max(64, (H // 64) * 64)
                    side_w = max(64, (W // 64) * 64)
                    top = (H - side_h) // 2
                    left = (W - side_w) // 2
                    crop = gt[:, top:top + side_h, left:left + side_w]
                    recon, tinfo = tiled_full_image(model, crop, device, learnable, eval_m, amp_dtype)
                    target = crop
                    cond_meta["method"] = "tiled_stitched_64x64_on_center_crop"
                    cond_meta["fallback_used"] = True
                    cond_meta["fallback_reason"] = f"{type(exc).__name__}: {exc}"
                    cond_meta["crop_box_tlhw"] = [top, left, side_h, side_w]
                    cond_meta.update(tinfo)
                    save_gray(crop, OUT / f"{ds}_GT.png")  # keep GT aligned with recon
                    entry["original_size_hw"] = [side_h, side_w]

            cond_meta["metrics_vs_gt"] = metric_pair(recon, target)
            save_gray(recon, OUT / f"{ds}_{suffix}.png")
            entry["conditions"][key] = cond_meta
            m = cond_meta["metrics_vs_gt"]
            print(f"[{ds}/{key}] method={cond_meta['method']} "
                  f"PSNR={m['psnr']} SSIM={m['ssim']}", flush=True)

        meta["datasets"][ds] = entry

    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote metadata → {OUT / 'metadata.json'}", flush=True)

    build_combined_figure(meta)


if __name__ == "__main__":
    main()
