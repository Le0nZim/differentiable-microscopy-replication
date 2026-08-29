#!/usr/bin/env python3
"""Regenerate paper Fig. 9 (HumanMCF7 x16): Ground Truth / wSwinIR / wCNN (+ learned H_t).

Key fix vs the legacy report: the wide reconstruction is stitched with OVERLAP-ADD tiling
(Hann window + reflect padding) instead of naive non-overlapping tiles, which removes the
64-px (wCNN) / 256-px (wSwinIR) seam grid that dominated the previous Fig. 9. Also saves a
naive-vs-overlap comparison so the artifact source is explicit.

Consistent display normalization: GT / wSwinIR / wCNN share one viridis lo/hi (from GT).

Usage:
  python scripts/figure08_mcf7/reproduce_fig9.py --config configs/figure08_mcf7/reproduce_fig9_tubulin.yaml --device cuda:0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
try:
    _VIRIDIS = matplotlib.colormaps["viridis"]
except Exception:  # pragma: no cover
    import matplotlib.cm as _cm
    _VIRIDIS = _cm.get_cmap("viridis")

from datasets.mcf7_channel2 import MCF7Channel2Config, MCF7Channel2Dataset, _load_tiff, _preprocess
from evaluation.metrics import mse as mse_metric
from evaluation.metrics import psnr as psnr_metric
from evaluation.metrics import ssim as ssim_metric
from evaluation.tiled_inference import naive_tiled_recon, overlap_tiled_recon
from utils.device import resolve_device

_spec = importlib.util.spec_from_file_location(
    "fig89_train", ROOT / "scripts" / "fig89_mcf7_swinir_fix_train.py")
TR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TR)

GREEN, RED = (40, 200, 90), (220, 60, 60)


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _font(size: int):
    for cand in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if Path(cand).exists():
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


def _to_viridis(gray, lo, hi):
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    return (_VIRIDIS(norm)[..., :3] * 255.0 + 0.5).astype(np.uint8)


def _frame(img, color, width):
    d = ImageDraw.Draw(img)
    for k in range(width):
        d.rectangle([k, k, img.width - 1 - k, img.height - 1 - k], outline=color)


def _load_model(condition, cfg, runs_dir, device):
    backbone, isz, up_mode, _ = TR.CONDITIONS[condition]
    model = (TR._build_swinir_model(cfg, isz) if backbone == "swinir"
             else TR._build_conventional_model(cfg, isz, up_mode)).to(device)
    model(torch.zeros(1, 1, isz, isz, device=device))
    ckpt = runs_dir / condition / "checkpoints" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint for {condition}: {ckpt}")
    ck = torch.load(ckpt, map_location=device)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    if not any(k.startswith("pattern_generator") for k in state):
        print(f"WARNING: {ckpt} has NO pattern_generator state (H_t not restored!)")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, isz


def _metrics(a, b):
    ta = torch.from_numpy(np.ascontiguousarray(a))[None, None]
    tb = torch.from_numpy(np.ascontiguousarray(b))[None, None]
    return {"psnr": float(psnr_metric(ta, tb).item()), "ssim": float(ssim_metric(ta, tb).item()),
            "mse": float(mse_metric(ta, tb).item())}


def _pattern_grid(patterns, tile_px, gap, color):
    soft = patterns.squeeze(1).clamp(0, 1).numpy()
    Tn = soft.shape[0]
    cols = 2
    rows = (Tn + cols - 1) // cols
    W = cols * tile_px + (cols - 1) * gap
    H = rows * tile_px + (rows - 1) * gap
    canvas = np.full((H, W, 3), color, dtype=np.uint8)
    for t in range(Tn):
        r, c = divmod(t, cols)
        small = np.array(Image.fromarray((soft[t] * 255).astype(np.uint8)).resize((tile_px, tile_px), Image.BILINEAR))
        binar = np.where(small > 127, 255, 0).astype(np.uint8)
        canvas[r * (tile_px + gap):r * (tile_px + gap) + tile_px,
               c * (tile_px + gap):c * (tile_px + gap) + tile_px] = np.array(Image.fromarray(binar).convert("RGB"))
    return Image.fromarray(canvas)


def _compose_rows(rows, row_w, lo, hi):
    row_h = max(1, round(row_w * rows[0][1].shape[0] / rows[0][1].shape[1]))
    gap = 6
    n = len(rows)
    canvas = np.full((n * row_h + (n - 1) * gap, row_w, 3), 255, dtype=np.uint8)
    lf = _font(max(20, row_h // 7))
    for r, (label, arr, color) in enumerate(rows):
        img = Image.fromarray(_to_viridis(arr, lo, hi)).resize((row_w, row_h), Image.LANCZOS)
        if color is not None:
            _frame(img, color, max(3, row_h // 60))
        d = ImageDraw.Draw(img)
        tw = d.textlength(label, font=lf)
        bx, by = row_w - tw - 22, 10
        d.rectangle([bx - 8, by - 4, bx + tw + 8, by + lf.size + 6], fill=(0, 0, 0))
        d.text((bx, by), label, fill=(255, 255, 255), font=lf)
        canvas[r * (row_h + gap):r * (row_h + gap) + row_h, :] = np.array(img)
    return canvas, row_h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--runs-dir", default=None, help="override reproduce.runs_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    rep = cfg.get("reproduce", {})
    runs_dir = ROOT / (args.runs_dir or rep.get("runs_dir", "experiments/figure08_mcf7/runs"))
    conds = rep.get("conditions", ["wswinir", "wcnn64"])
    eval_m = float(rep.get("eval_sigmoid_m", cfg["training"].get("eval_sigmoid_m", 8.0)))
    overlap_frac = float(rep.get("tile_overlap_frac", 0.25))
    device = resolve_device(args.device)
    out_dir = Path(args.out) if args.out else ROOT / "results" / "reproduced_figures" / cfg["experiment"]["tag"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | channel: {cfg['dataset'].get('bbbc021_channel')} | conditions: {conds}")

    swin, _ = _load_model(conds[0], cfg, runs_dir, device)
    cnn, _ = _load_model(conds[1], cfg, runs_dir, device)

    ds_cfg = dict(cfg["dataset"]); ds_cfg["seed"] = 42; ds_cfg["patch_size"] = 256; ds_cfg["image_size"] = 256
    ds = MCF7Channel2Dataset.from_dict(ds_cfg, split="test")
    src_index = int(rep.get("fig9_src_index", 0))
    top, left = int(rep.get("fig9_top", 384)), int(rep.get("fig9_left", 0))
    height, width = int(rep.get("fig9_height", 256)), int(rep.get("fig9_width", 1280))
    src_path = Path(ds.specs[src_index][0])
    full = _preprocess(_load_tiff(src_path), MCF7Channel2Config.from_dict(ds_cfg))
    _, Hs, Ws = full.shape
    top = min(top, max(0, Hs - height)); left = min(left, max(0, Ws - width))
    field = full[:, top:top + height, left:left + width].unsqueeze(0)
    gt = field[0, 0].numpy()

    ov_swin = max(16, int(256 * overlap_frac))
    ov_cnn = max(16, int(64 * overlap_frac))
    rec_swin = overlap_tiled_recon(swin, field, device, eval_m, 256, ov_swin)
    rec_cnn = overlap_tiled_recon(cnn, field, device, eval_m, 64, ov_cnn)
    naive_swin = naive_tiled_recon(swin, field, device, eval_m, 256)
    naive_cnn = naive_tiled_recon(cnn, field, device, eval_m, 64)

    lo, hi = float(np.percentile(gt, 1.0)), float(np.percentile(gt, 99.5))

    # ---- paper-style figure (overlap-add), GT / wSwinIR / wCNN + patterns ----
    row_w = 1000
    left_canvas, row_h = _compose_rows(
        [("Ground Truth", gt, None), ("wSwinIR", rec_swin, GREEN), ("wCNN", rec_cnn, RED)], row_w, lo, hi)
    left_h = left_canvas.shape[0]
    pat_swin = torch.load(runs_dir / conds[0] / "illumination" / "patterns.pt", map_location="cpu")
    pat_cnn = torch.load(runs_dir / conds[1] / "illumination" / "patterns.pt", map_location="cpu")
    pgap = 4
    tile_px = ((left_h - pgap) // 2 - pgap) // 2
    swin_grid = _pattern_grid(pat_swin, tile_px, pgap, GREEN)
    cnn_grid = _pattern_grid(pat_cnn, tile_px, pgap, RED)
    _frame(swin_grid, GREEN, max(3, tile_px // 40)); _frame(cnn_grid, RED, max(3, tile_px // 40))
    pat_w = max(swin_grid.width, cnn_grid.width)
    pat_canvas = np.full((left_h, pat_w, 3), 255, dtype=np.uint8)
    pat_canvas[0:swin_grid.height, 0:swin_grid.width] = np.array(swin_grid)
    y_red = swin_grid.height + pgap
    pat_canvas[y_red:y_red + cnn_grid.height, 0:cnn_grid.width] = np.array(cnn_grid)
    sep = 10
    canvas = np.full((left_h, row_w + sep + pat_w, 3), 255, dtype=np.uint8)
    canvas[:, :row_w] = left_canvas
    canvas[:, row_w + sep:row_w + sep + pat_w] = pat_canvas
    Image.fromarray(canvas).save(out_dir / "figure9_paper_style.png")
    print(f"Saved {out_dir / 'figure9_paper_style.png'}")

    # ---- naive-vs-overlap comparison (documents the tiling artifact) ----
    cmp_canvas, _ = _compose_rows([
        ("wCNN naive (non-overlap seams)", naive_cnn, RED),
        ("wCNN overlap-add (fixed)", rec_cnn, GREEN),
        ("wSwinIR naive (non-overlap seams)", naive_swin, RED),
        ("wSwinIR overlap-add (fixed)", rec_swin, GREEN),
    ], row_w, lo, hi)
    Image.fromarray(cmp_canvas).save(out_dir / "figure9_overlap_vs_naive.png")
    print(f"Saved {out_dir / 'figure9_overlap_vs_naive.png'}")

    metrics = {
        "source": src_path.name, "top": top, "left": left, "height": height, "width": width,
        "eval_sigmoid_m": eval_m, "overlap_px": {"wswinir": ov_swin, "wcnn": ov_cnn},
        "display_norm": "identical viridis lo/hi from GT p1/p99.5 for GT/wSwinIR/wCNN",
        "overlap_add": {"wSwinIR_vs_gt": _metrics(rec_swin, gt), "wCNN_vs_gt": _metrics(rec_cnn, gt)},
        "naive_tiling": {"wSwinIR_vs_gt": _metrics(naive_swin, gt), "wCNN_vs_gt": _metrics(naive_cnn, gt)},
    }
    (out_dir / "figure9_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
