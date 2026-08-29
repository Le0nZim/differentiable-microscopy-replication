#!/usr/bin/env python3
"""Regenerate paper Fig. 8 (HumanMCF7 x16): P = GT / Q = with SwinIR / R = w/O SwinIR.

Loads a `configs/reproduce_fig8_*.yaml`, loads the trained checkpoints named by the
config (learned H_t included), and renders the paper-style 3-row grid with IDENTICAL
per-column display normalization for GT and both reconstructions (stated on the figure).
Uses model.eval(), no augmentation, no noise, fixed indices.

Usage:
  python scripts/figure08_mcf7/reproduce_fig8.py --config configs/figure08_mcf7/reproduce_fig8_tubulin.yaml --device cuda:0
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

from datasets.mcf7_channel2 import MCF7Channel2Dataset
from evaluation.metrics import mse as mse_metric
from evaluation.metrics import psnr as psnr_metric
from evaluation.metrics import ssim as ssim_metric
from utils.device import resolve_device

_spec = importlib.util.spec_from_file_location(
    "fig89_train", ROOT / "scripts" / "fig89_mcf7_swinir_fix_train.py")
TR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TR)


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _font(size: int):
    for cand in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if Path(cand).exists():
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


def _to_viridis(gray: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    return (_VIRIDIS(norm)[..., :3] * 255.0 + 0.5).astype(np.uint8)


def _load_model(condition: str, cfg: dict, runs_dir: Path, device):
    backbone, isz, up_mode, _ = TR.CONDITIONS[condition]
    if backbone == "swinir":
        model = TR._build_swinir_model(cfg, isz).to(device)
    else:
        model = TR._build_conventional_model(cfg, isz, up_mode).to(device)
    model(torch.zeros(1, 1, isz, isz, device=device))
    ckpt = runs_dir / condition / "checkpoints" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint for {condition}: {ckpt}")
    ck = torch.load(ckpt, map_location=device)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    if not any(k.startswith("pattern_generator") for k in state):
        print(f"WARNING: checkpoint {ckpt} has NO pattern_generator state (H_t not restored!)")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, isz, (ck.get("epoch") if isinstance(ck, dict) else None)


@torch.no_grad()
def _recon(model, x, m):
    return model(x, sigmoid_m=m, apply_noise=False)["x_recon"].float().clamp(0, 1)


def _metrics(a, b):
    ta = torch.from_numpy(np.ascontiguousarray(a))[None, None]
    tb = torch.from_numpy(np.ascontiguousarray(b))[None, None]
    return {"psnr": float(psnr_metric(ta, tb).item()),
            "ssim": float(ssim_metric(ta, tb).item()),
            "mse": float(mse_metric(ta, tb).item())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--indices", type=int, nargs="*", default=None)
    ap.add_argument("--runs-dir", default=None, help="override reproduce.runs_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    rep = cfg.get("reproduce", {})
    runs_dir = ROOT / (args.runs_dir or rep.get("runs_dir", "experiments/figure08_mcf7/runs"))
    conds = rep.get("conditions", ["wswinir", "transpose256"])
    eval_m = float(rep.get("eval_sigmoid_m", cfg["training"].get("eval_sigmoid_m", 8.0)))
    device = resolve_device(args.device)
    out_dir = Path(args.out) if args.out else ROOT / "results" / "reproduced_figures" / cfg["experiment"]["tag"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | channel: {cfg['dataset'].get('bbbc021_channel')} | conditions: {conds}")

    q_model, isz, q_ep = _load_model(conds[0], cfg, runs_dir, device)
    r_model, _, r_ep = _load_model(conds[1], cfg, runs_dir, device)

    ds_cfg = dict(cfg["dataset"]); ds_cfg["seed"] = 42; ds_cfg["patch_size"] = 256; ds_cfg["image_size"] = 256
    ds = MCF7Channel2Dataset.from_dict(ds_cfg, split="test")
    indices = args.indices
    if not indices:
        scores = sorted(((float(ds[i].std()), i) for i in range(min(24, len(ds)))), reverse=True)
        top = [i for _, i in scores]
        indices = [top[0], top[len(top) // 2], top[-1]]
    print(f"Fig8 indices (deterministic): {indices}")

    cols = []
    for idx in indices:
        x = ds[idx].unsqueeze(0).to(device)
        cols.append({"idx": int(idx), "gt": x.squeeze().cpu().numpy(),
                     "Q": _recon(q_model, x, eval_m).squeeze().cpu().numpy(),
                     "R": _recon(r_model, x, eval_m).squeeze().cpu().numpy()})

    cell, gap = 256, 6
    H = 3 * cell + 2 * gap
    W = len(cols) * cell + (len(cols) - 1) * gap
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    rows = [("P: Ground Truth", "gt"), ("Q: with SwinIR", "Q"), ("R: w/O SwinIR", "R")]
    for c, col in enumerate(cols):
        # IDENTICAL display normalization for GT + Q + R within a column (from GT percentiles).
        lo, hi = float(np.percentile(col["gt"], 1.0)), float(np.percentile(col["gt"], 99.5))
        for r, (_lbl, key) in enumerate(rows):
            img = Image.fromarray(_to_viridis(col[key], lo, hi)).resize((cell, cell), Image.LANCZOS)
            canvas[r * (cell + gap):r * (cell + gap) + cell, c * (cell + gap):c * (cell + gap) + cell] = np.array(img)
    out = Image.fromarray(canvas)
    d = ImageDraw.Draw(out)
    font = _font(28)
    for r, (lbl, _k) in enumerate(rows):
        ty = r * (cell + gap) + cell - 34
        d.text((12, ty + 2), lbl, fill=(0, 0, 0), font=font)
        d.text((10, ty), lbl, fill=(255, 255, 255), font=font)
    out.save(out_dir / "figure8_paper_style.png")
    print(f"Saved {out_dir / 'figure8_paper_style.png'}")

    metrics = {"indices": indices, "eval_sigmoid_m": eval_m, "q_epoch": q_ep, "r_epoch": r_ep,
               "display_norm": "identical per-column viridis from GT p1/p99.5 (GT, Q, R share lo/hi)",
               "columns": [{"idx": col["idx"], "Q_vs_gt": _metrics(col["Q"], col["gt"]),
                            "R_vs_gt": _metrics(col["R"], col["gt"])} for col in cols]}
    (out_dir / "figure8_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
