#!/usr/bin/env python3
"""Evaluate + render Figures 8 & 9 for the MCF7 SwinIR high-res reconstruction fix.

Loads the isolated-run checkpoints from
experiments/figure08_mcf7/runs/{wswinir,transpose256,wcnn64}/,
renders the paper layouts (viridis), writes a quantitative CSV, and fills REPORT.md.

  Figure 8 : rows P (GT) / Q (wSwinIR) / R (transpose-conv), 3 x 256x256 crops.
  Figure 9 : wide field GT / wSwinIR / wCNN + learned illumination patterns strip.

Usage:
  python scripts/figure08_mcf7/report.py --device cuda:0
  python scripts/figure08_mcf7/report.py --device cuda:0 --fig8-indices 3 7 11
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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

from datasets.mcf7_channel2 import (
    MCF7Channel2Config,
    MCF7Channel2Dataset,
    _load_tiff,
    _preprocess,
)
from evaluation.metrics import mse as mse_metric
from evaluation.metrics import psnr as psnr_metric
from evaluation.metrics import ssim as ssim_metric
from utils.device import resolve_device

_spec = importlib.util.spec_from_file_location(
    "fig89_train", ROOT / "scripts" / "fig89_mcf7_swinir_fix_train.py")
TR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TR)

EXP = ROOT / "experiments/figure08_mcf7"
CFG = ROOT / "configs/figure08_mcf7/swinir_fix.yaml"
RUNS = EXP / "runs"
FIGS = EXP / "figures"
GREEN, RED = (40, 200, 90), (220, 60, 60)


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _font(size: int):
    for cand in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(cand).exists():
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


def _to_viridis(gray: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    return (_VIRIDIS(norm)[..., :3] * 255.0 + 0.5).astype(np.uint8)


def _frame(img: Image.Image, color, width: int) -> None:
    d = ImageDraw.Draw(img)
    for k in range(width):
        d.rectangle([k, k, img.width - 1 - k, img.height - 1 - k], outline=color)


def _load(condition: str, cfg: dict, device):
    backbone, isz, up, _ = TR.CONDITIONS[condition]
    if backbone == "swinir":
        model = TR._build_swinir_model(cfg, isz).to(device)
    else:
        model = TR._build_conventional_model(cfg, isz, up).to(device)
    model(torch.zeros(1, 1, isz, isz, device=device))
    ck = torch.load(RUNS / condition / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    model.eval()
    return model, isz


@torch.no_grad()
def _recon(model, x, eval_m):
    return model(x, sigmoid_m=eval_m, apply_noise=False)["x_recon"].float().clamp(0, 1)


@torch.no_grad()
def _tiled_recon(model, field: torch.Tensor, device, eval_m: float, tile: int) -> np.ndarray:
    H, W = field.shape[-2:]
    padH, padW = (tile - H % tile) % tile, (tile - W % tile) % tile
    f = F.pad(field, (0, padW, 0, padH), mode="reflect")
    Hp, Wp = f.shape[-2:]
    tiles, coords = [], []
    for i in range(0, Hp, tile):
        for j in range(0, Wp, tile):
            tiles.append(f[:, :, i:i + tile, j:j + tile])
            coords.append((i, j))
    batch = torch.cat(tiles, 0)
    bs = 8 if tile >= 256 else 64
    recons = []
    for k in range(0, batch.shape[0], bs):
        rec = model(batch[k:k + bs].to(device), sigmoid_m=eval_m, apply_noise=False)["x_recon"].float().clamp(0, 1)
        recons.append(rec.cpu())
    recons = torch.cat(recons, 0)
    out = torch.zeros(1, 1, Hp, Wp)
    for (i, j), r in zip(coords, recons):
        out[:, :, i:i + tile, j:j + tile] = r
    return out[0, 0, :H, :W].numpy()


def _pair(a, b):
    ta, tb = torch.from_numpy(np.ascontiguousarray(a))[None, None], torch.from_numpy(np.ascontiguousarray(b))[None, None]
    return {"psnr": float(psnr_metric(ta, tb).item()), "ssim": float(ssim_metric(ta, tb).item()),
            "mse": float(mse_metric(ta, tb).item())}


# ---------------------------------------------------------------------------
# Figure 8
# ---------------------------------------------------------------------------
def render_fig8(cfg, device, eval_m, indices, cell=256, gap=6):
    q_model, _ = _load("wswinir", cfg, device)
    r_model, _ = _load("transpose256", cfg, device)
    ds_cfg = dict(cfg["dataset"]); ds_cfg["seed"] = 42; ds_cfg["patch_size"] = 256; ds_cfg["image_size"] = 256
    ds = MCF7Channel2Dataset.from_dict(ds_cfg, split="test")
    if not indices:
        scores = sorted(((float(ds[i].std()), i) for i in range(min(24, len(ds)))), reverse=True)
        top = [i for _, i in scores[:max(6, len(scores) // 2)]]
        indices = [top[0], top[len(top) // 2], top[-1]]
    print(f"Fig8 columns: {indices}", flush=True)

    rows = [("P: Ground Truth", None), ("Q: with SwinIR", "Q"), ("R: w/O SwinIR", "R")]
    cols = []
    for idx in indices:
        x = ds[idx].unsqueeze(0).to(device)
        cols.append({"idx": int(idx), "gt": x.squeeze().cpu().numpy(),
                     "Q": _recon(q_model, x, eval_m).squeeze().cpu().numpy(),
                     "R": _recon(r_model, x, eval_m).squeeze().cpu().numpy()})
    H = 3 * cell + 2 * gap
    W = len(cols) * cell + (len(cols) - 1) * gap
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    for c, col in enumerate(cols):
        lo, hi = float(np.percentile(col["gt"], 1.0)), float(np.percentile(col["gt"], 99.5))
        for r, (_lbl, mkey) in enumerate(rows):
            gray = col["gt"] if mkey is None else col[mkey]
            img = Image.fromarray(_to_viridis(gray, lo, hi)).resize((cell, cell), Image.LANCZOS)
            canvas[r * (cell + gap):r * (cell + gap) + cell, c * (cell + gap):c * (cell + gap) + cell] = np.array(img)
    out = Image.fromarray(canvas)
    d = ImageDraw.Draw(out)
    font = _font(max(18, cell // 11))
    for r, (lbl, _m) in enumerate(rows):
        ty = r * (cell + gap) + cell - max(28, cell // 8)
        d.text((12, ty + 2), lbl, fill=(0, 0, 0), font=font)
        d.text((10, ty), lbl, fill=(255, 255, 255), font=font)
    FIGS.mkdir(parents=True, exist_ok=True)
    out.save(FIGS / "figure8_paper_style.png")
    print(f"Saved {FIGS / 'figure8_paper_style.png'}", flush=True)

    col_metrics = [{"idx": col["idx"], "Q_vs_gt": _pair(col["Q"], col["gt"]),
                    "R_vs_gt": _pair(col["R"], col["gt"])} for col in cols]
    return {"indices": indices, "columns": col_metrics}


# ---------------------------------------------------------------------------
# Figure 9
# ---------------------------------------------------------------------------
def _pattern_grid(patterns: torch.Tensor, tile_px: int, gap: int, color) -> Image.Image:
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


def render_fig9(cfg, device, eval_m, src_index, top, left, height, width, row_w=1000):
    swin, _ = _load("wswinir", cfg, device)
    cnn, _ = _load("wcnn64", cfg, device)
    ds_cfg = dict(cfg["dataset"]); ds_cfg["seed"] = 42; ds_cfg["patch_size"] = 256; ds_cfg["image_size"] = 256
    ds = MCF7Channel2Dataset.from_dict(ds_cfg, split="test")
    src_path = Path(ds.specs[src_index][0])
    full = _preprocess(_load_tiff(src_path), MCF7Channel2Config.from_dict(ds_cfg))
    _, Hs, Ws = full.shape
    top = min(top, max(0, Hs - height)); left = min(left, max(0, Ws - width))
    field = full[:, top:top + height, left:left + width].unsqueeze(0)
    gt = field[0, 0].numpy()
    rec_swin = _tiled_recon(swin, field, device, eval_m, 256)
    rec_cnn = _tiled_recon(cnn, field, device, eval_m, 64)
    lo, hi = float(np.percentile(gt, 1.0)), float(np.percentile(gt, 99.5))

    rows = [("Ground Truth", gt, None), ("wSwinIR", rec_swin, GREEN), ("wCNN", rec_cnn, RED)]
    row_h = max(1, round(row_w * height / width))
    gap = 6
    n = len(rows)
    left_h = n * row_h + (n - 1) * gap
    left_canvas = np.full((left_h, row_w, 3), 255, dtype=np.uint8)
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
        left_canvas[r * (row_h + gap):r * (row_h + gap) + row_h, :] = np.array(img)

    pat_swin = torch.load(RUNS / "wswinir" / "illumination" / "patterns.pt", map_location="cpu")
    pat_cnn = torch.load(RUNS / "wcnn64" / "illumination" / "patterns.pt", map_location="cpu")
    pgap = 4
    block_h = (left_h - pgap) // 2
    tile_px = (block_h - pgap) // 2
    swin_grid = _pattern_grid(pat_swin, tile_px, pgap, GREEN)
    cnn_grid = _pattern_grid(pat_cnn, tile_px, pgap, RED)
    _frame(swin_grid, GREEN, max(3, tile_px // 40))
    _frame(cnn_grid, RED, max(3, tile_px // 40))
    pat_w = max(swin_grid.width, cnn_grid.width)
    pat_canvas = np.full((left_h, pat_w, 3), 255, dtype=np.uint8)
    pat_canvas[0:swin_grid.height, 0:swin_grid.width] = np.array(swin_grid)
    y_red = swin_grid.height + pgap
    pat_canvas[y_red:y_red + cnn_grid.height, 0:cnn_grid.width] = np.array(cnn_grid)

    sep = 10
    canvas = np.full((left_h, row_w + sep + pat_w, 3), 255, dtype=np.uint8)
    canvas[:, :row_w] = left_canvas
    canvas[:, row_w + sep:row_w + sep + pat_w] = pat_canvas
    FIGS.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(FIGS / "figure9_paper_style.png")
    print(f"Saved {FIGS / 'figure9_paper_style.png'}", flush=True)
    return {"source": src_path.name, "top": top, "left": left, "height": height, "width": width,
            "wSwinIR_vs_gt": _pair(rec_swin, gt), "wCNN_vs_gt": _pair(rec_cnn, gt)}


# ---------------------------------------------------------------------------
def _result(condition: str) -> dict:
    p = RUNS / condition / "result.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def write_csv(fig8, fig9):
    FIGS.mkdir(parents=True, exist_ok=True)
    rows = []
    for cond, label in [("wswinir", "Q/wSwinIR (locality+SwinIR, full loss)"),
                        ("transpose256", "R (transpose-conv+ReconCNN, L1)"),
                        ("wcnn64", "wCNN (locality+ReconCNN@64, L1)")]:
        r = _result(cond)
        if r:
            rows.append({"condition": cond, "label": label, "image_size": r.get("image_size"),
                         "full_loss": r.get("full_loss"), "embed_dim": r.get("swinir_embed_dim"),
                         "epochs": r.get("epochs"), "test_psnr": round(r.get("test_psnr", float("nan")), 3),
                         "test_ssim": round(r.get("test_ssim", float("nan")), 4),
                         "test_mse": round(r.get("test_mse", float("nan")), 6)})
    out = EXP / "metrics_summary.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["condition", "label", "image_size", "full_loss",
                                           "embed_dim", "epochs", "test_psnr", "test_ssim", "test_mse"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}", flush=True)
    return rows


def update_report(csv_rows, fig8, fig9):
    rep = EXP / "REPORT.md"
    if not rep.exists():
        return
    text = rep.read_text(encoding="utf-8")

    tbl = ["| condition | model | img | loss | embed | epochs | test PSNR | test SSIM | test MSE |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in csv_rows:
        tbl.append(f"| {r['condition']} | {r['label']} | {r['image_size']} | "
                   f"{'full' if r['full_loss'] else 'L1'} | {r['embed_dim'] or '-'} | {r['epochs']} | "
                   f"{r['test_psnr']} | {r['test_ssim']} | {r['test_mse']} |")
    block = "\n".join(tbl)

    def _by(cond):
        return next((r for r in csv_rows if r["condition"] == cond), None)

    q, rr, wc = _by("wswinir"), _by("transpose256"), _by("wcnn64")
    headline = ""
    if q and rr:
        headline = (f"SwinIR (Q) test PSNR **{q['test_psnr']} dB** / SSIM **{q['test_ssim']}** vs "
                    f"transpose-conv (R) {rr['test_psnr']} dB / {rr['test_ssim']} "
                    f"(**{q['test_psnr'] - rr['test_psnr']:+.2f} dB**, {q['test_ssim'] - rr['test_ssim']:+.4f} SSIM).")
    text = _replace(text, "RESULTS_TABLE", block)
    text = _replace(text, "HEADLINE", headline)
    text = _replace(text, "FIG8_JSON", "```json\n" + json.dumps(fig8, indent=2) + "\n```" if fig8 else "_pending_")
    text = _replace(text, "FIG9_JSON", "```json\n" + json.dumps(fig9, indent=2) + "\n```" if fig9 else "_pending_")
    text = _replace(text, "GENERATED", datetime.now(timezone.utc).isoformat())
    rep.write_text(text, encoding="utf-8")
    print(f"Updated {rep}", flush=True)


def _replace(text: str, tag: str, value: str) -> str:
    start, end = f"<!--{tag}-->", f"<!--/{tag}-->"
    if start in text and end in text:
        pre = text.split(start)[0]
        post = text.split(end, 1)[1]
        return f"{pre}{start}\n{value}\n{end}{post}"
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fig8-indices", type=int, nargs="*", default=None)
    ap.add_argument("--fig9-src-index", type=int, default=0)
    ap.add_argument("--fig9-top", type=int, default=384)
    ap.add_argument("--fig9-left", type=int, default=0)
    ap.add_argument("--fig9-height", type=int, default=256)
    ap.add_argument("--fig9-width", type=int, default=1280)
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    cfg = _load_yaml(CFG)
    eval_m = float(cfg["training"].get("eval_sigmoid_m", 8.0))
    device = resolve_device(args.device)
    print(f"Device: {device}", flush=True)

    fig8 = fig9 = None
    have = {c: (RUNS / c / "checkpoints" / "best.pt").exists() for c in TR.CONDITIONS}
    print(f"Checkpoints present: {have}", flush=True)
    if not args.skip_figures:
        if have["wswinir"] and have["transpose256"]:
            fig8 = render_fig8(cfg, device, eval_m, args.fig8_indices)
        if have["wswinir"] and have["wcnn64"]:
            fig9 = render_fig9(cfg, device, eval_m, args.fig9_src_index, args.fig9_top,
                               args.fig9_left, args.fig9_height, args.fig9_width)

    csv_rows = write_csv(fig8, fig9)
    if fig8 or fig9:
        (EXP / "figures_metadata.json").write_text(
            json.dumps({"fig8": fig8, "fig9": fig9,
                        "generated_utc": datetime.now(timezone.utc).isoformat()}, indent=2),
            encoding="utf-8")
    update_report(csv_rows, fig8, fig9)


if __name__ == "__main__":
    main()
