#!/usr/bin/env python3
"""Export the individual panels that make up Fig. 7 / 8 / 9 / 10.

Uses the same scripts and checkpoints that produced the promoted figures:

  Fig. 7  paper_ready_results/scripts/table02_swinir_sr/render_full_image_eval.py
          + AM-4 full-run illumination tensors
  Fig. 8  scripts/figure08_mcf7/reproduce_fig8.py
          config: configs/figure08_mcf7/reproduce_fig8_tubulin.yaml
          indices 12 6 4, v3 wSwinIR checkpoint (epoch 46)
  Fig. 9  scripts/figure08_mcf7/reproduce_fig9.py
          config: configs/figure08_mcf7/reproduce_fig9_tubulin.yaml
  Fig. 10 scripts/figure10_ablation/reproduce.py
          qualitative_tensors.pt / H_t.pt, samples i0=17 i1=11

Each figure directory gets a ``raw_components/`` folder of standalone PNGs
(no axes, labels, or borders) plus a ``manifest.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import matplotlib

matplotlib.use("Agg")
try:
    _VIRIDIS = matplotlib.colormaps["viridis"]
except Exception:  # pragma: no cover
    import matplotlib.cm as _cm

    _VIRIDIS = _cm.get_cmap("viridis")

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIG07 = WS / "paper_ready_results/02_main_figures/fig07_swinir_standard_sr"
FIG08 = ROOT / "results/reproduced_figures/fig8_v3_matched"
FIG09 = ROOT / "results/reproduced_figures/fig9_v3"
FIG10 = ROOT / "results/reproduced_figures/fig10"

AM4 = ROOT / "experiments/table02_swinir_sr/full"
FIG10_RUNS = ROOT / "experiments/figure10_ablation/runs"
# Fig 8/9 v3 was generated from the superpixel-8 recipe (yaml later switched to sp=1).
# Checkpoints live under quarantine; Q is the epoch-46 full retrain (q_epoch=46),
# R is transpose256_sp8 (r_epoch=51), wCNN is wcnn64_sp8.
Q_FIG89 = WS / "quarantine_obsolete/replication/experiments/figure08_mcf7"
V3_WSWINIR = Q_FIG89 / "runs_full/wswinir"
V3_TRANSPOSE = Q_FIG89 / "runs/transpose256_sp8"
V3_WCNN = Q_FIG89 / "runs/wcnn64_sp8"
V3_SUPERPIXEL = 8

CFG8 = ROOT / "configs/figure08_mcf7/reproduce_fig8_tubulin.yaml"
CFG9 = ROOT / "configs/figure08_mcf7/reproduce_fig9_tubulin.yaml"

DATASETS = ["Set5", "Set14", "BSD100", "Urban100", "Manga109"]
FIG8_INDICES = [12, 6, 4]
FIG10_I0, FIG10_I1 = 17, 11


def _load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _to_viridis(gray: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    return (_VIRIDIS(norm)[..., :3] * 255.0 + 0.5).astype(np.uint8)


def _save_rgb(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(np.asarray(arr)).save(path)


def _save_gray01(arr: np.ndarray, path: Path) -> None:
    g = np.clip(np.asarray(arr), 0.0, 1.0)
    Image.fromarray((g * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def _write_manifest(out: Path, payload: dict) -> None:
    (out / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        payload.get("readme", "Individual panels of the promoted figure. See manifest.json.\n"),
        encoding="utf-8",
    )


def _v3_runs_dir() -> Path:
    """Assemble the exact runs_dir used to render fig8_v3_matched / fig9_v3."""
    tmp = Path(tempfile.mkdtemp(prefix="fig89_v3_runs_"))
    mapping = {
        "wswinir": V3_WSWINIR,
        "transpose256": V3_TRANSPOSE,
        "wcnn64": V3_WCNN,
    }
    for name, src in mapping.items():
        if not (src / "checkpoints" / "best.pt").exists():
            raise FileNotFoundError(f"missing v3 checkpoint for {name}: {src}")
        (tmp / name).symlink_to(src)
    return tmp


def _v3_cfg(cfg: dict) -> dict:
    """The promoted v3 figures were rendered with superpixel_factor=8."""
    cfg = dict(cfg)
    pg = dict(cfg.get("pattern_generator", {}))
    pg["superpixel_factor"] = V3_SUPERPIXEL
    cfg["pattern_generator"] = pg
    return cfg


# --------------------------------------------------------------------------- #
# Fig. 7
# --------------------------------------------------------------------------- #
def export_fig07() -> None:
    src = FIG07 / "full_image_eval"
    out = FIG07 / "raw_components"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    tiles: dict = {"full_image": {}, "illumination": {}, "patch_level": {}}

    for ds in DATASETS:
        tiles["full_image"][ds] = {}
        for suf in ("GT", "woLI", "withLI"):
            src_p = src / f"{ds}_{suf}.png"
            dst = out / f"full_{ds}_{suf}.png"
            shutil.copy2(src_p, dst)
            tiles["full_image"][ds][suf] = dst.name

    for kind, run in (("fixed", "swinir_wo_li"), ("learned", "swinir_with_li")):
        ht = torch.load(AM4 / run / "illumination" / "patterns.pt", map_location="cpu")
        tiles["illumination"][kind] = []
        for t in range(ht.shape[0]):
            arr = ht[t, 0].numpy()
            p = out / f"Ht_{kind}_t{t}.png"
            _save_gray01(arr, p)
            tiles["illumination"][kind].append(p.name)

    patch = FIG07 / "patch_level_eval"
    for name, dst_name in (
        ("fig07_GT.png", "patch_GT.png"),
        ("fig07_wo_LI.png", "patch_woLI.png"),
        ("fig07_with_LI.png", "patch_withLI.png"),
    ):
        shutil.copy2(patch / name, out / dst_name)
        tiles["patch_level"][dst_name] = dst_name

    _write_manifest(
        out,
        {
            "figure": "7",
            "script": "paper_ready_results/scripts/table02_swinir_sr/render_full_image_eval.py",
            "run": str(AM4.relative_to(WS)),
            "note": (
                "full_* are the native-resolution grayscale tiles of the 5x3 grid "
                "in full_image_eval/fig07_full_image_qualitative.png (GT / SwinIR w/o LI "
                "/ SwinIR with LI). Ht_* are the 64x64 illumination patterns (top row of "
                "the illumination panel = fixed/w/o LI, bottom = learned/with LI). "
                "patch_* are the earlier 64x64 single-patch renders."
            ),
            "layout": {
                "rows": DATASETS,
                "cols": ["GT", "woLI", "withLI"],
                "file_pattern": "full_{dataset}_{col}.png",
            },
            "tiles": tiles,
            "readme": (
                "# Figure 7 — raw components\n\n"
                "Standalone panels of the promoted Fig. 7 (full-image tiled 64x64 SwinIR x16).\n\n"
                "- `full_{Set5,Set14,BSD100,Urban100,Manga109}_{GT,woLI,withLI}.png` — 5x3 grid\n"
                "- `Ht_fixed_t{0-3}.png` / `Ht_learned_t{0-3}.png` — illumination patterns\n"
                "- `patch_{GT,woLI,withLI}.png` — 64x64 patch-level renders\n"
            ),
        },
    )
    print(f"[fig07] wrote {len(list(out.glob('*.png')))} tiles -> {out}", flush=True)


# --------------------------------------------------------------------------- #
# Fig. 8
# --------------------------------------------------------------------------- #
def export_fig08(device) -> None:
    f8 = _load_mod("reproduce_fig8", ROOT / "scripts/figure08_mcf7/reproduce_fig8.py")
    cfg = _v3_cfg(f8._load_yaml(CFG8))
    runs_dir = _v3_runs_dir()
    try:
        eval_m = float(cfg["reproduce"].get("eval_sigmoid_m", 8.0))
        q_model, _, q_ep = f8._load_model("wswinir", cfg, runs_dir, device)
        r_model, _, r_ep = f8._load_model("transpose256", cfg, runs_dir, device)

        ds_cfg = dict(cfg["dataset"])
        ds_cfg["seed"] = 42
        ds_cfg["patch_size"] = 256
        ds_cfg["image_size"] = 256
        ds = f8.MCF7Channel2Dataset.from_dict(ds_cfg, split="test")

        out = FIG08 / "raw_components"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        tiles: dict = {"P": [], "Q": [], "R": [], "P_gray": [], "Q_gray": [], "R_gray": []}
        columns = []
        for j, idx in enumerate(FIG8_INDICES):
            x = ds[idx].unsqueeze(0).to(device)
            gt = x.squeeze().cpu().numpy()
            q = f8._recon(q_model, x, eval_m).squeeze().cpu().numpy()
            r = f8._recon(r_model, x, eval_m).squeeze().cpu().numpy()
            lo, hi = float(np.percentile(gt, 1.0)), float(np.percentile(gt, 99.5))
            for key, arr in (("P", gt), ("Q", q), ("R", r)):
                vis = out / f"{key}_col{j}.png"
                gray = out / f"{key}_col{j}_gray.png"
                _save_rgb(_to_viridis(arr, lo, hi), vis)
                _save_gray01(arr, gray)
                tiles[key].append(vis.name)
                tiles[f"{key}_gray"].append(gray.name)
            columns.append(
                {
                    "col": j,
                    "idx": int(idx),
                    "lo": lo,
                    "hi": hi,
                    "Q_vs_gt": f8._metrics(q, gt),
                    "R_vs_gt": f8._metrics(r, gt),
                }
            )
        _write_manifest(
            out,
            {
                "figure": "8",
                "script": "scripts/figure08_mcf7/reproduce_fig8.py",
                "config": str(CFG8.relative_to(ROOT)),
                "indices": FIG8_INDICES,
                "eval_sigmoid_m": eval_m,
                "q_epoch": q_ep,
                "r_epoch": r_ep,
                "wswinir_checkpoint": str(V3_WSWINIR / "checkpoints/best.pt"),
                "display_norm": "identical per-column viridis from GT p1/p99.5",
                "layout": {
                    "rows": ["P (GT)", "Q (with SwinIR)", "R (w/o SwinIR)"],
                    "cols": FIG8_INDICES,
                    "file_pattern": "{P,Q,R}_col{j}.png",
                },
                "columns": columns,
                "tiles": tiles,
                "readme": (
                    "# Figure 8 — raw components\n\n"
                    "3x3 grid of `figure8_paper_style.png`, no labels.\n\n"
                    "- `P_col{0,1,2}.png` — Ground Truth (viridis, GT p1/p99.5)\n"
                    "- `Q_col{0,1,2}.png` — with SwinIR (same per-column lo/hi)\n"
                    "- `R_col{0,1,2}.png` — w/o SwinIR (transpose-conv)\n"
                    "- `*_gray.png` — the same tiles as unstretched grayscale [0,1]\n"
                    f"- columns = test indices {FIG8_INDICES}\n"
                ),
            },
        )
        print(f"[fig08] wrote {len(list(out.glob('*.png')))} tiles -> {out}  q_epoch={q_ep}", flush=True)
    finally:
        shutil.rmtree(runs_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Fig. 9
# --------------------------------------------------------------------------- #
def export_fig09(device) -> None:
    f9 = _load_mod("reproduce_fig9", ROOT / "scripts/figure08_mcf7/reproduce_fig9.py")
    cfg = _v3_cfg(f9._load_yaml(CFG9))
    runs_dir = _v3_runs_dir()
    try:
        rep = cfg.get("reproduce", {})
        eval_m = float(rep.get("eval_sigmoid_m", 8.0))
        overlap_frac = float(rep.get("tile_overlap_frac", 0.25))
        swin, _ = f9._load_model("wswinir", cfg, runs_dir, device)
        cnn, _ = f9._load_model("wcnn64", cfg, runs_dir, device)

        ds_cfg = dict(cfg["dataset"])
        ds_cfg["seed"] = 42
        ds_cfg["patch_size"] = 256
        ds_cfg["image_size"] = 256
        ds = f9.MCF7Channel2Dataset.from_dict(ds_cfg, split="test")
        src_index = int(rep.get("fig9_src_index", 0))
        top, left = int(rep.get("fig9_top", 384)), int(rep.get("fig9_left", 0))
        height, width = int(rep.get("fig9_height", 256)), int(rep.get("fig9_width", 1280))
        src_path = Path(ds.specs[src_index][0])
        full = f9._preprocess(f9._load_tiff(src_path), f9.MCF7Channel2Config.from_dict(ds_cfg))
        _, Hs, Ws = full.shape
        top = min(top, max(0, Hs - height))
        left = min(left, max(0, Ws - width))
        field = full[:, top : top + height, left : left + width].unsqueeze(0)
        gt = field[0, 0].numpy()

        ov_swin = max(16, int(256 * overlap_frac))
        ov_cnn = max(16, int(64 * overlap_frac))
        rec_swin = f9.overlap_tiled_recon(swin, field, device, eval_m, 256, ov_swin)
        rec_cnn = f9.overlap_tiled_recon(cnn, field, device, eval_m, 64, ov_cnn)
        naive_swin = f9.naive_tiled_recon(swin, field, device, eval_m, 256)
        naive_cnn = f9.naive_tiled_recon(cnn, field, device, eval_m, 64)
        lo, hi = float(np.percentile(gt, 1.0)), float(np.percentile(gt, 99.5))

        out = FIG09 / "raw_components"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)

        rows = {
            "row_GT": gt,
            "row_wSwinIR": rec_swin,
            "row_wCNN": rec_cnn,
            "row_wCNN_naive": naive_cnn,
            "row_wCNN_overlap": rec_cnn,
            "row_wSwinIR_naive": naive_swin,
            "row_wSwinIR_overlap": rec_swin,
        }
        tiles: dict = {"rows": {}, "rows_gray": {}, "Ht_wSwinIR": [], "Ht_wCNN": []}
        for name, arr in rows.items():
            vis = out / f"{name}.png"
            gray = out / f"{name}_gray.png"
            _save_rgb(_to_viridis(arr, lo, hi), vis)
            _save_gray01(arr, gray)
            tiles["rows"][name] = vis.name
            tiles["rows_gray"][name] = gray.name

        pat_swin = torch.load(runs_dir / "wswinir" / "illumination" / "patterns.pt", map_location="cpu")
        pat_cnn = torch.load(runs_dir / "wcnn64" / "illumination" / "patterns.pt", map_location="cpu")
        for tag, pat in (("wSwinIR", pat_swin), ("wCNN", pat_cnn)):
            soft = pat.squeeze(1).clamp(0, 1).numpy()
            for t in range(soft.shape[0]):
                p = out / f"Ht_{tag}_t{t}.png"
                _save_gray01(soft[t], p)
                binar = out / f"Ht_{tag}_t{t}_binarized.png"
                Image.fromarray(np.where(soft[t] > 0.5, 255, 0).astype(np.uint8), mode="L").save(binar)
                tiles[f"Ht_{tag}"].append({"gray": p.name, "binarized": binar.name})

        _write_manifest(
            out,
            {
                "figure": "9",
                "script": "scripts/figure08_mcf7/reproduce_fig9.py",
                "config": str(CFG9.relative_to(ROOT)),
                "source": src_path.name,
                "crop": {"top": top, "left": left, "height": height, "width": width},
                "eval_sigmoid_m": eval_m,
                "overlap_px": {"wswinir": ov_swin, "wcnn": ov_cnn},
                "display_norm": "identical viridis lo/hi from GT p1/p99.5",
                "lo": lo,
                "hi": hi,
                "metrics": {
                    "overlap_add": {
                        "wSwinIR_vs_gt": f9._metrics(rec_swin, gt),
                        "wCNN_vs_gt": f9._metrics(rec_cnn, gt),
                    },
                    "naive_tiling": {
                        "wSwinIR_vs_gt": f9._metrics(naive_swin, gt),
                        "wCNN_vs_gt": f9._metrics(naive_cnn, gt),
                    },
                },
                "layout_paper_style": ["row_GT", "row_wSwinIR", "row_wCNN", "Ht_wSwinIR_t*", "Ht_wCNN_t*"],
                "layout_overlap_vs_naive": [
                    "row_wCNN_naive",
                    "row_wCNN_overlap",
                    "row_wSwinIR_naive",
                    "row_wSwinIR_overlap",
                ],
                "tiles": tiles,
                "readme": (
                    "# Figure 9 — raw components\n\n"
                    "Wide-field panels of `figure9_paper_style.png` and "
                    "`figure9_overlap_vs_naive.png`, no labels or colored frames.\n\n"
                    "- `row_GT.png` / `row_wSwinIR.png` / `row_wCNN.png` — paper-style left column\n"
                    "- `row_*_naive.png` / `row_*_overlap.png` — tiling comparison\n"
                    "- `Ht_wSwinIR_t{0-3}.png` / `Ht_wCNN_t{0-3}.png` — illumination patterns\n"
                    "- `*_binarized.png` — thresholded display version used in the figure\n"
                    "- `*_gray.png` — unstretched grayscale [0,1]\n"
                ),
            },
        )
        print(f"[fig09] wrote {len(list(out.glob('*.png')))} tiles -> {out}", flush=True)
    finally:
        shutil.rmtree(runs_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Fig. 10
# --------------------------------------------------------------------------- #
def export_fig10() -> None:
    f10 = _load_mod("reproduce_fig10", ROOT / "scripts/figure10_ablation/reproduce.py")
    data = {L: f10._load(FIG10_RUNS, L, 42) for L in f10.LETTERS}
    out = FIG10 / "raw_components"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    tiles: dict = {"a": {}, "b": {}, "c": {}, "a_gray": {}, "c_gray": {}}
    for L in f10.LETTERS:
        recon = data[L]["recon"]
        ht = data[L]["H_t"]
        a = recon[FIG10_I0, 0].numpy()
        c = recon[FIG10_I1, 0].numpy()
        p = ht[0, 0].numpy()
        _save_rgb(_to_viridis(a, 0.0, 1.0), out / f"a_{L}.png")
        _save_gray01(p, out / f"b_{L}.png")
        _save_rgb(_to_viridis(c, 0.0, 1.0), out / f"c_{L}.png")
        _save_gray01(a, out / f"a_{L}_gray.png")
        _save_gray01(c, out / f"c_{L}_gray.png")
        tiles["a"][L] = f"a_{L}.png"
        tiles["b"][L] = f"b_{L}.png"
        tiles["c"][L] = f"c_{L}.png"
        tiles["a_gray"][L] = f"a_{L}_gray.png"
        tiles["c_gray"][L] = f"c_{L}_gray.png"

    _write_manifest(
        out,
        {
            "figure": "10",
            "script": "scripts/figure10_ablation/reproduce.py",
            "runs": str(FIG10_RUNS.relative_to(ROOT)),
            "samples": {"i0": FIG10_I0, "i1": FIG10_I1, "pattern_index": 0},
            "display_norm": "recon viridis vmin=0 vmax=1; H_t grayscale vmin=0 vmax=1",
            "metrics": {
                L: {"test_ssim": data[L]["test_ssim"], "test_mse": data[L]["test_mse"]}
                for L in f10.LETTERS
            },
            "layout": {
                "rows": ["a (test recon #1)", "b (learned H_t)", "c (test recon #2)"],
                "cols": ["A", "B", "C", "D"],
                "file_pattern": "{a,b,c}_{A,B,C,D}.png",
            },
            "tiles": tiles,
            "readme": (
                "# Figure 10 — raw components\n\n"
                "3x4 grid of `figure10_paper_style.png`, no labels or green highlight.\n\n"
                "- `a_{A,B,C,D}.png` — test reconstruction #1 (viridis, [0,1])\n"
                "- `b_{A,B,C,D}.png` — learned illumination H_t[0] (grayscale)\n"
                "- `c_{A,B,C,D}.png` — test reconstruction #2 (viridis, [0,1])\n"
                "- `*_gray.png` — reconstructions as unstretched grayscale\n"
                f"- samples: i0={FIG10_I0}, i1={FIG10_I1} (same as the promoted figure)\n"
            ),
        },
    )
    print(f"[fig10] wrote {len(list(out.glob('*.png')))} tiles -> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Fig 7/8/9/10 individual panels")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--figures", nargs="*", default=["fig07", "fig08", "fig09", "fig10"])
    args = ap.parse_args()

    wanted = {f.lower().replace("figure", "fig") for f in args.figures}
    if "fig07" in wanted or "7" in wanted:
        export_fig07()
    if "fig10" in wanted or "10" in wanted:
        export_fig10()

    need_gpu = wanted & {"fig08", "fig8", "8", "fig09", "fig9", "9"}
    if need_gpu:
        from utils.device import resolve_device

        device = resolve_device(args.device)
        print(f"[export] device={device}", flush=True)
        if "fig08" in wanted or "fig8" in wanted or "8" in wanted:
            export_fig08(device)
        if "fig09" in wanted or "fig9" in wanted or "9" in wanted:
            export_fig09(device)


if __name__ == "__main__":
    main()
