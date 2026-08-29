#!/usr/bin/env python3
"""Render a paper-style Figure 3 (content-aware reconstruction) for BBBC022.

Reproduces the components of paper Fig. 3 from the already-trained
``experiments/ablations/bbbc022_content_aware`` matrix (4 compressions x 4
illuminations). One common test specimen is pushed through every trained model
so that every cell shows the *same* field (as in the paper).

Components produced (all viridis for fluorescence, gray for patterns):
    A  - reconstruction grid: rows = x16/x64/x256/x1024, cols = Hadamard /
         All ones / Pseudo-random / Learnable, with method corner markers.
    B  - detection (measurement y_down) grid, normalized by max of field.
    C1 - example fixed illumination patterns (Hadamard / All ones / Pseudo-random).
    C2 - example learned illumination patterns (downscale 8/16/32/64).
    D  - SSIM vs compression scatter (markers per method, size per downscale;
         filled markers = "+SwinIR" refined results).
    E  - MSE  vs compression scatter (open = base, filled = "+SwinIR").
    F  - ground-truth test image.
A composite mirroring the paper layout is also written.

Two "+SwinIR" columns (Pseudo-random + SwinIR, Learnable + SwinIR) are added to
panel A and as filled markers in D/E, mirroring the paper's Fig. 3 / Table S1.
These use the SwinIR refiners trained by ``scripts/train_fig03_swinir_columns.py``
(a SwinIR with upscale=1 appended to the *frozen* trained microscope and trained
to super-resolve its output) so each "+SwinIR" cell shares the identical base
reconstruction as its non-SwinIR counterpart.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.swinir.refinement_model import MicroscopeSwinIRRefinement  # noqa: E402
from evaluation.metrics import ssim as ssim_metric  # noqa: E402
from models.microscope import DifferentiableMicroscope  # noqa: E402
from training.dataloaders import build_dataloader  # noqa: E402
from utils.device import resolve_device  # noqa: E402
from utils.experiment_config import load_experiment_config  # noqa: E402

# --- layout constants ---------------------------------------------------------
# (key, label, marker, color)  -- paper column order (B; base columns of A)
METHODS = [
    ("hadamard_fixed", "Hadamard", "s", "#1f77b4"),
    ("uniform_all_ones", "All ones", "^", "#1f77b4"),
    ("random_fixed", "Pseudo-random", "o", "#d62728"),
    ("learnable_frequency", "Learnable", "o", "#2ca02c"),
]
# Patterns that additionally get a "+SwinIR" refiner column (paper Fig 3 / Table S1).
SWINIR_PATTERNS = [
    ("random_fixed", "Pseudo-random\n+ SwinIR", "o", "#d62728"),
    ("learnable_frequency", "Learnable\n+ SwinIR", "o", "#2ca02c"),
]
# Panel-A columns: 4 base illuminations + 2 "+SwinIR" refined columns.
PANEL_A_COLS = [(k, lab, mk, c, False) for k, lab, mk, c in METHODS] + [
    (f"{k}_sw", lab, mk, c, True) for k, lab, mk, c in SWINIR_PATTERNS
]
# Eval sigmoid sharpness per pattern (matches base results.csv / refiner training).
EVAL_M = {"random_fixed": None, "learnable_frequency": 10.0}
# SwinIR config used by the refiners (must match train_fig03_swinir_columns.py).
SWINIR_CFG = {
    "upscale": 1, "in_chans": 1, "img_size": 256, "window_size": 8, "upsampler": "",
    "embed_dim": 96, "depths": [2, 2, 2, 2, 2, 2], "num_heads": [3, 3, 3, 3, 3, 3],
    "mlp_ratio": 2, "resi_connection": "1conv", "img_range": 1.0,
}
# rows top->bottom in panels A/B
COMPS = [("x16", 8), ("x64", 16), ("x256", 32), ("x1024", 64)]
# x-axis order (left->right) in panels D/E  (high compression -> low)
COMP_AXIS = ["x1024", "x256", "x64", "x16"]
DOWNSCALE = {"x16": 8, "x64": 16, "x256": 32, "x1024": 64}
SIZE_BY_DS = {8: 70, 16: 130, 32: 200, 64: 290}
FLUO_CMAP = "viridis"


def run_dir(exp_root: Path, comp: str, method: str) -> Path:
    return exp_root / f"bbbc022_{comp}_{method}_seed42"


def load_model(config: dict, ckpt: Path, device: torch.device, image_size: int) -> DifferentiableMicroscope:
    model = DifferentiableMicroscope.from_run_config(config).to(device)
    # Lazily size the impulse-PSF buffers before loading the checkpoint.
    model(torch.zeros(1, 1, image_size, image_size, device=device), sigmoid_m=10.0, apply_noise=False)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def read_metrics(exp_root: Path) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    with open(exp_root / "results.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            out[(row["compression"], row["pattern"])] = {
                "mse": float(row["test_mse"]),
                "ssim": float(row["test_ssim"]),
            }
    return out


def read_swinir_metrics(exp_root: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Read refined SwinIR test metrics (compression, pattern) -> {mse, ssim}."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    csv_path = exp_root / "swinir" / "swinir_results.csv"
    if not csv_path.exists():
        return out
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[(row["compression"], row["pattern"])] = {
                "mse": float(row["swinir_mse"]),
                "ssim": float(row["swinir_ssim"]),
            }
    return out


def select_test_specimen(loader, percentile: float, override: int | None) -> tuple[torch.Tensor, int]:
    """Concatenate the test split and pick a representative (median-ish density) field."""
    specimens = []
    for batch in loader:
        specimen = batch if torch.is_tensor(batch) else batch[0]
        specimens.append(specimen)
    x = torch.cat(specimens, dim=0)  # [N,1,H,W]
    if override is not None:
        return x[override : override + 1], override
    frac = (x > 0.3).float().mean(dim=(1, 2, 3)).cpu().numpy()
    target = np.percentile(frac, percentile)
    idx = int(np.argmin(np.abs(frac - target)))
    return x[idx : idx + 1], idx


def robust_norm(img: np.ndarray, lo: float = 1.0, hi: float = 99.7) -> tuple[float, float]:
    return float(np.percentile(img, lo)), float(np.percentile(img, hi))


# --- component renderers ------------------------------------------------------
def render_grid(grids: dict, out_path: Path, *, columns, title: str, normalize_each: bool,
                annotate_ssim: bool = False):
    """grids[(comp,colkey)] -> {img,ssim}. Render rows=COMPS, cols=`columns`.

    Each column entry is (colkey, label, marker, color[, is_swinir]); when
    is_swinir is True the corner marker is drawn filled (paper "+SwinIR" style).
    """
    nrows, ncols = len(COMPS), len(columns)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.05 * ncols, 2.05 * nrows))
    for r, (comp, _ds) in enumerate(COMPS):
        for c, col in enumerate(columns):
            mkey, mlabel, marker, color = col[0], col[1], col[2], col[3]
            is_swinir = col[4] if len(col) > 4 else False
            ax = axes[r, c]
            img = grids[(comp, mkey)]["img"]
            if normalize_each:
                vmax = float(img.max()) or 1.0
                ax.imshow(img / vmax, cmap=FLUO_CMAP, vmin=0.0, vmax=1.0, interpolation="nearest")
            else:
                ax.imshow(img, cmap=FLUO_CMAP, vmin=0.0, vmax=1.0, interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("white")
                s.set_linewidth(0.6)
            # corner method marker (paper-style); filled = "+SwinIR".
            ax.scatter([0.13], [0.87], transform=ax.transAxes, marker=marker, s=150,
                       facecolors=color if is_swinir else "none", edgecolors=color,
                       linewidths=1.8, zorder=6)
            if annotate_ssim and "ssim" in grids[(comp, mkey)]:
                ax.text(0.97, 0.04, f"SSIM {grids[(comp, mkey)]['ssim']:.2f}",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                        color="white", weight="bold")
            if r == 0:
                ax.set_title(mlabel, fontsize=11, color=color, weight="bold", pad=8)
            if c == 0:
                ax.set_ylabel(comp, fontsize=13, weight="bold", rotation=90, labelpad=8)
    fig.suptitle(title, fontsize=13, weight="bold", y=0.998)
    fig.subplots_adjust(left=0.045, right=0.995, top=0.90, bottom=0.01, wspace=0.03, hspace=0.03)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def render_patterns(imgs: list[np.ndarray], labels: list[str], out_path: Path, *, title: str,
                    vertical: bool, fixed_range: bool = False):
    n = len(imgs)
    if vertical:
        fig, axes = plt.subplots(n, 1, figsize=(2.4, 2.4 * n))
    else:
        fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6))
    axes = np.atleast_1d(axes).ravel()
    for ax, img, lab in zip(axes, imgs, labels):
        if fixed_range:
            vmax = float(img.max()) if float(img.max()) > 0 else 1.0
            lo, hi = 0.0, vmax
        else:
            lo, hi = robust_norm(img, 1, 99)
            if hi <= lo:
                hi = lo + 1e-6
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(lab, fontsize=11)
    fig.suptitle(title, fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def render_scatter(metrics: dict, out_path: Path, *, key: str, ylabel: str, title: str,
                   swinir_metrics: dict | None = None):
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    xpos = {c: i for i, c in enumerate(COMP_AXIS)}
    for x in range(len(COMP_AXIS)):
        ax.axvline(x, color="#9ecae1", ls="--", lw=0.8, alpha=0.7, zorder=0)
    for mkey, mlabel, marker, color in METHODS:
        xs, ys, ss = [], [], []
        for comp in COMP_AXIS:
            if (comp, mkey) in metrics:
                xs.append(xpos[comp])
                ys.append(metrics[(comp, mkey)][key])
                ss.append(SIZE_BY_DS[DOWNSCALE[comp]])
        ax.scatter(xs, ys, s=ss, facecolors="none", edgecolors=color, linewidths=1.8,
                   marker=marker, zorder=3)
    # "+SwinIR" refined results: filled markers (paper Fig 3 / Table S1).
    if swinir_metrics:
        for mkey, _lab, marker, color in SWINIR_PATTERNS:
            xs, ys, ss = [], [], []
            for comp in COMP_AXIS:
                if (comp, mkey) in swinir_metrics:
                    xs.append(xpos[comp])
                    ys.append(swinir_metrics[(comp, mkey)][key])
                    ss.append(SIZE_BY_DS[DOWNSCALE[comp]])
            ax.scatter(xs, ys, s=ss, facecolors=color, edgecolors="black", linewidths=0.7,
                       marker=marker, alpha=0.9, zorder=5)
    ax.set_xticks(range(len(COMP_AXIS)))
    ax.set_xticklabels(COMP_AXIS)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def render_single(img: np.ndarray, out_path: Path, *, title: str, cmap: str = FLUO_CMAP, vmin=0.0, vmax=1.0):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def render_legend(out_path: Path):
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.axis("off")
    method_handles = [
        Line2D([0], [0], marker=m, color="none", markerfacecolor="none",
               markeredgecolor=c, markersize=11, markeredgewidth=1.8, label=lab)
        for _k, lab, m, c in METHODS
    ]
    method_handles += [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728",
               markeredgecolor="black", markersize=11, markeredgewidth=0.7,
               label="Pseudo-random + SwinIR"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2ca02c",
               markeredgecolor="black", markersize=11, markeredgewidth=0.7,
               label="Learnable + SwinIR"),
    ]
    size_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="black", markersize=np.sqrt(SIZE_BY_DS[ds]),
               markeredgewidth=1.2, label=f"downscale {ds}x{ds}")
        for ds in (8, 16, 32, 64)
    ]
    leg1 = ax.legend(handles=method_handles, title="Illumination", loc="upper left",
                     frameon=False, fontsize=10, title_fontsize=11)
    ax.add_artist(leg1)
    ax.legend(handles=size_handles, title="Marker size", loc="lower left",
              frameon=False, fontsize=9, title_fontsize=11, labelspacing=1.1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def compose(comp_dir: Path, out_path: Path):
    """Stitch component PNGs into a single paper-style composite."""
    def im(name):
        return mpimg.imread(comp_dir / name)

    fig = plt.figure(figsize=(17, 17))
    gs = fig.add_gridspec(3, 2, height_ratios=[4.2, 4.2, 2.0], width_ratios=[3.0, 1.05],
                          hspace=0.04, wspace=0.03, left=0.02, right=0.99, top=0.95, bottom=0.02)

    def place(ax, name, letter):
        ax.imshow(im(name))
        ax.axis("off")
        ax.text(-0.01, 1.0, letter, transform=ax.transAxes, fontsize=22, weight="bold",
                va="top", ha="right")

    place(fig.add_subplot(gs[0, 0]), "panel_A_reconstructions.png", "A")
    place(fig.add_subplot(gs[0, 1]), "panel_C2_learned_patterns.png", "C2")
    place(fig.add_subplot(gs[1, 0]), "panel_B_detections.png", "B")

    gs_de = gs[1, 1].subgridspec(2, 1, hspace=0.18)
    place(fig.add_subplot(gs_de[0]), "panel_D_ssim.png", "D")
    place(fig.add_subplot(gs_de[1]), "panel_E_mse.png", "E")

    gs_bot = gs[2, :].subgridspec(1, 3, width_ratios=[2.0, 1.0, 1.1], wspace=0.05)
    place(fig.add_subplot(gs_bot[0]), "panel_C1_fixed_patterns.png", "C1")
    place(fig.add_subplot(gs_bot[1]), "panel_F_ground_truth.png", "F")
    ax_leg = fig.add_subplot(gs_bot[2])
    ax_leg.imshow(im("panel_legend.png"))
    ax_leg.axis("off")

    fig.suptitle("Figure 3 (ours, BBBC022 Hoechst substitute) — content-aware reconstruction "
                 "(+ SwinIR columns; photon count = 10000)",
                 fontsize=15, weight="bold", y=0.985)
    fig.savefig(out_path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", default=str(ROOT / "experiments/figure03_content_aware/base"))
    ap.add_argument("--out-dir", default=str(ROOT / "paper/figure03_content_aware"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--test-index", type=int, default=None)
    ap.add_argument("--percentile", type=float, default=65.0)
    args = ap.parse_args()

    exp_root = Path(args.exp_root)
    out_dir = Path(args.out_dir)
    comp_dir = out_dir / "components"
    rendered_dir = out_dir / "rendered"
    comp_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    metrics = read_metrics(exp_root)
    swinir_metrics = read_swinir_metrics(exp_root)

    # Common test specimen (identical dataset config across all runs).
    ref_cfg = load_experiment_config(run_dir(exp_root, "x16", "hadamard_fixed") / "config.yaml")
    ref_cfg["experiment"]["device"] = args.device
    image_size = ref_cfg["dataset"]["image_size"]
    test_loader = build_dataloader(ref_cfg, "test")
    specimen, chosen_idx = select_test_specimen(test_loader, args.percentile, args.test_index)
    specimen = specimen.to(device)
    print(f"[fig3] common test specimen index = {chosen_idx}", flush=True)

    recon_grid: dict = {}
    detect_grid: dict = {}
    learned_patterns: dict[str, np.ndarray] = {}
    fixed_patterns: dict[str, np.ndarray] = {}

    for comp, _ds in COMPS:
        for mkey, _lab, _mk, _col in METHODS:
            cfg = load_experiment_config(run_dir(exp_root, comp, mkey) / "config.yaml")
            cfg["experiment"]["device"] = args.device
            model = load_model(cfg, run_dir(exp_root, comp, mkey) / "checkpoints" / "best.pt", device, image_size)
            with torch.no_grad():
                out = model(specimen, sigmoid_m=10.0, apply_noise=False)
            recon = out["x_recon"][0, 0].clamp(0, 1).cpu().numpy()
            y_field = out["y_down"][0].mean(dim=0).cpu().numpy()  # [h,w] mean over T detections
            s = float(ssim_metric(out["x_recon"], specimen).item())
            recon_grid[(comp, mkey)] = {"img": recon, "ssim": s}
            detect_grid[(comp, mkey)] = {"img": y_field}
            if mkey == "learnable_frequency":
                # Paper C2 shows a single example learned pattern per downscale
                # (not the mean over T, which washes out structure). Pick the
                # most-structured channel by spatial variance.
                p_all = out["patterns"][:, 0].cpu().numpy()  # [T,H,W]
                ch = int(np.argmax(p_all.reshape(p_all.shape[0], -1).var(axis=1)))
                learned_patterns[comp] = p_all[ch]
            if comp == "x16" and mkey in ("hadamard_fixed", "uniform_all_ones", "random_fixed"):
                p_all = out["patterns"][:, 0].cpu().numpy()  # [T,H,W]
                if mkey == "hadamard_fixed":
                    # channel 0 is the DC (all-ones) Hadamard row; show the most structured one.
                    ch = int(np.argmax(p_all.reshape(p_all.shape[0], -1).var(axis=1)))
                    fixed_patterns[mkey] = p_all[ch]
                else:
                    fixed_patterns[mkey] = p_all[0]
            # "+SwinIR" refined column: append the trained refiner to the (frozen)
            # microscope and push the SAME specimen through it.
            if mkey in ("random_fixed", "learnable_frequency"):
                refiner_ckpt = exp_root / "swinir" / f"{comp}_{mkey}" / "refiner_best.pt"
                eval_m = EVAL_M[mkey]
                if refiner_ckpt.exists():
                    ref = MicroscopeSwinIRRefinement(model, dict(SWINIR_CFG), {"mode": "direct"}).to(device)
                    payload = torch.load(refiner_ckpt, map_location=device, weights_only=False)
                    ref.offline_refiner.load_state_dict(payload["refiner_state_dict"])
                    ref.eval()
                    with torch.no_grad():
                        rout = ref(specimen, sigmoid_m=eval_m, apply_noise=False)
                    rrec = rout["x_recon"][0, 0].clamp(0, 1).cpu().numpy()
                    s_sw = float(ssim_metric(rout["x_recon"].clamp(0, 1), specimen).item())
                    recon_grid[(comp, f"{mkey}_sw")] = {"img": rrec, "ssim": s_sw}
                    del ref
                else:
                    print(f"[fig3] WARNING: missing refiner {refiner_ckpt}; using base recon for "
                          f"{comp}/{mkey}_sw column", flush=True)
                    recon_grid[(comp, f"{mkey}_sw")] = {"img": recon, "ssim": s}
            del model
            torch.cuda.empty_cache()

    gt = specimen[0, 0].cpu().numpy()

    # --- render components ---
    render_grid(recon_grid, comp_dir / "panel_A_reconstructions.png", columns=PANEL_A_COLS,
                title="A  Reconstructions (same test field, viridis) — last 2 cols = + SwinIR",
                normalize_each=False, annotate_ssim=True)
    render_grid(detect_grid, comp_dir / "panel_B_detections.png",
                columns=[(k, lab, mk, c) for k, lab, mk, c in METHODS],
                title="B  Detections / measurements (normalized by max of field)", normalize_each=True)
    render_patterns([fixed_patterns["hadamard_fixed"], fixed_patterns["uniform_all_ones"], fixed_patterns["random_fixed"]],
                    ["Hadamard", "All ones", "Pseudo-random"], comp_dir / "panel_C1_fixed_patterns.png",
                    title="C1  Example fixed illumination patterns", vertical=False, fixed_range=True)
    render_patterns([learned_patterns[c] for c, _ in COMPS],
                    [f"downscale {DOWNSCALE[c]}x{DOWNSCALE[c]}" for c, _ in COMPS],
                    comp_dir / "panel_C2_learned_patterns.png",
                    title="C2  Learned patterns", vertical=True)
    render_scatter(metrics, comp_dir / "panel_D_ssim.png", key="ssim", ylabel="SSIM",
                   title="D  SSIM vs compression", swinir_metrics=swinir_metrics)
    render_scatter(metrics, comp_dir / "panel_E_mse.png", key="mse", ylabel="MSE",
                   title="E  MSE vs compression", swinir_metrics=swinir_metrics)
    render_single(gt, comp_dir / "panel_F_ground_truth.png", title="F  Ground truth")
    render_legend(comp_dir / "panel_legend.png")

    compose(comp_dir, rendered_dir / "fig03_ours.png")
    print(f"[fig3] wrote components to {comp_dir}")
    print(f"[fig3] wrote composite to {rendered_dir / 'fig03_ours.png'}")


if __name__ == "__main__":
    main()
