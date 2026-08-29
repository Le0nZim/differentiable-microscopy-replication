#!/usr/bin/env python3
"""Export the individual panels that make up Figure 4 as standalone image tiles.

Produces one clean PNG per cell of the paper-style Figure 4 grid (no axes, no
padding) so the figure can be rebuilt panel-by-panel in PowerPoint / Illustrator:

  * ``A_col{j}.png``            - row A: ground-truth test image (viridis, display
                                  percentile-stretched); ``A_col{j}_raw.png`` is the
                                  same tile with NO stretch (vmin/vmax = 0/1).
  * ``B_col{j}.png``            - row B: TrackMate pseudo-GT mask (raw MIP > 506,
                                  4-connected, DP-simplified), viridis {0,1}.
  * ``{C1,C2,D1,D2,E1,E2}_col{j}.png`` - predicted segmentation masks for the
                                  3 compressions x 2 illuminations at the
                                  val-selected threshold, viridis {0,1}.
  * ``F_{C1..E2}.png``          - row F: representative illumination pattern Hᵗ
                                  (eval m=10), grayscale.

Also writes a labelled paper-style composite ``figure4_paper_layout_viridis.png``
(+ panel-F strip and a combined ``figure4_full_viridis.png``) and a
``tiles_manifest.json`` describing the grid, thresholds, and file names.

Reads only the trained cells under
``experiments/figure04_segmentation/task_aware/runs/`` and reuses the helpers
in ``scripts/figure04_segmentation/report.py``. Run from the ``replication/`` directory:

    .venv/bin/python scripts/figure04_segmentation/export_tiles.py --device cuda:0 --seed 42 --k 5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Reuse the report module's model/data helpers.
_spec = importlib.util.spec_from_file_location("fig4report", ROOT / "scripts/figure04_segmentation/report.py")
rep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rep)

EXP = ROOT / "experiments/figure04_segmentation/task_aware"
FIGS = EXP / "figures"
TILES = FIGS / "plot_tiles"

# (compression, pattern) -> paper panel id
SHORT = {
    ("x64", "random_fixed"): "C1",
    ("x64", "learnable_frequency"): "C2",
    ("x256", "random_fixed"): "D1",
    ("x256", "learnable_frequency"): "D2",
    ("x1024", "random_fixed"): "E1",
    ("x1024", "learnable_frequency"): "E2",
}
LABELS = {
    "A": "A  GT image",
    "B": "B  pseudo-GT mask",
    "C1": "C1  ×64 pseudo-random",
    "C2": "C2  ×64 learnable",
    "D1": "D1  ×256 pseudo-random",
    "D2": "D2  ×256 learnable",
    "E1": "E1  ×1024 pseudo-random",
    "E2": "E2  ×1024 learnable",
}
MASK_CMAP = "viridis"
GT_CMAP = "viridis"
PATTERN_CMAP = "gray"


def _save_tile(arr: np.ndarray, path: Path, cmap: str, vmin, vmax) -> None:
    """Write a colormapped tile at native pixel resolution (no axes/padding)."""
    plt.imsave(path, np.asarray(arr), cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")


def _stretch(img: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(img, (lo_p, hi_p))
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Figure 4 panels as individual tiles")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5, help="number of shared test-image columns")
    args = ap.parse_args()

    device = rep.resolve_device(args.device)
    seed = args.seed
    TILES.mkdir(parents=True, exist_ok=True)

    imgs, masks = rep._test_examples(seed, args.k)
    k = len(imgs)
    x = torch.stack(imgs).to(device)

    manifest: dict = {
        "seed": seed,
        "k": k,
        "colormaps": {"A_and_masks": MASK_CMAP, "F_patterns": PATTERN_CMAP},
        "note": (
            "Individual panels of the paper-style Figure 4 (latest TrackMate-mask "
            "re-run). A_col*.png are percentile-stretched for display; "
            "A_col*_raw.png are unstretched (vmin/vmax=0/1). Masks are binary {0,1} "
            "shown with viridis (0=purple, 1=yellow). F_*.png are illumination "
            "patterns (grayscale). Grid layout: rows [A, B, C1, C2, D1, D2, E1, E2] "
            "x k columns; row F = one pattern per C1..E2 cell."
        ),
        "row_order": ["A", "B", "C1", "C2", "D1", "D2", "E1", "E2"],
        "thresholds": {},
        "tiles": {"A": [], "A_raw": [], "B": [], "C1": [], "C2": [], "D1": [], "D2": [], "E1": [], "E2": [], "F": {}},
    }

    # ----- rows A (GT) and B (pseudo-GT mask) ----- #
    gt_np = [imgs[j][0].numpy() for j in range(k)]
    mask_np = [masks[j][0].numpy() for j in range(k)]
    for j in range(k):
        a_path = TILES / f"A_col{j}.png"
        a_raw_path = TILES / f"A_col{j}_raw.png"
        b_path = TILES / f"B_col{j}.png"
        _save_tile(_stretch(gt_np[j]), a_path, GT_CMAP, 0.0, 1.0)
        _save_tile(gt_np[j], a_raw_path, GT_CMAP, 0.0, 1.0)
        _save_tile(mask_np[j], b_path, MASK_CMAP, 0.0, 1.0)
        manifest["tiles"]["A"].append(a_path.name)
        manifest["tiles"]["A_raw"].append(a_raw_path.name)
        manifest["tiles"]["B"].append(b_path.name)

    # ----- rows C1..E2 (predicted masks) + F (patterns) ----- #
    preds: dict[str, np.ndarray] = {}
    patterns: dict[str, np.ndarray] = {}
    for comp, pattern, _label in rep.ROW_ORDER:
        short = SHORT[(comp, pattern)]
        summ = rep._summary(comp, pattern, seed)
        if summ is None:
            raise FileNotFoundError(f"missing run summary for {comp}/{pattern}; train first")
        learnable = pattern == "learnable_frequency"
        model = rep._load_model(comp, pattern, learnable, seed, device)
        thr = float(summ.get("selected_threshold", 0.5))
        manifest["thresholds"][short] = thr
        with torch.no_grad():
            out = model(x, sigmoid_m=rep.EVAL_M, apply_noise=False)
            pmask = (out["seg_prob"] > thr).float().cpu().numpy()[:, 0]
            pat = model.microscope.pattern_generator(sigmoid_m=rep.EVAL_M).detach().cpu().numpy()[0, 0]
        preds[short] = pmask
        patterns[short] = pat
        for j in range(k):
            p = TILES / f"{short}_col{j}.png"
            _save_tile(pmask[j], p, MASK_CMAP, 0.0, 1.0)
            manifest["tiles"][short].append(p.name)
        fpath = TILES / f"F_{short}.png"
        _save_tile(pat, fpath, PATTERN_CMAP, float(pat.min()), float(pat.max()))
        manifest["tiles"]["F"][short] = fpath.name
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (TILES / "tiles_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[tiles] wrote {len(list(TILES.glob('*.png')))} tiles -> {TILES}", flush=True)

    # ----- labelled paper-style composite (viridis) ----- #
    row_ids = ["A", "B", "C1", "C2", "D1", "D2", "E1", "E2"]
    row_arr = {
        "A": [_stretch(g) for g in gt_np],
        "B": mask_np,
        **{sid: [preds[sid][j] for j in range(k)] for sid in ["C1", "C2", "D1", "D2", "E1", "E2"]},
    }
    n_rows = len(row_ids)
    fig, axes = plt.subplots(n_rows, k, figsize=(2.1 * k, 2.1 * n_rows))
    if k == 1:
        axes = axes.reshape(n_rows, 1)
    for r, sid in enumerate(row_ids):
        for j in range(k):
            axes[r, j].imshow(row_arr[sid][j], cmap=MASK_CMAP, vmin=0, vmax=1)
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
        axes[r, 0].set_ylabel(LABELS[sid], rotation=0, ha="right", va="center", fontsize=10, labelpad=8)
    fig.suptitle(
        "Figure 4 (BBBC022 proxy, TrackMate pseudo-GT): pseudo-random vs. learnable Hᵗ",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.07, 0, 1, 0.98))
    grid_path = FIGS / "figure4_paper_layout_viridis.png"
    fig.savefig(grid_path, dpi=130)
    plt.close(fig)

    # ----- panel F strip (patterns for C1..E2) ----- #
    f_ids = ["C1", "C2", "D1", "D2", "E1", "E2"]
    figf, axf = plt.subplots(1, len(f_ids), figsize=(2.4 * len(f_ids), 2.7))
    for a, sid in zip(axf, f_ids):
        a.imshow(patterns[sid], cmap=PATTERN_CMAP)
        a.set_title(LABELS[sid].split("  ", 1)[-1], fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
    figf.suptitle("F) Representative illumination patterns Hᵗ (eval m=10)", fontsize=12)
    figf.tight_layout(rect=(0, 0, 1, 0.9))
    f_path = FIGS / "figure4_panel_F_viridis.png"
    figf.savefig(f_path, dpi=130)
    plt.close(figf)

    # ----- combined (grid + F) ----- #
    grid_img = plt.imread(grid_path)
    f_img = plt.imread(f_path)
    gh, gw = grid_img.shape[:2]
    fh, fw = f_img.shape[:2]
    scale = gw / fw
    figc = plt.figure(figsize=(gw / 130, (gh + fh * scale) / 130))
    gs = figc.add_gridspec(2, 1, height_ratios=[gh, fh * scale], hspace=0.02)
    a0 = figc.add_subplot(gs[0])
    a0.imshow(grid_img)
    a0.axis("off")
    a1 = figc.add_subplot(gs[1])
    a1.imshow(f_img)
    a1.axis("off")
    full_path = FIGS / "figure4_full_viridis.png"
    figc.savefig(full_path, dpi=130, bbox_inches="tight")
    plt.close(figc)
    print(f"[tiles] wrote composites: {grid_path.name}, {f_path.name}, {full_path.name}", flush=True)


if __name__ == "__main__":
    main()
