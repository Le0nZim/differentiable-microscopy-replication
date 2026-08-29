#!/usr/bin/env python3
"""Render paper-style Figure 10 (ablation A/B/C/D) from the trained Fig-10 runs.

Layout mirrors the paper (3 rows x 4 cols):
    row a : reconstruction of held-out test sample #1  (viridis)
    row b : one learned illumination pattern H_t        (grayscale)
    row c : reconstruction of held-out test sample #2  (viridis)
    cols  : A, B, C, D   (C = proposed method, highlighted with a green box)

Reads the artifacts saved by scripts/figure10_ablation/train.py (run_am3_table3.run_one):
    runs/<L>_seed<seed>/figures/qualitative_tensors.pt   {gt, recon}
    runs/<L>_seed<seed>/learned_patterns/H_t.pt          [T,1,H,W]
    runs/<L>_seed<seed>/metrics/run_summary.json         test_ssim / test_mse

Also emits a Table-3 comparison panel (ours-on-substitute vs paper-U2OS).
Everything is explicitly labelled as a BBBC022 substitute reproduction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/figure10_ablation"

LETTERS = ["A", "B", "C", "D"]
COL_TITLE = {
    "A": "A\nfixed $H_t$ + Tr.Conv.\n+ freq",
    "B": "B\nlearnable $H_t$ + Tr.Conv.\n+ freq",
    "C": "C\nlearnable $H_t$ + locality\n+ freq  (proposed)",
    "D": "D\nlearnable $H_t$ + locality\nNO freq",
}
PAPER_TABLE3 = {
    "A": {"ssim": 0.7872, "mse": 0.0042},
    "B": {"ssim": 0.7950, "mse": 0.0038},
    "C": {"ssim": 0.8426, "mse": 0.0029},
    "D": {"ssim": 0.7857, "mse": 0.0041},
}
DEFAULT_DATA_LABEL = "BBBC022 Hoechst SUBSTITUTE (same data as repo Fig 3)"
DEFAULT_SHORT_LABEL = "substitute"
DEFAULT_TABLE_LABEL = "BBBC022 substitute, Fig-3 data"


def _load(runs: Path, letter: str, seed: int):
    d = runs / f"{letter}_seed{seed}"
    qt = torch.load(d / "figures" / "qualitative_tensors.pt", map_location="cpu")
    ht = torch.load(d / "learned_patterns" / "H_t.pt", map_location="cpu")
    summ = json.loads((d / "metrics" / "run_summary.json").read_text())
    return {
        "gt": qt["gt"].float(),
        "recon": qt["recon"].float().clamp(0, 1),
        "H_t": ht.float(),
        "test_ssim": summ.get("test_ssim"),
        "test_mse": summ.get("test_mse"),
    }


def _pick_two_samples(gt: torch.Tensor) -> tuple[int, int]:
    """Pick two high-content, distinct test crops for the two recon rows."""
    n = gt.shape[0]
    means = gt.view(n, -1).mean(dim=1)
    order = torch.argsort(means, descending=True).tolist()
    i0 = order[0]
    # second pick: most different from i0 among the top-content half
    top = order[: max(2, n // 2)]
    flat = gt.view(n, -1)
    best, bj = -1.0, top[1] if len(top) > 1 else order[min(1, n - 1)]
    for j in top[1:]:
        dist = float((flat[j] - flat[i0]).pow(2).mean())
        if dist > best:
            best, bj = dist, j
    return i0, bj


def render_qualitative(data: dict, out_path: Path, *, i0: int, i1: int, pat_idx: int,
                       data_label: str = DEFAULT_DATA_LABEL) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(11.0, 8.6))
    plt.subplots_adjust(left=0.055, right=0.995, top=0.90, bottom=0.11,
                        wspace=0.03, hspace=0.03)

    for col, L in enumerate(LETTERS):
        d = data[L]
        recon = d["recon"]
        ht = d["H_t"]
        n = recon.shape[0]
        a = recon[min(i0, n - 1), 0].numpy()
        c = recon[min(i1, n - 1), 0].numpy()
        p = ht[min(pat_idx, ht.shape[0] - 1), 0].numpy()

        axes[0, col].imshow(a, cmap="viridis", vmin=0, vmax=1)
        axes[1, col].imshow(p, cmap="gray", vmin=0, vmax=1)
        axes[2, col].imshow(c, cmap="viridis", vmin=0, vmax=1)

        for r in range(3):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])

        ssim = d["test_ssim"]
        mse = d["test_mse"]
        metric = (f"SSIM {ssim:.3f} | MSE {mse:.4f}"
                  if ssim is not None and mse is not None else "")
        axes[2, col].set_xlabel(COL_TITLE[L] + (f"\n{metric}" if metric else ""),
                                fontsize=9, labelpad=6)

    # row labels a / b / c
    for r, lbl in enumerate(["a", "b", "c"]):
        axes[r, 0].set_ylabel(lbl, rotation=0, fontsize=15, fontweight="bold",
                              labelpad=14, va="center")

    # green highlight box around column C (proposed method), spanning all 3 rows
    ci = LETTERS.index("C")
    boxes = [axes[r, ci].get_position() for r in range(3)]
    x0 = min(b.x0 for b in boxes)
    x1 = max(b.x1 for b in boxes)
    y0 = min(b.y0 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    pad = 0.006
    fig.add_artist(Rectangle((x0 - pad, y0 - pad), (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad,
                             transform=fig.transFigure, fill=False,
                             edgecolor="#00b050", linewidth=3.0, zorder=10))

    fig.suptitle(
        "Figure 10 (reproduction) - Ablation A/B/C/D at x16 (T=4, 8x8:1) on the "
        f"{data_label}\n"
        "rows: a) test recon #1   b) learned $H_t$   c) test recon #2   |   "
        "SUBSTITUTE data, NOT paper U2OS - only A/B/C/D ordering is comparable",
        fontsize=10.5)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def render_table3(data: dict, out_path: Path, *, table_label: str = DEFAULT_TABLE_LABEL,
                  short_label: str = DEFAULT_SHORT_LABEL) -> None:
    ours_ssim = [data[L]["test_ssim"] for L in LETTERS]
    ours_mse = [data[L]["test_mse"] for L in LETTERS]
    paper_ssim = [PAPER_TABLE3[L]["ssim"] for L in LETTERS]
    paper_mse = [PAPER_TABLE3[L]["mse"] for L in LETTERS]

    fig, (axt, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 4.2),
                                        gridspec_kw={"width_ratios": [1.25, 1, 1]})
    x = np.arange(4)
    w = 0.38

    # text table
    axt.axis("off")
    rows = [["Var", "ours SSIM", "ours MSE", "paper SSIM", "paper MSE"]]
    for i, L in enumerate(LETTERS):
        s = f"{ours_ssim[i]:.4f}" if ours_ssim[i] is not None else "n/a"
        m = f"{ours_mse[i]:.4f}" if ours_mse[i] is not None else "n/a"
        rows.append([L, s, m, f"{paper_ssim[i]:.4f}", f"{paper_mse[i]:.4f}"])
    tbl = axt.table(cellText=rows, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for j in range(5):
        tbl[0, j].set_facecolor("#dddddd")
    # highlight C row
    for j in range(5):
        tbl[3, j].set_facecolor("#e2f4e2")
    axt.set_title(f"Table 3 comparison\n({short_label} vs paper-U2OS)", fontsize=10)

    ax1.bar(x - w / 2, [s if s is not None else 0 for s in ours_ssim], w,
            label=f"ours ({short_label})", color="#4a90d9")
    ax1.bar(x + w / 2, paper_ssim, w, label="paper (U2OS)", color="#bbbbbb")
    ax1.set_xticks(x)
    ax1.set_xticklabels(LETTERS)
    ax1.set_title("SSIM (higher better)")
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(x - w / 2, [m if m is not None else 0 for m in ours_mse], w,
            label=f"ours ({short_label})", color="#4a90d9")
    ax2.bar(x + w / 2, paper_mse, w, label="paper (U2OS)", color="#bbbbbb")
    ax2.set_xticks(x)
    ax2.set_xticklabels(LETTERS)
    ax2.set_title("MSE (lower better)")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Figure 10 / Table 3 ablation - ours ({table_label}) vs paper (U2OS). "
        "Magnitudes differ (different data); compare the A->B->C ORDERING.",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(EXP / "runs"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--i0", type=int, default=-1, help="test index for recon row a (-1=auto)")
    ap.add_argument("--i1", type=int, default=-1, help="test index for recon row c (-1=auto)")
    ap.add_argument("--pattern-index", type=int, default=0)
    ap.add_argument("--out-dir", default=str(EXP / "figures"))
    ap.add_argument("--data-label", default=DEFAULT_DATA_LABEL,
                    help="dataset description used in the qualitative-panel title")
    ap.add_argument("--table-label", default=DEFAULT_TABLE_LABEL,
                    help="dataset description used in the Table-3 comparison title")
    ap.add_argument("--short-label", default=DEFAULT_SHORT_LABEL,
                    help="short dataset tag used in the Table-3 legends")
    ap.add_argument("--mirror-dir", default=str(ROOT / "results/reproduced_figures/fig10"),
                    help="second directory the figures are rendered into ('' to skip)")
    args = ap.parse_args()

    runs = Path(args.runs)
    data = {L: _load(runs, L, args.seed) for L in LETTERS}

    # sanity: GT should be identical across variants (deterministic test loader)
    ref_gt = data["A"]["gt"]
    for L in LETTERS[1:]:
        diff = float((data[L]["gt"] - ref_gt).abs().max())
        if diff > 1e-4:
            print(f"WARN: GT for {L} differs from A (max abs {diff:.2e}); "
                  "sample indices may not align.", flush=True)

    i0, i1 = args.i0, args.i1
    if i0 < 0 or i1 < 0:
        ai0, ai1 = _pick_two_samples(ref_gt)
        i0 = ai0 if i0 < 0 else i0
        i1 = ai1 if i1 < 0 else i1
    print(f"using test recon samples i0={i0}, i1={i1}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_qualitative(data, out_dir / "figure10_paper_style.png",
                       i0=i0, i1=i1, pat_idx=args.pattern_index, data_label=args.data_label)
    render_table3(data, out_dir / "figure10_table3_comparison.png",
                  table_label=args.table_label, short_label=args.short_label)

    # mirror into the shared reproduced-figures dir
    if args.mirror_dir:
        shared = Path(args.mirror_dir)
        shared.mkdir(parents=True, exist_ok=True)
        render_qualitative(data, shared / "figure10_paper_style.png",
                           i0=i0, i1=i1, pat_idx=args.pattern_index, data_label=args.data_label)
        render_table3(data, shared / "figure10_table3_comparison.png",
                      table_label=args.table_label, short_label=args.short_label)


if __name__ == "__main__":
    main()
