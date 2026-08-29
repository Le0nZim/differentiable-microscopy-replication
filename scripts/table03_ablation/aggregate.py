#!/usr/bin/env python3
"""Aggregate AM-3 outputs into the required deliverables.

Produces, under experiments/table03_ablation/:
  * aggregate_summary.json   (top-level: proxy ablation + patchmnist + trainsize + overfit)
  * metrics_by_seed.csv      (every run, every track)
  * figures/qualitative_ABCD.png   (GT + A/B/C/D reconstructions, shared examples)
  * figures/learned_patterns_BCD.png (learned H_t for B/C/D)
  * figures/curves_ABCD.png  (train/val MSE curves for A/B/C/D, seed 42)

Read-only over experiment data; writes only summary/figure artifacts.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/table03_ablation"
TRACK2 = OUT / "track2_proxy_bbbc022"
PATCH = OUT / "patchmnist_sanity"
TRAIN = OUT / "trainsize_sweep"
OVERFIT = OUT / "overfit_diagnostics"
PAPER_MSE = {"A": 0.0042, "B": 0.0038, "C": 0.0029, "D": 0.0041}
PAPER_SSIM = {"A": 0.7872, "B": 0.7950, "C": 0.8426, "D": 0.7857}


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _runs(root: Path) -> list[dict]:
    out = []
    for p in sorted(root.glob("*/metrics/run_summary.json")):
        out.append(json.loads(p.read_text()))
    return out


def metrics_by_seed_csv() -> None:
    rows = []
    for track, root in [("proxy_bbbc022", TRACK2), ("patchmnist_sanity", PATCH), ("trainsize", TRAIN), ("overfit", OVERFIT)]:
        for r in _runs(root):
            rows.append({
                "track": track,
                "variant": r.get("variant"),
                "seed": r.get("seed"),
                "run_dir": Path(r.get("run_dir", "")).name,
                "test_mse": r.get("test_mse"),
                "test_ssim": r.get("test_ssim"),
                "best_val_mse": r.get("best_val_mse"),
                "min_train_mse": r.get("min_train_mse"),
                "overfit_gap": r.get("overfit_gap"),
                "best_m": r.get("best_m"),
            })
    with (OUT / "metrics_by_seed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["track", "variant", "seed", "run_dir", "test_mse",
                                          "test_ssim", "best_val_mse", "min_train_mse",
                                          "overfit_gap", "best_m"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote metrics_by_seed.csv ({len(rows)} rows)")


def qualitative_panel(seed: int = 42) -> None:
    variants = ["A", "B", "C", "D"]
    tensors = {}
    for L in variants:
        p = TRACK2 / f"{L}_seed{seed}" / "figures" / "qualitative_tensors.pt"
        if p.exists():
            tensors[L] = torch.load(p, map_location="cpu")
    if not tensors:
        print("no qualitative tensors yet")
        return
    gt = tensors[variants[0]]["gt"] if variants[0] in tensors else next(iter(tensors.values()))["gt"]
    n = min(4, gt.shape[0])
    rows = 1 + len(variants)
    fig, axes = plt.subplots(rows, n, figsize=(2.3 * n, 2.3 * rows))
    for i in range(n):
        axes[0, i].imshow(gt[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
    axes[0, 0].set_title("GT", loc="left")
    for r, L in enumerate(variants, start=1):
        for i in range(n):
            ax = axes[r, i]
            if L in tensors:
                ax.imshow(tensors[L]["recon"][i, 0].numpy().clip(0, 1), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        label = {"A": "A fixed+transpose", "B": "B learn+transpose",
                 "C": "C learn+locality(best in paper)", "D": "D learn+locality, no-freq"}[L]
        axes[r, 0].set_title(label, loc="left", fontsize=8)
    fig.suptitle(f"AM-3 BBBC022 proxy ×16 ablation (seed {seed}) — NOT paper U2OS reproduction")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "qualitative_ABCD.png", dpi=120)
    plt.close(fig)
    print("wrote figures/qualitative_ABCD.png")


def patterns_panel(seed: int = 42) -> None:
    variants = ["B", "C", "D"]
    pats = {}
    for L in variants:
        p = TRACK2 / f"{L}_seed{seed}" / "learned_patterns" / "H_t.pt"
        if p.exists():
            pats[L] = torch.load(p, map_location="cpu")
    if not pats:
        print("no learned patterns yet")
        return
    T = next(iter(pats.values())).shape[0]
    rows = len(variants)
    fig, axes = plt.subplots(rows, T, figsize=(2.2 * T, 2.2 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    for r, L in enumerate(variants):
        for t in range(T):
            ax = axes[r, t]
            if L in pats:
                ax.imshow(pats[L][t, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        axes[r, 0].set_title(f"{L}: learned H_t", loc="left", fontsize=9)
    fig.suptitle(f"AM-3 learned illumination patterns B/C/D (seed {seed})")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "learned_patterns_BCD.png", dpi=120)
    plt.close(fig)
    print("wrote figures/learned_patterns_BCD.png")


def curves_panel(seed: int = 42) -> None:
    variants = ["A", "B", "C", "D"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for L in variants:
        p = TRACK2 / f"{L}_seed{seed}" / "metrics" / "step_log.csv"
        if not p.exists():
            continue
        rows = list(csv.DictReader(p.open()))
        steps = [int(r["step"]) for r in rows]
        ax.plot(steps, [float(r["train_mse"]) for r in rows], "--", alpha=0.6, label=f"{L} train")
        ax.plot(steps, [float(r["val_mse"]) for r in rows], "-", label=f"{L} val")
    ax.set_xlabel("step"); ax.set_ylabel("MSE"); ax.set_yscale("log")
    ax.set_title(f"AM-3 BBBC022 proxy: train/val MSE (seed {seed})")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "curves_ABCD.png", dpi=120)
    plt.close(fig)
    print("wrote figures/curves_ABCD.png")


def top_summary() -> None:
    summary = {
        "title": "AM-3 (Table 3 / Fig. 10) resolution summary",
        "data_status": "U2OS unavailable; BBBC022 (U2OS Cell-Painting, Hoechst) SUBSTITUTE proxy.",
        "paper_table3": {"mse": PAPER_MSE, "ssim": PAPER_SSIM, "best": "C"},
        "track2_proxy_bbbc022": _load(TRACK2 / "aggregate_summary.json"),
        "patchmnist_sanity": _load(PATCH / "aggregate_summary.json"),
        "trainsize_sweep": _load(TRAIN / "aggregate_summary.json"),
        "overfit_gate": _load(OVERFIT / "overfit_gate.json"),
    }
    (OUT / "aggregate_summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote aggregate_summary.json")


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    metrics_by_seed_csv()
    qualitative_panel()
    patterns_panel()
    curves_panel()
    top_summary()


if __name__ == "__main__":
    main()
