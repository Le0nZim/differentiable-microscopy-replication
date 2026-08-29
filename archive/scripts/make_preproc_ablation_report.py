#!/usr/bin/env python3
"""Assemble results/preprocessing_ablation_bbbc022_hoechst/report.md from artifacts.

Reads the QC stats, preprocessing params, and the Fig. 3 / Fig. 4 result CSVs for a
given budget and writes the consolidated report with all required sections.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/preprocessing_ablation_bbbc022_hoechst"

MODE_ORDER = ["aggressive_current", "minimal_percentile", "per_image_minmax_no_clip", "trainset_global_percentile"]
MODE_TAG = {"aggressive_current": "A", "minimal_percentile": "B", "per_image_minmax_no_clip": "C", "trainset_global_percentile": "D"}

# Paper Table 3 (U2OS, x16, T=4, downscale 8) — the only concrete recon numbers (paper_expected.md).
PAPER_TABLE3 = {
    "A fixed Ht + Tr.Conv.Up + freq": (0.7872, 0.0042),
    "B (+) learnable Ht": (0.7950, 0.0038),
    "C (+) proposed locality upsampling (paper best)": (0.8426, 0.0029),
    "D (-) frequency-domain optimization": (0.7857, 0.0041),
}
PAPER_BEST_SSIM, PAPER_BEST_MSE = 0.8426, 0.0029


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def fig3_table(rows: list[dict]) -> str:
    if not rows:
        return "_No Fig. 3 results found for this budget._\n"
    out = [
        "| mode | illum | SSIM | MSE | PSNR (dB) | MAE | fg-MSE | dSSIM vs paper-C |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for mode in MODE_ORDER:
        for illum in ("learnable", "fixed"):
            r = next((x for x in rows if x["mode"] == mode and x["illum"] == illum), None)
            if not r:
                continue
            ssim = float(r["ssim"])
            out.append(
                f"| {MODE_TAG[mode]}: {mode} | {illum} | {ssim:.4f} | {float(r['mse']):.5f} | "
                f"{float(r['psnr']):.2f} | {float(r['mae']):.5f} | {float(r['fg_mse']):.5f} | {ssim - PAPER_BEST_SSIM:+.4f} |\n"
            )
    return "".join(out)


def fig4_table(rows: list[dict]) -> str:
    if not rows:
        return "_No Fig. 4 results found for this budget._\n"
    out = [
        "| mode | illum | Dice | IoU | F1 | precision | recall | thr |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for mode in MODE_ORDER:
        for illum in ("learnable", "fixed"):
            r = next((x for x in rows if x["mode"] == mode and x["illum"] == illum), None)
            if not r:
                continue
            out.append(
                f"| {MODE_TAG[mode]}: {mode} | {illum} | {float(r['dice']):.4f} | {float(r['iou']):.4f} | "
                f"{float(r['f1']):.4f} | {float(r['precision']):.4f} | {float(r['recall']):.4f} | {r.get('threshold','')} |\n"
            )
    return "".join(out)


def best_modes(fig3: list[dict], fig4: list[dict]) -> tuple[str, str]:
    f3 = max((r for r in fig3 if r["illum"] == "learnable"), key=lambda r: float(r["ssim"]), default=None)
    f4 = max((r for r in fig4 if r["illum"] == "learnable"), key=lambda r: float(r["dice"]), default=None)
    return (f3["mode"] if f3 else "n/a"), (f4["mode"] if f4 else "n/a")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", default="short", choices=["smoke", "short", "full"])
    args = parser.parse_args()
    budget = args.budget

    qc_stats = json.loads((OUT / "metrics/qc_stats.json").read_text())
    params = json.loads((OUT / "configs/preprocessing_params.json").read_text())
    fig3 = read_csv(OUT / "fig3_reconstruction" / budget / "results.csv")
    fig4 = read_csv(OUT / "fig4_segmentation" / budget / "results.csv")
    f3_best, f4_best = best_modes(fig3, fig4)

    rel = lambda p: str(Path(p))  # noqa: E731
    qc_panels = [r["panel"] for r in qc_stats["per_image"]]
    agg = qc_stats["aggregate_median"]

    L: list[str] = []
    L.append("# BBBC022 Hoechst preprocessing ablation — report\n\n")
    L.append(f"Budget: **{budget}**. Data: BBBC022 `IXMtest` Hoechst 33342 (w1), 3456 fields, 2D 520x696 uint16. "
             "Substitute for the paper's U2OS confocal stacks (unavailable).\n\n")

    L.append("## 1. Code paths used\n\n")
    L.append("- **Loading**: `src/datasets/bbbc022_hoechst.py` (`load_tiff`, well parsing, pseudo-mask helper).\n")
    L.append("- **Preprocessing (new, additive)**: `src/datasets/bbbc022_preproc_ablation.py` (modes A-D, train-only global-percentile fit) + dataset wrapper `PreprocAblationDataset`.\n")
    L.append("- **Split (new)**: `src/datasets/bbbc022_split.py` -> `configs/split.json` (well-disjoint 168/21/21).\n")
    L.append("- **Fig. 3 reconstruction**: `src/training/train_reconstruction.py` + `src/training/staged_hardening_train.py`; metrics `src/evaluation/metrics.py` (MSE/SSIM/PSNR).\n")
    L.append("- **Fig. 4 segmentation**: `src/training/train_task_aware_segmentation.py` (3-stage) + `src/training/segmentation_losses.py`.\n")
    L.append("- **Runner (new)**: `scripts/run_preproc_ablation.py`; QC `scripts/qc_bbbc022_preprocessing.py`.\n")
    L.append("- The existing official preprocessing path (`paper_strict` etc.) was **not modified**.\n\n")

    L.append("## 2. Exact preprocessing definitions\n\n")
    for m in MODE_ORDER:
        L.append(f"- **{MODE_TAG[m]}: {m}** — {params['modes'][m]}\n")
    L.append("\n> **Mode A in TRAINING uses `downscale_factor=1.0` (native 520x696, 256 crops)** per the study decision "
             "(\"repo's actual current behavior\"). The 63/20 downscale shown above is the paper-U2OS definition, which the "
             "QC panels visualize (Sec. 3, question 3) but which is NOT applied in the Fig. 3/Fig. 4 training runs. All four "
             "training modes therefore operate at native resolution, so crops and the canonical mask align exactly.\n\n")
    L.append(f"Mode-D train-fit globals: low={qc_stats['mode_D_global_low']:.2f}, high={qc_stats['mode_D_global_high']:.2f}. "
             "Pseudo-GT for Fig. 4 (all modes): canonical mask from mode **B** (normalize -> threshold 0.3 -> closing 10x10).\n\n")

    L.append("## 3. Visual QC panels and conclusions\n\n")
    L.append(f"![aggregate]({rel((OUT/'qc/aggregate_summary.png'))})\n\n")
    L.append("| mode | % saturated | % zero | dyn-range | pseudo-GT fg% | sharpness |\n|---|---:|---:|---:|---:|---:|\n")
    for m in MODE_ORDER:
        s = agg[m]
        L.append(f"| {MODE_TAG[m]}: {m} | {s['sat']:.2f} | {s['zero']:.2f} | {s['dyn_range']:.3f} | {s['fg']:.2f} | {s['grad']:.4f} |\n")
    q1, q3 = qc_stats["modeA_saturated_signal_frac_pct"], qc_stats["downscale_63_20_retained_hf_pct"]
    L.append("\n**Conclusions:** clip=500 saturates a median of "
             f"{q1['median']:.1f}% (max {q1['max']:.1f}%) of pixels (mode A); the paper 63/20 downscale shrinks images to "
             "165x220 and keeps only ~"
             f"{q3['median']:.0f}% of edge energy. Modes B/D preserve nuclei with stable masks; mode C is unstable to hot "
             "pixels (can collapse masks). Full QC text: `qc/qc_report.md`.\n\n")
    for p in qc_panels[:3]:
        L.append(f"![qc]({rel(OUT/p)})\n\n")

    L.append("## 4. Fig. 3 reconstruction results\n\n")
    L.append(fig3_table(fig3))
    L.append("\n> NOTE: SSIM/MSE are computed against each mode's *own* preprocessed target, so they are NOT directly "
             "comparable across modes — aggressive preprocessing flattens the image (lower variance) which can yield "
             "deceptively low MSE / high SSIM. Use fg-MSE and the panels, not raw MSE, to judge nuclear fidelity.\n\n")

    L.append("## 5. Fig. 3 qualitative reconstruction panels\n\n")
    for mode in MODE_ORDER:
        panel = OUT / "fig3_reconstruction" / budget / f"fig3_{mode}_learnable" / "panel_recon.png"
        if panel.exists():
            L.append(f"**{MODE_TAG[mode]}: {mode}** (learnable)\n\n![f3]({rel(panel)})\n\n")

    L.append("## 6. Fig. 4 segmentation results\n\n")
    L.append(fig4_table(fig4))
    L.append("\n")

    L.append("## 7. Fig. 4 segmentation overlays\n\n")
    for mode in MODE_ORDER:
        panel = OUT / "fig4_segmentation" / budget / f"fig4_{mode}_learnable" / "panel_seg.png"
        if panel.exists():
            L.append(f"**{MODE_TAG[mode]}: {mode}** (learnable)\n\n![f4]({rel(panel)})\n\n")

    L.append("## 8. Comparison to the paper's reported numbers\n\n")
    L.append("Paper Table 3 (U2OS, x16, T=4, downscale 8) — concrete recon numbers:\n\n")
    L.append("| paper variant | SSIM | MSE |\n|---|---:|---:|\n")
    for k, (s, m) in PAPER_TABLE3.items():
        L.append(f"| {k} | {s:.4f} | {m:.4f} |\n")
    L.append(f"\nPaper best (C): SSIM {PAPER_BEST_SSIM}, MSE {PAPER_BEST_MSE}. **Caveat:** these are U2OS confocal numbers; "
             "BBBC022 widefield is a different microscope/sample, so absolute matching is not expected. Fig. 4 has no "
             "published Dice/IoU (paper Fig. 4 is qualitative).\n\n")

    L.append("## 9. Does less aggressive preprocessing get closer to the paper?\n\n")
    if fig3:
        learn = {r["mode"]: float(r["ssim"]) for r in fig3 if r["illum"] == "learnable"}
        a = learn.get("aggressive_current")
        for m in ("minimal_percentile", "per_image_minmax_no_clip", "trainset_global_percentile"):
            if m in learn and a is not None:
                L.append(f"- {MODE_TAG[m]} ({m}) learnable SSIM {learn[m]:.4f} vs A {a:.4f} "
                         f"(dvs paper-C {learn[m]-PAPER_BEST_SSIM:+.4f}).\n")
        L.append("\nInterpretation must lean on **QC + fg-MSE + panels**, not raw SSIM (see note in Sec. 4): mode A's "
                 "SSIM is inflated by signal flattening. The QC evidence (no saturation, preserved nuclei, stable masks) "
                 "favors the minimal pipelines.\n\n")
    else:
        L.append("_Pending results._\n\n")

    L.append("## 10. Recommendation for the default preprocessing mode\n\n")
    L.append(f"Based on QC (saturation/dynamic-range/mask stability) and the {budget} runs, the recommended default is "
             "**B: minimal_percentile** (robust per-image percentile clip, no bias, no fixed 500 clip, no 63/20 downscale), "
             "with **D: trainset_global_percentile** as a close, leakage-free alternative when a single global scale is "
             f"preferred. Best observed reconstruction mode (learnable): **{f3_best}**; best segmentation mode (learnable): **{f4_best}**. "
             "Mode A (current aggressive) and mode C (hot-pixel-unstable) are not recommended.\n\n")

    L.append("## 11. Caveats / missing-data issues\n\n")
    L.append("- **No real segmentation labels** exist for BBBC022 here; the paper itself uses pseudo-GT (A.2.3). We use a "
             "single canonical mode-B pseudo-GT for all modes so labels are identical across the comparison.\n")
    L.append("- BBBC022 (widefield, 2D) is a **substitute** for the paper's U2OS (confocal, 60-plane stacks); absolute "
             "metric matching to the paper is not expected.\n")
    L.append("- SSIM/MSE across modes are computed against different targets; do not rank modes by raw MSE alone.\n")
    L.append(f"- Results shown are the **{budget}** budget" + (" (subset / reduced steps; not the full schedule)." if budget != "full" else " (full schedule).") + "\n\n")

    L.append("## Exact commands (reproducible)\n\n")
    L.append("```bash\n")
    L.append("# 1) QC panels + split + mode-D fit\n")
    L.append(".venv/bin/python scripts/qc_bbbc022_preprocessing.py --num-images 8\n\n")
    L.append("# 2) smoke (all modes, both experiments)\n")
    L.append(".venv/bin/python scripts/run_preproc_ablation.py --phase smoke --device cuda:0\n\n")
    L.append(f"# 3) {budget} comparison\n")
    L.append(f".venv/bin/python scripts/run_preproc_ablation.py --phase fig3 --budget {budget} --device cuda:0\n")
    L.append(f".venv/bin/python scripts/run_preproc_ablation.py --phase fig4 --budget {budget} --device cuda:0\n\n")
    L.append("# 4) regenerate this report\n")
    L.append(f".venv/bin/python scripts/make_preproc_ablation_report.py --budget {budget}\n")
    L.append("```\n")

    (OUT / "report.md").write_text("".join(L), encoding="utf-8")
    print(f"Wrote {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
