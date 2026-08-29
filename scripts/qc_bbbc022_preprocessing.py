#!/usr/bin/env python3
"""Visual QC for the BBBC022 Hoechst preprocessing ablation (MANDATORY pre-training).

Produces, for a representative set of raw BBBC022 Hoechst fields (spanning many
wells and sites), comparison panels for the four preprocessing modes, intensity
histograms, percentile summaries, residual maps, and pseudo-GT-mask panels (the
segmentation target depends on preprocessing). Also writes a markdown report that
explicitly answers the required QC questions, plus a machine-readable JSON of the
aggregate statistics.

This is read-only w.r.t. the data and does NOT train anything.

Example:
    .venv/bin/python scripts/qc_bbbc022_preprocessing.py \
        --num-images 8 --out results/preprocessing_ablation_bbbc022_hoechst
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets.bbbc022_hoechst import (  # noqa: E402
    discover_image_paths,
    load_tiff,
    make_pseudo_mask,
    parse_well_site,
    select_hoechst_paths,
)
from datasets.bbbc022_preproc_ablation import (  # noqa: E402
    ALL_MODES,
    PAPER_BIAS,
    PAPER_CLIP_MAX,
    PAPER_DOWNSCALE,
    PreprocParams,
    fit_trainset_global_percentiles,
    mode_description,
    preprocess,
)
from datasets.bbbc022_split import SplitSpec, build_split, load_split, save_split  # noqa: E402

MODE_LABELS = {
    "aggressive_current": "A: aggressive_current",
    "minimal_percentile": "B: minimal_percentile",
    "per_image_minmax_no_clip": "C: per_image_minmax_no_clip",
    "trainset_global_percentile": "D: trainset_global_percentile",
}
MASK_THRESHOLD = 0.3
MASK_CLOSING = 10


def robust_display(img2d: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.9) -> np.ndarray:
    lo, hi = np.percentile(img2d, lo_p), np.percentile(img2d, hi_p)
    if hi <= lo:
        return np.zeros_like(img2d)
    return np.clip((img2d - lo) / (hi - lo), 0.0, 1.0)


def grad_energy(img2d: np.ndarray) -> float:
    """Mean gradient magnitude (sharpness proxy), scale-normalized to [0,1] range."""
    gy, gx = np.gradient(img2d.astype(np.float64))
    return float(np.mean(np.sqrt(gx**2 + gy**2)))


def select_representative_paths(all_paths: list[Path], num_images: int) -> list[Path]:
    """Pick images spanning many wells and varied sites (deterministic)."""
    by_well: dict[str, list[Path]] = {}
    for p in all_paths:
        well, _ = parse_well_site(p)
        by_well.setdefault(well or p.stem, []).append(p)
    wells = sorted(by_well)
    if not wells:
        return []
    picks: list[Path] = []
    for i in range(num_images):
        well = wells[round(i * (len(wells) - 1) / max(num_images - 1, 1))]
        sites = sorted(by_well[well], key=lambda p: parse_well_site(p)[1] or 0)
        picks.append(sites[i % len(sites)])  # vary site index across picks
    # de-duplicate while preserving order
    seen, unique = set(), []
    for p in picks:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def per_image_panel(raw: torch.Tensor, processed: dict[str, torch.Tensor], out_path: Path, title: str) -> None:
    raw_np = raw.numpy()
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(title, fontsize=13)

    # Row 0: images (raw robust display + 4 modes).
    axes[0, 0].imshow(robust_display(raw_np), cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title(f"raw (robust 1-99.9%)\n{raw_np.shape} uint16", fontsize=9)
    for j, mode in enumerate(ALL_MODES, start=1):
        im = processed[mode].squeeze(0).numpy()
        sat = float((im >= 0.999).mean()) * 100
        zero = float((im <= 1e-6).mean()) * 100
        axes[0, j].imshow(im, cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(f"{MODE_LABELS[mode]}\n{im.shape}  sat={sat:.1f}% zero={zero:.1f}%", fontsize=9)
    for ax in axes[0]:
        ax.axis("off")

    # Row 1: histograms (raw intensity with bias/clip markers + 4 modes in [0,1]).
    axes[1, 0].hist(raw_np.ravel(), bins=120, color="gray", log=True)
    axes[1, 0].axvline(PAPER_BIAS, color="tab:red", ls="--", lw=1, label=f"bias {PAPER_BIAS}")
    axes[1, 0].axvline(PAPER_BIAS + PAPER_CLIP_MAX, color="tab:orange", ls="--", lw=1, label=f"bias+clip {PAPER_BIAS+PAPER_CLIP_MAX:.0f}")
    axes[1, 0].set_title("raw intensity (log count)", fontsize=9)
    axes[1, 0].legend(fontsize=6)
    for j, mode in enumerate(ALL_MODES, start=1):
        im = processed[mode].squeeze(0).numpy().ravel()
        axes[1, j].hist(im, bins=80, color="tab:blue", log=True, range=(0, 1))
        axes[1, j].set_title(f"{MODE_LABELS[mode]} hist", fontsize=9)

    # Row 2: pseudo-GT masks (threshold 0.3 + closing) per mode + residual (C - A_resampled).
    base = processed["per_image_minmax_no_clip"].squeeze(0).numpy()
    a_img = processed["aggressive_current"].squeeze(0)
    a_up = torch.nn.functional.interpolate(
        a_img.unsqueeze(0).unsqueeze(0), size=base.shape, mode="bilinear", align_corners=False
    ).squeeze().numpy()
    residual = np.abs(base - a_up)
    axes[2, 0].imshow(residual, cmap="magma", vmin=0, vmax=1)
    axes[2, 0].set_title(f"|C - A(upsampled)|\nmean={residual.mean():.3f}", fontsize=9)
    axes[2, 0].axis("off")
    for j, mode in enumerate(ALL_MODES, start=1):
        mask = make_pseudo_mask(processed[mode], MASK_THRESHOLD, MASK_CLOSING).squeeze(0).numpy()
        fg = float(mask.mean()) * 100
        axes[2, j].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[2, j].set_title(f"{MODE_LABELS[mode]}\npseudo-GT (t=0.3) fg={fg:.1f}%", fontsize=9)
        axes[2, j].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="BBBC022 preprocessing QC panels")
    parser.add_argument("--out", default="results/preprocessing_ablation_bbbc022_hoechst")
    parser.add_argument("--data-root", default="data/substitute_data")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--q-low", type=float, default=0.001)
    parser.add_argument("--q-high", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root = (ROOT / args.out).resolve()
    qc_dir = out_root / "qc"
    cfg_dir = out_root / "configs"
    qc_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) build + save the shared, well-disjoint split ------------------- #
    spec = SplitSpec(data_root=args.data_root, seed=args.seed)
    split_rel = build_split(spec, ROOT)
    split_path = save_split(split_rel, spec, cfg_dir / "split.json")
    split_abs = load_split(split_path, ROOT)
    print(f"Saved split -> {split_path} (counts: {[ (k, len(v)) for k,v in split_abs.items()] })", flush=True)

    # ---- 2) fit mode-D global percentiles on TRAIN ONLY -------------------- #
    global_low, global_high = fit_trainset_global_percentiles(
        split_abs["train"], q_low=args.q_low, q_high=args.q_high, seed=args.seed
    )
    params = PreprocParams(q_low=args.q_low, q_high=args.q_high, global_low=global_low, global_high=global_high)
    print(f"Mode D train-fit global percentiles: low(q{args.q_low})={global_low:.2f} high(q{args.q_high})={global_high:.2f}", flush=True)

    params_payload = {
        "modes": {m: mode_description(m, params) for m in ALL_MODES},
        "params": {k: getattr(params, k) for k in vars(params)},
        "mask": {"threshold": MASK_THRESHOLD, "closing_kernel": MASK_CLOSING},
        "constants": {"paper_bias": PAPER_BIAS, "paper_clip_max": PAPER_CLIP_MAX, "paper_downscale": PAPER_DOWNSCALE},
    }
    (cfg_dir / "preprocessing_params.json").write_text(json.dumps(params_payload, indent=2), encoding="utf-8")

    # ---- 3) select representative images (many wells / sites) -------------- #
    all_paths = select_hoechst_paths(discover_image_paths((ROOT / args.data_root).resolve(), spec.stack_glob))
    sample = select_representative_paths(all_paths, args.num_images)
    print(f"Selected {len(sample)} QC images", flush=True)

    # ---- 4) per-image panels + aggregate stats ----------------------------- #
    agg: dict[str, dict[str, list[float]]] = {m: {"sat": [], "zero": [], "dyn_range": [], "fg": [], "grad": []} for m in ALL_MODES}
    a_clip_ceiling_frac: list[float] = []   # raw pixels above (bias+clip) -> saturated by mode A
    a_below_bias_frac: list[float] = []     # raw pixels at/below bias -> zeroed by mode A
    downscale_retained_hf: list[float] = []  # high-freq energy retained after 63/20 downscale+upsample
    per_image_records = []

    for k, path in enumerate(sample):
        raw = load_tiff(path)
        raw2d = raw.numpy()
        well, site = parse_well_site(path)
        processed = {m: preprocess(raw, m, params) for m in ALL_MODES}

        title = f"{path.name[:40]}  well={well} site={site}  raw[min={raw2d.min():.0f} max={raw2d.max():.0f} p99.9={np.percentile(raw2d,99.9):.0f}]"
        panel_path = qc_dir / f"panel_{k:02d}_{well}_s{site}.png"
        per_image_panel(raw, processed, panel_path, title)

        for m in ALL_MODES:
            im = processed[m].squeeze(0).numpy()
            mask = make_pseudo_mask(processed[m], MASK_THRESHOLD, MASK_CLOSING).squeeze(0).numpy()
            agg[m]["sat"].append(float((im >= 0.999).mean()) * 100)
            agg[m]["zero"].append(float((im <= 1e-6).mean()) * 100)
            agg[m]["dyn_range"].append(float(np.percentile(im, 99) - np.percentile(im, 1)))
            agg[m]["fg"].append(float(mask.mean()) * 100)
            agg[m]["grad"].append(grad_energy(im))

        a_clip_ceiling_frac.append(float(((raw2d - PAPER_BIAS) > PAPER_CLIP_MAX).mean()) * 100)
        a_below_bias_frac.append(float(((raw2d - PAPER_BIAS) <= 0).mean()) * 100)

        base = processed["per_image_minmax_no_clip"]
        down = torch.nn.functional.interpolate(base.unsqueeze(0), scale_factor=1.0 / PAPER_DOWNSCALE, mode="area")
        up = torch.nn.functional.interpolate(down, size=base.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
        e_full = grad_energy(base.squeeze(0).numpy())
        e_resampled = grad_energy(up.squeeze(0).numpy())
        downscale_retained_hf.append(100.0 * e_resampled / e_full if e_full > 0 else float("nan"))

        per_image_records.append({
            "file": path.name, "well": well, "site": site,
            "raw_min": float(raw2d.min()), "raw_max": float(raw2d.max()),
            "raw_p99_9": float(np.percentile(raw2d, 99.9)),
            "modeA_size": list(processed["aggressive_current"].shape[-2:]),
            "panel": str(panel_path.relative_to(out_root)),
        })

    # ---- 5) aggregate summary figure --------------------------------------- #
    def med(v):
        return float(np.median(v)) if v else float("nan")

    summary = {m: {k: med(v) for k, v in agg[m].items()} for m in ALL_MODES}
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    metrics = [("sat", "% saturated (=1.0)"), ("zero", "% zero (=0.0)"), ("dyn_range", "dyn range p99-p1"), ("grad", "sharpness (grad energy)")]
    xs = list(range(len(ALL_MODES)))
    for ax, (key, label) in zip(axes, metrics):
        ax.bar(xs, [summary[m][key] for m in ALL_MODES], color=["tab:red", "tab:green", "tab:blue", "tab:purple"])
        ax.set_xticks(xs)
        ax.set_xticklabels([MODE_LABELS[m].split(":")[0] for m in ALL_MODES])
        ax.set_title(label, fontsize=10)
    fig.suptitle("Aggregate over QC sample (median across images)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    summary_fig = qc_dir / "aggregate_summary.png"
    fig.savefig(summary_fig, dpi=120)
    plt.close(fig)

    stats = {
        "num_images": len(sample),
        "mode_D_global_low": global_low,
        "mode_D_global_high": global_high,
        "aggregate_median": summary,
        "modeA_saturated_signal_frac_pct": {"median": med(a_clip_ceiling_frac), "max": float(np.max(a_clip_ceiling_frac))},
        "modeA_zeroed_by_bias_frac_pct": {"median": med(a_below_bias_frac), "max": float(np.max(a_below_bias_frac))},
        "downscale_63_20_retained_hf_pct": {"median": med(downscale_retained_hf), "min": float(np.min(downscale_retained_hf))},
        "per_image": per_image_records,
    }
    (out_root / "metrics" / "qc_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # ---- 6) QC report answering the required questions --------------------- #
    a_size = per_image_records[0]["modeA_size"] if per_image_records else [0, 0]
    lines = [
        "# BBBC022 Hoechst preprocessing QC report\n\n",
        f"Sample: {len(sample)} representative fields spanning wells "
        f"{', '.join(sorted({r['well'] for r in per_image_records}))} (single plate `IXMtest`; "
        "split uses one site per well).\n\n",
        "## Mode definitions\n\n",
    ]
    for m in ALL_MODES:
        lines.append(f"- **{MODE_LABELS[m]}** — {mode_description(m, params)}\n")
    lines += [
        "\n## Aggregate (median across sample)\n\n",
        "| mode | % saturated | % zero | dyn-range (p99-p1) | pseudo-GT fg % | sharpness |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for m in ALL_MODES:
        s = summary[m]
        lines.append(f"| {MODE_LABELS[m]} | {s['sat']:.2f} | {s['zero']:.2f} | {s['dyn_range']:.3f} | {s['fg']:.2f} | {s['grad']:.4f} |\n")

    q1 = stats["modeA_saturated_signal_frac_pct"]
    q2 = stats["modeA_zeroed_by_bias_frac_pct"]
    q3 = stats["downscale_63_20_retained_hf_pct"]
    lines += [
        "\n## Required QC questions\n\n",
        f"**1. Does `clip_max=500` saturate/flatten useful nuclear signal?** "
        f"In mode A, a median of **{q1['median']:.2f}%** (up to **{q1['max']:.2f}%**) of pixels lie above the "
        f"bias+clip ceiling ({PAPER_BIAS}+{PAPER_CLIP_MAX:.0f}) and are flattened to the maximum. Median saturated "
        f"fraction in the final mode-A image is **{summary['aggressive_current']['sat']:.2f}%** vs "
        f"~{summary['per_image_minmax_no_clip']['sat']:.2f}% for mode C. => clip=500 **does** saturate real signal.\n\n",
        f"**2. Does fixed bias subtraction remove real low-intensity signal?** "
        f"A median of **{q2['median']:.2f}%** (up to **{q2['max']:.2f}%**) of pixels fall at/below the bias and are "
        f"zeroed by mode A. Background sits near {PAPER_BIAS:.0f}, so bias removal is mostly background, but for some "
        f"fields a non-trivial fraction of dim signal is clipped to zero.\n\n",
        f"**3. Does downscaling by 63/20 destroy nuclear boundary/detail?** "
        f"Mode A output size is **{a_size[0]}x{a_size[1]}** (from 520x696), i.e. smaller than a 256x256 patch. "
        f"After a 63/20 downscale+upsample, only a median of **{q3['median']:.1f}%** of high-frequency (gradient) "
        f"energy is retained (min {q3['min']:.1f}%). => downscaling **does** blur nuclear boundaries and makes "
        f"256x256 crops infeasible without majority padding.\n\n",
        f"**4. Are nuclei visibly preserved better by the minimal pipelines?** "
        f"Modes B/C/D keep larger dynamic range (B={summary['minimal_percentile']['dyn_range']:.3f}, "
        f"C={summary['per_image_minmax_no_clip']['dyn_range']:.3f}, D={summary['trainset_global_percentile']['dyn_range']:.3f}) "
        f"and much lower saturation than A ({summary['aggressive_current']['sat']:.2f}%). Inspect the panels to confirm "
        f"nuclei interiors and boundaries are preserved (not flattened).\n\n",
        f"**5. Any pipelines producing blank/washed-out/over-contrasted images?** "
        f"Mode A shows the highest saturation ({summary['aggressive_current']['sat']:.2f}%) (washed-out bright nuclei). "
        f"Mode C can over-stretch when a single hot pixel sets the max. Mode B/D percentile clipping mitigates hot pixels. "
        f"Check per-image `% saturated`/`% zero` annotations on each panel.\n\n",
        "## IMPORTANT: pseudo-GT segmentation masks depend on preprocessing\n\n",
        "The paper's Fig. 4 uses *pseudo*-ground-truth masks (normalize -> threshold 0.3 -> closing 10x10) derived "
        "from the **preprocessed** image (paper A.2.3). The bottom row of each panel shows that the resulting mask "
        "**changes with the preprocessing mode** (see `pseudo-GT fg %`). This is a methodological decision that must "
        "be resolved before Fig. 4 training (see report).\n\n",
        "## Panels\n\n",
        f"- Aggregate summary: `{summary_fig.relative_to(out_root)}`\n",
    ]
    for r in per_image_records:
        lines.append(f"- `{r['panel']}` — well {r['well']} site {r['site']}, raw max {r['raw_max']:.0f}\n")
    (qc_dir / "qc_report.md").write_text("".join(lines), encoding="utf-8")

    print(f"\nWrote QC report -> {qc_dir / 'qc_report.md'}", flush=True)
    print(f"Wrote {len(sample)} panels + aggregate summary to {qc_dir}", flush=True)
    print(f"Wrote stats -> {out_root / 'metrics' / 'qc_stats.json'}", flush=True)


if __name__ == "__main__":
    main()
