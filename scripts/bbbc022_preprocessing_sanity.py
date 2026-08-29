#!/usr/bin/env python3
"""Phase 2: Compare preprocessing modes A/B/C and select official substitute mode."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets.bbbc022_hoechst import (
    discover_image_paths,
    load_tiff,
    preprocess_image,
    select_hoechst_paths,
)

OUT = ROOT / "experiments/ablations"
EXAMPLES = OUT / "preprocessing_examples"
MODES = ["paper_strict", "bbbc022_calibrated", "raw_normalized"]


def _mode_stats(images: list[torch.Tensor]) -> dict:
    stacked = torch.stack(images)
    return {
        "n": len(images),
        "mean": float(stacked.mean()),
        "std": float(stacked.std()),
        "fraction_zero": float((stacked <= 1e-6).float().mean()),
        "fraction_saturated": float((stacked >= 0.999).float().mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    paths = select_hoechst_paths(discover_image_paths(ROOT / "data/substitute_data", "**/*.tif"))
    random.seed(42)
    sample_paths = random.sample(paths, min(100, len(paths)))

    mode_results: dict[str, dict] = {}
    for mode in MODES:
        processed = []
        for path in sample_paths:
            raw = load_tiff(path)
            img = preprocess_image(
                raw,
                mode,  # type: ignore[arg-type]
                bias=134.28,
                clip_max=500.0,
                background_percentile=1.0,
                clip_percentile=99.9,
            )
            processed.append(img.squeeze(0))
        mode_results[mode] = _mode_stats(processed)

    # Prefer paper_strict unless collapsed
    chosen = "paper_strict"
    reason = "Mode A (paper_strict) retains reasonable dynamic range on 100-image sample."
    if mode_results["paper_strict"]["fraction_zero"] > 0.5 or mode_results["paper_strict"]["std"] < 0.05:
        chosen = "bbbc022_calibrated"
        reason = (
            "Mode A collapsed dynamic range; using Mode B (bbbc022_calibrated) as official substitute preprocessing."
        )

    fig, axes = plt.subplots(3, 5, figsize=(12, 7))
    for row, mode in enumerate(MODES):
        for col, path in enumerate(sample_paths[:5]):
            raw = load_tiff(path)
            img = preprocess_image(raw, mode, bias=134.28, clip_max=500.0, background_percentile=1.0, clip_percentile=99.9)
            axes[row, col].imshow(img.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(mode, fontsize=8)
    plt.tight_layout()
    plt.savefig(EXAMPLES / "modes_abc_grid.png", dpi=120)
    plt.close()

    report = {
        "modes": mode_results,
        "chosen_official_mode": chosen,
        "selection_reason": reason,
        "deviations": {
            "no_z_stack_mip": "2D BBBC022 fields; MIP is no-op",
            "no_63_20_downscale": "Native resolution preserved; 256 crops extracted",
            "hoechst_not_dapi": "Hoechst 33342 substitute stain",
            "clip_max_500_mode_a": "Paper clip; may be harsh on BBBC022 uint16 scale",
        },
        "split": "168/21/21 via well-aware assignment",
    }
    (OUT / "preprocessing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# BBBC022 Preprocessing Report\n\n",
        f"**Official substitute mode:** `{chosen}`\n\n",
        f"{reason}\n\n",
        "## Mode statistics (100-image sample)\n\n",
        "| Mode | mean | std | frac_zero | frac_saturated |\n|---|---:|---:|---:|---:|\n",
    ]
    for mode, st in mode_results.items():
        md.append(f"| {mode} | {st['mean']:.4f} | {st['std']:.4f} | {st['fraction_zero']:.4f} | {st['fraction_saturated']:.4f} |\n")
    md.append("\n## Logged deviations from paper U2OS\n\n")
    for k, v in report["deviations"].items():
        md.append(f"- **{k}**: {v}\n")
    (OUT / "preprocessing_report.md").write_text("".join(md), encoding="utf-8")
    print(f"Chosen mode: {chosen}")
    print(f"Wrote {OUT / 'preprocessing_report.md'}")


if __name__ == "__main__":
    main()
