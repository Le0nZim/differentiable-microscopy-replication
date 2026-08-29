#!/usr/bin/env python3
"""PatchMNIST Fig. 5-style upsampling analysis: locality-aware vs transpose."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.train_reconstruction import train
from utils.device import resolve_device
from utils.experiment_config import load_experiment_config

OUT = ROOT / "experiments/upsampling_ablation/patchmnist_upsampling_analysis"

# Paper Fig. 5 (read off the published plot): image sizes {128, 256, 512}
# and train counts {600, 3000, 6000}. We keep 64 as an extra size.
IMAGE_SIZES = [64, 128, 256, 512]
TRAIN_COUNTS = [600, 3000, 6000]
UPSAMPLING_MODES = ["transpose_conv", "locality_aware"]


def _run_id(image_size: int, num_train: int, upsampling: str) -> str:
    short = "locality" if upsampling == "locality_aware" else "transpose"
    return f"fig5_is{image_size}_n{num_train}_{short}"


def _build_config(image_size: int, num_train: int, upsampling: str, seed: int, device: str) -> dict:
    config = load_experiment_config(ROOT / "configs/paper_aligned_patchmnist_x8.yaml")
    config["dataset"]["image_size"] = image_size
    config["dataset"]["num_train"] = num_train
    config["dataset"]["seed"] = seed
    config["experiment"]["seed"] = seed
    config["pattern_generator"]["mode"] = "random_fixed"
    config["pattern_generator"]["seed"] = seed
    config["pattern_generator"]["num_patterns"] = 8
    config["inverse_model"]["upsampling"]["mode"] = upsampling
    config["inverse_model"]["upsampling"]["downscale_factor"] = 8
    config["inverse_model"]["upsampling"]["num_patterns"] = 8
    config["inverse_model"]["reconstruction"]["in_channels"] = 8
    config["training"]["learn_patterns"] = False
    config["training"]["use_staged_hardening"] = False
    config["training"]["fixed_sigmoid_m"] = 10.0
    config["training"]["max_steps"] = 4000
    config["training"]["log_every"] = 200
    config["experiment"]["device"] = device
    run_id = _run_id(image_size, num_train, upsampling)
    config["experiment"]["run_id"] = run_id
    config["experiment"]["fig5_note"] = (
        "Fig. 5 grid matched to the published plot (sizes 128/256/512, n_train 600/3000/6000) "
        "with extra image size 64; step budget still unspecified in the paper"
    )
    return config


def _write_figures(rows: list[dict]) -> None:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for image_size in IMAGE_SIZES:
        subset = [r for r in rows if r["image_size"] == image_size]
        trains = sorted({r["num_train"] for r in subset})
        for metric, ylabel, fname in [
            ("test_mse", "Test MSE", f"mse_vs_train_is{image_size}.png"),
            ("test_ssim", "Test SSIM", f"ssim_vs_train_is{image_size}.png"),
        ]:
            plt.figure(figsize=(7, 4))
            for upsampling, label in [("locality_aware", "locality-aware"), ("transpose_conv", "transpose")]:
                pts = sorted(
                    (r["num_train"], r[metric]) for r in subset if r["upsampling"] == upsampling
                )
                if not pts:
                    continue
                xs, ys = zip(*pts)
                plt.plot(xs, ys, marker="o", label=label)
            plt.xscale("log")
            if metric == "test_mse":
                plt.yscale("log")
            plt.xlabel("# train images")
            plt.ylabel(ylabel)
            plt.title(f"PatchMNIST ×8, image_size={image_size}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_dir / fname, dpi=150)
            plt.close()


def _write_report(rows: list[dict], aggregate: dict) -> None:
    lines = [
        "# Fig. 5-style upsampling analysis (PatchMNIST)\n\n",
        "**Paper source:** Sec 5.4, Fig. 5 (`paper_sources/paper.md`).\n\n",
        "## Grid (matched to published Fig. 5, plus extra size 64)\n\n",
        "- Compression: ×8 (d=8, T=8)\n",
        "- Pattern: `random_fixed` (noise off)\n",
        "- Image sizes: 64 (extra), 128, 256, 512 (paper)\n",
        "- Train images: 600, 3000, 6000 (val/test fixed at 375/375)\n",
        "- Upsampling: locality-aware vs transpose convolution\n",
        "- Seed: 42\n",
        "- max_steps: 4000 (paper step budget for Fig. 5 not specified — logged deviation)\n\n",
        "## Qualitative trends\n\n",
    ]
    for image_size in IMAGE_SIZES:
        at_3000 = [
            r for r in rows if r["image_size"] == image_size and r["num_train"] == 3000
        ]
        if len(at_3000) == 2:
            loc = next(r for r in at_3000 if r["upsampling"] == "locality_aware")
            tr = next(r for r in at_3000 if r["upsampling"] == "transpose_conv")
            winner = "locality" if loc["test_mse"] < tr["test_mse"] else "transpose"
            lines.append(
                f"- image_size={image_size}, n_train=3000: {winner} wins "
                f"(MSE locality={loc['test_mse']:.4f}, transpose={tr['test_mse']:.4f})\n"
            )
    lines.append(
        "\nPaper claim: locality should outperform transpose at larger image size with sufficient training data.\n"
        "Compare trends qualitatively; exact numeric match not expected because the step budget is still unspecified.\n"
    )
    (OUT / "fig5_style_report.md").write_text("".join(lines), encoding="utf-8")


FIELDNAMES = ["run_id", "image_size", "num_train", "upsampling", "test_mse", "test_ssim", "seed"]


def _write_results_csv(rows: list[dict]) -> None:
    with (OUT / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(device: str, seed: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for image_size in IMAGE_SIZES:
        for num_train in TRAIN_COUNTS:
            for upsampling in UPSAMPLING_MODES:
                config = _build_config(image_size, num_train, upsampling, seed, device)
                run_id = config["experiment"]["run_id"]
                out_dir = OUT / f"{run_id}_seed{seed}"
                if (out_dir / "metrics/run_summary.json").exists():
                    print(f"\n=== skip existing {run_id} seed={seed} ===", flush=True)
                    summary = json.loads((out_dir / "metrics/run_summary.json").read_text())
                else:
                    print(f"\n=== {run_id} seed={seed} ===", flush=True)
                    summary = train(config, str(out_dir))
                row = {
                    "run_id": run_id,
                    "image_size": image_size,
                    "num_train": num_train,
                    "upsampling": upsampling,
                    "test_mse": summary.get("test_mse"),
                    "test_ssim": summary.get("test_ssim"),
                    "seed": seed,
                }
                rows.append(row)
                _write_results_csv(rows)

    aggregate = {}
    for image_size in IMAGE_SIZES:
        for num_train in TRAIN_COUNTS:
            key = f"is{image_size}_n{num_train}"
            subset = [r for r in rows if r["image_size"] == image_size and r["num_train"] == num_train]
            loc_mse = next((r["test_mse"] for r in subset if r["upsampling"] == "locality_aware"), None)
            tr_mse = next((r["test_mse"] for r in subset if r["upsampling"] == "transpose_conv"), None)
            aggregate[key] = {
                "locality_mse": loc_mse,
                "transpose_mse": tr_mse,
                "locality_beats_transpose": loc_mse is not None and tr_mse is not None and loc_mse < tr_mse,
            }

    payload = {
        "label": "PatchMNIST Fig. 5-style upsampling analysis (paper grid + extra size 64)",
        "paper_section": "5.4",
        "grid": {
            "image_sizes": IMAGE_SIZES,
            "train_counts": TRAIN_COUNTS,
            "upsampling_modes": UPSAMPLING_MODES,
            "seed": seed,
            "max_steps": 4000,
            "approximate": False,
            "paper_matched": True,
            "extra_image_sizes": [64],
        },
        "results": rows,
        "aggregate": aggregate,
    }
    (OUT / "aggregate_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_figures(rows)
    _write_report(rows, aggregate)
    print(f"Wrote {OUT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    resolve_device(args.device)
    run_analysis(args.device, args.seed)


if __name__ == "__main__":
    main()
