#!/usr/bin/env python3
"""BBBC022 Hoechst substitute U2OS-style experiment runner."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.staged_hardening_train import train_staged_hardening
from training.train_reconstruction import train
from utils.device import resolve_device
from utils.experiment_config import expand_experiment_matrix, load_experiment_config

OUT = ROOT / "experiments/ablations"
ABLATION_ROOT = OUT / "bbbc022_ablation_x16"
CONTENT_ROOT = OUT / "bbbc022_content_aware"

ABLATION_VARIANTS = [
    "bbbc022_x16_A_random_transpose",
    "bbbc022_x16_B_learnable_transpose",
    "bbbc022_x16_C_learnable_locality_freq",
    "bbbc022_x16_D_learnable_locality_spatial",
]

CONTENT_COMPRESSIONS = [
    ("x16", 8, 4),
    ("x64", 16, 4),
    ("x256", 32, 4),
    ("x1024", 64, 4),
]

CONTENT_PATTERNS = ["uniform_all_ones", "random_fixed", "hadamard_fixed", "learnable_frequency"]


def _is_learnable(config: dict) -> bool:
    return config["pattern_generator"]["mode"] in {"learnable_frequency", "learnable_spatial"}


def _run_one(config: dict, output_dir: str) -> dict:
    if config["training"].get("use_staged_hardening") and _is_learnable(config):
        return train_staged_hardening(config, output_dir)
    return train(config, output_dir)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0, "n": len(values)}


def _apply_official_preprocessing(config: dict) -> None:
    report_path = OUT / "preprocessing_report.json"
    if report_path.exists():
        mode = json.loads(report_path.read_text())["chosen_official_mode"]
        config["dataset"]["preprocessing_mode"] = mode


def phase_setup() -> None:
    subprocess.check_call([sys.executable, str(ROOT / "scripts/bbbc022_data_discovery.py")])
    subprocess.check_call([sys.executable, str(ROOT / "scripts/bbbc022_preprocessing_sanity.py")])


def phase_tiny_overfit(device: str) -> None:
    out_root = OUT / "validation/tiny_overfit"
    out_root.mkdir(parents=True, exist_ok=True)
    base = load_experiment_config(ROOT / "configs/base_bbbc022_substitute.yaml")
    _apply_official_preprocessing(base)
    base["experiment"]["device"] = device
    base["training"]["batch_size"] = 4
    base["training"]["max_steps"] = 800
    base["dataset"]["max_train_samples"] = 8
    base["dataset"]["max_val_samples"] = 8
    base["dataset"]["max_test_samples"] = 8
    results = []
    for run_id, mode, upsampling, learn, staged in [
        ("random_transpose", "random_fixed", "transpose_conv", False, False),
        ("learnable_locality", "learnable_frequency", "locality_aware", True, True),
    ]:
        config = copy.deepcopy(base)
        config["pattern_generator"]["mode"] = mode
        config["inverse_model"]["upsampling"]["mode"] = upsampling
        config["training"]["learn_patterns"] = learn
        config["training"]["use_staged_hardening"] = staged
        config["experiment"]["run_id"] = f"tiny_overfit_{run_id}"
        out_dir = str(out_root / f"tiny_overfit_{run_id}")
        print(f"\n=== tiny overfit {run_id} ===", flush=True)
        summary = _run_one(config, out_dir)
        results.append({**summary, "run_id": run_id})
    gate = {
        "runs": results,
        "pass": all(r.get("test_mse", 1.0) < 0.5 for r in results),
        "note": "BBBC022 substitute tiny overfit gate",
    }
    (out_root / "gate_summary.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    if not gate["pass"]:
        (OUT / "TINY_OVERFIT_FAILURE.md").write_text(
            "# Tiny overfit failed\n\nDo not run full ablation until resolved.\n",
            encoding="utf-8",
        )
        raise RuntimeError("Tiny overfit gate failed")
    print("Tiny overfit gate: PASS", flush=True)


def phase_ablation(device: str, seeds: list[int]) -> None:
    out_root = ABLATION_ROOT
    out_root.mkdir(parents=True, exist_ok=True)
    matrix = ROOT / "configs/matrices/bbbc022_x16_ablation_matrix.yaml"
    all_results = []
    by_variant: dict[str, list] = {v: [] for v in ABLATION_VARIANTS}
    for seed in seeds:
        for base_config in expand_experiment_matrix(matrix):
            config = copy.deepcopy(base_config)
            _apply_official_preprocessing(config)
            run_id = config["experiment"]["run_id"]
            config["experiment"]["seed"] = seed
            config["dataset"]["seed"] = seed
            config["pattern_generator"]["seed"] = seed
            config["experiment"]["device"] = device
            out_dir = str(out_root / f"{run_id}_seed{seed}")
            if (Path(out_dir) / "metrics/run_summary.json").exists():
                summary = json.loads((Path(out_dir) / "metrics/run_summary.json").read_text())
            else:
                print(f"\n=== {run_id} seed={seed} ===", flush=True)
                summary = _run_one(config, out_dir)
            summary["seed"] = seed
            summary["variant"] = run_id
            all_results.append(summary)
            by_variant[run_id].append(summary)
    aggregate = {v: {k: _stats([float(r[k]) for r in rows if r.get(k) is not None]) for k in ["test_mse", "test_ssim", "pattern_delta", "H_t_binary_fraction"]} for v, rows in by_variant.items()}
    output = {"seeds": seeds, "results": all_results, "aggregate": aggregate, "label": "BBBC022 substitute not paper U2OS"}
    (out_root / "aggregate_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    with (out_root / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "seed", "test_mse", "test_ssim", "pattern_delta", "H_t_binary_fraction"], extrasaction="ignore")
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    md = ["# BBBC022 Substitute ×16 Ablation A–D\n\n", "**Not comparable to paper Table 3 numerically.**\n\n"]
    for v in ABLATION_VARIANTS:
        a = aggregate.get(v, {}).get("test_mse", {})
        md.append(f"- {v}: MSE={a.get('mean', float('nan')):.4f}\n")
    (out_root / "ABCD_report.md").write_text("".join(md), encoding="utf-8")


def phase_content_aware(device: str, seeds: list[int]) -> None:
    out_root = CONTENT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)
    all_results = []
    for seed in seeds:
        for comp_name, d, t in CONTENT_COMPRESSIONS:
            if 256 // d < 2:
                continue
            for pattern in CONTENT_PATTERNS:
                config = load_experiment_config(ROOT / "configs/base_bbbc022_substitute.yaml")
                _apply_official_preprocessing(config)
                config["forward_model"]["downscale_factor"] = d
                config["pattern_generator"]["mode"] = pattern
                config["pattern_generator"]["num_patterns"] = t
                config["inverse_model"]["upsampling"]["downscale_factor"] = d
                config["inverse_model"]["upsampling"]["num_patterns"] = t
                config["inverse_model"]["upsampling"]["mode"] = "locality_aware"
                config["inverse_model"]["reconstruction"]["in_channels"] = t
                config["experiment"]["seed"] = seed
                config["dataset"]["seed"] = seed
                config["pattern_generator"]["seed"] = seed
                config["experiment"]["device"] = device
                learnable = pattern == "learnable_frequency"
                config["training"]["learn_patterns"] = learnable
                config["training"]["use_staged_hardening"] = learnable
                run_id = f"bbbc022_{comp_name}_{pattern}"
                config["experiment"]["run_id"] = run_id
                out_dir = str(out_root / f"{run_id}_seed{seed}")
                if (Path(out_dir) / "metrics/run_summary.json").exists():
                    summary = json.loads((Path(out_dir) / "metrics/run_summary.json").read_text())
                else:
                    print(f"\n=== content {run_id} seed={seed} ===", flush=True)
                    summary = _run_one(config, out_dir)
                summary.update({"variant": run_id, "seed": seed, "compression": comp_name, "pattern": pattern})
                all_results.append(summary)
    (out_root / "aggregate_summary.json").write_text(json.dumps({"results": all_results}, indent=2), encoding="utf-8")
    with (out_root / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "seed", "compression", "pattern", "test_mse", "test_ssim"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    (out_root / "figure3_style_report.md").write_text(
        "# Fig. 3-style content-aware report (BBBC022 substitute)\n\nQualitative substitute-data transfer only. Do not compare numeric values to paper U2OS Fig. 3.\n",
        encoding="utf-8",
    )


def phase_swinir_status() -> None:
    (OUT / "swinir_status.md").write_text(
        "# SwinIR status\n\n**NOT IMPLEMENTED** in this repository.\n\n"
        "Would need: SwinIR model weights, integration after locality upsampling or as recon replacement, "
        "training scripts for Fig. 3 / Fig. 8 / Fig. 9 style experiments on BBBC022 substitute data.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["setup", "tiny_overfit", "ablation", "content_aware", "swinir", "all"], required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()
    print(f"Device: {resolve_device(args.device)}", flush=True)
    if args.phase in ("setup", "all"):
        phase_setup()
    if args.phase in ("tiny_overfit", "all"):
        phase_tiny_overfit(args.device)
    if args.phase in ("ablation", "all"):
        phase_ablation(args.device, args.seeds)
    if args.phase in ("content_aware", "all"):
        phase_content_aware(args.device, args.seeds)
    if args.phase in ("swinir", "all"):
        phase_swinir_status()


if __name__ == "__main__":
    main()
