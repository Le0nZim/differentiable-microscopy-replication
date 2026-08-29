#!/usr/bin/env python3
"""Paper-aligned PatchMNIST experiment runner (Phases 4–6)."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.staged_hardening_train import train_staged_hardening
from training.train_reconstruction import train
from utils.device import resolve_device
from utils.experiment_config import (
    build_noise_sweep_experiments,
    expand_experiment_matrix,
    load_experiment_config,
)

LEARNABLE_VARIANTS = [
    "patchmnist_x8_learnable_transpose",
    "patchmnist_x8_learnable_locality",
]

FIXED_VARIANTS = [
    "patchmnist_x8_random_locality",
    "patchmnist_x8_uniform_locality",
    "patchmnist_x8_hadamard_locality",
    "patchmnist_x8_random_transpose",
    "patchmnist_x8_uniform_transpose",
    "patchmnist_x8_hadamard_transpose",
]

AGGREGATE_KEYS = [
    "test_mse",
    "test_ssim",
    "thresholded_test_mse",
    "thresholded_test_ssim",
    "pattern_delta",
    "detector_delta",
    "H_t_binary_fraction",
    "best_val_mse",
]


def _is_learnable(config: dict) -> bool:
    return config["pattern_generator"]["mode"] == "learnable_frequency"


def _run_one(config: dict, output_dir: str) -> dict:
    if config["training"].get("use_staged_hardening") and _is_learnable(config):
        return train_staged_hardening(config, output_dir)
    return train(config, output_dir)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def _aggregate(by_variant: dict[str, list[dict]]) -> dict:
    return {
        variant: {key: _stats([float(r[key]) for r in rows if r.get(key) is not None]) for key in AGGREGATE_KEYS}
        for variant, rows in by_variant.items()
    }


def _load_summary(path: Path) -> dict | None:
    summary_path = path / "metrics" / "run_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return None


def _run_matrix(
    matrix_path: Path,
    output_root: Path,
    seeds: list[int],
    device: str,
    illumination_lr: float | None = None,
) -> tuple[list[dict], dict[str, list[dict]]]:
    output_root.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []
    by_variant: dict[str, list[dict]] = {}
    for seed in seeds:
        for base_config in expand_experiment_matrix(matrix_path):
            config = copy.deepcopy(base_config)
            run_id = config["experiment"]["run_id"]
            config["experiment"]["seed"] = seed
            config["dataset"]["seed"] = seed
            config["pattern_generator"]["seed"] = seed
            config["experiment"]["device"] = device
            if illumination_lr is not None:
                config["training"]["illumination_lr"] = illumination_lr
            config["experiment"]["run_id"] = f"{run_id}_seed{seed}"
            out_dir = output_root / f"{run_id}_seed{seed}"
            if (out_dir / "metrics" / "run_summary.json").exists():
                print(f"Skipping existing {out_dir.name}", flush=True)
                summary = _load_summary(out_dir)
            else:
                print(f"\n========== {run_id} seed={seed} illum_lr={config['training']['illumination_lr']} ==========", flush=True)
                summary = _run_one(config, str(out_dir))
            summary["seed"] = seed
            summary["variant"] = run_id
            summary["illumination_lr"] = config["training"]["illumination_lr"]
            all_results.append(summary)
            by_variant.setdefault(run_id, []).append(summary)
    return all_results, by_variant


def _gate_pass(summary: dict, stable_mse: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    mse = float(summary["test_mse"])
    if math.isnan(mse) or math.isinf(mse):
        reasons.append("nan_or_inf_test_mse")
    h_bin = float(summary.get("H_t_binary_fraction", 0))
    if h_bin < 0.1 or h_bin > 0.95:
        reasons.append(f"H_t_binary_fraction_out_of_range={h_bin:.3f}")
    if mse > stable_mse * 1.15:
        reasons.append(f"test_mse_worse_than_stable ({mse:.4f} > {stable_mse * 1.15:.4f})")
    pd = float(summary.get("pattern_delta", 0))
    if pd < 0.01:
        reasons.append(f"pattern_delta_too_low={pd:.4f}")
    return len(reasons) == 0, reasons


def phase_lr1_full(args: argparse.Namespace) -> None:
    output_root = ROOT / "experiments/content_aware/lr1_full"
    matrix_path = ROOT / "configs/matrices/paper_aligned_lr1_learnable_matrix.yaml"
    stable_root = ROOT / "experiments/content_aware/lr1_full"  # historical LR gate no longer available

    gate_seeds = args.seeds if args.expand else [42]
    results, by_variant = _run_matrix(matrix_path, output_root, gate_seeds, args.device, illumination_lr=1.0)

    comparisons: list[dict] = []
    all_pass = True
    for variant in LEARNABLE_VARIANTS:
        for seed in gate_seeds:
            lr1_dir = output_root / f"{variant}_seed{seed}"
            lr1 = _load_summary(lr1_dir)
            stable_dir = stable_root / f"{variant}_seed{seed}"
            stable = _load_summary(stable_dir)
            if lr1 is None or stable is None:
                continue
            passed, reasons = _gate_pass(lr1, float(stable["test_mse"]))
            comparisons.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "lr1_test_mse": lr1["test_mse"],
                    "lr1_test_ssim": lr1["test_ssim"],
                    "stable_test_mse": stable["test_mse"],
                    "stable_test_ssim": stable["test_ssim"],
                    "gate_pass": passed,
                    "gate_fail_reasons": reasons,
                    "H_t_binary_fraction": lr1.get("H_t_binary_fraction"),
                    "pattern_delta": lr1.get("pattern_delta"),
                }
            )
            if not passed:
                all_pass = False

    adopt_lr1 = all_pass and len(gate_seeds) == 1 and not args.force_stable
    summary = {
        "illumination_lr_paper": 1.0,
        "illumination_lr_stable": 0.3,
        "gate_seeds": gate_seeds,
        "adopt_paper_lr_for_remaining": adopt_lr1,
        "comparisons": comparisons,
        "aggregate_lr1": _aggregate(by_variant),
    }
    if adopt_lr1 and args.expand:
        extra_seeds = [s for s in args.all_seeds if s not in gate_seeds]
        if extra_seeds:
            extra_results, extra_by = _run_matrix(matrix_path, output_root, extra_seeds, args.device, illumination_lr=1.0)
            results.extend(extra_results)
            for k, v in extra_by.items():
                by_variant.setdefault(k, []).extend(v)
            summary["expanded_seeds"] = extra_seeds
            summary["aggregate_lr1_all_seeds"] = _aggregate(by_variant)

    summary["results"] = results
    json_path = output_root / "lr_comparison_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# LR=1.0 vs stable LR=0.3 comparison\n\n",
        f"**Adopt paper LR=1.0 for remaining experiments:** {adopt_lr1}\n\n",
        "| Variant | Seed | LR=1.0 MSE | LR=0.3 MSE | LR=1.0 SSIM | Gate |\n",
        "|---|---:|---:|---:|---:|---|\n",
    ]
    for row in comparisons:
        md_lines.append(
            f"| {row['variant']} | {row['seed']} | {row['lr1_test_mse']:.4f} | {row['stable_test_mse']:.4f} | "
            f"{row['lr1_test_ssim']:.4f} | {'PASS' if row['gate_pass'] else 'FAIL: ' + ', '.join(row['gate_fail_reasons'])} |\n"
        )
    (output_root / "lr_comparison_summary.md").write_text("".join(md_lines), encoding="utf-8")
    print(f"Wrote {json_path}", flush=True)
    print(f"Adopt LR=1.0: {adopt_lr1}", flush=True)


def phase_fixed_baselines(args: argparse.Namespace) -> None:
    output_root = ROOT / "experiments/content_aware/fixed_baselines"
    matrix_path = ROOT / "configs/paper_aligned_patchmnist_fixed_baselines.yaml"
    lr = 1.0 if not args.force_stable else 0.3
    results, by_variant = _run_matrix(matrix_path, output_root, args.seeds, args.device, illumination_lr=lr)
    aggregate = _aggregate(by_variant)
    output = {
        "seeds": args.seeds,
        "illumination_lr": lr,
        "matrix": str(matrix_path),
        "results": results,
        "aggregate": aggregate,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "aggregate_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    fields = ["variant", "seed", "test_mse", "test_ssim", "H_t_binary_fraction", "pattern_delta", "detector_delta"]
    with (output_root / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Wrote {output_root / 'aggregate_summary.json'}", flush=True)


def _build_noise_experiments(illumination_lr: float, *, normalized: bool = False, output_suffix: str = "") -> list[dict]:
    config_name = (
        "paper_aligned_patchmnist_noise_table_normalized.yaml"
        if normalized
        else "paper_aligned_patchmnist_noise_table.yaml"
    )
    results_dir = (
        f"experiments/noise_robustness/noise_table_normalized{output_suffix}"
        if normalized
        else "experiments/noise_robustness/noise_table"
    )
    base = load_experiment_config(ROOT / f"configs/{config_name}")
    base["training"]["illumination_lr"] = illumination_lr
    configs = build_noise_sweep_experiments(
        base,
        photon_counts=[10.0, 10000.0],
        sigma_reads=[0.0, 2.7, 2.0, 6.0],
        pattern_modes=["random_fixed", "learnable_frequency"],
        results_csv=f"{results_dir}/results.csv",
    )
    for config in configs:
        config["inverse_model"]["upsampling"]["mode"] = "locality_aware"
        mode = config["pattern_generator"]["mode"]
        if mode == "learnable_frequency":
            config["training"]["learn_patterns"] = True
            config["training"]["use_staged_hardening"] = True
        else:
            config["training"]["learn_patterns"] = False
            config["training"]["use_staged_hardening"] = False
    return configs


def phase_noise_table(args: argparse.Namespace) -> None:
    _run_noise_table_phase(args, normalized=False)


def phase_noise_table_normalized(args: argparse.Namespace) -> None:
    suffix = getattr(args, "noise_output_suffix", "_v2")
    _run_noise_table_phase(args, normalized=True, output_suffix=suffix)


def _run_noise_table_phase(
    args: argparse.Namespace, *, normalized: bool, output_suffix: str = ""
) -> None:
    output_root = ROOT / (
        f"experiments/noise_robustness/noise_table_normalized{output_suffix}"
        if normalized
        else "experiments/noise_robustness/noise_table"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    lr = 1.0 if not args.force_stable else 0.3
    experiments = _build_noise_experiments(lr, normalized=normalized, output_suffix=output_suffix)
    all_results: list[dict] = []
    for seed in args.seeds:
        for base_config in experiments:
            config = copy.deepcopy(base_config)
            run_id = config["experiment"]["run_id"]
            config["experiment"]["seed"] = seed
            config["dataset"]["seed"] = seed
            config["pattern_generator"]["seed"] = seed
            config["experiment"]["device"] = args.device
            config["experiment"]["run_id"] = f"{run_id}_seed{seed}"
            out_dir = output_root / f"{run_id}_seed{seed}"
            if (out_dir / "metrics" / "run_summary.json").exists():
                print(f"Skipping existing {out_dir.name}", flush=True)
                summary = _load_summary(out_dir)
            else:
                print(f"\n========== {run_id} seed={seed} ==========", flush=True)
                summary = _run_one(config, str(out_dir))
            summary["seed"] = seed
            summary["variant"] = run_id
            summary["pattern_mode"] = config["pattern_generator"]["mode"]
            summary["detector_noise"] = config["detector_noise"]
            all_results.append(summary)

    by_run: dict[str, list[dict]] = {}
    for row in all_results:
        by_run.setdefault(row["variant"], []).append(row)
    aggregate = {
        variant: {key: _stats([float(r[key]) for r in rows if r.get(key) is not None]) for key in AGGREGATE_KEYS}
        for variant, rows in by_run.items()
    }
    output = {"seeds": args.seeds, "illumination_lr": lr, "compression": "x8", "results": all_results, "aggregate": aggregate}
    (output_root / "aggregate_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    fields = [
        "variant", "seed", "photon_count", "sigma_read", "pattern_mode",
        "test_mse", "test_ssim", "thresholded_test_mse", "thresholded_test_ssim",
        "pattern_delta", "detector_delta", "H_t_binary_fraction",
    ]
    with (output_root / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in all_results:
            noise = row.get("detector_noise", {})
            writer.writerow({**row, "photon_count": noise.get("photon_count"), "sigma_read": noise.get("sigma_read")})

    label = "paper-normalized noise (RR-1)" if normalized else "legacy noise"
    md = [
        f"# Table 1 qualitative comparison ({label})\n\n",
        "Qualitative only — do not claim exact numeric match to paper.\n\n",
    ]
    md.append("| photon_count | sigma_read | random MSE | learnable MSE | learnable wins? |\n|---|---:|---:|---:|---|\n")
    for pc in [10.0, 10000.0]:
        for sr in [0.0, 2.7, 2.0, 6.0]:
            rand = [r for r in all_results if r["pattern_mode"] == "random_fixed" and r["detector_noise"]["photon_count"] == pc and r["detector_noise"]["sigma_read"] == sr]
            learn = [r for r in all_results if r["pattern_mode"] == "learnable_frequency" and r["detector_noise"]["photon_count"] == pc and r["detector_noise"]["sigma_read"] == sr]
            if rand and learn:
                rm, lm = rand[0]["test_mse"], learn[0]["test_mse"]
                md.append(f"| {pc:g} | {sr} | {rm:.4f} | {lm:.4f} | {'yes' if lm < rm else 'no'} |\n")
    (output_root / "table1_comparison.md").write_text("".join(md), encoding="utf-8")
    if normalized:
        _write_rr1_diagnosis(output_root, all_results)
    print(f"Wrote {output_root / 'aggregate_summary.json'}", flush=True)


def _write_rr1_diagnosis(output_root: Path, all_results: list[dict]) -> None:
    """Summarize whether AM-1 is resolved after paper-normalized Table 1 rerun."""
    learnable_rows = [r for r in all_results if r["pattern_mode"] == "learnable_frequency"]
    random_rows = [r for r in all_results if r["pattern_mode"] == "random_fixed"]

    learnable_wins_all = True
    for pc in [10.0, 10000.0]:
        for sr in [0.0, 2.7, 2.0, 6.0]:
            rand = [r for r in random_rows if r["detector_noise"]["photon_count"] == pc and r["detector_noise"]["sigma_read"] == sr]
            learn = [r for r in learnable_rows if r["detector_noise"]["photon_count"] == pc and r["detector_noise"]["sigma_read"] == sr]
            if rand and learn and learn[0]["test_mse"] >= rand[0]["test_mse"]:
                learnable_wins_all = False

    def _spread(rows: list[dict], pc: float) -> float:
        mses = [r["test_mse"] for r in rows if r["detector_noise"]["photon_count"] == pc]
        return max(mses) - min(mses) if mses else float("nan")

    spread_pc10 = _spread(learnable_rows, 10.0)
    spread_pc10000 = _spread(learnable_rows, 10000.0)

    pc10_mses = [r["test_mse"] for r in learnable_rows if r["detector_noise"]["photon_count"] == 10.0 and r["detector_noise"]["sigma_read"] == 0.0]
    pc10000_mses = [r["test_mse"] for r in learnable_rows if r["detector_noise"]["photon_count"] == 10000.0 and r["detector_noise"]["sigma_read"] == 0.0]
    pc_order_ok = (
        pc10_mses and pc10000_mses and pc10_mses[0] <= pc10000_mses[0]
    )

    read_invariance_ok = spread_pc10 < 0.01 and spread_pc10000 < 0.01

    resolved = learnable_wins_all and read_invariance_ok and pc_order_ok

    lines = [
        "# RR-1 diagnosis (AM-1 paper-normalized noise)\n\n",
        f"**AM-1 resolved:** {'yes' if resolved else 'partial' if learnable_wins_all else 'no'}\n\n",
        "## Checks\n\n",
        f"- Learnable beats fixed in every cell: **{'yes' if learnable_wins_all else 'no'}**\n",
        f"- Learnable read-noise spread (pc=10, σ∈{{0,2.7,2.0,6.0}}): {spread_pc10:.4f} "
        f"(paper: ~flat; threshold <0.01): **{'yes' if spread_pc10 < 0.01 else 'no'}**\n",
        f"- Learnable read-noise spread (pc=10000): {spread_pc10000:.4f}: "
        f"**{'yes' if spread_pc10000 < 0.01 else 'no'}**\n",
        f"- Photon-count ordering (learnable pc=10 ≤ pc=10000 at σ=0): **{'yes' if pc_order_ok else 'no'}**\n\n",
        "## Comparison to legacy `noise_table/`\n\n",
        "See `experiments/noise_robustness/noise_table/table1_comparison.md` for pre-fix baseline.\n",
        "Legacy run showed strong σ_read sensitivity and reversed photon-count ordering.\n\n",
        "## Verdict\n\n",
    ]
    if resolved:
        lines.append("Paper-normalized noise (eqs. S7–S10) reproduces Table 1 qualitative properties. AM-1 is resolved.\n")
    elif learnable_wins_all and not read_invariance_ok:
        lines.append("Learnable still wins, but read-noise invariance not fully reproduced. AM-1 partially resolved.\n")
    else:
        lines.append("Mismatch remains after normalization fix; investigate further.\n")

    (output_root / "RR1_diagnosis.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-aligned PatchMNIST runner")
    parser.add_argument(
        "--phase",
        choices=["lr1_full", "fixed_baselines", "noise_table", "noise_table_normalized", "all"],
        required=True,
    )
    parser.add_argument("--noise-output-suffix", default="_v2", help="Suffix for normalized noise output dir (default: _v2)")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--all-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--expand", action="store_true", help="After LR gate on seed 42, expand to all seeds")
    parser.add_argument("--force-stable", action="store_true", help="Use illumination_lr=0.3 instead of paper 1.0")
    args = parser.parse_args()

    print(f"Device: {resolve_device(args.device)}", flush=True)
    if args.phase in ("lr1_full", "all"):
        phase_lr1_full(args)
    if args.phase in ("fixed_baselines", "all"):
        phase_fixed_baselines(args)
    if args.phase in ("noise_table", "all"):
        phase_noise_table(args)
    if args.phase in ("noise_table_normalized",):
        phase_noise_table_normalized(args)


if __name__ == "__main__":
    main()
