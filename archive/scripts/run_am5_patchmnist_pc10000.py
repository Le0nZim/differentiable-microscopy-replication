#!/usr/bin/env python3
"""AM-5 runner: PatchMNIST content-aware matrix with paper-aligned pc=10000 noise.

Reruns the *exact* frozen ×8 PatchMNIST content-aware fixed-vs-learnable matrix
(learnable transpose/locality + fixed random/uniform/hadamard × transpose/locality)
changing ONLY the detector-noise setting to the AM-1 (RR-1 v3) paper-aligned
normalized differentiable Poisson model at photon_count=10000.

Reuses the same trainers (`train_staged_hardening` for learnable,
`train` for fixed) and the same matrix-expansion path as
`scripts/run_paper_aligned_patchmnist.py`, so seeds, splits, architecture,
optimizer, schedule, upsampling, max_steps, and metrics are all identical to the
frozen runs. Writes to a NEW output dir; never touches frozen folders.

Examples
--------
# Full matrix, 3 seeds, single GPU:
python scripts/run_am5_patchmnist_pc10000.py --device cuda:1

# Parallelise across 2 GPUs (run both in separate terminals):
python scripts/run_am5_patchmnist_pc10000.py --device cuda:0 --shard 0 --num-shards 2
python scripts/run_am5_patchmnist_pc10000.py --device cuda:1 --shard 1 --num-shards 2

# Aggregate only (after all shards finish):
python scripts/run_am5_patchmnist_pc10000.py --aggregate-only
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
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
from utils.experiment_config import expand_experiment_matrix

OUTPUT_ROOT = ROOT / "experiments/content_aware/am5_patchmnist_pc10000_resolution"
LEARNABLE_MATRIX = ROOT / "configs/matrices/am5_patchmnist_pc10000_learnable_matrix.yaml"
FIXED_MATRIX = ROOT / "configs/matrices/am5_patchmnist_pc10000_fixed_matrix.yaml"

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


def assert_paper_aligned_noise(config: dict) -> None:
    """Hard runtime guard: the active config must be the AM-1 paper-aligned pc=10000 setting."""
    noise = config["detector_noise"]
    problems = []
    if noise.get("mode") == "noise_free":
        problems.append("detector_noise.mode is noise_free")
    if noise.get("mode") not in {"differentiable_poisson", "differentiable_poisson_plus_read"}:
        problems.append(f"unexpected detector_noise.mode={noise.get('mode')!r}")
    if not noise.get("apply_noise", False):
        problems.append("apply_noise is not True")
    if float(noise.get("photon_count", 0)) != 10000.0:
        problems.append(f"photon_count != 10000 (got {noise.get('photon_count')!r})")
    if noise.get("noise_normalization") != "paper_v3":
        problems.append(
            f"noise_normalization must be paper_v3 (AM-1 path), got {noise.get('noise_normalization')!r}"
        )
    if problems:
        raise ValueError(
            "AM-5 noise guard failed for run "
            f"{config.get('experiment', {}).get('run_id')!r}: " + "; ".join(problems)
        )


def _run_one(config: dict, output_dir: str) -> dict:
    assert_paper_aligned_noise(config)
    if config["training"].get("use_staged_hardening") and _is_learnable(config):
        return train_staged_hardening(config, output_dir)
    return train(config, output_dir)


def _jobs(seeds: list[int]) -> list[tuple[Path, dict, int]]:
    """Flat list of (matrix_path, base_config, seed) for every matrix entry × seed."""
    jobs: list[tuple[Path, dict, int]] = []
    for matrix_path in (LEARNABLE_MATRIX, FIXED_MATRIX):
        for base_config in expand_experiment_matrix(matrix_path):
            for seed in seeds:
                jobs.append((matrix_path, base_config, seed))
    return jobs


def _load_summary(path: Path) -> dict | None:
    summary_path = path / "metrics" / "run_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return None


def run_matrix(args: argparse.Namespace) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = _jobs(args.seeds)
    for idx, (matrix_path, base_config, seed) in enumerate(jobs):
        if args.num_shards > 1 and (idx % args.num_shards) != args.shard:
            continue
        config = copy.deepcopy(base_config)
        run_id = config["experiment"]["run_id"]
        config["experiment"]["seed"] = seed
        config["dataset"]["seed"] = seed
        config["pattern_generator"]["seed"] = seed
        config["experiment"]["device"] = args.device
        config["experiment"]["run_id"] = f"{run_id}_seed{seed}"
        out_dir = OUTPUT_ROOT / f"{run_id}_seed{seed}"
        if (out_dir / "metrics" / "run_summary.json").exists():
            print(f"Skipping existing {out_dir.name}", flush=True)
            continue
        print(f"\n========== AM-5 {run_id} seed={seed} (matrix={matrix_path.name}) ==========", flush=True)
        _run_one(config, str(out_dir))


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def aggregate(args: argparse.Namespace) -> None:
    """Collect every run_summary.json into metrics_by_run.csv + aggregate_summary.json."""
    rows: list[dict] = []
    by_variant: dict[str, list[dict]] = {}

    for matrix_path in (LEARNABLE_MATRIX, FIXED_MATRIX):
        for base_config in expand_experiment_matrix(matrix_path):
            run_id = base_config["experiment"]["run_id"]
            noise = base_config["detector_noise"]
            pattern_mode = base_config["pattern_generator"]["mode"]
            upsampling = base_config["inverse_model"]["upsampling"]["mode"]
            learnable = _is_learnable(base_config)
            for seed in args.seeds:
                out_dir = OUTPUT_ROOT / f"{run_id}_seed{seed}"
                summary = _load_summary(out_dir)
                if summary is None:
                    print(f"WARN missing summary: {out_dir.name}", flush=True)
                    continue
                row = {
                    "run_id": run_id,
                    "variant": run_id,
                    "seed": seed,
                    "pattern_mode": pattern_mode,
                    "upsampling": upsampling,
                    "learnable": learnable,
                    "photon_count": noise.get("photon_count"),
                    "noise_mode": noise.get("mode"),
                    "noise_normalization": noise.get("noise_normalization"),
                    "apply_noise": noise.get("apply_noise"),
                    "sigma_read": noise.get("sigma_read"),
                    "test_mse": summary.get("test_mse"),
                    "test_ssim": summary.get("test_ssim"),
                    "thresholded_test_mse": summary.get("thresholded_test_mse"),
                    "thresholded_test_ssim": summary.get("thresholded_test_ssim"),
                    "best_val_mse": summary.get("best_val_mse"),
                    "pattern_delta": summary.get("pattern_delta"),
                    "detector_delta": summary.get("detector_delta"),
                    "H_t_binary_fraction": summary.get("H_t_binary_fraction"),
                }
                rows.append(row)
                by_variant.setdefault(run_id, []).append(summary | {"seed": seed})

    fields = [
        "run_id", "seed", "pattern_mode", "upsampling", "learnable",
        "photon_count", "noise_mode", "noise_normalization", "apply_noise", "sigma_read",
        "test_mse", "test_ssim", "thresholded_test_mse", "thresholded_test_ssim",
        "best_val_mse", "pattern_delta", "detector_delta", "H_t_binary_fraction",
    ]
    csv_path = OUTPUT_ROOT / "metrics_by_run.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["run_id"], r["seed"])):
            writer.writerow(row)

    aggregate_summary = {
        "output_dir": str(OUTPUT_ROOT),
        "seeds": args.seeds,
        "compression": "x8",
        "noise": {
            "mode": "differentiable_poisson",
            "noise_normalization": "paper_v3",
            "photon_count": 10000.0,
            "sigma_read": 0.0,
            "apply_noise": True,
        },
        "num_runs": len(rows),
        "aggregate_by_variant": {
            variant: {
                key: _stats([float(r[key]) for r in runs if r.get(key) is not None])
                for key in AGGREGATE_KEYS
            }
            for variant, runs in by_variant.items()
        },
    }
    (OUTPUT_ROOT / "aggregate_summary.json").write_text(
        json.dumps(aggregate_summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {OUTPUT_ROOT / 'aggregate_summary.json'}", flush=True)
    print(f"Aggregated {len(rows)} runs across {len(by_variant)} variants.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="AM-5 PatchMNIST pc=10000 content-aware runner")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--shard", type=int, default=0, help="0-based shard index")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of parallel shards")
    parser.add_argument("--aggregate-only", action="store_true", help="Skip training, only aggregate")
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip aggregation (use when running parallel shards; aggregate once at the end)",
    )
    args = parser.parse_args()

    if not args.aggregate_only:
        print(f"Device: {resolve_device(args.device)}", flush=True)
        run_matrix(args)
    if not args.no_aggregate:
        aggregate(args)


if __name__ == "__main__":
    main()
