#!/usr/bin/env python3
"""Rerun the historical soft-pattern Figure 3 matrix with equal update budgets.

Uses the well-disjoint BBBC022 substitute and fresh crops on each loader pass.
No method ranking is an implementation acceptance criterion. The old results
used unequal budgets and cached augmentation; they remain historical artifacts.
For binary masks, explicit dose accounting, and held-out exact-noise evaluation,
use scripts/audit/run_controlled.py instead (see docs/SCIENTIFIC_AUDIT.md).
New runs go to experiments/audited_v1/figure03_content_aware/base/.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.staged_hardening_train import train_staged_hardening  # noqa: E402
from training.train_reconstruction import train  # noqa: E402
from utils.device import resolve_device  # noqa: E402
from utils.experiment_config import load_experiment_config, sync_derived_config_fields  # noqa: E402

SPLIT_PATH = ROOT / "configs/_shared/splits/split_fig03_large.json"
CONTENT_ROOT = ROOT / "experiments/audited_v1/figure03_content_aware/base"

COMPRESSIONS = [("x16", 8, 4), ("x64", 16, 4), ("x256", 32, 4), ("x1024", 64, 4)]
PATTERNS = ["uniform_all_ones", "random_fixed", "hadamard_fixed", "learnable_frequency"]

# Equal inverse-update budgets. This alone does not equalize dose or deployment.
FULL_STAGED = {"inverse_warmup_steps": 1500, "joint_soft_steps": 5000,
               "harden_m_values": [2, 4, 8], "harden_steps_per_m": 1200}
FULL_FIXED_STEPS = 10100
BATCH = 32


def _is_learnable(config: dict) -> bool:
    return config["pattern_generator"]["mode"] in {"learnable_frequency", "learnable_spatial"}


def _run_one(config: dict, output_dir: str) -> dict:
    if config["training"].get("use_staged_hardening") and _is_learnable(config):
        return train_staged_hardening(config, output_dir)
    return train(config, output_dir)


def _apply_minimal_dataset(config: dict) -> None:
    ds = config["dataset"]
    ds["name"] = "bbbc022_preproc_ablation"
    ds["split_path"] = str(SPLIT_PATH)
    ds["repo_root"] = str(ROOT)
    ds["preproc_mode"] = "minimal_percentile"
    ds["q_low"] = 0.001
    ds["q_high"] = 0.999
    ds["patch_size"] = 256
    ds["image_size"] = 256
    ds["return_mask"] = False
    ds["train_random_crops"] = True
    ds["random_flips"] = True
    # Fresh crops/flips require a new DataLoader iterator, not itertools.cycle.
    ds["epoch_varying_train_crops"] = True


def build_config(comp_name: str, d: int, t: int, pattern: str, seed: int, device: str) -> dict:
    config = load_experiment_config(ROOT / "configs/_shared/base_bbbc022_substitute.yaml")
    config["forward_model"]["downscale_factor"] = d
    config["pattern_generator"]["mode"] = pattern
    config["pattern_generator"]["num_patterns"] = t
    config["inverse_model"]["upsampling"]["downscale_factor"] = d
    config["inverse_model"]["upsampling"]["num_patterns"] = t
    config["inverse_model"]["upsampling"]["mode"] = "locality_aware"
    config["inverse_model"]["reconstruction"]["in_channels"] = t
    config["experiment"]["seed"] = seed
    config["dataset"]["seed"] = 42
    config["training"]["loader_seed"] = seed + 1000
    config["pattern_generator"]["seed"] = seed
    config["experiment"]["device"] = device
    config["experiment"]["run_id"] = f"bbbc022_{comp_name}_{pattern}"
    config["experiment"]["compression"] = float(comp_name[1:])

    learnable = pattern == "learnable_frequency"
    config["training"]["learn_patterns"] = learnable
    config["training"]["use_staged_hardening"] = learnable
    config["training"]["batch_size"] = BATCH
    if learnable:
        config["training"].pop("max_steps", None)
        config["training"]["staged_hardening"] = copy.deepcopy(FULL_STAGED)
    else:
        config["training"]["max_steps"] = FULL_FIXED_STEPS
        config["training"].pop("staged_hardening", None)

    _apply_minimal_dataset(config)
    return sync_derived_config_fields(config)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--resume", action="store_true", help="skip runs that already have run_summary.json")
    args = ap.parse_args()

    print(f"Device: {resolve_device(args.device)}", flush=True)
    CONTENT_ROOT.mkdir(parents=True, exist_ok=True)

    all_results = []
    t_start = time.time()
    for seed in args.seeds:
        for comp_name, d, t in COMPRESSIONS:
            for pattern in PATTERNS:
                run_id = f"bbbc022_{comp_name}_{pattern}"
                out_dir = CONTENT_ROOT / f"{run_id}_seed{seed}"
                summary_path = out_dir / "metrics/run_summary.json"
                if args.resume and summary_path.exists():
                    summary = json.loads(summary_path.read_text())
                    print(f"[skip] {run_id} seed={seed} (exists)", flush=True)
                else:
                    config = build_config(comp_name, d, t, pattern, seed, args.device)
                    t0 = time.time()
                    print(f"\n=== minimal {run_id} seed={seed} (downscale={d}, T={t}) ===", flush=True)
                    summary = _run_one(config, str(out_dir))
                    print(f"[done] {run_id} in {time.time() - t0:.0f}s  "
                          f"MSE={summary.get('test_mse')}  SSIM={summary.get('test_ssim')}", flush=True)
                summary.update({"variant": run_id, "seed": seed, "compression": comp_name, "pattern": pattern})
                all_results.append(summary)

    with (CONTENT_ROOT / "results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["variant", "seed", "compression", "pattern", "test_mse", "test_ssim"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    (CONTENT_ROOT / "aggregate_summary.json").write_text(
        json.dumps({"preprocessing": "minimal_percentile", "results": all_results}, indent=2))
    print(f"\nAll runs complete in {time.time() - t_start:.0f}s -> {CONTENT_ROOT}/results.csv", flush=True)


if __name__ == "__main__":
    main()
