#!/usr/bin/env python3
"""Train the Fig.3 content-aware 4x4 matrix (paper-quality BBBC022 substitute).

Same architecture / optimizer / LR / batch / staged-hardening as the original
content-aware pipeline and the ``minimal_percentile`` preprocessing (float32 ->
robust [0.1%, 99.9%] percentile clip -> min-max). Two changes recover the paper's
Fig.3 D/E behaviour (learnable best at *every* compression, gap widening with
compression) on the BBBC022 substitute:

1. **Larger well-disjoint split** (`split_fig03_large.json`, 1980 train / 40 val /
   60 test, multi-site train, strictly well-disjoint) instead of 168/21/21.
2. **Per-epoch random crops/flips** (`epoch_varying_train_crops=True`) instead of
   one fixed patch per image.

Together these remove the overfitting wall (train MSE ~0.0005 vs val ~0.0018 on
the old 168-image split) that previously capped every illumination method at the
same val ceiling, hiding the learnable advantage — exactly the regime where
PatchMNIST (3000 imgs) already shows learnable beating fixed ~3x.

Matrix: 4 compressions {x16,x64,x256,x1024} x 4 illuminations
{uniform_all_ones, random_fixed, hadamard_fixed, learnable_frequency}.

Output root: experiments/figure03_content_aware/base/
(results.csv is written in the same schema the Fig.3 renderer expects).
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
CONTENT_ROOT = ROOT / "experiments/figure03_content_aware/base"

COMPRESSIONS = [("x16", 8, 4), ("x64", 16, 4), ("x256", 32, 4), ("x1024", 64, 4)]
PATTERNS = ["uniform_all_ones", "random_fixed", "hadamard_fixed", "learnable_frequency"]

# Budget. Larger than the legacy paper_strict/minimal matrices because the new
# split (1980 well-disjoint train images) + per-epoch random crops removed the
# overfitting wall, so longer training now improves *validation* (it no longer
# just memorises 168 fixed patches).
FULL_STAGED = {"inverse_warmup_steps": 1500, "joint_soft_steps": 5000,
               "harden_m_values": [2, 4, 8], "harden_steps_per_m": 1200}
FULL_FIXED_STEPS = 4000
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
    # Draw fresh crops/flips every epoch (true augmentation). Without this the
    # train split returns one identical patch per image every epoch, which (with
    # the small legacy split) caused the overfitting that hid the learnable gain.
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
    config["dataset"]["seed"] = seed
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
