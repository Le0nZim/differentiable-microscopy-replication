#!/usr/bin/env python3
"""Fig. 5 upsampling analysis on the paper grid with disjoint val/test PatchMNIST pools.

Grid follows the published Fig. 5: image sizes {128, 256, 512} and train counts
{600, 3000, 6000}; size 64 is kept as an extra. The grid is shardable so the two
GPUs can split the (expensive) 512 runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.train_reconstruction import train
from utils.device import resolve_device
from utils.experiment_config import load_experiment_config

DEFAULT_OUT = ROOT / "experiments/figure05_upsampling"

PAPER_IMAGE_SIZES = [128, 256, 512]
EXTRA_IMAGE_SIZES = [64]
TRAIN_COUNTS = [600, 3000, 6000]
UPSAMPLING_MODES = ["transpose_conv", "locality_aware"]

FIELDNAMES = ["run_id", "image_size", "num_train", "upsampling", "test_mse", "test_ssim", "seed"]


def _run_id(image_size: int, num_train: int, upsampling: str) -> str:
    short = "locality" if upsampling == "locality_aware" else "transpose"
    return f"fig5_is{image_size}_n{num_train}_{short}"


def _build_config(image_size: int, num_train: int, upsampling: str, seed: int, device: str) -> dict:
    config = load_experiment_config(ROOT / "configs/figure05_upsampling/base.yaml")
    config["dataset"]["image_size"] = image_size
    config["dataset"]["num_train"] = num_train
    config["dataset"]["seed"] = seed
    # Val and test draw from disjoint halves of the MNIST test digit pool so the
    # best_val checkpoint is not selected on digits that reappear in the test set.
    config["dataset"]["disjoint_val_test"] = True
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
    config["experiment"]["run_id"] = _run_id(image_size, num_train, upsampling)
    config["experiment"]["fig5_note"] = (
        "Fig. 5 paper grid (sizes 128/256/512, n_train 600/3000/6000; size 64 extra) "
        "with disjoint val/test MNIST digit pools; step budget unspecified in the paper"
    )
    return config


def collect_rows(out_root: Path) -> list[dict]:
    """Read every completed run under out_root into result rows."""
    rows: list[dict] = []
    for summary_path in sorted(out_root.glob("fig5_is*/metrics/run_summary.json")):
        run_dir = summary_path.parents[1]
        stem = run_dir.name.replace("_seed", "|").split("|")[0]
        parts = stem.split("_")  # fig5, is<size>, n<count>, <mode>
        image_size = int(parts[1][2:])
        num_train = int(parts[2][1:])
        upsampling = "locality_aware" if parts[3] == "locality" else "transpose_conv"
        summary = json.loads(summary_path.read_text())
        rows.append(
            {
                "run_id": stem,
                "image_size": image_size,
                "num_train": num_train,
                "upsampling": upsampling,
                "test_mse": summary.get("test_mse"),
                "test_ssim": summary.get("test_ssim"),
                "seed": int(run_dir.name.rsplit("_seed", 1)[1]),
            }
        )
    rows.sort(key=lambda r: (r["image_size"], r["num_train"], r["upsampling"]))
    return rows


def write_results_csv(out_root: Path, rows: list[dict]) -> None:
    with (out_root / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def run_grid(out_root: Path, sizes: list[int], counts: list[int], device: str, seed: int) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for image_size in sizes:
        for num_train in counts:
            for upsampling in UPSAMPLING_MODES:
                config = _build_config(image_size, num_train, upsampling, seed, device)
                run_id = config["experiment"]["run_id"]
                out_dir = out_root / f"{run_id}_seed{seed}"
                if (out_dir / "metrics/run_summary.json").exists():
                    print(f"\n=== skip existing {run_id} seed={seed} ===", flush=True)
                    continue
                print(f"\n=== {run_id} seed={seed} device={device} ===", flush=True)
                train(config, str(out_dir))
    print(f"SHARD_DONE sizes={sizes} counts={counts}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sizes", default="", help="comma-separated image sizes for this shard")
    parser.add_argument("--counts", default="", help="comma-separated train counts for this shard")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.output_root)
    if args.collect_only:
        rows = collect_rows(out_root)
        write_results_csv(out_root, rows)
        print(f"Collected {len(rows)} runs -> {out_root / 'results.csv'}")
        return

    sizes = [int(s) for s in args.sizes.split(",") if s] or (EXTRA_IMAGE_SIZES + PAPER_IMAGE_SIZES)
    counts = [int(c) for c in args.counts.split(",") if c] or TRAIN_COUNTS
    resolve_device(args.device)
    run_grid(out_root, sizes, counts, args.device, args.seed)


if __name__ == "__main__":
    main()
