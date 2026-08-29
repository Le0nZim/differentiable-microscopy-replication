#!/usr/bin/env python3
"""Figure 10 / Table 3 ablation (A/B/C/D) on PatchMNIST.

Companion to ``scripts/figure10_ablation/train.py``. The protocol, the variant
knobs and the checkpoint-selection rule are byte-for-byte the same code path
(``run_am3_table3.run_one`` + ``apply_variant`` + ``default_phases``, reached
through the Fig-10 driver). The ONLY difference is the dataset:

  * Fig-10  : BBBC022 Hoechst substitute, ``split_fig03_large`` (1980/40/60),
              ``minimal_percentile`` preprocessing.
  * here    : PatchMNIST exactly as the Figure-6 / Table-1 noise-robustness
              experiments generate and load it (256px canvases of 32px digits on
              a 20x20 grid, 3000/375/375, disjoint val/test digit pools) -- the
              dataset block of ``configs/table01_noise_robustness/noise_table.yaml``.

Why: on the BBBC022 substitute, variant D (no frequency-domain optimization) came
out BEST, contradicting the paper. BBBC022 nuclei tile the FOV near-uniformly, so
the spatial-invariance benefit the paper attributes to frequency-domain
optimization is weak there. PatchMNIST has sparse, high-contrast, translation-
varying structure, so this run tests whether the D-wins result is a
substitute-data artifact.

Everything is written under ``experiments/figure10_ablation_patchmnist/``; the
BBBC022 Fig-10 outputs are never touched. Run CUDA outside the sandbox.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_spec = importlib.util.spec_from_file_location(
    "fig10_train", ROOT / "scripts/figure10_ablation/train.py"
)
_fig10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fig10)
VARIANT_LABEL = _fig10.VARIANT_LABEL
apply_variant = _fig10.apply_variant
default_phases = _fig10.default_phases
run_one = _fig10.run_one
write_aggregate = _fig10.write_aggregate

from utils.experiment_config import load_experiment_config  # noqa: E402

EXP = ROOT / "experiments/figure10_ablation_patchmnist"
CFG = ROOT / "configs/figure10_ablation_patchmnist/ablation.yaml"

AGG_LABEL = (
    "Figure 10 / Table 3 ablation on PatchMNIST (the Fig-6 / Table-1 data) "
    "- NOT paper U2OS reproduction; only A/B/C/D ORDERING is meaningful."
)
AGG_DATA = {
    "dataset": "patchmnist (256px canvas, 32px digits, 20x20 grid; no extra preprocessing)",
    "split": "3000 train / 375 val / 375 test, disjoint_val_test digit pools",
    "compression": "x16 (d=8, T=4)",
    "source_of_data_block": "configs/table01_noise_robustness/noise_table.yaml",
    "noise": "noise_free (Fig-10 ablation protocol; only the data was swapped)",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig 10 / Table 3 A/B/C/D on PatchMNIST")
    ap.add_argument("--variants", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="scale factor on the uniform Algorithm-1 phase step counts")
    ap.add_argument("--out", default=str(EXP / "runs"))
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="only (re)build aggregate_summary.json from existing run summaries")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    phases = default_phases(scale=args.scale)

    print("Fig10-PatchMNIST ablation: variants="
          f"{args.variants} device={args.device} "
          f"scale={args.scale} cuda={torch.cuda.is_available()}", flush=True)
    phase_s = ", ".join(f"{p['name']}:{p['steps']}" for p in phases)
    print(f"  phases (per variant): {phase_s}", flush=True)

    for letter in args.variants:
        run_dir = out_root / f"{letter}_seed{args.seed}"
        summary_path = run_dir / "metrics" / "run_summary.json"
        if (summary_path.exists() and not args.force) or args.aggregate_only:
            continue
        base = load_experiment_config(CFG)
        base["experiment"]["device"] = args.device
        cfg = apply_variant(base, letter)
        cfg["experiment"]["seed"] = args.seed
        cfg["dataset"]["seed"] = args.seed
        cfg["pattern_generator"]["seed"] = args.seed
        cfg["experiment"]["run_id"] = f"fig10pm_{letter}_seed{args.seed}"
        print(f"\n=== Fig10-PatchMNIST variant {letter}: {VARIANT_LABEL[letter]} "
              f"(seed {args.seed}, {args.device}) ===", flush=True)
        run_one(cfg, run_dir, letter=letter, phases=phases,
                seed=args.seed, log_every=args.log_every)

    # Collect every available variant summary for the aggregate (not just this call's).
    all_rows: list[dict] = []
    for letter in ["A", "B", "C", "D"]:
        sp = out_root / f"{letter}_seed{args.seed}" / "metrics" / "run_summary.json"
        if sp.exists():
            all_rows.append(json.loads(sp.read_text()))
    write_aggregate(out_root, all_rows, [args.seed], label=AGG_LABEL, data=AGG_DATA)


if __name__ == "__main__":
    main()
