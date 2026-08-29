#!/usr/bin/env python3
"""Figure 10 / Table 3 ablation (A/B/C/D) on the SAME BBBC022 substitute data the
repo's Figure 3 reproduction uses.

This is a thin driver that REUSES the validated, audited uniform A/B/C/D protocol
from ``run_am3_table3`` (``run_one`` + ``apply_variant`` + ``default_phases``):

  * scaled Algorithm-1 phases (inverse warmup -> joint soft -> m hardening 2/4/8),
  * L1 loss, illumination lr 1.0, inverse lr 0.001, grad clip 1.0,
  * machine-checkable variant wiring audit (variant_audit.check_variant),
  * GLOBAL best-val-MSE checkpoint selection (identical rule for every variant).

The ONLY thing that changes vs the historical AM-3 run is the dataset: instead of
the small 168-image ``paper_strict`` BBBC022 set, this uses the exact substitute
data Figure 3 consumes (``bbbc022_preproc_ablation`` + ``minimal_percentile`` +
the well-disjoint ``split_fig03_large`` split, 1980/40/60 images). See
``configs/figure10_ablation/ablation.yaml``.

NOT a numeric reproduction of the paper's U2OS Table 3 (U2OS data unavailable);
only the A/B/C/D ordering is meaningful. Run CUDA outside the sandbox.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_spec = importlib.util.spec_from_file_location(
    "table03_run", ROOT / "scripts/table03_ablation/run.py"
)
_am3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_am3)
VARIANT_LABEL = _am3.VARIANT_LABEL
apply_variant = _am3.apply_variant
default_phases = _am3.default_phases
run_one = _am3.run_one

from utils.experiment_config import load_experiment_config  # noqa: E402

EXP = ROOT / "experiments/figure10_ablation"
CFG = ROOT / "configs/figure10_ablation/ablation.yaml"

# Paper Table 3 (U2OS) for side-by-side ordering comparison only.
PAPER_TABLE3 = {
    "A": {"ssim": 0.7872, "mse": 0.0042},
    "B": {"ssim": 0.7950, "mse": 0.0038},
    "C": {"ssim": 0.8426, "mse": 0.0029},
    "D": {"ssim": 0.7857, "mse": 0.0041},
}

DEFAULT_AGG_LABEL = (
    "Figure 10 / Table 3 ablation on BBBC022 Hoechst SUBSTITUTE (Fig-3 data) "
    "- NOT paper U2OS reproduction; only A/B/C/D ORDERING is meaningful."
)
DEFAULT_AGG_DATA = {
    "dataset": "bbbc022_preproc_ablation (minimal_percentile)",
    "split": "split_fig03_large (well-disjoint 1980/40/60)",
    "compression": "x16 (d=8, T=4)",
}


def _stats(values: list[float]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def write_aggregate(
    out_root: Path,
    rows: list[dict],
    seeds: list[int],
    *,
    label: str = DEFAULT_AGG_LABEL,
    data: dict | None = None,
) -> None:
    by_variant: dict[str, list[dict]] = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)
    agg = {}
    for letter in ["A", "B", "C", "D"]:
        rs = by_variant.get(letter, [])
        agg[letter] = {
            "label": VARIANT_LABEL[letter],
            "test_ssim": _stats([r["test_ssim"] for r in rs]),
            "test_mse": _stats([r["test_mse"] for r in rs]),
            "best_val_mse": _stats([r["best_val_mse"] for r in rs]),
            "overfit_gap": _stats([r["overfit_gap"] for r in rs]),
        }
    present = {L: agg[L] for L in agg if agg[L]["test_mse"]["mean"] is not None}
    best_letter = None
    if present:
        best_letter = min(present, key=lambda L: present[L]["test_mse"]["mean"])
    out = {
        "label": label,
        "data": dict(DEFAULT_AGG_DATA if data is None else data),
        "seeds": seeds,
        "aggregate": agg,
        "our_best_variant_by_test_mse": best_letter,
        "paper_table3_u2os": PAPER_TABLE3,
        "paper_best_variant": "C",
        "ordering_matches_paper_best": best_letter == "C",
    }
    (out_root / "aggregate_summary.json").write_text(json.dumps(out, indent=2))
    print("\n=== Fig10 aggregate ===", flush=True)
    for letter in ["A", "B", "C", "D"]:
        a = agg[letter]
        s = a["test_ssim"]["mean"]
        m = a["test_mse"]["mean"]
        s = f"{s:.4f}" if s is not None else "  n/a "
        m = f"{m:.5f}" if m is not None else "  n/a  "
        print(f"  {letter} ({a['label']:<42}): SSIM {s}  MSE {m}", flush=True)
    print(f"  our best (min test MSE): {best_letter}  |  paper best: C  |  "
          f"ordering match: {best_letter == 'C'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig 10 / Table 3 A/B/C/D on Fig-3 substitute data")
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

    print("Fig10 ablation: variants="
          f"{args.variants} device={args.device} "
          f"scale={args.scale} cuda={torch.cuda.is_available()}", flush=True)
    phase_s = ", ".join(f"{p['name']}:{p['steps']}" for p in phases)
    print(f"  phases (per variant): {phase_s}", flush=True)

    rows: list[dict] = []
    for letter in args.variants:
        run_dir = out_root / f"{letter}_seed{args.seed}"
        summary_path = run_dir / "metrics" / "run_summary.json"
        if (summary_path.exists() and not args.force) or args.aggregate_only:
            if summary_path.exists():
                rows.append(json.loads(summary_path.read_text()))
            continue
        base = load_experiment_config(CFG)
        base["experiment"]["device"] = args.device
        cfg = apply_variant(base, letter)
        cfg["experiment"]["seed"] = args.seed
        cfg["dataset"]["seed"] = args.seed
        cfg["pattern_generator"]["seed"] = args.seed
        cfg["experiment"]["run_id"] = f"fig10_{letter}_seed{args.seed}"
        print(f"\n=== Fig10 variant {letter}: {VARIANT_LABEL[letter]} "
              f"(seed {args.seed}, {args.device}) ===", flush=True)
        rows.append(run_one(cfg, run_dir, letter=letter, phases=phases,
                            seed=args.seed, log_every=args.log_every))

    # Collect every available variant summary for the aggregate (not just this call's).
    all_rows: list[dict] = []
    for letter in ["A", "B", "C", "D"]:
        sp = out_root / f"{letter}_seed{args.seed}" / "metrics" / "run_summary.json"
        if sp.exists():
            all_rows.append(json.loads(sp.read_text()))
    write_aggregate(out_root, all_rows, [args.seed])


if __name__ == "__main__":
    main()
