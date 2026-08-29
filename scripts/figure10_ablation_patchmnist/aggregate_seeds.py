#!/usr/bin/env python3
"""Multi-seed aggregate for the PatchMNIST Fig-10 ablation.

``train.py --aggregate-only`` only ever summarizes a single seed (it mirrors the
BBBC022 Fig-10 driver). The C-vs-D margin on PatchMNIST is small enough that the
comparison needs a seed spread, so this collects every ``<L>_seed<N>`` run
directory present and reports per-variant mean/std plus the pairwise C-vs-D
verdict.

    python scripts/figure10_ablation_patchmnist/aggregate_seeds.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments/figure10_ablation_patchmnist/runs"
LETTERS = ["A", "B", "C", "D"]


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "n": 0, "values": []}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
        "values": values,
    }


def collect(runs: Path) -> dict[str, list[dict]]:
    by_variant: dict[str, list[dict]] = {L: [] for L in LETTERS}
    for path in sorted(runs.glob("*_seed*/metrics/run_summary.json")):
        summary = json.loads(path.read_text())
        letter = summary["variant"]
        if letter in by_variant:
            by_variant[letter].append(summary)
    for rows in by_variant.values():
        rows.sort(key=lambda r: r["seed"])
    return by_variant


def main() -> None:
    ap = argparse.ArgumentParser(description="multi-seed aggregate for the PatchMNIST Fig-10 ablation")
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--out", default=None, help="output JSON (default: <runs>/aggregate_multiseed.json)")
    args = ap.parse_args()

    runs = Path(args.runs)
    by_variant = collect(runs)

    agg = {}
    for letter in LETTERS:
        rows = by_variant[letter]
        agg[letter] = {
            "seeds": [r["seed"] for r in rows],
            "test_mse": _stats([r["test_mse"] for r in rows]),
            "test_ssim": _stats([r["test_ssim"] for r in rows]),
        }

    c_mse = agg["C"]["test_mse"]
    d_mse = agg["D"]["test_mse"]
    verdict = None
    if c_mse["mean"] is not None and d_mse["mean"] is not None:
        c_vals, d_vals = c_mse["values"], d_mse["values"]
        verdict = {
            "d_beats_c_on_mean": d_mse["mean"] < c_mse["mean"],
            "d_beats_c_on_every_seed": max(d_vals) < min(c_vals),
            "relative_mse_gap_d_vs_c": (c_mse["mean"] - d_mse["mean"]) / c_mse["mean"],
        }

    present = {L: agg[L]["test_mse"]["mean"] for L in LETTERS if agg[L]["test_mse"]["mean"] is not None}
    best = min(present, key=present.get) if present else None

    out = {
        "label": "Multi-seed aggregate, Figure 10 / Table 3 ablation on PatchMNIST",
        "aggregate": agg,
        "best_variant_by_mean_test_mse": best,
        "paper_best_variant": "C",
        "c_vs_d": verdict,
    }
    out_path = Path(args.out) if args.out else runs / "aggregate_multiseed.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(f"{'V':<3}{'seeds':>12}{'MSE mean':>11}{'MSE std':>10}{'SSIM mean':>11}{'SSIM std':>10}")
    for letter in LETTERS:
        a = agg[letter]
        if a["test_mse"]["mean"] is None:
            continue
        print(f"{letter:<3}{str(a['seeds']):>12}{a['test_mse']['mean']:>11.5f}"
              f"{a['test_mse']['std']:>10.5f}{a['test_ssim']['mean']:>11.4f}"
              f"{a['test_ssim']['std']:>10.4f}")
    print(f"\nbest by mean test MSE: {best} (paper best: C)")
    if verdict:
        print(f"D beats C on every seed: {verdict['d_beats_c_on_every_seed']}  "
              f"(relative MSE gap {verdict['relative_mse_gap_d_vs_c']:.1%})")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
