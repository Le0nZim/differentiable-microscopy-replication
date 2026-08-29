#!/usr/bin/env python3
"""Figure 6 / Table 1 noise robustness WITHOUT frequency-domain optimization.

Table 1 / Figure 6 compare ``random_fixed`` illumination against
``learnable_frequency`` -- so the learnable arm of the published noise-robustness
result carries the paper's frequency-domain parameterization. The Figure-10
ablation found that dropping it (``learnable_spatial``, the paper's variant D)
*improves* reconstruction on both BBBC022 and PatchMNIST. This study adds the
missing third arm to the noise sweep and asks the same question under detector
noise.

It reuses the Table-1 runner verbatim for everything that trains
(``_configure_run_kind`` -> ``build_noise_sweep_experiments`` ->
``train_staged_hardening``); only the pattern mode and the output root differ.
The frozen Table-1 runs are read back read-only for the comparison, never
retrained.

By default a SUBSET of the grid is run: the four corner cells
(photon_count x sigma_read over {10, 10000} x {0.0, 6.0}), which span both photon
regimes and both read-noise extremes, plus seeds 43/44 at the extreme cell
(pc=10, sigma=6.0) to match Table 1's multi-seed protocol. ``--cells full`` runs
all eight cells instead.

Usage (two GPUs, then aggregate):

    python scripts/figure06_noise_robustness_no_freq/run.py --device cuda:0 --shard 0 --num-shards 2
    python scripts/figure06_noise_robustness_no_freq/run.py --device cuda:1 --shard 1 --num-shards 2
    python scripts/figure06_noise_robustness_no_freq/run.py --aggregate-only --device cuda:0
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from evaluation.eval_reconstruction import evaluate_reconstruction  # noqa: E402
from evaluation.variant_audit import pattern_is_learnable  # noqa: E402
from models.microscope import DifferentiableMicroscope  # noqa: E402
from training.dataloaders import build_dataloader  # noqa: E402
from utils.device import resolve_device  # noqa: E402
from utils.experiment_config import build_noise_sweep_experiments, load_experiment_config  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "table01_run", ROOT / "scripts/table01_noise_robustness/run.py"
)
_t01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_t01)
_configure_run_kind = _t01._configure_run_kind
_run_one = _t01._run_one

OUTPUT_ROOT = ROOT / "experiments/figure06_noise_robustness_no_freq"
CONFIG_PATH = ROOT / "configs/figure06_noise_robustness_no_freq/noise_table.yaml"
BASELINE_ROOT = ROOT / "experiments/table01_noise_robustness"

MODE = "learnable_spatial"
BASELINE_MODES = ["random_fixed", "learnable_frequency"]
ALL_MODES = BASELINE_MODES + [MODE]
MODE_LABEL = {
    "random_fixed": "fixed random",
    "learnable_frequency": "learnable + freq (paper)",
    "learnable_spatial": "learnable, NO freq",
}

PHOTON_COUNTS = [10.0, 10000.0]
SIGMA_READS = [0.0, 2.7, 2.0, 6.0]  # paper Table-1 order
CORNER_SIGMA_READS = [0.0, 6.0]

MAIN_SEED = 42
EXTRA_SEEDS = [43, 44]
EXTREME_CELL = (10.0, 6.0)
EVAL_NOISE_SEEDS = [101, 202, 303, 404, 505]


def _cells(which: str) -> list[tuple[float, float]]:
    sigmas = SIGMA_READS if which == "full" else CORNER_SIGMA_READS
    return [(pc, sr) for pc in PHOTON_COUNTS for sr in sigmas]


def build_run_specs(which: str) -> list[dict]:
    """Deterministic, ordered list of runs for the requested cell subset."""
    base = load_experiment_config(CONFIG_PATH)
    base["training"]["illumination_lr"] = 1.0
    wanted = set(_cells(which))
    grid = build_noise_sweep_experiments(
        base,
        photon_counts=PHOTON_COUNTS,
        sigma_reads=SIGMA_READS,
        pattern_modes=[MODE],
        results_csv=f"{OUTPUT_ROOT.relative_to(ROOT)}/results.csv",
    )

    specs: list[dict] = []
    for cell in grid:
        pc = cell["detector_noise"]["photon_count"]
        sr = cell["detector_noise"]["sigma_read"]
        if (pc, sr) not in wanted:
            continue
        cell = _configure_run_kind(cell)
        seeds = [MAIN_SEED] + EXTRA_SEEDS if (pc, sr) == EXTREME_CELL else [MAIN_SEED]
        for seed in seeds:
            run_id = cell["experiment"]["run_id"]
            specs.append(
                {
                    "run_id": f"{run_id}_seed{seed}",
                    "base_run_id": run_id,
                    "seed": seed,
                    "photon_count": pc,
                    "sigma_read": sr,
                    "pattern_mode": cell["pattern_generator"]["mode"],
                    "config": cell,
                }
            )
    return specs


def _materialize_config(spec: dict, device: str) -> dict:
    config = copy.deepcopy(spec["config"])
    seed = spec["seed"]
    config["experiment"]["seed"] = seed
    config["dataset"]["seed"] = seed
    config["pattern_generator"]["seed"] = seed
    config["experiment"]["device"] = device
    config["experiment"]["run_id"] = spec["run_id"]
    return config


def run_shard(args: argparse.Namespace) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    specs = build_run_specs(args.cells)
    for idx, spec in enumerate(specs):
        if args.num_shards > 1 and idx % args.num_shards != args.shard:
            continue
        out_dir = OUTPUT_ROOT / spec["run_id"]
        if (out_dir / "metrics" / "run_summary.json").exists():
            print(f"[shard {args.shard}] skip existing {spec['run_id']}", flush=True)
            continue
        config = _materialize_config(spec, args.device)
        print(
            f"\n[shard {args.shard}] ===== {spec['run_id']} "
            f"(pc={spec['photon_count']}, sr={spec['sigma_read']}, "
            f"mode={spec['pattern_mode']}) =====",
            flush=True,
        )
        _run_one(config, out_dir)
    print(f"[shard {args.shard}] done", flush=True)


# --------------------------------------------------------------------------- #
# Aggregation against the frozen Table-1 baselines
# --------------------------------------------------------------------------- #


def _run_root(mode: str) -> Path:
    return OUTPUT_ROOT if mode == MODE else BASELINE_ROOT


def _run_id(mode: str, pc: float, sr: float, seed: int) -> str:
    """Same naming rule as ``build_noise_sweep_experiments`` + the seed suffix."""
    base = f"patchmnist_noise_{mode}_pc{int(pc)}_sr{sr}".replace(".", "p")
    return f"{base}_seed{seed}"


def _summary(mode: str, pc: float, sr: float, seed: int) -> dict | None:
    run_id = _run_id(mode, pc, sr, seed)
    path = _run_root(mode) / run_id / "metrics" / "run_summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _mse(mode: str, pc: float, sr: float, seed: int = MAIN_SEED) -> float | None:
    s = _summary(mode, pc, sr, seed)
    return float(s["test_mse"]) if s else None


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "n": 0, "values": []}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
        "values": values,
    }


def _load_trained_model(mode: str, pc: float, sr: float, seed: int, device: torch.device):
    """Rebuild a microscope and load its best checkpoint (PSF buffers are lazy)."""
    run_dir = _run_root(mode) / _run_id(mode, pc, sr, seed)
    cfg_path = run_dir / "config.yaml"
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not cfg_path.exists() or not ckpt_path.exists():
        return None, None
    config = load_experiment_config(cfg_path)
    model = DifferentiableMicroscope.from_run_config(config).to(device)
    img = int(config["dataset"]["image_size"])
    with torch.no_grad():
        model.forward_model._ensure_psfs(torch.zeros(1, 1, img, img, device=device))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def _eval_sigmoid_m(config: dict) -> float | None:
    mode = config["pattern_generator"]["mode"]
    if pattern_is_learnable(mode):
        return float(config["training"].get("sharpen_eval_m", 10.0))
    return config["training"].get("fixed_sigmoid_m")


@torch.no_grad()
def _eval_noise_robustness(mode: str, pc: float, sr: float, device: torch.device) -> dict | None:
    """Re-evaluate a trained checkpoint under several independent noise draws."""
    model, config = _load_trained_model(mode, pc, sr, MAIN_SEED, device)
    if model is None:
        return None
    test_loader = build_dataloader(config, "test")
    sigmoid_m = _eval_sigmoid_m(config)
    mses: list[float] = []
    for s in EVAL_NOISE_SEEDS:
        set_seed(s)
        mse, _ = evaluate_reconstruction(
            model, test_loader, device, apply_noise=True, sigmoid_m=sigmoid_m
        )
        mses.append(float(mse))
    return {"eval_seeds": EVAL_NOISE_SEEDS, **_stats(mses)}


def aggregate(args: argparse.Namespace) -> None:
    cells = _cells(args.cells)

    rows: list[dict] = []
    missing: list[str] = []
    for pc, sr in cells:
        seeds = [MAIN_SEED] + EXTRA_SEEDS if (pc, sr) == EXTREME_CELL else [MAIN_SEED]
        for mode in ALL_MODES:
            for seed in seeds:
                s = _summary(mode, pc, sr, seed)
                if s is None:
                    if mode == MODE:
                        missing.append(_run_id(mode, pc, sr, seed))
                    continue
                rows.append(
                    {
                        "run_id": _run_id(mode, pc, sr, seed),
                        "seed": seed,
                        "photon_count": pc,
                        "sigma_read": sr,
                        "pattern_mode": mode,
                        "test_mse": s["test_mse"],
                        "test_ssim": s.get("test_ssim"),
                        "H_t_binary_fraction": s.get("H_t_binary_fraction"),
                        "pattern_delta": s.get("pattern_delta"),
                        "source": "this study" if mode == MODE else "table01 (frozen)",
                    }
                )

    if missing:
        print(f"WARNING: {len(missing)} no-freq runs missing summaries:", flush=True)
        for m in missing:
            print(f"  - {m}", flush=True)

    fields = [
        "run_id", "seed", "photon_count", "sigma_read", "pattern_mode",
        "test_mse", "test_ssim", "H_t_binary_fraction", "pattern_delta", "source",
    ]
    with (OUTPUT_ROOT / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    table = []
    for pc, sr in cells:
        rand = _mse("random_fixed", pc, sr)
        freq = _mse("learnable_frequency", pc, sr)
        spat = _mse(MODE, pc, sr)
        if rand is None or freq is None or spat is None:
            continue
        table.append(
            {
                "photon_count": pc,
                "sigma_read": sr,
                "random_mse": rand,
                "frequency_mse": freq,
                "spatial_mse": spat,
                "ratio_spatial_over_frequency": spat / freq,
                "spatial_beats_frequency": spat < freq,
                "spatial_beats_random": spat < rand,
            }
        )

    extreme_pc, extreme_sr = EXTREME_CELL
    extreme = {}
    for mode in ALL_MODES:
        vals = [
            v for v in (_mse(mode, extreme_pc, extreme_sr, s)
                        for s in [MAIN_SEED] + EXTRA_SEEDS) if v is not None
        ]
        extreme[mode] = _stats(vals)

    device = resolve_device(args.device)
    eval_noise = {}
    for mode in ALL_MODES:
        res = _eval_noise_robustness(mode, extreme_pc, extreme_sr, device)
        if res is not None:
            eval_noise[mode] = res

    freq_ex = extreme.get("learnable_frequency", {}).get("mean")
    spat_ex = extreme.get(MODE, {}).get("mean")
    gates = {
        "cells_compared": len(table),
        "spatial_beats_frequency_all_cells": bool(table) and all(c["spatial_beats_frequency"] for c in table),
        "spatial_beats_random_all_cells": bool(table) and all(c["spatial_beats_random"] for c in table),
        "spatial_beats_frequency_at_extreme_cell_meanseed": bool(
            freq_ex is not None and spat_ex is not None and spat_ex < freq_ex
        ),
        "median_ratio_spatial_over_frequency": (
            statistics.median([c["ratio_spatial_over_frequency"] for c in table]) if table else None
        ),
    }

    payload = {
        "label": (
            "Figure 6 / Table 1 noise robustness WITHOUT frequency-domain optimization "
            "(learnable_spatial), vs the frozen Table-1 random_fixed and learnable_frequency arms."
        ),
        "output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
        "baseline_root": str(BASELINE_ROOT.relative_to(ROOT)),
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "cells_subset": args.cells,
        "noise_normalization": "paper_v3",
        "main_seed": MAIN_SEED,
        "extra_seeds": EXTRA_SEEDS,
        "extreme_cell": {"photon_count": extreme_pc, "sigma_read": extreme_sr},
        "table_seed42": table,
        "extreme_multiseed": extreme,
        "extreme_eval_noise": eval_noise,
        "gates": gates,
        "missing_runs": missing,
        "all_rows": rows,
    }
    (OUTPUT_ROOT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_md(table, extreme, eval_noise, gates)
    _make_plots(table, eval_noise)
    print(f"Wrote {OUTPUT_ROOT / 'results.json'}", flush=True)
    print(f"Gates: {json.dumps(gates, indent=2)}", flush=True)


def _write_md(table: list[dict], extreme: dict, eval_noise: dict, gates: dict) -> None:
    lines = [
        "# Figure 6 / Table 1 noise robustness — no frequency-domain optimization\n\n",
        "PatchMNIST ×8, T=8, d=8, batch=32, illum LR=1.0, inverse LR=0.001, gamma=10, "
        "`paper_v3` detector normalization — identical to Table 1.\n",
        "`learnable_spatial` runs are new; `random_fixed` and `learnable_frequency` are read "
        "from the frozen `experiments/table01_noise_robustness/` runs.\n\n",
        "| photon_count | sigma_read | fixed MSE | learnable+freq MSE | learnable NO-freq MSE | no-freq / freq | no-freq wins? |\n",
        "|---|---:|---:|---:|---:|---:|---|\n",
    ]
    for c in table:
        lines.append(
            f"| {c['photon_count']:g} | {c['sigma_read']} | {c['random_mse']:.4f} | "
            f"{c['frequency_mse']:.4f} | {c['spatial_mse']:.4f} | "
            f"{c['ratio_spatial_over_frequency']:.3f} | "
            f"{'yes' if c['spatial_beats_frequency'] else 'NO'} |\n"
        )
    lines.append("\n## Extreme cell pc=10, σ=6 — multi-seed (training) robustness\n\n")
    lines.append("| method | per-seed MSE | mean | std | n |\n|---|---|---:|---:|---:|\n")
    for mode in ALL_MODES:
        st = extreme.get(mode, {})
        if not st.get("n"):
            continue
        per = ", ".join(f"{v:.4f}" for v in st["values"])
        lines.append(f"| {MODE_LABEL[mode]} | {per} | {st['mean']:.4f} | {st['std']:.4f} | {st['n']} |\n")
    if eval_noise:
        lines.append("\n## Extreme cell pc=10, σ=6 — eval-noise robustness (seed-42 checkpoint, 5 noise draws)\n\n")
        lines.append("| method | mean | std | n |\n|---|---:|---:|---:|\n")
        for mode in ALL_MODES:
            st = eval_noise.get(mode)
            if not st:
                continue
            lines.append(f"| {MODE_LABEL[mode]} | {st['mean']:.4f} | {st['std']:.4f} | {st['n']} |\n")
    lines.append("\n## Gate snapshot\n\n")
    for k, v in gates.items():
        lines.append(f"- `{k}`: {v}\n")
    (OUTPUT_ROOT / "results.md").write_text("".join(lines), encoding="utf-8")


def _make_plots(table: list[dict], eval_noise: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        print(f"Plotting skipped ({exc})", flush=True)
        return

    plots_dir = OUTPUT_ROOT / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    keys = [("random_mse", "random_fixed"), ("frequency_mse", "learnable_frequency"),
            ("spatial_mse", MODE)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, pc in zip(axes, PHOTON_COUNTS):
        cells = sorted([c for c in table if c["photon_count"] == pc], key=lambda c: c["sigma_read"])
        for (key, mode), marker in zip(keys, ["o", "s", "^"]):
            if cells:
                ax.plot([c["sigma_read"] for c in cells], [c[key] for c in cells],
                        marker=marker, label=MODE_LABEL[mode])
        ax.set_title(f"photon_count={pc:g}")
        ax.set_xlabel("read noise σ_read")
        ax.set_ylabel("test MSE")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Noise robustness with and without frequency-domain optimization")
    fig.tight_layout()
    fig.savefig(plots_dir / "mse_vs_read_noise.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(table))
    w = 0.27
    for i, (key, mode) in enumerate(keys):
        ax.bar(x + (i - 1) * w, [c[key] for c in table], w, label=MODE_LABEL[mode])
    ax.set_xticks(x)
    ax.set_xticklabels([f"pc={c['photon_count']:g}\nσ={c['sigma_read']:g}" for c in table], fontsize=8)
    ax.set_ylabel("test MSE (lower better)")
    ax.set_title("Per-cell test MSE: does removing frequency-domain optimization help?")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "per_cell_mse.png", dpi=130)
    plt.close(fig)

    _qualitative_panel(plots_dir)


@torch.no_grad()
def _qualitative_panel(plots_dir: Path) -> None:
    """Figure-6-style panel at the extreme cell, with the no-freq arm added."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    device = resolve_device("cuda:0" if torch.cuda.is_available() else "cpu")
    pc, sr = EXTREME_CELL
    entries = []
    for mode in ALL_MODES:
        model, config = _load_trained_model(mode, pc, sr, MAIN_SEED, device)
        if model is None:
            continue
        batch = next(iter(build_dataloader(config, "test"))).to(device)
        set_seed(7)
        out = model(batch, sigmoid_m=_eval_sigmoid_m(config), apply_noise=True)
        entries.append((mode, batch, out))
    if not entries:
        return

    gt = entries[0][1][0, 0].cpu().numpy()
    fig, axes = plt.subplots(1, len(entries) + 1, figsize=(3.0 * (len(entries) + 1), 3.4))
    axes[0].imshow(gt, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("ground truth", fontsize=10)
    axes[0].axis("off")
    for i, (mode, _, out) in enumerate(entries, start=1):
        axes[i].imshow(out["x_recon"][0, 0].cpu().numpy().clip(0, 1), cmap="viridis", vmin=0, vmax=1)
        axes[i].set_title(MODE_LABEL[mode], fontsize=10)
        axes[i].axis("off")
    fig.suptitle(f"Figure 6 style — extreme cell pc={pc:g}, σ_read={sr} (first test batch)", fontsize=11)
    fig.tight_layout()
    fig.savefig(plots_dir / "fig06_style_extreme_cell.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fig-6 / Table-1 noise sweep without freq-domain optimization")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--cells", choices=["corners", "full"], default="corners",
                        help="corners = {10,10000} x {0.0,6.0}; full = the whole Table-1 grid")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    print(f"Device: {resolve_device(args.device)}", flush=True)
    print(f"Config: {CONFIG_PATH}", flush=True)
    print(f"Output: {OUTPUT_ROOT}  (baselines read from {BASELINE_ROOT})", flush=True)
    if args.aggregate_only:
        aggregate(args)
        return
    run_shard(args)


if __name__ == "__main__":
    main()
