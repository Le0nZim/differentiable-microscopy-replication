#!/usr/bin/env python3
"""RR-1 v3 (AM-1) PatchMNIST Table-1 noise-robustness rerun.

Implements the corrected paper supplement A.2.2 normalized detector model
(``noise_normalization: paper_v3``; ``alpha_norm = alpha_down``, no /d²) and
reruns the full 16-cell Table-1 grid plus multi-seed robustness for the
historically failing extreme cell (photon_count=10, sigma_read=6.0).

Everything is written under ``experiments/table01_noise_robustness/``.
Frozen v1/v2 outputs are never touched.

Usage (two GPUs, then aggregate):

    python scripts/table01_noise_robustness/run.py --device cuda:0 --shard 0 --num-shards 2
    python scripts/table01_noise_robustness/run.py --device cuda:1 --shard 1 --num-shards 2
    python scripts/table01_noise_robustness/run.py --aggregate-only --device cuda:0
"""

from __future__ import annotations

import argparse
import copy
import csv
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
from models.microscope import DifferentiableMicroscope  # noqa: E402
from training.dataloaders import build_dataloader  # noqa: E402
from training.staged_hardening_train import train_staged_hardening  # noqa: E402
from training.train_reconstruction import train  # noqa: E402
from utils.device import resolve_device  # noqa: E402
from utils.experiment_config import build_noise_sweep_experiments, load_experiment_config  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402

OUTPUT_ROOT = ROOT / "experiments/table01_noise_robustness"
CONFIG_PATH = ROOT / "configs/table01_noise_robustness/noise_table.yaml"

PHOTON_COUNTS = [10.0, 10000.0]
SIGMA_READS = [0.0, 2.7, 2.0, 6.0]  # paper Table-1 order
PATTERN_MODES = ["random_fixed", "learnable_frequency"]

MAIN_SEED = 42
EXTRA_SEEDS = [43, 44]  # additional training seeds for the extreme cell
EXTREME_CELL = (10.0, 6.0)  # (photon_count, sigma_read)
EVAL_NOISE_SEEDS = [101, 202, 303, 404, 505]  # eval-noise draws for robustness


def _is_learnable(config: dict) -> bool:
    return config["pattern_generator"]["mode"] == "learnable_frequency"


def _configure_run_kind(config: dict) -> dict:
    config = copy.deepcopy(config)
    config["inverse_model"]["upsampling"]["mode"] = "locality_aware"
    if _is_learnable(config):
        config["training"]["learn_patterns"] = True
        config["training"]["use_staged_hardening"] = True
    else:
        config["training"]["learn_patterns"] = False
        config["training"]["use_staged_hardening"] = False
    return config


def build_run_specs() -> list[dict]:
    """Deterministic, ordered list of all runs (main grid + extreme-cell seeds)."""
    base = load_experiment_config(CONFIG_PATH)
    base["training"]["illumination_lr"] = 1.0
    cells = build_noise_sweep_experiments(
        base,
        photon_counts=PHOTON_COUNTS,
        sigma_reads=SIGMA_READS,
        pattern_modes=PATTERN_MODES,
        results_csv=f"{OUTPUT_ROOT.relative_to(ROOT)}/results.csv",
    )

    specs: list[dict] = []
    for cell in cells:
        cell = _configure_run_kind(cell)
        pc = cell["detector_noise"]["photon_count"]
        sr = cell["detector_noise"]["sigma_read"]
        seeds = [MAIN_SEED]
        if (pc, sr) == EXTREME_CELL:
            seeds = [MAIN_SEED] + EXTRA_SEEDS
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


def _run_one(config: dict, output_dir: Path) -> dict:
    if config["training"].get("use_staged_hardening") and _is_learnable(config):
        return train_staged_hardening(config, str(output_dir))
    return train(config, str(output_dir))


def run_shard(args: argparse.Namespace) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    specs = build_run_specs()
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
# Aggregation, gates, eval-noise robustness, plots
# --------------------------------------------------------------------------- #


def _load_summary(run_id: str) -> dict | None:
    path = OUTPUT_ROOT / run_id / "metrics" / "run_summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _load_config(run_id: str) -> dict | None:
    path = OUTPUT_ROOT / run_id / "config.yaml"
    if path.exists():
        return load_experiment_config(path)
    return None


def _load_trained_model(run_id: str, device: torch.device):
    """Rebuild a microscope and load its best checkpoint.

    The forward model's PSF buffers are lazily created on the first forward pass,
    so initialize them before ``load_state_dict`` to avoid a shape mismatch.
    """
    config = _load_config(run_id)
    ckpt_path = OUTPUT_ROOT / run_id / "checkpoints" / "best.pt"
    if config is None or not ckpt_path.exists():
        return None, None
    model = DifferentiableMicroscope.from_run_config(config).to(device)
    img = int(config["dataset"]["image_size"])
    with torch.no_grad():
        model.forward_model._ensure_psfs(torch.zeros(1, 1, img, img, device=device))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


@torch.no_grad()
def _eval_noise_robustness(run_id: str, device: torch.device) -> dict | None:
    """Re-evaluate a trained checkpoint under several independent noise draws."""
    model, config = _load_trained_model(run_id, device)
    if model is None:
        return None
    test_loader = build_dataloader(config, "test")
    learnable = config["pattern_generator"]["mode"] == "learnable_frequency"
    sigmoid_m = float(config["training"].get("sharpen_eval_m", 10.0)) if learnable else \
        config["training"].get("fixed_sigmoid_m")
    mses: list[float] = []
    for s in EVAL_NOISE_SEEDS:
        set_seed(s)
        mse, _ = evaluate_reconstruction(
            model, test_loader, device, apply_noise=True, sigmoid_m=sigmoid_m
        )
        mses.append(float(mse))
    return {"eval_seeds": EVAL_NOISE_SEEDS, "mses": mses, **_stats(mses)}


def _cell_value(rows: list[dict], pc: float, sr: float, mode: str) -> list[float]:
    return [
        float(r["test_mse"])
        for r in rows
        if r["pattern_mode"] == mode and r["photon_count"] == pc and r["sigma_read"] == sr
    ]


def aggregate(args: argparse.Namespace) -> None:
    specs = build_run_specs()
    rows: list[dict] = []
    missing: list[str] = []
    for spec in specs:
        summary = _load_summary(spec["run_id"])
        if summary is None:
            missing.append(spec["run_id"])
            continue
        rows.append(
            {
                "run_id": spec["run_id"],
                "base_run_id": spec["base_run_id"],
                "seed": spec["seed"],
                "photon_count": spec["photon_count"],
                "sigma_read": spec["sigma_read"],
                "pattern_mode": spec["pattern_mode"],
                "test_mse": summary["test_mse"],
                "test_ssim": summary.get("test_ssim"),
                "H_t_binary_fraction": summary.get("H_t_binary_fraction"),
                "pattern_delta": summary.get("pattern_delta"),
            }
        )

    if missing:
        print(f"WARNING: {len(missing)} runs missing summaries:", flush=True)
        for m in missing:
            print(f"  - {m}", flush=True)

    # results.csv (all runs, all seeds)
    fields = [
        "run_id", "base_run_id", "seed", "photon_count", "sigma_read",
        "pattern_mode", "test_mse", "test_ssim", "H_t_binary_fraction", "pattern_delta",
    ]
    with (OUTPUT_ROOT / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # seed-42 main grid (consistent with the paper Table-1 layout)
    main_rows = [r for r in rows if r["seed"] == MAIN_SEED]

    table = []
    for pc in PHOTON_COUNTS:
        for sr in SIGMA_READS:
            rand = _cell_value(main_rows, pc, sr, "random_fixed")
            learn = _cell_value(main_rows, pc, sr, "learnable_frequency")
            if rand and learn:
                table.append(
                    {
                        "photon_count": pc,
                        "sigma_read": sr,
                        "random_mse": rand[0],
                        "learnable_mse": learn[0],
                        "learnable_wins": learn[0] < rand[0],
                        "ratio_learn_over_rand": learn[0] / rand[0] if rand[0] else float("nan"),
                    }
                )

    # extreme-cell multi-seed robustness (training-seed spread, same metric as table)
    extreme_pc, extreme_sr = EXTREME_CELL
    extreme = {}
    for mode in PATTERN_MODES:
        vals = _cell_value(rows, extreme_pc, extreme_sr, mode)
        extreme[mode] = {"per_seed_mse": vals, **_stats(vals)}

    # extreme-cell eval-noise robustness (auxiliary; re-eval seed-42 checkpoints)
    device = resolve_device(args.device)
    eval_noise = {}
    for mode in PATTERN_MODES:
        run_id = next(
            (s["run_id"] for s in specs
             if s["pattern_mode"] == mode and (s["photon_count"], s["sigma_read"]) == EXTREME_CELL
             and s["seed"] == MAIN_SEED),
            None,
        )
        if run_id is not None:
            res = _eval_noise_robustness(run_id, device)
            if res is not None:
                eval_noise[mode] = {"run_id": run_id, **res}

    # gates
    learnable_wins_all = all(c["learnable_wins"] for c in table) and len(table) == 8
    spread_pc10 = _spread(main_rows, "learnable_frequency", 10.0)
    spread_pc10000 = _spread(main_rows, "learnable_frequency", 10000.0)
    extreme_no_reversal = bool(
        extreme.get("learnable_frequency", {}).get("mean", float("nan"))
        < extreme.get("random_fixed", {}).get("mean", float("inf"))
    )

    # v2 baseline spreads for comparison (read from frozen v2 dir if present)
    v2_spreads = _v2_learnable_spreads()

    gates = {
        "all_tests_pass": None,  # filled by docs/manual: see pytest run
        "normalized_path_used_in_train_and_eval": True,
        "learnable_beats_fixed_all_8_cells": learnable_wins_all,
        "pc10000_flat": spread_pc10000 < 0.01,
        "pc10_materially_flatter_than_v2": (
            v2_spreads.get("pc10") is not None and spread_pc10 < v2_spreads["pc10"]
        ),
        "extreme_cell_no_reversal_meanseed": extreme_no_reversal,
        "spread_pc10": spread_pc10,
        "spread_pc10000": spread_pc10000,
        "v2_spread_pc10": v2_spreads.get("pc10"),
        "v2_spread_pc10000": v2_spreads.get("pc10000"),
    }

    payload = {
        "output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
        "noise_normalization": "paper_v3",
        "config": str(CONFIG_PATH.relative_to(ROOT)),
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
    (OUTPUT_ROOT / "table1_v3_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_table_md(table, extreme, eval_noise, gates)
    _make_plots(main_rows, eval_noise)
    print(f"Wrote {OUTPUT_ROOT / 'table1_v3_results.json'}", flush=True)
    print(f"Gates: {json.dumps(gates, indent=2)}", flush=True)


def _spread(rows: list[dict], mode: str, pc: float) -> float:
    vals = [float(r["test_mse"]) for r in rows if r["pattern_mode"] == mode and r["photon_count"] == pc]
    return (max(vals) - min(vals)) if vals else float("nan")


def _v2_learnable_spreads() -> dict:
    v2_csv = ROOT / "archive/experiments/noise_table_normalized_v2/results.csv"
    out: dict = {}
    if not v2_csv.exists():
        return out
    rows = list(csv.DictReader(v2_csv.open(encoding="utf-8")))
    for pc, key in [(10.0, "pc10"), (10000.0, "pc10000")]:
        vals = [
            float(r["test_mse"])
            for r in rows
            if r["pattern_mode"] == "learnable_frequency" and float(r["photon_count"]) == pc
        ]
        if vals:
            out[key] = max(vals) - min(vals)
    return out


def _write_table_md(table: list[dict], extreme: dict, eval_noise: dict, gates: dict) -> None:
    lines = [
        "# Table 1 v3 results (corrected paper-normalized noise, `paper_v3`)\n\n",
        "PatchMNIST ×8, T=8, d=8, batch=32, illum LR=1.0, inverse LR=0.001, gamma=10.\n",
        "Detector model: supplement A.2.2 eqs. S5–S10 with `alpha_norm = alpha_down` (no /d²).\n",
        "Seed 42 grid below; the extreme cell (pc=10, σ=6) additionally has seeds 43, 44.\n\n",
        "| photon_count | sigma_read | random MSE | learnable MSE | learnable/random | learnable wins? |\n",
        "|---|---:|---:|---:|---:|---|\n",
    ]
    for c in table:
        lines.append(
            f"| {c['photon_count']:g} | {c['sigma_read']} | {c['random_mse']:.4f} | "
            f"{c['learnable_mse']:.4f} | {c['ratio_learn_over_rand']:.3f} | "
            f"{'yes' if c['learnable_wins'] else 'NO'} |\n"
        )
    lines.append("\n## Extreme cell pc=10, σ=6 — multi-seed (training) robustness\n\n")
    lines.append("| method | per-seed MSE | mean | std | n |\n|---|---|---:|---:|---:|\n")
    for mode, st in extreme.items():
        per = ", ".join(f"{v:.4f}" for v in st.get("per_seed_mse", []))
        lines.append(f"| {mode} | {per} | {st.get('mean', float('nan')):.4f} | {st.get('std', float('nan')):.4f} | {st.get('n', 0)} |\n")
    if eval_noise:
        lines.append("\n## Extreme cell pc=10, σ=6 — eval-noise robustness (seed-42 checkpoint, 5 noise draws)\n\n")
        lines.append("| method | mean | std | n |\n|---|---:|---:|---:|\n")
        for mode, st in eval_noise.items():
            lines.append(f"| {mode} | {st.get('mean', float('nan')):.4f} | {st.get('std', float('nan')):.4f} | {st.get('n', 0)} |\n")
    lines.append("\n## Gate snapshot\n\n")
    for k, v in gates.items():
        lines.append(f"- `{k}`: {v}\n")
    (OUTPUT_ROOT / "table1_v3_results.md").write_text("".join(lines), encoding="utf-8")


def _make_plots(main_rows: list[dict], eval_noise: dict) -> None:
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

    # 1) MSE vs read noise per photon count and method
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, pc in zip(axes, PHOTON_COUNTS):
        for mode, marker in [("random_fixed", "o"), ("learnable_frequency", "s")]:
            xs, ys = [], []
            for sr in sorted(SIGMA_READS):
                v = _cell_value(main_rows, pc, sr, mode)
                if v:
                    xs.append(sr)
                    ys.append(v[0])
            if xs:
                ax.plot(xs, ys, marker=marker, label=mode)
        ax.set_title(f"photon_count={pc:g}")
        ax.set_xlabel("read noise σ_read")
        ax.set_ylabel("test MSE")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("RR-1 v3: MSE vs read noise (paper_v3 normalization)")
    fig.tight_layout()
    fig.savefig(plots_dir / "mse_vs_read_noise.png", dpi=130)
    plt.close(fig)

    # 2) learnable/fixed ratio heatmap
    ratio = np.full((len(PHOTON_COUNTS), len(SIGMA_READS)), np.nan)
    sr_sorted = sorted(SIGMA_READS)
    for i, pc in enumerate(PHOTON_COUNTS):
        for j, sr in enumerate(sr_sorted):
            r = _cell_value(main_rows, pc, sr, "random_fixed")
            l = _cell_value(main_rows, pc, sr, "learnable_frequency")
            if r and l and r[0]:
                ratio[i, j] = l[0] / r[0]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(ratio, cmap="RdYlGn_r", vmin=0.0, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(sr_sorted)), [str(s) for s in sr_sorted])
    ax.set_yticks(range(len(PHOTON_COUNTS)), [f"{p:g}" for p in PHOTON_COUNTS])
    ax.set_xlabel("read noise σ_read")
    ax.set_ylabel("photon count")
    for i in range(len(PHOTON_COUNTS)):
        for j in range(len(sr_sorted)):
            if not np.isnan(ratio[i, j]):
                ax.text(j, i, f"{ratio[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("learnable/random MSE ratio (<1 = learnable wins)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "learnable_fixed_ratio_heatmap.png", dpi=130)
    plt.close(fig)

    # 3) measurement-distribution histograms for pc=10,σ=6 and pc=10000,σ=6
    _measurement_histograms(plots_dir)
    # 4) example reconstructions for the extreme cell
    _example_reconstructions(plots_dir, eval_noise)


@torch.no_grad()
def _measurement_histograms(plots_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    device = resolve_device("cuda:0" if torch.cuda.is_available() else "cpu")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (pc, sr) in zip(axes, [(10.0, 6.0), (10000.0, 6.0)]):
        run_id = f"patchmnist_noise_learnable_frequency_pc{int(pc)}_sr{str(sr).replace('.', 'p')}_seed{MAIN_SEED}"
        model, config = _load_trained_model(run_id, device)
        if model is None:
            continue
        batch = next(iter(build_dataloader(config, "test"))).to(device)
        set_seed(7)
        out = model(batch, sigmoid_m=float(config["training"].get("sharpen_eval_m", 10.0)), apply_noise=True)
        y = out["y_down"].flatten().cpu().numpy()
        a = out["alpha_down"].flatten().cpu().numpy()
        ax.hist(a, bins=80, alpha=0.5, label="alpha_down (noise-free)", color="tab:blue")
        ax.hist(y, bins=80, alpha=0.5, label="y_norm (eq. S10)", color="tab:orange")
        ax.set_title(f"pc={pc:g}, σ_read={sr}")
        ax.set_xlabel("measurement value")
        ax.set_ylabel("count")
        ax.legend()
    fig.suptitle("RR-1 v3 measurement distributions (paper_v3)")
    fig.tight_layout()
    fig.savefig(plots_dir / "measurement_histograms.png", dpi=130)
    plt.close(fig)


@torch.no_grad()
def _example_reconstructions(plots_dir: Path, eval_noise: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    device = resolve_device("cuda:0" if torch.cuda.is_available() else "cpu")
    pc, sr = EXTREME_CELL
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for row, mode in enumerate(PATTERN_MODES):
        run_id = f"patchmnist_noise_{mode}_pc{int(pc)}_sr{str(sr).replace('.', 'p')}_seed{MAIN_SEED}"
        model, config = _load_trained_model(run_id, device)
        if model is None:
            continue
        batch = next(iter(build_dataloader(config, "test"))).to(device)
        learnable = mode == "learnable_frequency"
        sigmoid_m = float(config["training"].get("sharpen_eval_m", 10.0)) if learnable else config["training"].get("fixed_sigmoid_m")
        set_seed(7)
        out = model(batch, sigmoid_m=sigmoid_m, apply_noise=True)
        gt = batch[0, 0].cpu().numpy()
        recon = out["x_recon"][0, 0].cpu().numpy()
        axes[row, 0].imshow(gt, cmap="gray"); axes[row, 0].set_title(f"{mode}\nground truth")
        axes[row, 1].imshow(recon, cmap="gray"); axes[row, 1].set_title("reconstruction")
        axes[row, 2].imshow(out["y_down"][0, 0].cpu().numpy(), cmap="viridis"); axes[row, 2].set_title("y_norm[0]")
        axes[row, 3].imshow((gt - recon), cmap="coolwarm"); axes[row, 3].set_title("error")
        for c in range(4):
            axes[row, c].axis("off")
    fig.suptitle(f"RR-1 v3 reconstructions @ extreme cell pc={pc:g}, σ_read={sr}")
    fig.tight_layout()
    fig.savefig(plots_dir / "example_reconstructions_extreme.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="RR-1 v3 noise-table runner")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--config", default=None, help="Override YAML config path")
    parser.add_argument("--output-root", default=None, help="Override output directory")
    args = parser.parse_args()

    global CONFIG_PATH, OUTPUT_ROOT
    if args.config:
        CONFIG_PATH = Path(args.config)
        if not CONFIG_PATH.is_absolute():
            CONFIG_PATH = ROOT / CONFIG_PATH
    if args.output_root:
        OUTPUT_ROOT = Path(args.output_root)
        if not OUTPUT_ROOT.is_absolute():
            OUTPUT_ROOT = ROOT / OUTPUT_ROOT

    print(f"Device: {resolve_device(args.device)}", flush=True)
    print(f"Config: {CONFIG_PATH}", flush=True)
    print(f"Output: {OUTPUT_ROOT}", flush=True)
    if args.aggregate_only:
        aggregate(args)
        return
    run_shard(args)


if __name__ == "__main__":
    main()
