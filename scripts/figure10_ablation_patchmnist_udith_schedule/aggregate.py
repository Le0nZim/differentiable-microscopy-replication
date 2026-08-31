#!/usr/bin/env python3
"""Aggregate C/D Udith-schedule runs and write the comparison report + figures."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/figure10_ablation_patchmnist_udith_schedule"
RUNS = EXP / "runs"
FROZEN = ROOT / "experiments/figure10_ablation_patchmnist"

CURRENT_C = {"mse_mean": 0.006948, "mse_std": 0.000062, "ssim_mean": 0.9357, "ssim_std": 0.0012}
CURRENT_D = {"mse_mean": 0.006553, "mse_std": 0.000021, "ssim_mean": 0.9401, "ssim_std": 0.0002}

SEEDS = [42, 43, 44]


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "n": 0, "values": []}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
        "values": values,
    }


def _load_summary(runs: Path, letter: str, seed: int) -> dict | None:
    path = runs / f"{letter}_seed{seed}" / "metrics" / "run_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _load_diag(runs: Path, letter: str, seed: int) -> dict | None:
    path = runs / f"{letter}_seed{seed}" / "metrics" / "diagnostics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def collect(runs: Path) -> dict:
    per: dict[str, list[dict]] = {"C": [], "D": []}
    for letter in ("C", "D"):
        for seed in SEEDS:
            s = _load_summary(runs, letter, seed)
            if s is None:
                continue
            d = _load_diag(runs, letter, seed) or {}
            per[letter].append({
                "seed": seed,
                "test_mse": s["test_mse"],
                "test_ssim": s["test_ssim"],
                "best_m": s.get("best_m", d.get("best_m")),
                "best_step": s.get("best_step", d.get("best_step")),
                "H_t_binary_fraction": s.get(
                    "H_t_binary_fraction",
                    (d.get("final_pattern_stats") or {}).get("binary_fraction"),
                ),
                "tau_displacement": s.get("tau_displacement", d.get("final_tau_displacement")),
                "Ht_displacement": s.get("Ht_displacement", d.get("final_Ht_displacement_vs_m1_init")),
            })
        per[letter].sort(key=lambda r: r["seed"])

    paired = []
    by_c = {r["seed"]: r for r in per["C"]}
    by_d = {r["seed"]: r for r in per["D"]}
    for seed in SEEDS:
        if seed in by_c and seed in by_d:
            c, d = by_c[seed], by_d[seed]
            d_minus_c_mse = d["test_mse"] - c["test_mse"]
            d_minus_c_ssim = d["test_ssim"] - c["test_ssim"]
            paired.append({
                "seed": seed,
                "C_mse": c["test_mse"],
                "D_mse": d["test_mse"],
                "C_ssim": c["test_ssim"],
                "D_ssim": d["test_ssim"],
                "D_minus_C_mse": d_minus_c_mse,
                "D_minus_C_ssim": d_minus_c_ssim,
                "C_minus_D_mse": -d_minus_c_mse,
                "relative_D_minus_C_over_C": (
                    d_minus_c_mse / c["test_mse"] if c["test_mse"] else None
                ),
                "C_beats_D": c["test_mse"] < d["test_mse"],
            })
    n_c_wins = sum(1 for p in paired if p["C_beats_D"])
    if not paired:
        who = "incomplete"
    elif n_c_wins == len(paired):
        who = "C_beats_D_all_seeds"
    elif n_c_wins == 0:
        who = "D_beats_C_all_seeds"
    else:
        who = "mixed"

    payload = {
        "label": (
            "Udith legacy schedule under the current controlled PatchMNIST "
            "optimizer/data recipe"
        ),
        "seeds_present": {
            "C": [r["seed"] for r in per["C"]],
            "D": [r["seed"] for r in per["D"]],
        },
        "per_seed": per,
        "aggregate": {
            "C": {
                "test_mse": _stats([r["test_mse"] for r in per["C"]]),
                "test_ssim": _stats([r["test_ssim"] for r in per["C"]]),
            },
            "D": {
                "test_mse": _stats([r["test_mse"] for r in per["D"]]),
                "test_ssim": _stats([r["test_ssim"] for r in per["D"]]),
            },
        },
        "paired_D_minus_C": paired,
        "paired_C_minus_D": paired,
        "paired_D_minus_C_aggregate": {
            "test_mse": _stats([p["D_minus_C_mse"] for p in paired]),
            "test_ssim": _stats([p["D_minus_C_ssim"] for p in paired]),
        },
        "verdict": who,
        "current_schedule_frozen": {"C": CURRENT_C, "D": CURRENT_D},
        "interpretation_boundaries": [
            "This tests whether the legacy schedule changes the corrected C/D ordering on PatchMNIST.",
            "It is not a U2OS reproduction.",
            "It does not validate the historical Table-3 D condition.",
            "It does not test the full historical optimizer loop.",
        ],
        "experiment_label": (
            "Udith legacy schedule under the current controlled PatchMNIST "
            "optimizer/data recipe"
        ),
        "not_a_full_reproduction_of_udith_legacy_training_loop": True,
        "phase_boundary_evals_eligible_for_global_best": True,
        "step_121500_always_evaluated": True,
        "train_iterator_note": (
            "The inherited runner uses itertools.cycle(train_loader), if still present. "
            "It caches and repeats the first loader traversal. This remains fair because "
            "both schedules and both C/D arms use it, but differs from Udith’s original "
            "per-epoch loader recreation."
        ),
        "do_not_compare_fourier_param_norms_to_spatial_param_norms": True,
        "physical_illumination_displacement": "tau_displacement = ||τ − τ0||_2; Ht_displacement = ||H_t − H_t0||_2",
    }
    return payload


def write_comparison_csv(payload: dict, path: Path) -> None:
    rows = []
    for letter in ("C", "D"):
        agg = payload["aggregate"][letter]["test_mse"]
        ssim = payload["aggregate"][letter]["test_ssim"]
        frozen = payload["current_schedule_frozen"][letter]
        rows.append({
            "variant": letter,
            "schedule": "udith_legacy_121500",
            "mse_mean": agg["mean"],
            "mse_std": agg["std"],
            "ssim_mean": ssim["mean"],
            "ssim_std": ssim["std"],
            "n": agg["n"],
        })
        rows.append({
            "variant": letter,
            "schedule": "current_8500",
            "mse_mean": frozen["mse_mean"],
            "mse_std": frozen["mse_std"],
            "ssim_mean": frozen["ssim_mean"],
            "ssim_std": frozen["ssim_std"],
            "n": 3,
        })
    for p in payload["paired_D_minus_C"]:
        rows.append({
            "variant": "D-C",
            "schedule": f"udith_seed{p['seed']}",
            "mse_mean": p["D_minus_C_mse"],
            "mse_std": "",
            "ssim_mean": p["D_minus_C_ssim"],
            "ssim_std": "",
            "n": 1,
            "relative_over_C": p["relative_D_minus_C_over_C"],
            "C_beats_D": p["C_beats_D"],
        })
    fields = ["variant", "schedule", "mse_mean", "mse_std", "ssim_mean", "ssim_std", "n",
              "relative_over_C", "C_beats_D"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_step_log(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics" / "step_log.csv"
    if not path.exists():
        return []
    return list(csv.DictReader(path.open()))


def render_cd_panel(runs: Path, out_path: Path, seed: int = 42) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.4))
    plt.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.12, wspace=0.04, hspace=0.04)
    titles = {
        "C": "C  learnable $H_t$ + locality + freq",
        "D": "D  learnable $H_t$ + locality  NO freq",
    }
    for col, letter in enumerate(("C", "D")):
        d = runs / f"{letter}_seed{seed}"
        qt = torch.load(d / "figures" / "qualitative_tensors.pt", map_location="cpu", weights_only=False)
        ht = torch.load(d / "learned_patterns" / "H_t.pt", map_location="cpu", weights_only=False)
        summ = json.loads((d / "metrics" / "run_summary.json").read_text())
        recon = qt["recon"].float().clamp(0, 1)
        n = recon.shape[0]
        axes[0, col].imshow(recon[0, 0].numpy(), cmap="viridis", vmin=0, vmax=1)
        axes[1, col].imshow(ht[0, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[2, col].imshow(recon[min(1, n - 1), 0].numpy(), cmap="viridis", vmin=0, vmax=1)
        for r in range(3):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])
        axes[2, col].set_xlabel(
            f"{titles[letter]}\nSSIM {summ['test_ssim']:.4f} | MSE {summ['test_mse']:.5f}",
            fontsize=9,
        )
    axes[0, 0].set_ylabel("a", rotation=0, fontsize=14, fontweight="bold", labelpad=12)
    axes[1, 0].set_ylabel("b", rotation=0, fontsize=14, fontweight="bold", labelpad=12)
    axes[2, 0].set_ylabel("c", rotation=0, fontsize=14, fontweight="bold", labelpad=12)
    fig.suptitle(
        "C vs D under Udith legacy schedule (PatchMNIST, seed 42)\n"
        "a) test recon #1   b) learned $H_t$   c) test recon #2",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_curves(runs: Path, out_dir: Path, seed: int = 42) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    fig2, ax2 = plt.subplots(figsize=(8.5, 4.2))
    for letter, color in (("C", "#1f77b4"), ("D", "#d62728")):
        rows = _read_step_log(runs / f"{letter}_seed{seed}")
        if not rows:
            continue
        steps = [int(r["step"]) for r in rows]
        ax.plot(steps, [float(r["val_mse"]) for r in rows], label=f"{letter} val MSE", color=color)
        ax2.plot(steps, [float(r["H_t_binary_fraction"]) for r in rows],
                 label=f"{letter} binary fraction", color=color)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("validation MSE")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Validation MSE vs optimizer step (Udith legacy schedule, seed 42)")
    fig.tight_layout()
    fig.savefig(out_dir / "val_mse_curves.png", dpi=140)
    plt.close(fig)

    ax2.set_xlabel("optimizer step")
    ax2.set_ylabel(r"$H_t$ binary fraction")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Pattern binary fraction vs optimizer step (seed 42)")
    fig2.tight_layout()
    fig2.savefig(out_dir / "binary_fraction_curves.png", dpi=140)
    plt.close(fig2)


def render_schedule_comparison(payload: dict, out_path: Path) -> None:
    letters = ["C", "D"]
    x = range(len(letters))
    udith = [payload["aggregate"][L]["test_mse"]["mean"] for L in letters]
    current = [payload["current_schedule_frozen"][L]["mse_mean"] for L in letters]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    w = 0.35
    ax.bar([i - w / 2 for i in x], current, w, label="current 8,500-step", color="#bbbbbb")
    ax.bar([i + w / 2 for i in x], [u if u is not None else 0 for u in udith], w,
           label="Udith 121,500-step", color="#4a90d9")
    ax.set_xticks(list(x))
    ax.set_xticklabels(letters)
    ax.set_ylabel("test MSE (lower better)")
    ax.set_title("Current schedule vs Udith legacy schedule (PatchMNIST C/D)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def write_report(payload: dict, path: Path) -> None:
    c = payload["aggregate"]["C"]
    d = payload["aggregate"]["D"]
    verdict = payload["verdict"]
    if verdict == "C_beats_D_all_seeds":
        conclusion = (
            "C wins on every seed. The C/D ordering is schedule-sensitive, and the "
            "legacy schedule favors frequency-domain parameterization under this dataset."
        )
    elif verdict == "D_beats_C_all_seeds":
        conclusion = (
            "D still wins on every seed. The historical schedule alone does not explain "
            "the reversal on PatchMNIST."
        )
    elif verdict == "mixed":
        conclusion = (
            "Results are mixed across seeds. Report schedule sensitivity without declaring "
            "either parameterization intrinsically superior."
        )
    else:
        conclusion = "Incomplete: not all C/D seeds have finished."

    lines = [
        "# Udith legacy schedule: corrected C vs D on PatchMNIST",
        "",
        "**Label:** Udith legacy schedule under the current controlled PatchMNIST "
        "optimizer/data recipe. This is **not** a full reproduction of Udith’s "
        "legacy training loop.",
        "",
        "**Status:** schedule-only causal experiment. Isolated from "
        "`experiments/figure10_ablation_patchmnist/`.",
        "",
        "## Checkpointing and data iterator (same for C and D)",
        "",
        "- Phase-boundary evaluations **are eligible for global-best checkpoint "
        "selection**. They use the same `best.update` path as regular `log_every` "
        "rows. They are **not** diagnostic-only. This is not changed between C and D.",
        "- Step **121,500** is always evaluated: it is the last step of the last "
        "phase (the five-step \(m=8\) tail). It is **not** on the `log_every=200` "
        "grid (`121500 % 200 == 100`); last-step logging is what includes it.",
        "- The inherited runner uses `itertools.cycle(train_loader)`. It caches and "
        "repeats the first loader traversal. This remains fair because both schedules "
        "and both C/D arms use it, but differs from Udith’s original per-epoch loader "
        "recreation.",
        "- Physical illumination displacement is \(\\|\tau-\\tau_0\\|_2\) and "
        "\(\\|H_t-H_{t,0}\\|_2\). Do **not** compare raw Fourier-parameter norms "
        "against spatial-parameter norms.",
        "",
        "## Interpretation boundaries",
        "",
        "- This tests whether the **legacy schedule** changes the corrected C/D ordering on PatchMNIST.",
        "- It is **not** a U2OS reproduction.",
        "- It does **not** validate the historical Table-3 D condition "
        "(historical D retained an IFFT; ours is direct spatial `learnable_spatial`).",
        "- It does **not** test the full historical optimizer loop "
        "(still the current single Adam, simultaneous update, grad-clip 1.0).",
        "- Do not edit the manuscript or replace any current paper figure from these numbers "
        "until the report is reviewed.",
        "",
        "## What was held fixed vs the completed 8,500-step PatchMNIST run",
        "",
        "Dataset (PatchMNIST 3000/375/375, disjoint val/test), ×16, T=4, d=8, impulse PSFs, "
        "noise-free, locality-aware upsampling, L1, batch 32, illum LR 1.0, inverse LR 0.001, "
        "single Adam with parameter groups, grad-clip 1.0, simultaneous update, global "
        "best-val-MSE checkpointing, validation cadence `log_every=200`.",
        "",
        "The only causal change is the training schedule: 121,500 optimizer steps matching "
        "24,300 legacy epochs × 5 minibatches/epoch, with accumulating m = 1,2,3,4,5,6,7,8 "
        "and a faithful 5-step m=8 tail. C and D share a paired τ₀ initialization "
        "(C: W₀=FFT2(τ₀), D: τ₀).",
        "",
        "## Schedule (optimizer steps)",
        "",
        "| Global steps | State |",
        "| --- | --- |",
        "| 1–60,750 | illumination frozen, m=1 |",
        "| 60,751–97,195 | joint training, m=1 |",
        "| 97,196–101,245 | m=2 |",
        "| 101,246–105,295 | m=3 |",
        "| 105,296–109,345 | m=4 |",
        "| 109,346–113,395 | m=5 |",
        "| 113,396–117,445 | m=6 |",
        "| 117,446–121,495 | m=7 |",
        "| 121,496–121,500 | m=8 |",
        "",
        "## Per-seed paired differences (D − C)",
        "",
        "| seed | C MSE | D MSE | D−C MSE | C SSIM | D SSIM | D−C SSIM | C best m/step | D best m/step | C bin.frac | D bin.frac | C ‖τ−τ₀‖ | D ‖τ−τ₀‖ |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    by_c = {r["seed"]: r for r in payload["per_seed"]["C"]}
    by_d = {r["seed"]: r for r in payload["per_seed"]["D"]}
    for p in payload["paired_D_minus_C"]:
        seed = p["seed"]
        c_row, d_row = by_c[seed], by_d[seed]
        lines.append(
            f"| {seed} | {_fmt(p['C_mse'])} | {_fmt(p['D_mse'])} | {_fmt(p['D_minus_C_mse'])} | "
            f"{_fmt(c_row['test_ssim'], 4)} | {_fmt(d_row['test_ssim'], 4)} | "
            f"{_fmt(p['D_minus_C_ssim'], 4)} | "
            f"{c_row['best_m']}/{c_row['best_step']} | {d_row['best_m']}/{d_row['best_step']} | "
            f"{_fmt(c_row['H_t_binary_fraction'], 3)} | {_fmt(d_row['H_t_binary_fraction'], 3)} | "
            f"{_fmt(c_row['tau_displacement'], 2)} | {_fmt(d_row['tau_displacement'], 2)} |"
        )
    dc = payload.get("paired_D_minus_C_aggregate") or {}
    dc_mse = dc.get("test_mse") or {}
    dc_ssim = dc.get("test_ssim") or {}
    lines += [
        "",
        "## Mean ± std",
        "",
        f"- **C (Udith):** MSE {_fmt(c['test_mse']['mean'])} ± {_fmt(c['test_mse']['std'])}, "
        f"SSIM {_fmt(c['test_ssim']['mean'], 4)} ± {_fmt(c['test_ssim']['std'], 4)} "
        f"(n={c['test_mse']['n']}).",
        f"- **D (Udith):** MSE {_fmt(d['test_mse']['mean'])} ± {_fmt(d['test_mse']['std'])}, "
        f"SSIM {_fmt(d['test_ssim']['mean'], 4)} ± {_fmt(d['test_ssim']['std'], 4)} "
        f"(n={d['test_mse']['n']}).",
        f"- **D − C (paired):** MSE {_fmt(dc_mse.get('mean'))} ± {_fmt(dc_mse.get('std'))}, "
        f"SSIM {_fmt(dc_ssim.get('mean'), 4)} ± {_fmt(dc_ssim.get('std'), 4)} "
        f"(n={dc_mse.get('n', 0)}).",
        f"- **C (current 8,500):** MSE {CURRENT_C['mse_mean']} ± {CURRENT_C['mse_std']}, "
        f"SSIM {CURRENT_C['ssim_mean']} ± {CURRENT_C['ssim_std']}.",
        f"- **D (current 8,500):** MSE {CURRENT_D['mse_mean']} ± {CURRENT_D['mse_std']}, "
        f"SSIM {CURRENT_D['ssim_mean']} ± {CURRENT_D['ssim_std']}.",
        "",
        "## Pattern diagnostics (best checkpoint)",
        "",
        "Physical displacements only (\(\\|\tau-\\tau_0\\|_2\), \(\\|H_t-H_{t,0}\\|_2\)). "
        "Raw Fourier-parameter norms are not compared to spatial-parameter norms.",
        "",
        "| seed | C binary frac | D binary frac | C ‖τ−τ₀‖ | D ‖τ−τ₀‖ | C ‖H−H₀‖ | D ‖H−H₀‖ |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for seed in SEEDS:
        if seed not in by_c or seed not in by_d:
            continue
        cr, dr = by_c[seed], by_d[seed]
        lines.append(
            f"| {seed} | {_fmt(cr['H_t_binary_fraction'], 3)} | {_fmt(dr['H_t_binary_fraction'], 3)} | "
            f"{_fmt(cr['tau_displacement'], 2)} | {_fmt(dr['tau_displacement'], 2)} | "
            f"{_fmt(cr['Ht_displacement'], 2)} | {_fmt(dr['Ht_displacement'], 2)} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        f"`{verdict}`",
        "",
        conclusion,
        "",
        "## Figures",
        "",
        "- `figures/cd_panel_udith_schedule.png`",
        "- `figures/val_mse_curves.png`",
        "- `figures/binary_fraction_curves.png`",
        "- `figures/current_vs_udith_mse.png`",
        "",
        "Reproduce:",
        "",
        "```bash",
        "PY=path/to/python python scripts/figure10_ablation_patchmnist_udith_schedule/train.py "
        "--protocol shared_warmup --device cuda:0 --allow-gpu0 --skip-gpu-check "
        "--seeds 42 43 44 --variants C D",
        "```",
        "",
    ]
    path.write_text("\n".join(lines))


def write_all(runs: Path | None = None) -> dict:
    runs = Path(runs) if runs is not None else RUNS
    exp = runs.parent if runs.name == "runs" else EXP
    figures = exp / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    payload = collect(runs)
    (exp / "aggregate_multiseed.json").write_text(json.dumps(payload, indent=2))
    write_comparison_csv(payload, exp / "comparison_current_vs_udith_schedule.csv")
    if payload["verdict"] != "incomplete":
        try:
            render_cd_panel(runs, figures / "cd_panel_udith_schedule.png")
            render_curves(runs, figures)
            render_schedule_comparison(payload, figures / "current_vs_udith_mse.png")
        except FileNotFoundError as exc:
            print(f"figure render skipped: {exc}", flush=True)
    write_report(payload, exp / "UDITH_SCHEDULE_REPORT.md")
    print(f"verdict={payload['verdict']}", flush=True)
    print(f"wrote {exp / 'aggregate_multiseed.json'}", flush=True)
    print(f"wrote {exp / 'UDITH_SCHEDULE_REPORT.md'}", flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS))
    args = ap.parse_args()
    write_all(Path(args.runs))


if __name__ == "__main__":
    main()
