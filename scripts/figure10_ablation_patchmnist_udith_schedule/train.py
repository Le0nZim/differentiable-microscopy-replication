#!/usr/bin/env python3
"""Schedule-only C vs D experiment: Udith legacy 121,500-step accumulating-m.

Reuses ``apply_variant`` + ``run_one`` from the audited AM-3 / Fig-10 path.
The only causal change versus ``experiments/figure10_ablation_patchmnist`` is
the training schedule (and the paired τ₀ initialization that makes C/D start
from the same physical illumination).

Default device is cuda:1. GPU 0 is allowed only with ``--allow-gpu0``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
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
run_one = _am3.run_one

from training.dataloaders import build_dataloader  # noqa: E402
from training.paired_pattern_init import (  # noqa: E402
    apply_shared_tau0,
    generate_shared_tau0,
    paired_initialization_audit,
    tensor_sha256,
)
from training.shared_warmup_checkpoint import (  # noqa: E402
    compare_frozen_interval,
    read_step_log,
)
from models.microscope import DifferentiableMicroscope  # noqa: E402
from training.udith_legacy_schedule import (  # noqa: E402
    EXPECTED_GLOBAL_BOUNDS,
    phase_global_bounds,
    schedule_provenance,
    udith_legacy_phases,
    udith_legacy_phases_smoke,
)
from utils.experiment_config import load_experiment_config  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402

EXP = ROOT / "experiments/figure10_ablation_patchmnist_udith_schedule"
CFG = ROOT / "configs/figure10_ablation_patchmnist_udith_schedule/ablation.yaml"

EXPERIMENT_LABEL = (
    "Udith legacy schedule under the current controlled PatchMNIST optimizer/data recipe"
)


def _refuse_gpu0(device: str, *, allow: bool = False) -> None:
    if allow:
        return
    if device.strip().lower() in {"cuda:0", "cuda0", "0"}:
        raise SystemExit(
            "Refusing GPU 0 unless --allow-gpu0 is set (reserved for an explicit seed-43 split)."
        )


def gpu1_is_available() -> tuple[bool, str]:
    """True if GPU 1 exists and has no compute process we would interrupt."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi", "-i", "1",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            check=False, capture_output=True, text=True,
        )
    except FileNotFoundError:
        return False, "nvidia-smi not found"
    if proc.returncode != 0:
        return False, proc.stderr.strip() or "nvidia-smi -i 1 failed"
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    # Empty output, or the placeholder "[N/A]" row, means no compute apps.
    busy = [ln for ln in lines if ln and "N/A" not in ln]
    if busy:
        return False, "GPU 1 has compute apps: " + "; ".join(busy)
    return True, "GPU 1 idle (no compute apps)"


def _materialize(letter: str, seed: int, device: str) -> dict:
    base = load_experiment_config(CFG)
    base["experiment"]["device"] = device
    cfg = apply_variant(base, letter)
    cfg["experiment"]["seed"] = seed
    cfg["dataset"]["seed"] = seed
    cfg["pattern_generator"]["seed"] = seed
    cfg["experiment"]["run_id"] = f"fig10pm_udith_{letter}_seed{seed}"
    return cfg


def audit_seed(seed: int, device: str, out_root: Path) -> dict:
    """Build C and D from the same seed, apply shared τ₀, hash the first batch.

    Construction order matches ``run_one``: set_seed → model → apply τ₀ → dataloaders.
    """
    cfg_c = _materialize("C", seed, device)
    cfg_d = _materialize("D", seed, device)
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    image_size = int(cfg_c["dataset"]["image_size"])
    tau0 = generate_shared_tau0(
        int(cfg_c["pattern_generator"]["num_patterns"]), image_size, image_size, seed
    )

    set_seed(seed)
    model_c = DifferentiableMicroscope.from_run_config(cfg_c).to(torch_device)
    apply_shared_tau0(model_c, tau0)
    batch_c = next(iter(build_dataloader(cfg_c, "train")))

    set_seed(seed)
    model_d = DifferentiableMicroscope.from_run_config(cfg_d).to(torch_device)
    apply_shared_tau0(model_d, tau0)
    batch_d = next(iter(build_dataloader(cfg_d, "train")))

    audit = paired_initialization_audit(
        model_c, model_d, first_batch_c=batch_c, first_batch_d=batch_d
    )
    audit["seed"] = seed
    audit["tau0_sha256"] = tensor_sha256(tau0)
    audit["device"] = str(torch_device)
    path = out_root / f"seed{seed}_paired_initialization_audit.json"
    path.write_text(json.dumps(audit, indent=2))
    print(
        f"paired init seed={seed} pass={audit['pass']} "
        f"max|Δτ|={audit['max_abs_tau_difference']:.3e} "
        f"max|ΔH|={audit['max_abs_Ht_difference']:.3e}",
        flush=True,
    )
    if not audit["pass"]:
        raise RuntimeError(
            f"Paired initialization audit failed for seed {seed}: {audit['problems']}"
        )
    return audit


def _copy_sidecar(run_dir: Path, audit: dict, provenance: dict) -> None:
    (run_dir / "paired_initialization_audit.json").write_text(json.dumps(audit, indent=2))
    (run_dir / "schedule_provenance.json").write_text(json.dumps(provenance, indent=2))


def _finite(value: float) -> bool:
    return value is not None and math.isfinite(float(value))


def seed42_gates(run_dir_c: Path, run_dir_d: Path) -> dict:
    """Post-seed-42 checks required before launching 43/44."""
    problems: list[str] = []
    bounds = {b["name"]: b for b in EXPECTED_GLOBAL_BOUNDS}
    for letter, run_dir in [("C", run_dir_c), ("D", run_dir_d)]:
        log_path = run_dir / "metrics" / "step_log.csv"
        rows = list(csv.DictReader(log_path.open()))
        if not rows:
            problems.append(f"{letter}: empty step log")
            continue
        ms = [float(r["m"]) for r in rows]
        for prev, cur in zip(ms, ms[1:]):
            if cur < prev - 1e-9:
                problems.append(f"{letter}: logged m reset ({prev} -> {cur})")
                break
        for key in ("loss", "train_mse", "val_mse", "val_ssim", "H_t_mean", "H_t_std"):
            if any(not _finite(float(r[key])) for r in rows):
                problems.append(f"{letter}: non-finite {key}")
        warmup = [r for r in rows if r["phase"] == "inverse_warmup_m1"]
        if not warmup:
            problems.append(f"{letter}: no warmup log rows")
        else:
            for r in warmup:
                if abs(float(r["illum_delta"])) > 1e-12:
                    problems.append(f"{letter}: warmup illum_delta={r['illum_delta']} != 0")
                    break
                if "tau_displacement" in r and abs(float(r["tau_displacement"])) > 1e-12:
                    problems.append(f"{letter}: warmup tau_displacement={r['tau_displacement']} != 0")
                    break
        after = [r for r in rows if int(r["step"]) > 60_750]
        if after and "tau_displacement" in after[0]:
            if max(float(r["tau_displacement"]) for r in after) <= 0.0:
                problems.append(f"{letter}: illumination did not change after step 60750")
        # Phase-end rows must exist at each documented boundary.
        by_step = {int(r["step"]): r for r in rows}
        for name, b in bounds.items():
            row = by_step.get(int(b["end"]))
            if row is None:
                problems.append(f"{letter}: missing phase-end log at step {b['end']} ({name})")
            elif row["phase"] != name:
                problems.append(
                    f"{letter}: step {b['end']} phase={row['phase']} expected {name}"
                )
            elif abs(float(row["m"]) - float(b["m"])) > 1e-9:
                problems.append(
                    f"{letter}: step {b['end']} m={row['m']} expected {b['m']}"
                )
        diag = json.loads((run_dir / "metrics" / "diagnostics.json").read_text())
        if diag.get("saw_nonfinite"):
            problems.append(f"{letter}: diagnostics.saw_nonfinite")
        summ = json.loads((run_dir / "metrics" / "run_summary.json").read_text())
        if not _finite(summ["test_mse"]) or not _finite(summ["test_ssim"]):
            problems.append(f"{letter}: non-finite test metrics")
    warmup_cmp = compare_frozen_interval(
        read_step_log(run_dir_c / "metrics" / "step_log.csv"),
        read_step_log(run_dir_d / "metrics" / "step_log.csv"),
    )
    payload = {
        "pass": not problems and warmup_cmp["pass"],
        "problems": problems,
        "frozen_interval": warmup_cmp,
        "phase_boundary_evals_eligible_for_global_best": True,
        "experiment_label": EXPERIMENT_LABEL,
    }
    if warmup_cmp["materially_diverged"]:
        payload["problems"] = problems + [
            "C vs D frozen-interval losses/val metrics diverged materially; "
            "do not launch seeds 43/44; branch both arms from one shared warmup "
            "checkpoint including inverse Adam state"
        ]
        payload["pass"] = False
    return payload


def train_one(
    letter: str,
    seed: int,
    device: str,
    out_root: Path,
    phases: list[dict],
    *,
    log_every: int,
    force: bool,
    audit: dict,
    provenance: dict,
    run_dir: Path | None = None,
    warmup_checkpoint_out: Path | None = None,
    resume_from_warmup: Path | None = None,
) -> dict:
    run_dir = Path(run_dir) if run_dir is not None else out_root / f"{letter}_seed{seed}"
    summary_path = run_dir / "metrics" / "run_summary.json"
    ckpt_ready = warmup_checkpoint_out is None or Path(warmup_checkpoint_out).exists()
    if summary_path.exists() and ckpt_ready and not force:
        print(f"skip existing {run_dir}", flush=True)
        _copy_sidecar(run_dir, audit, provenance)
        return json.loads(summary_path.read_text())
    cfg = _materialize(letter, seed, device)
    image_size = int(cfg["dataset"]["image_size"])
    tau0 = generate_shared_tau0(
        int(cfg["pattern_generator"]["num_patterns"]), image_size, image_size, seed
    )
    print(f"\n=== Udith-schedule {letter} seed={seed} device={device} "
          f"steps={sum(p['steps'] for p in phases)} "
          f"resume={resume_from_warmup is not None} "
          f"save_warmup={warmup_checkpoint_out is not None} ===", flush=True)
    summary = run_one(
        cfg,
        run_dir,
        letter=letter,
        phases=phases,
        seed=seed,
        log_every=log_every,
        log_phase_boundaries=True,
        extra_physical_metrics=True,
        shared_tau0=tau0,
        save_config_yaml=True,
        schedule_provenance=provenance,
        warmup_checkpoint_out=warmup_checkpoint_out,
        resume_from_warmup=resume_from_warmup,
    )
    _copy_sidecar(run_dir, audit, provenance)
    return summary


def train_shared_warmup_seed(
    seed: int,
    device: str,
    out_root: Path,
    phases: list[dict],
    *,
    log_every: int,
    force: bool,
    audit: dict,
    provenance: dict,
) -> dict:
    """One frozen warmup (spatial/D arm), then fork C and D with shared inverse Adam."""
    warmup_dir = out_root / f"warmup_seed{seed}"
    ckpt = warmup_dir / "warmup_checkpoint.pt"
    warmup_phases = [p for p in phases if p["freeze_illum"]]
    if not warmup_phases:
        raise RuntimeError("shared-warmup protocol requires a freeze_illum phase")
    provenance = dict(provenance)
    provenance["protocol"] = "shared_warmup"
    provenance["warmup_arm"] = "D"
    provenance["warmup_arm_reason"] = (
        "Spatial (corrected D) frozen warmup so the inverse is trained against exact τ₀; "
        "C then sets W=FFT2(τ₀) and both arms load the same inverse weights and inverse Adam."
    )
    provenance["experiment_label"] = EXPERIMENT_LABEL
    if force or not ckpt.exists():
        train_one(
            "D", seed, device, out_root, warmup_phases,
            log_every=log_every, force=force, audit=audit, provenance=provenance,
            run_dir=warmup_dir, warmup_checkpoint_out=ckpt,
        )
    else:
        print(f"reuse warmup checkpoint {ckpt}", flush=True)
    summaries = {}
    for letter in ("C", "D"):
        summaries[letter] = train_one(
            letter, seed, device, out_root, phases,
            log_every=log_every, force=force, audit=audit, provenance=provenance,
            resume_from_warmup=ckpt,
        )
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser(description="Udith-schedule C vs D on PatchMNIST")
    ap.add_argument("--variants", nargs="+", default=["C", "D"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="non-scientific short pass through every phase")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--skip-gpu-check", action="store_true")
    ap.add_argument("--allow-gpu0", action="store_true",
                    help="Permit --device cuda:0.")
    ap.add_argument(
        "--protocol",
        choices=["independent", "shared_warmup"],
        default="independent",
        help="independent: C and D each run their own warmup. shared_warmup: one "
        "frozen warmup (D arm), then both C and D load inverse+Adam and train the "
        "joint/hardening half.",
    )
    ap.add_argument(
        "--allow-divergent-warmup-seeds",
        action="store_true",
        help="Override the block on independent-warmup seeds 43/44.",
    )
    args = ap.parse_args()

    _refuse_gpu0(args.device, allow=args.allow_gpu0)
    if args.smoke:
        out_root = Path(args.out) if args.out else EXP / "smoke"
        phases = udith_legacy_phases_smoke()
        seeds = args.seeds[:1]
        log_every = 1
    else:
        default_out = (
            EXP / "runs_shared_warmup"
            if args.protocol == "shared_warmup"
            else EXP / "runs"
        )
        out_root = Path(args.out) if args.out else default_out
        phases = udith_legacy_phases()
        seeds = args.seeds
        log_every = args.log_every

    if (
        (not args.smoke)
        and args.protocol == "independent"
        and any(s in (43, 44) for s in seeds)
        and not args.allow_divergent_warmup_seeds
    ):
        raise SystemExit(
            "Independent-warmup C vs D trajectories diverged materially on seed 42. "
            "Seeds 43/44 are blocked. Use --protocol shared_warmup "
            "(or --allow-divergent-warmup-seeds to override)."
        )

    out_root.mkdir(parents=True, exist_ok=True)
    provenance = schedule_provenance()
    provenance["experiment_label"] = EXPERIMENT_LABEL
    provenance["protocol"] = args.protocol
    provenance["phase_boundary_evals_eligible_for_global_best"] = True
    provenance["train_iterator"] = (
        "itertools.cycle(train_loader) caches and repeats the first loader traversal"
    )
    if args.smoke:
        provenance["smoke"] = True
        provenance["smoke_phases"] = phases

    print(
        f"{EXPERIMENT_LABEL}. protocol={args.protocol} variants="
        f"{args.variants} seeds={seeds} device={args.device} "
        f"smoke={args.smoke} cuda={torch.cuda.is_available()}",
        flush=True,
    )
    print("phases: " + ", ".join(f"{p['name']}:{p['steps']}" for p in phases), flush=True)
    print("bounds:", json.dumps(phase_global_bounds(phases)), flush=True)

    if args.aggregate_only:
        spec_agg = importlib.util.spec_from_file_location(
            "udith_agg", Path(__file__).resolve().parent / "aggregate.py"
        )
        agg_mod = importlib.util.module_from_spec(spec_agg)
        spec_agg.loader.exec_module(agg_mod)
        agg_mod.write_all(Path(args.out) if args.out else out_root)
        return

    if (not args.smoke) and args.device.startswith("cuda") and not args.skip_gpu_check:
        ok, msg = gpu1_is_available()
        print(f"GPU 1 check: {msg}", flush=True)
        if args.device.endswith(":1") and not ok:
            raise SystemExit(f"Not launching full training: {msg}")

    for seed in seeds:
        audit = audit_seed(seed, args.device, out_root)
        if args.protocol == "shared_warmup":
            train_shared_warmup_seed(
                seed, args.device, out_root, phases,
                log_every=log_every, force=args.force,
                audit=audit, provenance=provenance,
            )
        else:
            summaries = {}
            for letter in args.variants:
                summaries[letter] = train_one(
                    letter, seed, args.device, out_root, phases,
                    log_every=log_every, force=args.force,
                    audit=audit, provenance=provenance,
                )
        if (
            (not args.smoke)
            and args.protocol == "independent"
            and seed == 42
            and set(args.variants) >= {"C", "D"}
        ):
            gates = seed42_gates(out_root / "C_seed42", out_root / "D_seed42")
            (out_root / "seed42_gates.json").write_text(json.dumps(gates, indent=2))
            print("seed-42 gates:", json.dumps(gates), flush=True)
            if not gates["pass"]:
                raise SystemExit(f"Seed 42 gates failed: {gates['problems']}")

    if not args.smoke:
        spec_agg = importlib.util.spec_from_file_location(
            "udith_agg", Path(__file__).resolve().parent / "aggregate.py"
        )
        agg_mod = importlib.util.module_from_spec(spec_agg)
        spec_agg.loader.exec_module(agg_mod)
        n_done = sum(
            1
            for seed in seeds
            for letter in args.variants
            if (out_root / f"{letter}_seed{seed}" / "metrics" / "run_summary.json").exists()
        )
        if n_done == len(seeds) * len(args.variants) and set(args.variants) >= {"C", "D"}:
            agg_mod.write_all(out_root)


if __name__ == "__main__":
    main()
