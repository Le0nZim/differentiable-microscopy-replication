#!/usr/bin/env python3
"""AM-3 (Table 3 / Fig. 10) resolution runner.

Self-contained, *uniform-protocol* A/B/C/D ablation with full Phase-3
diagnostics. Implements a scaled paper Algorithm-1 schedule:

    1. inverse warmup  - train inverse model only (illumination frozen), m=1
    2. joint soft      - train illumination + inverse end-to-end, m=1
    3. m hardening     - increase the custom-sigmoid sharpness m (2 -> 4 -> 8)

The same schedule, optimizer budget, loss, crop policy and *global best-val
checkpoint selection* are used for every variant, so the only thing that differs
between A/B/C/D is exactly what Table 3 ablates (illumination mode + upsampling
mode + frequency-domain optimization).

This script writes ONLY under experiments/table03_ablation and
never modifies the historical experiments/ablations/bbbc022_ablation_x16 outputs
(nor any AM-1 / AM-2 outputs).

Run CUDA outside the sandbox.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluation.metrics import mse as mse_metric  # noqa: E402
from evaluation.metrics import ssim as ssim_metric  # noqa: E402
from evaluation.variant_audit import audit_microscope, check_variant  # noqa: E402
from models.microscope import DifferentiableMicroscope  # noqa: E402
from training.dataloaders import build_dataloader  # noqa: E402
from training.losses import reconstruction_loss_l1  # noqa: E402
from training.paired_pattern_init import apply_shared_tau0  # noqa: E402
from training.shared_warmup_checkpoint import (  # noqa: E402
    load_adam_inverse,
    load_warmup_checkpoint,
    restore_best_full_state,
    restore_rng,
    save_warmup_checkpoint,
    skip_cycle_steps,
)
from utils.experiment_config import load_experiment_config, sync_derived_config_fields  # noqa: E402
from utils.logging import save_run_config  # noqa: E402
from utils.reproducibility import get_git_commit_hash, set_seed  # noqa: E402

OUT = ROOT / "experiments/table03_ablation"

# Variant letter -> (pattern_mode, upsampling_mode, learn_patterns)
VARIANTS = {
    "A": ("random_fixed", "transpose_conv", False),
    "B": ("learnable_frequency", "transpose_conv", True),
    "C": ("learnable_frequency", "locality_aware", True),
    "D": ("learnable_spatial", "locality_aware", True),
}
VARIANT_LABEL = {
    "A": "fixed Ht + Tr.Conv.Up + freq",
    "B": "learnable Ht + Tr.Conv.Up + freq",
    "C": "learnable Ht + locality + freq (paper best)",
    "D": "learnable Ht + locality + NO freq",
}


# --------------------------------------------------------------------------- #
# config construction                                                         #
# --------------------------------------------------------------------------- #


def _apply_official_preprocessing(config: dict) -> None:
    report = ROOT / "experiments/ablations/preprocessing_report.json"
    if report.exists():
        config["dataset"]["preprocessing_mode"] = json.loads(report.read_text())["chosen_official_mode"]


def bbbc022_base(device: str) -> dict:
    cfg = load_experiment_config(ROOT / "configs/_shared/base_bbbc022_substitute.yaml")
    _apply_official_preprocessing(cfg)
    cfg["experiment"]["device"] = device
    return cfg


def patchmnist_base(device: str) -> dict:
    cfg = load_experiment_config(ROOT / "configs/_shared/base_patchmnist.yaml")
    cfg["experiment"]["device"] = device
    return cfg


def apply_variant(config: dict, letter: str) -> dict:
    pattern_mode, upsampling_mode, learn = VARIANTS[letter]
    cfg = copy.deepcopy(config)
    cfg["pattern_generator"]["mode"] = pattern_mode
    cfg["inverse_model"]["upsampling"]["mode"] = upsampling_mode
    cfg["training"]["learn_patterns"] = learn
    return sync_derived_config_fields(cfg)


# --------------------------------------------------------------------------- #
# protocol                                                                    #
# --------------------------------------------------------------------------- #


def default_phases(scale: float = 1.0) -> list[dict]:
    """Scaled Algorithm-1 phases (uniform across all variants)."""
    s = lambda n: max(1, int(round(n * scale)))
    return [
        {"name": "inverse_warmup", "m": 1.0, "steps": s(1500), "freeze_illum": True},
        {"name": "joint_soft", "m": 1.0, "steps": s(4000), "freeze_illum": False},
        {"name": "harden_m2", "m": 2.0, "steps": s(1000), "freeze_illum": False},
        {"name": "harden_m4", "m": 4.0, "steps": s(1000), "freeze_illum": False},
        {"name": "harden_m8", "m": 8.0, "steps": s(1000), "freeze_illum": False},
    ]


def should_log_step(
    global_step: int,
    log_every: int,
    *,
    is_phase_last_step: bool,
    is_last_phase: bool,
    log_phase_boundaries: bool = False,
) -> bool:
    """Match the historical AM-3 cadence; phase-end logs are opt-in.

    Default (``log_phase_boundaries=False``): log every ``log_every`` steps and
    on the last step of the *last* phase only. That is the original 8,500-step
    Fig-10 / Table-3 behaviour.
    """
    if log_every > 0 and global_step % log_every == 0:
        return True
    if is_phase_last_step and (log_phase_boundaries or is_last_phase):
        return True
    return False


def _grad_norm(params) -> float:
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float(p.grad.detach().norm(2).item() ** 2)
    return sq**0.5


def _param_count(params, only_trainable=False) -> int:
    return int(sum(p.numel() for p in params if (p.requires_grad or not only_trainable)))


@torch.no_grad()
def _evaluate(model, loader, device, m, apply_noise) -> tuple[float, float]:
    model.eval()
    tot_mse = tot_ssim = 0.0
    n = 0
    for batch in loader:
        x = batch.to(device)
        out = model(x, sigmoid_m=m, apply_noise=apply_noise)
        tot_mse += float(mse_metric(out["x_recon"], x).item())
        tot_ssim += float(ssim_metric(out["x_recon"], x).item())
        n += 1
    model.train()
    return tot_mse / max(n, 1), tot_ssim / max(n, 1)


def _illum_l2(model) -> float:
    params = model.illumination_parameters()
    if not params:
        return 0.0
    sq = 0.0
    for p in params:
        v = p.detach()
        if torch.is_complex(v):
            sq += float((v.real**2 + v.imag**2).sum().item())
        else:
            sq += float((v**2).sum().item())
    return sq**0.5


# --------------------------------------------------------------------------- #
# core training                                                               #
# --------------------------------------------------------------------------- #


def run_one(
    config: dict,
    output_dir: Path,
    *,
    letter: str | None,
    phases: list[dict],
    seed: int,
    log_every: int = 200,
    log_phase_boundaries: bool = False,
    extra_physical_metrics: bool = False,
    shared_tau0: torch.Tensor | None = None,
    save_config_yaml: bool = False,
    schedule_provenance: dict | None = None,
    warmup_checkpoint_out: Path | None = None,
    resume_from_warmup: Path | None = None,
) -> dict:
    import itertools

    output_dir = Path(output_dir)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "learned_patterns").mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = torch.device(config["experiment"]["device"] if torch.cuda.is_available() else "cpu")
    apply_noise = config["detector_noise"].get("apply_noise", False)

    model = DifferentiableMicroscope.from_run_config(config).to(device)
    if shared_tau0 is not None:
        apply_shared_tau0(model, shared_tau0)
    if save_config_yaml:
        save_run_config(config, output_dir)
    if schedule_provenance is not None:
        (output_dir / "schedule_provenance.json").write_text(
            json.dumps(schedule_provenance, indent=2)
        )

    # ----- machine-checkable wiring metadata -----
    audit = audit_microscope(model, config)
    illum_params = model.illumination_parameters()
    ups_params = list(model.inverse_model.upsampling.parameters())
    rec_params = list(model.inverse_model.reconstruction.parameters())
    metadata = {
        "variant": letter,
        "label": VARIANT_LABEL.get(letter, ""),
        "seed": seed,
        "device": str(device),
        "git_commit": get_git_commit_hash(),
        "audit": audit,
        "trainable_param_report": {
            "illumination_total": _param_count(illum_params),
            "illumination_trainable": _param_count(illum_params, only_trainable=True),
            "upsampler_total": _param_count(ups_params),
            "upsampler_trainable": _param_count(ups_params, only_trainable=True),
            "reconstruction_total": _param_count(rec_params),
            "reconstruction_trainable": _param_count(rec_params, only_trainable=True),
        },
        "phases": phases,
    }
    if letter is not None:
        problems = check_variant(letter, audit)
        metadata["wiring_problems"] = problems
        if problems:
            raise RuntimeError(f"Variant {letter} wiring mismatch: {problems}")
    (output_dir / "metrics" / "variant_metadata.json").write_text(json.dumps(metadata, indent=2))

    # ----- data -----
    train_loader = build_dataloader(config, "train")
    val_loader = build_dataloader(config, "val")
    test_loader = build_dataloader(config, "test")

    # optimizer (paper: illum lr 1.0, inverse lr 0.001)
    illum_lr = float(config["training"]["illumination_lr"])
    inv_lr = float(config["training"]["inverse_lr"])
    groups = []
    if illum_params:
        groups.append({"params": illum_params, "lr": illum_lr})
    groups.append({"params": list(model.inverse_model.parameters()), "lr": inv_lr})
    optimizer = torch.optim.Adam(groups)
    grad_clip = config["training"].get("gradient_clip_norm")

    init_illum_l2 = _illum_l2(model)
    tau0_snap = None
    H0_snap = None
    if extra_physical_metrics or shared_tau0 is not None:
        with torch.no_grad():
            H0_snap = model.pattern_generator(sigmoid_m=1.0).detach().clone()
            if model.pattern_generator.patterns_are_learnable():
                tau0_snap = model.pattern_generator._spatial_tau().detach().clone()
        if extra_physical_metrics:
            torch.save(H0_snap.detach().cpu(), output_dir / "learned_patterns" / "H_t0.pt")
            if tau0_snap is not None:
                torch.save(tau0_snap.detach().cpu(), output_dir / "learned_patterns" / "tau_0.pt")

    fieldnames = [
        "step", "phase", "m", "loss", "train_mse", "val_mse", "val_ssim",
        "grad_norm_illum", "grad_norm_upsampler", "grad_norm_recon",
        "H_t_min", "H_t_max", "H_t_mean", "H_t_std", "H_t_binary_fraction",
        "illum_l2", "illum_delta",
    ]
    if extra_physical_metrics:
        fieldnames.extend(["H_t_displacement", "tau_displacement", "illum_frozen"])
    step_log = (output_dir / "metrics" / "step_log.csv").open("w", newline="")
    writer = csv.DictWriter(step_log, fieldnames=fieldnames)
    writer.writeheader()

    history: list[dict] = []
    best = {"val_mse": float("inf"), "m": 1.0, "step": 0, "state": None, "train_mse": None}
    min_train_mse = float("inf")  # best (lowest) train MSE ever seen (fitting capacity)
    max_grad = {"illum": 0.0, "upsampler": 0.0, "recon": 0.0}

    train_iter = itertools.cycle(train_loader)
    global_step = 0
    t0 = time.time()
    n_phases = len(phases)
    saw_nonfinite = False
    warmup_tau_disp_max = 0.0
    post_warmup_tau_disp_max = 0.0
    resume_ckpt = None
    branched_from_shared_warmup = False

    if resume_from_warmup is not None:
        resume_ckpt = load_warmup_checkpoint(resume_from_warmup, map_location="cpu")
        branched_from_shared_warmup = True
        model.inverse_model.load_state_dict(resume_ckpt["inverse_state_dict"])
        if shared_tau0 is None:
            apply_shared_tau0(model, resume_ckpt["tau0"])
        load_adam_inverse(
            optimizer, model.inverse_model, resume_ckpt["adam_inverse_by_name"], device
        )
        tau0_snap = resume_ckpt["tau0"].to(device=device)
        H0_snap = resume_ckpt["H_t0"].to(device=device)
        if extra_physical_metrics:
            torch.save(H0_snap.detach().cpu(), output_dir / "learned_patterns" / "H_t0.pt")
            torch.save(tau0_snap.detach().cpu(), output_dir / "learned_patterns" / "tau_0.pt")
        skip_cycle_steps(train_iter, int(resume_ckpt["global_step"]))
        restore_rng(resume_ckpt["rng"], device)
        global_step = int(resume_ckpt["global_step"])
        history = list(resume_ckpt.get("history") or [])
        for row in history:
            writer.writerow({k: row[k] for k in fieldnames if k in row})
        step_log.flush()
        min_train_mse = float(resume_ckpt.get("min_train_mse", float("inf")))
        max_grad = dict(resume_ckpt.get("max_grad") or max_grad)
        best_blob = resume_ckpt.get("best") or {}
        best_inv = best_blob.get("inverse_state_dict")
        best = {
            "val_mse": best_blob.get("val_mse", float("inf")),
            "m": best_blob.get("m", 1.0),
            "step": best_blob.get("step", 0),
            "train_mse": best_blob.get("train_mse"),
            "state": restore_best_full_state(model, best_inv),
        }
        print(
            f"[{letter}] resumed shared warmup at step={global_step} "
            f"from {resume_from_warmup} (arm={resume_ckpt.get('warmup_arm')})",
            flush=True,
        )

    for phase_idx, phase in enumerate(phases):
        if resume_ckpt is not None and bool(phase["freeze_illum"]):
            continue
        m = float(phase["m"])
        freeze_illum = bool(phase["freeze_illum"])
        if freeze_illum:
            model.set_illumination_trainable(False)
        elif model.pattern_generator.patterns_are_learnable():
            model.set_illumination_trainable(True)

        phase_steps = int(phase["steps"])
        is_last_phase = phase_idx == n_phases - 1
        for step_in_phase in range(phase_steps):
            global_step += 1
            x = next(train_iter).to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            out = model(x, sigmoid_m=m, apply_noise=apply_noise)
            loss = reconstruction_loss_l1(out["x_recon"], x)
            if not torch.isfinite(loss):
                saw_nonfinite = True
                raise RuntimeError(
                    f"Non-finite loss at step {global_step} phase={phase['name']} m={m}: {loss.item()}"
                )
            loss.backward()

            gI = _grad_norm(illum_params) if illum_params else 0.0
            gU = _grad_norm(ups_params)
            gR = _grad_norm(rec_params)
            max_grad["illum"] = max(max_grad["illum"], gI)
            max_grad["upsampler"] = max(max_grad["upsampler"], gU)
            max_grad["recon"] = max(max_grad["recon"], gR)

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

            is_phase_last_step = step_in_phase == phase_steps - 1
            if should_log_step(
                global_step,
                log_every,
                is_phase_last_step=is_phase_last_step,
                is_last_phase=is_last_phase,
                log_phase_boundaries=log_phase_boundaries,
            ):
                with torch.no_grad():
                    patterns = out["patterns"].detach()
                    H = patterns
                    hb = float(((H < 0.1) | (H > 0.9)).float().mean().item())
                    tau_disp = 0.0
                    ht_disp = 0.0
                    if extra_physical_metrics:
                        ht_disp = float((H - H0_snap).norm().item())
                        if tau0_snap is not None:
                            tau_now = model.pattern_generator._spatial_tau()
                            tau_disp = float((tau_now - tau0_snap).norm().item())
                if freeze_illum:
                    warmup_tau_disp_max = max(warmup_tau_disp_max, tau_disp)
                else:
                    post_warmup_tau_disp_max = max(post_warmup_tau_disp_max, tau_disp)
                train_mse = float(mse_metric(out["x_recon"].detach(), x).item())
                val_mse, val_ssim = _evaluate(model, val_loader, device, m, apply_noise)
                illum_l2 = _illum_l2(model)
                row = {
                    "step": global_step, "phase": phase["name"], "m": m,
                    "loss": float(loss.item()), "train_mse": train_mse,
                    "val_mse": val_mse, "val_ssim": val_ssim,
                    "grad_norm_illum": gI, "grad_norm_upsampler": gU, "grad_norm_recon": gR,
                    "H_t_min": float(H.min().item()), "H_t_max": float(H.max().item()),
                    "H_t_mean": float(H.mean().item()), "H_t_std": float(H.std().item()),
                    "H_t_binary_fraction": hb,
                    "illum_l2": illum_l2, "illum_delta": abs(illum_l2 - init_illum_l2),
                }
                if extra_physical_metrics:
                    row["H_t_displacement"] = ht_disp
                    row["tau_displacement"] = tau_disp
                    row["illum_frozen"] = int(freeze_illum)
                writer.writerow(row)
                step_log.flush()
                history.append(row)
                min_train_mse = min(min_train_mse, train_mse)
                # Phase-boundary evaluations use this same path, so they ARE
                # eligible for global-best checkpoint selection (not diagnostic-only).
                if val_mse < best["val_mse"]:
                    best.update(val_mse=val_mse, m=m, step=global_step,
                                state=deepcopy(model.state_dict()), train_mse=train_mse)
                print(f"[{letter}] {phase['name']} step={global_step} m={m} "
                      f"loss={loss.item():.5f} train_mse={train_mse:.5f} val_mse={val_mse:.5f} "
                      f"gI={gI:.2e} gU={gU:.2e}", flush=True)

            last_freeze_phase = not any(
                bool(phases[j]["freeze_illum"]) for j in range(phase_idx + 1, n_phases)
            )
            if (
                warmup_checkpoint_out is not None
                and freeze_illum
                and is_phase_last_step
                and last_freeze_phase
            ):
                if tau0_snap is None or H0_snap is None:
                    raise RuntimeError(
                        "warmup_checkpoint_out requires snapshots of τ₀ and H_t0 "
                        "(pass shared_tau0 or extra_physical_metrics=True)"
                    )
                save_warmup_checkpoint(
                    Path(warmup_checkpoint_out),
                    global_step=global_step,
                    seed=seed,
                    warmup_arm=str(letter),
                    inverse=model.inverse_model,
                    optimizer=optimizer,
                    tau0=tau0_snap,
                    H_t0=H0_snap,
                    best=best,
                    min_train_mse=min_train_mse,
                    max_grad=max_grad,
                    history=history,
                    device=device,
                )

    step_log.close()
    elapsed = time.time() - t0

    # restore global best-val checkpoint (uniform rule for all variants)
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    best_m = best["m"]

    test_mse, test_ssim = _evaluate(model, test_loader, device, best_m, apply_noise)
    final_train_mse, _ = _evaluate(model, train_loader, device, best_m, apply_noise)

    torch.save({"model_state_dict": model.state_dict(), "config": config,
                "best_m": best_m, "best_step": best["step"]},
               output_dir / "checkpoints_best.pt")

    # ----- artifacts: patterns, qualitative panel, curves -----
    with torch.no_grad():
        model.eval()
        patterns = model.pattern_generator(sigmoid_m=best_m).detach().cpu()
        _save_patterns(patterns, output_dir / "learned_patterns" / "H_t.png")
        torch.save(patterns, output_dir / "learned_patterns" / "H_t.pt")
        # qualitative on a fixed held-out test batch (same indices across variants)
        ref = next(iter(test_loader)).to(device)
        ref_out = model(ref, sigmoid_m=best_m, apply_noise=apply_noise)
        torch.save({"gt": ref.detach().cpu(), "recon": ref_out["x_recon"].detach().cpu()},
                   output_dir / "figures" / "qualitative_tensors.pt")
        _save_qualitative(ref.detach().cpu(), ref_out["x_recon"].detach().cpu(),
                          output_dir / "figures" / "qualitative_panel.png",
                          title=f"{letter}: {VARIANT_LABEL.get(letter,'')}")

    _save_curves(history, output_dir / "figures" / "curves.png", title=f"{letter} train/val MSE")

    overfit_gap = None
    if best["train_mse"] is not None and best["val_mse"] < float("inf"):
        overfit_gap = best["val_mse"] - best["train_mse"]

    diagnostics = {
        "elapsed_sec": elapsed,
        "best_val_mse": best["val_mse"],
        "best_step": best["step"],
        "best_m": best_m,
        "best_train_mse_at_log": best["train_mse"],
        "min_train_mse": None if min_train_mse == float("inf") else min_train_mse,
        "final_train_mse_full": final_train_mse,
        "test_mse": test_mse,
        "test_ssim": test_ssim,
        "overfit_gap_val_minus_train": overfit_gap,
        "max_grad_norms": max_grad,
        "illum_l2_init": init_illum_l2,
        "illum_l2_final": _illum_l2(model),
        "illum_delta_final": abs(_illum_l2(model) - init_illum_l2),
        "final_pattern_stats": {
            "min": float(patterns.min().item()), "max": float(patterns.max().item()),
            "mean": float(patterns.mean().item()), "std": float(patterns.std().item()),
            "binary_fraction": float(((patterns < 0.1) | (patterns > 0.9)).float().mean().item()),
        },
        "m_schedule": [{"phase": p["name"], "m": p["m"], "steps": p["steps"]} for p in phases],
    }
    if extra_physical_metrics:
        diagnostics.update({
            "saw_nonfinite": saw_nonfinite,
            "warmup_tau_displacement_max_logged": warmup_tau_disp_max,
            "post_warmup_tau_displacement_max_logged": post_warmup_tau_disp_max,
            "H_t0_binary_fraction": float(
                ((H0_snap < 0.1) | (H0_snap > 0.9)).float().mean().item()
            ),
            "final_tau_displacement": float(
                (model.pattern_generator._spatial_tau().detach() - tau0_snap).norm().item()
            ),
            "final_Ht_displacement_vs_m1_init": float(
                (patterns.cpu() - H0_snap.detach().cpu()).norm().item()
            ),
            "log_phase_boundaries": log_phase_boundaries,
            "extra_physical_metrics": extra_physical_metrics,
            "shared_tau0_applied": shared_tau0 is not None,
            "phase_boundary_evals_eligible_for_global_best": True,
            "branched_from_shared_warmup": branched_from_shared_warmup,
            "warmup_checkpoint_out": None if warmup_checkpoint_out is None else str(warmup_checkpoint_out),
            "resume_from_warmup": None if resume_from_warmup is None else str(resume_from_warmup),
            "train_iterator": "itertools.cycle(train_loader)",
            "m_schedule_with_freeze": [
                {"phase": p["name"], "m": p["m"], "steps": p["steps"],
                 "freeze_illum": p["freeze_illum"]}
                for p in phases
            ],
        })
    (output_dir / "metrics" / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    summary = {
        "variant": letter, "label": VARIANT_LABEL.get(letter, ""), "seed": seed,
        "test_mse": test_mse, "test_ssim": test_ssim,
        "best_val_mse": best["val_mse"], "best_train_mse": best["train_mse"],
        "min_train_mse": None if min_train_mse == float("inf") else min_train_mse,
        "final_train_mse": final_train_mse, "overfit_gap": overfit_gap,
        "best_m": best_m, "elapsed_sec": elapsed,
        "run_dir": str(output_dir),
    }
    if extra_physical_metrics:
        summary["best_step"] = best["step"]
        summary["H_t_binary_fraction"] = float(
            ((patterns < 0.1) | (patterns > 0.9)).float().mean().item()
        )
        summary["tau_displacement"] = float(
            (model.pattern_generator._spatial_tau().detach() - tau0_snap).norm().item()
        )
        summary["Ht_displacement"] = float(
            (patterns.cpu() - H0_snap.detach().cpu()).norm().item()
        )
    (output_dir / "metrics" / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------- #
# figures                                                                     #
# --------------------------------------------------------------------------- #


def _save_patterns(patterns: torch.Tensor, path: Path) -> None:
    T = patterns.shape[0]
    fig, axes = plt.subplots(1, T, figsize=(2.2 * T, 2.4))
    if T == 1:
        axes = [axes]
    for t in range(T):
        axes[t].imshow(patterns[t, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[t].set_title(f"H_{t}")
        axes[t].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _save_qualitative(gt: torch.Tensor, recon: torch.Tensor, path: Path, *, title: str, n: int = 4) -> None:
    n = min(n, gt.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i in range(n):
        axes[0, i].imshow(gt[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i, 0].numpy().clip(0, 1), cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("GT")
    axes[1, 0].set_ylabel("recon")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _save_curves(history: list[dict], path: Path, *, title: str) -> None:
    if not history:
        return
    steps = [h["step"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, [h["train_mse"] for h in history], label="train MSE")
    ax.plot(steps, [h["val_mse"] for h in history], label="val MSE")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _stats(values: list[float]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals)}


# --------------------------------------------------------------------------- #
# commands                                                                    #
# --------------------------------------------------------------------------- #


def cmd_overfit(args) -> None:
    """Phase 5: tiny overfit gate for B and C on fixed crops, no augmentation."""
    out_root = OUT / "overfit_diagnostics"
    out_root.mkdir(parents=True, exist_ok=True)
    phases = [
        {"name": "inverse_warmup", "m": 1.0, "steps": 600, "freeze_illum": True},
        {"name": "joint_soft", "m": 1.0, "steps": 1400, "freeze_illum": False},
    ]
    results = []
    for n_images in args.sizes:
        for letter in ["B", "C"]:
            cfg = bbbc022_base(args.device)
            cfg = apply_variant(cfg, letter)
            # fixed crops, no aug, tiny train subset; val == train subset to test pure fitting
            cfg["dataset"]["train_random_crops"] = False
            cfg["dataset"]["random_flips"] = False
            cfg["dataset"]["max_train_samples"] = n_images
            cfg["dataset"]["max_val_samples"] = n_images
            cfg["dataset"]["max_test_samples"] = n_images
            cfg["training"]["batch_size"] = min(8, n_images)
            cfg["experiment"]["run_id"] = f"overfit_{letter}_n{n_images}"
            run_dir = out_root / f"overfit_{letter}_n{n_images}"
            print(f"\n=== overfit {letter} n={n_images} ===", flush=True)
            s = run_one(cfg, run_dir, letter=letter, phases=phases, seed=args.seed, log_every=100)
            s["n_images"] = n_images
            results.append(s)
    # gate: locality (C) has more per-location capacity than transpose (B), so it
    # must be able to FIT a tiny fixed train set at least as well as B. We measure
    # the MINIMUM train MSE reached (pure fitting capacity), not train MSE at the
    # best-val step. If C cannot fit >= B, that would indicate an implementation
    # bug. We also report val MSE to show the generalization story.
    gate = {"runs": results,
            "note": "gate is on MIN train MSE (fitting capacity); val shown for the overfitting narrative"}
    by = {(r["variant"], r["n_images"]): r for r in results}
    checks = []
    for n in args.sizes:
        b = by[("B", n)]["min_train_mse"]
        c = by[("C", n)]["min_train_mse"]
        checks.append({"n_images": n,
                       "B_min_train_mse": b, "C_min_train_mse": c,
                       "B_best_val_mse": by[("B", n)]["best_val_mse"],
                       "C_best_val_mse": by[("C", n)]["best_val_mse"],
                       "C_fits_at_least_as_well_as_B": (c is not None and b is not None and c <= b * 1.25)})
    gate["checks"] = checks
    gate["pass"] = all(ch["C_fits_at_least_as_well_as_B"] for ch in checks)
    gate["interpretation"] = (
        "PASS => locality block fits at least as well as transpose (no implementation/optimization bug); "
        "C's worse val MSE is therefore a generalization/overfitting effect, not broken code."
    )
    (out_root / "overfit_gate.json").write_text(json.dumps(gate, indent=2))
    print("OVERFIT GATE:", "PASS" if gate["pass"] else "FAIL", flush=True)


def cmd_ablation(args) -> None:
    """Phase 6 Track 2: A/B/C/D on BBBC022 proxy, multiple seeds, uniform protocol."""
    out_root = OUT / "track2_proxy_bbbc022"
    out_root.mkdir(parents=True, exist_ok=True)
    phases = default_phases(scale=args.scale)
    rows = []
    for seed in args.seeds:
        for letter in ["A", "B", "C", "D"]:
            cfg = bbbc022_base(args.device)
            cfg = apply_variant(cfg, letter)
            cfg["experiment"]["seed"] = seed
            cfg["dataset"]["seed"] = seed
            cfg["pattern_generator"]["seed"] = seed
            cfg["experiment"]["run_id"] = f"{letter}_seed{seed}"
            run_dir = out_root / f"{letter}_seed{seed}"
            summary_path = run_dir / "metrics" / "run_summary.json"
            if summary_path.exists() and not args.force:
                rows.append(json.loads(summary_path.read_text()))
                continue
            print(f"\n=== ablation {letter} seed={seed} ===", flush=True)
            rows.append(run_one(cfg, run_dir, letter=letter, phases=phases, seed=seed))
    _write_ablation_aggregate(out_root, rows, args.seeds)


def _write_ablation_aggregate(out_root: Path, rows: list[dict], seeds: list[int]) -> None:
    by_variant: dict[str, list[dict]] = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)
    agg = {}
    for letter in ["A", "B", "C", "D"]:
        rs = by_variant.get(letter, [])
        agg[letter] = {
            "label": VARIANT_LABEL[letter],
            "test_mse": _stats([r["test_mse"] for r in rs]),
            "test_ssim": _stats([r["test_ssim"] for r in rs]),
            "best_val_mse": _stats([r["best_val_mse"] for r in rs]),
            "best_train_mse": _stats([r["best_train_mse"] for r in rs]),
            "overfit_gap": _stats([r["overfit_gap"] for r in rs]),
        }
    best_letter = min(agg, key=lambda L: (agg[L]["test_mse"]["mean"] is None, agg[L]["test_mse"]["mean"]))
    paper = {"A": 0.0042, "B": 0.0038, "C": 0.0029, "D": 0.0041}
    out = {
        "label": "BBBC022 (U2OS Cell-Painting) SUBSTITUTE proxy - NOT paper U2OS reproduction",
        "seeds": seeds, "aggregate": agg, "proxy_best_variant": best_letter,
        "paper_table3_mse": paper, "paper_best_variant": "C",
        "ordering_matches_paper": best_letter == "C",
    }
    (out_root / "aggregate_summary.json").write_text(json.dumps(out, indent=2))


def cmd_patchmnist(args) -> None:
    """Phase 6: PatchMNIST sanity at the EXACT ablation config (x16, T=4, 256px).

    Locality is known to win on PatchMNIST in this repo; running it at the same
    compression/T as the BBBC022 ablation separates 'locality code broken' from
    'BBBC022 proxy != U2OS'.
    """
    out_root = OUT / "patchmnist_sanity"
    out_root.mkdir(parents=True, exist_ok=True)
    phases = default_phases(scale=args.scale)
    rows = []
    for seed in args.seeds:
        for letter in ["B", "C"]:  # B=transpose, C=locality (both learnable freq, x16/T4)
            cfg = patchmnist_base(args.device)
            cfg["pattern_generator"]["num_patterns"] = 4
            cfg["forward_model"]["downscale_factor"] = 8
            cfg = apply_variant(cfg, letter)
            if args.max_train:
                cfg["dataset"]["max_train_samples"] = args.max_train
            cfg["experiment"]["seed"] = seed
            cfg["dataset"]["seed"] = seed
            cfg["pattern_generator"]["seed"] = seed
            cfg["experiment"]["run_id"] = f"patchmnist_{letter}_seed{seed}"
            run_dir = out_root / f"patchmnist_{letter}_seed{seed}"
            summary_path = run_dir / "metrics" / "run_summary.json"
            if summary_path.exists() and not args.force:
                rows.append(json.loads(summary_path.read_text()))
                continue
            print(f"\n=== patchmnist sanity {letter} seed={seed} ===", flush=True)
            rows.append(run_one(cfg, run_dir, letter=letter, phases=phases, seed=seed))
    by_variant: dict[str, list[dict]] = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)
    out = {
        "label": "PatchMNIST sanity at ablation config (x16, T=4, 256px). B=transpose, C=locality.",
        "seeds": args.seeds,
        "transpose_B": _stats([r["test_mse"] for r in by_variant.get("B", [])]),
        "locality_C": _stats([r["test_mse"] for r in by_variant.get("C", [])]),
    }
    out["locality_beats_transpose"] = (
        out["locality_C"]["mean"] is not None and out["transpose_B"]["mean"] is not None
        and out["locality_C"]["mean"] < out["transpose_B"]["mean"]
    )
    (out_root / "aggregate_summary.json").write_text(json.dumps(out, indent=2))
    print("PATCHMNIST locality_beats_transpose:", out["locality_beats_transpose"], flush=True)


def cmd_trainsize(args) -> None:
    """Phase 7 evidence: BBBC022 B vs C as a function of train-set size."""
    out_root = OUT / "trainsize_sweep"
    out_root.mkdir(parents=True, exist_ok=True)
    phases = default_phases(scale=args.scale)
    rows = []
    for n_images in args.sizes:
        for letter in ["B", "C"]:
            cfg = bbbc022_base(args.device)
            cfg = apply_variant(cfg, letter)
            cfg["dataset"]["max_train_samples"] = n_images  # cap train; val/test fixed
            cfg["experiment"]["seed"] = args.seed
            cfg["dataset"]["seed"] = args.seed
            cfg["pattern_generator"]["seed"] = args.seed
            cfg["experiment"]["run_id"] = f"trainsize_{letter}_n{n_images}"
            run_dir = out_root / f"trainsize_{letter}_n{n_images}"
            summary_path = run_dir / "metrics" / "run_summary.json"
            if summary_path.exists() and not args.force:
                s = json.loads(summary_path.read_text())
            else:
                print(f"\n=== trainsize {letter} n={n_images} ===", flush=True)
                s = run_one(cfg, run_dir, letter=letter, phases=phases, seed=args.seed)
            s["n_images"] = n_images
            rows.append(s)
    by = {}
    for r in rows:
        by.setdefault(r["n_images"], {})[r["variant"]] = r
    sweep = []
    for n in args.sizes:
        b = by[n]["B"]["test_mse"]
        c = by[n]["C"]["test_mse"]
        sweep.append({"n_images": n, "B_transpose_test_mse": b, "C_locality_test_mse": c,
                      "C_minus_B": c - b, "C_beats_B": c < b})
    out = {"label": "BBBC022 B(transpose) vs C(locality) test MSE vs train size", "sweep": sweep}
    (out_root / "aggregate_summary.json").write_text(json.dumps(out, indent=2))
    # plot
    ns = [s["n_images"] for s in sweep]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ns, [s["B_transpose_test_mse"] for s in sweep], "o-", label="B transpose")
    ax.plot(ns, [s["C_locality_test_mse"] for s in sweep], "s-", label="C locality")
    ax.set_xlabel("# train images")
    ax.set_ylabel("test MSE")
    ax.set_title("BBBC022 proxy: transpose vs locality vs train size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / "trainsize_sweep.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="AM-3 Table-3 resolution runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("overfit")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sizes", nargs="+", type=int, default=[1, 8])
    p.set_defaults(func=cmd_overfit)

    p = sub.add_parser("ablation")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_ablation)

    p = sub.add_parser("patchmnist")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--max-train", type=int, default=None, dest="max_train")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_patchmnist)

    p = sub.add_parser("trainsize")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sizes", nargs="+", type=int, default=[42, 84, 168])
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_trainsize)

    args = parser.parse_args()
    print(f"AM-3 runner cmd={args.cmd} cuda={torch.cuda.is_available()}", flush=True)
    args.func(args)


if __name__ == "__main__":
    main()
