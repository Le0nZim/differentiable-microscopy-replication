"""Shared inverse-warmup checkpoint so C and D can fork with identical Adam state.

Independent C vs D GPU warmups can diverge even when construction hashes match
(FFT round-trip + non-deterministic kernels). If that happens, both arms resume
from one frozen-illumination checkpoint, including inverse Adam.
"""

from __future__ import annotations

import csv
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


# Larger than construction-time FFT round-trip (~1e-6) and larger than typical
# log-cadence float noise; smaller than the 8,500-step C/D test-MSE gap (~4e-4).
WARMUP_ALIGN_MAX_ABS_LOSS = 1e-4
WARMUP_ALIGN_MAX_ABS_VAL_MSE = 1e-4

COMPARE_KEYS = (
    "loss",
    "train_mse",
    "val_mse",
    "val_ssim",
    "grad_norm_upsampler",
    "grad_norm_recon",
    "H_t_min",
    "H_t_max",
    "H_t_mean",
    "H_t_std",
    "H_t_binary_fraction",
    "tau_displacement",
    "H_t_displacement",
    "illum_delta",
)


def inverse_from_full_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "inverse_model."
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def snapshot_adam_inverse(optimizer: Optimizer, inverse: nn.Module) -> dict[str, dict]:
    """Adam state for inverse parameters, keyed by ``named_parameters`` names."""
    id_to_name = {id(p): n for n, p in inverse.named_parameters()}
    named: dict[str, dict] = {}
    for param, state in optimizer.state.items():
        name = id_to_name.get(id(param))
        if name is None:
            continue
        named[name] = {
            k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
            for k, v in state.items()
        }
    return named


def load_adam_inverse(
    optimizer: Optimizer,
    inverse: nn.Module,
    named: dict[str, dict],
    device: torch.device,
) -> None:
    name_to_param = dict(inverse.named_parameters())
    missing = [n for n in named if n not in name_to_param]
    if missing:
        raise KeyError(f"Adam inverse snapshot has names not in module: {missing[:8]}")
    for name, state in named.items():
        restored = {}
        for key, value in state.items():
            restored[key] = value.to(device) if torch.is_tensor(value) else value
        optimizer.state[name_to_param[name]] = restored


def snapshot_rng(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        payload["cuda_all"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng(payload: dict[str, Any], device: torch.device) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    if "cuda_all" in payload and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_all"])


def skip_cycle_steps(train_iter, n_steps: int) -> None:
    """Advance ``itertools.cycle(train_loader)`` by ``n_steps`` (no training)."""
    for i in range(int(n_steps)):
        next(train_iter)
        if (i + 1) % 10_000 == 0:
            print(f"  skipped {i + 1}/{n_steps} cached loader steps", flush=True)


def save_warmup_checkpoint(
    path: Path,
    *,
    global_step: int,
    seed: int,
    warmup_arm: str,
    inverse: nn.Module,
    optimizer: Optimizer,
    tau0: torch.Tensor,
    H_t0: torch.Tensor,
    best: dict,
    min_train_mse: float,
    max_grad: dict,
    history: list[dict],
    device: torch.device,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    best_inv = None
    if best.get("state") is not None:
        best_inv = inverse_from_full_state(best["state"])
        best_inv = {k: v.detach().cpu().clone() for k, v in best_inv.items()}
    payload = {
        "global_step": int(global_step),
        "seed": int(seed),
        "warmup_arm": warmup_arm,
        "inverse_state_dict": {k: v.detach().cpu().clone() for k, v in inverse.state_dict().items()},
        "adam_inverse_by_name": snapshot_adam_inverse(optimizer, inverse),
        "tau0": tau0.detach().cpu().clone(),
        "H_t0": H_t0.detach().cpu().clone(),
        "best": {
            "val_mse": best["val_mse"],
            "m": best["m"],
            "step": best["step"],
            "train_mse": best["train_mse"],
            "inverse_state_dict": best_inv,
        },
        "min_train_mse": min_train_mse,
        "max_grad": dict(max_grad),
        "history": history,
        "rng": snapshot_rng(device),
        "note": (
            "Shared frozen-illumination warmup. Fork C (W=FFT2(τ₀)) and D (τ=τ₀) "
            "from inverse weights + inverse Adam in this file."
        ),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    print(f"wrote shared warmup checkpoint {path} at step {global_step}", flush=True)


def load_warmup_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict:
    path = Path(path)
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_best_full_state(
    model: nn.Module,
    best_inverse: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor] | None:
    """Build this arm's full ``state_dict`` corresponding to the warmup-era best."""
    if best_inverse is None:
        return None
    current_inv = deepcopy(model.inverse_model.state_dict())
    model.inverse_model.load_state_dict(best_inverse)
    full = deepcopy(model.state_dict())
    model.inverse_model.load_state_dict(current_inv)
    return full


def compare_frozen_interval(
    c_rows: list[dict],
    d_rows: list[dict],
    *,
    warmup_end: int = 60_750,
    max_abs_loss: float = WARMUP_ALIGN_MAX_ABS_LOSS,
    max_abs_val_mse: float = WARMUP_ALIGN_MAX_ABS_VAL_MSE,
) -> dict[str, Any]:
    """Align logged warmup rows and report max |C−D| per metric."""

    def _by_step(rows: list[dict]) -> dict[int, dict]:
        out = {}
        for row in rows:
            step = int(row["step"])
            if step <= warmup_end:
                out[step] = row
        return out

    c_by, d_by = _by_step(c_rows), _by_step(d_rows)
    overlap = sorted(set(c_by) & set(d_by))
    max_abs: dict[str, float] = {k: 0.0 for k in COMPARE_KEYS}
    at_step: dict[str, int | None] = {k: None for k in COMPARE_KEYS}
    first_loss_split = None
    for step in overlap:
        cr, dr = c_by[step], d_by[step]
        for key in COMPARE_KEYS:
            if key not in cr or key not in dr:
                continue
            if cr[key] in ("", None) or dr[key] in ("", None):
                continue
            delta = abs(float(cr[key]) - float(dr[key]))
            if delta >= max_abs[key]:
                max_abs[key] = delta
                at_step[key] = step
        loss_delta = abs(float(cr["loss"]) - float(dr["loss"]))
        if loss_delta > 1e-8 and first_loss_split is None:
            first_loss_split = {
                "step": step,
                "C_loss": float(cr["loss"]),
                "D_loss": float(dr["loss"]),
                "abs_delta": loss_delta,
            }

    material = bool(overlap) and (
        max_abs["loss"] > max_abs_loss or max_abs["val_mse"] > max_abs_val_mse
    )
    return {
        "n_overlap": len(overlap),
        "c_warmup_steps_logged": len(c_by),
        "d_warmup_steps_logged": len(d_by),
        "last_overlap_step": overlap[-1] if overlap else None,
        "warmup_end": warmup_end,
        "max_abs": max_abs,
        "max_abs_at_step": at_step,
        "first_loss_split": first_loss_split,
        "thresholds": {
            "loss": max_abs_loss,
            "val_mse": max_abs_val_mse,
        },
        "materially_diverged": material,
        "pass": bool(overlap) and not material,
    }


def read_step_log(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return list(csv.DictReader(path.open()))
