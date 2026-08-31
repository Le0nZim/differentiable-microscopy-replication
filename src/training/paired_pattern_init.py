"""Paired C/D illumination initialization: shared τ₀, W₀ = FFT2(τ₀) vs τ₀."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from models.microscope import DifferentiableMicroscope
from utils.reproducibility import set_seed


def generate_shared_tau0(
    num_patterns: int,
    height: int,
    width: int,
    seed: int,
) -> torch.Tensor:
    """τ₀ ~ N(0,1) with an isolated generator, matching PatternGenerator init."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.randn(num_patterns, 1, height, width, generator=generator)


@torch.no_grad()
def apply_shared_tau0(model: DifferentiableMicroscope, tau0: torch.Tensor) -> None:
    """Write the shared τ₀ into C (W = FFT2(τ₀)) or D (τ = τ₀)."""
    pg = model.pattern_generator
    device = next(model.parameters()).device
    tau0 = tau0.to(device=device)
    if pg.mode == "learnable_frequency":
        assert pg.W is not None
        real_dtype = torch.float32 if pg.W.dtype == torch.complex64 else torch.float64
        weights = torch.fft.fft2(tau0.to(dtype=real_dtype))
        if weights.dtype != pg.W.dtype:
            weights = weights.to(dtype=pg.W.dtype)
        if tuple(weights.shape) != tuple(pg.W.shape):
            raise ValueError(
                f"FFT2(τ₀) shape {tuple(weights.shape)} != W shape {tuple(pg.W.shape)}"
            )
        pg.W.copy_(weights)
    elif pg.mode == "learnable_spatial":
        assert pg.tau is not None
        if tuple(tau0.shape) != tuple(pg.tau.shape):
            raise ValueError(
                f"τ₀ shape {tuple(tau0.shape)} != tau shape {tuple(pg.tau.shape)}"
            )
        pg.tau.copy_(tau0.to(dtype=pg.tau.dtype))
    else:
        raise ValueError(f"apply_shared_tau0 does not apply to mode {pg.mode}")


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Stable content hash (shape + dtype + raw bytes)."""
    cpu = tensor.detach().contiguous().cpu()
    payload = hashlib.sha256()
    payload.update(str(tuple(cpu.shape)).encode())
    payload.update(str(cpu.dtype).encode())
    payload.update(cpu.numpy().tobytes())
    return payload.hexdigest()


def module_state_sha256(module: torch.nn.Module) -> str:
    payload = hashlib.sha256()
    for key, value in module.state_dict().items():
        payload.update(key.encode())
        payload.update(tensor_sha256(value).encode())
    return payload.hexdigest()


@torch.no_grad()
def physical_tau(model: DifferentiableMicroscope) -> torch.Tensor:
    return model.pattern_generator._spatial_tau().detach()


def paired_initialization_audit(
    model_c: DifferentiableMicroscope,
    model_d: DifferentiableMicroscope,
    *,
    first_batch_c: torch.Tensor,
    first_batch_d: torch.Tensor,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Compare C (frequency) and D (spatial) at initialization."""
    tau_c = physical_tau(model_c).cpu()
    tau_d = physical_tau(model_d).cpu()
    h_c = model_c.pattern_generator(sigmoid_m=1.0).detach().cpu()
    h_d = model_d.pattern_generator(sigmoid_m=1.0).detach().cpu()

    max_tau_diff = float((tau_c - tau_d).abs().max().item())
    max_ht_diff = float((h_c - h_d).abs().max().item())
    inverse_hash_c = module_state_sha256(model_c.inverse_model)
    inverse_hash_d = module_state_sha256(model_d.inverse_model)
    ups_hash_c = module_state_sha256(model_c.inverse_model.upsampling)
    ups_hash_d = module_state_sha256(model_d.inverse_model.upsampling)
    batch_hash_c = tensor_sha256(first_batch_c)
    batch_hash_d = tensor_sha256(first_batch_d)

    problems: list[str] = []
    if model_c.pattern_generator.W is None:
        problems.append("C is missing frequency-domain W")
    if model_d.pattern_generator.tau is None:
        problems.append("D is missing spatial tau")
    if max_tau_diff > atol:
        problems.append(f"max |τ_C - τ_D| = {max_tau_diff:.3e} > atol {atol}")
    if max_ht_diff > atol:
        problems.append(f"max |H_C - H_D| = {max_ht_diff:.3e} > atol {atol}")
    if inverse_hash_c != inverse_hash_d:
        problems.append("inverse-model state hashes differ")
    if ups_hash_c != ups_hash_d:
        problems.append("upsampler state hashes differ")
    if batch_hash_c != batch_hash_d:
        problems.append("first training-batch hashes differ")

    return {
        "pass": not problems,
        "problems": problems,
        "atol": atol,
        "max_abs_tau_difference": max_tau_diff,
        "max_abs_Ht_difference": max_ht_diff,
        "inverse_state_hash": inverse_hash_c,
        "inverse_state_hash_C": inverse_hash_c,
        "inverse_state_hash_D": inverse_hash_d,
        "upsampler_state_hash": ups_hash_c,
        "upsampler_state_hash_C": ups_hash_c,
        "upsampler_state_hash_D": ups_hash_d,
        "first_batch_hash": batch_hash_c,
        "first_batch_hash_C": batch_hash_c,
        "first_batch_hash_D": batch_hash_d,
        "C_has_W": model_c.pattern_generator.W is not None,
        "D_has_tau": model_d.pattern_generator.tau is not None,
        "C_has_tau_param": model_c.pattern_generator.tau is not None,
        "D_has_W": model_d.pattern_generator.W is not None,
    }


def build_paired_models(
    config_c: dict,
    config_d: dict,
    *,
    seed: int,
    device: torch.device,
) -> tuple[DifferentiableMicroscope, DifferentiableMicroscope, torch.Tensor]:
    """Construct C and D from the same seed and shared τ₀."""
    image_size = int(config_c["dataset"]["image_size"])
    num_patterns = int(config_c["pattern_generator"]["num_patterns"])
    tau0 = generate_shared_tau0(num_patterns, image_size, image_size, seed)

    set_seed(seed)
    model_c = DifferentiableMicroscope.from_run_config(config_c).to(device)
    apply_shared_tau0(model_c, tau0)

    set_seed(seed)
    model_d = DifferentiableMicroscope.from_run_config(config_d).to(device)
    apply_shared_tau0(model_d, tau0)
    return model_c, model_d, tau0
