"""Machine-checkable audit of the Table-3 ablation variant wiring (AM-3).

The paper's Table 3 / Fig. 10 ablation (U2OS, x16 compression, T=4, 8x8
downscaling) compares four variants:

    A: fixed H_t          + transpose upsampling      + frequency-domain opt.
    B: learnable H_t      + transpose upsampling      + frequency-domain opt.
    C: learnable H_t      + proposed locality upsamp.  + frequency-domain opt.   (best)
    D: learnable H_t      + proposed locality upsamp.  + NO frequency-domain opt.

This module exposes the canonical expected mapping plus helpers that introspect
an actual built :class:`DifferentiableMicroscope` so tests and the runner can
*prove* that each variant is wired correctly (no config leakage, no accidental
freezing, no accidental frequency-domain optimization in D, etc.).
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from models.locality_upsampling import LocalityAwareUpsampling, TransposeConvUpsampling
from models.microscope import DifferentiableMicroscope

# Pattern-generator modes that carry learnable illumination parameters.
LEARNABLE_PATTERN_MODES = {"learnable_frequency", "learnable_spatial"}
# Pattern-generator modes that optimize the illumination in the Fourier domain
# (eq. 6: tau = real(ifft(W))).
FREQUENCY_PATTERN_MODES = {"learnable_frequency"}

# Canonical Table-3 expected configuration for each variant. ``frequency`` means
# "frequency-domain optimization is used"; ``learnable`` means "H_t is learned".
#
# NOTE on ``frequency`` for variant A: Fig. 10's caption lists A as "fixed H_t +
# Tr.Conv.Up + frequency domain optimization", but A's H_t is *fixed* pseudo-
# random, so there is no W being optimized in Fourier space. ``frequency`` here
# means the *operational* fact "frequency-domain optimization is actively
# training W", which is therefore False for A. The discriminating ablation is D,
# which keeps H_t learnable but drops the frequency-domain parameterization.
TABLE3_EXPECTED: dict[str, dict[str, Any]] = {
    "A": {
        "pattern_mode": "random_fixed",
        "learnable": False,
        "frequency": False,  # fixed H_t => no W optimized in Fourier space (see note above)
        "upsampling": "transpose_conv",
    },
    "B": {
        "pattern_mode": "learnable_frequency",
        "learnable": True,
        "frequency": True,
        "upsampling": "transpose_conv",
    },
    "C": {
        "pattern_mode": "learnable_frequency",
        "learnable": True,
        "frequency": True,
        "upsampling": "locality_aware",
    },
    "D": {
        "pattern_mode": "learnable_spatial",
        "learnable": True,
        "frequency": False,
        "upsampling": "locality_aware",
    },
}


def pattern_is_learnable(pattern_mode: str) -> bool:
    return pattern_mode in LEARNABLE_PATTERN_MODES


def uses_frequency_domain_optimization(pattern_mode: str) -> bool:
    """Frequency-domain optimization (eq. 6) is only active for learnable_frequency.

    For a *fixed* pattern (variant A) there is no W being optimized, but the
    forward path the paper draws is still the frequency-domain branch; callers
    that care about "is W being *trained*" should also check ``learnable``.
    """
    return pattern_mode in FREQUENCY_PATTERN_MODES


def _count_params(module: nn.Module, *, only_trainable: bool = False) -> int:
    return sum(
        p.numel()
        for p in module.parameters()
        if (p.requires_grad or not only_trainable)
    )


def describe_config(config: dict[str, Any]) -> dict[str, Any]:
    """Pure-config descriptor of a variant (no model build required)."""
    pattern_mode = config["pattern_generator"]["mode"]
    upsampling_mode = config["inverse_model"]["upsampling"]["mode"]
    downscale = int(config["forward_model"]["downscale_factor"])
    num_patterns = int(config["pattern_generator"]["num_patterns"])
    learn_patterns = bool(config["training"].get("learn_patterns", pattern_is_learnable(pattern_mode)))
    return {
        "pattern_mode": pattern_mode,
        "learnable_patterns": pattern_is_learnable(pattern_mode) and learn_patterns,
        "frequency_domain_optimization": uses_frequency_domain_optimization(pattern_mode)
        and learn_patterns,
        "upsampling_mode": upsampling_mode,
        "downscale_factor": downscale,
        "num_patterns": num_patterns,
        "compression": (downscale * downscale) / num_patterns,
    }


def audit_microscope(model: DifferentiableMicroscope, config: dict[str, Any]) -> dict[str, Any]:
    """Introspect a *built* microscope to verify the variant wiring.

    Returns a structured descriptor that records the *actual* module types,
    parameter counts and learnability flags observed on the instantiated model,
    not just what the config asked for.
    """
    upsampler = model.inverse_model.upsampling.upsampler
    if isinstance(upsampler, LocalityAwareUpsampling):
        actual_upsampling = "locality_aware"
    elif isinstance(upsampler, TransposeConvUpsampling):
        actual_upsampling = "transpose_conv"
    else:  # pragma: no cover - defensive
        actual_upsampling = type(upsampler).__name__

    pg = model.pattern_generator
    has_W = getattr(pg, "W", None) is not None
    has_tau = getattr(pg, "tau", None) is not None

    illumination_params = model.illumination_parameters()
    n_illum = sum(p.numel() for p in illumination_params)
    n_illum_trainable = sum(p.numel() for p in illumination_params if p.requires_grad)

    cfg = describe_config(config)
    return {
        **cfg,
        "actual_upsampling_module": actual_upsampling,
        "pattern_generator_class": type(pg).__name__,
        "pattern_has_frequency_W": has_W,
        "pattern_has_spatial_tau": has_tau,
        "num_illumination_params": int(n_illum),
        "num_illumination_trainable_params": int(n_illum_trainable),
        "num_upsampler_params": _count_params(model.inverse_model.upsampling),
        "num_reconstruction_params": _count_params(model.inverse_model.reconstruction),
        "num_total_params": _count_params(model),
    }


def expected_for_variant(letter: str) -> dict[str, Any]:
    return dict(TABLE3_EXPECTED[letter.upper()])


def check_variant(letter: str, descriptor: dict[str, Any]) -> list[str]:
    """Return a list of human-readable mismatch strings (empty == wiring OK)."""
    expected = expected_for_variant(letter)
    problems: list[str] = []

    if descriptor["pattern_mode"] != expected["pattern_mode"]:
        problems.append(
            f"{letter}: pattern_mode={descriptor['pattern_mode']} != expected {expected['pattern_mode']}"
        )
    if descriptor.get("actual_upsampling_module", descriptor["upsampling_mode"]) != expected[
        "upsampling"
    ]:
        problems.append(
            f"{letter}: upsampling={descriptor.get('actual_upsampling_module')} != expected {expected['upsampling']}"
        )
    if descriptor["learnable_patterns"] != expected["learnable"]:
        problems.append(
            f"{letter}: learnable_patterns={descriptor['learnable_patterns']} != expected {expected['learnable']}"
        )
    if descriptor["frequency_domain_optimization"] != expected["frequency"]:
        problems.append(
            f"{letter}: frequency_domain_optimization="
            f"{descriptor['frequency_domain_optimization']} != expected {expected['frequency']}"
        )
    # Cross-variant invariants enforced by the paper's design.
    if expected["learnable"] and descriptor.get("num_illumination_trainable_params", 0) == 0:
        problems.append(f"{letter}: expected trainable illumination params but found 0")
    if not expected["learnable"] and descriptor.get("num_illumination_trainable_params", 0) != 0:
        problems.append(
            f"{letter}: fixed variant must have 0 trainable illumination params, "
            f"found {descriptor.get('num_illumination_trainable_params')}"
        )
    return problems
