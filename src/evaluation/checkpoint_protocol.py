"""Recover the acquisition state associated with checkpoint weights."""

from __future__ import annotations

import math


def checkpoint_sigmoid_m(checkpoint: dict, *, override: float | None = None) -> float:
    """Legacy checkpoints without sharpness require an explicit evaluation value.

    In particular, `epoch` was sometimes an epoch and sometimes m. Guessing from
    that field can silently evaluate different physical masks.
    """
    value = override
    if value is None:
        value = checkpoint.get("sigmoid_m", checkpoint.get("best_m"))
    if value is None:
        raise ValueError("Checkpoint has no saved sigmoid_m; supply an explicit evaluation sharpness")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Checkpoint sigmoid_m must be finite and positive")
    return value
