"""Compute device selection for training and evaluation."""

from __future__ import annotations

import os
from typing import Any

import torch


def resolve_device(device_spec: str | None = None) -> torch.device:
    """Resolve a torch device from a config value or environment.

    Priority:
        1. Explicit ``device_spec`` (e.g. ``cuda:1``, ``gpu1``, ``1``)
        2. ``DIFF_MICROSCOPY_DEVICE`` environment variable
        3. ``CUDA_VISIBLE_DEVICES`` if set (uses logical ``cuda:0``)
        4. Default ``cuda:0`` when CUDA is available, else CPU
    """
    spec = device_spec or os.environ.get("DIFF_MICROSCOPY_DEVICE")
    if spec is None:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    normalized = spec.strip().lower()
    if normalized in {"cpu", "cuda", "gpu"}:
        if normalized == "cpu":
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        raise RuntimeError(f"Requested device {spec!r} but CUDA is not available.")

    if normalized.startswith("cuda:"):
        index = int(normalized.split(":", 1)[1])
        _assert_cuda_index_available(index)
        return torch.device(f"cuda:{index}")

    if normalized.startswith("gpu"):
        index = int(normalized.removeprefix("gpu"))
        _assert_cuda_index_available(index)
        return torch.device(f"cuda:{index}")

    if normalized.isdigit():
        index = int(normalized)
        _assert_cuda_index_available(index)
        return torch.device(f"cuda:{index}")

    raise ValueError(
        f"Unsupported device spec {spec!r}. Use cpu, cuda, cuda:N, gpuN, or an integer GPU index."
    )


def _assert_cuda_index_available(index: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(f"Requested cuda:{index} but CUDA is not available.")
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested cuda:{index} but only {torch.cuda.device_count()} GPU(s) are visible."
        )


def device_from_config(config: dict[str, Any]) -> torch.device:
    """Read device preference from experiment config."""
    experiment_cfg = config.get("experiment", {})
    return resolve_device(experiment_cfg.get("device"))
