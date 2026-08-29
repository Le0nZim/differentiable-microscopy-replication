"""Normalize user-facing config aliases into the internal schema."""

from __future__ import annotations

import copy
from typing import Any


def normalize_run_config(config: dict[str, Any]) -> dict[str, Any]:
    """Apply shorthand fields used by validation configs."""
    config = copy.deepcopy(config)

    dataset = config.setdefault("dataset", {})
    if "num_train_samples" in dataset:
        dataset["num_train"] = dataset["num_train_samples"]
        dataset["max_train_samples"] = dataset["num_train_samples"]
    if "num_val_samples" in dataset:
        dataset["num_val"] = dataset["num_val_samples"]
        dataset["max_val_samples"] = dataset["num_val_samples"]
    if "num_test_samples" in dataset:
        dataset["num_test"] = dataset["num_test_samples"]

    training = config.setdefault("training", {})
    if "learning_rate_inverse" in training:
        training["inverse_lr"] = training["learning_rate_inverse"]
    if "learning_rate_illumination" in training:
        training["illumination_lr"] = training["learning_rate_illumination"]
    if "max_steps" in training and "num_epochs" not in training:
        training.setdefault("num_epochs", 1)

    pattern_mode = config.pop("pattern_mode", None)
    if pattern_mode is not None:
        config.setdefault("pattern_generator", {})["mode"] = pattern_mode

    upsampling = config.pop("upsampling", None)
    if upsampling is not None:
        config.setdefault("inverse_model", {}).setdefault("upsampling", {})["mode"] = upsampling

    downscale_factor = config.pop("downscale_factor", None)
    if downscale_factor is not None:
        config.setdefault("forward_model", {})["downscale_factor"] = downscale_factor
        config.setdefault("inverse_model", {}).setdefault("upsampling", {})["downscale_factor"] = downscale_factor

    num_patterns = config.pop("num_patterns", None)
    if num_patterns is not None:
        config.setdefault("pattern_generator", {})["num_patterns"] = num_patterns
        config.setdefault("inverse_model", {}).setdefault("upsampling", {})["num_patterns"] = num_patterns
        config.setdefault("inverse_model", {}).setdefault("reconstruction", {})["in_channels"] = num_patterns

    noise_mode = config.pop("noise_mode", None)
    if noise_mode is not None:
        noise = config.setdefault("detector_noise", {})
        if noise_mode in {"none", "noise_free", "off"}:
            noise["mode"] = "noise_free"
            noise["apply_noise"] = False
        else:
            noise["mode"] = noise_mode
            noise["apply_noise"] = True

    learn_patterns = config.pop("learn_patterns", None)
    if learn_patterns is not None:
        training["learn_patterns"] = bool(learn_patterns)

    return config
