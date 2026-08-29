"""SwinIR import smoke test."""

from __future__ import annotations

import pytest


def test_swinir_import():
    pytest.importorskip("timm")
    from baselines.swinir.model_wrapper import build_swinir_from_config

    model = build_swinir_from_config({"img_size": 32, "upscale": 1, "in_chans": 1})
    assert model is not None
