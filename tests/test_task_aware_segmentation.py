"""Tests for the task-aware segmentation head, losses, and staged trainer wiring."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.segmentation_head import SegmentationHead, SegmentationHeadConfig
from models.task_aware_microscope import TaskAwareMicroscope
from training.segmentation_losses import (
    TaskAwareLossWeights,
    bce_with_logits_loss,
    soft_dice_loss,
    task_aware_segmentation_loss,
)


def _tiny_config() -> dict:
    return {
        "dataset": {"image_size": 32, "name": "patchmnist"},
        "pattern_generator": {
            "mode": "learnable_frequency",
            "num_patterns": 4,
            "sigmoid_m": 1.0,
            "seed": 1,
        },
        "forward_model": {"downscale_factor": 4, "use_impulse_psfs": True},
        "detector_noise": {"mode": "noise_free", "apply_noise": False},
        "inverse_model": {
            "upsampling": {"mode": "transpose_conv", "downscale_factor": 4, "num_patterns": 4},
            "reconstruction": {
                "in_channels": 4,
                "hidden_channels": [16, 16, 8, 8, 4, 1],
                "kernel_size": 3,
                "padding": 1,
            },
        },
        "sigmoid_schedule": {"epoch_baseline": 0, "epoch_cutoff": 0, "epoch_step": 1, "m_init": 1.0},
        "segmentation_head": {"in_channels": 1, "hidden_channels": [8, 8, 1], "kernel_size": 3, "padding": 1},
    }


# --------------------------------------------------------------------------- #
# segmentation head
# --------------------------------------------------------------------------- #
def test_segmentation_head_output_shape_is_logits():
    head = SegmentationHead(SegmentationHeadConfig(hidden_channels=[8, 8, 1]))
    x = torch.rand(2, 1, 32, 32)
    out = head(x)
    assert out.shape == x.shape
    # The head emits raw logits: the final layer is a bare Conv2d (no Sigmoid).
    assert isinstance(head.net[-1], nn.Conv2d)
    assert not any(isinstance(m, nn.Sigmoid) for m in head.net)


def test_segmentation_head_rejects_non_unit_final_channel():
    try:
        SegmentationHead(SegmentationHeadConfig(hidden_channels=[8, 8, 2]))
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-unit final channel")


# --------------------------------------------------------------------------- #
# task-aware microscope forward
# --------------------------------------------------------------------------- #
def test_task_aware_microscope_forward_keys_and_shapes():
    model = TaskAwareMicroscope.from_run_config(_tiny_config())
    specimen = torch.rand(2, 1, 32, 32)
    outputs = model(specimen, sigmoid_m=1.0, apply_noise=False)
    for key in ("x_recon", "seg_logits", "seg_prob"):
        assert outputs[key].shape == specimen.shape
    prob = outputs["seg_prob"]
    assert prob.min() >= 0.0 and prob.max() <= 1.0
    # seg_prob == sigmoid(seg_logits)
    assert torch.allclose(prob, torch.sigmoid(outputs["seg_logits"]), atol=1e-6)


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def test_loss_functions_smoke():
    logits = torch.randn(2, 1, 8, 8)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    assert bce_with_logits_loss(logits, target).item() >= 0.0
    dice = soft_dice_loss(torch.sigmoid(logits), target)
    assert 0.0 <= dice.item() <= 1.0


def test_task_aware_loss_combines_weights():
    outputs = {
        "seg_logits": torch.randn(2, 1, 8, 8),
        "seg_prob": torch.rand(2, 1, 8, 8),
        "x_recon": torch.rand(2, 1, 8, 8),
    }
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    specimen = torch.rand(2, 1, 8, 8)
    weights = TaskAwareLossWeights(seg_bce_weight=1.0, seg_dice_weight=0.5, reconstruction_l1_weight=0.1)
    total, comps = task_aware_segmentation_loss(outputs, target, weights, specimen=specimen)
    assert "bce" in comps and "dice" in comps and "recon_l1" in comps
    assert abs(comps["total"] - (1.0 * comps["bce"] + 0.5 * comps["dice"] + 0.1 * comps["recon_l1"])) < 1e-5
    assert total.requires_grad is False  # all inputs are leaf tensors w/o grad


def test_loss_weight_config_is_read_including_historical_alias():
    # Explicit weights are read.
    w = TaskAwareLossWeights.from_config({"seg_bce_weight": 2.0, "seg_dice_weight": 0.5})
    assert w.seg_bce_weight == 2.0 and w.seg_dice_weight == 0.5
    # The previously-DEAD `segmentation_bce_weight` alias is now honored.
    w2 = TaskAwareLossWeights.from_config({"segmentation_bce_weight": 3.0})
    assert w2.seg_bce_weight == 3.0


# --------------------------------------------------------------------------- #
# stage 2 vs stage 3 freeze / unfreeze + gradient flow
# --------------------------------------------------------------------------- #
def test_stage2_freezes_microscope_and_trains_only_head():
    model = TaskAwareMicroscope.from_run_config(_tiny_config())
    model.set_microscope_trainable(False)
    model.set_segmentation_trainable(True)
    report = model.trainable_parameter_report()
    assert report["illumination"]["all_frozen"]
    assert report["inverse_model"]["all_frozen"]
    assert report["segmentation_head"]["all_trainable"]

    specimen = torch.rand(2, 1, 32, 32)
    mask = (specimen > 0.3).float()
    outputs = model(specimen, sigmoid_m=1.0, apply_noise=False)
    loss = bce_with_logits_loss(outputs["seg_logits"], mask)
    loss.backward()
    assert model.segmentation_head.net[0].weight.grad is not None
    assert model.microscope.pattern_generator.W.grad is None
    for param in model.microscope.inverse_model.parameters():
        assert param.grad is None


def test_stage3_finetunes_all_and_illumination_receives_gradient():
    model = TaskAwareMicroscope.from_run_config(_tiny_config())
    model.set_segmentation_trainable(True)
    model.set_inverse_trainable(True)
    model.set_illumination_trainable(True)
    report = model.trainable_parameter_report()
    assert report["illumination"]["all_trainable"]
    assert report["inverse_model"]["all_trainable"]
    assert report["segmentation_head"]["all_trainable"]

    specimen = torch.rand(2, 1, 32, 32)
    mask = (specimen > 0.3).float()
    outputs = model(specimen, sigmoid_m=1.0, apply_noise=False)
    weights = TaskAwareLossWeights(seg_bce_weight=1.0, seg_dice_weight=0.5)
    loss, _ = task_aware_segmentation_loss(outputs, mask, weights)
    loss.backward()

    # Illumination parameters (frequency-domain W) get the segmentation gradient.
    assert model.microscope.pattern_generator.W.grad is not None
    assert float(model.microscope.pattern_generator.W.grad.abs().sum()) > 0.0
    # Inverse model and seg head also receive gradients.
    assert any(p.grad is not None for p in model.microscope.inverse_model.parameters())
    assert model.segmentation_head.net[0].weight.grad is not None


def test_fixed_variant_has_no_illumination_parameters():
    config = _tiny_config()
    config["pattern_generator"]["mode"] = "random_fixed"
    model = TaskAwareMicroscope.from_run_config(config)
    assert model.illumination_parameters() == []
    report = model.trainable_parameter_report()
    # No illumination params -> reported as frozen (nothing to train).
    assert report["illumination"]["all_frozen"]


def test_trainer_reads_loss_weights_from_training_cfg():
    """Regression: the staged trainer must read the seg loss weights from config."""
    import inspect

    from training import train_task_aware_segmentation as trainer_mod

    source = inspect.getsource(trainer_mod.train_task_aware_segmentation)
    assert "TaskAwareLossWeights.from_config(training_cfg)" in source
