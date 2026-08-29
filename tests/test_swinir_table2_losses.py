"""Tests for Table 2 loss stack and SwinIR LI forward/backward."""

from __future__ import annotations

import pytest
import torch


def _model(learnable: bool):
    from baselines.swinir.table2_pipeline import SwinIRTable2Model, default_table2_config

    cfg = default_table2_config(learnable=learnable, image_size=32)
    cfg["swinir"]["img_size"] = 32
    model = SwinIRTable2Model(cfg)
    model(torch.zeros(1, 1, 32, 32))  # build lazy locality upsampling
    return model


def test_pixel_loss_finite():
    from baselines.swinir.losses import pixel_loss

    pred = torch.rand(2, 1, 16, 16, requires_grad=True)
    target = torch.rand(2, 1, 16, 16)
    loss = pixel_loss(pred, target, "l1")
    loss.backward()
    assert torch.isfinite(loss).item()
    assert pred.grad is not None and torch.isfinite(pred.grad).all().item()


def test_swinir_wo_li_forward_backward():
    pytest.importorskip("timm")
    model = _model(learnable=False)
    x = torch.rand(2, 1, 32, 32)
    out = model(x, sigmoid_m=None, apply_noise=False)
    assert out["x_recon"].shape == (2, 1, 32, 32)
    loss = out["x_recon"].mean()
    loss.backward()
    assert torch.isfinite(out["x_recon"]).all().item()


def test_swinir_with_li_forward_backward():
    pytest.importorskip("timm")
    model = _model(learnable=True)
    assert len(model.illumination_parameters()) > 0
    x = torch.rand(2, 1, 32, 32)
    out = model(x, sigmoid_m=8.0, apply_noise=False)
    loss = torch.nn.functional.l1_loss(out["x_recon"], x)
    loss.backward()
    grads = [p.grad for p in model.illumination_parameters()]
    assert any(g is not None and torch.isfinite(g).all().item() for g in grads)


def test_perceptual_loss_finite():
    pytest.importorskip("torchvision")
    from torchvision.models import VGG19_Weights

    try:
        VGG19_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
        available = True
    except Exception:
        available = False
    from baselines.swinir.losses import VGGPerceptualLoss

    loss_fn = VGGPerceptualLoss(weights_available=available)
    pred = torch.rand(2, 1, 32, 32, requires_grad=True)
    target = torch.rand(2, 1, 32, 32)
    loss = loss_fn(pred, target)
    loss.backward()
    assert torch.isfinite(loss).item()
    assert pred.grad is not None and torch.isfinite(pred.grad).all().item()


def test_adversarial_loss_and_discriminator_finite():
    from baselines.swinir.losses import GANLoss, VGGStyleDiscriminator

    disc = VGGStyleDiscriminator(in_chans=1)
    gan = GANLoss(kind="vanilla")
    real = torch.rand(2, 1, 64, 64)
    fake = torch.rand(2, 1, 64, 64)
    d_out_real = disc(real)
    d_out_fake = disc(fake)
    d_loss = gan(d_out_real, True) + gan(d_out_fake, False)
    d_loss.backward()
    assert torch.isfinite(d_loss).item()
    assert d_out_real.shape == (2, 1)


def test_loss_stack_assembly():
    from baselines.swinir.losses import build_loss_stack

    stack = build_loss_stack(
        {"pixel_weight": 1.0, "perceptual_weight": 0.0, "gan_weight": 0.1, "in_chans": 1},
        torch.device("cpu"),
    )
    assert "discriminator" in stack
    assert "gan_loss" in stack
    assert "perceptual" not in stack


def test_psnr_ssim_eval_finite():
    from evaluation.metrics import psnr, ssim

    pred = torch.rand(1, 1, 32, 32)
    target = torch.rand(1, 1, 32, 32)
    assert torch.isfinite(psnr(pred, target)).item()
    assert torch.isfinite(ssim(pred, target)).item()
