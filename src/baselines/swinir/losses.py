"""Loss stack for paper Table 2 SwinIR training.

Paper Sec 5.6: "we utilize pixel loss, adversarial loss, and perceptual loss to train the
network end-to-end ... All other configurations are similar to SwinIR configurations [26]."

Exact discriminator architecture, VGG perceptual layer set/weights, and loss weights are
NOT specified in the paper (PAPER_UNSPECIFIED). We use the standard SwinIR real-world SR /
ESRGAN/Real-ESRGAN defaults and log this as PAPER_UNSPECIFIED_FALLBACK.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pixel loss
# ---------------------------------------------------------------------------
def pixel_loss(pred: torch.Tensor, target: torch.Tensor, kind: str = "l1") -> torch.Tensor:
    if kind == "l1":
        return F.l1_loss(pred, target)
    if kind == "l2":
        return F.mse_loss(pred, target)
    raise ValueError(f"Unsupported pixel loss: {kind}")


# ---------------------------------------------------------------------------
# Perceptual loss (VGG19) — ESRGAN/SwinIR default layer set and weights
# ---------------------------------------------------------------------------
_VGG19_LAYER_IDX = {
    "conv1_2": 3,
    "conv2_2": 8,
    "conv3_4": 17,
    "conv4_4": 26,
    "conv5_4": 35,
}
_DEFAULT_LAYER_WEIGHTS = {
    "conv1_2": 0.1,
    "conv2_2": 0.1,
    "conv3_4": 1.0,
    "conv4_4": 1.0,
    "conv5_4": 1.0,
}


class VGGPerceptualLoss(nn.Module):
    """VGG19 feature-space L1 perceptual loss (ESRGAN/SwinIR-style).

    Grayscale inputs are repeated to 3 channels and ImageNet-normalized.
    """

    def __init__(
        self,
        layer_weights: dict[str, float] | None = None,
        weights_available: bool = True,
    ) -> None:
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        self.layer_weights = layer_weights or dict(_DEFAULT_LAYER_WEIGHTS)
        self.max_idx = max(_VGG19_LAYER_IDX[name] for name in self.layer_weights)
        weights = VGG19_Weights.IMAGENET1K_V1 if weights_available else None
        features = vgg19(weights=weights).features[: self.max_idx + 1]
        for p in features.parameters():
            p.requires_grad = False
        self.features = features.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def _extract(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats: dict[str, torch.Tensor] = {}
        idx_to_name = {v: k for k, v in _VGG19_LAYER_IDX.items() if k in self.layer_weights}
        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in idx_to_name:
                feats[idx_to_name[i]] = h
        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = self._extract(self._prep(pred.clamp(0, 1)))
        with torch.no_grad():
            target_f = self._extract(self._prep(target.clamp(0, 1)))
        loss = pred.new_zeros(())
        for name, w in self.layer_weights.items():
            loss = loss + w * F.l1_loss(pred_f[name], target_f[name])
        return loss


# ---------------------------------------------------------------------------
# Discriminator (VGG-style PatchGAN) + GAN loss
# PAPER_UNSPECIFIED_FALLBACK: paper says "similar to SwinIR [26]"; SwinIR real-world SR uses
# a UNet-SN discriminator. We use a compact VGG-style discriminator with spectral norm.
# ---------------------------------------------------------------------------
def _sn_conv(in_c: int, out_c: int, stride: int) -> nn.Module:
    return nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False))


class VGGStyleDiscriminator(nn.Module):
    """Compact spectral-norm VGG-style discriminator for 1-channel images."""

    def __init__(self, in_chans: int = 1, base: int = 64) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        chans = [in_chans, base, base, base * 2, base * 2, base * 4, base * 4]
        for i in range(1, len(chans)):
            stride = 1 if i % 2 == 1 else 2
            layers.append(_sn_conv(chans[i - 1], chans[i], stride))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(chans[-1] * 16, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


class GANLoss(nn.Module):
    """Vanilla (BCE-with-logits) GAN loss."""

    def __init__(self, kind: str = "vanilla") -> None:
        super().__init__()
        self.kind = kind
        self.bce = nn.BCEWithLogitsLoss()

    def _target(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        return torch.full_like(pred, 1.0 if is_real else 0.0)

    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        if self.kind == "lsgan":
            target = self._target(pred, is_real)
            return F.mse_loss(pred, target)
        return self.bce(pred, self._target(pred, is_real))


def build_loss_stack(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Assemble enabled losses from config.

    cfg keys: pixel_weight, perceptual_weight, gan_weight, gan_type, vgg_weights_available.
    """
    stack: dict[str, Any] = {
        "pixel_weight": float(cfg.get("pixel_weight", 1.0)),
        "pixel_kind": cfg.get("pixel_kind", "l1"),
        "perceptual_weight": float(cfg.get("perceptual_weight", 0.0)),
        "gan_weight": float(cfg.get("gan_weight", 0.0)),
    }
    if stack["perceptual_weight"] > 0:
        stack["perceptual"] = VGGPerceptualLoss(
            weights_available=bool(cfg.get("vgg_weights_available", True))
        ).to(device)
    if stack["gan_weight"] > 0:
        stack["discriminator"] = VGGStyleDiscriminator(in_chans=int(cfg.get("in_chans", 1))).to(device)
        stack["gan_loss"] = GANLoss(kind=cfg.get("gan_type", "vanilla")).to(device)
    return stack
