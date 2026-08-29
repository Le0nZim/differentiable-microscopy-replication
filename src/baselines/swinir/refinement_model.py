"""SwinIR refinement appended after DifferentiableMicroscope base reconstruction."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from baselines.swinir.model_wrapper import build_swinir_from_config
from models.microscope import DifferentiableMicroscope


class OfflineSwinIRRefinement(nn.Module):
    """Pure post-processing: refine fixed x_base tensors (offline training)."""

    def __init__(self, swinir_cfg: dict[str, Any], refinement_cfg: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.swinir = build_swinir_from_config(swinir_cfg)
        refinement_cfg = refinement_cfg or {}
        self.refinement_mode = refinement_cfg.get("mode", "direct")
        alpha_init = float(refinement_cfg.get("alpha_init", 0.0))
        alpha_learnable = bool(refinement_cfg.get("alpha_learnable", True))
        if self.refinement_mode == "residual":
            self.alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=alpha_learnable)
        else:
            self.register_buffer("alpha", torch.tensor(1.0))

    def refine(self, x_base: torch.Tensor) -> torch.Tensor:
        delta = self.swinir(x_base)
        if self.refinement_mode == "residual":
            return torch.clamp(x_base + self.alpha * delta, 0.0, 1.0)
        return delta

    def forward(self, x_base: torch.Tensor) -> torch.Tensor:
        return self.refine(x_base)

    def parameters_for_training(self) -> list[nn.Parameter]:
        params = list(self.swinir.parameters())
        if self.refinement_mode == "residual" and isinstance(self.alpha, nn.Parameter):
            params.append(self.alpha)
        return params


class MicroscopeSwinIRRefinement(nn.Module):
    """Freeze (or optionally train) base microscope; refine x_base with SwinIR (upscale=1)."""

    def __init__(
        self,
        microscope: DifferentiableMicroscope,
        swinir_cfg: dict[str, Any],
        refinement_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.microscope = microscope
        refinement_cfg = refinement_cfg or {}
        self.refinement_mode = refinement_cfg.get("mode", "direct")
        self.offline_refiner = OfflineSwinIRRefinement(swinir_cfg, refinement_cfg)
        self.swinir = self.offline_refiner.swinir
        self._freeze_base = True

    def set_freeze_base(self, freeze: bool) -> None:
        self._freeze_base = freeze
        for param in self.microscope.parameters():
            param.requires_grad = not freeze

    def swinir_parameters(self) -> list[nn.Parameter]:
        return self.offline_refiner.parameters_for_training()

    def refine_base(self, x_base: torch.Tensor) -> torch.Tensor:
        return self.offline_refiner.refine(x_base)

    def forward(
        self,
        specimen: torch.Tensor,
        *,
        sigmoid_m: float | None = None,
        apply_noise: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if self._freeze_base:
            with torch.no_grad():
                base_out = self.microscope(specimen, sigmoid_m=sigmoid_m, apply_noise=apply_noise)
        else:
            base_out = self.microscope(specimen, sigmoid_m=sigmoid_m, apply_noise=apply_noise)
        x_base = base_out["x_recon"]
        x_refined = self.refine_base(x_base)
        return {
            **base_out,
            "x_base": x_base,
            "x_recon": x_refined,
        }


def build_refinement_from_config(config: dict[str, Any]) -> MicroscopeSwinIRRefinement:
    microscope = DifferentiableMicroscope.from_run_config(config)
    swinir_cfg = dict(config.get("swinir", {}))
    image_size = config["dataset"]["image_size"]
    swinir_cfg.setdefault("upscale", 1)
    swinir_cfg.setdefault("in_chans", 1)
    swinir_cfg.setdefault("img_size", image_size)
    refinement_cfg = dict(config.get("refinement", {}))
    return MicroscopeSwinIRRefinement(microscope, swinir_cfg, refinement_cfg)


def build_offline_refiner_from_config(config: dict[str, Any]) -> OfflineSwinIRRefinement:
    swinir_cfg = dict(config.get("swinir", {}))
    image_size = config["dataset"]["image_size"]
    swinir_cfg.setdefault("upscale", 1)
    swinir_cfg.setdefault("in_chans", 1)
    swinir_cfg.setdefault("img_size", image_size)
    return OfflineSwinIRRefinement(swinir_cfg, dict(config.get("refinement", {})))
