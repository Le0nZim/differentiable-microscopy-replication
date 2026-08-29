"""Table 2 pipeline: SwinIR replaces ψ with compressive LI forward model on SR patches."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from baselines.swinir.model_wrapper import build_swinir_from_config
from models.detector_noise import DetectorNoise
from models.forward_model import ForwardModel
from models.inverse_model import InverseModelConfig
from models.locality_upsampling import LocalityUpsampling, LocalityUpsamplingConfig
from models.pattern_generator import PatternGenerator


class SwinIRTable2Model(nn.Module):
    """Paper Table 2 style: LI forward + locality upsample + SwinIR inverse (upscale=1)."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.image_size = int(config["image_size"])
        self.downscale = int(config["forward_model"]["downscale_factor"])
        self.num_patterns = int(config["pattern_generator"]["num_patterns"])
        height_down = self.image_size // self.downscale
        width_down = self.image_size // self.downscale

        pg_cfg = dict(config["pattern_generator"])
        pg_cfg.setdefault("height", self.image_size)
        pg_cfg.setdefault("width", self.image_size)
        self.pattern_generator = PatternGenerator.from_dict(pg_cfg)
        self.forward_model = ForwardModel.from_dict(config["forward_model"])
        self.detector_noise = DetectorNoise.from_dict(config.get("detector_noise", {"mode": "noise_free"}))
        up_cfg = LocalityUpsamplingConfig(
            mode=config["inverse_model"]["upsampling"]["mode"],
            downscale_factor=self.downscale,
            num_patterns=self.num_patterns,
            use_mixing_cnn=False,
        )
        self.upsampling = LocalityUpsampling(up_cfg, height_down, width_down)
        self.fuse = nn.Conv2d(self.num_patterns, 1, kernel_size=1)
        swinir_cfg = dict(config["swinir"])
        swinir_cfg["img_size"] = self.image_size
        self.swinir = build_swinir_from_config(swinir_cfg)

    def forward(
        self,
        specimen: torch.Tensor,
        *,
        sigmoid_m: float | None = None,
        apply_noise: bool = False,
    ) -> dict[str, torch.Tensor]:
        patterns = self.pattern_generator(sigmoid_m=sigmoid_m)
        alpha_down = self.forward_model(specimen, patterns)
        y_down = self.detector_noise(alpha_down, apply_noise=apply_noise)
        y_up = self.upsampling(y_down)
        fused = self.fuse(y_up)
        x_recon = self.swinir(fused)
        return {
            "x_recon": x_recon,
            "patterns": patterns,
            "y_down": y_down,
        }

    def illumination_parameters(self) -> list[nn.Parameter]:
        if not self.pattern_generator.patterns_are_learnable():
            return []
        return list(self.pattern_generator.parameters())

    def swinir_parameters(self) -> list[nn.Parameter]:
        return list(self.upsampling.parameters()) + list(self.fuse.parameters()) + list(self.swinir.parameters())


def default_table2_config(*, learnable: bool, image_size: int = 64) -> dict[str, Any]:
    return {
        "image_size": image_size,
        "pattern_generator": {
            "mode": "learnable_frequency" if learnable else "random_fixed",
            "num_patterns": 4,
            "sigmoid_m": 1.0,
            "random_fixed_m": 10.0,
            "seed": 42,
        },
        "forward_model": {"downscale_factor": 8, "use_impulse_psfs": True},
        "detector_noise": {"mode": "noise_free", "apply_noise": False},
        "inverse_model": {
            "upsampling": {
                "mode": "locality_aware",
                "downscale_factor": 8,
                "num_patterns": 4,
            },
        },
        "swinir": {
            "upscale": 1,
            "in_chans": 1,
            "img_size": image_size,
            "window_size": 8,
            "upsampler": "",
            "embed_dim": 96,
            "depths": [2, 2, 2, 2, 2, 2],
            "num_heads": [3, 3, 3, 3, 3, 3],
        },
        "training": {
            "illumination_lr": 0.1 if learnable else 0.0,
            "swinir_lr": 2.0e-4,
            "learn_patterns": learnable,
        },
    }
