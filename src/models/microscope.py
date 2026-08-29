"""End-to-end differentiable compressive fluorescence microscope."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .detector_noise import DetectorNoise, DetectorNoiseConfig
from .forward_model import ForwardModel, ForwardModelConfig
from .inverse_model import InverseModel, InverseModelConfig
from .pattern_generator import PatternGenerator, PatternGeneratorConfig, SigmoidSchedule


@dataclass
class MicroscopeConfig:
    """Configuration for the full differentiable microscope."""

    pattern_generator: PatternGeneratorConfig
    forward_model: ForwardModelConfig
    detector_noise: DetectorNoiseConfig
    inverse_model: InverseModelConfig
    image_size: int

    @classmethod
    def from_dict(cls, data: dict) -> "MicroscopeConfig":
        detector_cfg = DetectorNoiseConfig.from_dict(data["detector_noise"])
        forward_cfg = ForwardModelConfig.from_dict(data["forward_model"])
        if detector_cfg.downscale_factor == 1 and forward_cfg.downscale_factor != 1:
            detector_cfg = DetectorNoiseConfig(
                **{**detector_cfg.__dict__, "downscale_factor": forward_cfg.downscale_factor}
            )
        return cls(
            pattern_generator=PatternGeneratorConfig.from_dict(data["pattern_generator"]),
            forward_model=forward_cfg,
            detector_noise=detector_cfg,
            inverse_model=InverseModelConfig.from_dict(data["inverse_model"]),
            image_size=data["dataset"]["image_size"],
        )


class DifferentiableMicroscope(nn.Module):
    """Full forward + inverse differentiable microscopy pipeline."""

    def __init__(self, config: MicroscopeConfig) -> None:
        super().__init__()
        self.config = config
        downscale = config.forward_model.downscale_factor
        if config.image_size % downscale != 0:
            raise ValueError("image_size must be divisible by downscale_factor")

        height_down = config.image_size // downscale
        self.pattern_generator = PatternGenerator(config.pattern_generator)
        self.forward_model = ForwardModel(config.forward_model)
        self.detector_noise = DetectorNoise(config.detector_noise)
        self.inverse_model = InverseModel(config.inverse_model, height_down, height_down)

    @classmethod
    def from_run_config(cls, config: dict[str, Any]) -> "DifferentiableMicroscope":
        pattern_cfg = dict(config["pattern_generator"])
        pattern_cfg["height"] = config["dataset"]["image_size"]
        pattern_cfg["width"] = config["dataset"]["image_size"]

        inverse_cfg = dict(config["inverse_model"])
        inverse_cfg["upsampling"] = dict(inverse_cfg["upsampling"])
        inverse_cfg["reconstruction"] = dict(inverse_cfg["reconstruction"])
        inverse_cfg["upsampling"]["num_patterns"] = pattern_cfg["num_patterns"]
        inverse_cfg["upsampling"]["downscale_factor"] = config["forward_model"]["downscale_factor"]
        inverse_cfg["reconstruction"]["in_channels"] = pattern_cfg["num_patterns"]

        microscope_cfg = MicroscopeConfig(
            pattern_generator=PatternGeneratorConfig.from_dict(pattern_cfg),
            forward_model=ForwardModelConfig.from_dict(config["forward_model"]),
            detector_noise=DetectorNoiseConfig.from_dict(config["detector_noise"]),
            inverse_model=InverseModelConfig.from_dict(inverse_cfg),
            image_size=config["dataset"]["image_size"],
        )
        down = config["forward_model"]["downscale_factor"]
        if microscope_cfg.detector_noise.downscale_factor == 1 and down != 1:
            microscope_cfg.detector_noise.downscale_factor = down
        return cls(microscope_cfg)

    def forward(
        self,
        specimen: torch.Tensor,
        *,
        sigmoid_m: float | None = None,
        apply_noise: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the full acquisition and reconstruction pipeline.

        Args:
            specimen: Ground-truth image X with shape [B, 1, H, W].
            sigmoid_m: Optional override for custom sigmoid sharpness.
            apply_noise: Optional override for detector noise application.

        Returns:
            Dictionary with keys:
                x_recon: [B, 1, H, W]
                patterns: [T, 1, H, W]
                alpha_down: [B, T, H_down, W_down]
                y_down: [B, T, H_down, W_down]
        """
        patterns = self.pattern_generator(sigmoid_m=sigmoid_m)
        alpha_down = self.forward_model(specimen, patterns)
        y_down = self.detector_noise(alpha_down, apply_noise=apply_noise)
        x_recon = self.inverse_model(y_down)
        return {
            "x_recon": x_recon,
            "patterns": patterns,
            "alpha_down": alpha_down,
            "y_down": y_down,
        }

    def illumination_parameters(self) -> list[nn.Parameter]:
        if not self.pattern_generator.patterns_are_learnable():
            return []
        return list(self.pattern_generator.parameters())

    def inverse_parameters(self) -> list[nn.Parameter]:
        return list(self.inverse_model.parameters())

    def set_illumination_trainable(self, trainable: bool) -> None:
        for param in self.illumination_parameters():
            param.requires_grad = trainable
