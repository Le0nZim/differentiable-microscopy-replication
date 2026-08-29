"""Track illumination-pattern changes during training."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from models.microscope import DifferentiableMicroscope


@dataclass
class PatternSnapshot:
    """Initial and final illumination parameters for delta metrics."""

    initial_patterns: torch.Tensor | None
    final_patterns: torch.Tensor | None
    initial_w: torch.Tensor | None
    final_w: torch.Tensor | None
    initial_y_down: torch.Tensor | None = None
    final_y_down: torch.Tensor | None = None
    max_grad_norm_w: float = 0.0

    def pattern_delta(self) -> float | None:
        if self.initial_patterns is None or self.final_patterns is None:
            return None
        return float((self.final_patterns - self.initial_patterns).abs().mean().item())

    def w_delta(self) -> float | None:
        if self.initial_w is None or self.final_w is None:
            return None
        return float((self.final_w - self.initial_w).abs().mean().item())

    def detector_delta(self) -> float | None:
        if self.initial_y_down is None or self.final_y_down is None:
            return None
        return float((self.final_y_down - self.initial_y_down).abs().mean().item())

    def to_dict(self) -> dict[str, float | None]:
        return {
            "pattern_delta": self.pattern_delta(),
            "w_delta": self.w_delta(),
            "detector_delta": self.detector_delta(),
            "max_grad_norm_W": self.max_grad_norm_w,
        }


@torch.no_grad()
def capture_pattern_snapshot(
    model: DifferentiableMicroscope,
    *,
    sigmoid_m: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    patterns = model.pattern_generator(sigmoid_m=sigmoid_m).detach().cpu().clone()
    w_tensor: torch.Tensor | None = None
    if hasattr(model.pattern_generator, "W") and model.pattern_generator.W is not None:
        w_tensor = model.pattern_generator.W.detach().cpu().clone()
    return patterns, w_tensor


@torch.no_grad()
def capture_detector_snapshot(
    model: DifferentiableMicroscope,
    specimen: torch.Tensor,
    *,
    sigmoid_m: float,
    apply_noise: bool,
) -> torch.Tensor:
    model.eval()
    outputs = model(specimen, sigmoid_m=sigmoid_m, apply_noise=apply_noise)
    model.train()
    return outputs["y_down"].detach().cpu().clone()


def finalize_pattern_snapshot(
    snapshot: PatternSnapshot,
    model: DifferentiableMicroscope,
    *,
    sigmoid_m: float,
    reference_specimen: torch.Tensor | None = None,
    apply_noise: bool = True,
) -> PatternSnapshot:
    final_patterns, final_w = capture_pattern_snapshot(model, sigmoid_m=sigmoid_m)
    snapshot.final_patterns = final_patterns
    snapshot.final_w = final_w
    if reference_specimen is not None:
        snapshot.final_y_down = capture_detector_snapshot(
            model,
            reference_specimen,
            sigmoid_m=sigmoid_m,
            apply_noise=apply_noise,
        )
    return snapshot
