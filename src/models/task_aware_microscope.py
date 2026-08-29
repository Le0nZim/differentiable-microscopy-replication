"""Microscope + segmentation head for task-aware training (paper B.0.1).

Forward path: ``specimen -> microscope (forward + inverse) -> x_recon ->
segmentation head -> seg_logits``. The forward output dict exposes
``x_recon``, ``seg_logits`` and ``seg_prob`` (plus the microscope's intermediate
tensors), which the staged trainer consumes.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .microscope import DifferentiableMicroscope
from .segmentation_head import SegmentationHead


class TaskAwareMicroscope(nn.Module):
    """Content-aware microscope with an appended convolutional segmentation head."""

    def __init__(
        self,
        microscope: DifferentiableMicroscope,
        segmentation_head: SegmentationHead,
    ) -> None:
        super().__init__()
        self.microscope = microscope
        self.segmentation_head = segmentation_head

    @classmethod
    def from_run_config(cls, config: dict[str, Any]) -> "TaskAwareMicroscope":
        microscope = DifferentiableMicroscope.from_run_config(config)
        seg_cfg = config.get("segmentation_head", {"in_channels": 1})
        return cls(microscope, SegmentationHead.from_dict(seg_cfg))

    def forward(
        self,
        specimen: torch.Tensor,
        *,
        sigmoid_m: float | None = None,
        apply_noise: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.microscope(specimen, sigmoid_m=sigmoid_m, apply_noise=apply_noise)
        seg_logits = self.segmentation_head(outputs["x_recon"])
        outputs["seg_logits"] = seg_logits
        outputs["seg_prob"] = torch.sigmoid(seg_logits)
        # Backward-compatible alias (older code referenced ``seg_pred``).
        outputs["seg_pred"] = outputs["seg_prob"]
        return outputs

    # --- freeze / unfreeze helpers (paper B.0.1 stage control) ---------------

    def set_microscope_trainable(self, trainable: bool) -> None:
        for param in self.microscope.parameters():
            param.requires_grad = trainable

    def set_inverse_trainable(self, trainable: bool) -> None:
        for param in self.microscope.inverse_model.parameters():
            param.requires_grad = trainable

    def set_illumination_trainable(self, trainable: bool) -> None:
        self.microscope.set_illumination_trainable(trainable)

    def set_segmentation_trainable(self, trainable: bool) -> None:
        for param in self.segmentation_head.parameters():
            param.requires_grad = trainable

    # --- parameter group accessors ------------------------------------------

    def illumination_parameters(self) -> list[nn.Parameter]:
        return self.microscope.illumination_parameters()

    def inverse_parameters(self) -> list[nn.Parameter]:
        return self.microscope.inverse_parameters()

    def segmentation_parameters(self) -> list[nn.Parameter]:
        return list(self.segmentation_head.parameters())

    def trainable_parameter_report(self) -> dict[str, dict[str, int | bool]]:
        """Summarize requires_grad state per logical group (for stage evidence)."""

        def summarize(params: list[nn.Parameter]) -> dict[str, int | bool]:
            total = sum(int(p.numel()) for p in params)
            trainable = sum(int(p.numel()) for p in params if p.requires_grad)
            return {
                "num_params": int(len(params)),
                "total_elems": int(total),
                "trainable_elems": int(trainable),
                "all_trainable": bool(params and trainable == total),
                "all_frozen": bool(trainable == 0),
            }

        return {
            "illumination": summarize(self.illumination_parameters()),
            "inverse_model": summarize(self.inverse_parameters()),
            "segmentation_head": summarize(self.segmentation_parameters()),
        }
