"""Training stage helpers."""

from __future__ import annotations

from models.microscope import DifferentiableMicroscope
from models.pattern_generator import SigmoidSchedule


def configure_training_stage(
    model: DifferentiableMicroscope,
    schedule: SigmoidSchedule,
    epoch: int,
    *,
    force_freeze_patterns: bool = False,
) -> float:
    """Apply Stage A/B freezing and return the sigmoid sharpness for this epoch."""
    if force_freeze_patterns or schedule.should_freeze_patterns(epoch):
        model.set_illumination_trainable(False)
    else:
        model.set_illumination_trainable(True)
    return schedule.step(epoch)
