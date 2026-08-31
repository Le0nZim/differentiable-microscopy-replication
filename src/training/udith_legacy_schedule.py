"""Udith Haputhanthri legacy Algorithm-1 schedule, converted to optimizer steps.

Source (do not execute the notebook; this module encodes the derived schedule):

    udithhaputhanthri/CompressiveDabbaMu@d4308c0
    4_20230102_ablation-2.ipynb
    modules/m_inc_procs.py :: inc_m_class  (accumulating m, not a reset)

The notebook used 24,300 *legacy epochs* on 168 training images, batch 32,
``drop_last=True`` => ``floor(168/32) = 5`` optimizer updates per legacy epoch.

This module converts those epoch constants into optimizer *steps* using that
factor of 5. It does **not** call :class:`models.pattern_generator.SigmoidSchedule`,
whose ``step()`` resets ``m`` to 1 on non-trigger epochs.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------- #
# Historical epoch constants (from the ablation notebook @ d4308c0)           #
# --------------------------------------------------------------------------- #

LEGACY_REPO = "udithhaputhanthri/CompressiveDabbaMu"
LEGACY_COMMIT = "d4308c0"
LEGACY_NOTEBOOK = "4_20230102_ablation-2.ipynb"
LEGACY_SCHEDULER = "modules/m_inc_procs.py :: inc_m_class"

LEGACY_TRAIN_IMAGES = 168
LEGACY_BATCH_SIZE = 32
LEGACY_DROP_LAST = True
UPDATES_PER_LEGACY_EPOCH = LEGACY_TRAIN_IMAGES // LEGACY_BATCH_SIZE  # 5

LEGACY_TOTAL_EPOCHS = 300 * 81  # 24_300
LEGACY_TRAIN_H_AFTER_EPOCH = 150 * 81  # 12_150  (train_H_iter = epoch > 150*81)
LEGACY_M_THRESHOLD_EPOCH = 230 * 81  # 18_630
LEGACY_M_STEP_EPOCHS = 10 * 81  # 810

# Converted optimizer-step budget.
TOTAL_STEPS = LEGACY_TOTAL_EPOCHS * UPDATES_PER_LEGACY_EPOCH  # 121_500
WARMUP_STEPS = LEGACY_TRAIN_H_AFTER_EPOCH * UPDATES_PER_LEGACY_EPOCH  # 60_750
JOINT_TRAINABLE_STEPS = TOTAL_STEPS - WARMUP_STEPS  # 60_750
M_THRESHOLD_STEP = LEGACY_M_THRESHOLD_EPOCH * UPDATES_PER_LEGACY_EPOCH  # 93_150
M_STEP_STEPS = LEGACY_M_STEP_EPOCHS * UPDATES_PER_LEGACY_EPOCH  # 4_050

# Global-step inclusive bounds matching the causal-experiment table.
# (1-indexed global steps, as in the training loop: global_step += 1 then train.)
EXPECTED_GLOBAL_BOUNDS: list[dict[str, Any]] = [
    {"name": "inverse_warmup_m1", "m": 1.0, "start": 1, "end": 60_750, "freeze_illum": True},
    {"name": "joint_m1", "m": 1.0, "start": 60_751, "end": 97_195, "freeze_illum": False},
    {"name": "harden_m2", "m": 2.0, "start": 97_196, "end": 101_245, "freeze_illum": False},
    {"name": "harden_m3", "m": 3.0, "start": 101_246, "end": 105_295, "freeze_illum": False},
    {"name": "harden_m4", "m": 4.0, "start": 105_296, "end": 109_345, "freeze_illum": False},
    {"name": "harden_m5", "m": 5.0, "start": 109_346, "end": 113_395, "freeze_illum": False},
    {"name": "harden_m6", "m": 6.0, "start": 113_396, "end": 117_445, "freeze_illum": False},
    {"name": "harden_m7", "m": 7.0, "start": 117_446, "end": 121_495, "freeze_illum": False},
    {"name": "harden_m8", "m": 8.0, "start": 121_496, "end": 121_500, "freeze_illum": False},
]


def udith_legacy_phases() -> list[dict]:
    """Full 121,500-step accumulating-m schedule (five updates per legacy epoch).

    The five-step ``m=8`` phase is faithful: historical code first reaches m=8
    at legacy epoch 24,300, its final epoch. Do not extend it.
    """
    phases = [
        {
            "name": "inverse_warmup_m1",
            "m": 1.0,
            "steps": 60_750,
            "freeze_illum": True,
        },
        {
            "name": "joint_m1",
            "m": 1.0,
            "steps": 36_445,
            "freeze_illum": False,
        },
        {
            "name": "harden_m2",
            "m": 2.0,
            "steps": 4_050,
            "freeze_illum": False,
        },
        {
            "name": "harden_m3",
            "m": 3.0,
            "steps": 4_050,
            "freeze_illum": False,
        },
        {
            "name": "harden_m4",
            "m": 4.0,
            "steps": 4_050,
            "freeze_illum": False,
        },
        {
            "name": "harden_m5",
            "m": 5.0,
            "steps": 4_050,
            "freeze_illum": False,
        },
        {
            "name": "harden_m6",
            "m": 6.0,
            "steps": 4_050,
            "freeze_illum": False,
        },
        {
            "name": "harden_m7",
            "m": 7.0,
            "steps": 4_050,
            "freeze_illum": False,
        },
        {
            "name": "harden_m8",
            "m": 8.0,
            "steps": 5,
            "freeze_illum": False,
        },
    ]
    assert sum(p["steps"] for p in phases) == 121_500
    assert sum(p["steps"] for p in phases if not p["freeze_illum"]) == 60_750
    return phases


def udith_legacy_phases_smoke() -> list[dict]:
    """Non-scientific 1–2 step stand-in that still visits every named phase."""
    return [
        {"name": "inverse_warmup_m1", "m": 1.0, "steps": 2, "freeze_illum": True},
        {"name": "joint_m1", "m": 1.0, "steps": 2, "freeze_illum": False},
        {"name": "harden_m2", "m": 2.0, "steps": 1, "freeze_illum": False},
        {"name": "harden_m3", "m": 3.0, "steps": 1, "freeze_illum": False},
        {"name": "harden_m4", "m": 4.0, "steps": 1, "freeze_illum": False},
        {"name": "harden_m5", "m": 5.0, "steps": 1, "freeze_illum": False},
        {"name": "harden_m6", "m": 6.0, "steps": 1, "freeze_illum": False},
        {"name": "harden_m7", "m": 7.0, "steps": 1, "freeze_illum": False},
        {"name": "harden_m8", "m": 8.0, "steps": 1, "freeze_illum": False},
    ]


def phase_global_bounds(phases: list[dict]) -> list[dict[str, Any]]:
    """Inclusive 1-indexed [start, end] global-step bounds for each phase."""
    bounds: list[dict[str, Any]] = []
    cursor = 1
    for phase in phases:
        steps = int(phase["steps"])
        end = cursor + steps - 1
        bounds.append(
            {
                "name": phase["name"],
                "m": float(phase["m"]),
                "start": cursor,
                "end": end,
                "steps": steps,
                "freeze_illum": bool(phase["freeze_illum"]),
            }
        )
        cursor = end + 1
    return bounds


def m_sequence(phases: list[dict]) -> list[float]:
    """Unique m values in the order they first appear."""
    out: list[float] = []
    for phase in phases:
        m = float(phase["m"])
        if not out or m != out[-1]:
            out.append(m)
    return out


def schedule_provenance() -> dict[str, Any]:
    """JSON-serializable record of the epoch→step conversion."""
    phases = udith_legacy_phases()
    return {
        "source_repository": LEGACY_REPO,
        "source_commit": LEGACY_COMMIT,
        "source_notebook": LEGACY_NOTEBOOK,
        "source_scheduler": LEGACY_SCHEDULER,
        "scheduler_semantics": (
            "inc_m_class accumulates m (m += 1) after epoch_threshold on epoch_steps "
            "cadence; it does NOT reset m to 1. Contrast with SigmoidSchedule.step()."
        ),
        "legacy_epoch_constants": {
            "epochs": LEGACY_TOTAL_EPOCHS,
            "train_H_after_epoch": LEGACY_TRAIN_H_AFTER_EPOCH,
            "epoch_threshold": LEGACY_M_THRESHOLD_EPOCH,
            "epoch_steps": LEGACY_M_STEP_EPOCHS,
            "notebook_excerpt": {
                "epochs": "300 * 81",
                "train_H_iter": "int(epoch > 150 * 81)",
                "m_inc_proc": "inc_m_class(epoch_threshold=230 * 81, epoch_steps=10 * 81)",
            },
        },
        "five_batches_per_epoch_derivation": {
            "train_images": LEGACY_TRAIN_IMAGES,
            "batch_size": LEGACY_BATCH_SIZE,
            "drop_last": LEGACY_DROP_LAST,
            "updates_per_legacy_epoch": UPDATES_PER_LEGACY_EPOCH,
            "formula": "floor(168 / 32) = 5",
        },
        "converted_step_counts": {
            "total_steps": TOTAL_STEPS,
            "warmup_frozen_m1": WARMUP_STEPS,
            "trainable_steps": JOINT_TRAINABLE_STEPS,
            "joint_m1_steps": 36_445,
            "harden_steps_per_m_2_through_7": M_STEP_STEPS,
            "harden_m8_steps": 5,
            "first_m_increment_legacy_epoch": 19_440,
            "derivation_note": (
                "inc_m_class uses epoch > epoch_threshold (18,630) and a cadence "
                "of 810 epochs, so m ticks at 19440, 20250, ..., 24300 "
                "(seven increments: m=1→8). Each of m=2..7 lasts 810 epochs = "
                "4,050 steps; m=8 is only the final legacy epoch (5 steps)."
            ),
        },
        "phases": phases,
        "global_step_bounds": phase_global_bounds(phases),
        "m_sequence": m_sequence(phases),
        "not_a_literal_patchmnist_epoch_run": True,
        "uses_sigmoid_schedule_step": False,
    }
