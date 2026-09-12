"""Explicit acquisition accounting; photon scale is not a sequence dose."""

from __future__ import annotations

import torch


def dose_multiplier(patterns: torch.Tensor, mode: str, sequence_dose: float) -> torch.Tensor:
    """Return a scalar exposure/power factor for mean incident sequence dose.

    Masks stay binary. This does not equalize detected photons for every
    specimen or impose a peak-power or photobleaching constraint.
    """
    if mode == "fixed_peak":
        return patterns.new_tensor(1.0)
    if mode != "equal_mean" or sequence_dose <= 0:
        raise ValueError("Use fixed_peak or equal_mean with positive sequence_dose")
    incident = patterns.sum(dim=0).mean()
    if not torch.isfinite(incident) or incident <= 0:
        raise ValueError("Cannot normalize the dose of an all-dark or nonfinite sequence")
    return sequence_dose / incident


@torch.no_grad()
def acquisition_metadata(patterns: torch.Tensor, *, downscale: int, photon_scale: float,
                         dose_mode: str = "fixed_peak", sequence_dose: float = 1.0) -> dict:
    t, _, h, w = patterns.shape
    if h % downscale or w % downscale:
        raise ValueError("Acquisition requires complete detector bins")
    multiplier = float(dose_multiplier(patterns, dose_mode, sequence_dose))
    incident = patterns.sum(dim=0) * photon_scale * multiplier
    detections = t * (h // downscale) * (w // downscale)
    return {
        "num_patterns": t, "physical_exposures": t,
        "detector_shape": [h // downscale, w // downscale],
        "scalar_measurements": detections, "compression": h * w / detections,
        "binary": bool(torch.all((patterns == 0) | (patterns == 1))),
        "mean_fill_factor": float(patterns.mean()), "dose_mode": dose_mode,
        "exposure_power_multiplier": multiplier,
        "incident_sequence_mean_photon_scale": float(incident.mean()),
        "incident_sequence_max_photon_scale": float(incident.max()),
        "peak_on_pixel_photon_scale": photon_scale * multiplier,
        "note": "Incident dose is in model photon-scale units; detected signal depends on the specimen. No calibrated acquisition time or peak-power cap is assumed.",
    }
