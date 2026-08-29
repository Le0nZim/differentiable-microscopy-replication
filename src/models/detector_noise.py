"""Differentiable photodetector noise model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

NoiseMode = Literal["noise_free", "differentiable_poisson", "differentiable_poisson_plus_read"]
NoiseNormalization = Literal["legacy", "paper", "paper_v3"]


@dataclass
class DetectorNoiseConfig:
    """Configuration for detector noise simulation."""

    mode: NoiseMode = "noise_free"
    photon_count: float = 10_000.0
    gamma: float = 10.0
    sigma_read: float = 0.0
    apply_noise: bool = True
    noise_normalization: NoiseNormalization = "legacy"
    downscale_factor: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "DetectorNoiseConfig":
        payload = {key: data[key] for key in cls.__dataclass_fields__ if key in data}
        if "gamma_background" in data and "gamma" not in payload:
            payload["gamma"] = data["gamma_background"]
        return cls(**payload)


class DetectorNoise(nn.Module):
    """Apply optional Poisson and read noise to demagnified measurements."""

    def __init__(self, config: DetectorNoiseConfig) -> None:
        super().__init__()
        self.config = config

    @classmethod
    def from_dict(cls, data: dict) -> "DetectorNoise":
        return cls(DetectorNoiseConfig.from_dict(data))

    def forward(
        self,
        alpha_down: torch.Tensor,
        *,
        apply_noise: bool | None = None,
        poisson_noise: torch.Tensor | None = None,
        read_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add detector noise to demagnified measurements.

        Args:
            alpha_down: Noise-free measurements with shape [B, T, H_down, W_down].
            apply_noise: Override config.apply_noise when provided.
            poisson_noise: Optional fixed z ~ N(0, 1) for deterministic tests.
            read_noise: Optional fixed z_read ~ N(0, 1) for deterministic tests.

        Returns:
            y_down: Noisy (or noise-free) measurements with shape [B, T, H_down, W_down].
        """
        use_noise = self.config.apply_noise if apply_noise is None else apply_noise
        if not use_noise or self.config.mode == "noise_free":
            return alpha_down

        return self._apply_noise(alpha_down, poisson_noise=poisson_noise, read_noise=read_noise)

    def _apply_noise(
        self,
        alpha_down: torch.Tensor,
        *,
        poisson_noise: torch.Tensor | None,
        read_noise: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.config.noise_normalization == "paper":
            # AM-1 v2: alpha_norm = alpha_down / d² (over-normalizes signal to [0, 1]).
            return self._apply_noise_paper(
                alpha_down,
                poisson_noise=poisson_noise,
                read_noise=read_noise,
                alpha_divisor=float(self.config.downscale_factor**2),
            )
        if self.config.noise_normalization == "paper_v3":
            # AM-1 v3: alpha_norm = alpha_down (no division); see _apply_noise_paper docstring.
            return self._apply_noise_paper(
                alpha_down,
                poisson_noise=poisson_noise,
                read_noise=read_noise,
                alpha_divisor=1.0,
            )
        return self._apply_noise_legacy(
            alpha_down,
            poisson_noise=poisson_noise,
            read_noise=read_noise,
        )

    def _apply_noise_legacy(
        self,
        alpha_down: torch.Tensor,
        *,
        poisson_noise: torch.Tensor | None,
        read_noise: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pre-AM-1 behavior: alpha_down treated as already on the inverse-model scale."""
        photon_count = self.config.photon_count
        gamma = self.config.gamma

        if photon_count <= 0:
            raise ValueError(f"photon_count must be positive, got {photon_count}")

        alpha_norm = alpha_down
        alpha_scaled = alpha_norm / photon_count

        if poisson_noise is None:
            poisson_noise = torch.randn_like(alpha_down)
        else:
            if poisson_noise.shape != alpha_down.shape:
                raise ValueError("poisson_noise must match alpha_down shape")

        poisson_mean = alpha_norm + gamma / photon_count
        poisson_std = torch.sqrt(alpha_scaled + (gamma / (photon_count**2)))
        y_poiss = poisson_mean + poisson_std * poisson_noise

        if self.config.mode == "differentiable_poisson":
            return y_poiss

        if self.config.mode == "differentiable_poisson_plus_read":
            if read_noise is None:
                read_noise = torch.randn_like(alpha_down)
            else:
                if read_noise.shape != alpha_down.shape:
                    raise ValueError("read_noise must match alpha_down shape")
            return y_poiss + self.config.sigma_read * read_noise

        raise ValueError(f"Unsupported noise mode: {self.config.mode}")

    def _apply_noise_paper(
        self,
        alpha_down: torch.Tensor,
        *,
        poisson_noise: torch.Tensor | None,
        read_noise: torch.Tensor | None,
        alpha_divisor: float,
    ) -> torch.Tensor:
        """Paper supplement A.2.2 eqs. S5–S10: normalized Poisson + read noise.

        Given the normalized signal ``alpha_norm = alpha_down / alpha_divisor``::

            y_poiss_norm = alpha_norm + gamma/k
                           + sqrt(alpha_norm/k + gamma/k^2) * z_poiss   (eq. S7)
            y_read_norm  = (sigma_read / k) * z_read                    (eq. S9)
            y_norm       = y_poiss_norm + y_read_norm                   (eq. S10)

        with ``k`` the photon count and ``z_poiss``, ``z_read`` independent
        standard normals.

        Choice of ``alpha_divisor`` (this is the only difference between the two
        paper modes, and the crux of AM-1 / RR-1):

        * ``noise_normalization="paper"`` (AM-1 v2) uses ``alpha_divisor = d²``.
          This was based on the assumption ``alpha_norm ∈ [0, 1]``.
        * ``noise_normalization="paper_v3"`` (AM-1 v3, correct) uses
          ``alpha_divisor = 1`` — i.e. ``alpha_norm = alpha_down`` directly.

        Why ``paper_v3`` is the faithful reading of eq. S5. The paper writes
        ``alpha_down = k · alpha_down^norm`` where
        ``alpha_down^norm = ψ(X, H_t^norm)`` with the *normalized* excitation
        pattern ``H_t^norm ∈ [0, 1]`` (eq. S5) and ``ψ`` the sum-pooling forward
        operator (eq. 8). This repo's forward model already computes ``ψ`` with
        binary patterns ``∈ {0, 1} = H_t^norm`` and image ``∈ [0, 1]``, so its
        output **is** ``alpha_down^norm`` (range ``[0, d²]``) — it never carried
        the physical factor ``k``. Hence no further division is required. The
        paper constrains ``H_t^norm ∈ [0, 1]``, not ``alpha_down^norm``; with
        sum-pooling ``alpha_down^norm ∈ [0, d²]``. Dividing again by ``d²`` (v2)
        shrinks the signal to ``[0, 1]`` while the read term ``sigma_read/k`` is
        unchanged, so at low photon count (e.g. k=10, sigma_read=6 → 0.6) the
        read noise swamps the signal — the residual pc=10 read-noise spread.
        """
        photon_count = self.config.photon_count
        gamma = self.config.gamma

        if photon_count <= 0:
            raise ValueError(f"photon_count must be positive, got {photon_count}")
        if alpha_divisor <= 0:
            raise ValueError(f"alpha_divisor must be positive, got {alpha_divisor}")

        k = photon_count
        alpha_norm = alpha_down / alpha_divisor
        # Numerical safety before sqrt only; this does NOT clip the output noise
        # (forward output is already non-negative, so this is a no-op in practice).
        alpha_norm = alpha_norm.clamp_min(0.0)

        if poisson_noise is None:
            poisson_noise = torch.randn_like(alpha_down)
        else:
            if poisson_noise.shape != alpha_down.shape:
                raise ValueError("poisson_noise must match alpha_down shape")

        poisson_mean = alpha_norm + gamma / k
        poisson_std = torch.sqrt(alpha_norm / k + gamma / (k**2))
        y_poiss_norm = poisson_mean + poisson_std * poisson_noise

        if self.config.mode == "differentiable_poisson":
            return y_poiss_norm

        if self.config.mode == "differentiable_poisson_plus_read":
            if read_noise is None:
                read_noise = torch.randn_like(alpha_down)
            else:
                if read_noise.shape != alpha_down.shape:
                    raise ValueError("read_noise must match alpha_down shape")
            y_read_norm = (self.config.sigma_read / k) * read_noise
            return y_poiss_norm + y_read_norm

        raise ValueError(f"Unsupported noise mode: {self.config.mode}")
