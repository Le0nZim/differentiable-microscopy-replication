"""Excitation pattern generation for compressive fluorescence microscopy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

PatternMode = Literal[
    "learnable_frequency",
    "learnable_spatial",
    "random_fixed",
    "uniform_all_ones",
    "hadamard_fixed",
]


@dataclass
class PatternGeneratorConfig:
    """Configuration for illumination pattern generation."""

    mode: PatternMode = "learnable_frequency"
    num_patterns: int = 8
    height: int = 256
    width: int = 256
    sigmoid_m: float = 1.0
    random_fixed_m: float = 10.0
    seed: int = 42
    # Optical super-pixel size: the learnable pattern is generated on a coarse
    # (height // superpixel_factor, width // superpixel_factor) grid and block-
    # upsampled (nearest) to full resolution. This reproduces the paper's coarse
    # binary illumination patterns: detail finer than one demagnification super-
    # pixel is washed out by sum-pooling, so the effective DOF is the super-pixel
    # grid. superpixel_factor=1 (default) keeps full per-pixel patterns.
    superpixel_factor: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "PatternGeneratorConfig":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


class SigmoidSchedule:
    """Custom sigmoid sharpness schedule (paper Algorithm 1)."""

    def __init__(
        self,
        epoch_baseline: int = 12150,
        epoch_cutoff: int = 18630,
        epoch_step: int = 810,
        m_init: float = 1.0,
    ) -> None:
        self.epoch_baseline = epoch_baseline
        self.epoch_cutoff = epoch_cutoff
        self.epoch_step = epoch_step
        self.m_init = m_init
        self._m = m_init

    @classmethod
    def from_dict(cls, data: dict) -> "SigmoidSchedule":
        return cls(
            epoch_baseline=data.get("epoch_baseline", 12150),
            epoch_cutoff=data.get("epoch_cutoff", 18630),
            epoch_step=data.get("epoch_step", 810),
            m_init=data.get("m_init", 1.0),
        )

    def reset(self) -> None:
        """Reset sharpness to its initial value."""
        self._m = self.m_init

    def should_freeze_patterns(self, epoch: int) -> bool:
        """Return True during Stage A (inverse model only)."""
        return epoch <= self.epoch_baseline

    def step(self, epoch: int) -> float:
        """Update and return sigmoid sharpness m for the given epoch."""
        if epoch <= self.epoch_baseline:
            return self._m

        if epoch > self.epoch_cutoff and epoch % self.epoch_step == 0:
            self._m += 1.0
        else:
            self._m = self.m_init
        return self._m

    def get_m(self) -> float:
        """Return the current sharpness value."""
        return self._m


def _generate_hadamard_matrix(size: int) -> torch.Tensor:
    """Build a Sylvester Hadamard matrix of shape [size, size]."""
    if size & (size - 1) != 0:
        raise ValueError(f"Hadamard size must be a power of 2, got {size}")

    matrix = torch.ones(1, 1)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            [
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1),
            ],
            dim=0,
        )
    return matrix


def _hadamard_patterns(num_patterns: int, height: int, width: int) -> torch.Tensor:
    """Build fixed Hadamard patterns mapped to [0, 1]. Shape: [T, 1, H, W]."""
    num_pixels = height * width
    hadamard_size = 1 << (num_pixels - 1).bit_length()
    hadamard = _generate_hadamard_matrix(hadamard_size)

    patterns = []
    for pattern_idx in range(num_patterns):
        row_idx = pattern_idx % hadamard_size
        row = hadamard[row_idx, :num_pixels]
        pattern = (row.reshape(height, width) + 1.0) * 0.5
        patterns.append(pattern)

    # [T, 1, H, W]
    return torch.stack(patterns, dim=0).unsqueeze(1)


def _random_fixed_patterns(
    num_patterns: int,
    height: int,
    width: int,
    sigmoid_m: float,
    seed: int,
) -> torch.Tensor:
    """Build fixed pseudo-random patterns via sigmoid(m * tau_0). Shape: [T, 1, H, W]."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    tau_0 = torch.randn(num_patterns, 1, height, width, generator=generator)
    return torch.sigmoid(sigmoid_m * tau_0)


def _uniform_patterns(num_patterns: int, height: int, width: int) -> torch.Tensor:
    """Build all-ones patterns. Shape: [T, 1, H, W]."""
    return torch.ones(num_patterns, 1, height, width)


def _init_frequency_weights(
    num_patterns: int,
    height: int,
    width: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Initialize W = fft2(tau_0) with tau_0 ~ N(0, 1). Shape: [T, 1, H, W] complex."""
    tau_0 = torch.randn(num_patterns, 1, height, width, generator=generator)
    return torch.fft.fft2(tau_0)


class PatternGenerator(nn.Module):
    """Generate excitation patterns H_t with shape [T, 1, H, W] in [0, 1]."""

    def __init__(self, config: PatternGeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.mode = config.mode
        self.num_patterns = config.num_patterns
        self.height = config.height
        self.width = config.width
        self.sigmoid_m = config.sigmoid_m

        self.superpixel_factor = max(1, int(config.superpixel_factor))
        if config.height % self.superpixel_factor != 0 or config.width % self.superpixel_factor != 0:
            raise ValueError(
                f"height ({config.height}) and width ({config.width}) must be divisible "
                f"by superpixel_factor ({self.superpixel_factor})"
            )
        # patterns are generated at this (coarse) resolution, then block-upsampled
        self.gen_height = config.height // self.superpixel_factor
        self.gen_width = config.width // self.superpixel_factor

        generator = torch.Generator()
        generator.manual_seed(config.seed)

        if self.mode == "learnable_frequency":
            init_w = _init_frequency_weights(
                config.num_patterns, self.gen_height, self.gen_width, generator
            )
            self.W = nn.Parameter(init_w)
            self.tau = None
            self.register_buffer("fixed_patterns", None)
        elif self.mode == "learnable_spatial":
            init_tau = torch.randn(
                config.num_patterns, 1, self.gen_height, self.gen_width, generator=generator
            )
            self.tau = nn.Parameter(init_tau)
            self.W = None
            self.register_buffer("fixed_patterns", None)
        else:
            self.W = None
            self.tau = None
            fixed = self._build_fixed_patterns(generator)
            self.register_buffer("fixed_patterns", fixed)

    @classmethod
    def from_dict(cls, data: dict) -> "PatternGenerator":
        return cls(PatternGeneratorConfig.from_dict(data))

    def _build_fixed_patterns(self, generator: torch.Generator) -> torch.Tensor:
        if self.mode == "random_fixed":
            return _random_fixed_patterns(
                self.num_patterns,
                self.gen_height,
                self.gen_width,
                self.config.random_fixed_m,
                self.config.seed,
            )
        if self.mode == "uniform_all_ones":
            return _uniform_patterns(self.num_patterns, self.gen_height, self.gen_width)
        if self.mode == "hadamard_fixed":
            return _hadamard_patterns(self.num_patterns, self.gen_height, self.gen_width)
        raise ValueError(f"Unsupported fixed pattern mode: {self.mode}")

    def _upsample(self, patterns: torch.Tensor) -> torch.Tensor:
        """Block-upsample coarse patterns to full resolution (no-op if sp==1)."""
        if self.superpixel_factor == 1:
            return patterns
        return F.interpolate(patterns, size=(self.height, self.width), mode="nearest")

    def _spatial_tau(self) -> torch.Tensor:
        if self.mode == "learnable_frequency":
            assert self.W is not None
            # W: [T, 1, H, W] complex -> tau: [T, 1, H, W] real
            return torch.fft.ifft2(self.W).real
        if self.mode == "learnable_spatial":
            assert self.tau is not None
            return self.tau
        raise RuntimeError(f"Mode {self.mode} does not have learnable spatial parameters")

    def forward(self, sigmoid_m: float | None = None) -> torch.Tensor:
        """Return excitation patterns H_t with shape [T, 1, H, W].

        Patterns are generated on the (gen_height, gen_width) super-pixel grid and
        then block-upsampled to (height, width).
        """
        m = self.sigmoid_m if sigmoid_m is None else sigmoid_m

        if self.mode in {"learnable_frequency", "learnable_spatial"}:
            tau = self._spatial_tau()
            # tau: [T, 1, gh, gw] -> H_t: [T, 1, gh, gw]
            coarse = torch.sigmoid(m * tau)
        else:
            assert self.fixed_patterns is not None
            coarse = self.fixed_patterns

        return self._upsample(coarse)

    def patterns_are_learnable(self) -> bool:
        return self.mode in {"learnable_frequency", "learnable_spatial"}

    @torch.no_grad()
    def load_frequency_weights(self, weights: torch.Tensor) -> None:
        """Overwrite learnable-frequency weights W from a saved tensor."""
        if self.mode != "learnable_frequency" or self.W is None:
            raise ValueError("load_frequency_weights requires learnable_frequency mode")
        if weights.shape != self.W.shape:
            raise ValueError(f"Expected W shape {tuple(self.W.shape)}, got {tuple(weights.shape)}")
        self.W.copy_(weights.to(device=self.W.device, dtype=self.W.dtype))

    @torch.no_grad()
    def export_frequency_checkpoint(self, *, sigmoid_m: float) -> dict[str, torch.Tensor | float | int]:
        """Export W and H_t for paired-initialization experiments."""
        if self.W is None:
            raise ValueError("export_frequency_checkpoint requires learnable_frequency mode")
        patterns = self.forward(sigmoid_m=sigmoid_m).detach().cpu()
        return {
            "W": self.W.detach().cpu().clone(),
            "H_t": patterns,
            "sigmoid_m": sigmoid_m,
            "seed": self.config.seed,
            "num_patterns": self.num_patterns,
            "height": self.height,
            "width": self.width,
        }

    @classmethod
    def load_frequency_checkpoint(cls, path: str | Path) -> dict[str, torch.Tensor | float | int]:
        """Load a paired-init checkpoint written by export_frequency_checkpoint."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = {"W", "H_t", "sigmoid_m"}
        if not required.issubset(payload.keys()):
            raise ValueError(f"Pattern init checkpoint missing keys: {required - set(payload.keys())}")
        return payload
