"""Excitation pattern generation for compressive fluorescence microscopy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    # upsampled (nearest) to full resolution. This imposes coarse illumination.
    # It is a different sensing constraint, not a consequence
    # of demagnification: sub-bin pattern variation encodes sub-bin structure.
    # A mask constant within each detector bin gives only a scaled bin sum.
    superpixel_factor: int = 1
    hadamard_tile_size: int | None = None
    binarization: Literal["soft", "ste"] = "soft"

    @classmethod
    def from_dict(cls, data: dict) -> "PatternGeneratorConfig":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


class SigmoidSchedule:
    """Monotone sharpness, as in the original code and hardening description.

    Computed from the epoch rather than call count, so repeated calls and resume
    do not change the schedule. The separate Udith step protocol is untouched.
    """

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
        if epoch_step < 1 or epoch_baseline < 0 or epoch_cutoff < 0 or m_init <= 0:
            raise ValueError("Invalid sigmoid schedule")
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
        threshold = max(self.epoch_baseline, self.epoch_cutoff)
        increments = max(0, epoch // self.epoch_step - threshold // self.epoch_step)
        self._m = self.m_init + increments
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


def _hadamard_patterns(num_patterns: int, height: int, width: int,
                       tile_size: int | None = None) -> torch.Tensor:
    """Generate only requested Sylvester rows, using H[r,c]=(-1)^popcount(r&c).

    tile_size=d reproduces the original detector-bin tiling. None preserves
    historical image-wide ordering, without allocating an (H*W)^2 matrix.
    Binary rows are single exposures; no complementary subtraction is implied.
    """
    ph, pw = (height, width) if tile_size is None else (tile_size, tile_size)
    if min(ph, pw) < 1 or height % ph or width % pw:
        raise ValueError("Hadamard tiles must evenly divide the pattern")
    num_pixels = ph * pw
    hadamard_size = 1 << (num_pixels - 1).bit_length()
    if num_patterns > hadamard_size:
        raise ValueError("Requested more distinct Hadamard rows than the basis contains")
    columns = torch.arange(num_pixels, dtype=torch.int64)

    patterns = []
    for pattern_idx in range(num_patterns):
        bits = columns & pattern_idx
        parity = torch.zeros_like(columns)
        while torch.any(bits):
            parity ^= bits & 1
            bits >>= 1
        pattern = (1 - parity).float().reshape(ph, pw).repeat(height // ph, width // pw)
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

        if min(config.num_patterns, config.height, config.width, config.superpixel_factor) < 1:
            raise ValueError("Pattern dimensions, count and superpixel_factor must be positive")
        if config.binarization not in {"soft", "ste"}:
            raise ValueError("binarization must be soft or ste")

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
            return _hadamard_patterns(self.num_patterns, self.gen_height, self.gen_width,
                                      self.config.hadamard_tile_size)
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
        if m <= 0:
            raise ValueError("sigmoid_m must be positive")

        if self.mode in {"learnable_frequency", "learnable_spatial"}:
            tau = self._spatial_tau()
            # tau: [T, 1, gh, gw] -> H_t: [T, 1, gh, gw]
            coarse = torch.sigmoid(m * tau)
        else:
            assert self.fixed_patterns is not None
            coarse = self.fixed_patterns

        if self.config.binarization == "ste":
            binary = (coarse > 0.5).to(coarse.dtype)
            coarse = coarse + (binary - coarse).detach() if self.training else binary
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
