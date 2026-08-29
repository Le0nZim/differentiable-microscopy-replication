"""Save illumination-pattern diagnostics after learnable-pattern runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image


def save_pattern_inspection(
    patterns: torch.Tensor,
    output_dir: str | Path,
    *,
    prefix: str = "H_t",
    sigmoid_m: float | None = None,
) -> dict[str, str]:
    """Save raw, binarized, mean, FFT magnitude, and histogram for H_t.

    Args:
        patterns: [T, 1, H, W] in [0, 1].
    """
    output_dir = Path(output_dir) / "learned_patterns"
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns_cpu = patterns.detach().cpu().float()
    binarized = (patterns_cpu > 0.5).float()
    mean_pattern = patterns_cpu.mean(dim=0, keepdim=True)

    raw_path = output_dir / f"{prefix}_raw.png"
    bin_path = output_dir / f"{prefix}_binarized.png"
    mean_path = output_dir / f"{prefix}_mean.png"
    fft_path = output_dir / f"{prefix}_fft_magnitude.png"
    hist_path = output_dir / f"{prefix}_histogram.png"
    stats_path = output_dir / f"{prefix}_stats.json"

    nrow = max(1, int(patterns_cpu.shape[0] ** 0.5))
    save_image(patterns_cpu, raw_path, nrow=nrow)
    save_image(binarized, bin_path, nrow=nrow)
    save_image(mean_pattern, mean_path, nrow=1)

    fft_mag = torch.fft.fftshift(torch.fft.fft2(patterns_cpu), dim=(-2, -1)).abs()
    fft_mag = fft_mag / fft_mag.max().clamp_min(1e-8)
    save_image(fft_mag, fft_path, nrow=nrow)

    values = patterns_cpu.reshape(-1).numpy()
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(values, bins=50, range=(0.0, 1.0))
    axis.set_xlabel("H_t value")
    axis.set_ylabel("count")
    axis.set_title(f"{prefix} value histogram")
    figure.tight_layout()
    figure.savefig(hist_path, dpi=150)
    plt.close(figure)

    near_binary = ((patterns_cpu < 0.1) | (patterns_cpu > 0.9)).float().mean()
    per_pattern_mean = patterns_cpu.mean(dim=(2, 3)).squeeze(1)
    stats = {
        "min": float(patterns_cpu.min().item()),
        "max": float(patterns_cpu.max().item()),
        "mean": float(patterns_cpu.mean().item()),
        "binary_fraction": float(near_binary.item()),
        "per_pattern_mean": [float(v) for v in per_pattern_mean.tolist()],
        "patterns_differ": bool(torch.std(per_pattern_mean) > 1e-4),
        "sigmoid_m": sigmoid_m,
    }
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    torch.save(patterns_cpu, output_dir / f"{prefix}_raw.pt")
    torch.save(binarized, output_dir / f"{prefix}_binarized.pt")
    torch.save(mean_pattern, output_dir / f"{prefix}_mean.pt")
    torch.save(fft_mag, output_dir / f"{prefix}_fft_magnitude.pt")

    return {
        "raw": str(raw_path),
        "binarized": str(bin_path),
        "mean": str(mean_path),
        "fft_magnitude": str(fft_path),
        "histogram": str(hist_path),
        "stats": str(stats_path),
    }
