"""Compression ratio sanity checks."""

from __future__ import annotations


def compression_ratio(downscale_factor: int, num_patterns: int) -> float:
    """compression = downscale_factor^2 / number_of_patterns."""
    return (downscale_factor**2) / num_patterns


def test_compression_ratio_examples():
    assert compression_ratio(8, 8) == 8.0
    assert compression_ratio(32, 16) == 64.0
    assert compression_ratio(16, 4) == 64.0
