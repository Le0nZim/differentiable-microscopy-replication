"""Export the RAW "detector measurement" image used by the "Compression
bottleneck" panel as a plain PNG -- no card/border, no captions, no title, no
arrows, nothing added. Just the image content itself, for the user to place
and label directly in PowerPoint.

The other raw image (the high-resolution sample) already exists, unmodified,
as `components/panel_X_ground_truth_x16.png` -- reuse that file directly
rather than duplicating it here.

Self-contained (does not depend on the earlier, no-longer-present
`render_fig01_compression_bottleneck.py`): loads the same GT PNG and applies
the same block-average pixelation.

Usage (from `replication/`):
    .venv/bin/python scripts/render_fig01_compression_bottleneck_parts.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG01 = REPO_ROOT / "paper_ready_results/02_main_figures/fig01_differentiable_microscopy_schematic"
OUT_COMPONENTS = FIG01 / "components"
GT_PNG = OUT_COMPONENTS / "panel_X_ground_truth_x16.png"
PREFIX = "panel_compbottleneck"

DETECTOR_GRID = 10  # raw (10x10) detector-measurement pixelation
UPSCALE_TO = 512    # nearest-neighbor upscale so the pixelation is visible, no smoothing


def _load_sample() -> np.ndarray:
    if GT_PNG.exists():
        img = np.asarray(Image.open(GT_PNG).convert("RGB"), dtype=np.float32) / 255.0
        return img
    raise FileNotFoundError(f"Expected ground-truth sample at {GT_PNG}; run render_fig01_assets.py first.")


def _block_avg_rgb(rgb: np.ndarray, g: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    hs, ws = h // g, w // g
    rgb = rgb[: hs * g, : ws * g]
    return rgb.reshape(g, hs, g, ws, 3).mean(axis=(1, 3))


def main() -> None:
    OUT_COMPONENTS.mkdir(parents=True, exist_ok=True)
    sample = _load_sample()  # HxWx3 float RGB in [0,1] (same array as panel_X_ground_truth_x16.png)

    detector_tile = _block_avg_rgb(sample, DETECTOR_GRID)  # small (10x10x3), raw pixelation
    detector_img = Image.fromarray((np.clip(detector_tile, 0, 1) * 255).astype(np.uint8))
    detector_img = detector_img.resize((UPSCALE_TO, UPSCALE_TO), Image.NEAREST)
    detector_png = OUT_COMPONENTS / f"{PREFIX}_detector.png"
    detector_img.save(detector_png)
    print("wrote", detector_png.name, f"(raw grid {DETECTOR_GRID}x{DETECTOR_GRID}, "
          f"nearest-neighbor upscaled to {UPSCALE_TO}x{UPSCALE_TO}, no border/text)")


if __name__ == "__main__":
    main()
