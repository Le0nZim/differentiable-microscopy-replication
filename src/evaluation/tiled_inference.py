"""Simulated tiled acquisition and reconstruction.

Every model call reacquires a specimen tile through its optical encoder. Overlap
therefore adds measurements and changes the acquisition, not just the stitching.
Use non-overlapping tiles for the nominal compression; report the complete count
from tiled_acquisition_budget for overlap diagnostics.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import math


def tiled_acquisition_budget(height: int, width: int, tile: int, downscale: int,
                             num_patterns: int, overlap: int = 0) -> dict:
    if min(height, width, tile, downscale, num_patterns) < 1 or not 0 <= overlap < tile:
        raise ValueError("Invalid tiled acquisition geometry")
    if tile % downscale:
        raise ValueError("Tile must contain complete detector bins")
    stride = tile - overlap
    nh = max(1, math.ceil((height - tile) / stride) + 1)
    nw = max(1, math.ceil((width - tile) / stride) + 1)
    values = nh * nw * num_patterns * (tile // downscale)**2
    return {"acquired_tiles": nh * nw, "scalar_measurements": values,
            "nominal_compression_per_tile": downscale**2 / num_patterns,
            "effective_field_compression": height * width / values,
            "overlap_pixels": overlap,
            "note": "Counts repeated acquisitions and padded detector values; not decoder-only overlap."}


def _hann_2d(tile: int, device, dtype) -> torch.Tensor:
    """Separable 2-D Hann window in (0, 1], shape [tile, tile]."""
    w = torch.hann_window(tile, periodic=False, device=device, dtype=dtype)
    # Avoid exact zeros at the border so the overlap-add normalization is stable.
    w = w.clamp_min(1e-3)
    return torch.outer(w, w)


@torch.no_grad()
def naive_tiled_recon(model, field: torch.Tensor, device, eval_m: float, tile: int) -> np.ndarray:
    """Non-overlapping tiling (produces visible seams). field: [1,1,H,W]."""
    model.eval()
    H, W = field.shape[-2:]
    padH, padW = (tile - H % tile) % tile, (tile - W % tile) % tile
    f = F.pad(field, (0, padW, 0, padH), mode="reflect")
    Hp, Wp = f.shape[-2:]
    out = torch.zeros(1, 1, Hp, Wp)
    bs = 8 if tile >= 256 else 64
    tiles, coords = [], []
    for i in range(0, Hp, tile):
        for j in range(0, Wp, tile):
            tiles.append(f[:, :, i:i + tile, j:j + tile])
            coords.append((i, j))
    batch = torch.cat(tiles, 0)
    recons = []
    for k in range(0, batch.shape[0], bs):
        rec = model(batch[k:k + bs].to(device), sigmoid_m=eval_m, apply_noise=False)["x_recon"].float().clamp(0, 1)
        recons.append(rec.cpu())
    recons = torch.cat(recons, 0)
    for (i, j), r in zip(coords, recons):
        out[:, :, i:i + tile, j:j + tile] = r
    return out[0, 0, :H, :W].numpy()


@torch.no_grad()
def overlap_tiled_recon(
    model,
    field: torch.Tensor,
    device,
    eval_m: float,
    tile: int,
    overlap: int | None = None,
) -> np.ndarray:
    """Reacquire overlapping tiles and blend; this increases measurements.

    Args:
        field: [1, 1, H, W] normalized target field.
        tile: model patch size (must match training size).
        overlap: pixels of overlap between adjacent tiles. Defaults to tile//4
            (>= 16 px for a 64-px tile, as recommended).
    """
    model.eval()
    if overlap is None:
        overlap = max(16, tile // 4)
    overlap = min(overlap, tile - 1)
    stride = tile - overlap

    H, W = field.shape[-2:]
    # Pad so an integer number of strided tiles covers the field, plus a tile margin
    # on the near edge, using reflect padding (never zero padding).
    padH = (stride - (H - tile) % stride) % stride if H > tile else tile - H
    padW = (stride - (W - tile) % stride) % stride if W > tile else tile - W
    f = F.pad(field, (0, max(0, padW), 0, max(0, padH)), mode="reflect")
    Hp, Wp = f.shape[-2:]

    win = _hann_2d(tile, torch.device("cpu"), torch.float32)
    acc = torch.zeros(1, 1, Hp, Wp)
    wsum = torch.zeros(1, 1, Hp, Wp)

    rows = list(range(0, max(1, Hp - tile + 1), stride))
    cols = list(range(0, max(1, Wp - tile + 1), stride))
    if rows[-1] != Hp - tile:
        rows.append(Hp - tile)
    if cols[-1] != Wp - tile:
        cols.append(Wp - tile)

    tiles, coords = [], []
    for i in rows:
        for j in cols:
            tiles.append(f[:, :, i:i + tile, j:j + tile])
            coords.append((i, j))
    batch = torch.cat(tiles, 0)
    bs = 8 if tile >= 256 else 64
    recons = []
    for k in range(0, batch.shape[0], bs):
        rec = model(batch[k:k + bs].to(device), sigmoid_m=eval_m, apply_noise=False)["x_recon"].float().clamp(0, 1)
        recons.append(rec.cpu())
    recons = torch.cat(recons, 0)
    for (i, j), r in zip(coords, recons):
        acc[:, :, i:i + tile, j:j + tile] += r[0] * win
        wsum[:, :, i:i + tile, j:j + tile] += win
    out = acc / wsum.clamp_min(1e-6)
    return out[0, 0, :H, :W].numpy()
