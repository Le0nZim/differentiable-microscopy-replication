"""BBBC022 Hoechst 33342 substitute U2OS-style dataset (not paper U2OS reproduction)."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

SplitName = Literal["train", "val", "test"]
PreprocessMode = Literal["paper_strict", "bbbc022_calibrated", "raw_normalized"]
MaskMode = Literal["threshold_closing", "trackmate"]

_WELL_SITE_RE = re.compile(r"IXMtest_([A-P]\d{2})_s(\d+)_", re.IGNORECASE)

# Process-level cache of full-image TrackMate masks keyed by (path, params) so the
# expensive contour tracing runs once per image even when the dataset is rebuilt
# for every train/val/test split and every experiment cell in a single process.
_TRACKMATE_CACHE: dict[tuple, torch.Tensor] = {}


@dataclass
class BBBC022HoechstConfig:
    """BBBC022 Hoechst substitute configuration."""

    data_root: str = "data/substitute_data"
    stack_glob: str = "**/*.tif"
    preprocessing_mode: PreprocessMode = "paper_strict"
    bias: float = 134.28
    clip_max: float = 500.0
    downscale_factor: float = 1.0
    background_percentile: float = 1.0
    clip_percentile: float = 99.9
    patch_size: int = 256
    num_train_images: int = 168
    num_val_images: int = 21
    num_test_images: int = 21
    seed: int = 42
    train_random_crops: bool = True
    random_flips: bool = True
    split_by_well: bool = True
    return_mask: bool = False
    mask_mode: MaskMode = "threshold_closing"
    mask_threshold: float = 0.3
    mask_closing_kernel: int = 10
    # TrackMate-style pseudo-GT (used when mask_mode == "trackmate"): threshold the
    # *raw* MIP intensity (no normalization), 4-connect, then simplify each region
    # contour (resample at ~mask_smooth_interval px, Douglas-Peucker eps).
    mask_raw_threshold: float = 506.0
    mask_smooth_interval: float = 2.0
    mask_dp_epsilon: float = 0.5

    @classmethod
    def from_dict(cls, data: dict) -> "BBBC022HoechstConfig":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


def parse_well_site(path: Path) -> tuple[str | None, int | None]:
    match = _WELL_SITE_RE.search(path.name)
    if not match:
        return None, None
    return match.group(1).upper(), int(match.group(2))


def discover_image_paths(data_root: Path, stack_glob: str) -> list[Path]:
    paths = sorted(data_root.glob(stack_glob)) if "**" in stack_glob else sorted(data_root.glob(stack_glob))
    if not paths:
        paths = sorted(data_root.rglob("*.tif"))
    return paths


def select_hoechst_paths(paths: list[Path]) -> list[Path]:
    """Confirm Hoechst-only channel pack (BBBC022 w1 folder)."""
    if not paths:
        return []
    parent_names = {p.parent.name.lower() for p in paths}
    if any("w1" in name or "hoechst" in name or "bbbc022" in name for name in parent_names):
        return paths
    # Single-channel pack assumed Hoechst when all TIFFs in BBBC022 substitute folder
    if all(p.suffix.lower() in {".tif", ".tiff"} for p in paths):
        return paths
    raise ValueError("Could not confirm Hoechst channel selection from paths/metadata")


def assign_split_paths(
    paths: list[Path],
    config: BBBC022HoechstConfig,
) -> dict[SplitName, list[Path]]:
    total_needed = config.num_train_images + config.num_val_images + config.num_test_images
    if len(paths) < total_needed:
        raise FileNotFoundError(f"Need {total_needed} images, found {len(paths)}")

    if config.split_by_well:
        by_well: dict[str, list[Path]] = {}
        for path in paths:
            well, site = parse_well_site(path)
            key = well or path.stem
            by_well.setdefault(key, []).append(path)
        wells = sorted(by_well.keys())
        gen = torch.Generator().manual_seed(config.seed)
        perm = torch.randperm(len(wells), generator=gen).tolist()
        shuffled_wells = [wells[i] for i in perm]
        train_wells = shuffled_wells[: config.num_train_images]
        val_wells = shuffled_wells[config.num_train_images : config.num_train_images + config.num_val_images]
        test_wells = shuffled_wells[
            config.num_train_images + config.num_val_images : total_needed
        ]

        def pick_one(well: str) -> Path:
            candidates = sorted(by_well[well], key=lambda p: parse_well_site(p)[1] or 0)
            return candidates[0]

        return {
            "train": [pick_one(w) for w in train_wells],
            "val": [pick_one(w) for w in val_wells],
            "test": [pick_one(w) for w in test_wells],
        }

    sorted_paths = sorted(paths)
    return {
        "train": sorted_paths[: config.num_train_images],
        "val": sorted_paths[config.num_train_images : config.num_train_images + config.num_val_images],
        "test": sorted_paths[config.num_train_images + config.num_val_images : total_needed],
    }


def load_tiff(path: Path) -> torch.Tensor:
    try:
        import tifffile
        import numpy as np
    except ImportError as exc:
        raise ImportError("tifffile required for BBBC022 loading") from exc
    array_np = np.asarray(tifffile.imread(path))
    if array_np.dtype.byteorder not in ("=", "|"):
        array_np = array_np.astype(array_np.dtype.newbyteorder("="))
    return torch.from_numpy(array_np.astype(np.float32))


def maximum_intensity_projection(stack: torch.Tensor) -> torch.Tensor:
    if stack.ndim == 2:
        return stack
    if stack.ndim == 3:
        return torch.amax(stack, dim=0)
    raise ValueError(f"Expected [H,W] or [Z,H,W], got {tuple(stack.shape)}")


def preprocess_image(
    image: torch.Tensor,
    mode: PreprocessMode,
    *,
    bias: float,
    clip_max: float,
    background_percentile: float,
    clip_percentile: float,
) -> torch.Tensor:
    img = maximum_intensity_projection(image).float()
    if mode == "paper_strict":
        img = img - bias
        img = torch.clamp(img, min=0.0, max=clip_max)
    elif mode == "bbbc022_calibrated":
        bg = torch.quantile(img.flatten(), background_percentile / 100.0)
        img = img - bg
        hi = torch.quantile(img.flatten(), clip_percentile / 100.0)
        img = torch.clamp(img, min=0.0, max=hi)
    elif mode == "raw_normalized":
        pass
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}")

    min_val = img.min()
    max_val = img.max()
    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = torch.zeros_like(img)
    return img.unsqueeze(0)


def pad_for_crop(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, h, w = image.shape
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    if pad_h == 0 and pad_w == 0:
        return image
    return F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0.0)


def make_pseudo_mask(image: torch.Tensor, threshold: float, closing_kernel: int) -> torch.Tensor:
    """Paper-inspired pseudo segmentation: threshold + morphological closing."""
    binary = (image >= threshold).float()
    if closing_kernel <= 1:
        return binary
    k = closing_kernel if closing_kernel % 2 == 1 else closing_kernel + 1
    pad = k // 2
    dilated = F.max_pool2d(binary, kernel_size=k, stride=1, padding=pad)
    closed = -F.max_pool2d(-dilated, kernel_size=k, stride=1, padding=pad)
    return closed


# --------------------------------------------------------------------------- #
# TrackMate-style thresholded pseudo-GT (raw intensity threshold + simplified
# contours). Pure NumPy/Pillow so it needs no extra deps. Classification uses the
# original pixel values (no normalization); regions are 4-connected; each region
# boundary is resampled at ~2 px then Douglas-Peucker simplified and filled.
# --------------------------------------------------------------------------- #
def _label_4connected(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected-component labelling with 4-connectivity. Labels start at 1."""
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    n = 0
    for y in range(h):
        row = binary[y]
        for x in range(w):
            if not row[x] or labels[y, x]:
                continue
            n += 1
            q = deque([(y, x)])
            labels[y, x] = n
            while q:
                cy, cx = q.popleft()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = n
                        q.append((ny, nx))
    return labels, n


def _extract_boundary_contour(mask: np.ndarray) -> np.ndarray | None:
    """Moore-neighbour trace of a single 4-connected blob. Returns Nx2 (x, y)."""
    pad = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    h, w = pad.shape
    ys, xs = np.where(pad)
    if len(xs) == 0:
        return None
    i0 = np.lexsort((xs, ys))[0]
    y0, x0 = int(ys[i0]), int(xs[i0])
    nbrs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

    def backtrack_dir(py, px, cy, cx):
        dy, dx = py - cy, px - cx
        for i, (ady, adx) in enumerate(nbrs):
            if (ady, adx) == (dy, dx):
                return i
        return 6

    contour = [(x0 - 1, y0 - 1)]
    cy, cx = y0, x0
    py, px = y0, x0 - 1
    start = (y0, x0)
    first_move = None
    max_steps = mask.size * 8
    for step in range(max_steps):
        bd = backtrack_dir(py, px, cy, cx)
        found = False
        for k in range(8):
            i = (bd + 1 + k) % 8
            dy, dx = nbrs[i]
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and pad[ny, nx]:
                contour.append((nx - 1, ny - 1))
                py, px = cy, cx
                cy, cx = ny, nx
                if first_move is None:
                    first_move = (cy, cx)
                found = True
                break
        if not found:
            break
        if step > 2 and len(contour) > 3 and (cy, cx) == start:
            break

    pts = np.asarray(contour, dtype=np.float64)
    if len(pts) < 3:
        return None
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    return pts


def _resample_contour(pts: np.ndarray, interval: float) -> np.ndarray:
    """Resample a polyline at ~`interval`-pixel arc-length spacing."""
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = seg > 1e-9
    if not np.any(keep):
        return pts[:1]
    pts2 = np.vstack([pts[0], pts[1:][keep]])
    seg = np.linalg.norm(np.diff(pts2, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < interval:
        return pts2
    n = max(int(np.floor(total / interval)), 1)
    targets = np.linspace(0.0, total, n + 1)
    out = []
    j = 0
    for t in targets:
        while j < len(cum) - 2 and cum[j + 1] < t:
            j += 1
        t0, t1 = cum[j], cum[j + 1]
        if t1 <= t0:
            out.append(pts2[j])
        else:
            a = (t - t0) / (t1 - t0)
            out.append((1 - a) * pts2[j] + a * pts2[j + 1])
    out = np.asarray(out, dtype=np.float64)
    if not np.allclose(out[0], out[-1]):
        out = np.vstack([out, out[0]])
    return out


def _douglas_peucker(pts: np.ndarray, epsilon: float) -> np.ndarray:
    """Douglas-Peucker polyline simplification (handles closed contours)."""
    if len(pts) <= 3:
        return pts
    closed = np.allclose(pts[0], pts[-1])
    work = pts[:-1] if closed else pts

    def _dp(points: np.ndarray) -> np.ndarray:
        if len(points) <= 2:
            return points
        start, end = points[0], points[-1]
        line = end - start
        denom = float(np.dot(line, line))
        if denom < 1e-12:
            dists = np.linalg.norm(points - start, axis=1)
        else:
            t = np.clip(((points - start) @ line) / denom, 0.0, 1.0)
            proj = start + t[:, None] * line
            dists = np.linalg.norm(points - proj, axis=1)
        idx = int(np.argmax(dists))
        if dists[idx] > epsilon:
            left = _dp(points[: idx + 1])
            right = _dp(points[idx:])
            return np.vstack([left[:-1], right])
        return np.vstack([start, end])

    simp = _dp(work)
    if closed and not np.allclose(simp[0], simp[-1]):
        simp = np.vstack([simp, simp[0]])
    return simp


def _fill_polygons(shape: tuple[int, int], polygons: list[np.ndarray]) -> np.ndarray:
    """Rasterize filled polygons into a binary mask (float {0,1})."""
    from PIL import Image, ImageDraw

    img = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        if poly is None or len(poly) < 3:
            continue
        xy = [(float(x), float(y)) for x, y in poly]
        draw.polygon(xy, outline=1, fill=1)
    return np.asarray(img, dtype=np.float32)


def make_trackmate_mask(
    intensity: np.ndarray,
    threshold: float,
    smooth_interval: float,
    dp_epsilon: float,
) -> np.ndarray:
    """TrackMate-style detection: threshold raw intensity, 4-connect, simplify
    each region contour, and fill. Returns a float {0,1} mask (H, W)."""
    fg = np.asarray(intensity, dtype=np.float64) > float(threshold)
    labels, n = _label_4connected(fg)
    polygons: list[np.ndarray] = []
    for lab in range(1, n + 1):
        region = labels == lab
        contour = _extract_boundary_contour(region)
        if contour is None or len(contour) < 3:
            continue
        smoothed = _resample_contour(contour, smooth_interval)
        simplified = _douglas_peucker(smoothed, dp_epsilon)
        polygons.append(simplified)
    return _fill_polygons(intensity.shape, polygons)


class BBBC022HoechstDataset(Dataset):
    """BBBC022 Hoechst substitute U2OS-style dataset."""

    def __init__(self, config: BBBC022HoechstConfig, split: SplitName = "train") -> None:
        self.config = config
        self.split = split
        self.data_root = Path(config.data_root)
        all_paths = select_hoechst_paths(discover_image_paths(self.data_root, config.stack_glob))
        split_paths = assign_split_paths(all_paths, config)
        self.paths = split_paths[split]
        self.images = [self._load_and_preprocess(p) for p in self.paths]
        # Precompute full-image TrackMate masks (once, cached) so per-item access is
        # just a crop. Only needed when masks are requested in trackmate mode.
        self.mask_full: list[torch.Tensor] | None = None
        if self.config.mask_mode == "trackmate" and self.config.return_mask:
            self.mask_full = [self._trackmate_full(p) for p in self.paths]

    @classmethod
    def from_dict(cls, data: dict, split: SplitName) -> "BBBC022HoechstDataset":
        return cls(BBBC022HoechstConfig.from_dict(data), split=split)

    def _load_and_preprocess(self, path: Path) -> torch.Tensor:
        raw = load_tiff(path)
        image = preprocess_image(
            raw,
            self.config.preprocessing_mode,
            bias=self.config.bias,
            clip_max=self.config.clip_max,
            background_percentile=self.config.background_percentile,
            clip_percentile=self.config.clip_percentile,
        )
        if self.config.downscale_factor != 1.0:
            image = F.interpolate(
                image.unsqueeze(0),
                scale_factor=1.0 / self.config.downscale_factor,
                mode="area",
            ).squeeze(0)
        return pad_for_crop(image, self.config.patch_size)

    def _load_raw_mip(self, path: Path) -> torch.Tensor:
        """Raw MIP intensity (NO normalization) aligned spatially to ``images``."""
        raw = load_tiff(path)
        image = maximum_intensity_projection(raw).float().unsqueeze(0)
        if self.config.downscale_factor != 1.0:
            image = F.interpolate(
                image.unsqueeze(0),
                scale_factor=1.0 / self.config.downscale_factor,
                mode="area",
            ).squeeze(0)
        return pad_for_crop(image, self.config.patch_size)[0]

    def _trackmate_full(self, path: Path) -> torch.Tensor:
        key = (
            str(Path(path).resolve()),
            float(self.config.mask_raw_threshold),
            float(self.config.mask_smooth_interval),
            float(self.config.mask_dp_epsilon),
            float(self.config.downscale_factor),
            int(self.config.patch_size),
        )
        cached = _TRACKMATE_CACHE.get(key)
        if cached is None:
            raw_full = self._load_raw_mip(path).cpu().numpy()
            mask = make_trackmate_mask(
                raw_full,
                self.config.mask_raw_threshold,
                self.config.mask_smooth_interval,
                self.config.mask_dp_epsilon,
            )
            cached = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))
            _TRACKMATE_CACHE[key] = cached
        return cached

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        _, height, width = image.shape
        patch_size = self.config.patch_size
        generator = torch.Generator()
        generator.manual_seed(
            self.config.seed + index + {"train": 0, "val": 10_000, "test": 20_000}[self.split]
        )
        if self.split == "train" and self.config.train_random_crops:
            top = int(torch.randint(0, height - patch_size + 1, (1,), generator=generator).item())
            left = int(torch.randint(0, width - patch_size + 1, (1,), generator=generator).item())
        else:
            top = max(0, (height - patch_size) // 2)
            left = max(0, (width - patch_size) // 2)
        patch = image[:, top : top + patch_size, left : left + patch_size]

        # For trackmate masks, crop the matching window from the precomputed
        # full-image mask so it stays pixel-aligned with the specimen crop.
        mask: torch.Tensor | None = None
        if self.config.return_mask and self.config.mask_mode == "trackmate" and self.mask_full is not None:
            mask = self.mask_full[index][top : top + patch_size, left : left + patch_size].unsqueeze(0)

        if self.split == "train" and self.config.random_flips:
            if torch.rand((), generator=generator).item() > 0.5:
                patch = torch.flip(patch, dims=[-1])
                if mask is not None:
                    mask = torch.flip(mask, dims=[-1])
            if torch.rand((), generator=generator).item() > 0.5:
                patch = torch.flip(patch, dims=[-2])
                if mask is not None:
                    mask = torch.flip(mask, dims=[-2])

        if self.config.return_mask:
            if mask is None:
                mask = make_pseudo_mask(patch, self.config.mask_threshold, self.config.mask_closing_kernel)
            return patch, mask.contiguous()
        return patch
