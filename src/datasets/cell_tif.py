"""Sanity-check dataset built from the local cell.tif file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from datasets.u2os import U2OSPreprocessor


@dataclass
class CellTifConfig:
    """Configuration for cell.tif sanity-check patches."""

    image_path: str = "data/cell.tif"
    bias: float = 134.28
    clip_max: float = 500.0
    downscale_factor: float = 63.0 / 20.0
    patch_size: int = 128
    num_patches: int = 8
    seed: int = 42

    @classmethod
    def from_dict(cls, data: dict) -> "CellTifConfig":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


def load_cell_tif_image(path: str | Path) -> torch.Tensor:
    """Load a TIFF image as a float tensor. Returns shape [1, H, W]."""
    path = Path(path)
    try:
        import tifffile
        import numpy as np

        array = tifffile.imread(path)
        array_np = np.asarray(array)
        if array_np.dtype.byteorder not in ("=", "|"):
            array_np = array_np.astype(array_np.dtype.newbyteorder("="))
        array = torch.from_numpy(array_np.astype(np.float32))
    except ImportError:
        from PIL import Image
        import numpy as np

        array_np = np.array(Image.open(path))
        if array_np.dtype.byteorder not in ("=", "|"):
            array_np = array_np.astype(array_np.dtype.newbyteorder("="))
        array = torch.from_numpy(array_np.astype(np.float32))

    if array.ndim == 3:
        return U2OSPreprocessor.maximum_intensity_projection(array.float()).unsqueeze(0)
    if array.ndim == 2:
        return array.float().unsqueeze(0)

    raise ValueError(f"Unsupported cell.tif shape: {tuple(array.shape)}")


def preprocess_cell_tif_image(
    image: torch.Tensor,
    *,
    bias: float,
    clip_max: float,
    downscale_factor: float,
) -> torch.Tensor:
    """Apply U2OS-style preprocessing to a single-channel image [1, H, W]."""
    if image.ndim == 2:
        image = image.unsqueeze(0)

    processed = image.float() - bias
    processed = torch.clamp(processed, min=0.0, max=clip_max)
    min_val = processed.min()
    max_val = processed.max()
    if max_val > min_val:
        processed = (processed - min_val) / (max_val - min_val)
    else:
        processed = torch.zeros_like(processed)

    processed = processed.unsqueeze(0)
    processed = torch.nn.functional.interpolate(
        processed,
        scale_factor=1.0 / downscale_factor,
        mode="area",
    )
    return processed.squeeze(0)


class CellTifDataset(Dataset):
    """Generate fixed random crops from ``data/cell.tif`` for preprocessing sanity checks."""

    def __init__(self, config: CellTifConfig | None = None) -> None:
        self.config = config or CellTifConfig()
        raw_image = load_cell_tif_image(self.config.image_path)
        self.image = preprocess_cell_tif_image(
            raw_image,
            bias=self.config.bias,
            clip_max=self.config.clip_max,
            downscale_factor=self.config.downscale_factor,
        )
        self._crop_origins = self._sample_crop_origins()

    def _sample_crop_origins(self) -> list[tuple[int, int]]:
        _, height, width = self.image.shape
        patch_size = self.config.patch_size
        if height < patch_size or width < patch_size:
            raise ValueError(
                f"Preprocessed cell.tif shape {tuple(self.image.shape)} is smaller than "
                f"patch_size={patch_size}"
            )

        generator = torch.Generator()
        generator.manual_seed(self.config.seed)
        origins: list[tuple[int, int]] = []
        for _ in range(self.config.num_patches):
            top = int(torch.randint(0, height - patch_size + 1, (1,), generator=generator).item())
            left = int(torch.randint(0, width - patch_size + 1, (1,), generator=generator).item())
            origins.append((top, left))
        return origins

    def __len__(self) -> int:
        return len(self._crop_origins)

    def __getitem__(self, index: int) -> torch.Tensor:
        top, left = self._crop_origins[index]
        return U2OSPreprocessor.crop_patch(self.image, self.config.patch_size, top, left)
