"""U2OS cell dataset interface and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

SplitName = Literal["train", "val", "test"]


@dataclass
class U2OSConfig:
    """Configuration for U2OS preprocessing and loading.

    Preprocessing pipeline (paper / spec):
        1. Input stack shape: [Z, H, W] (expected 60 x 2304 x 2304).
        2. Maximum intensity projection over Z.
        3. Subtract camera bias (134.28).
        4. Clip at 500.
        5. Min-max normalize to [0, 1].
        6. Downscale by factor 63/20 (~731 x 731).
        7. Split: 168 train / 21 val / 21 test full images.
        8. Training: random 256x256 crops + random H/V flips.
        9. Val/test: fixed 256x256 crops.
    """

    data_root: str = "data/u2os"
    stack_glob: str = "**/*.tif"
    bias: float = 134.28
    clip_max: float = 500.0
    downscale_factor: float = 63.0 / 20.0
    patch_size: int = 256
    num_train_images: int = 168
    num_val_images: int = 21
    num_test_images: int = 21
    seed: int = 42
    train_random_crops: bool = True
    random_flips: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "U2OSConfig":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


class U2OSPreprocessor:
    """Documented U2OS preprocessing steps from the paper reproduction plan."""

    @staticmethod
    def maximum_intensity_projection(stack: torch.Tensor) -> torch.Tensor:
        """Project a Z-stack to a single image. stack: [Z, H, W] -> [H, W]."""
        if stack.ndim != 3:
            raise ValueError(f"Expected stack shape [Z, H, W], got {tuple(stack.shape)}")
        return torch.amax(stack, dim=0)

    @staticmethod
    def preprocess_stack(
        stack: torch.Tensor,
        bias: float = 134.28,
        clip_max: float = 500.0,
        downscale_factor: float = 63.0 / 20.0,
    ) -> torch.Tensor:
        """Apply the full U2OS preprocessing chain. Returns [1, H', W'] in [0, 1]."""
        if stack.ndim == 2:
            image = stack.float()
        elif stack.ndim == 3:
            image = U2OSPreprocessor.maximum_intensity_projection(stack).float()
        else:
            raise ValueError(f"Expected stack shape [H, W] or [Z, H, W], got {tuple(stack.shape)}")

        image = image - bias
        image = torch.clamp(image, min=0.0, max=clip_max)

        min_val = image.min()
        max_val = image.max()
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = torch.zeros_like(image)

        image = image.unsqueeze(0).unsqueeze(0)
        if downscale_factor != 1.0:
            scale = 1.0 / downscale_factor
            image = F.interpolate(image, scale_factor=scale, mode="area")
        return image.squeeze(0)

    @staticmethod
    def crop_patch(
        image: torch.Tensor,
        patch_size: int,
        top: int,
        left: int,
    ) -> torch.Tensor:
        """Extract a fixed crop. image: [1, H, W] -> [1, patch_size, patch_size]."""
        return image[:, top : top + patch_size, left : left + patch_size]


class U2OSDataset(Dataset):
    """Placeholder U2OS loader with documented preprocessing.

    This class expects preprocessed or raw stack files under ``data_root``.
    Until the full U2OS release files are available, use ``U2OSPreprocessor``
    directly on stacks or raise a clear error from ``__init__``.
    """

    def __init__(self, config: U2OSConfig, split: SplitName = "train") -> None:
        self.config = config
        self.split = split
        self.data_root = Path(config.data_root)
        self.images = self._load_split_images()

    @classmethod
    def from_dict(cls, data: dict, split: SplitName) -> "U2OSDataset":
        return cls(U2OSConfig.from_dict(data), split=split)

    def _expected_count(self) -> int:
        if self.split == "train":
            return self.config.num_train_images
        if self.split == "val":
            return self.config.num_val_images
        if self.split == "test":
            return self.config.num_test_images
        raise ValueError(f"Unsupported split: {self.split}")

    def _discover_stack_paths(self) -> list[Path]:
        if "**" in self.config.stack_glob:
            paths = sorted(self.data_root.glob(self.config.stack_glob))
        else:
            paths = sorted(self.data_root.glob(self.config.stack_glob))
        if not paths:
            paths = sorted(self.data_root.rglob("*.tif"))
        return paths

    def _split_path_ranges(self, stack_paths: list[Path]) -> tuple[int, int]:
        train_count = self.config.num_train_images
        val_count = self.config.num_val_images
        test_count = self.config.num_test_images
        total_needed = train_count + val_count + test_count

        if len(stack_paths) < total_needed:
            raise FileNotFoundError(
                f"Expected at least {total_needed} U2OS images total, found {len(stack_paths)}"
            )

        if self.split == "train":
            return 0, train_count
        if self.split == "val":
            return train_count, train_count + val_count
        if self.split == "test":
            return train_count + val_count, total_needed
        raise ValueError(f"Unsupported split: {self.split}")

    def _load_split_images(self) -> list[torch.Tensor]:
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"U2OS data directory not found: {self.data_root}. "
                "Place U2OS stacks under this path before training."
            )

        stack_paths = self._discover_stack_paths()
        if not stack_paths:
            raise FileNotFoundError(
                f"No U2OS stack files matching {self.config.stack_glob!r} in {self.data_root}"
            )

        start, end = self._split_path_ranges(stack_paths)
        selected_paths = stack_paths[start:end]

        images: list[torch.Tensor] = []
        for path in selected_paths:
            stack = self._load_stack(path)
            images.append(
                U2OSPreprocessor.preprocess_stack(
                    stack,
                    bias=self.config.bias,
                    clip_max=self.config.clip_max,
                    downscale_factor=self.config.downscale_factor,
                )
            )
        return images

    def _load_stack(self, path: Path) -> torch.Tensor:
        try:
            import tifffile
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "Reading U2OS TIFF stacks requires the optional 'tifffile' package."
            ) from exc

        array_np = np.asarray(tifffile.imread(path))
        if array_np.dtype.byteorder not in ("=", "|"):
            array_np = array_np.astype(array_np.dtype.newbyteorder("="))
        stack = torch.from_numpy(array_np.astype(np.float32))
        if stack.ndim == 2:
            return stack
        if stack.ndim == 3:
            return stack
        raise ValueError(f"Unsupported TIFF shape {tuple(stack.shape)} in {path}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = self.images[index]
        _, height, width = image.shape
        patch_size = self.config.patch_size

        if height < patch_size or width < patch_size:
            raise ValueError(
                f"Image shape {tuple(image.shape)} is smaller than patch_size={patch_size}"
            )

        generator = torch.Generator()
        generator.manual_seed(self.config.seed + index + {"train": 0, "val": 10_000, "test": 20_000}[self.split])

        if self.split == "train" and self.config.train_random_crops:
            top = int(torch.randint(0, height - patch_size + 1, (1,), generator=generator).item())
            left = int(torch.randint(0, width - patch_size + 1, (1,), generator=generator).item())
        else:
            top = (height - patch_size) // 2
            left = (width - patch_size) // 2

        patch = U2OSPreprocessor.crop_patch(image, patch_size, top, left)

        if self.split == "train" and self.config.random_flips:
            if torch.rand((), generator=generator).item() > 0.5:
                patch = torch.flip(patch, dims=[-1])
            if torch.rand((), generator=generator).item() > 0.5:
                patch = torch.flip(patch, dims=[-2])

        return patch
