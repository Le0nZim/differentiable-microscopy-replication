"""PatchMNIST dataset for compressive microscopy debugging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import InterpolationMode, Resize, ToTensor

SplitName = Literal["train", "val", "test"]


@dataclass
class PatchMNISTConfig:
    """Configuration for PatchMNIST generation."""

    data_root: str = "data/mnist"
    image_size: int = 256
    digit_size: int = 32
    grid_size: int = 20
    num_train: int = 3000
    num_val: int = 375
    num_test: int = 375
    seed: int = 42
    download: bool = True
    # When True, val and test draw from disjoint halves of the MNIST test
    # digit pool (same 10k file, no shared digit indices). Train is unchanged.
    disjoint_val_test: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "PatchMNISTConfig":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})

    @property
    def canvas_size(self) -> int:
        return self.grid_size * self.digit_size

    @property
    def max_crop_origin(self) -> int:
        return self.canvas_size - self.image_size


_MNIST_CACHE: dict[tuple[str, bool], torch.Tensor] = {}


def _load_mnist_tensors(
    data_root: str,
    train: bool,
    download: bool,
) -> torch.Tensor:
    """Load MNIST images as float tensors in [0, 1]. Shape: [N, 32, 32]."""
    cache_key = (str(data_root), train)
    cached = _MNIST_CACHE.get(cache_key)
    if cached is not None:
        return cached
    resize = Resize((32, 32), interpolation=InterpolationMode.BILINEAR, antialias=True)
    to_tensor = ToTensor()
    mnist = datasets.MNIST(
        root=data_root,
        train=train,
        download=download,
        transform=lambda image: to_tensor(resize(image)),
    )
    images = torch.stack([mnist[idx][0] for idx in range(len(mnist))], dim=0).squeeze(1)
    _MNIST_CACHE[cache_key] = images
    return images


def _build_tiled_canvas(
    digits: torch.Tensor,
    indices: torch.Tensor,
    grid_size: int,
    digit_size: int,
) -> torch.Tensor:
    """Tile selected digits into a grid canvas. Shape: [grid_size*digit_size, grid_size*digit_size]."""
    selected = digits[indices]
    grid = selected.view(grid_size, grid_size, digit_size, digit_size)
    return grid.permute(0, 2, 1, 3).contiguous().reshape(
        grid_size * digit_size, grid_size * digit_size
    )


def _extract_patch(
    canvas: torch.Tensor,
    patch_size: int,
    top: int,
    left: int,
) -> torch.Tensor:
    """Crop a square patch from a canvas. Shape: [1, patch_size, patch_size]."""
    patch = canvas[top : top + patch_size, left : left + patch_size]
    return patch.unsqueeze(0)


def _mnist_test_pools(config: PatchMNISTConfig, n_digits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition MNIST-test indices into disjoint val and test pools."""
    gen = torch.Generator().manual_seed(int(config.seed) + 10007)
    perm = torch.randperm(n_digits, generator=gen)
    n_val = n_digits // 2
    return perm[:n_val], perm[n_val:]


def generate_patchmnist_split(
    config: PatchMNISTConfig,
    split: SplitName,
) -> list[torch.Tensor]:
    """Generate PatchMNIST images for one split."""
    if split == "train":
        count = config.num_train
        mnist_train = True
        seed = config.seed
    elif split == "val":
        count = config.num_val
        mnist_train = False
        seed = config.seed + 1
    elif split == "test":
        count = config.num_test
        mnist_train = False
        seed = config.seed + 2
    else:
        raise ValueError(f"Unsupported split: {split}")

    digits = _load_mnist_tensors(config.data_root, train=mnist_train, download=config.download)
    generator = torch.Generator()
    generator.manual_seed(seed)
    index_pool: torch.Tensor | None = None
    if split != "train" and config.disjoint_val_test:
        val_pool, test_pool = _mnist_test_pools(config, digits.shape[0])
        index_pool = val_pool if split == "val" else test_pool

    patches: list[torch.Tensor] = []
    cells_per_canvas = config.grid_size * config.grid_size
    max_origin = config.max_crop_origin

    for _ in range(count):
        if index_pool is None:
            digit_indices = torch.randint(0, digits.shape[0], (cells_per_canvas,), generator=generator)
        else:
            local = torch.randint(0, int(index_pool.numel()), (cells_per_canvas,), generator=generator)
            digit_indices = index_pool[local]
        canvas = _build_tiled_canvas(
            digits,
            digit_indices,
            config.grid_size,
            config.digit_size,
        )
        top = int(torch.randint(0, max_origin + 1, (1,), generator=generator).item())
        left = int(torch.randint(0, max_origin + 1, (1,), generator=generator).item())
        patches.append(_extract_patch(canvas, config.image_size, top, left))

    return patches


class PatchMNISTDataset(Dataset):
    """PatchMNIST images with shape [1, H, W] in [0, 1]."""

    def __init__(self, config: PatchMNISTConfig, split: SplitName = "train") -> None:
        self.config = config
        self.split = split
        self.images = generate_patchmnist_split(config, split)

    @classmethod
    def from_dict(cls, data: dict, split: SplitName) -> "PatchMNISTDataset":
        return cls(PatchMNISTConfig.from_dict(data), split=split)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        # image: [1, H, W]
        return self.images[index]
