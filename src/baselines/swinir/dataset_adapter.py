"""Dataset adapter for SwinIR smoke / SR evaluation."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class SRImageFolderDataset(Dataset):
    """Load grayscale HR PNGs from canonical data/sr layout."""

    def __init__(
        self,
        root: str | Path,
        *,
        patch_size: int = 64,
        max_images: int | None = None,
        seed: int = 42,
        grayscale: bool = True,
        random_crops: bool = False,
    ) -> None:
        self.root = Path(root)
        self.patch_size = patch_size
        self.grayscale = grayscale
        self.random_crops = random_crops
        paths = sorted(self.root.glob("*.png")) + sorted(self.root.glob("*.jpg"))
        if max_images is not None:
            paths = paths[:max_images]
        if not paths:
            raise FileNotFoundError(f"No images in {self.root}")
        self.paths = paths
        self.seed = seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index]
        img = read_image(str(path)).float() / 255.0
        if self.grayscale and img.shape[0] == 3:
            img = img.mean(dim=0, keepdim=True)
        _, h, w = img.shape
        ps = self.patch_size
        gen = None if self.random_crops else torch.Generator().manual_seed(self.seed + index)
        if h >= ps and w >= ps:
            top = int(torch.randint(0, h - ps + 1, (1,), generator=gen).item())
            left = int(torch.randint(0, w - ps + 1, (1,), generator=gen).item())
            img = img[:, top : top + ps, left : left + ps]
        else:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(ps, ps), mode="bilinear", align_corners=False
            ).squeeze(0)
        return img


def write_sr_manifest_csv(manifest_path: Path, rows: list[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
