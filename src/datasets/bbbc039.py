"""BBBC039 manual nucleus annotations, with aligned image/foreground crops.

The released PNG red channel encodes nuclei. Its positive labels are united
for this binary semantic task; instance-separation metrics are not implied.
See https://bbbc.broadinstitute.org/BBBC039 and the linked decoding example.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .augmentation import training_crop_generator
from .bbbc022_hoechst import load_tiff
from .bbbc022_preproc_ablation import PreprocParams, preprocess_minimal_percentile


def load_manual_foreground(path):
    with Image.open(path) as im:
        labels = np.array(im)
    if labels.ndim == 3 and labels.shape[-1] in (3, 4):
        labels = labels[..., 0]  # Ignore opaque alpha; never OR the RGBA channels.
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Expected an integer annotation PNG: {path}")
    return torch.from_numpy(np.ascontiguousarray(labels > 0)).unsqueeze(0)


@dataclass
class BBBC039Config:
    split_path: str
    patch_size: int = 256
    seed: int = 42
    return_mask: bool = True
    train_random_crops: bool = True
    random_flips: bool = True
    q_low: float = .001
    q_high: float = .999


class BBBC039Dataset(Dataset):
    def __init__(self, config, split):
        self.config, self.split = config, split
        manifest = json.loads(Path(config.split_path).read_text())
        if manifest.get("label_source") != "BBBC039 manual nucleus annotations":
            raise ValueError("BBBC039 requires a prepared manual-annotation manifest")
        self.records = manifest["splits"][split]
        if not self.records:
            raise ValueError(f"Empty BBBC039 {split} split")
        self.paths = [Path(r["image"]) for r in self.records]
        params = PreprocParams(q_low=config.q_low, q_high=config.q_high)
        self.images, self.masks = [], []
        for record in self.records:
            image = preprocess_minimal_percentile(load_tiff(Path(record["image"])), params)
            mask = load_manual_foreground(record["mask"])
            if image.shape != mask.shape or min(image.shape[-2:]) < config.patch_size:
                raise ValueError(f"Image/mask shape or crop-size mismatch: {record['image']}")
            self.images.append(image)
            self.masks.append(mask)

    @classmethod
    def from_dict(cls, data, split):
        return cls(BBBC039Config(**{k: v for k, v in data.items() if k in BBBC039Config.__dataclass_fields__}), split)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image, mask = self.images[index], self.masks[index]
        _, height, width = image.shape
        patch = self.config.patch_size
        generator = training_crop_generator(self) if self.split == "train" else None
        if self.split == "train" and self.config.train_random_crops:
            top = int(torch.randint(height - patch + 1, (1,), generator=generator))
            left = int(torch.randint(width - patch + 1, (1,), generator=generator))
        else:
            top, left = (height - patch) // 2, (width - patch) // 2
        image = image[:, top:top+patch, left:left+patch]
        mask = mask[:, top:top+patch, left:left+patch]
        if self.split == "train" and self.config.random_flips:
            for dim in (-1, -2):
                if torch.rand((), generator=generator) > .5:
                    image, mask = image.flip(dim), mask.flip(dim)
        image = image.contiguous()
        return (image, mask.float().contiguous()) if self.config.return_mask else image
