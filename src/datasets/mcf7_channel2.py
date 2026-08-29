"""Human MCF7 BBBC021 dataset loader (single fluorescence channel, explicit).

Paper Sec 5.1: "Images from **channel-2** of the dataset were utilized ... 3000, 100,
and 100 image patches ... train, validation, and test".  In BBBC021 the channel token
``w2`` == **Tubulin** (see ``bbbc021_channels.py`` and ``BBBC021_v1_image.csv`` column
order DAPI / Tubulin / Actin).  So the paper-faithful default is ``tubulin``.

This loader is now EXPLICIT about the channel it loads:

    * ``bbbc021_channel`` in {``actin``, ``tubulin``, ``dapi``} (default ``tubulin``).
    * It never loads "all channels" and never silently averages a 3-channel composite
      to grayscale -- a 3-D image raises instead.
    * On construction it prints the exact channel, the ``w#`` token, the metadata
      columns used, the resolved image directory, and the first 20 image paths.
    * It asserts every resolved file exists, that exactly one channel is loaded per
      sample, and that each loaded target is 2-D grayscale before preprocessing.

Preprocessing beyond background-subtract + percentile-clip float normalize is
PAPER_UNSPECIFIED.  Normalization is a single deterministic per-image
background/percentile scaling (NOT per-crop min-max) so relative fluorescence
structure is preserved; per-image min-max is reserved for visualization only.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets.bbbc021_channels import (
    CHANNEL_TABLE,
    build_triplet_index,
    channel_dir_name,
    normalize_channel,
    wtoken_from_filename,
)

SplitName = Literal["train", "val", "test"]


@dataclass
class MCF7Channel2Config:
    data_root: str = "data/mcf7_bbbc021"
    # ``images_dir`` is auto-resolved from ``bbbc021_channel`` when left at the default.
    images_dir: str = "channel2_selected"
    manifest_csv: str = "manifests/mcf7_channel2_manifest.csv"
    image_metadata_csv: str = "raw_zips/BBBC021_v1_image.csv"
    # Explicit fluorescence channel: actin (w4), tubulin (w2, paper channel-2), dapi (w1).
    bbbc021_channel: str = "tubulin"
    patch_size: int = 256
    num_train: int = 3000
    num_val: int = 100
    num_test: int = 100
    seed: int = 42
    train_random_crops: bool = True
    random_flips: bool = True
    split_by_well: bool = True
    clip_percentile: float = 99.9
    background_percentile: float = 1.0
    verbose: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "MCF7Channel2Config":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})

    def resolved_channel(self) -> str:
        return normalize_channel(self.bbbc021_channel)

    def resolved_images_dir(self) -> str:
        """Image directory for the requested channel.

        Tubulin keeps the historical ``channel2_selected`` dir (paper channel-2).
        Actin/DAPI use ``channel4_actin`` / ``channel1_dapi`` (extract via
        ``tools/audit_bbbc021_channels.py``).
        """
        channel = self.resolved_channel()
        if channel == "tubulin":
            # Preserve any explicitly configured dir for backwards compatibility.
            return self.images_dir or channel_dir_name("tubulin")
        return channel_dir_name(channel)


def _load_tiff(path: Path, *, assert_grayscale: bool = True) -> torch.Tensor:
    """Load a single-channel BBBC021 TIF as a 2-D float tensor.

    Raises on a 3-D (RGB / composite / z-stack) image instead of silently
    averaging channels -- averaging would mix fluorescence channels.
    """
    import tifffile

    arr = tifffile.imread(path)
    tensor = torch.from_numpy(arr).float()
    if assert_grayscale and tensor.ndim != 2:
        raise ValueError(
            f"Expected a single-channel 2-D BBBC021 TIF at {path} but got shape "
            f"{tuple(tensor.shape)}. Refusing to silently collapse a multi-channel "
            f"image to grayscale (that would mix DAPI/Tubulin/Actin)."
        )
    if tensor.ndim == 3:
        # Only reached when assert_grayscale=False (explicit opt-in for previews).
        tensor = tensor.mean(dim=-1)
    return tensor


def _preprocess(image: torch.Tensor, config: MCF7Channel2Config) -> torch.Tensor:
    image = image.float()
    bg = torch.quantile(image.flatten(), config.background_percentile / 100.0)
    image = image - bg
    image = image.clamp_min(0)
    hi = torch.quantile(image.flatten(), config.clip_percentile / 100.0)
    if hi > 0:
        image = image / hi
    return image.clamp(0, 1).unsqueeze(0)


_LOG_ONCE: set[str] = set()


def _log_channel_provenance(config: MCF7Channel2Config, rows: list[dict], split: SplitName) -> None:
    channel = config.resolved_channel()
    info = CHANNEL_TABLE[channel]
    key = f"{config.data_root}:{channel}"
    if not config.verbose or key in _LOG_ONCE:
        return
    _LOG_ONCE.add(key)
    print("=" * 78, flush=True)
    print(f"[MCF7/BBBC021 loader] channel = {channel.upper()}  ({info['stain']})", flush=True)
    print(f"[MCF7/BBBC021 loader]   w-token             = {info['wtoken']}  "
          f"(channel_index={info['channel_index']})", flush=True)
    print(f"[MCF7/BBBC021 loader]   CSV filename column = {info['csv_file_col']}", flush=True)
    print(f"[MCF7/BBBC021 loader]   CSV path column     = {info['csv_path_col']}", flush=True)
    print(f"[MCF7/BBBC021 loader]   images_dir          = "
          f"{config.data_root}/{config.resolved_images_dir()}", flush=True)
    print(f"[MCF7/BBBC021 loader]   manifest_csv        = {config.manifest_csv}", flush=True)
    print(f"[MCF7/BBBC021 loader]   resolved image files: {len(rows)}", flush=True)
    print(f"[MCF7/BBBC021 loader]   first 20 image paths ({split} pool source):", flush=True)
    for row in rows[:20]:
        tok = wtoken_from_filename(Path(row["path"]).name)
        print(f"      [{tok}] {row['path']}", flush=True)
    print("=" * 78, flush=True)


def _load_manifest(root: Path, config: MCF7Channel2Config) -> list[dict]:
    channel = config.resolved_channel()
    wtoken = CHANNEL_TABLE[channel]["wtoken"]
    images_dir = config.resolved_images_dir()
    manifest_path = root / config.manifest_csv
    rows: list[dict] = []

    triplet_index = None
    if channel != "tubulin":
        csv_path = root / config.image_metadata_csv
        if not csv_path.exists():
            raise FileNotFoundError(
                f"channel={channel} requires the BBBC021 metadata CSV at {csv_path} to map "
                f"tubulin->{channel} filenames. Run tools/audit_bbbc021_channels.py to extract "
                f"the {channel} images first."
            )
        triplet_index = build_triplet_index(csv_path)

    missing = 0
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            tub_file = row["image_file"]
            if channel == "tubulin":
                img_name = tub_file
            else:
                triplet = triplet_index.get(tub_file)
                if triplet is None:
                    continue
                img_name = triplet.files[channel]
            img_path = root / images_dir / img_name
            if img_path.exists():
                row = dict(row)
                row["path"] = str(img_path)
                row["channel"] = channel
                rows.append(row)
            else:
                missing += 1

    if not rows:
        raise FileNotFoundError(
            f"No channel-{wtoken} ({channel}) images found under {root / images_dir}. "
            f"For actin/dapi you must extract them first via tools/audit_bbbc021_channels.py."
        )

    # HARD CHECK: every resolved file exists and carries the requested channel token.
    for row in rows:
        p = Path(row["path"])
        assert p.exists(), f"resolved image does not exist: {p}"
        tok = wtoken_from_filename(p.name)
        assert tok == wtoken, (
            f"channel mismatch: expected {wtoken} for '{channel}', got token {tok} in {p.name}"
        )
    if missing:
        print(f"[MCF7/BBBC021 loader] WARNING: {missing} manifest rows had no extracted "
              f"{channel} file (skipped).", flush=True)
    return rows


def _assign_well_splits(rows: list[dict], config: MCF7Channel2Config) -> dict[SplitName, list[dict]]:
    by_well: dict[str, list[dict]] = {}
    for row in rows:
        well = row.get("well") or row["image_file"]
        by_well.setdefault(well, []).append(row)
    wells = sorted(by_well.keys())
    gen = torch.Generator().manual_seed(config.seed)
    perm = torch.randperm(len(wells), generator=gen).tolist()
    ordered = [wells[i] for i in perm]
    n_val_wells = max(1, len(ordered) // 20)
    n_test_wells = max(1, len(ordered) // 20)
    val_wells = set(ordered[:n_val_wells])
    test_wells = set(ordered[n_val_wells : n_val_wells + n_test_wells])
    train_wells = set(ordered[n_val_wells + n_test_wells :])
    splits: dict[SplitName, list[dict]] = {"train": [], "val": [], "test": []}
    for well, items in by_well.items():
        if well in test_wells:
            splits["test"].extend(items)
        elif well in val_wells:
            splits["val"].extend(items)
        else:
            splits["train"].extend(items)
    return splits


def _crop_patch(
    image: torch.Tensor,
    patch_size: int,
    top: int,
    left: int,
) -> torch.Tensor:
    _, h, w = image.shape
    if h < patch_size or w < patch_size:
        image = F.interpolate(image.unsqueeze(0), size=(max(h, patch_size), max(w, patch_size)), mode="bilinear", align_corners=False).squeeze(0)
        _, h, w = image.shape
    top = min(top, h - patch_size)
    left = min(left, w - patch_size)
    return image[:, top : top + patch_size, left : left + patch_size]


def _build_patch_specs(
    pool: list[dict],
    count: int,
    patch_size: int,
    seed: int,
    *,
    deterministic: bool,
) -> list[tuple[Path, int, int]]:
    if not pool:
        raise ValueError("empty image pool for patch generation")
    gen = torch.Generator().manual_seed(seed)
    specs: list[tuple[Path, int, int]] = []
    for i in range(count):
        if i == 0 or (i + 1) % 100 == 0 or i + 1 == count:
            print(f"MCF7 patch specs: {i + 1}/{count}...", flush=True)
        row = pool[i % len(pool)] if deterministic else pool[int(torch.randint(0, len(pool), (1,), generator=gen).item())]
        path = Path(row["path"])
        image = _preprocess(_load_tiff(path), MCF7Channel2Config())
        _, h, w = image.shape
        if deterministic:
            top = ((i * 37) * 13) % max(1, h - patch_size + 1)
            left = ((i * 53) * 17) % max(1, w - patch_size + 1)
        else:
            top = int(torch.randint(0, max(1, h - patch_size + 1), (1,), generator=gen).item())
            left = int(torch.randint(0, max(1, w - patch_size + 1), (1,), generator=gen).item())
        specs.append((path, top, left))
    return specs


class MCF7Channel2Dataset(Dataset):
    def __init__(self, config: MCF7Channel2Config, split: SplitName = "train") -> None:
        self.config = config
        self.split = split
        self.channel = config.resolved_channel()
        root = Path(config.data_root)
        rows = _load_manifest(root, config)
        _log_channel_provenance(config, rows, split)
        pools = _assign_well_splits(rows, config)
        counts = {"train": config.num_train, "val": config.num_val, "test": config.num_test}
        self.specs = _build_patch_specs(
            pools[split],
            counts[split],
            config.patch_size,
            config.seed + {"train": 0, "val": 1, "test": 2}[split],
            deterministic=split != "train" or not config.train_random_crops,
        )
        self._cache: dict[str, torch.Tensor] = {}

    @classmethod
    def from_dict(cls, data: dict, split: SplitName) -> "MCF7Channel2Dataset":
        return cls(MCF7Channel2Config.from_dict(data), split=split)

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, index: int) -> torch.Tensor:
        path, top, left = self.specs[index]
        key = str(path)
        if key not in self._cache:
            self._cache[key] = _preprocess(_load_tiff(path), self.config)
        image = self._cache[key]
        # HARD CHECK: exactly one channel per training sample, 2-D grayscale content.
        assert image.ndim == 3 and image.shape[0] == 1, (
            f"target must be [1, H, W] (single channel), got {tuple(image.shape)}"
        )
        if self.split == "train" and self.config.train_random_crops:
            gen = torch.Generator().manual_seed(self.config.seed + index)
            _, h, w = image.shape
            ps = self.config.patch_size
            top = int(torch.randint(0, max(1, h - ps + 1), (1,), generator=gen).item())
            left = int(torch.randint(0, max(1, w - ps + 1), (1,), generator=gen).item())
        patch = _crop_patch(image, self.config.patch_size, top, left)
        if self.split == "train" and self.config.random_flips:
            gen = torch.Generator().manual_seed(self.config.seed + index + 1000)
            if float(torch.rand(1, generator=gen)) > 0.5:
                patch = torch.flip(patch, dims=[-1])
            if float(torch.rand(1, generator=gen)) > 0.5:
                patch = torch.flip(patch, dims=[-2])
        return patch
