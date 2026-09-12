"""Dataset and dataloader builders for training and evaluation."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from datasets.patchmnist import PatchMNISTDataset
from datasets.u2os import U2OSDataset
from datasets.bbbc022_hoechst import BBBC022HoechstDataset
from datasets.bbbc022_preproc_ablation import PreprocAblationDataset
from datasets.mcf7_channel2 import MCF7Channel2Dataset


def build_dataset(config: dict[str, Any], split: str):
    dataset_name = config["dataset"]["name"].lower()
    if dataset_name == "patchmnist":
        dataset = PatchMNISTDataset.from_dict(config["dataset"], split=split)
    elif dataset_name == "u2os":
        dataset = U2OSDataset.from_dict(config["dataset"], split=split)
    elif dataset_name in {"bbbc022_hoechst", "bbbc022_substitute"}:
        dataset = BBBC022HoechstDataset.from_dict(config["dataset"], split=split)
    elif dataset_name in {"bbbc022_preproc_ablation", "preproc_ablation"}:
        dataset = PreprocAblationDataset.from_dict(config["dataset"], split=split)
    elif dataset_name in {"mcf7_channel2", "mcf7"}:
        dataset = MCF7Channel2Dataset.from_dict(config["dataset"], split=split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if split == "train":
        max_samples = config["dataset"].get("max_train_samples")
        if max_samples is not None:
            dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    elif split in {"val", "test"}:
        max_samples = config["dataset"].get(f"max_{split}_samples")
        if max_samples is not None:
            dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return dataset


def build_dataloader(config: dict[str, Any], split: str) -> DataLoader:
    dataset = build_dataset(config, split)
    batch_size = config["training"]["batch_size"]
    shuffle = split == "train"
    # Opt in with a separate loader seed for paired model comparisons. Historical
    # Udith replay intentionally retains its original global-RNG sequence.
    seed = config["training"].get("loader_seed")
    generator = None if seed is None else torch.Generator().manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
