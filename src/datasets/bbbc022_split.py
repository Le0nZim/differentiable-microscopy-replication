"""Reproducible, metadata-aware (well-disjoint) split for BBBC022 Hoechst.

The split is built once and saved to disk as JSON so every preprocessing mode and
both experiments (Fig. 3 reconstruction, Fig. 4 segmentation) use the *exact same*
train/val/test files. It reuses the well-aware assignment from
``datasets.bbbc022_hoechst`` (one site per well, wells disjoint across splits) and
matches the paper's 168/21/21 image counts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from datasets.bbbc022_hoechst import (
    BBBC022HoechstConfig,
    assign_split_paths,
    discover_image_paths,
    parse_well_site,
    select_hoechst_paths,
)


@dataclass
class SplitSpec:
    data_root: str = "data/substitute_data"
    stack_glob: str = "**/*.tif"
    num_train_images: int = 168
    num_val_images: int = 21
    num_test_images: int = 21
    seed: int = 42


def build_split(spec: SplitSpec, repo_root: Path) -> dict[str, list[str]]:
    """Return {split: [relative tif paths]} using a well-disjoint assignment."""
    data_root = (repo_root / spec.data_root).resolve()
    paths = select_hoechst_paths(discover_image_paths(data_root, spec.stack_glob))
    config = BBBC022HoechstConfig(
        data_root=str(data_root),
        stack_glob=spec.stack_glob,
        num_train_images=spec.num_train_images,
        num_val_images=spec.num_val_images,
        num_test_images=spec.num_test_images,
        seed=spec.seed,
        split_by_well=True,
    )
    split_paths = assign_split_paths(paths, config)
    rel: dict[str, list[str]] = {}
    for split, items in split_paths.items():
        rel[split] = [str(p.resolve().relative_to(repo_root)) for p in items]
    return rel


def split_well_sets(split: dict[str, list[str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, items in split.items():
        wells = sorted({parse_well_site(Path(p))[0] or Path(p).stem for p in items})
        out[name] = wells
    return out


def save_split(split: dict[str, list[str]], spec: SplitSpec, out_path: Path) -> Path:
    well_sets = split_well_sets(split)
    train_w, val_w, test_w = (set(well_sets[s]) for s in ("train", "val", "test"))
    payload = {
        "description": "Well-disjoint BBBC022 Hoechst split (one site per well); shared by all preprocessing modes.",
        "spec": vars(spec),
        "counts": {k: len(v) for k, v in split.items()},
        "wells": well_sets,
        "disjoint": {
            "train_val": not (train_w & val_w),
            "train_test": not (train_w & test_w),
            "val_test": not (val_w & test_w),
        },
        "splits": split,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_split(path: Path, repo_root: Path) -> dict[str, list[Path]]:
    payload = json.loads(Path(path).read_text())
    return {split: [repo_root / rel for rel in items] for split, items in payload["splits"].items()}
