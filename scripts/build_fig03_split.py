#!/usr/bin/env python3
"""Build a larger well-disjoint BBBC022 Hoechst split for the Fig.3 matrix.

The legacy split (`results/preprocessing_ablation_bbbc022_hoechst/configs/split.json`)
uses only 168 train / 21 val / 21 test images (one site per well). Combined with
the per-index deterministic crop bug this caused severe overfitting (train MSE
~0.0005 vs val ~0.0018) that capped *every* illumination method at the same wall,
so the learnable advantage could not emerge — unlike PatchMNIST (3000 imgs), where
learnable beats fixed ~3x.

This builder keeps the split strictly **well-disjoint** (no well appears in two
splits, so no field/treatment leakage) but uses far more of the available
384 wells x 9 sites = 3456 images:

* train: ``num_train_wells`` wells x *all* their sites (content + field diversity)
* val:   ``num_val_wells``  wells x 1 site (deterministic center-crop eval)
* test:  ``num_test_wells`` wells x 1 site

The output JSON matches the schema consumed by
``datasets.bbbc022_split.load_split`` (a top-level ``splits`` dict of relative
paths), so the existing dataset/dataloader code reads it unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from datasets.bbbc022_hoechst import (  # noqa: E402
    discover_image_paths,
    parse_well_site,
    select_hoechst_paths,
)


def build(
    data_root: Path,
    stack_glob: str,
    num_train_wells: int,
    num_val_wells: int,
    num_test_wells: int,
    sites_per_train_well: int,
    seed: int,
) -> dict:
    paths = select_hoechst_paths(discover_image_paths(data_root, stack_glob))
    by_well: dict[str, list[Path]] = {}
    for p in paths:
        well, _site = parse_well_site(p)
        by_well.setdefault(well or p.stem, []).append(p)
    # sort sites within each well for determinism
    for w in by_well:
        by_well[w] = sorted(by_well[w], key=lambda p: parse_well_site(p)[1] or 0)

    wells = sorted(by_well.keys())
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(wells), generator=gen).tolist()
    shuffled = [wells[i] for i in perm]

    need = num_train_wells + num_val_wells + num_test_wells
    if len(shuffled) < need:
        raise ValueError(f"Need {need} wells, only {len(shuffled)} available")

    train_w = shuffled[:num_train_wells]
    val_w = shuffled[num_train_wells : num_train_wells + num_val_wells]
    test_w = shuffled[num_train_wells + num_val_wells : need]

    def rel(p: Path) -> str:
        return str(p.resolve().relative_to(ROOT))

    train_paths: list[str] = []
    for w in train_w:
        train_paths.extend(rel(p) for p in by_well[w][:sites_per_train_well])
    val_paths = [rel(by_well[w][0]) for w in val_w]
    test_paths = [rel(by_well[w][0]) for w in test_w]

    splits = {"train": train_paths, "val": val_paths, "test": test_paths}
    train_set, val_set, test_set = set(train_w), set(val_w), set(test_w)
    payload = {
        "description": (
            "Larger well-disjoint BBBC022 Hoechst split for the Fig.3 content-aware "
            "matrix (multi-site train; 1 site/well val+test). Strictly well-disjoint."
        ),
        "spec": {
            "data_root": str(data_root.relative_to(ROOT)) if data_root.is_absolute() else str(data_root),
            "stack_glob": stack_glob,
            "num_train_wells": num_train_wells,
            "num_val_wells": num_val_wells,
            "num_test_wells": num_test_wells,
            "sites_per_train_well": sites_per_train_well,
            "seed": seed,
        },
        "counts": {k: len(v) for k, v in splits.items()},
        "wells": {"train": sorted(train_w), "val": sorted(val_w), "test": sorted(test_w)},
        "disjoint": {
            "train_val": not (train_set & val_set),
            "train_test": not (train_set & test_set),
            "val_test": not (val_set & test_set),
        },
        "splits": splits,
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT / "data/substitute_data"))
    ap.add_argument("--stack-glob", default="**/*.tif")
    ap.add_argument("--num-train-wells", type=int, default=220)
    ap.add_argument("--num-val-wells", type=int, default=40)
    ap.add_argument("--num-test-wells", type=int, default=60)
    ap.add_argument("--sites-per-train-well", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default=str(ROOT / "results/preprocessing_ablation_bbbc022_hoechst/configs/split_fig03_large.json"),
    )
    args = ap.parse_args()

    payload = build(
        Path(args.data_root),
        args.stack_glob,
        args.num_train_wells,
        args.num_val_wells,
        args.num_test_wells,
        args.sites_per_train_well,
        args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"counts: {payload['counts']}  disjoint: {payload['disjoint']}")


if __name__ == "__main__":
    main()
