#!/usr/bin/env python3
"""Phase 1: BBBC022 substitute data discovery and Hoechst channel selection."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets.bbbc022_hoechst import discover_image_paths, parse_well_site, select_hoechst_paths

DATA_ROOT = ROOT / "data/substitute_data"
OUT = ROOT / "experiments/ablations"


def _stats(arr: np.ndarray) -> dict:
    flat = arr.astype(np.float64).ravel()
    return {
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "p1": float(np.percentile(flat, 1)),
        "p50": float(np.percentile(flat, 50)),
        "p99": float(np.percentile(flat, 99)),
        "p99_9": float(np.percentile(flat, 99.9)),
    }


def main() -> None:
    import tifffile

    OUT.mkdir(parents=True, exist_ok=True)
    paths = discover_image_paths(DATA_ROOT, "**/*.tif")
    hoechst_paths = select_hoechst_paths(paths)

    rows: list[dict] = []
    for path in hoechst_paths:
        arr = np.asarray(tifffile.imread(path))
        well, site = parse_well_site(path)
        parent = path.parent.name
        st = _stats(arr)
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "filename": path.name,
                "extension": path.suffix.lower(),
                "plate": "IXMtest",
                "well": well or "",
                "site": site if site is not None else "",
                "channel_folder": parent,
                "likely_hoechst": "w1" in parent.lower() or "bbbc022" in parent.lower(),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                **st,
            }
        )

    manifest_path = OUT / "data_manifest.csv"
    fields = list(rows[0].keys()) if rows else []
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    channel_note = (
        "BBBC022_v1_images_20585w1 is a single-channel ImageXpress pack (w1). "
        "All 3456 TIFFs treated as Hoechst 33342 nuclear stain substitute for paper DAPI."
    )
    selected = hoechst_paths
    selected_csv = OUT / "hoechst_selected_files.csv"
    with selected_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "well", "site", "channel_note"])
        for p in selected:
            w, s = parse_well_site(p)
            writer.writerow([str(p.relative_to(ROOT)), w, s, channel_note])

    summary = {
        "data_root": str(DATA_ROOT),
        "total_tif_files": len(rows),
        "hoechst_selected_count": len(selected),
        "channel_selection": channel_note,
        "stain": "Hoechst 33342 (substitute for paper DAPI)",
        "typical_shape": rows[0]["shape"] if rows else None,
        "typical_dtype": rows[0]["dtype"] if rows else None,
        "well_site_parseable": sum(1 for r in rows if r["well"]),
        "split_policy": "split_by_well=True in BBBC022HoechstDataset",
    }
    (OUT / "data_manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path} ({len(rows)} files)")
    print(f"Wrote {selected_csv}")
    print(f"Channel: {channel_note}")


if __name__ == "__main__":
    main()
