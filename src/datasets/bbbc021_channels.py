"""BBBC021 channel resolution + matched-triplet extraction helpers.

The BBBC021 (Human MCF7) dataset acquires THREE fluorescence channels per field
of view:

    * ``w1`` -> DAPI    (DNA / nuclei)
    * ``w2`` -> Tubulin (beta-tubulin / microtubules, cytoskeletal)
    * ``w4`` -> Actin   (F-actin, cytoskeletal)

The channel token (``w1`` / ``w2`` / ``w4``) is embedded in every image filename and
the mapping to the biological stain is fixed by the column order of
``BBBC021_v1_image.csv`` (``Image_FileName_DAPI``, ``Image_FileName_Tubulin``,
``Image_FileName_Actin``).

The paper (Sec 5.1) states it used **channel-2** of the dataset, which is
**Tubulin**.  This module makes the channel choice explicit and auditable and lets
us extract the *matched* Actin / DAPI files for the identical fields of view so we
can visually compare which channel matches the paper's Fig. 8 / 9 images.
"""

from __future__ import annotations

import csv
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Bbbc021Channel = Literal["actin", "tubulin", "dapi"]

# Canonical, fully-specified channel table. Do NOT guess these anywhere else.
CHANNEL_TABLE: dict[str, dict] = {
    "dapi": {
        "wtoken": "w1",
        "csv_file_col": "Image_FileName_DAPI",
        "csv_path_col": "Image_PathName_DAPI",
        "channel_index": 1,
        "stain": "DAPI (DNA / nuclei)",
        "is_cytoskeletal": False,
    },
    "tubulin": {
        "wtoken": "w2",
        "csv_file_col": "Image_FileName_Tubulin",
        "csv_path_col": "Image_PathName_Tubulin",
        "channel_index": 2,
        "stain": "beta-Tubulin (microtubules, cytoskeletal)",
        "is_cytoskeletal": True,
    },
    "actin": {
        "wtoken": "w4",
        "csv_file_col": "Image_FileName_Actin",
        "csv_path_col": "Image_PathName_Actin",
        "channel_index": 4,
        "stain": "F-actin (cytoskeletal)",
        "is_cytoskeletal": True,
    },
}

ALLOWED_CHANNELS = tuple(CHANNEL_TABLE.keys())


def normalize_channel(channel: str) -> str:
    """Validate + normalize a channel name; raise on anything unexpected."""
    key = str(channel).strip().lower()
    aliases = {
        "w1": "dapi",
        "dna": "dapi",
        "nuclei": "dapi",
        "channel1": "dapi",
        "channel_1": "dapi",
        "w2": "tubulin",
        "tub": "tubulin",
        "microtubule": "tubulin",
        "channel2": "tubulin",
        "channel_2": "tubulin",
        "w4": "actin",
        "f-actin": "actin",
        "channel4": "actin",
        "channel_4": "actin",
    }
    key = aliases.get(key, key)
    if key not in CHANNEL_TABLE:
        raise ValueError(
            f"Unknown bbbc021_channel={channel!r}. Allowed: {ALLOWED_CHANNELS} "
            f"(w1=dapi, w2=tubulin, w4=actin)."
        )
    return key


def channel_wtoken(channel: str) -> str:
    return CHANNEL_TABLE[normalize_channel(channel)]["wtoken"]


def channel_dir_name(channel: str) -> str:
    """Canonical on-disk subdir for a channel's extracted single-channel TIFs."""
    key = normalize_channel(channel)
    idx = CHANNEL_TABLE[key]["channel_index"]
    return f"channel{idx}_{key}"


def wtoken_from_filename(filename: str) -> str | None:
    """Return the ``w#`` channel token embedded in a BBBC021 filename, if any."""
    import re

    m = re.search(r"_(w[124])[0-9A-Fa-f]", filename)
    if m:
        return m.group(1)
    m = re.search(r"_(w[124])", filename)
    return m.group(1) if m else None


def channel_from_filename(filename: str) -> str | None:
    """Map a BBBC021 filename to its channel name via the ``w#`` token."""
    tok = wtoken_from_filename(filename)
    if tok is None:
        return None
    for name, info in CHANNEL_TABLE.items():
        if info["wtoken"] == tok:
            return name
    return None


@dataclass
class TripletRow:
    """One field of view with all three matched channel filenames."""

    plate: str
    well: str
    compound: str
    concentration: str
    replicate: str
    files: dict[str, str]  # channel -> filename


def build_triplet_index(csv_path: str | Path) -> dict[str, TripletRow]:
    """Index ``BBBC021_v1_image.csv`` keyed by the Tubulin (w2) filename.

    Each row of the CSV lists the DAPI / Tubulin / Actin filenames for the SAME
    field of view, so this lets us find the matched Actin / DAPI file given a
    Tubulin file (the channel the current reproduction loads).
    """
    csv_path = Path(csv_path)
    index: dict[str, TripletRow] = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        cols = reader.fieldnames or []
        for ch, info in CHANNEL_TABLE.items():
            if info["csv_file_col"] not in cols:
                raise KeyError(
                    f"Expected column {info['csv_file_col']!r} for channel {ch!r} "
                    f"in {csv_path.name}; found columns: {cols}"
                )
        for row in reader:
            files = {ch: row[CHANNEL_TABLE[ch]["csv_file_col"]] for ch in CHANNEL_TABLE}
            triplet = TripletRow(
                plate=row.get("Image_Metadata_Plate_DAPI", ""),
                well=row.get("Image_Metadata_Well_DAPI", ""),
                compound=row.get("Image_Metadata_Compound", ""),
                concentration=row.get("Image_Metadata_Concentration", ""),
                replicate=row.get("Replicate", ""),
                files=files,
            )
            index[files["tubulin"]] = triplet
    return index


def extract_member(zip_path: str | Path, member: str, dest_dir: str | Path) -> Path:
    """Extract a single ``member`` from a zip into ``dest_dir`` (cached)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / Path(member).name
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as src, out_path.open("wb") as dst:
            dst.write(src.read())
    return out_path


def resolve_matched_file(
    *,
    channel: str,
    tubulin_zip_internal: str,
    tubulin_filename: str,
    triplet_index: dict[str, TripletRow],
) -> str:
    """Return the zip-internal path of ``channel`` for the field of the given tubulin file.

    ``tubulin_zip_internal`` is e.g. ``Week4_27481/B02_s1_w2ABCD.tif``; the other
    channels live in the SAME folder with a different filename (read from the CSV).
    """
    channel = normalize_channel(channel)
    if channel == "tubulin":
        return tubulin_zip_internal
    triplet = triplet_index.get(tubulin_filename)
    if triplet is None:
        raise KeyError(
            f"Tubulin file {tubulin_filename!r} not present in BBBC021 CSV triplet index"
        )
    folder = str(Path(tubulin_zip_internal).parent).replace("\\", "/")
    other_name = triplet.files[channel]
    return f"{folder}/{other_name}"
