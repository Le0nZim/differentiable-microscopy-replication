"""Download and validate the public BBBC039 image/manual-mask pairs."""
import csv
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

from .common import file_sha, write_json

SOURCE = "https://bbbc.broadinstitute.org/BBBC039"
BASE_URL = "https://data.broadinstitute.org/bbbc/BBBC039/"
# SHA256 observed from the official v1 downloads on 2026-09-19.
ARCHIVES = {
    "images": "6f30a5d4fe38c928ded972704f085975f8dc0d65d9aa366df00e5a9d449fddd7",
    "masks": "f9e6043d8ca56344a4886f96a700d804d6ee982f31e2b2cd3194af2a053c2710",
    "metadata": "a2c1f900bed9ba92a99553efd4c2ae98598433691c7401d818653ab61110deb2",
}
SPLITS = {"train": ("training", 100), "val": ("validation", 50), "test": ("test", 50)}


def download(root, archives_dir=None):
    """Extract verified archives; never replace a different existing source file."""
    root = Path(root)
    archive_root = Path(archives_dir) if archives_dir else root / "raw_archives"
    archive_root.mkdir(parents=True, exist_ok=True)
    for name, expected in ARCHIVES.items():
        archive = archive_root / f"{name}.zip"
        if not archive.exists():
            if archives_dir:
                raise ValueError(f"Missing supplied archive: {archive}")
            part = archive.with_suffix(".zip.part")
            print(f"Downloading BBBC039 {name} from Broad...", flush=True)
            with urllib.request.urlopen(BASE_URL + archive.name, timeout=60) as src, part.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if file_sha(part) != expected:
                raise ValueError(f"BBBC039 archive checksum mismatch: {part}")
            part.replace(archive)
        if file_sha(archive) != expected:
            raise ValueError(f"BBBC039 archive checksum mismatch: {archive}")
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                rel = Path(item.filename)
                if item.is_dir() or rel.parts[0] == "__MACOSX" or any(p.startswith(".") for p in rel.parts):
                    continue
                if rel.is_absolute() or ".." in rel.parts or rel.parts[0] != name:
                    raise ValueError(f"Unexpected BBBC039 archive member: {rel}")
                dest = root / rel
                data = source.read(item)
                if dest.exists():
                    if dest.read_bytes() != data:
                        raise ValueError(f"Existing BBBC039 file differs; refusing replacement: {dest}")
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
        print(f"Verified and extracted BBBC039 {name}.", flush=True)
    manifest, _ = inspect(root)
    write_json(root / "download_provenance.json", {"source": SOURCE, "archive_sha256": ARCHIVES,
                                                  "counts": manifest["counts"]})
    return manifest


def inspect(root):
    """Verify every pair and preserve the published image and well partitions."""
    import numpy as np
    import tifffile
    from datasets.bbbc039 import load_manual_foreground

    root = Path(root)
    files, records, all_stems, all_wells = [], {}, set(), set()
    metadata = root / "metadata"
    if not (metadata / "filenames_and_plates.csv").exists():
        raise ValueError(f"BBBC039 manual annotations missing at {root}. Run: python paper.py download-segmentation-data")
    plates = {}
    for name, plate in csv.reader((metadata / "filenames_and_plates.csv").read_text().splitlines()):
        if name in plates:
            raise ValueError(f"Duplicate BBBC039 metadata identity: {name}")
        plates[name] = plate
    for split, (filename, expected) in SPLITS.items():
        listing = metadata / (filename + ".txt")
        names = [n.strip() for n in listing.read_text().splitlines() if n.strip()]
        if len(names) != expected:
            raise ValueError(f"BBBC039 {split}: expected {expected} official image names, got {len(names)}")
        records[split] = []
        wells = set()
        for name in names:
            if Path(name).name != name or not name.endswith(".png") or name not in plates:
                raise ValueError(f"Invalid BBBC039 split identity: {name}")
            stem = Path(name).stem
            match = re.fullmatch(r"IXMtest_([A-P]\d{2})_s\d+_w1.+", stem)
            if not match or stem in all_stems:
                raise ValueError(f"Wrong channel, duplicate image or split overlap: {name}")
            well = f"{plates[name]}:{match[1]}"
            if well in all_wells:
                raise ValueError(f"BBBC039 well shared across splits: {well}")
            wells.add(well)
            image_path, mask_path = root / "images" / (stem + ".tif"), root / "masks" / name
            image = tifffile.imread(image_path)
            mask = load_manual_foreground(mask_path)
            if image.ndim != 2 or image.shape != tuple(mask.shape[-2:]) or image.shape != (520, 696):
                raise ValueError(f"BBBC039 image/mask native shapes differ: {name}")
            if image.dtype != np.uint16 or not np.isfinite(image).all():
                raise ValueError(f"Expected original 16-bit BBBC039 image: {image_path}")
            records[split].append({"image": str(image_path.resolve()), "mask": str(mask_path.resolve()), "well": well,
                                   "image_sha256": file_sha(image_path), "mask_sha256": file_sha(mask_path)})
            files.extend([image_path, mask_path])
            all_stems.add(stem)
        files.append(listing)
        all_wells.update(wells)
    if {Path(n).stem for n in plates} != all_stems:
        raise ValueError("BBBC039 metadata and split coverage disagree")
    files.append(metadata / "filenames_and_plates.csv")
    manifest = {"dataset": "BBBC039v1", "source": SOURCE, "label_source": "BBBC039 manual nucleus annotations",
                "target": "binary union of positive red-channel annotation labels; alpha ignored",
                "split_policy": "official 100/50/50 image lists; plate+well disjointness verified",
                "evaluation_region": "center 256x256 crop per validation/test field",
                "counts": {s: len(v) for s, v in records.items()}, "splits": records}
    return manifest, files
