from __future__ import annotations

import csv
import gzip
import re
import shutil
import struct
import subprocess
from collections import defaultdict
from pathlib import Path

from .catalog import SR_SETS, requirements
from .common import ROOT, digest, file_sha, module, read_json, write_json

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MNIST_FILES = {
    "train-images-idx3-ubyte": (2051, 60000, 28, 28),
    "train-labels-idx1-ubyte": (2049, 60000),
    "t10k-images-idx3-ubyte": (2051, 10000, 28, 28),
    "t10k-labels-idx1-ubyte": (2049, 10000),
}


def images(path):
    path = Path(path)
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS) if path.is_dir() else []


def hr_files(root):
    """Recognize nested HR packs; do not import LR copies or multiple scales."""
    root = Path(root)
    found = []
    for p in images(root):
        parts = [s.lower() for s in p.relative_to(root).parts[:-1]]
        if any(s in {"lr", "lq", "x2", "x3", "x4"} or "bicubic" in s or "_lr" in s for s in parts):
            continue
        if re.search(r"(?:x[234]|[_-](?:lr|lq))(?:\.[^.]+)$", p.name, re.I):
            continue
        found.append(p)
    if not found:
        raise ValueError(f"No HR images in {root}; point this field to extracted HR originals.")
    # A repeated scene is generally a second scale or duplicate download.
    seen = set()
    for p in found:
        key = re.sub(r"[_-]hr$", "", p.stem, flags=re.I).lower()
        if key in seen:
            raise ValueError(f"Multiple copies of SR scene {key} under {root}. Select one HR folder in workstation.yaml.")
        seen.add(key)
    return found


def defaults(data_root):
    root = Path(data_root).expanduser().resolve()
    mcf = root / "mcf7_bbbc021"
    selected = mcf / "channel2_selected"
    if not images(selected) and images(mcf / "channel2_tubulin"):
        selected = mcf / "channel2_tubulin"
    flickr = root / "sr/train/Flickr2K"
    if not images(flickr) and images(root / "sr/train/FLICK2R"):
        flickr = root / "sr/train/FLICK2R"
    return {
        "version": 1,
        "data": {
            "mnist": str(root / "mnist"),
            "bbbc022": str(root / "substitute_data/BBBC022_v1_images_20585w1"),
            "mcf7_images": str(selected),
            "mcf7_manifest": str(mcf / "manifests/mcf7_channel2_manifest.csv"),
            "sr_train": {"DIV2K": str(root / "sr/train/DIV2K"), "Flickr2K": str(flickr)},
            "sr_test": {name: str(root / "sr/test" / name) for name in SR_SETS},
        },
        "run": {"output_root": str(ROOT / "runs/journal_v1"), "cache_root": str(ROOT / ".workstation-cache"),
                "device": "cuda:1", "seeds": [42, 43, 44], "data_seed": 42, "num_workers": 8, "min_free_gb": 20},
    }


def load_settings(path):
    import yaml
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or cfg.get("version") != 1:
        raise ValueError("Unsupported workstation config. Run: python paper.py init --data-root /path/to/data")
    if set(cfg) != {"version", "data", "run"}:
        raise ValueError("workstation.yaml must contain only version, data and run; check for misspelled fields.")
    required_data = {"mnist", "bbbc022", "mcf7_images", "mcf7_manifest", "sr_train", "sr_test"}
    required_run = {"output_root", "cache_root", "device", "seeds", "data_seed", "num_workers", "min_free_gb"}
    if set(cfg["data"]) != required_data or set(cfg["run"]) != required_run:
        raise ValueError("workstation.yaml has missing/unknown fields; compare with configs/workstation.example.yaml")
    def absolute(value):
        p = Path(value).expanduser()
        return str((path.parent / p).resolve())
    for key, value in cfg["data"].items():
        if isinstance(value, dict):
            cfg["data"][key] = {k: absolute(v) for k, v in value.items()}
        else:
            cfg["data"][key] = absolute(value)
    if set(cfg["data"]["sr_train"]) != {"DIV2K", "Flickr2K"} or set(cfg["data"]["sr_test"]) != set(SR_SETS):
        raise ValueError("SR requires DIV2K, Flickr2K and all five named test sets.")
    for key in ("output_root", "cache_root"):
        cfg["run"][key] = absolute(cfg["run"][key])
    seeds = cfg["run"]["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("run.seeds must be a nonempty list of unique nonnegative integers")
    if type(cfg["run"]["data_seed"]) is not int or cfg["run"]["data_seed"] < 0:
        raise ValueError("run.data_seed must be a nonnegative integer")
    if type(cfg["run"]["num_workers"]) is not int or cfg["run"]["num_workers"] < 0:
        raise ValueError("run.num_workers must be a nonnegative integer")
    if not isinstance(cfg["run"]["min_free_gb"], (int, float)) or cfg["run"]["min_free_gb"] < 0:
        raise ValueError("run.min_free_gb must be nonnegative")
    output = Path(cfg["run"]["output_root"])
    cache = Path(cfg["run"]["cache_root"])
    source_dirs = [Path(v) for k, v in cfg["data"].items() if isinstance(v, str) and k != "mcf7_manifest"]
    source_dirs += [Path(p) for k in ("sr_train", "sr_test") for p in cfg["data"][k].values()]
    for target in (output, cache):
        if target == ROOT or target in {ROOT / "experiments", ROOT / "paper", ROOT / "src"}:
            raise ValueError("Choose a dedicated output/cache directory, e.g. runs/paper_v1")
        for source in source_dirs:
            if target == source or source in target.parents or target in source.parents:
                raise ValueError(f"Output/cache must be separate from source data: {target} / {source}")
    if output == cache or output in cache.parents or cache in output.parents:
        raise ValueError("Output and cache directories must be separate.")
    return cfg


def inspect_mnist(root):
    raw = Path(root) / "MNIST/raw"
    paths = []
    for name, expected in MNIST_FILES.items():
        p = raw / name
        if not p.is_file():
            p = raw / (name + ".gz")
        if not p.is_file():
            raise ValueError(f"Missing {raw / name} (or .gz). Place both train and t10k IDX images/labels under MNIST/raw.")
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rb") as f:
            head = f.read(4 * len(expected))
        if len(head) != 4 * len(expected) or struct.unpack(">" + "I" * len(expected), head) != expected:
            raise ValueError(f"Invalid MNIST IDX header/count: {p}")
        if p.suffix != ".gz":
            expected_bytes = 4 * len(expected) + expected[1] * (784 if len(expected) == 4 else 1)
            if p.stat().st_size != expected_bytes:
                raise ValueError(f"Truncated MNIST IDX file: {p}")
        paths.append(p)
    return paths


def inspect_bbbc(root):
    files = images(root)
    if not files:
        raise ValueError(f"No BBBC022 TIFFs in {root}. Extract BBBC022_v1_images_20585w1 here.")
    by_well = defaultdict(list)
    for p in files:
        m = re.search(r"_([A-P]\d{2})_s(\d+)_w(\d)", p.name, re.I)
        if p.suffix.lower() not in {".tif", ".tiff"} or not m or m[3] != "1":
            raise ValueError(f"Expected BBBC022 w1 TIFF with well/site identity: {p}")
        by_well[m[1].upper()].append((int(m[2]), p))
    if len(by_well) < 320:
        raise ValueError(f"BBBC022 requires at least 320 wells for 220/40/60; found {len(by_well)}.")
    for well, items in by_well.items():
        if len({s for s, _ in items}) != len(items):
            raise ValueError(f"Repeated BBBC022 well/site {well}; select the single 20585w1 plate folder.")
        if len(items) < 9:
            raise ValueError(f"BBBC022 well {well} has {len(items)} sites; expected 9. Check extraction.")
    return files, {w: [p for _, p in sorted(v)] for w, v in by_well.items()}


def inspect_mcf(data):
    root = Path(data["mcf7_images"])
    available = images(root)
    if not available:
        raise ValueError(f"No MCF7 images in {root}. Set mcf7_images to channel2_selected or channel2_tubulin.")
    for p in available:
        if not re.search(r"_w2", p.name, re.I) or p.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"MCF7 requires tubulin w2 only: {p}")
    source = Path(data["mcf7_manifest"])
    if source.exists():
        with source.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if not rows or any(not r.get("image_file") or not r.get("well") for r in rows):
            raise ValueError(f"{source} must have nonempty image_file and well columns for every row.")
        by_name = defaultdict(list)
        for p in available:
            by_name[p.name].append(p)
        clean = []
        for row in rows:
            candidates = by_name.get(Path(row["image_file"]).name, [])
            if len(candidates) != 1:
                raise ValueError(f"MCF7 manifest image is missing or ambiguous: {row['image_file']} in {root}")
            clean.append({"image_file": str(candidates[0].resolve()), "well": row["well"]})
    else:
        clean = []
        for p in available:
            m = re.search(r"^(.*?)[_-]([A-P]\d{2})[_-]", p.name, re.I)
            if not m:
                raise ValueError(f"Cannot recover MCF7 well/plate from {p.name}; supply the image_file,well manifest.")
            clean.append({"image_file": str(p.resolve()), "well": f"{m[1]}:{m[2].upper()}"})
    if len({r["image_file"] for r in clean}) != len(clean):
        raise ValueError("Duplicate image_file rows in MCF7 manifest; remove duplicates before preparation.")
    if len({r["well"] for r in clean}) < 3:
        raise ValueError("MCF7 needs at least three wells for disjoint train/validation/test pools.")
    return clean, [Path(r["image_file"]) for r in clean]


def inspect_pixels(paths, *, grayscale=False):
    """Decode deterministic samples; inventories cover every input's size/mtime."""
    import numpy as np
    import tifffile
    from PIL import Image
    chosen = sorted({0, len(paths) // 2, len(paths) - 1})
    for i in chosen:
        p = paths[i]
        if p.stat().st_size < 150 and p.read_bytes().startswith(b"version https://git-lfs"):
            raise ValueError(f"Image is an LFS pointer: {p}")
        arr = tifffile.imread(p) if p.suffix.lower() in {".tif", ".tiff"} else np.asarray(Image.open(p))
        if arr.ndim not in ((2,) if grayscale else (2, 3)) or min(arr.shape[:2]) < 64 or not np.isfinite(arr).all():
            raise ValueError(f"Invalid image shape/values in {p}: {arr.shape}")


def inspect(cfg, stages):
    """Gather all actionable errors instead of stopping after the first path."""
    needed = requirements(stages)
    data = cfg["data"]
    evidence, inventories, resolved, errors = {}, {}, {}, []
    for need in needed:
        try:
            if need == "mnist":
                paths = inspect_mnist(data["mnist"])
                resolved[need] = str(Path(data["mnist"]))
                evidence[need] = "60,000 train / 10,000 test digits; IDX headers verified"
            elif need == "bbbc022":
                paths, groups = inspect_bbbc(data[need])
                inspect_pixels(paths, grayscale=True)
                resolved[need] = {w: [str(p.resolve()) for p in ps] for w, ps in groups.items()}
                evidence[need] = f"{len(paths)} w1 images, {len(groups)} wells; sample TIFFs decoded"
            elif need == "mcf7":
                rows, paths = inspect_mcf(data)
                inspect_pixels(paths, grayscale=True)
                resolved[need] = rows
                if Path(data["mcf7_manifest"]).is_file():
                    paths = paths + [Path(data["mcf7_manifest"])]
                evidence[need] = f"{len(rows)} tubulin images, {len({r['well'] for r in rows})} well groups"
            elif need == "sr":
                paths, resolved[need] = [], {"train": {}, "test": {}}
                for group in ("train", "test"):
                    for name, root in data["sr_" + group].items():
                        found = hr_files(root)
                        inspect_pixels(found)
                        resolved[need][group][name] = [str(p.resolve()) for p in found]
                        paths.extend(found)
                if len(paths) != len({p.resolve() for p in paths}):
                    raise ValueError("The same SR file appears in training and/or test roots; fix sr_train/sr_test paths.")
                for name, minimum in zip(SR_SETS, [5, 14, 100, 100, 109]):
                    count = len(resolved[need]["test"][name])
                    if count != minimum:
                        raise ValueError(f"{name} has {count} HR files, expected {minimum}; check for missing images or LR copies.")
                counts = {n: len(p) for n, p in resolved[need]["train"].items()}
                if counts["DIV2K"] != 800 or counts["Flickr2K"] != 2650:
                    raise ValueError(f"SR training expects DIV2K train=800 / Flickr2K=2650 HR originals; found {counts}. Select the train HR folders.")
                evidence[need] = f"HR train={counts}; all five test-set counts verified"
            elif need == "swinir":
                expected = re.search(r"^commit: (\w+)$", (ROOT / "SwinIR.COMMIT").read_text(), re.M)[1]
                r = subprocess.run(["git", "-C", str(ROOT / "SwinIR"), "rev-parse", "HEAD"], text=True, capture_output=True)
                if r.returncode or r.stdout.strip() != expected:
                    raise ValueError("Pinned SwinIR source is missing/wrong. Run: bash scripts/workstation/setup.sh")
                source = ROOT / "SwinIR/models/network_swinir.py"
                if not source.exists():
                    raise ValueError(f"Missing SwinIR source: {source}")
                paths = [source]
                evidence[need] = expected
            else:
                import torch
                from torchvision.models import VGG19_Weights
                filename = Path(VGG19_Weights.IMAGENET1K_V1.url).name
                p = Path(torch.hub.get_dir()) / "checkpoints" / filename
                if not p.exists():
                    raise ValueError("Pretrained VGG19 weights missing. Run: python paper.py download-weights (once, with internet).")
                # torchvision's pretrained file is ~548 MB; pointers/truncation must fail now.
                if p.stat().st_size < 500_000_000:
                    raise ValueError(f"VGG19 weights appear truncated: {p}. Re-download them.")
                expected_prefix = filename.rsplit("-", 1)[1].split(".")[0]
                if not file_sha(p).startswith(expected_prefix):
                    raise ValueError(f"VGG19 checkpoint SHA256 does not match torchvision: {p}")
                paths = [p]
                evidence[need] = str(p)
            inventories[need] = [{"path": str(p.resolve()), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in paths]
        except (ValueError, OSError, ImportError, RuntimeError) as exc:
            errors.append(f"{need}: {exc}")
    return {"evidence": evidence, "inventories": inventories, "resolved": resolved, "errors": errors}


def prepare(cfg, stages, inspection):
    import torch
    out = Path(cfg["run"]["output_root"]) / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    rs = inspection["resolved"]
    if "mnist" in rs:
        source = Path(rs["mnist"])
        if all((source / "MNIST/raw" / n).exists() for n in MNIST_FILES):
            result["mnist_root"] = str(source)
        else:
            dest = out / "mnist/MNIST/raw"
            dest.mkdir(parents=True, exist_ok=True)
            for name in MNIST_FILES:
                p = source / "MNIST/raw" / name
                if p.exists():
                    shutil.copyfile(p, dest / name)
                else:
                    with gzip.open(str(p) + ".gz", "rb") as a, (dest / name).open("wb") as b:
                        shutil.copyfileobj(a, b)
            inspect_mnist(out / "mnist")
            result["mnist_root"] = str(out / "mnist")
        result["mnist_sha256"] = digest([file_sha(p) for p in inspect_mnist(result["mnist_root"])])
    if "bbbc022" in rs:
        groups = rs["bbbc022"]
        wells = sorted(groups)
        gen = torch.Generator().manual_seed(cfg["run"]["data_seed"])
        order = [wells[i] for i in torch.randperm(len(wells), generator=gen).tolist()]
        # The last 64 wells were unused by the recorded 220/40/60 campaign.
        # Reserve 60 of those for the journal test, retaining train/validation.
        assigned = {"train": order[:220], "val": order[220:260], "test": order[320:380]}
        large = {s: [p for w in ws for p in groups[w][:9 if s == "train" else 1]] for s, ws in assigned.items()}
        # A single global partition prevents one family's training wells becoming
        # another family's held-out wells. Segmentation uses a nested small split.
        small = {s: [groups[w][0] for w in assigned[s][:n]] for s, n in zip(["train", "val", "test"], [168, 21, 21])}
        for name, splits in [("bbbc022_large", large), ("bbbc022_segmentation", small)]:
            p = out / (name + ".json")
            write_json(p, {"splits": splits, "data_seed": cfg["run"]["data_seed"],
                           "counts": {s: len(v) for s, v in splits.items()}, "global_wells": assigned,
                           "excluded_previous_test_wells": order[260:320],
                           "test_policy": "Previously unused reserve wells, for the same data_seed",
                           "label": "BBBC022 20585w1 substitute, not original U2OS"})
            result[name] = str(p)
    if "mcf7" in rs:
        p = out / "mcf7_tubulin.csv"
        with p.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_file", "well"])
            writer.writeheader()
            writer.writerows(rs["mcf7"])
        result["mcf7_manifest"] = str(p)
    if "sr" in rs:
        # Freeze parent-image splits independently of the model seed.
        paths = sorted(p for values in rs["sr"]["train"].values() for p in values)
        gen = torch.Generator().manual_seed(cfg["run"]["data_seed"])
        n_val = max(1, round(0.02 * len(paths)))
        idx = set(torch.randperm(len(paths), generator=gen).tolist()[:n_val])
        splits = {"train": [p for i, p in enumerate(paths) if i not in idx],
                  "val": [p for i, p in enumerate(paths) if i in idx]}
        p = out / "sr_split.json"
        write_json(p, {**splits, "test": rs["sr"]["test"], "data_seed": cfg["run"]["data_seed"]})
        result["sr_split"] = str(p)
        result["sr_test_roots"] = {}
        for name, paths in rs["sr"]["test"].items():
            folder = out / "sr_test" / name
            folder.mkdir(parents=True, exist_ok=True)
            for source in paths:
                # Flat views are read by the existing evaluator. No 44 GB data copy.
                dest = folder / Path(source).name
                if dest.is_symlink() and dest.resolve() == Path(source):
                    continue
                if dest.exists() or dest.is_symlink():
                    raise ValueError(f"Conflicting prepared SR link: {dest}")
                dest.symlink_to(source)
            result["sr_test_roots"][name] = str(folder)
    return result
