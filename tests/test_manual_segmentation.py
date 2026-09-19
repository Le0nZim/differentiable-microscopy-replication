"""Manual-label provenance, unchanged pseudo fallback, and real trainer routing."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import tifffile
import torch
import yaml

from datasets.bbbc039 import BBBC039Dataset, load_manual_foreground
from workflow import bbbc039
from workflow.catalog import jobs, requirements
from workflow.common import read_json, write_json
from workflow.data import defaults, inspect, load_settings, prepare
from workflow.worker import execute, materialize


def annotated_fixture(root, monkeypatch):
    monkeypatch.setattr(bbbc039, "SPLITS", {"train": ("training", 2), "val": ("validation", 1), "test": ("test", 1)})
    for folder in ("images", "masks", "metadata"):
        (root / folder).mkdir(parents=True)
    splits = {"training": ["A01", "A02"], "validation": ["B01"], "test": ["C01"]}
    plate_rows = []
    for split, wells in splits.items():
        names = []
        for well in wells:
            stem = f"IXMtest_{well}_s1_w1fixture"
            names.append(stem + ".png")
            plate_rows.append(stem + ".png,20585")
            yy, xx = np.indices((520, 696))
            foreground = ((xx // 23 + yy // 31) % 3 == 0)
            tifffile.imwrite(root / "images" / (stem + ".tif"), (foreground * 1500).astype(np.uint16))
            rgba = np.zeros((520, 696, 4), np.uint8)
            rgba[..., 0] = foreground * 2
            rgba[..., 3] = 255
            Image.fromarray(rgba).save(root / "masks" / (stem + ".png"))
        (root / "metadata" / (split + ".txt")).write_text("\n".join(names) + "\n")
    (root / "metadata/filenames_and_plates.csv").write_text("\n".join(plate_rows) + "\n")
    return root


def test_rgba_annotation_ignores_opaque_alpha_and_unites_nucleus_labels(tmp_path):
    a = np.zeros((20, 20, 4), np.uint8)
    a[..., 3] = 255
    a[3:7, 2:5, 0] = 1
    a[3:7, 5:8, 0] = 3
    p = tmp_path / "manual.png"
    Image.fromarray(a).save(p)
    actual = load_manual_foreground(p)
    assert actual.sum() == 24 and not actual[0, 0, 0]
    np.testing.assert_array_equal(actual[0].numpy(), a[..., 0] > 0)


def test_manual_split_validation_and_aligned_advancing_crops(tmp_path, monkeypatch):
    root = annotated_fixture(tmp_path / "data", monkeypatch)
    manifest, paths = bbbc039.inspect(root)
    assert manifest["counts"] == {"train": 2, "val": 1, "test": 1}
    assert len(paths) == 12
    p = tmp_path / "manual.json"
    write_json(p, manifest)
    cfg = {"split_path": str(p), "patch_size": 64, "seed": 42}
    a, b = BBBC039Dataset.from_dict(cfg, "train"), BBBC039Dataset.from_dict(cfg, "train")
    seen = []
    for _ in range(5):
        x, mask = a[0]
        torch.testing.assert_close(x, mask)
        torch.testing.assert_close(x, b[0][0])
        seen.append(x)
    assert any(not torch.equal(seen[0], x) for x in seen[1:])
    val = BBBC039Dataset.from_dict(cfg, "val")
    torch.testing.assert_close(val[0][0], val[0][1])
    torch.testing.assert_close(val[0][0], val[0][0])
    # Explicit missing masks and overlapping source identities fail, never synthesize labels.
    listing = root / "metadata/test.txt"
    original = listing.read_text()
    listing.write_text((root / "metadata/validation.txt").read_text())
    with pytest.raises(ValueError, match="overlap"):
        bbbc039.inspect(root)
    listing.write_text(original)
    (root / "masks" / original.strip()).unlink()
    with pytest.raises(FileNotFoundError):
        bbbc039.inspect(root)


def test_old_machine_settings_get_manual_default_and_pseudo_values_are_preserved(tmp_path):
    settings = defaults(tmp_path / "source")
    settings["data"].pop("bbbc039")
    settings["run"].pop("segmentation_labels")
    p = tmp_path / "workstation.yaml"
    p.write_text(yaml.safe_dump(settings))
    restored = load_settings(p)
    assert restored["run"]["segmentation_labels"] == "bbbc039"
    assert restored["data"]["bbbc039"] == str(tmp_path / "source/bbbc039")
    assert requirements(["segmentation"]) == ["bbbc039"]
    assert requirements(["segmentation"], segmentation_labels="pseudo_trackmate") == ["bbbc022"]
    restored["run"]["segmentation_labels"] = "pseudo_trackmate"
    job = jobs([42], ["segmentation"], segmentation_labels="pseudo_trackmate")[0]
    cfg = materialize(job, restored, {"bbbc022_segmentation": str(tmp_path / "pseudo.json")}, tmp_path / "out", {})
    ds = cfg["dataset"]
    assert ds["name"] == "bbbc022_hoechst"
    assert {k: ds[k] for k in ("mask_mode", "mask_raw_threshold", "mask_smooth_interval", "mask_dp_epsilon")} == {
        "mask_mode": "trackmate", "mask_raw_threshold": 506., "mask_smooth_interval": 2., "mask_dp_epsilon": .5}
    assert ds["downscale_factor"] == 1.


def test_missing_manual_dataset_fails_instead_of_using_pseudo(tmp_path):
    cfg = defaults(tmp_path)
    result = inspect(cfg, ["segmentation"])
    assert len(result["errors"]) == 1 and "download-segmentation-data" in result["errors"][0]
    assert not result["resolved"]


def test_manual_worker_runs_three_stages_from_paired_images(tmp_path, monkeypatch):
    from utils.experiment_config import sync_derived_config_fields
    from datasets import bbbc022_hoechst
    root = annotated_fixture(tmp_path / "bbbc039", monkeypatch)
    settings = defaults(tmp_path)
    settings["run"].update(device="cpu", output_root=str(tmp_path / "campaign"))
    inspection = inspect(settings, ["segmentation"])
    assert not inspection["errors"]
    prepared = prepare(settings, ["segmentation"], inspection)
    job = jobs([42], ["segmentation"])[0]
    out = tmp_path / "out"
    cfg = materialize(job, settings, prepared, out, {})
    cfg["dataset"].update(patch_size=16, image_size=16)
    cfg["forward_model"]["downscale_factor"] = 4
    cfg["pattern_generator"]["num_patterns"] = 2
    cfg["inverse_model"]["reconstruction"]["hidden_channels"] = [4, 4, 4, 4, 4, 1]
    cfg["segmentation_head"]["hidden_channels"] = [4, 4, 1]
    cfg["training"].update(batch_size=2, log_every=1)
    task = cfg["training"]["task_aware"]
    task["stage1"]["learnable"].update(inverse_warmup_steps=1, joint_soft_steps=1, harden_m_values=[8], harden_steps_per_m=1)
    task.update(seg_head_steps=1, finetune_steps=1, num_qualitative_samples=1)
    cfg = sync_derived_config_fields(cfg)
    def forbidden(*a, **k):
        raise AssertionError("Manual labels must not invoke a pseudo-mask generator")
    monkeypatch.setattr(bbbc022_hoechst, "make_pseudo_mask", forbidden)
    monkeypatch.setattr(bbbc022_hoechst, "make_trackmate_mask", forbidden)
    spec = tmp_path / "job.json"
    write_json(spec, {"job": job, "config": cfg, "output": str(out), "device": "cpu", "dependencies": {}})
    execute(spec)
    summary = read_json(out / "metrics/run_summary.json")
    assert summary["label_source"] == "BBBC039 manual nucleus annotations" and summary["test_num_samples"] == 1
    assert (out / "checkpoints/stage2_seg_head.pt").exists()
    assert (out / "checkpoints/stage3_finetuned.pt").exists()
