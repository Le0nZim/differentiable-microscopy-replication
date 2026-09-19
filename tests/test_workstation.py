"""Workstation data, dependency, completion and actual trainer integration checks."""
import copy
import csv
import json
import os
from pathlib import Path

import pytest
import torch
import yaml

from workflow.catalog import jobs, select_stages
from workflow.cli import freeze_campaign, record_artifacts, verify_completed
from workflow.common import module, read_json, write_json
from workflow.data import defaults, hr_files, inspect_mcf, load_settings, prepare
from workflow.worker import execute, materialize


def settings(tmp_path):
    cfg = defaults(tmp_path / "source data")
    cfg["run"].update(output_root=str(tmp_path / "campaign"), cache_root=str(tmp_path / "cache"), device="cpu", min_free_gb=0)
    return cfg


def prepared(tmp_path):
    return {"mnist_root": str(tmp_path / "mnist"), "mnist_sha256": "test-input-hash",
            "bbbc022_large": str(tmp_path / "large.json"), "bbbc022_segmentation": str(tmp_path / "small.json"),
            "mcf7_manifest": str(tmp_path / "mcf.csv"), "sr_split": str(tmp_path / "sr.json"),
            "sr_test_roots": {n: str(tmp_path / n) for n in ["Set5", "Set14", "BSD100", "Urban100", "Manga109"]}}


def test_entire_queue_materializes_without_datasets_or_historical_artifacts(tmp_path):
    cfg = settings(tmp_path)
    plan = jobs([42, 43, 44])
    assert len(plan) == 234
    positions = {job["id"]: i for i, job in enumerate(plan)}
    for job in plan + jobs([42], ["controlled"]):
        deps = {name: str(tmp_path / "fresh dependency" / name) for name in job["requires"]}
        out = tmp_path / "outputs" / job["id"]
        resolved = materialize(job, cfg, prepared(tmp_path), out, deps)
        assert resolved["experiment"]["output_dir"] == str(out)
        assert resolved["experiment"]["seed"] == job["seed"]
        if "dataset" in resolved:
            assert resolved["dataset"]["seed"] == 42
        if job["engine"] == "refiner":
            assert resolved["base_run_dir"] == deps[job["requires"][0]]
        for dep in job["requires"]:
            assert positions[dep] < positions[job["id"]]
        if "forward_model" in resolved and "compression" in resolved["experiment"]:
            assert resolved["experiment"]["compression"] == resolved["forward_model"]["downscale_factor"] ** 2 / resolved["pattern_generator"]["num_patterns"]
    assert select_stages(["content_swinir"]) == ["content", "content_swinir"]


def test_config_paths_spaces_and_source_overlap_protection(tmp_path):
    cfg = settings(tmp_path)
    p = tmp_path / "machine config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert load_settings(p)["data"]["mnist"] == str(tmp_path / "source data/mnist")
    cfg["run"]["output_root"] = str(tmp_path / "source data")
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="separate from source"):
        load_settings(p)


def test_hr_selection_excludes_lr_and_refuses_duplicate_scenes(tmp_path):
    for name in ["HR/0001.png", "LR_bicubic/X2/0001x2.png", "HR/0002x4.png"]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    assert hr_files(tmp_path) == [tmp_path / "HR/0001.png"]
    (tmp_path / "HR/0001.jpg").touch()
    with pytest.raises(ValueError, match="Multiple copies"):
        hr_files(tmp_path)


def test_mcf_rebases_manifest_and_never_skips_missing_or_wrong_channel(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    rows = []
    for well in ("A01", "B01", "C01"):
        name = f"plate_{well}_s1_w2abc.tif"
        (root / name).touch()
        rows.append({"image_file": "/old/workstation/" + name, "well": well})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_file", "well"])
        writer.writeheader()
        writer.writerows(rows)
    data = {"mcf7_images": str(root), "mcf7_manifest": str(manifest)}
    clean, paths = inspect_mcf(data)
    assert len(paths) == 3 and all(Path(r["image_file"]).parent == root for r in clean)
    paths[0].unlink()
    with pytest.raises(ValueError, match="missing or ambiguous"):
        inspect_mcf(data)
    (root / "plate_A01_s1_w1abc.tif").touch()
    with pytest.raises(ValueError, match="w2 only"):
        inspect_mcf(data)


def test_bbbc_small_split_is_nested_and_globally_well_disjoint(tmp_path):
    cfg = settings(tmp_path)
    groups = {f"{chr(65 + r)}{c:02}": [f"/external/IXMtest_{chr(65 + r)}{c:02}_s{s}_w1.tif" for s in range(1, 10)]
              for r in range(16) for c in range(1, 25)}
    result = prepare(cfg, ["content", "segmentation"], {"resolved": {"bbbc022": groups}})
    large = read_json(result["bbbc022_large"])["splits"]
    small = read_json(result["bbbc022_segmentation"])["splits"]
    assert [len(large[s]) for s in large] == [1980, 40, 60]
    assert [len(small[s]) for s in small] == [168, 21, 21]
    for split in small:
        assert set(small[split]) <= set(large[split])
        for other in large:
            if other != split:
                assert not {Path(p).name.split("_")[1] for p in small[split]} & {Path(p).name.split("_")[1] for p in large[other]}


def test_cache_matches_uncached_generation_and_does_not_mutate(tmp_path, monkeypatch):
    import datasets.patchmnist as pm
    digits = torch.rand(20, 32, 32, generator=torch.Generator().manual_seed(2))
    monkeypatch.setattr(pm, "_load_mnist_tensors", lambda *a, **k: digits)
    cfg = pm.PatchMNISTConfig(image_size=32, grid_size=2, num_train=3, num_val=2, num_test=2,
                              disjoint_val_test=True, download=False)
    expected = pm.generate_patchmnist_split(cfg, "test")
    cfg.cache_dir, cfg.source_sha256 = str(tmp_path), "digits-1"
    cached = pm.PatchMNISTDataset(cfg, "test")
    torch.testing.assert_close(torch.stack([cached[i] for i in range(2)]), torch.stack(expected), rtol=0, atol=0)
    cached[0].zero_()
    monkeypatch.setattr(pm, "_iter_patchmnist_split", lambda *a: (_ for _ in ()).throw(AssertionError("cache must be reused")))
    replay = pm.PatchMNISTDataset(cfg, "test")
    torch.testing.assert_close(replay[0], expected[0], rtol=0, atol=0)


def test_completed_artifacts_detect_corruption(tmp_path):
    p = tmp_path / "result.json"
    p.write_text('{"test_mse": 0.1}')
    write_json(tmp_path / "worker_complete.json", {"artifacts": ["result.json"]})
    entry = {"status": "completed", "output": str(tmp_path), "artifacts": record_artifacts(tmp_path)}
    assert verify_completed(entry)
    p.write_text('{"test_mse": 0.9}')
    assert not verify_completed(entry)


def test_campaign_refuses_changed_recipe_inputs_and_manifests(tmp_path):
    cfg = settings(tmp_path)
    Path(cfg["run"]["output_root"]).mkdir()
    inspection = {"resolved": {}, "inventories": {"mnist": [{"path": "/mnist", "bytes": 1}]}, "evidence": {}}
    freeze_campaign(cfg, ["patchmnist"], inspection)
    changed = copy.deepcopy(cfg)
    changed["run"]["data_seed"] = 43
    with pytest.raises(ValueError, match="different config/source"):
        freeze_campaign(changed, ["patchmnist"], inspection)
    inspection["inventories"]["mnist"][0]["bytes"] = 2
    with pytest.raises(ValueError, match="inputs changed"):
        freeze_campaign(cfg, ["patchmnist"], inspection)


@pytest.mark.parametrize("stage,mode", [("noise", "random_fixed"), ("content", "learnable_frequency"), ("patchmnist", "A")])
def test_worker_executes_real_small_cnn_and_ablation_trainers(tmp_path, monkeypatch, stage, mode):
    import training.dataloaders as data
    from utils.experiment_config import sync_derived_config_fields
    cfg = settings(tmp_path)
    job = next(j for j in jobs([42], [stage]) if j.get("mode", j.get("variant")) == mode)
    out = tmp_path / "attempt"
    resolved = materialize(job, cfg, prepared(tmp_path), out, {})
    resolved["dataset"]["image_size"] = 16
    resolved["forward_model"]["downscale_factor"] = 4
    resolved["pattern_generator"]["num_patterns"] = 2
    resolved["inverse_model"]["reconstruction"]["hidden_channels"] = [4, 4, 4, 4, 4, 1]
    resolved["training"].update(batch_size=2, max_steps=2, log_every=1, illumination_lr=.01)
    resolved["training"]["staged_hardening"] = {"inverse_warmup_steps": 1, "joint_soft_steps": 1, "harden_m_values": [2], "harden_steps_per_m": 1}
    resolved = sync_derived_config_fields(resolved)
    arrays = {s: torch.rand(n, 1, 16, 16) for s, n in [("train", 4), ("val", 3), ("test", 3)]}
    monkeypatch.setattr(data, "build_dataset", lambda c, s: arrays[s])
    if stage == "patchmnist":
        impl = module("scripts/table03_ablation/run.py")
        monkeypatch.setattr(impl, "default_phases", lambda: [{"name": "test", "steps": 2, "m": 1., "freeze_illum": True}])
    spec_path = tmp_path / "job.json"
    write_json(spec_path, {"job": job, "config": resolved, "output": str(out), "device": "cpu", "dependencies": {}})
    execute(spec_path)
    artifacts = record_artifacts(out)
    assert verify_completed({"status": "completed", "output": str(out), "artifacts": artifacts})
    assert read_json(out / "metrics/run_summary.json")["test_mse"] >= 0


def test_sequential_runner_retries_only_failed_job_and_reports(tmp_path, monkeypatch):
    import workflow.cli as cli
    cfg = settings(tmp_path)
    cfg["run"]["seeds"] = [42]
    plan = jobs([42], ["sr"])
    monkeypatch.setattr(cli, "validate", lambda *a, **k: {"resolved": {}, "inventories": {}, "evidence": {}})
    import workflow.worker as worker
    monkeypatch.setattr(worker, "materialize", lambda *a: {"test_stub": True})
    calls = []

    class Process:
        def __init__(self, command, **kwargs):
            spec = read_json(command[-1])
            calls.append(spec["job"]["id"])
            self.returncode = 1 if len(calls) == 2 else 0
            out = Path(spec["output"])
            write_json(out / "result.json", {"test_mse": 0.25})
            write_json(out / "worker_complete.json", {"artifacts": ["result.json"]})

        def poll(self):
            return self.returncode

    monkeypatch.setattr(cli.subprocess, "Popen", Process)
    # git_commit uses subprocess.run internally; avoid replacing that unrelated call.
    monkeypatch.setattr(cli, "git_commit", lambda: "test-revision")
    with pytest.raises(RuntimeError, match="Job failed"):
        cli.run_campaign(cfg, ["sr"])
    state = read_json(Path(cfg["run"]["output_root"]) / "state.json")
    assert state[plan[0]["id"]]["status"] == "completed"
    assert state[plan[1]["id"]]["status"] == "failed"
    cli.run_campaign(cfg, ["sr"])
    assert calls == [plan[0]["id"], plan[1]["id"], plan[1]["id"]]
    state = read_json(Path(cfg["run"]["output_root"]) / "state.json")
    assert state[plan[1]["id"]]["attempt"] == 2
    assert read_json(Path(cfg["run"]["output_root"]) / "report/summary.json")["completed"] == 2


def test_worker_executes_small_sr_trainer_with_frozen_manifest(tmp_path):
    from PIL import Image
    import numpy as np
    cfg = settings(tmp_path)
    job = jobs([42], ["sr"])[0]
    prep = prepared(tmp_path)
    paths = []
    rng = np.random.default_rng(7)
    for i in range(4):
        p = tmp_path / f"sample_{i}.png"
        Image.fromarray(rng.integers(0, 256, (64, 64), dtype=np.uint8)).save(p)
        paths.append(str(p))
    write_json(prep["sr_split"], {"train": paths[:2], "val": paths[2:3]})
    test_root = tmp_path / "test"
    test_root.mkdir()
    (test_root / "sample.png").symlink_to(paths[3])
    prep["sr_test_roots"] = {"Set5": str(test_root)}
    out = tmp_path / "attempt"
    resolved = materialize(job, cfg, prep, out, {})
    resolved["model"]["swinir"].update(embed_dim=12, depths=[1], num_heads=[3])
    resolved["training"].update(iterations=2, micro_batch_size=1, grad_accum=1, effective_batch_size=1,
                               num_workers=0, eval_num_workers=0, val_every=1, ckpt_every=1, log_every=1, amp_dtype="none")
    resolved["training"]["loss"].update(perceptual_weight=0, gan_weight=0)
    spec = tmp_path / "job.json"
    write_json(spec, {"job": job, "config": resolved, "output": str(out), "device": "cpu", "dependencies": {}})
    execute(spec)
    result = read_json(out / "swinir_wo_li/result.json")
    assert result["iterations_reached"] == 2 and result["per_dataset_best"]["Set5"]["images"] == 1
    assert record_artifacts(out)


def test_worker_executes_mcf_cnn_with_current_paths_and_data_seed(tmp_path):
    import numpy as np
    import tifffile
    cfg = settings(tmp_path)
    cfg["run"]["data_seed"] = 17
    job = next(j for j in jobs([43], ["mcf7"]) if j.get("condition") == "wcnn64")
    prep = prepared(tmp_path)
    rows = []
    rng = np.random.default_rng(8)
    for well in ["A01", "B01", "C01", "D01"]:
        p = tmp_path / f"plate_{well}_s1_w2abc.tif"
        tifffile.imwrite(p, rng.integers(0, 4000, (64, 64), dtype=np.uint16))
        rows.append({"image_file": str(p), "well": well})
    with Path(prep["mcf7_manifest"]).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_file", "well"])
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "attempt"
    resolved = materialize(job, cfg, prep, out, {})
    resolved["dataset"].update(num_train=5, num_val=2, num_test=2, verbose=False)
    resolved["training"].update(conv_batch_size=2, amp_dtype="none")
    resolved["algorithm1"].update(epochs=2, epoch_baseline=0, epoch_step=1)
    spec = tmp_path / "job.json"
    write_json(spec, {"job": job, "config": resolved, "output": str(out), "device": "cpu", "dependencies": {}})
    execute(spec)
    result = read_json(out / "result.json")
    assert result["epochs"] == 2
    assert all(row["train_images"] == 5 and row["opt_steps"] == 3 for row in result["history"])
    assert record_artifacts(out)
    loaders = module("scripts/figure08_mcf7/train.py")._loaders(resolved, 64, 2, 43, 4, 2, 2)
    assert loaders["test"].dataset.config.seed == 17


def test_segmentation_worker_trains_all_three_stages_without_old_checkpoint(tmp_path, monkeypatch):
    import training.dataloaders as data
    from torch.utils.data import TensorDataset
    from utils.experiment_config import sync_derived_config_fields
    cfg = settings(tmp_path)
    job = next(j for j in jobs([42], ["segmentation"]) if j["mode"] == "random_fixed")
    out = tmp_path / "attempt"
    resolved = materialize(job, cfg, prepared(tmp_path), out, {})
    resolved["dataset"]["image_size"] = 16
    resolved["forward_model"]["downscale_factor"] = 4
    resolved["pattern_generator"]["num_patterns"] = 2
    resolved["inverse_model"]["reconstruction"]["hidden_channels"] = [4, 4, 4, 4, 4, 1]
    resolved["segmentation_head"]["hidden_channels"] = [4, 4, 1]
    resolved["training"].update(batch_size=2, log_every=1)
    task = resolved["training"]["task_aware"]
    task["stage1"]["fixed_steps"] = 2
    task.update(seg_head_steps=2, finetune_steps=2, num_qualitative_samples=2)
    resolved = sync_derived_config_fields(resolved)
    arrays = {s: torch.rand(n, 1, 16, 16) for s, n in [("train", 4), ("val", 3), ("test", 3)]}
    monkeypatch.setattr(data, "build_dataset", lambda c, s: TensorDataset(arrays[s], (arrays[s] > .5).float()) if c["dataset"].get("return_mask") else arrays[s])
    spec = tmp_path / "job.json"
    write_json(spec, {"job": job, "config": resolved, "output": str(out), "device": "cpu", "dependencies": {}})
    execute(spec)
    assert (out / "stage1_content_aware/checkpoints/best.pt").exists()
    assert (out / "checkpoints/stage2_seg_head.pt").exists()
    assert (out / "checkpoints/stage3_finetuned.pt").exists()
    assert record_artifacts(out)


def test_refiner_worker_loads_explicit_seed43_dependency(tmp_path, monkeypatch):
    from models.microscope import DifferentiableMicroscope
    from utils.experiment_config import sync_derived_config_fields
    from baselines.swinir import fig3_refine_stage as stage
    cfg = settings(tmp_path)
    plan = jobs([43], ["content_swinir"])
    job = next(j for j in plan if j["engine"] == "refiner")
    base = tmp_path / "fresh_seed43_base"
    base.mkdir()
    base_cfg = materialize(next(j for j in plan if j["id"] == job["requires"][0]), cfg, prepared(tmp_path), base, {})
    base_cfg["dataset"]["image_size"] = 16
    base_cfg["forward_model"]["downscale_factor"] = 4
    base_cfg["pattern_generator"]["num_patterns"] = 2
    base_cfg["inverse_model"]["reconstruction"]["hidden_channels"] = [4, 4, 4, 4, 4, 1]
    base_cfg = sync_derived_config_fields(base_cfg)
    (base / "config.yaml").write_text(yaml.safe_dump(base_cfg))
    (base / "checkpoints").mkdir()
    model = DifferentiableMicroscope.from_run_config(base_cfg)
    torch.save({"model_state_dict": model.state_dict(), "sigmoid_m": 10.}, base / "checkpoints/best.pt")
    write_json(base / "metrics/run_summary.json", {"final_training_m": 10.})
    out = tmp_path / "attempt"
    resolved = materialize(job, cfg, prepared(tmp_path), out, {job["requires"][0]: str(base)})
    resolved["swinir"].update(embed_dim=12, depths=[1], num_heads=[3])
    resolved["train"].update(iterations=2, micro_batch_size=1, grad_accum=1, effective_batch_size=1,
                             val_every=1, ckpt_every=1, log_every=1, amp_dtype="none")
    resolved["loss"]["mode"] = "l1_only"
    tensors = torch.rand(3, 1, 64, 64)
    def cache(_model, run_cfg, split, *_args, **_kwargs):
        assert run_cfg["experiment"]["seed"] == 43
        return stage.PairCache(tensors * .8, tensors, torch.arange(3), split, 64)
    monkeypatch.setattr(stage, "build_pair_cache", cache)
    spec = tmp_path / "job.json"
    write_json(spec, {"job": job, "config": resolved, "output": str(out), "device": "cpu", "dependencies": {job["requires"][0]: str(base)}})
    execute(spec)
    assert record_artifacts(out)
