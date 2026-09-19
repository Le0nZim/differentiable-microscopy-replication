"""Regressions prompted by the September 19 workstation results."""
import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from datasets.bbbc022_hoechst import BBBC022HoechstConfig, BBBC022HoechstDataset
from datasets.bbbc022_preproc_ablation import PreprocAblationConfig, PreprocAblationDataset
from evaluation.eval_reconstruction import evaluate_reconstruction
from evaluation import pattern_hardening as variants
from models.microscope import DifferentiableMicroscope
from training.pattern_tracking import capture_detector_snapshot
from utils.experiment_config import load_experiment_config
from workflow.common import module


def tiny_config():
    cfg = load_experiment_config("configs/audit/controlled_patchmnist.yaml")
    cfg["experiment"].update(device="cpu", run_id="regression")
    cfg["dataset"]["image_size"] = 16
    cfg["forward_model"].update(downscale_factor=4, dose_mode="fixed_peak")
    cfg["pattern_generator"].update(num_patterns=2, hadamard_tile_size=4, binarization="soft")
    cfg["inverse_model"]["reconstruction"]["hidden_channels"] = [4, 4, 4, 4, 4, 1]
    cfg["detector_noise"].update(mode="noise_free", apply_noise=False)
    cfg["training"].update(batch_size=2, log_every=1, illumination_lr=.01)
    return cfg


@pytest.mark.parametrize("training", [False, True])
def test_pattern_diagnostics_and_export_do_not_change_bn_or_modes(tmp_path, monkeypatch, training):
    model = DifferentiableMicroscope.from_run_config(tiny_config())
    model.train(training)
    model.inverse_model.reconstruction.net[2].eval()  # preserve a mixed-mode tree
    modes = [m.training for m in model.modules()]
    before = copy.deepcopy(model.state_dict())
    x = torch.rand(3, 1, 16, 16)
    monkeypatch.setattr(variants, "save_pattern_inspection", lambda *a, **k: None)
    monkeypatch.setattr(variants, "save_measurement_grid", lambda *a, **k: None)
    capture_detector_snapshot(model, x, sigmoid_m=8, apply_noise=False)
    variants.evaluate_all_pattern_variants(model, {"val": DataLoader(x, batch_size=2)}, torch.device("cpu"),
                                          training_m=8, reference_specimen=x, apply_noise=False)
    variants.save_variant_artifacts(model, x, tmp_path, training_m=8, apply_noise=False)
    assert modes == [m.training for m in model.modules()]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_artifact_exception_restores_model_modes(tmp_path, monkeypatch):
    model = DifferentiableMicroscope.from_run_config(tiny_config()).train()
    model.inverse_model.eval()
    modes = [m.training for m in model.modules()]
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(variants, "save_pattern_inspection", fail)
    with pytest.raises(OSError, match="disk full"):
        variants.save_variant_artifacts(model, torch.rand(2, 1, 16, 16), tmp_path, training_m=8)
    assert modes == [m.training for m in model.modules()]


def test_staged_summary_replays_the_saved_checkpoint(tmp_path, monkeypatch):
    import training.dataloaders as data
    from training.staged_hardening_train import train_staged_hardening
    cfg = tiny_config()
    cfg["experiment"]["results_csv"] = str(tmp_path / "results.csv")
    cfg["training"]["staged_hardening"] = {
        "inverse_warmup_steps": 1, "joint_soft_steps": 1,
        "harden_m_values": [2], "harden_steps_per_m": 1,
    }
    gen = torch.Generator().manual_seed(917)
    arrays = {s: torch.rand(n, 1, 16, 16, generator=gen) for s, n in [("train", 4), ("val", 3), ("test", 5)]}
    monkeypatch.setattr(data, "build_dataset", lambda c, s: arrays[s])
    monkeypatch.setattr(variants, "save_pattern_inspection", lambda *a, **k: None)
    monkeypatch.setattr(variants, "save_measurement_grid", lambda *a, **k: None)
    summary = train_staged_hardening(cfg, tmp_path)
    saved = torch.load(tmp_path / "checkpoints/best.pt", weights_only=False)
    model = DifferentiableMicroscope.from_run_config(cfg)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    metrics = evaluate_reconstruction(model, DataLoader(arrays["test"], batch_size=2), torch.device("cpu"),
                                      sigmoid_m=saved["sigmoid_m"], apply_noise=False)
    assert metrics == pytest.approx((summary["test_mse"], summary["test_ssim"]), abs=1e-8)


def synthetic_dataset(kind, split="train", varying=True):
    if kind == "segmentation":
        ds = BBBC022HoechstDataset.__new__(BBBC022HoechstDataset)
        ds.config = BBBC022HoechstConfig(patch_size=16, epoch_varying_train_crops=varying,
                                         return_mask=True, mask_mode="trackmate")
    else:
        ds = PreprocAblationDataset.__new__(PreprocAblationDataset)
        ds.config = PreprocAblationConfig(patch_size=16, epoch_varying_train_crops=varying)
    ds.split = split
    ds.images = [torch.arange(4096).reshape(1, 64, 64).float() / 4096]
    ds.mask_full = [(ds.images[0][0] > .5).float()]
    return ds


@pytest.mark.parametrize("kind", ["segmentation", "reconstruction"])
def test_crops_vary_reproducibly_without_model_rng_and_keep_mask_alignment(kind):
    a, b = synthetic_dataset(kind), synthetic_dataset(kind)
    samples = []
    for _ in range(5):
        xa = a[0]
        torch.rand(173)  # model/noise randomness must not choose the other crop
        xb = b[0]
        if kind == "segmentation":
            torch.testing.assert_close(xa[1], (xa[0] > .5).float(), rtol=0, atol=0)
            torch.testing.assert_close(xa[1], xb[1], rtol=0, atol=0)
            xa, xb = xa[0], xb[0]
        torch.testing.assert_close(xa, xb, rtol=0, atol=0)
        samples.append(xa.numpy().tobytes())
    assert len(set(samples)) > 1
    for split in ("val", "test"):
        ds = synthetic_dataset(kind, split)
        a, b = ds[0], ds[0]
        if kind == "segmentation":
            a, b = a[0], b[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_refiner_gate_is_a_constraint_and_empty_evaluation_is_rejected():
    from baselines.swinir.fig3_refine_stage import PairCache, eval_cache, refinement_selection_score
    assert refinement_selection_score({"ref_ssim": .99, "ref_mse": .2}, base_mse=.1, mse_gate=True) is None
    assert refinement_selection_score({"ref_ssim": .7, "ref_mse": .05}, base_mse=.1, mse_gate=True) == .7
    cache = PairCache(torch.empty(0, 1, 16, 16), torch.empty(0, 1, 16, 16), torch.empty(0), "test", 16)
    with pytest.raises(ValueError, match="empty"):
        eval_cache(torch.nn.Identity(), cache, torch.device("cpu"))


def test_mcf7_metrics_weight_images_and_preserve_modes():
    from types import SimpleNamespace
    impl = module("scripts/figure08_mcf7/train.py")
    adapter = SimpleNamespace(model=torch.nn.Identity().eval(), forward=lambda x, m: torch.zeros_like(x))
    x = torch.tensor([.1, .2, .8]).view(3, 1, 1, 1).expand(-1, 1, 16, 16)
    a = impl._evaluate(adapter, DataLoader(x, batch_size=1), torch.device("cpu"), 8, None)
    b = impl._evaluate(adapter, DataLoader(x, batch_size=2), torch.device("cpu"), 8, None)
    assert a == pytest.approx(b)
    assert b["mse"] == pytest.approx((.01 + .04 + .64) / 3)
    assert b["num_images"] == 3 and not adapter.model.training
    capped = impl._evaluate(adapter, DataLoader(x, batch_size=2), torch.device("cpu"), 8, None, max_items=1)
    assert capped["num_images"] == 1
    with pytest.raises(ValueError, match="empty"):
        impl._evaluate(adapter, DataLoader(x[:0]), torch.device("cpu"), 8, None)


def test_refiner_selects_initial_candidate_and_saves_gan_state(tmp_path, monkeypatch):
    import yaml
    impl = module("scripts/figure03_content_aware/train_swinir.py")
    cfg = yaml.safe_load(Path("configs/figure03_content_aware/paper_faithful_pixel_perceptual_gan.yaml").read_text())
    cfg["train"].update(iterations=2, micro_batch_size=1, grad_accum=1, effective_batch_size=1,
                         val_every=1, ckpt_every=1, log_every=1, amp_dtype="none", warmup=0)
    class Refiner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.swinir = torch.nn.Conv2d(1, 1, 1)
            torch.nn.init.zeros_(self.swinir.weight)
            torch.nn.init.zeros_(self.swinir.bias)
        def forward(self, x):
            return x + self.swinir(x)
        def parameters_for_training(self):
            return list(self.parameters())
    base = torch.nn.Identity().eval()
    monkeypatch.setattr(impl.S, "load_frozen_base", lambda *a, **k: (base, {}, 8))
    monkeypatch.setattr(impl.S, "build_pair_cache", lambda *a, **k: [0])
    monkeypatch.setattr(impl.S, "build_refiner", lambda *a, **k: Refiner())
    monkeypatch.setattr(impl.S, "sample_patch_batch", lambda *a, **k: (torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)))
    def fake_eval(refiner, *args, **kwargs):
        initial = all(torch.count_nonzero(p) == 0 for p in refiner.parameters())
        return {"n": 1, "base_mse": .1, "base_ssim": .9, "base_psnr": 10.,
                "ref_mse": .1 if initial else .05, "ref_ssim": .9 if initial else .8, "ref_psnr": 10.}
    monkeypatch.setattr(impl.S, "eval_cache", fake_eval)
    monkeypatch.setattr(impl, "save_val_grid", lambda *a, **k: None)
    disc = torch.nn.Conv2d(1, 1, 1)
    monkeypatch.setattr(impl.S, "make_loss", lambda *a, **k: {"mode": "paper_pixel_perceptual_gan", "stack": {
        "discriminator": disc, "pixel_weight": 1., "pixel_kind": "l1", "gan_weight": .1,
        "gan_loss": lambda logits, real: (logits - float(real)).square().mean(),
    }})
    result = impl.train_cell(cfg, "x16", "random_fixed", torch.device("cpu"), tmp_path)
    assert result["best_step"] == 0 and result["selected_initial_checkpoint"]
    last_path = tmp_path / "x16/random_fixed/checkpoints/last.pt"
    last = torch.load(last_path, weights_only=False)
    assert last["discriminator"] and last["opt_d"]
    assert "patch_rng_state" in last
    del last["discriminator"]
    torch.save(last, last_path)
    with pytest.raises(ValueError, match="discriminator weights"):
        impl.train_cell(cfg, "x16", "random_fixed", torch.device("cpu"), tmp_path, iterations_override=3, resume=True)


def test_snapshot_inspector_checks_tables_and_distinguishes_lfs_pointers(tmp_path):
    inspector = module("scripts/audit/inspect_campaign_snapshot.py")
    out = tmp_path / "jobs/content/random_seed42/attempt_001"
    out.mkdir(parents=True)
    result = out / "result.json"
    result.write_text('{"test_mse": 0.2}\n')
    checksum = hashlib.sha256(result.read_bytes()).hexdigest()
    weight_hash = "a" * 64
    (out / "best.pt").write_text(f"version https://git-lfs.github.com/spec/v1\noid sha256:{weight_hash}\nsize 2000\n")
    (tmp_path / "campaign.json").write_text(json.dumps({"git_commit": "example", "source_sha256": "b" * 64}))
    (tmp_path / "plan.json").write_text('[{"id": "content/random_seed42"}]')
    (tmp_path / "state.json").write_text(json.dumps({"content/random_seed42": {
        "status": "completed", "attempt": 1, "artifacts": [
            {"path": "result.json", "bytes": result.stat().st_size, "sha256": checksum},
            {"path": "best.pt", "bytes": 2000, "sha256": weight_hash},
        ]}}))
    report = tmp_path / "report"
    report.mkdir()
    row = {"stage": "content", "condition": "random", "metric": "test_mse", "seed": "42",
           "value": .2, "job": "content/random_seed42", "source": str(result.relative_to(tmp_path))}
    def write_csv(path, data):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data))
            writer.writeheader()
            writer.writerow(data)
    write_csv(report / "metrics_by_seed.csv", row)
    write_csv(report / "aggregate.csv", {"stage": "content", "condition": "random", "metric": "test_mse",
                                         "n_seeds": 1, "mean": .2, "std": ""})
    checked = inspector.inspect_campaign(tmp_path)
    assert checked["artifact_integrity"] == {"bytes_verified": 1, "lfs_pointer_only": 1}
    assert not checked["artifact_problems"] and not checked["metric_problems"] and not checked["aggregate_problems"]
    assert not checked["checkpoint_replay_performed"]
    row["value"] = .3
    write_csv(report / "metrics_by_seed.csv", row)
    checked = inspector.inspect_campaign(tmp_path)
    assert checked["metric_problems"] and checked["aggregate_problems"]
    result.write_text('{"test_mse": 0.3}\n')  # same size, wrong hash
    checked = inspector.inspect_campaign(tmp_path)
    assert checked["artifact_problems"][0]["status"] == "mismatch"
