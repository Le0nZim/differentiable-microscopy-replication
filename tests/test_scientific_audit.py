"""Scientific regression tests for the September 2026 audit.

These check observable acquisition/evaluation behavior, not target rankings.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from datasets.bbbc022_hoechst import BBBC022HoechstConfig, assign_split_paths
from datasets.bbbc022_split import load_split
from evaluation.checkpoint_protocol import checkpoint_sigmoid_m
from evaluation.controlled import evaluate_controlled
from evaluation.eval_reconstruction import evaluate_reconstruction
from evaluation.metrics import ssim
from evaluation.tiled_inference import tiled_acquisition_budget
from models.acquisition import acquisition_metadata
from models.detector_noise import DetectorNoise, DetectorNoiseConfig
from models.forward_model import ForwardModel, ForwardModelConfig, sum_pool_nxn
from models.locality_upsampling import OriginalLocalityUpsampling, OriginalTransposeUpsampling
from models.microscope import DifferentiableMicroscope
from models.pattern_generator import PatternGenerator, PatternGeneratorConfig, SigmoidSchedule, _hadamard_patterns
from training.hardening_trainer import train_fixed_m_phase
from training.iteration import repeat_dataloader
from training.train_reconstruction import build_optimizer, save_checkpoint
from utils.experiment_config import load_experiment_config


def tiny_config():
    cfg = load_experiment_config("configs/audit/controlled_patchmnist.yaml")
    cfg["dataset"]["image_size"] = 16
    cfg["forward_model"].update(downscale_factor=4, dose_mode="fixed_peak")
    cfg["pattern_generator"].update(num_patterns=2, hadamard_tile_size=4)
    cfg["inverse_model"]["reconstruction"]["hidden_channels"] = [4, 4, 4, 4, 4, 1]
    cfg["detector_noise"].update(mode="noise_free", apply_noise=False)
    cfg["training"].update(log_every=1, illumination_lr=0.01)
    return cfg


def test_historical_drivers_now_prescribe_equal_updates_and_new_output_paths():
    def module_at(path):
        spec = importlib.util.spec_from_file_location(Path(path).stem + "_audit_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    f3 = module_at("scripts/figure03_content_aware/train_base.py")
    fixed = f3.build_config("x16", 8, 4, "random_fixed", 43, "cpu")
    learned = f3.build_config("x16", 8, 4, "learnable_frequency", 43, "cpu")
    phase = learned["training"]["staged_hardening"]
    total = phase["inverse_warmup_steps"] + phase["joint_soft_steps"] + len(phase["harden_m_values"]) * phase["harden_steps_per_m"]
    assert fixed["training"]["max_steps"] == total == 10100
    assert fixed["dataset"]["seed"] == learned["dataset"]["seed"] == 42
    assert "audited_v1" in str(f3.CONTENT_ROOT)
    t1 = module_at("scripts/table01_noise_robustness/run.py")
    runs = t1.build_run_specs()
    assert {r["config"]["training"]["max_steps"] for r in runs if r["pattern_mode"] == "random_fixed"} == {14850}
    assert "audited_v1" in str(t1.OUTPUT_ROOT)


@pytest.mark.parametrize("box", [False, True])
def test_checkpoint_restores_before_first_forward_without_resetting_psf(box):
    cfg = ForwardModelConfig(downscale_factor=2, use_impulse_psfs=not box)
    old = ForwardModel(cfg)
    x, h = torch.rand(2, 1, 8, 8), torch.rand(3, 1, 8, 8)
    if box:
        old.em_psf.copy_(torch.arange(9).reshape(1, 1, 3, 3) / 36)
    expected = old(x, h)
    fresh = ForwardModel(cfg)
    fresh.load_state_dict(old.state_dict(), strict=True)
    torch.testing.assert_close(fresh(x, h), expected)


def test_old_zero_placeholder_checkpoint_migration_is_narrow():
    model = ForwardModel(ForwardModelConfig())
    model.load_state_dict({"ex_psf": torch.zeros(1), "em_psf": torch.zeros(1)})
    assert model.ex_psf.shape == (1, 1, 3, 3)
    with pytest.raises(RuntimeError):
        model.load_state_dict({"ex_psf": torch.ones(1), "em_psf": torch.zeros(1)})


def test_asymmetric_psf_convolution_orientation():
    model = ForwardModel(ForwardModelConfig(downscale_factor=1))
    model.em_psf.zero_()
    model.em_psf[0, 0, 1, 2] = 1
    x = torch.zeros(1, 1, 7, 7)
    x[0, 0, 3, 3] = 1
    y = model(x, torch.ones_like(x))
    assert y[0, 0, 3, 4] == 1 and y.sum() == 1


def test_schedule_is_monotone_and_independent_of_call_count():
    schedule = SigmoidSchedule(20, 35, 5)
    assert [schedule.step(e) for e in (21, 40, 41, 45, 45, 50)] == [1, 2, 2, 3, 3, 4]
    assert SigmoidSchedule(20, 35, 5).step(50) == 4


def test_hadamard_rows_equal_independent_scipy_basis():
    from scipy.linalg import hadamard
    expected = torch.tensor((hadamard(64)[:8] + 1) / 2, dtype=torch.float32).reshape(8, 1, 8, 8)
    torch.testing.assert_close(_hadamard_patterns(8, 8, 8), expected)
    expected_tile = torch.tensor((hadamard(16)[:8] + 1) / 2, dtype=torch.float32).reshape(8, 1, 4, 4)
    torch.testing.assert_close(_hadamard_patterns(8, 8, 8, 4), expected_tile.repeat(1, 1, 2, 2))


def test_detector_tiled_hadamard_has_full_local_rank():
    p = _hadamard_patterns(16, 256, 256, 4)
    local_matrix = p[:, 0, :4, :4].reshape(16, 16)
    assert torch.linalg.matrix_rank(local_matrix) == 16
    # In the historical global row ordering, these first rows never encode y.
    old = _hadamard_patterns(16, 256, 256)[:, 0, :4, :4].reshape(16, 16)
    assert torch.linalg.matrix_rank(old) < 16


def test_hadamard_ordering_difference_does_not_affect_active_four_pattern_sweep():
    for d in (8, 16, 32, 64):
        torch.testing.assert_close(_hadamard_patterns(4, 256, 256, d),
                                   _hadamard_patterns(4, 256, 256))


def test_superpixel_constant_masks_lose_sub_bin_information():
    x1 = torch.zeros(1, 1, 4, 4); x1[0, 0, 0, 0] = 1
    x2 = torch.zeros_like(x1); x2[0, 0, 0, 1] = 1
    forward = ForwardModel(ForwardModelConfig(downscale_factor=4))
    coarse = torch.ones(2, 1, 4, 4)
    torch.testing.assert_close(forward(x1, coarse), forward(x2, coarse))
    encoded = _hadamard_patterns(2, 4, 4, 4)
    assert not torch.equal(forward(x1, encoded), forward(x2, encoded))


def test_binary_training_matches_deployed_forward_and_has_gradient():
    p = PatternGenerator(PatternGeneratorConfig(mode="learnable_spatial", num_patterns=3,
                                               height=16, width=16, binarization="ste"))
    h = p()
    assert torch.all((h == 0) | (h == 1))
    h.sum().backward()
    assert torch.isfinite(p.tau.grad).all() and p.tau.grad.abs().sum() > 0
    p.eval()
    assert torch.equal(h, p())


def test_mean_incident_dose_is_equal_but_detected_counts_need_not_be():
    uniform = torch.ones(2, 1, 4, 4)
    patterned = uniform.clone(); patterned[:, :, :, 2:] = 0
    for h in (uniform, patterned):
        a = acquisition_metadata(h, downscale=4, photon_scale=100, dose_mode="equal_mean", sequence_dose=1)
        assert a["incident_sequence_mean_photon_scale"] == 100
        assert a["scalar_measurements"] == 2 and a["compression"] == 8
    f = ForwardModel(ForwardModelConfig(downscale_factor=4, dose_mode="equal_mean", sequence_dose=1))
    x = torch.zeros(1, 1, 4, 4); x[:, :, :, :2] = 1
    assert f(x, patterned).sum() > f(x, uniform).sum()


def test_exact_poisson_gaussian_mean_and_variance():
    torch.manual_seed(221)
    noise = DetectorNoise(DetectorNoiseConfig(mode="exact_poisson_plus_read", noise_normalization="paper_v3",
                                             photon_count=20, gamma=10, sigma_read=3))
    x = torch.full((160000, 1, 1, 1), 0.4)
    y = noise(x)
    assert abs(float(y.mean()) - 0.9) < 0.004
    assert abs(float(y.var()) - (20*0.4 + 10 + 9)/20**2) < 0.002
    with pytest.raises(ValueError, match="evaluation"):
        noise(x.requires_grad_())


def test_bad_noise_units_cannot_silently_fall_back_to_legacy():
    with pytest.raises(ValueError, match="normalization"):
        DetectorNoise(DetectorNoiseConfig(noise_normalization="paepr_v3"))


def test_zero_signal_zero_background_has_finite_surrogate_gradient():
    noise = DetectorNoise(DetectorNoiseConfig(mode="differentiable_poisson", noise_normalization="paper_v3", gamma=0))
    x = torch.zeros(1, 1, 2, 2, requires_grad=True)
    y = noise(x, poisson_noise=torch.ones_like(x))
    assert torch.count_nonzero(y) == 0
    y.sum().backward()
    # The clamp's boundary subgradient is backend/version dependent; it must
    # be finite, while positive physical signals must still receive gradient.
    assert torch.isfinite(x.grad).all()
    positive = torch.ones_like(x, requires_grad=True)
    noise(positive, poisson_noise=torch.zeros_like(positive)).sum().backward()
    torch.testing.assert_close(positive.grad, torch.ones_like(positive))


def test_repeat_loader_gets_fresh_samples_instead_of_cached_tensors():
    class Counter:
        calls = 0
        def __len__(self): return 2
        def __getitem__(self, index):
            self.calls += 1
            return self.calls
    it = repeat_dataloader(DataLoader(Counter(), batch_size=2))
    assert next(it).tolist() == [1, 2]
    assert next(it).tolist() == [3, 4]
    with pytest.raises(ValueError, match="empty"):
        next(repeat_dataloader(DataLoader([])))


def test_metric_mean_does_not_overweight_last_short_batch():
    class ZeroModel(torch.nn.Module):
        def forward(self, x, **kwargs): return {"x_recon": torch.zeros_like(x)}
    x = torch.zeros(5, 1, 16, 16); x[-1] = 1
    values = [evaluate_reconstruction(ZeroModel(), DataLoader(x, batch_size=b), torch.device("cpu"))
              for b in (1, 2, 4, 5)]
    for m, s in values:
        assert m == pytest.approx(0.2)
        assert s == pytest.approx(values[0][1], abs=1e-6)


def test_ssim_border_convention_is_explicit():
    a, b = torch.zeros(1, 1, 16, 16), torch.ones(1, 1, 16, 16)
    reflect = ssim(a, b, padding_mode="reflect")
    assert float(reflect) == pytest.approx(0.0001 / 1.0001, rel=1e-4)
    assert float(ssim(a, b)) != pytest.approx(float(reflect), rel=1e-3)


def test_checkpoint_stores_physical_sharpness_separately_from_epoch(tmp_path):
    cfg = tiny_config(); model = DifferentiableMicroscope.from_run_config(cfg)
    opt = build_optimizer(model, cfg)
    model.pattern_generator.sigmoid_m = 8
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, opt, 45, cfg, step=123)
    payload = torch.load(path, weights_only=False)
    assert payload["epoch"] == 45 and payload["step"] == 123
    assert checkpoint_sigmoid_m(payload) == 8
    with pytest.raises(ValueError, match="sharpness"):
        checkpoint_sigmoid_m({"epoch": 8})


def test_selected_phase_restores_matching_adam_state(tmp_path, monkeypatch):
    import training.hardening_trainer as trainer
    cfg = tiny_config(); model = DifferentiableMicroscope.from_run_config(cfg)
    opt = build_optimizer(model, cfg)
    loader = DataLoader(torch.rand(4, 1, 16, 16), batch_size=2)
    scores = iter([(1.0, 0.5), (2.0, 0.5), (3.0, 0.5)])
    monkeypatch.setattr(trainer, "_evaluate_loader", lambda *a, **k: next(scores))
    train_fixed_m_phase(model, loader, loader, opt, torch.device("cpu"), cfg, tmp_path,
                        sigmoid_m=4, max_steps=3, phase_name="phase")
    best = torch.load(tmp_path / "checkpoints/phase/best.pt", weights_only=False)
    last = torch.load(tmp_path / "checkpoints/phase/last.pt", weights_only=False)
    assert best["step"] == 1 and last["step"] == 3
    assert {int(s["step"]) for s in best["optimizer_state_dict"]["state"].values()} == {1}
    assert {int(s["step"]) for s in last["optimizer_state_dict"]["state"].values()} == {3}
    assert best["sigmoid_m"] == 4


def test_controlled_eval_restores_training_rng_mode_and_config():
    cfg = tiny_config(); cfg["detector_noise"].update(mode="differentiable_poisson_plus_read", apply_noise=True)
    model = DifferentiableMicroscope.from_run_config(cfg).train()
    x = torch.rand(5, 1, 16, 16); loader = DataLoader(x, batch_size=2)
    rng = torch.get_rng_state().clone(); state = copy.deepcopy(model.state_dict())
    a = evaluate_controlled(model, loader, torch.device("cpu"), sigmoid_m=8, noise_seed=13,
                            noise_mode="exact_poisson_plus_read")
    b = evaluate_controlled(model, loader, torch.device("cpu"), sigmoid_m=8, noise_seed=13,
                            noise_mode="exact_poisson_plus_read")
    assert a == b and model.training
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.detector_noise.config.mode == "differentiable_poisson_plus_read"
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())


def test_original_locality_combines_channels_at_one_detector_site():
    module = OriginalLocalityUpsampling(3, 2, 2, 2)
    with torch.no_grad():
        module.weights.zero_(); module.bias.zero_()
        module.weights[1, 0] = torch.tensor([1, 2, 3, 4])
        module.weights[1, 2] = torch.tensor([4, 3, 2, 1])
    x = torch.zeros(1, 3, 2, 2); x[0, 0, 0, 1] = 2; x[0, 2, 0, 1] = 3
    image = module.project(x)
    torch.testing.assert_close(image[0, 0, :2, 2:], torch.tensor([[14., 13.], [12., 11.]]))
    assert image.sum() == 50


def test_original_transpose_validation_does_not_update_batchnorm():
    module = OriginalTransposeUpsampling(2, 8).eval()
    state = copy.deepcopy(module.state_dict())
    assert module(torch.rand(3, 2, 2, 2)).shape == (3, 2, 16, 16)
    assert all(torch.equal(v, module.state_dict()[k]) for k, v in state.items())


def test_overlap_reacquisition_lowers_effective_compression():
    a = tiled_acquisition_budget(256, 1280, 256, 8, 4)
    b = tiled_acquisition_budget(256, 1280, 256, 8, 4, overlap=64)
    assert a["effective_field_compression"] == 16
    assert a["acquired_tiles"] == 5 and b["acquired_tiles"] == 7
    assert b["effective_field_compression"] == pytest.approx(80/7)


def test_incomplete_detector_bins_are_rejected():
    with pytest.raises(ValueError, match="complete"):
        sum_pool_nxn(torch.ones(1, 1, 7, 8), 4)


def test_split_validation_checks_wells_not_just_file_count(tmp_path):
    paths = [Path(f"IXMtest_A01_s{i}_w1.tif") for i in range(1, 5)]
    cfg = BBBC022HoechstConfig(num_train_images=1, num_val_images=1, num_test_images=1)
    with pytest.raises(ValueError, match="distinct wells"):
        assign_split_paths(paths, cfg)
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"splits": {"train": [str(paths[0])], "val": [str(paths[1])], "test": ["IXMtest_A02_s1_w1.tif"]}}))
    with pytest.raises(ValueError, match="leakage"):
        load_split(path, tmp_path)
