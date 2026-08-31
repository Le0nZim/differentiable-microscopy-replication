"""Udith legacy-schedule experiment: phases, logging opt-in, paired init, frozen artifacts."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
from pathlib import Path

import pytest
import torch

from models.microscope import DifferentiableMicroscope
from training.paired_pattern_init import (
    apply_shared_tau0,
    generate_shared_tau0,
    paired_initialization_audit,
)
from training.udith_legacy_schedule import (
    EXPECTED_GLOBAL_BOUNDS,
    m_sequence,
    phase_global_bounds,
    schedule_provenance,
    udith_legacy_phases,
    udith_legacy_phases_smoke,
)
from utils.experiment_config import load_experiment_config
from utils.reproducibility import set_seed
from training.dataloaders import build_dataloader

ROOT = Path(__file__).resolve().parents[1]


def _load_am3():
    spec = importlib.util.spec_from_file_location(
        "table03_run_for_test", ROOT / "scripts/table03_ablation/run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


FROZEN_HASHES = {
    "experiments/figure10_ablation_patchmnist/runs/aggregate_multiseed.json":
        "2a52f00e73e2c0434d8b9f88ce215b79",
    "experiments/figure10_ablation_patchmnist/runs/aggregate_summary.json":
        "9ef4942c9163618a1b871b0c04afecbe",
    "experiments/figure10_ablation_patchmnist/runs/C_seed42/metrics/run_summary.json":
        "4cf4a59ccf58132dd4758f43ddb97b0e",
    "experiments/figure10_ablation_patchmnist/runs/D_seed42/metrics/run_summary.json":
        "0dbd01f36139d901d26b415370c99b92",
    "experiments/figure10_ablation_patchmnist/figures/figure10_paper_style.png":
        "3df9674f732c356f6f77e5413cf72ada",
    "experiments/figure10_ablation_patchmnist/figures/figure10_table3_comparison.png":
        "1da4b4ce4ef54ac2c409a1e1bc75a625",
    "experiments/figure10_ablation_patchmnist/FIG10_REPORT.md":
        "a97d521ee531ea0f77391164812555bd",
    "configs/figure10_ablation_patchmnist/ablation.yaml":
        "524d9714798e91d22019fe2f8cb6a095",
}


def test_total_schedule_length_is_121500():
    phases = udith_legacy_phases()
    assert sum(p["steps"] for p in phases) == 121_500


def test_illumination_frozen_exactly_60750_steps():
    phases = udith_legacy_phases()
    assert sum(p["steps"] for p in phases if p["freeze_illum"]) == 60_750


def test_illumination_trainable_exactly_60750_steps():
    phases = udith_legacy_phases()
    assert sum(p["steps"] for p in phases if not p["freeze_illum"]) == 60_750


def test_m_sequence_is_1_through_8():
    assert m_sequence(udith_legacy_phases()) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


def test_global_step_boundaries_match_table():
    bounds = phase_global_bounds(udith_legacy_phases())
    assert len(bounds) == len(EXPECTED_GLOBAL_BOUNDS)
    for got, expected in zip(bounds, EXPECTED_GLOBAL_BOUNDS):
        assert got["name"] == expected["name"]
        assert got["m"] == expected["m"]
        assert got["start"] == expected["start"]
        assert got["end"] == expected["end"]
        assert got["freeze_illum"] is expected["freeze_illum"]


def test_five_step_m8_phase_retained():
    m8 = [p for p in udith_legacy_phases() if p["name"] == "harden_m8"]
    assert len(m8) == 1
    assert m8[0]["steps"] == 5
    assert m8[0]["m"] == 8.0
    bounds = phase_global_bounds(udith_legacy_phases())
    assert bounds[-1]["start"] == 121_496
    assert bounds[-1]["end"] == 121_500


def test_smoke_phases_visit_every_named_phase():
    full = [p["name"] for p in udith_legacy_phases()]
    smoke = [p["name"] for p in udith_legacy_phases_smoke()]
    assert smoke == full
    assert [p["m"] for p in udith_legacy_phases_smoke()] == [p["m"] for p in udith_legacy_phases()]
    assert [p["freeze_illum"] for p in udith_legacy_phases_smoke()] == [
        p["freeze_illum"] for p in udith_legacy_phases()
    ]


def test_default_phases_still_8500():
    am3 = _load_am3()
    phases = am3.default_phases(1.0)
    assert [(p["name"], p["m"], p["steps"], p["freeze_illum"]) for p in phases] == [
        ("inverse_warmup", 1.0, 1500, True),
        ("joint_soft", 1.0, 4000, False),
        ("harden_m2", 2.0, 1000, False),
        ("harden_m4", 4.0, 1000, False),
        ("harden_m8", 8.0, 1000, False),
    ]
    assert sum(p["steps"] for p in phases) == 8500
    udith_names = [p["name"] for p in udith_legacy_phases()]
    assert [p["name"] for p in phases] != udith_names


def test_should_log_step_default_matches_legacy_predicate():
    am3 = _load_am3()
    # Historical: log iff step % log_every == 0 OR (last phase AND last step of that phase).
    log_every = 200
    cases = [
        (200, False, False, True),
        (1500, True, False, False),   # phase end of warmup: NOT logged by default
        (5500, True, False, False),
        (8500, True, True, True),     # last step of last phase
        (199, False, False, False),
        (1, False, False, False),
    ]
    for step, phase_end, last_phase, expected in cases:
        got = am3.should_log_step(
            step, log_every,
            is_phase_last_step=phase_end,
            is_last_phase=last_phase,
            log_phase_boundaries=False,
        )
        assert got is expected, (step, phase_end, last_phase, got)
    # Opt-in logs every phase end.
    assert am3.should_log_step(
        1500, 200, is_phase_last_step=True, is_last_phase=False, log_phase_boundaries=True
    ) is True


def test_run_one_opt_in_defaults_preserve_old_path():
    am3 = _load_am3()
    params = inspect.signature(am3.run_one).parameters
    assert params["log_phase_boundaries"].default is False
    assert params["extra_physical_metrics"].default is False
    assert params["shared_tau0"].default is None
    assert params["save_config_yaml"].default is False
    assert params["schedule_provenance"].default is None
    assert params["warmup_checkpoint_out"].default is None
    assert params["resume_from_warmup"].default is None


def test_provenance_records_five_batches_and_sources():
    prov = schedule_provenance()
    assert prov["source_repository"] == "udithhaputhanthri/CompressiveDabbaMu"
    assert prov["source_commit"] == "d4308c0"
    assert prov["source_notebook"] == "4_20230102_ablation-2.ipynb"
    assert prov["five_batches_per_epoch_derivation"]["updates_per_legacy_epoch"] == 5
    assert prov["converted_step_counts"]["total_steps"] == 121_500
    assert prov["uses_sigmoid_schedule_step"] is False
    assert prov["m_sequence"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


def test_existing_patchmnist_fig10_outputs_byte_for_byte():
    for rel, expected in FROZEN_HASHES.items():
        path = ROOT / rel
        assert path.exists(), f"missing frozen artifact {rel}"
        assert _md5(path) == expected, f"hash changed for {rel}"


def test_new_config_matches_patchmnist_protocol_except_schedule():
    old = load_experiment_config(ROOT / "configs/figure10_ablation_patchmnist/ablation.yaml")
    new = load_experiment_config(
        ROOT / "configs/figure10_ablation_patchmnist_udith_schedule/ablation.yaml"
    )
    for key in ("name", "data_root", "image_size", "digit_size", "grid_size",
                "num_train", "num_val", "num_test", "disjoint_val_test"):
        assert old["dataset"][key] == new["dataset"][key]
    assert old["forward_model"] == new["forward_model"]
    assert old["detector_noise"] == new["detector_noise"]
    assert old["training"]["batch_size"] == new["training"]["batch_size"] == 32
    assert old["training"]["illumination_lr"] == new["training"]["illumination_lr"] == 1.0
    assert old["training"]["inverse_lr"] == new["training"]["inverse_lr"] == 0.001
    assert old["training"]["gradient_clip_norm"] == new["training"]["gradient_clip_norm"] == 1.0
    assert old["training"]["loss"] == new["training"]["loss"] == "l1"
    assert old["pattern_generator"]["num_patterns"] == new["pattern_generator"]["num_patterns"] == 4
    assert old["forward_model"]["downscale_factor"] == 8
    assert new["inverse_model"]["upsampling"]["mode"] == "locality_aware"


def _paired_audit_for_seed(seed: int) -> dict:
    am3 = _load_am3()
    base = load_experiment_config(
        ROOT / "configs/figure10_ablation_patchmnist_udith_schedule/ablation.yaml"
    )
    base["experiment"]["device"] = "cpu"
    cfg_c = am3.apply_variant(base, "C")
    cfg_d = am3.apply_variant(base, "D")
    for cfg in (cfg_c, cfg_d):
        cfg["experiment"]["seed"] = seed
        cfg["dataset"]["seed"] = seed
        cfg["pattern_generator"]["seed"] = seed
    tau0 = generate_shared_tau0(4, 256, 256, seed)
    device = torch.device("cpu")

    set_seed(seed)
    model_c = DifferentiableMicroscope.from_run_config(cfg_c).to(device)
    apply_shared_tau0(model_c, tau0)
    batch_c = next(iter(build_dataloader(cfg_c, "train")))

    set_seed(seed)
    model_d = DifferentiableMicroscope.from_run_config(cfg_d).to(device)
    apply_shared_tau0(model_d, tau0)
    batch_d = next(iter(build_dataloader(cfg_d, "train")))

    return paired_initialization_audit(
        model_c, model_d, first_batch_c=batch_c, first_batch_d=batch_d
    )


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_paired_cd_initialization_audit(seed):
    audit = _paired_audit_for_seed(seed)
    assert audit["pass"], audit["problems"]
    assert audit["C_has_W"] is True
    assert audit["D_has_tau"] is True
    assert audit["C_has_tau_param"] is False
    assert audit["D_has_W"] is False
    assert audit["max_abs_tau_difference"] < 1e-5
    assert audit["max_abs_Ht_difference"] < 1e-5


def test_step_121500_is_always_logged():
    am3 = _load_am3()
    # 121500 is NOT on the log_every=200 grid (121500 % 200 == 100).
    # It is logged because it is the last step of the last phase, and also
    # because log_phase_boundaries=True logs every phase end.
    assert 121_500 % 200 == 100
    assert am3.should_log_step(
        121_500, 200,
        is_phase_last_step=True, is_last_phase=True, log_phase_boundaries=False,
    )
    assert am3.should_log_step(
        121_500, 200,
        is_phase_last_step=True, is_last_phase=True, log_phase_boundaries=True,
    )
    assert am3.should_log_step(
        121_500, 200,
        is_phase_last_step=True, is_last_phase=False, log_phase_boundaries=True,
    )


def test_run_one_still_uses_itertools_cycle():
    text = (ROOT / "scripts/table03_ablation/run.py").read_text()
    assert "train_iter = itertools.cycle(train_loader)" in text


def test_phase_boundary_evals_share_best_update_path():
    text = (ROOT / "scripts/table03_ablation/run.py").read_text()
    assert "eligible for global-best checkpoint selection" in text
    assert "if val_mse < best[\"val_mse\"]:" in text


def test_compare_frozen_interval_flags_material_divergence():
    from training.shared_warmup_checkpoint import compare_frozen_interval

    aligned = [
        {"step": "200", "loss": "0.1", "train_mse": "0.05", "val_mse": "0.06",
         "val_ssim": "0.5", "grad_norm_upsampler": "0.01", "grad_norm_recon": "0.1",
         "H_t_min": "0", "H_t_max": "1", "H_t_mean": "0.5", "H_t_std": "0.2",
         "H_t_binary_fraction": "0.03", "tau_displacement": "0",
         "H_t_displacement": "0", "illum_delta": "0"},
    ]
    cmp_ok = compare_frozen_interval(aligned, aligned)
    assert cmp_ok["pass"] is True
    assert cmp_ok["materially_diverged"] is False

    d = dict(aligned[0])
    d["loss"] = "0.12"
    d["val_mse"] = "0.07"
    cmp_bad = compare_frozen_interval(aligned, [d])
    assert cmp_bad["materially_diverged"] is True
    assert cmp_bad["pass"] is False
    assert cmp_bad["max_abs"]["loss"] == pytest.approx(0.02)
    assert cmp_bad["max_abs"]["val_mse"] == pytest.approx(0.01)


def test_adam_inverse_snapshot_roundtrip():
    from training.shared_warmup_checkpoint import load_adam_inverse, snapshot_adam_inverse

    torch.manual_seed(0)
    inverse = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))
    illum = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(
        [{"params": list(illum.parameters()), "lr": 1.0},
         {"params": list(inverse.parameters()), "lr": 0.001}]
    )
    x = torch.randn(5, 8)
    opt.zero_grad()
    inverse(x).sum().backward()
    opt.step()
    snap = snapshot_adam_inverse(opt, inverse)
    assert snap
    inverse2 = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))
    inverse2.load_state_dict(inverse.state_dict())
    illum2 = torch.nn.Linear(2, 2)
    opt2 = torch.optim.Adam(
        [{"params": list(illum2.parameters()), "lr": 1.0},
         {"params": list(inverse2.parameters()), "lr": 0.001}]
    )
    load_adam_inverse(opt2, inverse2, snap, torch.device("cpu"))
    names = dict(inverse.named_parameters())
    names2 = dict(inverse2.named_parameters())
    for name in names:
        s1 = opt.state[names[name]]
        s2 = opt2.state[names2[name]]
        for key in s1:
            v1, v2 = s1[key], s2[key]
            if torch.is_tensor(v1):
                assert torch.equal(v1.cpu(), v2.cpu()), (name, key)
            else:
                assert v1 == v2

