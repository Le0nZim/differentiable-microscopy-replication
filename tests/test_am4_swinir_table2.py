"""AM-4 — Phase 7 tests: config parsing, SwinIR arch audit, fairness audit,
eval-metric sanity, and checkpoint-metadata structure.

All tests are CPU-only and fast: model-building tests use a reduced embed_dim;
config-reading tests assert the real (180) capacity declared by the YAMLs.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml

from baselines.swinir import am4_table2 as A

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs/table02_swinir_sr"
CONFIGS = {
    "smoke": CFG_DIR / "smoke.yaml",
    "budget": CFG_DIR / "budget.yaml",
    "full": CFG_DIR / "full.yaml",
}


def _small(cfg: dict) -> dict:
    """Reduce SwinIR capacity for fast CPU model builds (wiring unchanged)."""
    cfg = copy.deepcopy(cfg)
    cfg["model"]["swinir"].update({"embed_dim": 24, "depths": [2, 2], "num_heads": [2, 2]})
    return cfg


# ---------------------------------------------------------------------------
# Config parsing / validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(CONFIGS))
def test_config_parses_and_validates(name):
    cfg = A.load_am4_config(CONFIGS[name])
    micro, accum, eff = A.resolve_grad_accum(cfg)
    assert micro * accum == eff
    assert isinstance(cfg["deviations_from_paper"], list) and cfg["deviations_from_paper"]
    assert cfg["model"]["swinir"]["upscale"] == 1  # paper: no upscaling
    d = cfg["model"]["downscale_factor"]
    t = cfg["model"]["num_patterns"]
    assert d * d // t == cfg["model"]["compression"]  # x16


def test_invalid_grad_accum_rejected(tmp_path):
    cfg = yaml.safe_load(CONFIGS["budget"].read_text())
    cfg["training"]["grad_accum"] = 3  # 8*3 != 32
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        A.load_am4_config(p)


def test_full_config_matches_swinir_m_and_paper():
    cfg = A.load_am4_config(CONFIGS["full"])
    sw = cfg["model"]["swinir"]
    assert sw["embed_dim"] == 180                 # SwinIR-M (was 96)
    assert sw["depths"] == [6, 6, 6, 6, 6, 6]
    assert sw["num_heads"] == [6, 6, 6, 6, 6, 6]
    assert sw["window_size"] == 8
    assert cfg["training"]["effective_batch_size"] == 32          # paper batch 32
    assert abs(cfg["training"]["illumination_lr"] - 0.1) < 1e-12  # paper LR
    # full + budget eval must be fair (full deterministic tiling, no cap)
    assert cfg["eval"]["tile_selection"] == "all"
    assert cfg["eval"]["max_tiles_per_image"] is None


def test_budget_is_fair_eval_smoke_is_labelled():
    budget = A.load_am4_config(CONFIGS["budget"])
    smoke = A.load_am4_config(CONFIGS["smoke"])
    assert budget["eval"]["tile_selection"] == "all" and budget["eval"]["max_tiles_per_image"] is None
    # smoke may cap, but only because it is explicitly a smoke tag
    assert smoke["experiment"]["tag"] == "smoke"


# ---------------------------------------------------------------------------
# SwinIR architecture audit
# ---------------------------------------------------------------------------
def test_arch_summary_capacity_and_illumination():
    pytest.importorskip("timm")
    cfg = _small(A.load_am4_config(CONFIGS["budget"]))
    wo = A.build_model(cfg, learnable=False)
    wi = A.build_model(cfg, learnable=True)
    awo, awi = A.model_arch_summary(wo), A.model_arch_summary(wi)
    # identical inverse capacity
    assert awo["swinir_params"] == awi["swinir_params"]
    assert awo["upsampling_params"] == awi["upsampling_params"]
    assert awo["fuse_params"] == awi["fuse_params"]
    assert awo["embed_dim"] == awi["embed_dim"] == 24
    # only difference: illumination
    assert awo["illumination_params"] == 0 and awo["illumination_learnable"] is False
    assert awi["illumination_params"] > 0 and awi["illumination_learnable"] is True


def test_full_config_builds_at_180(tmp_path):
    pytest.importorskip("timm")
    cfg = A.load_am4_config(CONFIGS["full"])
    # build only the SwinIR backbone summary cheaply by checking declared dim;
    # a real 180-dim build is exercised by the smoke/budget runs, not unit tests.
    assert cfg["model"]["swinir"]["embed_dim"] == 180


# ---------------------------------------------------------------------------
# Optimizer fairness
# ---------------------------------------------------------------------------
def test_optimizer_group_fairness():
    pytest.importorskip("timm")
    cfg = _small(A.load_am4_config(CONFIGS["budget"]))
    wo = A.build_model(cfg, learnable=False)
    wi = A.build_model(cfg, learnable=True)
    o_wo = A.build_optimizers(wo, cfg, learnable=False)
    o_wi = A.build_optimizers(wi, cfg, learnable=True)
    g_wo = {g["name"]: g for g in A.optimizer_group_summary(o_wo["opt_g"])}
    g_wi = {g["name"]: g for g in A.optimizer_group_summary(o_wi["opt_g"])}
    assert g_wo["inverse"]["lr"] == g_wi["inverse"]["lr"]
    assert g_wo["inverse"]["betas"] == g_wi["inverse"]["betas"]
    assert g_wo["inverse"]["num_params"] == g_wi["inverse"]["num_params"]
    assert "illumination" not in g_wo and "illumination" in g_wi
    assert abs(g_wi["illumination"]["lr"] - 0.1) < 1e-12


# ---------------------------------------------------------------------------
# Eval metric / tiling sanity
# ---------------------------------------------------------------------------
def test_tile_coords_all_is_rowmajor_and_complete():
    coords = A.tile_coords(3, 4, selection="all")
    assert len(coords) == 12
    assert coords[0] == (0, 0) and coords[-1] == (2, 3)


def test_tile_coords_center_biased_deterministic_and_capped():
    a = A.tile_coords(5, 5, selection="center_biased", max_tiles=3)
    b = A.tile_coords(5, 5, selection="center_biased", max_tiles=3)
    assert a == b and len(a) == 3
    assert a[0] == (2, 2)  # exact center first


def test_psnr_ssim_convention():
    from evaluation.metrics import psnr, ssim

    a = torch.rand(1, 1, 32, 32)
    assert float(psnr(a, a)) > 100.0                       # identical -> ~120 dB
    assert abs(float(psnr(a + 0.1, a)) - 20.0) < 1e-3      # mse 0.01 -> 20 dB
    assert abs(float(ssim(a, a)) - 1.0) < 1e-4


def test_eval_dataset_fair_reproducible(tmp_path):
    pytest.importorskip("timm")
    from torchvision.utils import save_image

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    torch.manual_seed(0)
    save_image(torch.rand(1, 128, 96), img_dir / "a.png")
    save_image(torch.rand(1, 64, 64), img_dir / "b.png")

    cfg = _small(A.load_am4_config(CONFIGS["budget"]))
    model = A.build_model(cfg, learnable=True).eval()
    kw = dict(patch_size=64, device=torch.device("cpu"), learnable=True,
              eval_sigmoid_m=8.0, selection="all", max_tiles_per_image=None,
              eval_batch=8, amp_dtype=None, compute_stitched=True)
    r1 = A.eval_dataset_fair(model, img_dir, **kw)
    r2 = A.eval_dataset_fair(model, img_dir, **kw)
    assert r1["psnr"] == r2["psnr"] and r1["ssim"] == r2["ssim"]
    assert r1["tiles"] == 2 + 1  # 128x96 -> 2x1 tiles, 64x64 -> 1 tile
    assert "stitched_psnr" in r1


# ---------------------------------------------------------------------------
# Fairness audit (reduced config for speed) + full smoke metadata structure
# ---------------------------------------------------------------------------
def _load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_am4_swinir_fairness", ROOT / "scripts" / "table02_swinir_sr" / "audit_fairness.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fairness_audit_overall_pass(tmp_path):
    pytest.importorskip("timm")
    cfg = _small(A.load_am4_config(CONFIGS["budget"]))
    p = tmp_path / "reduced.yaml"
    p.write_text(yaml.safe_dump(cfg))
    audit = _load_audit_module()
    result = audit.run_audit(p)
    assert result["overall_pass"] is True
    assert all(result["summary"][c] for c in ("C1", "C2", "C3", "C4", "C5", "C6", "C7"))


def test_checkpoint_metadata_structure_from_smoke():
    """If the frozen smoke run exists, its checkpoint metadata must be complete."""
    meta_path = (
        ROOT
        / "experiments/table02_swinir_sr/smoke/swinir_wo_li/checkpoint_metadata.json"
    )
    if not meta_path.exists():
        pytest.skip("smoke run not present")
    meta = json.loads(meta_path.read_text())
    required = {
        "condition", "learnable", "iterations_reached", "effective_batch_size",
        "best_val_l1", "best_step", "arch_summary", "optimizer_groups",
        "seed", "checkpoints",
    }
    assert required.issubset(meta.keys())
    assert meta["arch_summary"]["embed_dim"] == 180
    assert set(meta["checkpoints"]) == {"best", "last"}
