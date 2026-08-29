#!/usr/bin/env python3
"""AM-4 — Phase 3: evaluation/forward FAIRNESS audit (run BEFORE long training).

Machine-verifies that the LI and w/o-LI conditions differ ONLY by illumination
learnability, and that the evaluation + metric conventions are fair and
reproducible. Writes:
  experiments/swinir_or_highres/am4_swinir_table2_resolution/fairness_audit.json

Checks:
  C1 identical inverse architecture (SwinIR + upsampling + fuse param counts/dims)
  C2 only intended difference = learnable vs fixed illumination
  C3 identical optimizer settings for the SHARED (inverse) param group
  C4 identical compressive forward path (downscale/T/compression/#measurements/noise)
     -> w/o-LI is NOT an easier measurement path
  C5 evaluation covers all images / deterministic (no center-bias cap unless smoke)
  C6 PSNR/SSIM convention documented + metric sanity (identical, known-shift, ssim=1)
  C7 metrics + forward reproducible under a fixed seed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.swinir import am4_table2 as A
from baselines.swinir.losses import build_loss_stack
from evaluation.metrics import psnr as psnr_metric
from evaluation.metrics import ssim as ssim_metric

OUT_BASE = ROOT / "experiments/swinir_or_highres/am4_swinir_table2_resolution"


def run_audit(config_path: str | Path) -> dict:
    cfg = A.load_am4_config(config_path)
    seed = int(cfg["experiment"]["seed"])
    checks: list[dict] = []

    wo = A.build_model(cfg, learnable=False, seed=seed)
    wi = A.build_model(cfg, learnable=True, seed=seed)
    awo = A.model_arch_summary(wo)
    awi = A.model_arch_summary(wi)

    # C1: identical inverse architecture
    same_arch = (
        awo["swinir_params"] == awi["swinir_params"]
        and awo["upsampling_params"] == awi["upsampling_params"]
        and awo["fuse_params"] == awi["fuse_params"]
        and awo["embed_dim"] == awi["embed_dim"]
        and awo["window_size"] == awi["window_size"]
        and awo["upscale"] == awi["upscale"]
    )
    checks.append({
        "id": "C1", "name": "identical_inverse_architecture", "pass": bool(same_arch),
        "detail": {"wo": awo, "with": awi},
    })

    # C2: only intended difference = illumination learnability
    only_illum = (
        awo["illumination_learnable"] is False
        and awi["illumination_learnable"] is True
        and awo["illumination_params"] == 0
        and awi["illumination_params"] > 0
        and same_arch
    )
    checks.append({
        "id": "C2", "name": "only_difference_is_learnable_illumination", "pass": bool(only_illum),
        "detail": {
            "wo_illum_params": awo["illumination_params"], "with_illum_params": awi["illumination_params"],
            "wo_learnable": awo["illumination_learnable"], "with_learnable": awi["illumination_learnable"],
        },
    })

    # C3: identical optimizer settings for the shared (inverse) group
    loss_cfg = dict(cfg["training"]["loss"]); loss_cfg["in_chans"] = 1
    stack_wo = build_loss_stack(loss_cfg, torch.device("cpu"))
    stack_wi = build_loss_stack(loss_cfg, torch.device("cpu"))
    o_wo = A.build_optimizers(wo, cfg, learnable=False, discriminator=stack_wo.get("discriminator"))
    o_wi = A.build_optimizers(wi, cfg, learnable=True, discriminator=stack_wi.get("discriminator"))
    g_wo = {g["name"]: g for g in A.optimizer_group_summary(o_wo["opt_g"])}
    g_wi = {g["name"]: g for g in A.optimizer_group_summary(o_wi["opt_g"])}
    inv_same = (
        "inverse" in g_wo and "inverse" in g_wi
        and abs(g_wo["inverse"]["lr"] - g_wi["inverse"]["lr"]) < 1e-12
        and g_wo["inverse"]["betas"] == g_wi["inverse"]["betas"]
        and g_wo["inverse"]["num_params"] == g_wi["inverse"]["num_params"]
    )
    illum_only_in_li = ("illumination" not in g_wo) and ("illumination" in g_wi)
    illum_lr_ok = abs(g_wi.get("illumination", {}).get("lr", -1) - float(cfg["training"]["illumination_lr"])) < 1e-12
    checks.append({
        "id": "C3", "name": "identical_shared_optimizer_settings", "pass": bool(inv_same and illum_only_in_li and illum_lr_ok),
        "detail": {"wo_groups": g_wo, "with_groups": g_wi, "illumination_lr_matches_paper_0.1": bool(illum_lr_ok)},
    })

    # C4: identical compressive forward path (w/o-LI not easier)
    def meas_count(model):
        d = model.forward_model.downscale_factor
        ps = int(cfg["data"]["patch_size"])
        return model.num_patterns * (ps // d) * (ps // d)
    fwd_same = (
        wo.forward_model.downscale_factor == wi.forward_model.downscale_factor
        and wo.num_patterns == wi.num_patterns
        and meas_count(wo) == meas_count(wi)
        and type(wo.detector_noise).__name__ == type(wi.detector_noise).__name__
    )
    comp = int(cfg["model"]["compression"])
    comp_ok = meas_count(wo) == (int(cfg["data"]["patch_size"]) ** 2) // comp
    checks.append({
        "id": "C4", "name": "identical_forward_path_wo_li_not_easier", "pass": bool(fwd_same and comp_ok),
        "detail": {
            "downscale": wo.forward_model.downscale_factor, "num_patterns": wo.num_patterns,
            "measurements_per_image": meas_count(wo), "compression": comp,
            "expected_measurements": (int(cfg["data"]["patch_size"]) ** 2) // comp,
        },
    })

    # C5: evaluation fairness (full deterministic unless smoke)
    ev = cfg["eval"]
    sel = str(ev.get("tile_selection", "all"))
    cap = ev.get("max_tiles_per_image")
    is_smoke = str(cfg["experiment"].get("tag")) == "smoke"
    eval_fair = (sel == "all" and cap is None) or is_smoke
    checks.append({
        "id": "C5", "name": "evaluation_full_deterministic_or_labelled_smoke", "pass": bool(eval_fair),
        "detail": {"tile_selection": sel, "max_tiles_per_image": cap, "is_smoke": is_smoke,
                   "note": "full runs must use selection=all, no cap; smoke may cap (labelled)"},
    })

    # C6: PSNR/SSIM convention + metric sanity
    torch.manual_seed(0)
    a = torch.rand(2, 1, 32, 32)
    psnr_same = float(psnr_metric(a, a))            # identical -> ~120 dB (clamp 1e-12)
    psnr_shift = float(psnr_metric((a + 0.1).clamp(0, 1), a))  # not exact due to clamp, approx 20 dB
    b = (a + 0.1)  # unclamped known shift -> mse=0.01 -> psnr=20 exactly
    psnr_shift_exact = float(psnr_metric(b, a))
    ssim_same = float(ssim_metric(a, a))
    sanity = psnr_same > 100.0 and abs(psnr_shift_exact - 20.0) < 1e-3 and abs(ssim_same - 1.0) < 1e-4
    checks.append({
        "id": "C6", "name": "psnr_ssim_convention_and_sanity", "pass": bool(sanity),
        "detail": {
            "convention": "grayscale [B,1,H,W], data_range=1.0, NO border crop (upscale=1 SR border=0), recon clamp[0,1], per-tile then averaged; stitched full-image as secondary",
            "psnr_identical_db": round(psnr_same, 2),
            "psnr_known_shift_0.1_db": round(psnr_shift_exact, 4),
            "ssim_identical": round(ssim_same, 6),
        },
    })

    # C7: reproducibility under fixed seed (forward + metric)
    m1 = A.build_model(cfg, learnable=True, seed=123).eval()
    m2 = A.build_model(cfg, learnable=True, seed=123).eval()
    ps = int(cfg["data"]["patch_size"])
    torch.manual_seed(7)
    x = torch.rand(2, 1, ps, ps)
    with torch.no_grad():
        r1 = m1(x, sigmoid_m=8.0)["x_recon"]
        r2 = m2(x, sigmoid_m=8.0)["x_recon"]
    repro = bool(torch.allclose(r1, r2, atol=1e-6))
    checks.append({
        "id": "C7", "name": "fixed_seed_reproducible_eval_mode", "pass": repro,
        "detail": {
            "max_abs_diff": float((r1 - r2).abs().max()),
            "note": "eval() required: SwinIR uses drop_path_rate=0.1 stochastic depth in train mode (faithful SwinIR default)",
        },
    })

    overall = all(c["pass"] for c in checks)
    return {
        "overall_pass": bool(overall),
        "config": str(config_path),
        "tag": cfg["experiment"].get("tag"),
        "checks": checks,
        "summary": {c["id"]: c["pass"] for c in checks},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/swinir/am4_table2_budget.yaml"))
    ap.add_argument("--out", default=str(OUT_BASE / "fairness_audit.json"))
    args = ap.parse_args()
    result = run_audit(args.config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"overall_pass={result['overall_pass']}  -> {out}")
    if not result["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
