#!/usr/bin/env python3
"""Sanity gates for the MCF7 Figure 8/9 SwinIR high-res reconstruction fix.

MUST pass before launching long training (per the Figure-3 discipline). Gates:
  1. data_pairing_norm   : dataset returns [1,H,W] in [0,1]; deterministic val/test crops.
  2. no_leakage_split    : train/val/test image pools are disjoint (well-level).
  3. compression_x16     : forward measurement is T patterns of (H/d)x(H/d) -> ratio d^2/T = 16.
  4. loss_stack_finite   : build_loss_stack -> one full pixel+perceptual+GAN step is finite.
  5. swinir_psi_overfit  : SwinIR-as-psi pipeline overfits a few patches (SSIM up sharply)
                           -> confirms end-to-end wiring learns.
  6. baselines_build     : transpose256 (R) and wcnn64 (wCNN) construct + forward at right size.
  7. shapes_upscale1     : SwinIR output size == input size (upscale=1 image-to-image).

Writes experiments/figure08_mcf7/sanity/sanity_report.json (+ prints).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.swinir.losses import build_loss_stack, pixel_loss
from datasets.mcf7_channel2 import MCF7Channel2Config, MCF7Channel2Dataset, _assign_well_splits, _load_manifest
from evaluation.metrics import ssim as ssim_metric
from utils.device import resolve_device

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "fig89_train", ROOT / "scripts" / "fig89_mcf7_swinir_fix_train.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

EXP = ROOT / "experiments/figure08_mcf7"
CFG = ROOT / "configs/figure08_mcf7/swinir_fix.yaml"


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def gate_data_pairing_norm(cfg: dict) -> dict:
    ds_cfg = dict(cfg["dataset"]); ds_cfg["num_test"] = 8; ds_cfg["patch_size"] = 256; ds_cfg["image_size"] = 256
    ds = MCF7Channel2Dataset.from_dict(ds_cfg, split="test")
    x0 = ds[0]; x1 = ds[0]
    ok_shape = tuple(x0.shape) == (1, 256, 256)
    ok_range = float(x0.min()) >= 0.0 and float(x0.max()) <= 1.0 + 1e-5
    ok_det = torch.allclose(x0, x1)  # deterministic for test split
    ok_nontrivial = float(x0.std()) > 1e-3
    passed = ok_shape and ok_range and ok_det and ok_nontrivial
    return {"pass": bool(passed), "shape": list(x0.shape), "min": float(x0.min()),
            "max": float(x0.max()), "std": float(x0.std()), "deterministic": bool(ok_det)}


def gate_no_leakage_split(cfg: dict) -> dict:
    conf = MCF7Channel2Config.from_dict(dict(cfg["dataset"]))
    rows = _load_manifest(Path(conf.data_root), conf)
    splits = _assign_well_splits(rows, conf)

    def wells(items):
        return {(r.get("well") or r["image_file"]) for r in items}

    tw, vw, ew = wells(splits["train"]), wells(splits["val"]), wells(splits["test"])
    paths = {k: {r["path"] for r in v} for k, v in splits.items()}
    disjoint_wells = not (tw & vw) and not (tw & ew) and not (vw & ew)
    disjoint_paths = (not (paths["train"] & paths["val"]) and not (paths["train"] & paths["test"])
                      and not (paths["val"] & paths["test"]))
    return {"pass": bool(disjoint_wells and disjoint_paths),
            "n_train_wells": len(tw), "n_val_wells": len(vw), "n_test_wells": len(ew),
            "wells_disjoint": bool(disjoint_wells), "paths_disjoint": bool(disjoint_paths)}


def gate_compression_x16(cfg: dict, device) -> dict:
    model = T._build_swinir_model(cfg, 256).to(device)
    model(torch.zeros(1, 1, 256, 256, device=device))
    with torch.no_grad():
        out = model(torch.rand(1, 1, 256, 256, device=device), sigmoid_m=8.0)
    y = out["y_down"]  # [B, T, H/d, W/d]
    _, tpat, hd, wd = y.shape
    d = int(cfg["forward_model"]["downscale_factor"])
    ratio = (256 * 256) / (tpat * hd * wd)
    return {"pass": bool(abs(ratio - 16.0) < 1e-6 and hd == 256 // d),
            "y_down_shape": list(y.shape), "compression_ratio": float(ratio),
            "downscale": d, "num_patterns": tpat}


def gate_loss_stack_finite(cfg: dict, device) -> dict:
    model = T._build_swinir_model(cfg, 256).to(device)
    model(torch.zeros(1, 1, 256, 256, device=device))
    stack = build_loss_stack(dict(cfg["training"]["loss"]), device)
    disc = stack.get("discriminator")
    x = torch.rand(2, 1, 256, 256, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else T._null():
        rec = model(x, sigmoid_m=8.0)["x_recon"]
        pix = stack["pixel_weight"] * pixel_loss(rec, x, stack["pixel_kind"])
        perc = stack["perceptual_weight"] * stack["perceptual"](rec, x)
        adv = stack["gan_weight"] * stack["gan_loss"](disc(rec), True)
        dloss = stack["gan_loss"](disc(x), True) + stack["gan_loss"](disc(rec.detach()), False)
    vals = {"pixel": float(pix), "perceptual": float(perc), "adv": float(adv), "d": float(dloss)}
    finite = all(torch.isfinite(torch.tensor(v)) for v in vals.values())
    has_all = ("perceptual" in stack) and (disc is not None) and (stack["gan_weight"] > 0)
    return {"pass": bool(finite and has_all), **vals,
            "has_perceptual": "perceptual" in stack, "has_gan": disc is not None,
            "gan_weight": stack["gan_weight"], "perceptual_weight": stack["perceptual_weight"]}


def gate_swinir_psi_overfit(cfg: dict, device, steps: int = 500, probe_lr: float = 1e-3) -> dict:
    """Confirm the SwinIR-as-psi end-to-end pipeline can fit (memorize) a few patches.

    WIRING probe: verifies gradients reach illumination + upsampling + SwinIR and the
    pipeline can drive the reconstruction to the targets. Uses **pure L1** (not the training
    perceptual+GAN stack): perceptual loss deliberately trades pixel accuracy for feature
    matching and thus *suppresses* the pixel-structural SSIM, the wrong objective for a
    fit test (full loss validated separately by `loss_stack_finite`).

    The joint illumination (lr 0.1) + SwinIR optimization makes the per-step SSIM oscillate
    (the illumination keeps re-encoding), so we judge the probe by the **best SSIM reached**
    over the trajectory (robust) alongside the stable L1 collapse — not the noisy endpoint.
    """
    torch.manual_seed(0)
    model = T._build_swinir_model(cfg, 256).to(device)
    model(torch.zeros(1, 1, 256, 256, device=device))
    ds_cfg = dict(cfg["dataset"]); ds_cfg["num_train"] = 4; ds_cfg["patch_size"] = 256; ds_cfg["image_size"] = 256
    ds = MCF7Channel2Dataset.from_dict(ds_cfg, split="train")
    x = torch.stack([ds[i] for i in range(4)]).to(device)
    opt = torch.optim.Adam(
        [{"params": model.swinir_parameters(), "lr": probe_lr},
         {"params": model.illumination_parameters(), "lr": 0.1}])
    with torch.no_grad():
        rec0 = model(x, sigmoid_m=8.0)["x_recon"].float().clamp(0, 1)
        ssim0 = float(ssim_metric(rec0, x)); l1_0 = float(F.l1_loss(rec0, x))
    model.train()
    traj = []
    best_ssim = ssim0
    best_l1 = l1_0
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else T._null():
            rec = model(x, sigmoid_m=8.0)["x_recon"]
            loss = pixel_loss(rec, x, "l1")
        loss.backward()
        opt.step()
        if (s + 1) % 50 == 0:
            with torch.no_grad():
                sv = float(ssim_metric(rec.float().clamp(0, 1), x))
                lv = float(F.l1_loss(rec.float(), x))
            traj.append(round(sv, 4))
            best_ssim = max(best_ssim, sv)
            best_l1 = min(best_l1, lv)
            print(f"  overfit step {s + 1}/{steps} ssim={sv:.4f} l1={lv:.4f} best_ssim={best_ssim:.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        rec = model(x, sigmoid_m=8.0)["x_recon"].float().clamp(0, 1)
        ssim1 = float(ssim_metric(rec, x)); l1_1 = float(F.l1_loss(rec, x))
    best_ssim = max(best_ssim, ssim1)
    best_l1 = min(best_l1, l1_1)
    # clear learning/fit: best SSIM strong + large gain + L1 collapsed (robust to oscillation).
    passed = best_ssim > 0.8 and best_ssim > ssim0 + 0.4 and best_l1 < l1_0 * 0.4
    return {"pass": bool(passed), "loss": "pure_l1", "ssim_before": ssim0,
            "ssim_after_endpoint": ssim1, "best_ssim": best_ssim,
            "l1_before": l1_0, "l1_after_endpoint": l1_1, "best_l1": best_l1,
            "steps": steps, "probe_lr": probe_lr, "ssim_trajectory_per50": traj}


def gate_baselines_build(cfg: dict, device) -> dict:
    res = {}
    ok = True
    for cond, isz, up in [("transpose256", 256, "transpose_conv"), ("wcnn64", 64, "locality_aware")]:
        m = T._build_conventional_model(cfg, isz, up).to(device)
        m(torch.zeros(1, 1, isz, isz, device=device))
        with torch.no_grad():
            r = m(torch.rand(1, 1, isz, isz, device=device), sigmoid_m=8.0)["x_recon"]
        good = tuple(r.shape) == (1, 1, isz, isz)
        ok = ok and good
        res[cond] = {"out_shape": list(r.shape), "ok": bool(good)}
    return {"pass": bool(ok), **res}


def gate_shapes_upscale1(cfg: dict, device) -> dict:
    model = T._build_swinir_model(cfg, 256).to(device)
    model(torch.zeros(1, 1, 256, 256, device=device))
    with torch.no_grad():
        out = model(torch.rand(1, 1, 256, 256, device=device), sigmoid_m=8.0)
    r = out["x_recon"]
    return {"pass": bool(tuple(r.shape) == (1, 1, 256, 256)), "recon_shape": list(r.shape),
            "swinir_upscale": int(cfg["swinir"]["upscale"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overfit-steps", type=int, default=1000)
    ap.add_argument("--only", default=None, help="run a single gate by name and update the report")
    args = ap.parse_args()
    cfg = _load_yaml(CFG)
    device = resolve_device(args.device)
    print(f"Sanity device: {device}", flush=True)

    all_gates = {
        "data_pairing_norm": lambda: gate_data_pairing_norm(cfg),
        "no_leakage_split": lambda: gate_no_leakage_split(cfg),
        "compression_x16": lambda: gate_compression_x16(cfg, device),
        "loss_stack_finite": lambda: gate_loss_stack_finite(cfg, device),
        "shapes_upscale1": lambda: gate_shapes_upscale1(cfg, device),
        "baselines_build": lambda: gate_baselines_build(cfg, device),
        "swinir_psi_overfit": lambda: gate_swinir_psi_overfit(cfg, device, steps=args.overfit_steps),
    }
    out = EXP / "sanity" / "sanity_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.only:
        if args.only not in all_gates:
            raise SystemExit(f"unknown gate {args.only}; choices: {list(all_gates)}")
        prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"gates": {}}
        gates = prev.get("gates", {})
        print(f"[only] {args.only}...", flush=True)
        gates[args.only] = all_gates[args.only]()
    else:
        gates = {}
        for i, (name, fn) in enumerate(all_gates.items(), 1):
            print(f"[{i}/{len(all_gates)}] {name}...", flush=True)
            gates[name] = fn()

    all_pass = all(g["pass"] for g in gates.values())
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "device": str(device), "all_pass": bool(all_pass), "gates": gates}
    out = EXP / "sanity" / "sanity_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== SANITY SUMMARY ===", flush=True)
    for name, g in gates.items():
        print(f"  {'PASS' if g['pass'] else 'FAIL'}  {name}", flush=True)
    print(f"ALL_PASS={all_pass}  -> {out}", flush=True)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
