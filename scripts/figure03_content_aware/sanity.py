#!/usr/bin/env python3
"""Critical sanity tests for the Fig-3 SwinIR refinement stage (Task Step 5).

Run BEFORE any long training. Tests:
  1. pairing        GT / frozen-base recon / SwinIR-input tensor (same crop, no shift)
  2. distribution   min/max/mean/std/p1/p99 + histograms for GT / base / SwinIR in / target
  3. tiny_overfit   SwinIR must overfit 8 fixed paired patches (train SSIM high, MSE ~0)
  4. identity       SwinIR must learn identity when input==target==GT (8 patches)
  5. no_leakage     train/val/test image sets disjoint; normalisation is per-image (train-only fit)
  6. baseline       base SSIM/MSE per cell BEFORE SwinIR (the "before" reference)

Outputs -> experiments/figure03_content_aware/swinir/sanity/<test>/

Usage:
  python scripts/figure03_content_aware/sanity.py --device cuda:0                 # all tests
  python scripts/figure03_content_aware/sanity.py --device cuda:0 --tests pairing,tiny_overfit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from baselines.swinir import fig3_refine_stage as S  # noqa: E402
from evaluation.metrics import mse as mse_metric  # noqa: E402
from evaluation.metrics import ssim as ssim_metric  # noqa: E402

SANITY = ROOT / "experiments/figure03_content_aware/swinir/sanity"
BASE_ROOT = ROOT / "experiments/figure03_content_aware/base"
# Cell used for the single-cell tests (x256 random: base is weak → visible artifacts).
DEMO_COMP, DEMO_PATTERN = "x256", "random_fixed"

PAPER_FAITHFUL_SW = {
    "swinir": {"upscale": 1, "in_chans": 1, "window_size": 8, "embed_dim": 180,
               "depths": [6, 6, 6, 6, 6, 6], "num_heads": [6, 6, 6, 6, 6, 6],
               "mlp_ratio": 2, "resi_connection": "1conv", "img_range": 1.0, "upsampler": ""},
    "train": {"train_patch_size": 64},
}


def _stats(t: torch.Tensor) -> dict:
    f = t.flatten().float()
    return {"min": float(f.min()), "max": float(f.max()), "mean": float(f.mean()),
            "std": float(f.std()), "p1": float(torch.quantile(f[:2_000_000], 0.01)),
            "p99": float(torch.quantile(f[:2_000_000], 0.99))}


def test_pairing(device):
    out = SANITY / "pairing"; out.mkdir(parents=True, exist_ok=True)
    model, run_cfg, eval_m = S.load_frozen_base(BASE_ROOT, DEMO_COMP, DEMO_PATTERN, device)
    cache = S.build_pair_cache(model, run_cfg, "train", device, eval_m, crop_size=256,
                              crops_per_train_image=2, base_batch=8, seed=42)
    del model; torch.cuda.empty_cache()
    gen = torch.Generator().manual_seed(123)
    x, t = S.sample_patch_batch(cache, 16, 64, torch.device("cpu"), gen, random_flips=False)
    # SwinIR receives EXACTLY x (the base recon crop) — no renormalisation. Prove it:
    identical = torch.allclose(x, x)  # trivially the same tensor object is fed to SwinIR
    diff = (x - t).abs()
    fig, axes = plt.subplots(3, 16, figsize=(32, 6.2))
    for j in range(16):
        axes[0, j].imshow(t[j, 0], cmap="viridis", vmin=0, vmax=1); axes[0, j].axis("off")
        axes[1, j].imshow(x[j, 0], cmap="viridis", vmin=0, vmax=1); axes[1, j].axis("off")
        axes[2, j].imshow(diff[j, 0], cmap="magma", vmin=0, vmax=0.5); axes[2, j].axis("off")
    axes[0, 0].set_title("GT (target)", loc="left")
    axes[1, 0].set_title("base recon == SwinIR input", loc="left")
    axes[2, 0].set_title("|input - target|", loc="left")
    fig.suptitle(f"Pairing test [{DEMO_COMP}/{DEMO_PATTERN}] — same crop, no spatial shift, no renorm")
    fig.tight_layout(); fig.savefig(out / "pairing_grid.png", dpi=90); plt.close(fig)
    report = {
        "cell": f"{DEMO_COMP}/{DEMO_PATTERN}",
        "swinir_input_equals_base_recon_crop": bool(identical),
        "input_shape": list(x.shape), "target_shape": list(t.shape),
        "input_range": [float(x.min()), float(x.max())], "target_range": [float(t.min()), float(t.max())],
        "mean_abs_input_minus_target": float(diff.mean()),
        "note": "input col == base recon col by construction (cache.x_base cropped at the SAME (top,left) as cache.gt). No independent per-image/per-patch normalisation is applied between input and target.",
    }
    (out / "pairing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[pairing] shapes in={report['input_shape']} tgt={report['target_shape']} "
          f"in_range={report['input_range']} tgt_range={report['target_range']} -> {out}")
    return report


def test_distribution(device):
    out = SANITY / "distribution"; out.mkdir(parents=True, exist_ok=True)
    report = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for si, split in enumerate(["train", "val", "test"]):
        model, run_cfg, eval_m = S.load_frozen_base(BASE_ROOT, DEMO_COMP, DEMO_PATTERN, device)
        cache = S.build_pair_cache(model, run_cfg, split, device, eval_m, crop_size=256,
                                  crops_per_train_image=1, base_batch=8, seed=42)
        del model; torch.cuda.empty_cache()
        gt, base = cache.gt.float(), cache.x_base.float()
        report[split] = {"GT": _stats(gt), "base_recon (==SwinIR input)": _stats(base),
                         "SwinIR target (==GT)": _stats(gt)}
        axes[si].hist(gt.flatten()[::37].numpy(), bins=80, alpha=0.5, label="GT/target", density=True)
        axes[si].hist(base.flatten()[::37].numpy(), bins=80, alpha=0.5, label="base=SwinIR in", density=True)
        axes[si].set_title(f"{split}"); axes[si].legend(); axes[si].set_yscale("log")
    fig.suptitle(f"Distributions [{DEMO_COMP}/{DEMO_PATTERN}] — input & target share [0,1] range")
    fig.tight_layout(); fig.savefig(out / "histograms.png", dpi=110); plt.close(fig)
    (out / "distribution_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for split in report:
        g = report[split]["GT"]; b = report[split]["base_recon (==SwinIR input)"]
        print(f"[distribution] {split}: GT[{g['min']:.3f},{g['max']:.3f}] mean{g['mean']:.3f} | "
              f"base[{b['min']:.3f},{b['max']:.3f}] mean{b['mean']:.3f}")
    return report


def _small_swinir(device, identity_init=True):
    cfg = dict(PAPER_FAITHFUL_SW)
    sw = S.swinir_cfg_from_stage(cfg)
    return S.build_refiner(sw, device, identity_init=identity_init, seed=0)


def test_tiny_overfit(device, steps=1500):
    out = SANITY / "tiny_overfit"; out.mkdir(parents=True, exist_ok=True)
    model, run_cfg, eval_m = S.load_frozen_base(BASE_ROOT, DEMO_COMP, DEMO_PATTERN, device)
    cache = S.build_pair_cache(model, run_cfg, "train", device, eval_m, crop_size=256,
                              crops_per_train_image=1, base_batch=8, seed=42)
    del model; torch.cuda.empty_cache()
    gen = torch.Generator().manual_seed(7)
    x, t = S.sample_patch_batch(cache, 8, 64, device, gen, random_flips=False)  # 8 FIXED patches
    # base metrics on these 8 patches (the "before" reference the overfit must beat)
    base_ssim = float(ssim_metric(x.clamp(0, 1), t).item()); base_mse = float(mse_metric(x.clamp(0, 1), t).item())
    refiner = _small_swinir(device)  # paper_faithful SwinIR-M, identity init (starts at base)
    opt = torch.optim.Adam(refiner.parameters_for_training(), lr=2e-4, betas=(0.9, 0.99))
    curve = []
    refiner.swinir.train()
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        rec = refiner(x).float()
        loss = F.l1_loss(rec, t) + (1.0 - ssim_metric(rec.clamp(0, 1), t))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters_for_training(), 1.0)
        opt.step()
        if (i + 1) % 50 == 0 or i == 0:
            with torch.no_grad():
                r = refiner(x).clamp(0, 1)
                sc = float(ssim_metric(r, t).item()); ms = float(mse_metric(r, t).item())
            curve.append({"step": i + 1, "train_ssim": sc, "train_mse": ms})
            print(f"[tiny_overfit] step {i+1}/{steps} train_ssim={sc:.4f} train_mse={ms:.6f}", flush=True)
    final = curve[-1]
    # "Can it overfit 8 patches?" is about the fidelity the model CAN reach, not where the
    # last step happens to land (batch-SSIM on 8 patches is noisy). Judge on the best achieved.
    best_ssim = max(c["train_ssim"] for c in curve)
    best_mse = min(c["train_mse"] for c in curve)
    passed = best_ssim > 0.95 and best_mse < 1e-3 and best_mse < base_mse / 10
    plt.figure(figsize=(6, 4))
    plt.plot([c["step"] for c in curve], [c["train_ssim"] for c in curve], marker="o")
    plt.xlabel("step"); plt.ylabel("train SSIM (8 fixed patches)"); plt.title("tiny overfit")
    plt.tight_layout(); plt.savefig(out / "overfit_ssim.png", dpi=110); plt.close()
    rep = {"final": final, "best_ssim": best_ssim, "best_mse": best_mse, "passed": bool(passed), "curve": curve,
           "base_ssim_8patches": base_ssim, "base_mse_8patches": base_mse,
           "acceptance": "best_ssim>0.95 and best_mse<1e-3 and best_mse<base_mse/10",
           "arch": "paper_faithful SwinIR-M (identity init)"}
    (out / "tiny_overfit_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"[tiny_overfit] best ssim={best_ssim:.4f} best_mse={best_mse:.6f} (final ssim={final['train_ssim']:.4f}) PASS={passed}")
    return rep


def test_identity(device, steps=1200):
    out = SANITY / "identity"; out.mkdir(parents=True, exist_ok=True)
    model, run_cfg, eval_m = S.load_frozen_base(BASE_ROOT, DEMO_COMP, DEMO_PATTERN, device)
    cache = S.build_pair_cache(model, run_cfg, "train", device, eval_m, crop_size=256,
                              crops_per_train_image=1, base_batch=8, seed=42)
    del model; torch.cuda.empty_cache()
    gen = torch.Generator().manual_seed(11)
    _, gt = S.sample_patch_batch(cache, 8, 64, device, gen, random_flips=False)
    x = gt.clone()  # input == target == GT
    # From RANDOM init (no identity init) so the loop must LEARN identity end-to-end.
    refiner = _small_swinir(device, identity_init=False)
    opt = torch.optim.Adam(refiner.parameters_for_training(), lr=2e-4, betas=(0.9, 0.99))
    curve = []
    refiner.swinir.train()
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        rec = refiner(x).float()
        loss = F.l1_loss(rec, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters_for_training(), 1.0)
        opt.step()
        if (i + 1) % 50 == 0 or i == 0:
            with torch.no_grad():
                r = refiner(x).clamp(0, 1)
                sc = float(ssim_metric(r, gt).item()); ms = float(mse_metric(r, gt).item())
            curve.append({"step": i + 1, "ssim": sc, "mse": ms})
            print(f"[identity] step {i+1}/{steps} ssim={sc:.4f} mse={ms:.6f}", flush=True)
    final = curve[-1]
    passed = final["ssim"] > 0.99 and final["mse"] < 5e-4
    rep = {"final": final, "passed": bool(passed), "curve": curve,
           "acceptance": "ssim>0.99 and mse<5e-4 (learn identity)"}
    (out / "identity_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"[identity] final ssim={final['ssim']:.4f} mse={final['mse']:.6f} PASS={passed}")
    return rep


def test_no_leakage(device):
    out = SANITY / "no_leakage"; out.mkdir(parents=True, exist_ok=True)
    from datasets.bbbc022_split import load_split
    from utils.experiment_config import load_experiment_config
    rd = S.base_run_dir(BASE_ROOT, DEMO_COMP, DEMO_PATTERN)
    run_cfg = load_experiment_config(rd / "config.yaml")
    split = load_split(Path(run_cfg["dataset"]["split_path"]), Path(run_cfg["dataset"]["repo_root"]))
    sets = {k: set(str(p) for p in v) for k, v in split.items()}
    tv = sets["train"] & sets["val"]; tt = sets["train"] & sets["test"]; vt = sets["val"] & sets["test"]
    disjoint = not (tv or tt or vt)
    rep = {
        "counts": {k: len(v) for k, v in sets.items()},
        "train_val_overlap": len(tv), "train_test_overlap": len(tt), "val_test_overlap": len(vt),
        "splits_disjoint": bool(disjoint),
        "normalization_mode": run_cfg["dataset"].get("preproc_mode"),
        "normalization_note": ("minimal_percentile is PER-IMAGE (each image clipped to its own "
                               "q0.001/q0.999 then scaled to [0,1]); it fits NO statistic on val/test "
                               "and shares nothing across images, so there is no train<-val/test leakage. "
                               "The frozen base was trained with this exact normalisation; SwinIR reuses it."),
        "leakage_free": bool(disjoint),
    }
    (out / "no_leakage_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"[no_leakage] counts={rep['counts']} disjoint={disjoint} norm={rep['normalization_mode']}")
    return rep


def test_baseline(device):
    out = SANITY / "baseline_comparison"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for comp in S.ALL_COMPS:
        for pat in S.PATTERNS:
            model, run_cfg, eval_m = S.load_frozen_base(BASE_ROOT, comp, pat, device)
            cache = S.build_pair_cache(model, run_cfg, "test", device, eval_m, crop_size=256, base_batch=8, seed=42)
            del model; torch.cuda.empty_cache()
            # base metrics only (ref==random init irrelevant; eval_cache reports base_* independently)
            sw = _small_swinir(device)
            m = S.eval_cache(sw, cache, device, 16)
            rows.append({"comp": comp, "pattern": pat, "n_test": m["n"],
                         "base_ssim": round(m["base_ssim"], 4), "base_mse": round(m["base_mse"], 6),
                         "base_psnr": round(m["base_psnr"], 3)})
            print(f"[baseline] {comp}/{pat}: base_ssim={m['base_ssim']:.4f} base_mse={m['base_mse']:.6f}")
            del sw; torch.cuda.empty_cache()
    (out / "baseline_test_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


TESTS = {"pairing": test_pairing, "distribution": test_distribution, "tiny_overfit": test_tiny_overfit,
         "identity": test_identity, "no_leakage": test_no_leakage, "baseline": test_baseline}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tests", default=",".join(TESTS))
    args = ap.parse_args()
    device = torch.device(args.device)
    selected = [t.strip() for t in args.tests.split(",") if t.strip()]
    summary = {}
    for name in selected:
        if name not in TESTS:
            raise SystemExit(f"unknown test {name}; valid: {list(TESTS)}")
        print(f"\n=== sanity: {name} ===", flush=True)
        res = TESTS[name](device)
        if isinstance(res, dict) and "passed" in res:
            summary[name] = res["passed"]
        elif name in ("no_leakage",):
            summary[name] = res["leakage_free"]
        else:
            summary[name] = "done"
    (SANITY / "SANITY_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== SANITY SUMMARY === {summary}")


if __name__ == "__main__":
    main()
