#!/usr/bin/env python3
"""MCF7 Figure-9 wCNN condition: the conventional CNN reconstruction at 64x64.

Paper Fig 9 (Sec 5.6) compares, for a "fairer comparison":
  - wSwinIR : locality-aware upsampling + SwinIR @256x256  (== Fig-8 Q; reused)
  - wCNN    : the classical convolutional reconstruction model of Sec 4.3
              (locality-aware upsampling + ReconCNN), trained at 64x64 image size,
              learnable Ht, x16 compression. NO SwinIR.

This script trains ONLY the wCNN condition (wSwinIR is reused from the Fig-8 run).
It writes to a NEW frozen directory and does not touch prior runs. The conventional
model is fully convolutional, so at render time it is tiled over a wide field.

Usage:
  python scripts/run_mcf7_fig9_wcnn.py --device cuda:0 --epochs 72 \
      --epoch-baseline 46 --epoch-step 7 --batch-size 32 --val-subset 40
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets.mcf7_channel2 import MCF7Channel2Dataset
from evaluation.metrics import mse as mse_metric
from evaluation.metrics import psnr as psnr_metric
from evaluation.metrics import ssim as ssim_metric
from models.microscope import DifferentiableMicroscope
from utils.device import resolve_device
from utils.logging import save_measurement_grid

OUT = ROOT / "experiments/swinir_or_highres/mcf7_fig9_wcnn"
CFG = ROOT / "configs/mcf7_li_swinir_paper_direct.yaml"
IMG = 64  # paper: CNN reconstruction trained at 64x64


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _build_model(cfg: dict) -> DifferentiableMicroscope:
    """Conventional pipeline: locality-aware upsampling + ReconCNN at 64x64."""
    npat = int(cfg["pattern_generator"]["num_patterns"])
    down = int(cfg["forward_model"]["downscale_factor"])
    run_cfg = {
        "dataset": {"image_size": IMG},
        "pattern_generator": {
            "mode": "learnable_frequency",
            "num_patterns": npat,
            "sigmoid_m": 1.0,
            "random_fixed_m": float(cfg["pattern_generator"].get("random_fixed_m", 10.0)),
            "seed": int(cfg["pattern_generator"].get("seed", 42)),
            "superpixel_factor": int(cfg["pattern_generator"].get("superpixel_factor", 1)),
        },
        "forward_model": {
            "downscale_factor": down,
            "use_impulse_psfs": bool(cfg["forward_model"].get("use_impulse_psfs", True)),
        },
        "detector_noise": {"mode": "noise_free", "apply_noise": False},
        "inverse_model": {
            "upsampling": {
                "mode": "locality_aware",
                "downscale_factor": down,
                "num_patterns": npat,
            },
            "reconstruction": {
                "in_channels": npat,
                "hidden_channels": [64, 64, 32, 32, 16, 1],
                "kernel_size": 3,
                "padding": 1,
            },
        },
    }
    return DifferentiableMicroscope.from_run_config(run_cfg)


def _build_loaders(cfg: dict, seed: int, batch: int, n_train, n_val, n_test):
    ds_cfg = dict(cfg["dataset"])
    ds_cfg["seed"] = seed
    ds_cfg["patch_size"] = IMG
    ds_cfg["image_size"] = IMG
    ds_cfg["num_train"] = n_train
    ds_cfg["num_val"] = n_val
    ds_cfg["num_test"] = n_test
    loaders = {}
    print("Building MCF7 64x64 dataloaders (loads TIFFs; a few minutes)...", flush=True)
    for split, shuffle in (("train", True), ("val", False), ("test", False)):
        ds = MCF7Channel2Dataset.from_dict(ds_cfg, split=split)
        bs = batch if split == "train" else 1
        loaders[split] = DataLoader(ds, batch_size=bs, shuffle=shuffle)
        print(f"  -> {split}: {len(ds)} patches", flush=True)
    return loaders


def _schedule_m(epoch, baseline, m_values, step):
    if epoch < baseline:
        return 1.0, False
    idx = min((epoch - baseline) // max(1, step), len(m_values) - 1)
    return float(m_values[idx]), True


@torch.no_grad()
def _evaluate(model, loader, device, m, max_items=None):
    model.eval()
    mse_s = ssim_s = psnr_s = 0.0
    n = 0
    for batch in loader:
        x = batch.to(device)
        rec = model(x, sigmoid_m=m, apply_noise=False)["x_recon"].clamp(0, 1)
        mse_s += float(mse_metric(rec, x).item())
        ssim_s += float(ssim_metric(rec, x).item())
        psnr_s += float(psnr_metric(rec, x).item())
        n += 1
        if max_items is not None and n >= max_items:
            break
    return {"mse": mse_s / max(1, n), "ssim": ssim_s / max(1, n), "psnr": psnr_s / max(1, n)}


@torch.no_grad()
def _save_examples(model, loader, device, m, out_dir, n_examples):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for batch in loader:
        x = batch.to(device)
        rec = model(x, sigmoid_m=m, apply_noise=False)["x_recon"].clamp(0, 1)
        for j in range(x.shape[0]):
            if saved >= n_examples:
                return
            save_measurement_grid(x[j:j + 1], out_dir / f"gt_{saved:02d}.png")
            save_measurement_grid(rec[j:j + 1], out_dir / f"recon_{saved:02d}.png")
            saved += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=72)
    ap.add_argument("--epoch-baseline", type=int, default=46)
    ap.add_argument("--epoch-step", type=int, default=7)
    ap.add_argument("--max-steps-per-epoch", type=int, default=10_000_000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-train", type=int, default=3000)
    ap.add_argument("--num-val", type=int, default=100)
    ap.add_argument("--num-test", type=int, default=100)
    ap.add_argument("--val-subset", type=int, default=40)
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--pattern-superpixel", type=int, default=1,
                    help="optical super-pixel size for illumination patterns (e.g. 8 = downscale factor)")
    ap.add_argument("--out", type=str, default=None,
                    help="override output experiment directory (keeps prior runs intact)")
    args = ap.parse_args()

    global OUT
    if args.out:
        OUT = Path(args.out)

    cfg = _load_yaml(CFG)
    cfg["pattern_generator"]["superpixel_factor"] = int(args.pattern_superpixel)
    device = resolve_device(args.device)
    print(f"Device: {device}  | pattern superpixel: {args.pattern_superpixel}  | out: {OUT}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    m_values = [float(v) for v in cfg["algorithm1"]["m_values"]]
    eval_m = float(cfg["training"].get("eval_sigmoid_m", 8.0))
    illum_lr = float(cfg["training"]["illumination_lr"])
    inverse_lr = float(cfg["training"].get("inverse_lr", 1e-3))

    loaders = _build_loaders(cfg, args.seed, args.batch_size,
                             args.num_train, args.num_val, args.num_test)

    import shutil
    (OUT / "configs_used").mkdir(exist_ok=True)
    shutil.copy2(CFG, OUT / "configs_used" / CFG.name)

    torch.manual_seed(args.seed)
    model = _build_model(cfg).to(device)
    model(torch.zeros(1, 1, IMG, IMG, device=device))
    opt = torch.optim.Adam([
        {"params": model.inverse_parameters(), "lr": inverse_lr},
        {"params": model.illumination_parameters(), "lr": illum_lr},
    ])

    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")
    best_state = None
    t0 = datetime.now(timezone.utc)

    for epoch in range(args.epochs):
        ep_t0 = datetime.now(timezone.utc)
        m, unfreeze = _schedule_m(epoch, args.epoch_baseline, m_values, args.epoch_step)
        for p in model.illumination_parameters():
            p.requires_grad = unfreeze
        model.train()
        ep_loss, nb = 0.0, 0
        for batch in loaders["train"]:
            if nb >= args.max_steps_per_epoch:
                break
            x = batch.to(device)
            opt.zero_grad(set_to_none=True)
            rec = model(x, sigmoid_m=m, apply_noise=False)["x_recon"]
            loss = F.l1_loss(rec, x)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        val = _evaluate(model, loaders["val"], device, eval_m, max_items=args.val_subset)
        ep_sec = (datetime.now(timezone.utc) - ep_t0).total_seconds()
        history.append({"epoch": epoch, "m": m, "illum_unfrozen": unfreeze,
                        "train_l1": ep_loss / max(1, nb), "val_mse": val["mse"],
                        "val_ssim": val["ssim"], "epoch_sec": round(ep_sec, 1)})
        print(f"[wCNN] epoch {epoch}/{args.epochs} m={m} illum={unfreeze} steps={nb} "
              f"train_l1={ep_loss/max(1,nb):.5f} val_mse={val['mse']:.5f} "
              f"val_ssim={val['ssim']:.4f} ({ep_sec:.0f}s)", flush=True)
        if val["mse"] < best_val:
            best_val = val["mse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model": best_state, "epoch": epoch, "val_mse": best_val,
                        "condition": "wCNN", "backbone": "conventional_locality_cnn",
                        "image_size": IMG}, ckpt_dir / "best.pt")

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model": model.state_dict(), "condition": "wCNN", "image_size": IMG},
               ckpt_dir / "last.pt")

    test = _evaluate(model, loaders["test"], device, eval_m)
    _save_examples(model, loaders["test"], device, eval_m, OUT / "examples", args.n_examples)
    with torch.no_grad():
        patterns = model.pattern_generator(sigmoid_m=eval_m).detach().cpu()
    (OUT / "illumination").mkdir(parents=True, exist_ok=True)
    torch.save(patterns, OUT / "illumination" / "patterns.pt")

    result = {
        "condition": "wCNN", "backbone": "conventional_locality_cnn", "image_size": IMG,
        "epochs_run": args.epochs, "baseline": args.epoch_baseline, "step": args.epoch_step,
        "test_mse": test["mse"], "test_ssim": test["ssim"], "test_psnr": test["psnr"],
        "best_val_mse": best_val,
        "wall_seconds": (datetime.now(timezone.utc) - t0).total_seconds(),
        "checkpoint": str(ckpt_dir / "best.pt"),
    }
    (OUT / "result.json").write_text(json.dumps({**result, "history": history}, indent=2),
                                     encoding="utf-8")
    with (OUT / "results.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["condition", "backbone", "image_size",
                                           "epochs_run", "test_mse", "test_ssim",
                                           "test_psnr", "best_val_mse"])
        w.writeheader()
        w.writerow({k: result[k] for k in ["condition", "backbone", "image_size",
                                           "epochs_run", "test_mse", "test_ssim",
                                           "test_psnr", "best_val_mse"]})
    (OUT / "status.md").write_text(
        f"# MCF7 Fig-9 wCNN (conventional locality+CNN @64x64)\n\n**Status:** COMPLETE\n"
        f"**Updated:** {datetime.now(timezone.utc).isoformat()}\n\n"
        f"test PSNR {test['psnr']:.2f}, SSIM {test['ssim']:.4f}, MSE {test['mse']:.6f}\n",
        encoding="utf-8")
    print(f"[wCNN] DONE test PSNR={test['psnr']:.2f} SSIM={test['ssim']:.4f} "
          f"MSE={test['mse']:.6f}", flush=True)


if __name__ == "__main__":
    main()
