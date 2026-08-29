#!/usr/bin/env python3
"""Paper-direct MCF7 LI+SwinIR (Fig 8-9, Sec 5.6).

Faithful path: SwinIR is the reconstruction model (replaces CNN), upscale=1, fed by the
locality-aware upsampling block; trained END-TO-END with learnable illumination using the
Algorithm 1 schedule (epochs=230, epochbaseline=150, epochcutoff=150, epochstep=20), L1 loss.

This is DISTINCT from the residual/offline stabilization adaptation in
the removed residual/offline stabilization path and does NOT overwrite prior runs.

Compute scaling: ratio-preserving epoch scale + steps/epoch cap are logged as
IMPLEMENTATION_FALLBACK_COMPUTE_SCALED (paper batch=32 / 230 epochs at 256x256 SwinIR
end-to-end is single-GPU-infeasible). The schedule SHAPE (warmup fraction, hardening
cadence) is preserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.swinir.table2_pipeline import SwinIRTable2Model
from datasets.mcf7_channel2 import MCF7Channel2Dataset
from evaluation.metrics import mse as mse_metric
from evaluation.metrics import psnr as psnr_metric
from evaluation.metrics import ssim as ssim_metric
from utils.device import resolve_device
from utils.logging import save_measurement_grid

OUT = ROOT / "experiments/swinir_or_highres/mcf7_paper_direct_full"
CFG = ROOT / "configs/mcf7_li_swinir_paper_direct.yaml"


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _model_config(cfg: dict, learnable: bool) -> dict[str, Any]:
    return {
        "image_size": int(cfg["dataset"]["image_size"]),
        "pattern_generator": {
            "mode": "learnable_frequency" if learnable else "random_fixed",
            "num_patterns": int(cfg["pattern_generator"]["num_patterns"]),
            "sigmoid_m": 1.0,
            "random_fixed_m": float(cfg["pattern_generator"].get("random_fixed_m", 10.0)),
            "seed": int(cfg["pattern_generator"].get("seed", 42)),
        },
        "forward_model": {
            "downscale_factor": int(cfg["forward_model"]["downscale_factor"]),
            "use_impulse_psfs": bool(cfg["forward_model"].get("use_impulse_psfs", True)),
        },
        "detector_noise": {"mode": "noise_free", "apply_noise": False},
        "inverse_model": {
            "upsampling": {
                "mode": cfg["inverse_model"]["upsampling"]["mode"],
                "downscale_factor": int(cfg["forward_model"]["downscale_factor"]),
                "num_patterns": int(cfg["pattern_generator"]["num_patterns"]),
            }
        },
        "swinir": dict(cfg["swinir"]),
    }


def _build_loaders(cfg: dict, seed: int) -> dict[str, DataLoader]:
    ds_cfg = dict(cfg["dataset"])
    ds_cfg["seed"] = seed
    batch = int(cfg["training"]["batch_size"])
    loaders = {}
    print("Building MCF7 dataloaders (loads TIFFs; train split ~3000 patches, several minutes)...", flush=True)
    for split in ("train", "val", "test"):
        print(f"  -> {split} split...", flush=True)
        ds = MCF7Channel2Dataset.from_dict(ds_cfg, split=split)
        loaders[split] = DataLoader(ds, batch_size=batch, shuffle=(split == "train"))
        print(f"  -> {split}: {len(ds)} patches ready", flush=True)
    return loaders


def _schedule_m(epoch: int, baseline: int, m_values: list[float], step: int) -> tuple[float, bool]:
    """Return (sigmoid_m, illumination_unfrozen) for an epoch.

    Warmup [0, baseline): m=1, illumination frozen (inverse/SwinIR warmup).
    Hardening [baseline, end): unfreeze illumination, step m through m_values every `step` epochs.
    """
    if epoch < baseline:
        return 1.0, False
    idx = (epoch - baseline) // max(1, step)
    idx = min(idx, len(m_values) - 1)
    return float(m_values[idx]), True


@torch.no_grad()
def _evaluate(model: SwinIRTable2Model, loader: DataLoader, device: torch.device, m: float, learnable: bool) -> dict:
    model.eval()
    mse_s = ssim_s = psnr_s = 0.0
    n = 0
    for batch in loader:
        x = batch.to(device)
        out = model(x, sigmoid_m=m if learnable else None, apply_noise=False)
        rec = out["x_recon"].clamp(0, 1)
        mse_s += float(mse_metric(rec, x).item())
        ssim_s += float(ssim_metric(rec, x).item())
        psnr_s += float(psnr_metric(rec, x).item())
        n += 1
    return {"mse": mse_s / n, "ssim": ssim_s / n, "psnr": psnr_s / n}


def _save_examples(model: SwinIRTable2Model, loader: DataLoader, device: torch.device, m: float, learnable: bool, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device)
            out = model(x, sigmoid_m=m if learnable else None, apply_noise=False)
            rec = out["x_recon"].clamp(0, 1)
            save_measurement_grid(x[:1], out_dir / "ground_truth.png")
            save_measurement_grid(rec[:1], out_dir / "reconstruction.png")
            save_measurement_grid((rec[:1] - x[:1]).abs(), out_dir / "error_map.png")
            break


def _train_condition(name: str, learnable: bool, cfg: dict, loaders: dict, device: torch.device, epochs: int, baseline: int, step: int, m_values: list[float], max_steps_per_epoch: int, seed: int) -> dict:
    torch.manual_seed(seed)
    model = SwinIRTable2Model(_model_config(cfg, learnable)).to(device)
    model(torch.zeros(1, 1, cfg["dataset"]["image_size"], cfg["dataset"]["image_size"], device=device))

    illum_lr = float(cfg["training"]["illumination_lr"])
    swinir_lr = float(cfg["training"]["swinir_lr"])
    recon_params = model.swinir_parameters()
    param_groups = [{"params": recon_params, "lr": swinir_lr}]
    if learnable:
        param_groups.append({"params": model.illumination_parameters(), "lr": illum_lr})
    opt = torch.optim.Adam(param_groups)

    eval_m = float(cfg["training"].get("eval_sigmoid_m", 8.0))
    history = []
    best_val = float("inf")
    best_state = None
    ckpt_dir = OUT / f"mcf7_x16_{name}" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        m, unfreeze = _schedule_m(epoch, baseline, m_values, step)
        if learnable:
            for p in model.illumination_parameters():
                p.requires_grad = unfreeze
        model.train()
        ep_loss = 0.0
        nb = 0
        for batch in loaders["train"]:
            if nb >= max_steps_per_epoch:
                break
            x = batch.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x, sigmoid_m=m if learnable else None, apply_noise=False)
            loss = F.l1_loss(out["x_recon"], x)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        val = _evaluate(model, loaders["val"], device, eval_m if learnable else m, learnable)
        history.append({"epoch": epoch, "m": m, "illum_unfrozen": unfreeze, "train_loss": ep_loss / max(1, nb), "val_mse": val["mse"], "val_ssim": val["ssim"]})
        print(f"[{name}] epoch {epoch}/{epochs} m={m} illum={unfreeze} train_l1={ep_loss/max(1,nb):.5f} val_mse={val['mse']:.5f} val_ssim={val['ssim']:.4f}", flush=True)
        if val["mse"] < best_val:
            best_val = val["mse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_dir / "best.pt")

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), ckpt_dir / "last.pt")

    test = _evaluate(model, loaders["test"], device, eval_m if learnable else 1.0, learnable)
    _save_examples(model, loaders["test"], device, eval_m if learnable else 1.0, learnable, OUT / f"mcf7_x16_{name}" / "examples")

    finite = all(torch.isfinite(torch.tensor(h["train_loss"])).item() for h in history)
    return {
        "name": name,
        "learnable": learnable,
        "epochs_run": epochs,
        "test_mse": test["mse"],
        "test_ssim": test["ssim"],
        "test_psnr": test["psnr"],
        "best_val_mse": best_val,
        "loss_finite": finite,
        "history": history,
        "checkpoint": str(ckpt_dir / "best.pt"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-direct MCF7 LI+SwinIR runner")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=24, help="executed epochs (paper 230; ratio-preserving scale)")
    parser.add_argument("--epoch-baseline", type=int, default=15, help="warmup epochs (paper 150; ~0.65 of epochs)")
    parser.add_argument("--epoch-step", type=int, default=2, help="hardening cadence in epochs (paper 20; ~0.087 of epochs)")
    parser.add_argument("--max-steps-per-epoch", type=int, default=150, help="batches/epoch cap (compute scaling; logged)")
    parser.add_argument("--full-budget", action="store_true", help="use paper 230/150/20 epochs (very long)")
    args = parser.parse_args()

    print("MCF7 paper-direct LI+SwinIR starting...", flush=True)
    cfg = _load_yaml(CFG)
    device = resolve_device(args.device)
    print(f"Device: {device}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "status.md").write_text(
        f"# MCF7 paper-direct LI+SwinIR\n\n**Status:** RUNNING\n**Updated:** {datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )

    if args.full_budget:
        epochs, baseline, step = 230, 150, 20
        max_spe = 10_000_000
    else:
        epochs, baseline, step = args.epochs, args.epoch_baseline, args.epoch_step
        max_spe = args.max_steps_per_epoch

    m_values = [float(v) for v in cfg["algorithm1"]["m_values"]]
    loaders = _build_loaders(cfg, args.seed)

    import shutil

    (OUT / "configs_used").mkdir(exist_ok=True)
    shutil.copy2(CFG, OUT / "configs_used/mcf7_li_swinir_paper_direct.yaml")

    results = []
    for name, learnable in [("random_fixed_locality_swinir", False), ("learnable_locality_swinir", True)]:
        print(f"\n=== paper-direct {name} (learnable={learnable}) ===", flush=True)
        results.append(
            _train_condition(name, learnable, cfg, loaders, device, epochs, baseline, step, m_values, max_spe, args.seed)
        )

    no_li = next(r for r in results if not r["learnable"])
    li = next(r for r in results if r["learnable"])
    comparison = {
        "li_improves_mse": li["test_mse"] < no_li["test_mse"],
        "li_improves_ssim": li["test_ssim"] > no_li["test_ssim"],
        "all_finite": all(r["loss_finite"] for r in results),
    }

    paper_target = {"epochs": 230, "epoch_baseline": 150, "epoch_cutoff": 150, "epoch_step": 20, "batch_size": 32}
    executed = {"epochs": epochs, "epoch_baseline": baseline, "epoch_step": step, "max_steps_per_epoch": max_spe, "batch_size": int(cfg["training"]["batch_size"])}

    payload = {
        "label": "PAPER_ALIGNED_ATTEMPTED MCF7 LI+SwinIR (SwinIR replaces CNN, end-to-end, Algorithm 1, L1)",
        "paper_target": paper_target,
        "executed": executed,
        "compute_scaling": "IMPLEMENTATION_FALLBACK_COMPUTE_SCALED" if not args.full_budget else "FULL_BUDGET",
        "deviations": [
            "batch 32 -> 8 (256x256 SwinIR end-to-end memory); logged",
            "epochs 230 -> executed (ratio-preserving) unless --full-budget; logged",
            "steps/epoch capped (compute scaling); logged",
            "SwinIR embed_dim 96 depths [2]*6 (PAPER_UNSPECIFIED_ARCH_FALLBACK; vendor 180 infeasible at 256)",
            "MCF7 loss PAPER_UNSPECIFIED -> L1 (eq.12)",
            "MCF7 Ht LR PAPER_UNSPECIFIED -> 0.1 (Sec 5.6 SwinIR context)",
        ],
        "results": results,
        "comparison": comparison,
    }
    (OUT / "aggregate_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (OUT / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "learnable", "test_mse", "test_ssim", "test_psnr", "best_val_mse", "loss_finite"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in ["name", "learnable", "test_mse", "test_ssim", "test_psnr", "best_val_mse", "loss_finite"]})

    (OUT / "error_maps").mkdir(exist_ok=True)
    for r in results:
        src = OUT / f"mcf7_x16_{r['name']}" / "examples" / "error_map.png"
        if src.exists():
            shutil.copy2(src, OUT / "error_maps" / f"{r['name']}_error_map.png")

    (OUT / "report.md").write_text(
        "# MCF7 paper-direct LI+SwinIR\n\n"
        "**Label:** PAPER_ALIGNED_ATTEMPTED (SwinIR replaces CNN, end-to-end, Algorithm 1, L1)\n\n"
        f"- random/no-LI: test MSE {no_li['test_mse']:.5f}, SSIM {no_li['test_ssim']:.4f}, PSNR {no_li['test_psnr']:.2f}\n"
        f"- learnable/LI: test MSE {li['test_mse']:.5f}, SSIM {li['test_ssim']:.4f}, PSNR {li['test_psnr']:.2f}\n"
        f"- LI improves MSE: {comparison['li_improves_mse']}; LI improves SSIM: {comparison['li_improves_ssim']}\n\n"
        f"Paper target: {paper_target}\nExecuted: {executed}\n\n"
        "Compute-scaled (ratio-preserving) faithful attempt. Not exact Fig 8-9. "
        "Distinct from residual/offline stabilization adaptation.\n",
        encoding="utf-8",
    )
    (OUT / "status.md").write_text(
        f"# MCF7 paper-direct LI+SwinIR\n\n**Status:** COMPLETE\n**Updated:** {datetime.now(timezone.utc).isoformat()}\n\n"
        f"LI improves MSE: {comparison['li_improves_mse']}; SSIM: {comparison['li_improves_ssim']}; finite: {comparison['all_finite']}\n",
        encoding="utf-8",
    )
    print(json.dumps({"comparison": comparison, "results": [{k: r[k] for k in ['name', 'test_mse', 'test_ssim']} for r in results]}, indent=2))


if __name__ == "__main__":
    main()
