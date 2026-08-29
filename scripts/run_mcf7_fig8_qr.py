#!/usr/bin/env python3
"""MCF7 Figure-8 — the paper's actual Q vs R comparison (Sec 5.6, Fig 8).

Paper Fig 8 rows:
  P = Ground Truth
  Q = "SwinIR reconstruction model"          = locality-aware upsampling + SwinIR
  R = "simple transpose convolution           = transpose-conv upsampling + the
       reconstruction model"                    conventional ReconCNN (Sec 4.3), NO SwinIR

Both conditions: learnable_frequency Ht, x16 compression (d=8, T=4), 256x256
MCF7 channel-2 patches, L1 loss, Algorithm-1 sharpness schedule, Adam.

This trains the MISSING R (w/O SwinIR) condition and a fresh Q so Figure 8 can
be rendered exactly as in the paper. It writes to a NEW frozen directory and
does NOT touch the prior `mcf7_paper_direct_full` run.

Usage (parallel across 2 GPUs):
  python scripts/run_mcf7_fig8_qr.py --only-condition Q --device cuda:0 --full-budget
  python scripts/run_mcf7_fig8_qr.py --only-condition R --device cuda:1 --full-budget
  python scripts/run_mcf7_fig8_qr.py --aggregate-only            # writes summary
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
from models.microscope import DifferentiableMicroscope
from utils.device import resolve_device
from utils.logging import save_measurement_grid

OUT = ROOT / "experiments/swinir_or_highres/mcf7_fig8_qr"
CFG = ROOT / "configs/mcf7_li_swinir_paper_direct.yaml"

# condition key -> (subdir name, backbone)
CONDITIONS = {
    "Q": ("with_swinir_locality", "swinir"),
    "R": ("wo_swinir_transpose", "conventional"),
}


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def _build_swinir_model(cfg: dict) -> SwinIRTable2Model:
    """Q: locality-aware upsampling + SwinIR (learnable Ht)."""
    m_cfg = {
        "image_size": int(cfg["dataset"]["image_size"]),
        "pattern_generator": {
            "mode": "learnable_frequency",
            "num_patterns": int(cfg["pattern_generator"]["num_patterns"]),
            "sigmoid_m": 1.0,
            "random_fixed_m": float(cfg["pattern_generator"].get("random_fixed_m", 10.0)),
            "seed": int(cfg["pattern_generator"].get("seed", 42)),
            "superpixel_factor": int(cfg["pattern_generator"].get("superpixel_factor", 1)),
        },
        "forward_model": {
            "downscale_factor": int(cfg["forward_model"]["downscale_factor"]),
            "use_impulse_psfs": bool(cfg["forward_model"].get("use_impulse_psfs", True)),
        },
        "detector_noise": {"mode": "noise_free", "apply_noise": False},
        "inverse_model": {
            "upsampling": {
                "mode": "locality_aware",
                "downscale_factor": int(cfg["forward_model"]["downscale_factor"]),
                "num_patterns": int(cfg["pattern_generator"]["num_patterns"]),
            }
        },
        "swinir": dict(cfg["swinir"]),
    }
    return SwinIRTable2Model(m_cfg)


def _build_conventional_model(cfg: dict) -> DifferentiableMicroscope:
    """R: transpose-conv upsampling + conventional ReconCNN (learnable Ht, NO SwinIR)."""
    npat = int(cfg["pattern_generator"]["num_patterns"])
    down = int(cfg["forward_model"]["downscale_factor"])
    run_cfg = {
        "dataset": {"image_size": int(cfg["dataset"]["image_size"])},
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
                "mode": "transpose_conv",
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


class CondAdapter:
    """Uniform interface over the two model types."""

    def __init__(self, key: str, model: torch.nn.Module):
        self.key = key
        self.model = model

    def illum_params(self):
        return self.model.illumination_parameters()

    def recon_params(self):
        if isinstance(self.model, SwinIRTable2Model):
            return self.model.swinir_parameters()
        return self.model.inverse_parameters()

    def forward(self, x, m):
        return self.model(x, sigmoid_m=m, apply_noise=False)["x_recon"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _build_loaders(cfg: dict, seed: int, batch: int,
                   n_train: int, n_val: int, n_test: int) -> dict[str, DataLoader]:
    ds_cfg = dict(cfg["dataset"])
    ds_cfg["seed"] = seed
    ds_cfg["num_train"] = n_train
    ds_cfg["num_val"] = n_val
    ds_cfg["num_test"] = n_test
    loaders = {}
    print("Building MCF7 dataloaders (loads TIFFs; a few minutes)...", flush=True)
    for split, shuffle in (("train", True), ("val", False), ("test", False)):
        ds = MCF7Channel2Dataset.from_dict(ds_cfg, split=split)
        bs = batch if split == "train" else 1
        loaders[split] = DataLoader(ds, batch_size=bs, shuffle=shuffle)
        print(f"  -> {split}: {len(ds)} patches", flush=True)
    return loaders


def _schedule_m(epoch: int, baseline: int, m_values: list[float], step: int) -> tuple[float, bool]:
    if epoch < baseline:
        return 1.0, False
    idx = min((epoch - baseline) // max(1, step), len(m_values) - 1)
    return float(m_values[idx]), True


@torch.no_grad()
def _evaluate(adapter: CondAdapter, loader: DataLoader, device, m: float, max_items: int | None = None) -> dict:
    adapter.model.eval()
    mse_s = ssim_s = psnr_s = 0.0
    n = 0
    for batch in loader:
        x = batch.to(device)
        rec = adapter.forward(x, m).clamp(0, 1)
        mse_s += float(mse_metric(rec, x).item())
        ssim_s += float(ssim_metric(rec, x).item())
        psnr_s += float(psnr_metric(rec, x).item())
        n += 1
        if max_items is not None and n >= max_items:
            break
    return {"mse": mse_s / max(1, n), "ssim": ssim_s / max(1, n), "psnr": psnr_s / max(1, n)}


@torch.no_grad()
def _save_examples(adapter: CondAdapter, loader: DataLoader, device, m: float,
                   out_dir: Path, n_examples: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter.model.eval()
    saved = 0
    for batch in loader:
        x = batch.to(device)
        rec = adapter.forward(x, m).clamp(0, 1)
        for j in range(x.shape[0]):
            if saved >= n_examples:
                return
            save_measurement_grid(x[j:j + 1], out_dir / f"gt_{saved:02d}.png")
            save_measurement_grid(rec[j:j + 1], out_dir / f"recon_{saved:02d}.png")
            saved += 1


def _train_condition(key: str, cfg: dict, loaders: dict, device, *,
                     epochs: int, baseline: int, step: int, m_values: list[float],
                     max_steps: int, seed: int, n_examples: int,
                     val_subset: int | None = None) -> dict:
    subdir, backbone = CONDITIONS[key]
    torch.manual_seed(seed)
    model = (_build_swinir_model(cfg) if backbone == "swinir"
             else _build_conventional_model(cfg)).to(device)
    isz = int(cfg["dataset"]["image_size"])
    model(torch.zeros(1, 1, isz, isz, device=device))  # lazy-init buffers
    adapter = CondAdapter(key, model)

    illum_lr = float(cfg["training"]["illumination_lr"])
    recon_lr = float(cfg["training"]["swinir_lr"]) if backbone == "swinir" \
        else float(cfg["training"].get("inverse_lr", 1e-3))
    opt = torch.optim.Adam([
        {"params": adapter.recon_params(), "lr": recon_lr},
        {"params": adapter.illum_params(), "lr": illum_lr},
    ])

    eval_m = float(cfg["training"].get("eval_sigmoid_m", 8.0))
    cond_dir = OUT / subdir
    ckpt_dir = cond_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_val = float("inf")
    best_state = None
    t0 = datetime.now(timezone.utc)

    for epoch in range(epochs):
        ep_t0 = datetime.now(timezone.utc)
        m, unfreeze = _schedule_m(epoch, baseline, m_values, step)
        for p in adapter.illum_params():
            p.requires_grad = unfreeze
        model.train()
        ep_loss, nb = 0.0, 0
        for batch in loaders["train"]:
            if nb >= max_steps:
                break
            x = batch.to(device)
            opt.zero_grad(set_to_none=True)
            rec = adapter.forward(x, m)
            loss = F.l1_loss(rec, x)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        val = _evaluate(adapter, loaders["val"], device, eval_m, max_items=val_subset)
        ep_sec = (datetime.now(timezone.utc) - ep_t0).total_seconds()
        history.append({"epoch": epoch, "m": m, "illum_unfrozen": unfreeze,
                        "train_l1": ep_loss / max(1, nb), "val_mse": val["mse"],
                        "val_ssim": val["ssim"], "epoch_sec": round(ep_sec, 1),
                        "steps": nb})
        print(f"[{key}] epoch {epoch}/{epochs} m={m} illum={unfreeze} steps={nb} "
              f"train_l1={ep_loss/max(1,nb):.5f} val_mse={val['mse']:.5f} "
              f"val_ssim={val['ssim']:.4f} ({ep_sec:.0f}s)", flush=True)
        if val["mse"] < best_val:
            best_val = val["mse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model": best_state, "epoch": epoch, "val_mse": best_val,
                        "condition": key, "backbone": backbone}, ckpt_dir / "best.pt")

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model": model.state_dict(), "condition": key, "backbone": backbone},
               ckpt_dir / "last.pt")

    test = _evaluate(adapter, loaders["test"], device, eval_m)
    _save_examples(adapter, loaders["test"], device, eval_m, cond_dir / "examples", n_examples)

    # save learned illumination patterns
    with torch.no_grad():
        patterns = model.pattern_generator(sigmoid_m=eval_m).detach().cpu()
    (cond_dir / "illumination").mkdir(parents=True, exist_ok=True)
    torch.save(patterns, cond_dir / "illumination" / "patterns.pt")

    result = {
        "condition": key, "subdir": subdir, "backbone": backbone,
        "epochs_run": epochs, "baseline": baseline, "step": step,
        "test_mse": test["mse"], "test_ssim": test["ssim"], "test_psnr": test["psnr"],
        "best_val_mse": best_val,
        "wall_seconds": (datetime.now(timezone.utc) - t0).total_seconds(),
        "checkpoint": str(ckpt_dir / "best.pt"),
    }
    (cond_dir / "result.json").write_text(json.dumps({**result, "history": history}, indent=2),
                                          encoding="utf-8")
    print(f"[{key}] DONE test PSNR={test['psnr']:.2f} SSIM={test['ssim']:.4f} "
          f"MSE={test['mse']:.6f}", flush=True)
    return result


def _aggregate() -> None:
    results = []
    for key, (subdir, _) in CONDITIONS.items():
        rj = OUT / subdir / "result.json"
        if rj.exists():
            d = json.loads(rj.read_text(encoding="utf-8"))
            results.append({k: d[k] for k in
                            ["condition", "subdir", "backbone", "epochs_run",
                             "test_mse", "test_ssim", "test_psnr", "best_val_mse"]})
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "description": ("MCF7 Fig-8 paper comparison: Q=locality+SwinIR, "
                        "R=transpose-conv+conventional ReconCNN (no SwinIR). "
                        "Both learnable Ht, x16, 256x256, L1, Algorithm-1 schedule."),
        "results": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "aggregate_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (OUT / "results.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["condition", "subdir", "backbone",
                                           "epochs_run", "test_mse", "test_ssim",
                                           "test_psnr", "best_val_mse"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"Wrote aggregate → {OUT / 'aggregate_summary.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only-condition", choices=["Q", "R"], default=None)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--epoch-baseline", type=int, default=15)
    ap.add_argument("--epoch-step", type=int, default=2)
    ap.add_argument("--max-steps-per-epoch", type=int, default=10_000_000)
    ap.add_argument("--full-budget", action="store_true",
                    help="paper schedule 230/150/20, no step cap")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-train", type=int, default=None)
    ap.add_argument("--num-val", type=int, default=None)
    ap.add_argument("--num-test", type=int, default=None)
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--val-subset", type=int, default=None,
                    help="evaluate on at most this many val items per epoch (checkpoint selection)")
    ap.add_argument("--pattern-superpixel", type=int, default=1,
                    help="optical super-pixel size for illumination patterns (e.g. 8 = downscale factor)")
    ap.add_argument("--out", type=str, default=None,
                    help="override output experiment directory (keeps prior runs intact)")
    args = ap.parse_args()

    global OUT
    if args.out:
        OUT = Path(args.out)

    if args.aggregate_only:
        _aggregate()
        return

    cfg = _load_yaml(CFG)
    cfg["pattern_generator"]["superpixel_factor"] = int(args.pattern_superpixel)
    device = resolve_device(args.device)
    print(f"Device: {device}  | pattern superpixel: {args.pattern_superpixel}  | out: {OUT}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.full_budget:
        epochs, baseline, step, max_steps = 230, 150, 20, 10_000_000
    else:
        epochs, baseline, step = args.epochs, args.epoch_baseline, args.epoch_step
        max_steps = args.max_steps_per_epoch

    batch = args.batch_size or int(cfg["training"]["batch_size"])
    n_train = args.num_train if args.num_train is not None else int(cfg["dataset"]["num_train"])
    n_val = args.num_val if args.num_val is not None else int(cfg["dataset"]["num_val"])
    n_test = args.num_test if args.num_test is not None else int(cfg["dataset"]["num_test"])
    m_values = [float(v) for v in cfg["algorithm1"]["m_values"]]

    loaders = _build_loaders(cfg, args.seed, batch, n_train, n_val, n_test)

    import shutil
    (OUT / "configs_used").mkdir(exist_ok=True)
    shutil.copy2(CFG, OUT / "configs_used" / CFG.name)

    keys = [args.only_condition] if args.only_condition else ["Q", "R"]
    for key in keys:
        print(f"\n=== Fig-8 condition {key} ({CONDITIONS[key][0]}) ===", flush=True)
        _train_condition(key, cfg, loaders, device, epochs=epochs, baseline=baseline,
                         step=step, m_values=m_values, max_steps=max_steps,
                         seed=args.seed, n_examples=args.n_examples,
                         val_subset=args.val_subset)

    if not args.only_condition:
        _aggregate()


if __name__ == "__main__":
    main()
