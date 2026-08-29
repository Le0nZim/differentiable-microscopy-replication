#!/usr/bin/env python3
"""Train SwinIR refiners for the Figure-3 "+SwinIR" columns (paper-faithful).

Paper Fig. 3 / Table S1 procedure (paper_sources/paper.md, U2OS content-aware
section): *"We append the SwinIR model at the end of the trained end-to-end
model. Here we set the upscaling factor of the SwinIR to 1 (i.e. no upscaling).
Finally, we train the SwinIR to super-resolve the output of the trained
end-to-end model."*

We therefore reuse the **already-trained** ``bbbc022_content_aware_v2``
microscopes (frozen), append a SwinIR (upscale=1) built with the exact same
``build_swinir_from_config`` used by the Table-2 / Fig-7 pipeline, and train only
the SwinIR to map ``x_base -> x_gt``. The SwinIR columns therefore share the
*identical* base reconstruction as the non-SwinIR columns (a clean controlled
comparison, exactly as the paper intends).

Only the pseudo-random and learnable conditions get a SwinIR column (paper Fig 3
filled markers): ``random_fixed`` and ``learnable_frequency`` x {x16,x64,x256,x1024}.

Each cell writes:
    <out_root>/<comp>_<pattern>/refiner_best.pt   (SwinIR refiner weights)
    <out_root>/<comp>_<pattern>/metrics.json      (base vs refined MSE/SSIM)

Run (two GPUs in parallel):
    python scripts/train_fig03_swinir_columns.py --device cuda:0 --comps x16,x64
    python scripts/train_fig03_swinir_columns.py --device cuda:1 --comps x256,x1024
Then aggregate:
    python scripts/train_fig03_swinir_columns.py --aggregate-only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.swinir.refinement_model import MicroscopeSwinIRRefinement  # noqa: E402
from evaluation.metrics import ssim as ssim_metric  # noqa: E402
from models.microscope import DifferentiableMicroscope  # noqa: E402
from training.dataloaders import build_dataloader  # noqa: E402
from utils.experiment_config import load_experiment_config  # noqa: E402

# Paper Fig-3 SwinIR columns: only pseudo-random + learnable get a SwinIR refiner.
PATTERNS = ["random_fixed", "learnable_frequency"]
ALL_COMPS = ["x16", "x64", "x256", "x1024"]

# Eval/forward sigmoid sharpness per pattern (matches how the base results.csv
# was computed: learnable uses sharpen_eval_m=10; fixed patterns ignore m).
EVAL_M = {"random_fixed": None, "learnable_frequency": 10.0}

# SwinIR config: same family/capacity as the Table-2 / MCF7 (Fig 7/8) path,
# proven stable at 256x256 (configs/mcf7_li_swinir_paper_direct.yaml).
SWINIR_CFG = {
    "upscale": 1,
    "in_chans": 1,
    "img_size": 256,
    "window_size": 8,
    "upsampler": "",
    "embed_dim": 96,
    "depths": [2, 2, 2, 2, 2, 2],
    "num_heads": [3, 3, 3, 3, 3, 3],
    "mlp_ratio": 2,
    "resi_connection": "1conv",
    "img_range": 1.0,
}


def run_dir(exp_root: Path, comp: str, pattern: str) -> Path:
    return exp_root / f"bbbc022_{comp}_{pattern}_seed42"


def lr_at(step: int, *, base_lr: float, steps: int, warmup: int, min_frac: float) -> float:
    """Linear warmup then cosine decay to ``base_lr * min_frac`` (stabilises SwinIR)."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, steps - warmup)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_frac + (1.0 - min_frac) * cos)


def load_microscope(cfg: dict, ckpt: Path, device: torch.device, image_size: int, eval_m) -> DifferentiableMicroscope:
    model = DifferentiableMicroscope.from_run_config(cfg).to(device)
    model(torch.zeros(1, 1, image_size, image_size, device=device), sigmoid_m=eval_m or 10.0, apply_noise=False)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def evaluate(ref: MicroscopeSwinIRRefinement, loader, device: torch.device, eval_m) -> dict:
    ref.eval()
    base_mse = base_ssim = ref_mse = ref_ssim = 0.0
    n = 0
    for batch in loader:
        x = (batch if torch.is_tensor(batch) else batch[0]).to(device)
        out = ref(x, sigmoid_m=eval_m, apply_noise=False)
        xb = out["x_base"].clamp(0, 1)
        xr = out["x_recon"].clamp(0, 1)
        for j in range(x.shape[0]):
            t = x[j : j + 1]
            base_mse += F.mse_loss(xb[j : j + 1], t).item()
            ref_mse += F.mse_loss(xr[j : j + 1], t).item()
            base_ssim += float(ssim_metric(xb[j : j + 1], t).item())
            ref_ssim += float(ssim_metric(xr[j : j + 1], t).item())
            n += 1
    return {
        "base_mse": base_mse / max(1, n),
        "base_ssim": base_ssim / max(1, n),
        "ref_mse": ref_mse / max(1, n),
        "ref_ssim": ref_ssim / max(1, n),
        "n": n,
    }


def train_cell(
    exp_root: Path,
    out_root: Path,
    comp: str,
    pattern: str,
    device: torch.device,
    *,
    steps: int,
    batch: int,
    lr: float,
    val_every: int,
    warmup: int = 200,
    grad_clip: float = 1.0,
    min_lr_frac: float = 0.02,
) -> dict:
    rd = run_dir(exp_root, comp, pattern)
    cfg = load_experiment_config(rd / "config.yaml")
    cfg["experiment"]["device"] = str(device)
    image_size = int(cfg["dataset"]["image_size"])
    eval_m = EVAL_M[pattern]

    # SwinIR-sized batch for train/val/test loaders (256x256 SwinIR is heavy).
    cfg["training"]["batch_size"] = batch
    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "val")
    test_loader = build_dataloader(cfg, "test")

    microscope = load_microscope(cfg, rd / "checkpoints" / "best.pt", device, image_size, eval_m)
    ref = MicroscopeSwinIRRefinement(microscope, dict(SWINIR_CFG), {"mode": "direct"}).to(device)
    ref.set_freeze_base(True)
    # Keep the frozen microscope in EVAL mode throughout: its reconstruction CNN
    # has BatchNorm, whose running stats would otherwise be polluted by the
    # refinement batches during train-mode forward passes (they update even under
    # no_grad), making the "frozen" base drift. Only the SwinIR refiner trains.
    ref.microscope.eval()

    swinir_params = ref.swinir_parameters()
    opt = torch.optim.Adam(swinir_params, lr=lr, betas=(0.9, 0.99))

    cell_dir = out_root / f"{comp}_{pattern}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint selection: SSIM-primary with an MSE-no-regression gate. Minimising
    # MSE alone can pick a degenerate over-smoothed refiner (low MSE, low SSIM) at
    # extreme compression; the paper reports BOTH metrics improving with SwinIR, so
    # we maximise val SSIM among checkpoints that do not regress MSE vs the (frozen)
    # base, falling back to max SSIM if none beat base MSE.
    best_score = -float("inf")
    best_val_mse = float("inf")
    best_val_ssim = -float("inf")
    best_state = None
    step = 0
    t0 = time.time()
    print(f"[{comp}/{pattern}] start: train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"test={len(test_loader.dataset)} eval_m={eval_m} lr={lr} warmup={warmup} clip={grad_clip}", flush=True)
    while step < steps:
        for batch_data in train_loader:
            if step >= steps:
                break
            cur_lr = lr_at(step, base_lr=lr, steps=steps, warmup=warmup, min_frac=min_lr_frac)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            x = (batch_data if torch.is_tensor(batch_data) else batch_data[0]).to(device)
            ref.offline_refiner.train()
            ref.microscope.eval()  # keep frozen base (BatchNorm) stable
            opt.zero_grad(set_to_none=True)
            out = ref(x, sigmoid_m=eval_m, apply_noise=False)
            loss = F.l1_loss(out["x_recon"], x)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(swinir_params, grad_clip)
            opt.step()
            step += 1
            if step % val_every == 0 or step == steps:
                vm = evaluate(ref, val_loader, device, eval_m)
                gated = vm["ref_mse"] <= vm["base_mse"]
                score = vm["ref_ssim"] if gated else vm["ref_ssim"] - 1.0
                is_best = score > best_score
                if is_best:
                    best_score = score
                    best_val_mse = vm["ref_mse"]
                    best_val_ssim = vm["ref_ssim"]
                    best_state = {k: v.detach().cpu().clone() for k, v in ref.offline_refiner.state_dict().items()}
                tag = " *best" if is_best else ""
                print(f"[{comp}/{pattern}] step {step}/{steps} lr={cur_lr:.2e} loss={loss.item():.4f} "
                      f"val base_mse={vm['base_mse']:.5f} ref_mse={vm['ref_mse']:.5f} "
                      f"base_ssim={vm['base_ssim']:.4f} ref_ssim={vm['ref_ssim']:.4f} "
                      f"({(time.time()-t0)/60:.1f} min){tag}", flush=True)

    if best_state is not None:
        ref.offline_refiner.load_state_dict(best_state)
    torch.save({"refiner_state_dict": ref.offline_refiner.state_dict(),
                "swinir_cfg": SWINIR_CFG, "comp": comp, "pattern": pattern},
               cell_dir / "refiner_best.pt")

    tm = evaluate(ref, test_loader, device, eval_m)
    metrics = {
        "comp": comp,
        "pattern": pattern,
        "steps": steps,
        "batch": batch,
        "lr": lr,
        "eval_m": eval_m,
        "selection": "max_val_ssim_with_mse_no_regression_gate",
        "best_val_ref_mse": best_val_mse,
        "best_val_ref_ssim": best_val_ssim,
        "test": tm,
        "minutes": (time.time() - t0) / 60.0,
    }
    (cell_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[{comp}/{pattern}] DONE test: base_mse={tm['base_mse']:.5f} -> ref_mse={tm['ref_mse']:.5f} | "
          f"base_ssim={tm['base_ssim']:.4f} -> ref_ssim={tm['ref_ssim']:.4f} "
          f"({metrics['minutes']:.1f} min)", flush=True)

    del ref, microscope
    torch.cuda.empty_cache()
    return metrics


def aggregate(out_root: Path) -> None:
    rows = []
    for comp in ALL_COMPS:
        for pattern in PATTERNS:
            mp = out_root / f"{comp}_{pattern}" / "metrics.json"
            if not mp.exists():
                continue
            m = json.loads(mp.read_text(encoding="utf-8"))
            t = m["test"]
            rows.append({
                "compression": comp,
                "pattern": pattern,
                "base_mse": t["base_mse"],
                "base_ssim": t["base_ssim"],
                "swinir_mse": t["ref_mse"],
                "swinir_ssim": t["ref_ssim"],
                "n_test": t["n"],
            })
    out_csv = out_root / "swinir_results.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["compression", "pattern", "base_mse", "base_ssim",
                                           "swinir_mse", "swinir_ssim", "n_test"])
        w.writeheader()
        w.writerows(rows)
    print(f"[aggregate] wrote {out_csv} ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['compression']:>6} {r['pattern']:<20} "
              f"SSIM {r['base_ssim']:.4f} -> {r['swinir_ssim']:.4f} | "
              f"MSE {r['base_mse']:.5f} -> {r['swinir_mse']:.5f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", default=str(ROOT / "experiments/figure03_content_aware/base"))
    ap.add_argument("--out-root", default=None, help="default: <exp-root>/swinir")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--comps", default=",".join(ALL_COMPS), help="comma list of compressions to run")
    ap.add_argument("--patterns", default=",".join(PATTERNS), help="comma list of patterns to run")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--min-lr-frac", type=float, default=0.02)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    exp_root = Path(args.exp_root)
    out_root = Path(args.out_root) if args.out_root else exp_root / "swinir"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate(out_root)
        return

    device = torch.device(args.device)
    comps = [c.strip() for c in args.comps.split(",") if c.strip()]
    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    for comp in comps:
        for pattern in patterns:
            train_cell(exp_root, out_root, comp, pattern, device,
                       steps=args.steps, batch=args.batch, lr=args.lr, val_every=args.val_every,
                       warmup=args.warmup, grad_clip=args.grad_clip, min_lr_frac=args.min_lr_frac)

    aggregate(out_root)


if __name__ == "__main__":
    main()
