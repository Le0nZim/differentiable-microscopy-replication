#!/usr/bin/env python3
"""Train the paper-faithful Fig-3 SwinIR refinement stage (one config, many cells).

Protocol (paper §5.6 U2OS/Fig-3):
  For each compression x {x16,x64,x256,x1024} and illumination x {pseudo_random,
  learnable}: load the frozen content-aware base model, freeze it entirely (forward
  microscope + Ht + locality upsampling + recon CNN), and train ONLY a SwinIR
  (upscale=1) to map frozen_base(x).x_recon -> x_gt.

Reuses the validated Table-2/Fig-7 SwinIR arch + loss stack via
``src.baselines.swinir.fig3_refine_stage``. Nothing here mutates any frozen run.

Outputs per cell under <out-root>/<comp>/<illum>/:
  checkpoints/best.pt  checkpoints/last.pt  metrics.json  config.yaml  grids/*.png

Examples:
  # one cell
  python scripts/figure03_content_aware/train_swinir.py --config <cfg.yaml> --comps x256 \
      --patterns random_fixed --device cuda:0
  # a whole config across cells (sequential)
  python scripts/figure03_content_aware/train_swinir.py --config <cfg.yaml> --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.swinir import fig3_refine_stage as S  # noqa: E402

# Map the config/CLI illumination alias to the base-run pattern name.
ILLUM_ALIAS = {"pseudo_random": "random_fixed", "random_fixed": "random_fixed",
               "learnable": "learnable_frequency", "learnable_frequency": "learnable_frequency"}


def _autocast(device: torch.device, amp_dtype):
    if amp_dtype is not None and device.type == "cuda":
        return torch.autocast("cuda", dtype=amp_dtype)
    class _n:
        def __enter__(self): return None
        def __exit__(self, *a): return False
    return _n()


def _amp_dtype(name: str):
    name = str(name).lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    return None


def save_val_grid(refiner, cache, device, path: Path, n: int = 6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    refiner.eval()
    n = min(n, len(cache))
    with torch.no_grad():
        xb = cache.x_base[:n].to(device, torch.float32)
        gt = cache.gt[:n].to(device, torch.float32)
        rec = refiner(xb).clamp(0, 1).cpu()
    xb = xb.clamp(0, 1).cpu()
    gt = gt.cpu()
    fig, axes = plt.subplots(3, n, figsize=(2.1 * n, 6.4))
    if n == 1:
        axes = axes.reshape(3, 1)
    for j in range(n):
        for r, (img, lab) in enumerate([(gt[j, 0], "GT"), (xb[j, 0], "base"), (rec[j, 0], "base+SwinIR")]):
            axes[r, j].imshow(img, cmap="viridis", vmin=0, vmax=1)
            axes[r, j].axis("off")
            if j == 0:
                axes[r, j].set_ylabel(lab, rotation=90, labelpad=8, fontsize=11)
                axes[r, j].axis("on"); axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
    fig.suptitle("val: GT / frozen base / base+SwinIR (viridis [0,1])")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def train_cell(cfg: dict, comp: str, pattern: str, device: torch.device, out_root: Path,
               *, iterations_override: int | None = None, resume: bool = False) -> dict:
    tr = cfg["train"]
    seed = int(cfg.get("seed", 42))
    base_root = ROOT / cfg["base_exp_root"]
    iterations = int(iterations_override or tr["iterations"])
    micro = int(tr["micro_batch_size"]); accum = int(tr["grad_accum"])
    patch = int(tr["train_patch_size"])
    amp_dtype = _amp_dtype(tr.get("amp_dtype", "none"))
    eval_batch = int(cfg["eval"]["eval_batch"])

    cell_dir = out_root / comp / pattern
    ckpt_dir = cell_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{comp}/{pattern}] loading frozen base ...", flush=True)
    base_model, run_cfg, eval_m = S.load_frozen_base(base_root, comp, pattern, device)
    assert S.base_is_frozen(base_model), "base model must be fully frozen (requires_grad=False + eval)"

    print(f"[{comp}/{pattern}] building caches (frozen base forward) ...", flush=True)
    t_cache = time.time()
    train_cache = S.build_pair_cache(base_model, run_cfg, "train", device, eval_m,
                                     crop_size=256, crops_per_train_image=int(tr["cache_crops_per_train_image"]),
                                     base_batch=int(tr.get("base_batch", 8)), seed=seed)
    val_cache = S.build_pair_cache(base_model, run_cfg, "val", device, eval_m, crop_size=256,
                                   base_batch=int(tr.get("base_batch", 8)), seed=seed)
    test_cache = S.build_pair_cache(base_model, run_cfg, "test", device, eval_m, crop_size=256,
                                    base_batch=int(tr.get("base_batch", 8)), seed=seed)
    # free base weights ASAP (we only need the cached tensors now)
    del base_model
    torch.cuda.empty_cache()
    print(f"[{comp}/{pattern}] cache: train={len(train_cache)} val={len(val_cache)} test={len(test_cache)} "
          f"({time.time()-t_cache:.0f}s)", flush=True)

    # SwinIR refiner (direct mode = image-to-image, paper-faithful; identity init
    # so the untrained refiner == frozen base and can only improve it).
    swinir_cfg = S.swinir_cfg_from_stage(cfg)
    identity_init = bool(cfg["swinir"].get("identity_init", True))
    refiner = S.build_refiner(swinir_cfg, device, identity_init=identity_init, seed=seed)
    n_params = sum(p.numel() for p in refiner.swinir.parameters())
    print(f"[{comp}/{pattern}] SwinIR params={n_params/1e6:.2f}M embed_dim={swinir_cfg['embed_dim']} "
          f"depths={swinir_cfg['depths']} identity_init={identity_init}", flush=True)

    loss_spec = S.make_loss(cfg, device)
    disc: nn.Module | None = loss_spec.get("stack", {}).get("discriminator") if "stack" in loss_spec else None

    opt_g = torch.optim.Adam(refiner.parameters_for_training(), lr=float(tr["swinir_lr"]),
                             betas=tuple(tr.get("betas", [0.9, 0.99])))
    opt_d = torch.optim.Adam(disc.parameters(), lr=float(cfg["loss"].get("disc_lr", 2e-4)),
                             betas=tuple(tr.get("betas", [0.9, 0.99]))) if disc is not None else None

    # base val metrics for the MSE-no-regression gate
    base_val = S.eval_cache(refiner, val_cache, device, eval_batch)  # ref==random init irrelevant; we read base_*
    base_val_mse, base_val_ssim = base_val["base_mse"], base_val["base_ssim"]
    print(f"[{comp}/{pattern}] base val: ssim={base_val_ssim:.4f} mse={base_val_mse:.6f}", flush=True)

    gate = bool(cfg["selection"].get("mse_no_regression_gate", True))
    grad_clip = float(tr.get("grad_clip", 0.0))
    warmup = int(tr.get("warmup", 0)); min_lr_frac = float(tr.get("min_lr_frac", 0.02))
    base_lr = float(tr["swinir_lr"])
    val_every = int(tr.get("val_every", 1000)); ckpt_every = int(tr.get("ckpt_every", 5000))
    log_every = int(tr.get("log_every", 100))

    gen = torch.Generator().manual_seed(seed + 777)
    best_score = -float("inf"); best_val_ssim = -1.0; best_val_mse = float("inf"); best_step = -1
    start_step = 0
    history: list[dict] = []

    last_ckpt = ckpt_dir / "last.pt"
    if resume and last_ckpt.exists():
        st = torch.load(last_ckpt, map_location=device, weights_only=False)
        refiner.load_state_dict(st["refiner"])
        opt_g.load_state_dict(st["opt_g"])
        if opt_d is not None and st.get("opt_d") is not None:
            opt_d.load_state_dict(st["opt_d"])
        start_step = int(st.get("step", 0)); best_score = float(st.get("best_score", -1e9))
        best_val_ssim = float(st.get("best_val_ssim", -1.0)); best_val_mse = float(st.get("best_val_mse", 1e9))
        best_step = int(st.get("best_step", -1)); history = st.get("history", [])
        print(f"[{comp}/{pattern}] resumed @ step {start_step}", flush=True)

    def save_ckpt(path: Path, step: int, best: bool = False):
        payload = {"step": step, "swinir_cfg": swinir_cfg, "comp": comp, "pattern": pattern,
                   "loss_mode": cfg["loss"]["mode"], "best_score": best_score,
                   "best_val_ssim": best_val_ssim, "best_val_mse": best_val_mse, "best_step": best_step}
        if best:
            payload["refiner_state_dict"] = {k: v.detach().cpu().clone() for k, v in refiner.state_dict().items()}
        else:
            payload["refiner"] = refiner.state_dict()
            payload["opt_g"] = opt_g.state_dict()
            payload["opt_d"] = opt_d.state_dict() if opt_d is not None else None
            payload["history"] = history
        torch.save(payload, path)

    t0 = time.time(); step = start_step
    refiner.swinir.train()
    while step < iterations:
        cur_lr = S.lr_at(step, base_lr=base_lr, steps=iterations, warmup=warmup, min_frac=min_lr_frac)
        for g in opt_g.param_groups:
            g["lr"] = cur_lr
        if opt_d is not None:
            opt_d.zero_grad(set_to_none=True)
        opt_g.zero_grad(set_to_none=True)
        comps: dict[str, float] = {}
        for _ in range(accum):
            x, gt = S.sample_patch_batch(train_cache, micro, patch, device, gen,
                                         random_flips=bool(tr.get("random_flips", True)))
            with _autocast(device, amp_dtype):
                rec = refiner(x)
            rec_f = rec.float()
            if disc is not None:
                stack = loss_spec["stack"]
                d_loss = (stack["gan_loss"](disc(gt), True) + stack["gan_loss"](disc(rec_f.detach()), False)) / 2.0 / accum
                d_loss.backward()
                comps["d"] = comps.get("d", 0.0) + float(d_loss.item())
                for p in disc.parameters():
                    p.requires_grad_(False)
            g_loss, gc = S.compute_generator_loss(loss_spec, rec_f, gt, disc)
            (g_loss / accum).backward()
            if disc is not None:
                for p in disc.parameters():
                    p.requires_grad_(True)
            for k, v in gc.items():
                comps[k] = comps.get(k, 0.0) + v / accum
            comps["g_total"] = comps.get("g_total", 0.0) + float(g_loss.item()) / accum
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(refiner.parameters_for_training(), grad_clip)
        if opt_d is not None:
            opt_d.step()
        opt_g.step()
        step += 1

        if step % log_every == 0 or step == iterations:
            rate = (step - start_step) / max(1e-9, time.time() - t0)
            print(f"[{comp}/{pattern}] step {step}/{iterations} lr={cur_lr:.2e} "
                  + " ".join(f"{k}={v:.4f}" for k, v in comps.items())
                  + f" {rate:.2f} it/s", flush=True)

        if step % val_every == 0 or step == iterations:
            vm = S.eval_cache(refiner, val_cache, device, eval_batch)
            refiner.swinir.train()
            improves_ssim = vm["ref_ssim"] > base_val_ssim
            no_mse_regress = vm["ref_mse"] <= base_val_mse
            score = vm["ref_ssim"] if (not gate or no_mse_regress) else vm["ref_ssim"] - 1.0
            is_best = score > best_score
            if is_best:
                best_score = score; best_val_ssim = vm["ref_ssim"]; best_val_mse = vm["ref_mse"]; best_step = step
                save_ckpt(ckpt_dir / "best.pt", step, best=True)
            history.append({"step": step, **{f"val_{k}": vm[k] for k in ("ref_ssim", "ref_mse", "base_ssim", "base_mse")}})
            print(f"[{comp}/{pattern}] step {step} VAL ref_ssim={vm['ref_ssim']:.4f} (base {vm['base_ssim']:.4f}) "
                  f"ref_mse={vm['ref_mse']:.6f} (base {vm['base_mse']:.6f}) "
                  f"best_ssim={best_val_ssim:.4f}@{best_step}{' *' if is_best else ''}", flush=True)

        if step % ckpt_every == 0 or step == iterations:
            save_ckpt(last_ckpt, step)

    # ensure best exists
    if not (ckpt_dir / "best.pt").exists():
        save_ckpt(ckpt_dir / "best.pt", step, best=True)
        best_step = step

    # load best and test
    best = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    refiner.load_state_dict(best["refiner_state_dict"])
    tm = S.eval_cache(refiner, test_cache, device, eval_batch)
    save_val_grid(refiner, val_cache, device, cell_dir / "grids" / "val_grid.png", n=6)
    save_val_grid(refiner, test_cache, device, cell_dir / "grids" / "test_grid.png", n=6)

    metrics = {
        "comp": comp, "pattern": pattern, "loss_mode": cfg["loss"]["mode"],
        "config_name": cfg["name"], "iterations": iterations, "iterations_reached": step,
        "effective_batch": int(tr["effective_batch_size"]), "train_patch_size": patch,
        "swinir_params_M": round(n_params / 1e6, 3), "swinir_embed_dim": swinir_cfg["embed_dim"],
        "swinir_depths": swinir_cfg["depths"],
        "selection": f"max_val_ssim{'+mse_gate' if gate else ''}",
        "best_step": best_step, "best_val_ssim": best_val_ssim, "best_val_mse": best_val_mse,
        "base_val_ssim": base_val_ssim, "base_val_mse": base_val_mse,
        "test": tm,
        "delta_test_ssim": tm["ref_ssim"] - tm["base_ssim"],
        "delta_test_mse": tm["ref_mse"] - tm["base_mse"],
        "checkpoint_path": str((ckpt_dir / "best.pt").resolve()),
        "minutes": (time.time() - t0) / 60.0,
    }
    (cell_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (cell_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[{comp}/{pattern}] DONE test base->ref  SSIM {tm['base_ssim']:.4f}->{tm['ref_ssim']:.4f} "
          f"(+{metrics['delta_test_ssim']:+.4f}) | MSE {tm['base_mse']:.6f}->{tm['ref_mse']:.6f} "
          f"({metrics['delta_test_mse']:+.6f}) | {metrics['minutes']:.1f} min", flush=True)
    del refiner
    torch.cuda.empty_cache()
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", default=None, help="default: experiments/figure03_content_aware/swinir/<config-name>")
    ap.add_argument("--comps", default=",".join(S.ALL_COMPS))
    ap.add_argument("--patterns", default="random_fixed,learnable_frequency")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = S.load_stage_config(args.config)
    out_root = Path(args.out_root) if args.out_root else \
        ROOT / "experiments/figure03_content_aware/swinir" / cfg["name"]
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    comps = [c.strip() for c in args.comps.split(",") if c.strip()]
    patterns = [ILLUM_ALIAS[p.strip()] for p in args.patterns.split(",") if p.strip()]

    for comp in comps:
        for pattern in patterns:
            train_cell(cfg, comp, pattern, device, out_root,
                       iterations_override=args.iterations, resume=args.resume)


if __name__ == "__main__":
    main()
