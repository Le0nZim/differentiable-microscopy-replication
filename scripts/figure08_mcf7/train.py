#!/usr/bin/env python3
"""MCF7 Figure 8 & 9 — paper-faithful SwinIR high-resolution reconstruction (fix).

Ports the VALIDATED Table-2/Fig-7 SwinIR recipe (full pixel+perceptual+adversarial loss,
SwinIR-M capacity) onto the MCF7 end-to-end pipeline with the Algorithm-1 sharpness
schedule, fixing the "SwinIR resolution deviates" defect of the frozen mcf7_fig8_qr run.

Conditions (paper §5.6, Fig 8/9):
  wswinir      Q / wSwinIR : locality-aware upsampling + SwinIR(upscale=1), 256x256,
                            FULL loss (pixel + perceptual + adversarial). learnable Ht.
  transpose256 R (Fig 8)   : transpose-conv upsampling + conventional ReconCNN, 256x256,
                            L1 (conventional pipeline). learnable Ht.
  wcnn64       wCNN (Fig 9): locality-aware upsampling + conventional ReconCNN, 64x64,
                            L1. learnable Ht.

Isolated writes to experiments/figure08_mcf7/runs/<condition>/.
Does NOT touch any frozen run.

Usage:
  python scripts/figure08_mcf7/train.py --condition wswinir --device cuda:0 \
      --epochs 60 --epoch-baseline 39 --epoch-step 5 --max-steps-per-epoch 150
  python scripts/figure08_mcf7/train.py --condition transpose256 --device cuda:1 --epochs 60 ...
  python scripts/figure08_mcf7/train.py --condition wcnn64 --device cuda:1 --epochs 72 ...
  python scripts/figure08_mcf7/train.py --smoke --condition wswinir --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.swinir.losses import build_loss_stack, pixel_loss
from baselines.swinir.table2_pipeline import SwinIRTable2Model
from datasets.mcf7_channel2 import MCF7Channel2Dataset
from evaluation.metrics import mse as mse_metric
from evaluation.metrics import psnr as psnr_metric
from evaluation.metrics import ssim as ssim_metric
from models.microscope import DifferentiableMicroscope
from utils.device import resolve_device

EXP = ROOT / "experiments/figure08_mcf7"
CFG = ROOT / "configs/figure08_mcf7/swinir_fix.yaml"

# condition -> (backbone, image_size, upsampling_mode, uses_full_loss)
CONDITIONS = {
    "wswinir": ("swinir", 256, "locality_aware", True),
    "transpose256": ("conventional", 256, "transpose_conv", False),
    "wcnn64": ("conventional", 64, "locality_aware", False),
}


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _build_swinir_model(cfg: dict, image_size: int) -> SwinIRTable2Model:
    pg = cfg["pattern_generator"]
    fm = cfg["forward_model"]
    swin = dict(cfg["swinir"])
    swin["img_size"] = image_size
    m_cfg = {
        "image_size": image_size,
        "pattern_generator": {
            "mode": pg["mode"],
            "num_patterns": int(pg["num_patterns"]),
            "sigmoid_m": 1.0,
            "random_fixed_m": float(pg.get("random_fixed_m", 10.0)),
            "seed": int(pg.get("seed", 42)),
            "superpixel_factor": int(pg.get("superpixel_factor", 1)),
        },
        "forward_model": {
            "downscale_factor": int(fm["downscale_factor"]),
            "use_impulse_psfs": bool(fm.get("use_impulse_psfs", True)),
        },
        "detector_noise": {"mode": "noise_free", "apply_noise": False},
        "inverse_model": {
            "upsampling": {
                "mode": "locality_aware",
                "downscale_factor": int(fm["downscale_factor"]),
                "num_patterns": int(pg["num_patterns"]),
            }
        },
        "swinir": swin,
    }
    return SwinIRTable2Model(m_cfg)


def _build_conventional_model(cfg: dict, image_size: int, up_mode: str) -> DifferentiableMicroscope:
    pg = cfg["pattern_generator"]
    fm = cfg["forward_model"]
    npat = int(pg["num_patterns"])
    down = int(fm["downscale_factor"])
    run_cfg = {
        "dataset": {"image_size": image_size},
        "pattern_generator": {
            "mode": pg["mode"],
            "num_patterns": npat,
            "sigmoid_m": 1.0,
            "random_fixed_m": float(pg.get("random_fixed_m", 10.0)),
            "seed": int(pg.get("seed", 42)),
            "superpixel_factor": int(pg.get("superpixel_factor", 1)),
        },
        "forward_model": {
            "downscale_factor": down,
            "use_impulse_psfs": bool(fm.get("use_impulse_psfs", True)),
        },
        "detector_noise": {"mode": "noise_free", "apply_noise": False},
        "inverse_model": {
            "upsampling": {"mode": up_mode, "downscale_factor": down, "num_patterns": npat},
            "reconstruction": {
                "in_channels": npat,
                "hidden_channels": [64, 64, 32, 32, 16, 1],
                "kernel_size": 3,
                "padding": 1,
            },
        },
    }
    return DifferentiableMicroscope.from_run_config(run_cfg)


class Adapter:
    def __init__(self, backbone: str, model: torch.nn.Module):
        self.backbone = backbone
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
def _loaders(cfg: dict, image_size: int, micro: int, seed: int,
             n_train: int, n_val: int, n_test: int) -> dict[str, DataLoader]:
    ds_cfg = dict(cfg["dataset"])
    ds_cfg["seed"] = seed
    ds_cfg["patch_size"] = image_size
    ds_cfg["image_size"] = image_size
    ds_cfg["num_train"] = n_train
    ds_cfg["num_val"] = n_val
    ds_cfg["num_test"] = n_test
    out = {}
    print(f"Building MCF7 loaders @ {image_size}px (loads TIFFs; a few minutes)...", flush=True)
    for split, shuffle in (("train", True), ("val", False), ("test", False)):
        ds = MCF7Channel2Dataset.from_dict(ds_cfg, split=split)
        bs = micro if split == "train" else 1
        out[split] = DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=(split == "train"))
        print(f"  -> {split}: {len(ds)} patches", flush=True)
    return out


def _schedule_m(epoch: int, baseline: int, m_values: list[float], step: int) -> tuple[float, bool]:
    if epoch < baseline:
        return 1.0, False
    idx = min((epoch - baseline) // max(1, step), len(m_values) - 1)
    return float(m_values[idx]), True


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _autocast(device, amp_dtype):
    if amp_dtype is not None and device.type == "cuda":
        return torch.autocast(device.type, dtype=amp_dtype)
    return _null()


@torch.no_grad()
def _evaluate(adapter: Adapter, loader: DataLoader, device, m: float,
              amp_dtype, max_items: int | None = None) -> dict:
    adapter.model.eval()
    mse_s = ssim_s = psnr_s = 0.0
    n = 0
    for batch in loader:
        x = batch.to(device)
        with _autocast(device, amp_dtype):
            rec = adapter.forward(x, m)
        rec = rec.float().clamp(0, 1)
        mse_s += float(mse_metric(rec, x).item())
        ssim_s += float(ssim_metric(rec, x).item())
        psnr_s += float(psnr_metric(rec, x).item())
        n += 1
        if max_items is not None and n >= max_items:
            break
    adapter.model.train()
    return {"mse": mse_s / max(1, n), "ssim": ssim_s / max(1, n), "psnr": psnr_s / max(1, n)}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(cfg: dict, condition: str, device, *, epochs: int, baseline: int, step: int,
          max_steps_per_epoch: int | None, seed: int, out_dir: Path,
          n_train: int, n_val: int, n_test: int, val_subset: int | None,
          n_examples: int, gan_warmup_epochs: int | None = None) -> dict:
    backbone, image_size, up_mode, full_loss = CONDITIONS[condition]
    torch.manual_seed(seed)
    tr = cfg["training"]
    channel = str(cfg["dataset"].get("bbbc021_channel", "tubulin"))
    print(f"[{condition}] BBBC021 channel = {channel} | full_loss={full_loss} | "
          f"image_size={image_size} | GAN warmup epochs={gan_warmup_epochs}", flush=True)
    micro = int(tr["micro_batch_size"]) if full_loss else int(tr.get("conv_batch_size", 8))
    accum = int(tr["grad_accum"]) if full_loss else 1
    amp_dtype = torch.bfloat16 if str(tr.get("amp_dtype", "bfloat16")) == "bfloat16" else None
    m_values = [float(v) for v in cfg["algorithm1"]["m_values"]]
    eval_m = float(tr.get("eval_sigmoid_m", 8.0))

    if backbone == "swinir":
        model = _build_swinir_model(cfg, image_size).to(device)
    else:
        model = _build_conventional_model(cfg, image_size, up_mode).to(device)
    model(torch.zeros(1, 1, image_size, image_size, device=device))  # lazy-init buffers
    adapter = Adapter(backbone, model)

    betas = tuple(float(b) for b in tr.get("betas", [0.9, 0.99]))
    recon_lr = float(tr["swinir_lr"]) if backbone == "swinir" else float(tr.get("inverse_lr", 1e-3))
    opt_g = torch.optim.Adam(
        [{"params": adapter.recon_params(), "lr": recon_lr},
         {"params": adapter.illum_params(), "lr": float(tr["illumination_lr"])}],
        betas=betas,
    )

    stack = disc = opt_d = None
    if full_loss:
        loss_cfg = dict(tr["loss"])
        stack = build_loss_stack(loss_cfg, device)
        disc = stack.get("discriminator")
        if disc is not None:
            opt_d = torch.optim.Adam(disc.parameters(), lr=float(tr.get("disc_lr", 2e-4)), betas=betas)

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    loaders = _loaders(cfg, image_size, micro, seed, n_train, n_val, n_test)
    log_path = out_dir / "run.log"

    history: list[dict] = []
    best_val = float("inf")       # best val MSE (for L1 baselines / MSE gate)
    best_score = -float("inf")    # best val SSIM (for the perceptual+GAN SwinIR)
    best_state = None
    best_meta = {}
    t0 = time.time()
    finite = True

    def _log(msg: str) -> None:
        with log_path.open("a", encoding="utf-8") as h:
            h.write(msg + "\n")
        print(msg, flush=True)

    # GAN warmup: train pixel+perceptual only for the first `gan_warmup_epochs`, then
    # introduce the adversarial term. Adversarial loss applied from step 0 destabilizes
    # SwinIR training (the frozen run's val SSIM oscillated 0.29-0.69); a warmup lets the
    # generator reach a sensible conditional mean before the discriminator engages.
    if gan_warmup_epochs is None:
        gan_warmup_epochs = min(baseline, max(2, epochs // 4)) if full_loss else 0

    for epoch in range(epochs):
        ep_t0 = time.time()
        m, unfreeze = _schedule_m(epoch, baseline, m_values, step)
        for p in adapter.illum_params():
            p.requires_grad = unfreeze
        model.train()
        use_gan = full_loss and (opt_d is not None) and (epoch >= gan_warmup_epochs)
        comp = {"pixel": 0.0, "perceptual": 0.0, "adv": 0.0, "d": 0.0, "g_total": 0.0}
        opt_steps = 0
        it = iter(loaders["train"])
        exhausted = False
        while not exhausted:
            if max_steps_per_epoch is not None and opt_steps >= max_steps_per_epoch:
                break
            if use_gan:
                opt_d.zero_grad(set_to_none=True)
            opt_g.zero_grad(set_to_none=True)
            micro_done = 0
            for _ in range(accum):
                try:
                    x = next(it).to(device)
                except StopIteration:
                    exhausted = True
                    break
                with _autocast(device, amp_dtype):
                    rec = adapter.forward(x, m)
                    if full_loss:
                        if use_gan:
                            d_loss = (stack["gan_loss"](disc(x), True)
                                      + stack["gan_loss"](disc(rec.detach()), False)) / accum
                        g_pix = stack["pixel_weight"] * pixel_loss(rec, x, stack["pixel_kind"])
                        g_loss = g_pix
                        comp["pixel"] += float(g_pix.item()) / accum
                        if "perceptual" in stack:
                            g_perc = stack["perceptual_weight"] * stack["perceptual"](rec, x)
                            g_loss = g_loss + g_perc
                            comp["perceptual"] += float(g_perc.item()) / accum
                        if use_gan:
                            g_adv = stack["gan_weight"] * stack["gan_loss"](disc(rec), True)
                            g_loss = g_loss + g_adv
                            comp["adv"] += float(g_adv.item()) / accum
                        g_loss = g_loss / accum
                    else:
                        g_loss = F.l1_loss(rec, x) / accum
                        comp["pixel"] += float(g_loss.item())
                if use_gan:
                    d_loss.backward()
                    comp["d"] += float(d_loss.item())
                    for p in disc.parameters():
                        p.requires_grad_(False)
                g_loss.backward()
                if use_gan:
                    for p in disc.parameters():
                        p.requires_grad_(True)
                comp["g_total"] += float(g_loss.item())
                if not torch.isfinite(g_loss).item():
                    finite = False
                micro_done += 1
            if micro_done == 0:
                break
            if use_gan:
                opt_d.step()
            opt_g.step()
            opt_steps += 1

        val = _evaluate(adapter, loaders["val"], device, eval_m, amp_dtype, max_items=val_subset)
        ep_sec = time.time() - ep_t0
        rec_line = {"epoch": epoch, "m": m, "illum_unfrozen": unfreeze, "opt_steps": opt_steps,
                    **{k: round(v, 6) for k, v in comp.items()},
                    "val_mse": val["mse"], "val_ssim": val["ssim"], "val_psnr": val["psnr"],
                    "epoch_sec": round(ep_sec, 1)}
        history.append(rec_line)
        _log(f"[{condition}] ep {epoch}/{epochs} m={m} illum={unfreeze} steps={opt_steps} "
             f"g={comp['g_total']:.4f} pix={comp['pixel']:.4f} perc={comp['perceptual']:.4f} "
             f"adv={comp['adv']:.4f} d={comp['d']:.4f} val_ssim={val['ssim']:.4f} "
             f"val_psnr={val['psnr']:.2f} ({ep_sec:.0f}s)")
        # Checkpoint selection.
        #   * Perceptual + adversarial SwinIR (full_loss): select by MAX val SSIM. Selecting
        #     by MIN val MSE picks the blurry conditional-mean epoch (the exact defect that
        #     made the frozen Q "over-smoothed"). GAN/perceptual deliberately trades pixel
        #     MSE for high-frequency realism, so MSE is the wrong selection metric here.
        #   * L1 baselines (R / wCNN): keep MIN val MSE (== MAX SSIM for L1).
        improved = (val["ssim"] > best_score) if full_loss else (val["mse"] < best_val)
        if improved:
            best_val = min(best_val, val["mse"])
            best_score = max(best_score, val["ssim"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_meta = {"epoch": epoch, "val_mse": val["mse"], "val_ssim": val["ssim"],
                         "selection": "max_val_ssim" if full_loss else "min_val_mse"}
            torch.save({"model": best_state, **best_meta,
                        "condition": condition, "backbone": backbone, "image_size": image_size},
                       ckpt_dir / "best.pt")

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model": model.state_dict(), "condition": condition, "backbone": backbone,
                "image_size": image_size}, ckpt_dir / "last.pt")

    test = _evaluate(adapter, loaders["test"], device, eval_m, amp_dtype)
    _save_examples(adapter, loaders["test"], device, eval_m, amp_dtype, out_dir / "examples", n_examples)
    with torch.no_grad():
        patterns = model.pattern_generator(sigmoid_m=eval_m).detach().float().cpu()
    (out_dir / "illumination").mkdir(parents=True, exist_ok=True)
    torch.save(patterns, out_dir / "illumination" / "patterns.pt")

    result = {
        "condition": condition, "backbone": backbone, "image_size": image_size,
        "full_loss": full_loss, "epochs": epochs, "baseline": baseline, "step": step,
        "micro_batch": micro, "grad_accum": accum, "effective_batch": micro * accum,
        "max_steps_per_epoch": max_steps_per_epoch,
        "test_mse": test["mse"], "test_ssim": test["ssim"], "test_psnr": test["psnr"],
        "best_val_mse": best_val, "checkpoint_selection": best_meta,
        "gan_warmup_epochs": gan_warmup_epochs, "loss_finite": finite,
        "wall_seconds": round(time.time() - t0, 1),
        "checkpoint": str(ckpt_dir / "best.pt"),
        "swinir_embed_dim": int(cfg["swinir"]["embed_dim"]) if backbone == "swinir" else None,
        "pattern_superpixel": int(cfg["pattern_generator"].get("superpixel_factor", 1)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "history": history,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _log(f"[{condition}] DONE test PSNR={test['psnr']:.2f} SSIM={test['ssim']:.4f} "
         f"MSE={test['mse']:.6f} finite={finite}")
    return result


@torch.no_grad()
def _save_examples(adapter: Adapter, loader: DataLoader, device, m: float, amp_dtype,
                   out_dir: Path, n_examples: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter.model.eval()
    saved = 0
    for batch in loader:
        x = batch.to(device)
        with _autocast(device, amp_dtype):
            rec = adapter.forward(x, m)
        rec = rec.float().clamp(0, 1)
        for j in range(x.shape[0]):
            if saved >= n_examples:
                return
            torch.save({"gt": x[j].cpu(), "recon": rec[j].cpu()}, out_dir / f"pair_{saved:02d}.pt")
            saved += 1


def _smoke(cfg: dict, condition: str, device) -> None:
    """One-optimizer-step memory/speed probe for the given condition."""
    backbone, image_size, up_mode, full_loss = CONDITIONS[condition]
    tr = cfg["training"]
    micro = int(tr["micro_batch_size"]) if full_loss else 8
    accum = int(tr["grad_accum"]) if full_loss else 1
    amp_dtype = torch.bfloat16 if str(tr.get("amp_dtype", "bfloat16")) == "bfloat16" else None
    if backbone == "swinir":
        model = _build_swinir_model(cfg, image_size).to(device)
    else:
        model = _build_conventional_model(cfg, image_size, up_mode).to(device)
    model(torch.zeros(1, 1, image_size, image_size, device=device))
    adapter = Adapter(backbone, model)
    opt_g = torch.optim.Adam(
        [{"params": adapter.recon_params(), "lr": 2e-4},
         {"params": adapter.illum_params(), "lr": 0.1}])
    stack = disc = opt_d = None
    if full_loss:
        stack = build_loss_stack(dict(tr["loss"]), device)
        disc = stack.get("discriminator")
        opt_d = torch.optim.Adam(disc.parameters(), lr=2e-4) if disc is not None else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    for _ in range(accum):
        x = torch.rand(micro, 1, image_size, image_size, device=device)
        with _autocast(device, amp_dtype):
            rec = adapter.forward(x, 8.0)
            if full_loss:
                d_loss = (stack["gan_loss"](disc(x), True) + stack["gan_loss"](disc(rec.detach()), False)) / accum
                g = (stack["pixel_weight"] * pixel_loss(rec, x, "l1")
                     + stack["perceptual_weight"] * stack["perceptual"](rec, x)
                     + stack["gan_weight"] * stack["gan_loss"](disc(rec), True)) / accum
            else:
                g = F.l1_loss(rec, x)
        if opt_d is not None:
            d_loss.backward()
        g.backward()
    if opt_d is not None:
        opt_d.step()
    opt_g.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    n_recon = sum(p.numel() for p in adapter.recon_params())
    print(f"SMOKE [{condition}] image={image_size} micro={micro} accum={accum} full_loss={full_loss} "
          f"embed={cfg['swinir']['embed_dim'] if backbone=='swinir' else '-'} "
          f"recon_params={n_recon/1e6:.1f}M peak_mem={peak:.2f}GB "
          f"opt_step_time={dt:.2f}s (~{dt/max(1,accum):.2f}s/micro)", flush=True)
    print("SMOKE_OK", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=list(CONDITIONS), required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--epoch-baseline", type=int, default=39)
    ap.add_argument("--epoch-step", type=int, default=5)
    ap.add_argument("--gan-warmup-epochs", type=int, default=None,
                    help="pixel+perceptual-only warmup before the adversarial term engages "
                         "(default: min(baseline, epochs//4) for full-loss SwinIR)")
    ap.add_argument("--max-steps-per-epoch", type=int, default=None)
    ap.add_argument("--full-budget", action="store_true", help="paper 230/150/20 (no scaling)")
    ap.add_argument("--num-train", type=int, default=None)
    ap.add_argument("--num-val", type=int, default=None)
    ap.add_argument("--num-test", type=int, default=None)
    ap.add_argument("--val-subset", type=int, default=None)
    ap.add_argument("--n-examples", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    cfg = _load_yaml(CFG)
    device = resolve_device(args.device)
    print(f"Device: {device} | condition: {args.condition}", flush=True)

    if args.smoke:
        _smoke(cfg, args.condition, device)
        return

    if args.full_budget:
        epochs, baseline, step = 230, 150, 20
    else:
        epochs, baseline, step = args.epochs, args.epoch_baseline, args.epoch_step

    n_train = args.num_train if args.num_train is not None else int(cfg["dataset"]["num_train"])
    n_val = args.num_val if args.num_val is not None else int(cfg["dataset"]["num_val"])
    n_test = args.num_test if args.num_test is not None else int(cfg["dataset"]["num_test"])

    out_dir = Path(args.out) if args.out else (EXP / "runs" / args.condition)
    out_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    (out_dir / "configs_used").mkdir(exist_ok=True)
    shutil.copy2(CFG, out_dir / "configs_used" / CFG.name)

    train(cfg, args.condition, device, epochs=epochs, baseline=baseline, step=step,
          max_steps_per_epoch=args.max_steps_per_epoch, seed=args.seed, out_dir=out_dir,
          n_train=n_train, n_val=n_val, n_test=n_test, val_subset=args.val_subset,
          n_examples=args.n_examples, gan_warmup_epochs=args.gan_warmup_epochs)


if __name__ == "__main__":
    main()
