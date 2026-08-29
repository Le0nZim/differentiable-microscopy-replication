#!/usr/bin/env python3
"""Table 2 full paper-faithful attempt (Sec 5.6, Table 2, Fig 7).

SwinIR replaces psi with the compressive LI forward model (x16) + locality upsampling.
Trained end-to-end with pixel + perceptual (VGG19) + adversarial losses, for both
conditions: SwinIR without LI and SwinIR with LI. Evaluated (PSNR/SSIM) on Set5, Set14,
BSD100, Urban100, Manga109 via 64x64 grid tiling (LocalityUpsampling is fixed-size).

PAPER_UNSPECIFIED_FALLBACK: discriminator arch, VGG layers, loss weights, iteration count,
optimizer betas (Sec 5.6 only says "similar to SwinIR [26]"). Logged in deviations.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.swinir.dataset_adapter import SRImageFolderDataset
from baselines.swinir.losses import build_loss_stack, pixel_loss
from baselines.swinir.table2_pipeline import SwinIRTable2Model
from evaluation.metrics import psnr as psnr_metric
from evaluation.metrics import ssim as ssim_metric
from torchvision.io import read_image
from utils.device import resolve_device
from utils.logging import save_measurement_grid

OUT = ROOT / "experiments/swinir_or_highres/swinir_table2_full"
CFG = ROOT / "configs/swinir/table2_paper_full.yaml"


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _model_config(cfg: dict, learnable: bool) -> dict[str, Any]:
    m = cfg["model"]
    return {
        "image_size": int(m["image_size"]),
        "pattern_generator": {
            "mode": "learnable_frequency" if learnable else "random_fixed",
            "num_patterns": int(m["num_patterns"]),
            "sigmoid_m": 1.0,
            "random_fixed_m": 10.0,
            "seed": int(cfg["experiment"]["seed"]),
        },
        "forward_model": {"downscale_factor": int(m["downscale_factor"]), "use_impulse_psfs": True},
        "detector_noise": {"mode": "noise_free", "apply_noise": False},
        "inverse_model": {
            "upsampling": {
                "mode": "locality_aware",
                "downscale_factor": int(m["downscale_factor"]),
                "num_patterns": int(m["num_patterns"]),
            }
        },
        "swinir": dict(m["swinir"]),
    }


@torch.no_grad()
def _eval_dataset(model: SwinIRTable2Model, root: Path, ps: int, device: torch.device, m: float, learnable: bool, max_tiles_per_image: int = 9) -> dict:
    model.eval()
    paths = sorted(root.glob("*.png")) + sorted(root.glob("*.jpg"))
    psnr_s = ssim_s = 0.0
    n_tiles = 0
    for p in paths:
        img = read_image(str(p)).float() / 255.0
        if img.shape[0] == 3:
            img = img.mean(dim=0, keepdim=True)
        _, h, w = img.shape
        nh, nw = h // ps, w // ps
        if nh == 0 or nw == 0:
            continue
        img = img[:, : nh * ps, : nw * ps]
        # Center-biased grid sample, capped per image to bound eval cost.
        coords = [(ti, tj) for ti in range(nh) for tj in range(nw)]
        if len(coords) > max_tiles_per_image:
            ci, cj = nh / 2.0, nw / 2.0
            coords.sort(key=lambda c: (c[0] - ci) ** 2 + (c[1] - cj) ** 2)
            coords = coords[:max_tiles_per_image]
        tiles = [img[:, ti * ps : (ti + 1) * ps, tj * ps : (tj + 1) * ps] for ti, tj in coords]
        batch = torch.stack(tiles).to(device)
        for i in range(0, batch.shape[0], 16):
            chunk = batch[i : i + 16]
            out = model(chunk, sigmoid_m=m if learnable else None, apply_noise=False)
            rec = out["x_recon"].clamp(0, 1)
            for j in range(chunk.shape[0]):
                psnr_s += float(psnr_metric(rec[j : j + 1], chunk[j : j + 1]).item())
                ssim_s += float(ssim_metric(rec[j : j + 1], chunk[j : j + 1]).item())
                n_tiles += 1
    return {"psnr": psnr_s / max(1, n_tiles), "ssim": ssim_s / max(1, n_tiles), "tiles": n_tiles}


def _save_example(model: SwinIRTable2Model, root: Path, ps: int, device: torch.device, m: float, learnable: bool, out_path: Path) -> None:
    paths = sorted(root.glob("*.png")) + sorted(root.glob("*.jpg"))
    if not paths:
        return
    img = read_image(str(paths[0])).float() / 255.0
    if img.shape[0] == 3:
        img = img.mean(dim=0, keepdim=True)
    img = img[:, :ps, :ps].unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(img, sigmoid_m=m if learnable else None, apply_noise=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_measurement_grid(img, out_path.with_name(out_path.stem + "_gt.png"))
    save_measurement_grid(out["x_recon"].clamp(0, 1), out_path)


def _train_condition(name: str, learnable: bool, cfg: dict, train_loader: DataLoader, device: torch.device, iterations: int, log_path: Path) -> dict:
    seed = int(cfg["experiment"]["seed"])
    torch.manual_seed(seed)
    model = SwinIRTable2Model(_model_config(cfg, learnable)).to(device)
    ps = int(cfg["data"]["patch_size"])
    model(torch.zeros(1, 1, ps, ps, device=device))

    tr = cfg["training"]
    loss_cfg = dict(tr["loss"])
    loss_cfg["in_chans"] = 1
    stack = build_loss_stack(loss_cfg, device)

    g_params = [{"params": model.swinir_parameters(), "lr": float(tr["swinir_lr"])}]
    if learnable:
        g_params.append({"params": model.illumination_parameters(), "lr": float(tr["illumination_lr"])})
    opt_g = torch.optim.Adam(g_params, betas=(0.9, 0.99))
    opt_d = None
    if "discriminator" in stack:
        opt_d = torch.optim.Adam(stack["discriminator"].parameters(), lr=float(tr["disc_lr"]), betas=(0.9, 0.99))

    ckpt_dir = OUT / name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history = []
    finite = True
    step = 0
    t0 = time.time()
    log_every = int(tr["log_every"])
    model.train()
    while step < iterations:
        for batch in train_loader:
            if step >= iterations:
                break
            gt = batch.to(device)
            out = model(gt, sigmoid_m=8.0 if learnable else None, apply_noise=False)
            rec = out["x_recon"]

            if opt_d is not None:
                opt_d.zero_grad(set_to_none=True)
                d_real = stack["discriminator"](gt)
                d_fake = stack["discriminator"](rec.detach())
                d_loss = stack["gan_loss"](d_real, True) + stack["gan_loss"](d_fake, False)
                d_loss.backward()
                opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            g_pix = stack["pixel_weight"] * pixel_loss(rec, gt, stack["pixel_kind"])
            g_loss = g_pix
            comp = {"pixel": float(g_pix.item())}
            if "perceptual" in stack:
                g_perc = stack["perceptual_weight"] * stack["perceptual"](rec, gt)
                g_loss = g_loss + g_perc
                comp["perceptual"] = float(g_perc.item())
            if opt_d is not None:
                g_adv = stack["gan_weight"] * stack["gan_loss"](stack["discriminator"](rec), True)
                g_loss = g_loss + g_adv
                comp["adv"] = float(g_adv.item())
            g_loss.backward()
            opt_g.step()

            if not torch.isfinite(g_loss).item():
                finite = False
            if step % log_every == 0:
                rate = (step + 1) / (time.time() - t0)
                comp["g_total"] = float(g_loss.item())
                comp["step"] = step
                comp["rate_it_s"] = round(rate, 2)
                history.append(comp)
                with log_path.open("a", encoding="utf-8") as h:
                    h.write(f"[{name}] step {step}/{iterations} {json.dumps(comp)}\n")
                print(f"[{name}] step {step}/{iterations} g={g_loss.item():.4f} {comp} {rate:.2f} it/s", flush=True)
            step += 1

    torch.save(model.state_dict(), ckpt_dir / "last.pt")

    eval_m = float(tr["eval_sigmoid_m"])
    per_dataset = {}
    for ds_name, rel in cfg["data"]["test_roots"].items():
        per_dataset[ds_name] = _eval_dataset(model, ROOT / rel, ps, device, eval_m, learnable)
        _save_example(model, ROOT / rel, ps, device, eval_m, learnable, OUT / name / "examples" / f"{ds_name}.png")
        print(f"[{name}] {ds_name}: PSNR {per_dataset[ds_name]['psnr']:.2f} SSIM {per_dataset[ds_name]['ssim']:.4f}", flush=True)

    return {
        "name": name,
        "learnable": learnable,
        "iterations": iterations,
        "loss_finite": finite,
        "per_dataset": per_dataset,
        "history_tail": history[-5:],
        "checkpoint": str(ckpt_dir / "last.pt"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="tiny run for tests")
    args = parser.parse_args()

    cfg = _load_yaml(CFG)
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    device = resolve_device(args.device or cfg["experiment"]["device"])
    iterations = args.iterations or int(cfg["training"]["iterations"])
    if args.smoke:
        iterations = 4

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "configs_used").mkdir(exist_ok=True)
    shutil.copy2(CFG, OUT / "configs_used/table2_paper_full.yaml")
    log_path = OUT / "run.log"
    (OUT / "status.md").write_text(
        f"# Table 2 full\n\n**Status:** RUNNING\n**Iterations:** {iterations}\n**Updated:** {datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )

    ps = int(cfg["data"]["patch_size"])
    seed = int(cfg["experiment"]["seed"])
    max_imgs = 40 if args.smoke else None
    train_parts = []
    for idx, rel in enumerate(cfg["data"]["train_roots"]):
        train_parts.append(
            SRImageFolderDataset(ROOT / rel, patch_size=ps, max_images=max_imgs, seed=seed + idx, grayscale=True, random_crops=True)
        )
    train_ds = ConcatDataset(train_parts)
    train_loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True, num_workers=4, drop_last=True)
    print(f"train images: {len(train_ds)}", flush=True)

    results = []
    for cond in cfg["training"]["conditions"]:
        print(f"\n=== Table2 full: {cond['name']} (learnable={cond['learnable']}) ===", flush=True)
        results.append(_train_condition(cond["name"], bool(cond["learnable"]), cfg, train_loader, device, iterations, log_path))

    wo = next(r for r in results if not r["learnable"])
    wi = next(r for r in results if r["learnable"])
    datasets = list(cfg["data"]["test_roots"].keys())
    comparison = {
        ds: {
            "wo_li_psnr": wo["per_dataset"][ds]["psnr"],
            "with_li_psnr": wi["per_dataset"][ds]["psnr"],
            "wo_li_ssim": wo["per_dataset"][ds]["ssim"],
            "with_li_ssim": wi["per_dataset"][ds]["ssim"],
            "li_improves_psnr": wi["per_dataset"][ds]["psnr"] > wo["per_dataset"][ds]["psnr"],
        }
        for ds in datasets
    }

    payload = {
        "label": "PAPER_ALIGNED_ATTEMPTED Table 2 (pixel+perceptual+adversarial, x16, SwinIR replaces psi)",
        "paper_table2": {
            "Set5": {"wo_li": [14.03, 0.3079], "with_li": [26.74, 0.8113]},
            "Set14": {"wo_li": [13.64, 0.2258], "with_li": [23.60, 0.6930]},
            "BSD100": {"wo_li": [14.28, 0.2094], "with_li": [22.90, 0.6317]},
            "Urban100": {"wo_li": [13.51, 0.2146], "with_li": [21.51, 0.6402]},
            "Manga109": {"wo_li": [12.09, 0.1952], "with_li": [20.18, 0.6652]},
        },
        "results": results,
        "comparison": comparison,
        "all_finite": all(r["loss_finite"] for r in results),
        "deviations": [
            "iterations PAPER_UNSPECIFIED -> bounded (SwinIR real-SR uses ~5e5)",
            "batch 32 -> 16 (memory + GAN/VGG); logged",
            "SwinIR embed_dim 96 PAPER_UNSPECIFIED_ARCH_FALLBACK (vendor 180)",
            "discriminator VGG-style SN PAPER_UNSPECIFIED_FALLBACK (paper 'similar to SwinIR')",
            "VGG19 perceptual layers/weights ESRGAN default PAPER_UNSPECIFIED_FALLBACK",
            "GAN/perceptual weights ESRGAN default PAPER_UNSPECIFIED_FALLBACK",
            "eval via 64x64 grid tiling (LocalityUpsampling fixed-size), not full-image",
            "eval capped to 9 center-biased tiles/image to bound cost; logged",
            "grayscale (microscopy 1-channel forward model), not RGB",
            "1x1 conv fuse of T upsampled channels before SwinIR (in_chans=1)",
        ],
    }
    (OUT / "aggregate_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (OUT / "results.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.writer(h)
        w.writerow(["dataset", "condition", "psnr", "ssim"])
        for r in results:
            for ds in datasets:
                w.writerow([ds, r["name"], f"{r['per_dataset'][ds]['psnr']:.4f}", f"{r['per_dataset'][ds]['ssim']:.4f}"])

    lines = ["# Table 2 full paper-faithful attempt\n",
             "**Label:** PAPER_ALIGNED_ATTEMPTED (pixel+perceptual+adversarial, x16).\n",
             "| Dataset | w/o LI PSNR | with LI PSNR | w/o LI SSIM | with LI SSIM | LI improves |",
             "|---|---|---|---|---|---|"]
    for ds in datasets:
        c = comparison[ds]
        lines.append(f"| {ds} | {c['wo_li_psnr']:.2f} | {c['with_li_psnr']:.2f} | {c['wo_li_ssim']:.4f} | {c['with_li_ssim']:.4f} | {c['li_improves_psnr']} |")
    lines.append("\nPaper Table 2 reference values are in `aggregate_summary.json` (`paper_table2`).")
    lines.append("Numbers are NOT directly comparable to paper (grayscale, tiled eval, bounded iters, fallback arch). See deviations.\n")
    (OUT / "table2_report.md").write_text("\n".join(lines), encoding="utf-8")

    (OUT / "status.md").write_text(
        f"# Table 2 full\n\n**Status:** COMPLETE\n**Iterations:** {iterations}\n**Updated:** {datetime.now(timezone.utc).isoformat()}\n\n"
        f"all_finite: {payload['all_finite']}; LI improves PSNR on: "
        f"{[ds for ds in datasets if comparison[ds]['li_improves_psnr']]}\n",
        encoding="utf-8",
    )
    print(json.dumps({"comparison": comparison, "all_finite": payload["all_finite"]}, indent=2))


if __name__ == "__main__":
    main()
