#!/usr/bin/env python3
"""AM-4 — paper-faithful SwinIR Table-2 rerun (full / budget / smoke).

Removes the AM-4 concrete deviations of the frozen `swinir_table2_full` run:
  embed_dim 96 -> 180, batch 16 -> effective 32 (grad accumulation), 8000 iters
  -> config-driven (up to SwinIR ~5e5), capped center-biased tiled eval -> fair
  full deterministic tiling. Adds a val split + best-val/final checkpoints +
  checkpoint metadata + train/val curves + illumination saving + figures.

LI and w/o-LI are identical except the illumination mode (fair by construction;
verified by scripts/table02_swinir_sr/audit_fairness.py).

Usage:
  # full pipeline (both conditions sequentially, then aggregate + figures)
  python scripts/table02_swinir_sr/run.py --config configs/table02_swinir_sr/smoke.yaml

  # one condition on a specific GPU (parallelise across GPUs), then aggregate
  python scripts/table02_swinir_sr/run.py --config configs/table02_swinir_sr/budget.yaml \
      --only-condition swinir_wo_li --device cuda:0
  python scripts/table02_swinir_sr/run.py --config configs/table02_swinir_sr/budget.yaml \
      --only-condition swinir_with_li --device cuda:1
  python scripts/table02_swinir_sr/run.py --config configs/table02_swinir_sr/budget.yaml \
      --aggregate-only

  # resume a compute-limited full run from the latest checkpoint
  python scripts/table02_swinir_sr/run.py --config archive/configs/swinir/am4_table2_full.yaml --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.swinir import am4_table2 as A
from baselines.swinir.losses import build_loss_stack, pixel_loss

OUT_BASE = ROOT / "experiments/table02_swinir_sr"

PAPER_TABLE2 = {
    "Set5": {"wo_li": [14.03, 0.3079], "with_li": [26.74, 0.8113]},
    "Set14": {"wo_li": [13.64, 0.2258], "with_li": [23.60, 0.6930]},
    "BSD100": {"wo_li": [14.28, 0.2094], "with_li": [22.90, 0.6317]},
    "Urban100": {"wo_li": [13.51, 0.2146], "with_li": [21.51, 0.6402]},
    "Manga109": {"wo_li": [12.09, 0.1952], "with_li": [20.18, 0.6652]},
}


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def pick_device(requested: str | None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    # pick the GPU with most free memory
    best_idx, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_idx, best_free = i, free
    return torch.device(f"cuda:{best_idx}")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Training / evaluation for one condition
# ---------------------------------------------------------------------------
def train_condition(
    cfg: dict[str, Any],
    cond: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    *,
    resume: bool,
    iterations_override: int | None,
) -> dict[str, Any]:
    name = cond["name"]
    learnable = bool(cond["learnable"])
    cond_dir = out_dir / name
    ckpt_dir = cond_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (cond_dir / "examples").mkdir(parents=True, exist_ok=True)
    (cond_dir / "illumination").mkdir(parents=True, exist_ok=True)

    seed = int(cfg["experiment"]["seed"])
    tr = cfg["training"]
    micro, accum, eff = A.resolve_grad_accum(cfg)
    iterations = int(iterations_override or tr["iterations"])
    amp_dtype = A.amp_dtype_from_cfg(cfg)
    ps = int(cfg["data"]["patch_size"])
    train_m = float(tr.get("train_sigmoid_m", 8.0))
    eval_m = float(tr.get("eval_sigmoid_m", 8.0))

    # --- model + losses + optimizers ---
    model = A.build_model(cfg, learnable=learnable, seed=seed).to(device)
    loss_cfg = dict(tr["loss"])
    loss_cfg["in_chans"] = 1
    stack = build_loss_stack(loss_cfg, device)
    disc = stack.get("discriminator")
    opts = A.build_optimizers(model, cfg, learnable=learnable, discriminator=disc)
    opt_g, opt_d = opts["opt_g"], opts.get("opt_d")

    # --- data (identical for both conditions) ---
    train_paths, val_paths = A.split_train_val(cfg)
    train_ds = A.PathListSRDataset(train_paths, patch_size=ps, grayscale=True, random_crops=True, seed=seed)
    val_ds = A.PathListSRDataset(val_paths, patch_size=ps, grayscale=True, random_crops=False, seed=seed)
    g = torch.Generator().manual_seed(seed)  # identical batch order across conditions
    train_loader = DataLoader(
        train_ds, batch_size=micro, shuffle=True, num_workers=int(tr.get("num_workers", 4)),
        drop_last=True, generator=g, persistent_workers=int(tr.get("num_workers", 4)) > 0,
    )
    val_loader = DataLoader(val_ds, batch_size=max(8, micro), shuffle=False, num_workers=2)

    history: list[dict[str, Any]] = []
    start_step = 0
    best_val = float("inf")
    best_step = -1

    last_ckpt = ckpt_dir / "last.pt"
    if resume and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt_g.load_state_dict(state["opt_g"])
        if opt_d is not None and state.get("opt_d") is not None:
            opt_d.load_state_dict(state["opt_d"])
        start_step = int(state.get("step", 0))
        best_val = float(state.get("best_val", float("inf")))
        best_step = int(state.get("best_step", -1))
        history = state.get("history", [])
        print(f"[{name}] resumed from step {start_step} (best_val={best_val:.5f}@{best_step})", flush=True)

    log_every = int(tr.get("log_every", 100))
    val_every = int(tr.get("val_every", 5000))
    ckpt_every = int(tr.get("ckpt_every", 10000))
    log_path = out_dir / "run.log"

    def validate() -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for vb in val_loader:
                vb = vb.to(device)
                ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype and device.type == "cuda" else _null()
                with ctx:
                    rec = model(vb, sigmoid_m=eval_m if learnable else None, apply_noise=False)["x_recon"]
                tot += float(pixel_loss(rec.float().clamp(0, 1), vb, "l1").item()) * vb.shape[0]
                n += vb.shape[0]
        model.train()
        return tot / max(1, n)

    def save_ckpt(path: Path, step: int) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict() if opt_d is not None else None,
                "step": step,
                "best_val": best_val,
                "best_step": best_step,
                "history": history,
                "condition": name,
                "learnable": learnable,
            },
            path,
        )

    model.train()
    data_iter = iter(train_loader)

    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            return next(data_iter)

    t0 = time.time()
    finite = True
    step = start_step
    while step < iterations:
        if opt_d is not None:
            opt_d.zero_grad(set_to_none=True)
        opt_g.zero_grad(set_to_none=True)
        comp = {"pixel": 0.0, "perceptual": 0.0, "adv": 0.0, "d": 0.0, "g_total": 0.0}
        for _ in range(accum):
            gt = next_batch().to(device)
            ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype and device.type == "cuda" else _null()
            with ctx:
                out = model(gt, sigmoid_m=train_m if learnable else None, apply_noise=False)
                rec = out["x_recon"]
            # discriminator update (accumulated)
            if opt_d is not None:
                with ctx:
                    d_loss = (stack["gan_loss"](disc(gt), True) + stack["gan_loss"](disc(rec.detach()), False)) / accum
                d_loss.backward()
                comp["d"] += float(d_loss.item())
            # generator update (accumulated); freeze D grads during G backward
            with ctx:
                g_pix = stack["pixel_weight"] * pixel_loss(rec, gt, stack["pixel_kind"])
                g_loss = g_pix
                comp["pixel"] += float(g_pix.item()) / accum
                if "perceptual" in stack:
                    g_perc = stack["perceptual_weight"] * stack["perceptual"](rec, gt)
                    g_loss = g_loss + g_perc
                    comp["perceptual"] += float(g_perc.item()) / accum
                if opt_d is not None:
                    g_adv = stack["gan_weight"] * stack["gan_loss"](disc(rec), True)
                    g_loss = g_loss + g_adv
                    comp["adv"] += float(g_adv.item()) / accum
                g_loss = g_loss / accum
            if opt_d is not None:
                for p in disc.parameters():
                    p.requires_grad_(False)
            g_loss.backward()
            if opt_d is not None:
                for p in disc.parameters():
                    p.requires_grad_(True)
            comp["g_total"] += float(g_loss.item())
            if not torch.isfinite(g_loss).item():
                finite = False
        if opt_d is not None:
            opt_d.step()
        opt_g.step()
        step += 1

        if step % log_every == 0 or step == iterations:
            rate = (step - start_step) / max(1e-9, time.time() - t0)
            rec_line = {**comp, "step": step, "rate_it_s": round(rate, 3), "kind": "train"}
            history.append(rec_line)
            with log_path.open("a", encoding="utf-8") as h:
                h.write(f"[{name}] {json.dumps(rec_line)}\n")
            print(f"[{name}] step {step}/{iterations} g={comp['g_total']:.4f} pix={comp['pixel']:.4f} "
                  f"{rate:.2f} it/s", flush=True)

        if step % val_every == 0 or step == iterations:
            vloss = validate()
            history.append({"step": step, "val_l1": vloss, "kind": "val"})
            with log_path.open("a", encoding="utf-8") as h:
                h.write(f"[{name}] {json.dumps({'step': step, 'val_l1': vloss, 'kind': 'val'})}\n")
            print(f"[{name}] step {step} VAL_L1={vloss:.5f} (best {best_val:.5f}@{best_step})", flush=True)
            if vloss < best_val:
                best_val = vloss
                best_step = step
                save_ckpt(ckpt_dir / "best.pt", step)

        if step % ckpt_every == 0 or step == iterations:
            save_ckpt(last_ckpt, step)

    save_ckpt(last_ckpt, step)
    if not (ckpt_dir / "best.pt").exists():
        save_ckpt(ckpt_dir / "best.pt", step)
        best_step = step

    # --- save illumination patterns ---
    illum = _save_illumination(model, cond_dir / "illumination", eval_m, learnable)

    # --- evaluate (best-val checkpoint primary; final as cross-check) ---
    ev = cfg["eval"]
    sel = str(ev.get("tile_selection", "all"))
    max_tiles = ev.get("max_tiles_per_image")
    max_imgs = cfg["data"].get("max_test_images")
    eval_batch = int(ev.get("eval_batch", 64))
    stitched = bool(ev.get("stitched", True))

    def eval_all(tag: str) -> dict[str, Any]:
        per = {}
        for ds_name, rel in cfg["data"]["test_roots"].items():
            per[ds_name] = A.eval_dataset_fair(
                model, ROOT / rel, patch_size=ps, device=device, learnable=learnable,
                eval_sigmoid_m=eval_m, selection=sel, max_tiles_per_image=max_tiles,
                max_images=max_imgs, eval_batch=eval_batch, amp_dtype=amp_dtype, compute_stitched=stitched,
            )
            print(f"[{name}/{tag}] {ds_name}: PSNR {per[ds_name]['psnr']:.2f} SSIM {per[ds_name]['ssim']:.4f} "
                  f"({per[ds_name]['tiles']} tiles, {per[ds_name]['images']} imgs)", flush=True)
        return per

    # final-checkpoint eval (model currently holds final weights)
    per_final = eval_all("final")
    _save_examples(model, cfg, device, eval_m, learnable, cond_dir / "examples", ps, amp_dtype)

    # best-val checkpoint eval
    best_state = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    per_best = eval_all("best")

    # checkpoint metadata
    arch = A.model_arch_summary(model)
    meta = {
        "condition": name,
        "learnable": learnable,
        "iterations_target": iterations,
        "iterations_reached": step,
        "effective_batch_size": eff,
        "micro_batch_size": micro,
        "grad_accum": accum,
        "best_val_l1": best_val,
        "best_step": best_step,
        "arch_summary": arch,
        "optimizer_groups": A.optimizer_group_summary(opt_g),
        "amp_dtype": str(amp_dtype),
        "device": str(device),
        "torch_version": torch.__version__,
        "git_commit": git_commit(),
        "seed": seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "train_images": len(train_paths),
        "val_images": len(val_paths),
        "checkpoints": {"best": str(ckpt_dir / "best.pt"), "last": str(last_ckpt)},
    }
    (cond_dir / "checkpoint_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    result = {
        "name": name,
        "learnable": learnable,
        "loss_finite": finite,
        "iterations_reached": step,
        "iterations_target": iterations,
        "per_dataset_best": per_best,
        "per_dataset_final": per_final,
        "best_val_l1": best_val,
        "best_step": best_step,
        "illumination": illum,
        "arch_summary": arch,
        "optimizer_groups": A.optimizer_group_summary(opt_g),
        "history": history,
        "metadata": meta,
    }
    (cond_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _save_curves(history, cond_dir / "curves.png", name)
    return result


def _save_illumination(model, out_dir: Path, eval_m: float, learnable: bool) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        patterns = model.pattern_generator(sigmoid_m=eval_m if learnable else None).detach().float().cpu()
    torch.save(patterns, out_dir / "patterns.pt")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = patterns.shape[0]
        fig, axes = plt.subplots(1, t, figsize=(2.4 * t, 2.6))
        if t == 1:
            axes = [axes]
        for i in range(t):
            axes[i].imshow(patterns[i, 0], cmap="gray", vmin=0, vmax=1)
            axes[i].set_title(f"Ht[{i}]")
            axes[i].axis("off")
        fig.suptitle(f"Illumination ({'learnable' if learnable else 'fixed pseudo-random'})")
        fig.tight_layout()
        fig.savefig(out_dir / "patterns.png", dpi=110)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"illumination figure skipped: {exc}", flush=True)
    return {
        "mean": float(patterns.mean()),
        "std": float(patterns.std()),
        "min": float(patterns.min()),
        "max": float(patterns.max()),
        "binarization_frac_near_0_or_1": float(((patterns < 0.05) | (patterns > 0.95)).float().mean()),
        "shape": list(patterns.shape),
    }


@torch.no_grad()
def _save_examples(model, cfg, device, eval_m, learnable, out_dir: Path, ps: int, amp_dtype) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()  # SwinIR drop_path=0.1 stochastic depth must be disabled for inference
    from torchvision.utils import save_image

    for ds_name, rel in list(cfg["data"]["test_roots"].items())[:3]:
        paths = A.list_test_images(ROOT / rel, max_images=1)
        if not paths:
            continue
        img = A._load_gray(paths[0])
        img = img[:, :ps, :ps].unsqueeze(0).to(device)
        ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype and device.type == "cuda" else _null()
        with ctx:
            rec = model(img, sigmoid_m=eval_m if learnable else None, apply_noise=False)["x_recon"]
        # fixed [0,1] range so GT and recon are visually comparable
        save_image(img.float().clamp(0, 1), out_dir / f"{ds_name}_gt.png")
        save_image(rec.float().clamp(0, 1), out_dir / f"{ds_name}_recon.png")


def _save_curves(history: list[dict], path: Path, name: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tr = [(h["step"], h["g_total"]) for h in history if h.get("kind") == "train"]
        va = [(h["step"], h["val_l1"]) for h in history if h.get("kind") == "val"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        if tr:
            xs, ys = zip(*tr)
            ax[0].plot(xs, ys)
            ax[0].set_title(f"{name} train g_total")
            ax[0].set_xlabel("step"); ax[0].set_ylabel("g_total")
        if va:
            xs, ys = zip(*va)
            ax[1].plot(xs, ys, marker="o")
            ax[1].set_title(f"{name} val L1")
            ax[1].set_xlabel("step"); ax[1].set_ylabel("val L1")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"curves figure skipped: {exc}", flush=True)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# Aggregation + figures
# ---------------------------------------------------------------------------
def aggregate(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    conds = cfg["training"]["conditions"]
    results = {}
    for c in conds:
        rp = out_dir / c["name"] / "result.json"
        if not rp.exists():
            print(f"[aggregate] missing {rp}; skip", flush=True)
            continue
        results[c["name"]] = json.loads(rp.read_text(encoding="utf-8"))
    if "swinir_wo_li" not in results or "swinir_with_li" not in results:
        print("[aggregate] need both conditions present; aborting aggregate", flush=True)
        return {}

    wo, wi = results["swinir_wo_li"], results["swinir_with_li"]
    datasets = list(cfg["data"]["test_roots"].keys())
    which = "per_dataset_best"
    comparison = {}
    for ds in datasets:
        wo_p, wi_p = wo[which][ds]["psnr"], wi[which][ds]["psnr"]
        wo_s, wi_s = wo[which][ds]["ssim"], wi[which][ds]["ssim"]
        comparison[ds] = {
            "wo_li_psnr": wo_p, "with_li_psnr": wi_p,
            "wo_li_ssim": wo_s, "with_li_ssim": wi_s,
            "li_gain_psnr": wi_p - wo_p, "li_gain_ssim": wi_s - wo_s,
            "li_improves_psnr": wi_p > wo_p,
            "paper_wo_li": PAPER_TABLE2[ds]["wo_li"], "paper_with_li": PAPER_TABLE2[ds]["with_li"],
            "paper_li_gain_psnr": PAPER_TABLE2[ds]["with_li"][0] - PAPER_TABLE2[ds]["wo_li"][0],
        }

    payload = {
        "label": f"AM-4 SwinIR Table-2 ({cfg['experiment'].get('tag', '?')}) — checkpoint=best-val",
        "tag": cfg["experiment"].get("tag"),
        "paper_table2": PAPER_TABLE2,
        "comparison": comparison,
        "conditions": {
            n: {
                "iterations_reached": r["iterations_reached"],
                "iterations_target": r["iterations_target"],
                "best_val_l1": r["best_val_l1"],
                "best_step": r["best_step"],
                "loss_finite": r["loss_finite"],
                "arch_summary": r["arch_summary"],
                "optimizer_groups": r["optimizer_groups"],
                "illumination": r["illumination"],
                "per_dataset_best": r["per_dataset_best"],
                "per_dataset_final": r["per_dataset_final"],
            }
            for n, r in results.items()
        },
        "all_finite": all(r["loss_finite"] for r in results.values()),
        "deviations_from_paper": cfg.get("deviations_from_paper", []),
        "eval": dict(cfg["eval"]),
        "effective_batch_size": cfg["training"]["effective_batch_size"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "aggregate_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.writer(h)
        w.writerow(["dataset", "condition", "checkpoint", "psnr", "ssim", "stitched_psnr", "stitched_ssim", "tiles", "images"])
        for n, r in results.items():
            for ck in ("per_dataset_best", "per_dataset_final"):
                for ds in datasets:
                    d = r[ck][ds]
                    w.writerow([ds, n, ck.replace("per_dataset_", ""), f"{d['psnr']:.4f}", f"{d['ssim']:.4f}",
                                f"{d.get('stitched_psnr', float('nan')):.4f}", f"{d.get('stitched_ssim', float('nan')):.4f}",
                                d["tiles"], d["images"]])

    _write_report(cfg, out_dir, payload, comparison, datasets)
    _make_figures(cfg, out_dir, results, comparison, datasets)
    return payload


def _write_report(cfg, out_dir: Path, payload, comparison, datasets) -> None:
    tag = cfg["experiment"].get("tag", "?")
    lines = [
        f"# AM-4 SwinIR Table-2 — `{tag}` run summary\n",
        f"**Checkpoint:** best-val. **Effective batch:** {payload['effective_batch_size']}. "
        f"**All finite:** {payload['all_finite']}.\n",
        "| Dataset | w/o LI PSNR | with LI PSNR | LI gain PSNR | w/o LI SSIM | with LI SSIM | LI gain SSIM | paper LI gain PSNR |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ds in datasets:
        c = comparison[ds]
        lines.append(
            f"| {ds} | {c['wo_li_psnr']:.2f} | {c['with_li_psnr']:.2f} | {c['li_gain_psnr']:+.2f} | "
            f"{c['wo_li_ssim']:.4f} | {c['with_li_ssim']:.4f} | {c['li_gain_ssim']:+.4f} | {c['paper_li_gain_psnr']:.2f} |"
        )
    for n, cinfo in payload["conditions"].items():
        lines.append(
            f"\n- **{n}**: iters {cinfo['iterations_reached']}/{cinfo['iterations_target']}, "
            f"best_val_L1 {cinfo['best_val_l1']:.5f}@{cinfo['best_step']}, "
            f"embed_dim {cinfo['arch_summary']['embed_dim']}, "
            f"illum learnable={cinfo['arch_summary']['illumination_learnable']} "
            f"({cinfo['arch_summary']['illumination_params']} params)"
        )
    lines.append("\n**Deviations / assumptions:** see `aggregate_summary.json` and the config `deviations_from_paper`.")
    if tag != "full":
        lines.append(f"\n> NOTE: `{tag}` is a NON-FINAL diagnostic. Do NOT cite as a Table-2 reproduction.")
    (out_dir / f"{tag}_run_table.md").write_text("\n".join(lines), encoding="utf-8")


def _make_figures(cfg, out_dir: Path, results, comparison, datasets) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        print(f"figures skipped: {exc}", flush=True)
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = cfg["experiment"].get("tag", "?")

    # PSNR / SSIM bar charts (ours wo/with + paper wo/with)
    x = np.arange(len(datasets))
    width = 0.2
    for metric, idx in (("PSNR", 0), ("SSIM", 1)):
        fig, ax = plt.subplots(figsize=(11, 4.5))
        wo = [comparison[d]["wo_li_psnr" if metric == "PSNR" else "wo_li_ssim"] for d in datasets]
        wi = [comparison[d]["with_li_psnr" if metric == "PSNR" else "with_li_ssim"] for d in datasets]
        pwo = [PAPER_TABLE2[d]["wo_li"][idx] for d in datasets]
        pwi = [PAPER_TABLE2[d]["with_li"][idx] for d in datasets]
        ax.bar(x - 1.5 * width, wo, width, label="ours w/o LI")
        ax.bar(x - 0.5 * width, wi, width, label="ours with LI")
        ax.bar(x + 0.5 * width, pwo, width, label="paper w/o LI", alpha=0.6)
        ax.bar(x + 1.5 * width, pwi, width, label="paper with LI", alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels(datasets)
        ax.set_ylabel(metric); ax.set_title(f"AM-4 {tag}: {metric} (best-val) vs paper Table 2")
        ax.legend()
        fig.tight_layout(); fig.savefig(fig_dir / f"{metric.lower()}_bars.png", dpi=120); plt.close(fig)

    # combined illumination panel
    try:
        wo_p = torch.load(out_dir / "swinir_wo_li" / "illumination" / "patterns.pt")
        wi_p = torch.load(out_dir / "swinir_with_li" / "illumination" / "patterns.pt")
        t = wo_p.shape[0]
        fig, axes = plt.subplots(2, t, figsize=(2.3 * t, 4.8))
        for i in range(t):
            axes[0, i].imshow(wo_p[i, 0], cmap="gray", vmin=0, vmax=1); axes[0, i].axis("off")
            axes[0, i].set_title(f"fixed Ht[{i}]")
            axes[1, i].imshow(wi_p[i, 0], cmap="gray", vmin=0, vmax=1); axes[1, i].axis("off")
            axes[1, i].set_title(f"learned Ht[{i}]")
        fig.suptitle(f"AM-4 {tag}: illumination (top=fixed w/o LI, bottom=learned with LI)")
        fig.tight_layout(); fig.savefig(fig_dir / "illumination_panel.png", dpi=120); plt.close(fig)
    except Exception as exc:
        print(f"illumination panel skipped: {exc}", flush=True)

    # qualitative reconstruction panel (gt / wo-LI / with-LI) for available datasets
    try:
        from torchvision.io import read_image

        ds_avail = []
        for ds in datasets:
            gt = out_dir / "swinir_wo_li" / "examples" / f"{ds}_gt.png"
            wo = out_dir / "swinir_wo_li" / "examples" / f"{ds}_recon.png"
            wi = out_dir / "swinir_with_li" / "examples" / f"{ds}_recon.png"
            if gt.exists() and wo.exists() and wi.exists():
                ds_avail.append((ds, gt, wo, wi))
        if ds_avail:
            fig, axes = plt.subplots(len(ds_avail), 3, figsize=(7.5, 2.6 * len(ds_avail)))
            if len(ds_avail) == 1:
                axes = axes.reshape(1, 3)
            for r, (ds, gt, wo, wi) in enumerate(ds_avail):
                for cidx, (title, p) in enumerate([("GT", gt), ("w/o LI", wo), ("with LI", wi)]):
                    im = read_image(str(p)).float().mean(0) / 255.0
                    axes[r, cidx].imshow(im, cmap="gray", vmin=0, vmax=1); axes[r, cidx].axis("off")
                    if r == 0:
                        axes[r, cidx].set_title(title)
                axes[r, 0].set_ylabel(ds)
            fig.suptitle(f"AM-4 {tag}: reconstructions")
            fig.tight_layout(); fig.savefig(fig_dir / "qualitative_panel.png", dpi=120); plt.close(fig)
    except Exception as exc:
        print(f"qualitative panel skipped: {exc}", flush=True)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--only-condition", default=None, help="swinir_wo_li | swinir_with_li")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--output-base", default=None, help="Override experiments/.../am4_swinir_table2_resolution")
    args = ap.parse_args()

    cfg = A.load_am4_config(args.config)
    tag = cfg["experiment"].get("tag", Path(args.config).stem)
    base = Path(args.output_base) if args.output_base else OUT_BASE
    if not base.is_absolute():
        base = ROOT / base
    out_dir = base / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "configs_used").mkdir(exist_ok=True)
    shutil.copy2(args.config, out_dir / "configs_used" / Path(args.config).name)

    if args.aggregate_only:
        aggregate(cfg, out_dir)
        return

    device = pick_device(args.device or cfg["experiment"].get("device"))
    print(f"AM-4 [{tag}] device={device} effective_batch={cfg['training']['effective_batch_size']}", flush=True)

    (out_dir / "status.md").write_text(
        f"# AM-4 {tag}\n\n**Status:** RUNNING\n**Device:** {device}\n"
        f"**Updated:** {datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )

    conds = cfg["training"]["conditions"]
    if args.only_condition:
        conds = [c for c in conds if c["name"] == args.only_condition]
        if not conds:
            raise SystemExit(f"unknown condition {args.only_condition}")

    for cond in conds:
        print(f"\n=== AM-4 {tag}: {cond['name']} (learnable={cond['learnable']}) ===", flush=True)
        train_condition(cfg, cond, device, out_dir, resume=args.resume, iterations_override=args.iterations)

    # aggregate if both conditions now have results
    have_both = all((out_dir / c["name"] / "result.json").exists() for c in cfg["training"]["conditions"])
    if have_both:
        aggregate(cfg, out_dir)
        (out_dir / "status.md").write_text(
            f"# AM-4 {tag}\n\n**Status:** COMPLETE\n**Device:** {device}\n"
            f"**Updated:** {datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )
    else:
        (out_dir / "status.md").write_text(
            f"# AM-4 {tag}\n\n**Status:** PARTIAL (ran {[c['name'] for c in conds]})\n"
            f"Run the other condition then `--aggregate-only`.\n"
            f"**Updated:** {datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
