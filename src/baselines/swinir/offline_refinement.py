"""Offline SwinIR refinement on fixed base CNN outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from baselines.swinir.refinement_model import OfflineSwinIRRefinement
from evaluation.metrics import mse, psnr, ssim


class OfflineRefinementDataset(Dataset):
    def __init__(self, manifest_rows: list[dict]) -> None:
        self.rows = manifest_rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.rows[index]
        data = torch.load(row["file"], weights_only=False)
        return data["x_base"].squeeze(0), data["x_gt"].squeeze(0), row["patch_id"]


def load_manifest_rows(manifest_path: Path, variant: str, split: str) -> list[dict]:
    rows = []
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] == variant and row["split"] == split:
                rows.append(row)
    return rows


def swinir_loss_fn(name: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if name == "mse":
        return F.mse_loss(pred, target)
    if name == "l1":
        return F.l1_loss(pred, target)
    raise ValueError(name)


def build_offline_refiner(config: dict[str, Any]) -> OfflineSwinIRRefinement:
    swinir_cfg = dict(config.get("swinir", {}))
    swinir_cfg.setdefault("img_size", config["dataset"]["image_size"])
    return OfflineSwinIRRefinement(swinir_cfg, dict(config.get("refinement", {})))


@torch.no_grad()
def evaluate_offline_refiner(
    model: OfflineSwinIRRefinement,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    bm = bs = bp = rm = rs = rp = 0.0
    n = 0
    for x_base, x_gt, _ in loader:
        x_base = x_base.to(device)
        x_gt = x_gt.to(device)
        if x_base.ndim == 3:
            x_base = x_base.unsqueeze(0)
            x_gt = x_gt.unsqueeze(0)
        x_ref = model.refine(x_base)
        bm += float(mse(x_base, x_gt).item())
        bs += float(ssim(x_base, x_gt).item())
        bp += float(psnr(x_base, x_gt).item())
        rm += float(mse(x_ref, x_gt).item())
        rs += float(ssim(x_ref, x_gt).item())
        rp += float(psnr(x_ref, x_gt).item())
        n += 1
    return {
        "base_mse": bm / n,
        "base_ssim": bs / n,
        "base_psnr": bp / n,
        "refined_mse": rm / n,
        "refined_ssim": rs / n,
        "refined_psnr": rp / n,
        "delta_mse": (rm - bm) / n,
        "delta_ssim": (rs - bs) / n,
        "improves_mse": (rm / n) <= (bm / n),
        "improves_ssim": (rs / n) >= (bs / n),
        "promising": (rm / n) <= (bm / n) and (rs / n) >= (bs / n),
    }


def train_offline_refiner(
    config: dict[str, Any],
    manifest_path: Path,
    variant_label: str,
    device: torch.device,
    *,
    steps: int,
    lr: float,
    loss_name: str,
) -> tuple[OfflineSwinIRRefinement, dict[str, Any]]:
    model = build_offline_refiner(config).to(device)
    train_loader = DataLoader(
        OfflineRefinementDataset(load_manifest_rows(manifest_path, variant_label, "train")),
        batch_size=1,
        shuffle=True,
    )
    val_loader = DataLoader(
        OfflineRefinementDataset(load_manifest_rows(manifest_path, variant_label, "val")),
        batch_size=1,
        shuffle=False,
    )
    test_loader = DataLoader(
        OfflineRefinementDataset(load_manifest_rows(manifest_path, variant_label, "test")),
        batch_size=1,
        shuffle=False,
    )
    opt = torch.optim.Adam(model.parameters_for_training(), lr=lr)
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_state = None
    step = 0
    train_iter = iter(train_loader)
    model.train()
    while step < steps:
        try:
            x_base, x_gt, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x_base, x_gt, _ = next(train_iter)
        x_base = x_base.to(device)
        x_gt = x_gt.to(device)
        if x_base.ndim == 3:
            x_base = x_base.unsqueeze(0)
            x_gt = x_gt.unsqueeze(0)
        opt.zero_grad(set_to_none=True)
        pred = model.refine(x_base)
        loss = swinir_loss_fn(loss_name, pred, x_gt)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        opt.step()
        train_losses.append(float(loss.item()))
        step += 1
        if step % max(1, steps // 10) == 0 or step == steps:
            model.eval()
            vl = 0.0
            vn = 0
            with torch.no_grad():
                for vb, vg, _ in val_loader:
                    vb = vb.to(device)
                    vg = vg.to(device)
                    if vb.ndim == 3:
                        vb = vb.unsqueeze(0)
                        vg = vg.unsqueeze(0)
                    vl += float(swinir_loss_fn(loss_name, model.refine(vb), vg).item())
                    vn += 1
            val_losses.append(vl / max(vn, 1))
            if val_losses[-1] < best_val:
                best_val = val_losses[-1]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model.train()
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_offline_refiner(model, test_loader, device)
    alpha = float(model.alpha.item()) if hasattr(model, "alpha") else None
    summary = {
        "train_loss_start": train_losses[0] if train_losses else None,
        "train_loss_end": train_losses[-1] if train_losses else None,
        "val_loss_curve": val_losses,
        "best_val_loss": best_val,
        "alpha_final": alpha,
        "test": test_metrics,
        "loss_finite": all(torch.isfinite(torch.tensor(x)).item() for x in train_losses),
        "swinir_steps": len(train_losses),
        **{f"test_{k}": v for k, v in test_metrics.items() if k in {"refined_mse", "refined_ssim", "refined_psnr", "base_mse", "base_ssim"}},
    }
    return model, summary
