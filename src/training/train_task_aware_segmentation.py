"""Staged task-aware segmentation training (paper supplement B.0.1, §5.3).

Implements the paper's three-stage procedure end-to-end:

* **Stage 1 — content-aware reconstruction pretraining.** Train the
  microscope (excitation patterns + inverse model) with the L1 reconstruction
  objective, exactly like the repo's content-aware setup. Learnable variants use
  the staged-hardening schedule; the fixed pseudo-random variant keeps the
  illumination fixed and trains the inverse model only.
* **Stage 2 — segmentation-head-only training.** Append the convolutional
  segmentation head and train it on pseudo-ground-truth masks while the
  microscope / reconstruction components are frozen.
* **Stage 3 — end-to-end task-aware finetuning.** Unfreeze every learnable
  component (excitation pattern parameters for the learnable variant, the
  inverse model, and the segmentation head) and finetune with the segmentation
  task loss. The segmentation loss therefore backpropagates into the
  illumination parameters for the learnable variant.

The trainer logs ``requires_grad`` reports and gradient norms per stage so the
freeze (Stage 2) and end-to-end finetuning (Stage 3) can be verified.
"""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from evaluation.pattern_inspection import save_pattern_inspection
from models.task_aware_microscope import TaskAwareMicroscope
from training.dataloaders import build_dataloader
from training.segmentation_losses import (
    TaskAwareLossWeights,
    bce_with_logits_loss,
    task_aware_segmentation_loss,
)
from training.staged_hardening_train import train_staged_hardening
from training.train_reconstruction import train as train_reconstruction_model
from utils.device import device_from_config
from utils.logging import copy_assumptions, ensure_run_directory, save_patterns, save_run_config, save_run_metadata
from utils.reproducibility import get_git_commit_hash, set_seed


# --------------------------------------------------------------------------- #
# metric helpers
# --------------------------------------------------------------------------- #
def _dice(pred_bin: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    inter = (pred_bin * target).sum()
    return float((2 * inter + eps) / (pred_bin.sum() + target.sum() + eps))


def _iou(pred_bin: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    inter = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - inter
    return float((inter + eps) / (union + eps))


def _grad_norm(params: list[torch.nn.Parameter]) -> float:
    grads = [p.grad.detach() for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.norm(torch.stack([g.norm() for g in grads])).item())


def _empty_grad_norms() -> dict[str, dict]:
    return {k: {"max": 0.0, "first": None, "last": 0.0} for k in ("segmentation_head", "inverse_model", "illumination")}


def _merge_grad_norms(acc: dict[str, dict], new: dict[str, dict]) -> dict[str, dict]:
    for key in acc:
        acc[key]["max"] = max(acc[key]["max"], new[key]["max"])
        if acc[key]["first"] is None:
            acc[key]["first"] = new[key]["first"]
        acc[key]["last"] = new[key]["last"]
    return acc


def _build_stage3_phases(
    total_steps: int,
    soft_m: float,
    harden_m: list[float],
    soft_fraction: float,
    learnable: bool,
) -> list[tuple[float, int]]:
    """Soft->hard finetune schedule (learnable); single phase otherwise.

    For the learnable variant we spend ``soft_fraction`` of steps at a soft
    sigmoid sharpness (so the segmentation loss flows into the pattern
    parameters), then split the remainder across a hardening tail so the final
    patterns are near-binary for evaluation. For the fixed variant the sharpness
    is irrelevant (patterns are fixed), so one phase suffices.
    """
    if not learnable or not harden_m:
        return [(soft_m, total_steps)]
    soft_steps = max(1, int(round(total_steps * soft_fraction)))
    remaining = max(0, total_steps - soft_steps)
    per = max(1, remaining // len(harden_m))
    phases = [(soft_m, soft_steps)]
    used = 0
    for idx, m in enumerate(harden_m):
        steps = (remaining - used) if idx == len(harden_m) - 1 else per
        if steps > 0:
            phases.append((float(m), steps))
            used += steps
    return phases


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _collect_probs(
    model: TaskAwareMicroscope,
    loader: DataLoader,
    device: torch.device,
    *,
    sigmoid_m: float,
    apply_noise: bool,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return stacked ``(probs, masks)`` on CPU plus mean BCE-with-logits."""
    model.eval()
    probs: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    bce_total = 0.0
    n_batches = 0
    for specimen, mask in loader:
        specimen = specimen.to(device)
        mask = mask.to(device)
        outputs = model(specimen, sigmoid_m=sigmoid_m, apply_noise=apply_noise)
        bce_total += float(bce_with_logits_loss(outputs["seg_logits"], mask).item())
        n_batches += 1
        probs.append(outputs["seg_prob"].cpu())
        masks.append(mask.cpu())
    return torch.cat(probs), torch.cat(masks), bce_total / max(n_batches, 1)


def select_threshold(
    model: TaskAwareMicroscope,
    loader: DataLoader,
    device: torch.device,
    *,
    sigmoid_m: float,
    apply_noise: bool,
    thresholds: list[float],
) -> tuple[float, float]:
    """Pick the threshold maximizing mean per-sample Dice on ``loader``."""
    probs, masks, _ = _collect_probs(model, loader, device, sigmoid_m=sigmoid_m, apply_noise=apply_noise)
    best_t, best_dice = 0.5, -1.0
    for t in thresholds:
        pred = (probs > t).float()
        dice = sum(_dice(pred[i], masks[i]) for i in range(pred.shape[0])) / pred.shape[0]
        if dice > best_dice:
            best_dice, best_t = dice, float(t)
    return best_t, best_dice


@torch.no_grad()
def evaluate_segmentation(
    model: TaskAwareMicroscope,
    loader: DataLoader,
    device: torch.device,
    *,
    sigmoid_m: float,
    apply_noise: bool,
    threshold: float,
) -> dict[str, float]:
    probs, masks, bce = _collect_probs(model, loader, device, sigmoid_m=sigmoid_m, apply_noise=apply_noise)
    pred = (probs > threshold).float()
    dices = [_dice(pred[i], masks[i]) for i in range(pred.shape[0])]
    ious = [_iou(pred[i], masks[i]) for i in range(pred.shape[0])]
    return {
        "bce": bce,
        "dice": sum(dices) / len(dices),
        "iou": sum(ious) / len(ious),
        "threshold": float(threshold),
        "num_samples": int(pred.shape[0]),
    }


# --------------------------------------------------------------------------- #
# stage 1 (content-aware reconstruction pretraining)
# --------------------------------------------------------------------------- #
def _load_microscope_checkpoint(
    model: TaskAwareMicroscope,
    checkpoint_path: Path,
    device: torch.device,
    image_size: int,
) -> None:
    with torch.no_grad():
        dummy = torch.zeros(1, 1, image_size, image_size, device=device)
        model.microscope(dummy, sigmoid_m=10.0, apply_noise=False)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.microscope.load_state_dict(payload["model_state_dict"])


def _run_stage1_content_aware(
    config: dict[str, Any],
    stage1_dir: Path,
    *,
    learnable: bool,
) -> Path:
    """Train (or reuse) a content-aware microscope; return its best.pt path."""
    recon_config = copy.deepcopy(config)
    recon_config["dataset"]["return_mask"] = False
    recon_config.setdefault("experiment", {})
    recon_config["experiment"]["run_id"] = f"{config['experiment']['run_id']}_stage1"

    task_cfg = config["training"].get("task_aware", {})
    stage1_cfg = task_cfg.get("stage1", {})

    if learnable:
        learn_cfg = stage1_cfg.get("learnable", {})
        recon_config["training"]["learn_patterns"] = True
        recon_config["training"]["staged_hardening"] = {
            "inverse_warmup_steps": int(learn_cfg.get("inverse_warmup_steps", 1500)),
            "joint_soft_steps": int(learn_cfg.get("joint_soft_steps", 3500)),
            "harden_m_values": list(learn_cfg.get("harden_m_values", [2, 4, 8])),
            "harden_steps_per_m": int(learn_cfg.get("harden_steps_per_m", 800)),
        }
        recon_config["training"].pop("max_steps", None)
        train_staged_hardening(recon_config, stage1_dir)
    else:
        recon_config["training"]["learn_patterns"] = False
        recon_config["training"]["max_steps"] = int(stage1_cfg.get("fixed_steps", 4000))
        train_reconstruction_model(recon_config, stage1_dir)

    return stage1_dir / "checkpoints" / "best.pt"


# --------------------------------------------------------------------------- #
# stage 2 / 3 segmentation training loop
# --------------------------------------------------------------------------- #
def _train_segmentation_phase(
    model: TaskAwareMicroscope,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    phase_name: str,
    max_steps: int,
    sigmoid_m: float,
    apply_noise: bool,
    weights: TaskAwareLossWeights,
    log_every: int,
    grad_clip: float | None,
) -> dict[str, Any]:
    """Train with the task-aware segmentation loss; record gradient norms."""
    history: list[dict[str, float]] = []
    grad_norms = {
        "segmentation_head": {"max": 0.0, "first": None, "last": 0.0},
        "inverse_model": {"max": 0.0, "first": None, "last": 0.0},
        "illumination": {"max": 0.0, "first": None, "last": 0.0},
    }
    seg_params = model.segmentation_parameters()
    inv_params = model.inverse_parameters()
    illum_params = model.illumination_parameters()
    needs_specimen = weights.reconstruction_l1_weight != 0.0

    train_iter = itertools.cycle(loader)
    for step in range(1, max_steps + 1):
        specimen, mask = next(train_iter)
        specimen = specimen.to(device)
        mask = mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(specimen, sigmoid_m=sigmoid_m, apply_noise=apply_noise)
        loss, components = task_aware_segmentation_loss(
            outputs, mask, weights, specimen=specimen if needs_specimen else None
        )
        loss.backward()

        gn_seg = _grad_norm(seg_params)
        gn_inv = _grad_norm(inv_params)
        gn_illum = _grad_norm(illum_params)
        for key, value in (
            ("segmentation_head", gn_seg),
            ("inverse_model", gn_inv),
            ("illumination", gn_illum),
        ):
            if grad_norms[key]["first"] is None:
                grad_norms[key]["first"] = value
            grad_norms[key]["max"] = max(grad_norms[key]["max"], value)
            grad_norms[key]["last"] = value

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == max_steps:
            row = {"step": step, **components, "gn_seg": gn_seg, "gn_inv": gn_inv, "gn_illum": gn_illum}
            history.append(row)
            print(
                f"[{phase_name}] step={step}/{max_steps} total={components['total']:.5f} "
                f"bce={components['bce']:.5f} gn_seg={gn_seg:.3e} gn_inv={gn_inv:.3e} "
                f"gn_illum={gn_illum:.3e}",
                flush=True,
            )
    return {"history": history, "grad_norms": grad_norms}


# --------------------------------------------------------------------------- #
# qualitative panels
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _save_qualitative_panel(
    model: TaskAwareMicroscope,
    loader: DataLoader,
    device: torch.device,
    out_path: Path,
    *,
    sigmoid_m: float,
    apply_noise: bool,
    threshold: float,
    num_samples: int,
) -> None:
    model.eval()
    specimen, mask = next(iter(loader))
    specimen = specimen.to(device)
    mask = mask.to(device)
    outputs = model(specimen, sigmoid_m=sigmoid_m, apply_noise=apply_noise)
    recon = outputs["x_recon"]
    prob = outputs["seg_prob"]
    pred = (prob > threshold).float()

    n = min(num_samples, specimen.shape[0])
    cols = ["GT image", "pseudo mask", "reconstruction", "seg prob", f"pred mask (t={threshold:.2f})"]
    fig, axes = plt.subplots(n, len(cols), figsize=(3 * len(cols), 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i in range(n):
        panels = [
            specimen[i, 0].cpu(),
            mask[i, 0].cpu(),
            recon[i, 0].cpu(),
            prob[i, 0].cpu(),
            pred[i, 0].cpu(),
        ]
        for j, (img, title) in enumerate(zip(panels, cols)):
            axes[i, j].imshow(img.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def train_task_aware_segmentation(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    experiment_cfg = config["experiment"]
    training_cfg = config["training"]
    task_cfg = training_cfg.get("task_aware", {})
    seed = experiment_cfg["seed"]
    set_seed(seed)

    run_dir = ensure_run_directory(output_dir)
    save_run_config(config, run_dir)
    copy_assumptions(run_dir)
    save_run_metadata(
        run_dir,
        {
            "seed": seed,
            "git_commit": get_git_commit_hash(),
            "run_id": experiment_cfg["run_id"],
            "training_mode": "task_aware_segmentation_staged",
        },
    )

    device = device_from_config(config)
    print(f"Using device: {device}", flush=True)

    learnable = bool(training_cfg.get("learn_patterns", True))
    image_size = int(config["dataset"]["image_size"])
    apply_noise = config["detector_noise"].get("apply_noise", False)
    eval_m = float(task_cfg.get("eval_sigmoid_m", training_cfg.get("sharpen_eval_m", 10.0)))
    # Stage-3 illumination finetuning uses a soft->hard schedule (like the
    # content-aware procedure) so the segmentation loss can actually flow into
    # the frequency-domain pattern parameters before they re-binarize for eval.
    finetune_soft_m = float(task_cfg.get("finetune_sigmoid_m", 1.0))
    finetune_harden_m = [float(v) for v in task_cfg.get("finetune_harden_m_values", [2.0, 4.0, 8.0])]
    finetune_soft_fraction = float(task_cfg.get("finetune_soft_fraction", 0.6))
    log_every = int(training_cfg.get("log_every", 200))
    seg_steps = int(task_cfg.get("seg_head_steps", 1200))
    finetune_steps = int(task_cfg.get("finetune_steps", 2500))
    grad_clip = training_cfg.get("gradient_clip_norm")
    thresholds = [round(0.05 * k, 2) for k in range(2, 19)]  # 0.10 .. 0.90
    num_qual = int(task_cfg.get("num_qualitative_samples", 6))
    weights = TaskAwareLossWeights.from_config(training_cfg)

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ----- segmentation dataloaders (pseudo-mask targets) ------------------- #
    config["dataset"]["return_mask"] = True
    train_loader = build_dataloader(config, "train")
    val_loader = build_dataloader(config, "val")
    test_loader = build_dataloader(config, "test")

    # ----- model ------------------------------------------------------------ #
    model = TaskAwareMicroscope.from_run_config(config).to(device)

    # ======================= STAGE 1 =================================== #
    stage1_mode = str(task_cfg.get("stage1_mode", "train")).lower()
    explicit_ckpt = task_cfg.get("content_aware_checkpoint")
    stage1_dir = run_dir / "stage1_content_aware"
    if stage1_mode == "load" and explicit_ckpt:
        ckpt_path = Path(explicit_ckpt)
        print(f"[stage1] loading content-aware checkpoint: {ckpt_path}", flush=True)
    else:
        print(f"[stage1] training content-aware microscope (learnable={learnable})...", flush=True)
        ckpt_path = _run_stage1_content_aware(config, stage1_dir, learnable=learnable)
    _load_microscope_checkpoint(model, ckpt_path, device, image_size)

    # ======================= STAGE 2 =================================== #
    # Freeze microscope / reconstruction; train segmentation head only.
    model.set_microscope_trainable(False)
    model.set_segmentation_trainable(True)
    model.microscope.eval()  # freeze BatchNorm running stats too
    model.segmentation_head.train()
    stage2_report = model.trainable_parameter_report()
    print(f"[stage2] trainable report: {json.dumps(stage2_report)}", flush=True)
    assert stage2_report["illumination"]["all_frozen"], "Stage 2 illumination must be frozen"
    assert stage2_report["inverse_model"]["all_frozen"], "Stage 2 inverse model must be frozen"
    assert stage2_report["segmentation_head"]["all_trainable"], "Stage 2 seg head must be trainable"

    seg_head_lr = float(task_cfg.get("seg_head_lr", training_cfg.get("inverse_lr", 1e-3)))
    optimizer_s2 = torch.optim.Adam(model.segmentation_parameters(), lr=seg_head_lr)
    stage2_weights = TaskAwareLossWeights(
        seg_bce_weight=weights.seg_bce_weight,
        seg_dice_weight=weights.seg_dice_weight,
        reconstruction_l1_weight=0.0,
    )
    print(f"[stage2] segmentation-head-only training ({seg_steps} steps)...", flush=True)
    stage2 = _train_segmentation_phase(
        model, train_loader, optimizer_s2, device,
        phase_name="stage2_seg_head", max_steps=seg_steps, sigmoid_m=eval_m,
        apply_noise=apply_noise, weights=stage2_weights, log_every=log_every, grad_clip=grad_clip,
    )
    val_t2, _ = select_threshold(model, val_loader, device, sigmoid_m=eval_m, apply_noise=apply_noise, thresholds=thresholds)
    stage2_val = evaluate_segmentation(model, val_loader, device, sigmoid_m=eval_m, apply_noise=apply_noise, threshold=val_t2)
    print(f"[stage2] val dice={stage2_val['dice']:.4f} iou={stage2_val['iou']:.4f} (t={val_t2})", flush=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config, "stage": "stage2"}, run_dir / "checkpoints" / "stage2_seg_head.pt")
    # Evidence: microscope grads must be zero while head grads are nonzero.
    assert stage2["grad_norms"]["segmentation_head"]["max"] > 0.0, "Stage 2 seg head received no gradient"
    assert stage2["grad_norms"]["inverse_model"]["max"] == 0.0, "Stage 2 inverse model must not receive gradient"
    assert stage2["grad_norms"]["illumination"]["max"] == 0.0, "Stage 2 illumination must not receive gradient"

    # ======================= STAGE 3 =================================== #
    # Finetune all learnable components end-to-end with the segmentation loss.
    model.set_segmentation_trainable(True)
    model.set_inverse_trainable(True)
    if learnable:
        model.set_illumination_trainable(True)
    else:
        model.set_illumination_trainable(False)
    model.train()
    stage3_report = model.trainable_parameter_report()
    print(f"[stage3] trainable report: {json.dumps(stage3_report)}", flush=True)
    assert stage3_report["inverse_model"]["all_trainable"], "Stage 3 inverse model must be trainable"
    assert stage3_report["segmentation_head"]["all_trainable"], "Stage 3 seg head must be trainable"
    if learnable:
        assert stage3_report["illumination"]["all_trainable"], "Stage 3 illumination must be trainable (learnable variant)"

    groups: list[dict[str, Any]] = [
        {"params": model.segmentation_parameters(), "lr": seg_head_lr},
        {"params": model.inverse_parameters(), "lr": float(training_cfg["inverse_lr"])},
    ]
    if learnable and model.illumination_parameters():
        groups.append({"params": model.illumination_parameters(), "lr": float(training_cfg["illumination_lr"])})
    optimizer_s3 = torch.optim.Adam(groups)

    # Capture the (deployable, sharpened) patterns before finetuning so we can
    # show the segmentation loss actually moved the illumination.
    with torch.no_grad():
        h_before = model.microscope.pattern_generator(sigmoid_m=eval_m).detach().cpu()

    stage3_phases = _build_stage3_phases(finetune_steps, finetune_soft_m, finetune_harden_m, finetune_soft_fraction, learnable)
    print(f"[stage3] end-to-end finetuning ({finetune_steps} steps; phases={stage3_phases})...", flush=True)
    stage3_grad_norms = _empty_grad_norms()
    stage3_history: list[dict[str, float]] = []
    for phase_m, phase_steps in stage3_phases:
        phase = _train_segmentation_phase(
            model, train_loader, optimizer_s3, device,
            phase_name=f"stage3_finetune_all_m{phase_m:g}", max_steps=phase_steps, sigmoid_m=phase_m,
            apply_noise=apply_noise, weights=weights, log_every=log_every, grad_clip=grad_clip,
        )
        _merge_grad_norms(stage3_grad_norms, phase["grad_norms"])
        for row in phase["history"]:
            row["sigmoid_m"] = phase_m
        stage3_history.extend(phase["history"])
    stage3 = {"history": stage3_history, "grad_norms": stage3_grad_norms}

    with torch.no_grad():
        h_after = model.microscope.pattern_generator(sigmoid_m=eval_m).detach().cpu()
    pattern_delta_l2 = float(torch.norm(h_after - h_before).item())
    pattern_delta_rel = float(pattern_delta_l2 / (torch.norm(h_before).item() + 1e-8))
    print(
        f"[stage3] illumination pattern change (eval m={eval_m}): L2={pattern_delta_l2:.4e} "
        f"rel={pattern_delta_rel:.4e}",
        flush=True,
    )

    # Evidence: every finetuned group received gradient; illumination too (learnable).
    assert stage3["grad_norms"]["segmentation_head"]["max"] > 0.0, "Stage 3 seg head received no gradient"
    assert stage3["grad_norms"]["inverse_model"]["max"] > 0.0, "Stage 3 inverse model received no gradient"
    if learnable:
        assert stage3["grad_norms"]["illumination"]["max"] > 0.0, "Stage 3 illumination received no gradient"
        assert pattern_delta_l2 > 0.0, "Stage 3 illumination patterns did not change"

    torch.save({"model_state_dict": model.state_dict(), "config": config, "stage": "stage3"}, run_dir / "checkpoints" / "stage3_finetuned.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": config, "stage": "stage3"}, run_dir / "checkpoints" / "best.pt")

    # ======================= EVALUATION ================================ #
    final_t, final_val_dice = select_threshold(model, val_loader, device, sigmoid_m=eval_m, apply_noise=apply_noise, thresholds=thresholds)
    val_metrics = evaluate_segmentation(model, val_loader, device, sigmoid_m=eval_m, apply_noise=apply_noise, threshold=final_t)
    test_metrics = evaluate_segmentation(model, test_loader, device, sigmoid_m=eval_m, apply_noise=apply_noise, threshold=final_t)
    test_metrics_05 = evaluate_segmentation(model, test_loader, device, sigmoid_m=eval_m, apply_noise=apply_noise, threshold=0.5)
    print(
        f"[final] test dice={test_metrics['dice']:.4f} iou={test_metrics['iou']:.4f} "
        f"(val-selected t={final_t}); dice@0.5={test_metrics_05['dice']:.4f}",
        flush=True,
    )

    # ----- qualitative panel + learned patterns ----------------------------- #
    _save_qualitative_panel(
        model, test_loader, device, run_dir / "figures" / "qualitative_panel.png",
        sigmoid_m=eval_m, apply_noise=apply_noise, threshold=final_t, num_samples=num_qual,
    )
    with torch.no_grad():
        patterns = model.microscope.pattern_generator(sigmoid_m=eval_m).detach().cpu()
    save_patterns(patterns, run_dir, prefix="H_t_final")
    save_pattern_inspection(patterns, run_dir, prefix="H_t_final", sigmoid_m=eval_m)

    # ----- persist evidence + summary -------------------------------------- #
    stage_evidence = {
        "stage1": {
            "mode": stage1_mode if (stage1_mode == "load" and explicit_ckpt) else "train",
            "checkpoint": str(ckpt_path),
            "learnable": learnable,
        },
        "stage2_seg_head": {
            "steps": seg_steps,
            "trainable_report": stage2_report,
            "grad_norms": stage2["grad_norms"],
            "val_dice": stage2_val["dice"],
            "val_threshold": val_t2,
        },
        "stage3_finetune_all": {
            "steps": finetune_steps,
            "phases": [{"sigmoid_m": m, "steps": s} for m, s in stage3_phases],
            "trainable_report": stage3_report,
            "grad_norms": stage3["grad_norms"],
            "illumination_pattern_delta_l2": pattern_delta_l2,
            "illumination_pattern_delta_relative": pattern_delta_rel,
        },
        "loss_weights": vars(weights),
        "eval_sigmoid_m": eval_m,
        "finetune_soft_sigmoid_m": finetune_soft_m,
    }
    with (metrics_dir / "stage_evidence.json").open("w", encoding="utf-8") as handle:
        json.dump(stage_evidence, handle, indent=2)
    with (metrics_dir / "stage2_history.json").open("w", encoding="utf-8") as handle:
        json.dump(stage2["history"], handle, indent=2)
    with (metrics_dir / "stage3_history.json").open("w", encoding="utf-8") as handle:
        json.dump(stage3["history"], handle, indent=2)

    summary = {
        "run_dir": str(run_dir),
        "run_id": experiment_cfg["run_id"],
        "learnable": learnable,
        "compression": experiment_cfg.get("compression"),
        "content_aware_checkpoint": str(ckpt_path),
        "selected_threshold": final_t,
        "val_dice": val_metrics["dice"],
        "val_iou": val_metrics["iou"],
        "test_dice": test_metrics["dice"],
        "test_iou": test_metrics["iou"],
        "test_bce": test_metrics["bce"],
        "test_dice_at_0p5": test_metrics_05["dice"],
        "test_iou_at_0p5": test_metrics_05["iou"],
        "stage2_val_dice": stage2_val["dice"],
        "seg_head_steps": seg_steps,
        "finetune_steps": finetune_steps,
        "illumination_grad_norm_stage3_max": stage3["grad_norms"]["illumination"]["max"],
        "inverse_grad_norm_stage3_max": stage3["grad_norms"]["inverse_model"]["max"],
        "seg_head_grad_norm_stage2_max": stage2["grad_norms"]["segmentation_head"]["max"],
        "illumination_pattern_delta_l2": pattern_delta_l2,
        "illumination_pattern_delta_relative": pattern_delta_rel,
    }
    with (metrics_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
