#!/usr/bin/env python3
"""Run prospective controlled experiments; never reuse historical result folders.

No model ordering is a success criterion. --smoke is synthetic software
validation only, not a microscopy or PatchMNIST performance experiment.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import torch
from torch.utils.data import DataLoader
from evaluation.controlled import evaluate_controlled
from models.acquisition import acquisition_metadata
from models.microscope import DifferentiableMicroscope
from training.dataloaders import build_dataloader
from training.iteration import repeat_dataloader
from training.train_reconstruction import build_optimizer
from utils.experiment_config import load_experiment_config, sync_derived_config_fields
from utils.reproducibility import set_seed, get_git_commit_hash

MODES = ("uniform_all_ones", "random_fixed", "hadamard_fixed", "learnable_spatial", "learnable_frequency")


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def implementation_hash():
    """Include uncommitted source changes, which a git HEAD string cannot identify."""
    digest = hashlib.sha256()
    for path in sorted([Path(__file__).resolve(), *(ROOT / "src").rglob("*.py")]):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def dataset_evidence(loader):
    """Fingerprint the evaluated tensors, not an unaudited split label."""
    digest = hashlib.sha256()
    for x in loader.dataset:
        x = x[0] if isinstance(x, (tuple, list)) else x
        digest.update(x.detach().cpu().contiguous().numpy().tobytes())
    return {"num_images": len(loader.dataset), "tensor_sha256": digest.hexdigest()}


def smoke_loaders(config):
    # Deliberately unequal final batches also exercise metric weighting.
    g = torch.Generator().manual_seed(713)
    arrays = [torch.rand(n, 1, 16, 16, generator=g) for n in (9, 5, 7)]
    return tuple(DataLoader(x, batch_size=4, shuffle=i == 0,
                            generator=torch.Generator().manual_seed(314 + i))
                 for i, x in enumerate(arrays))


def run_one(base, *, mode, seed, output, device, smoke=False):
    cfg = copy.deepcopy(base)
    cfg["experiment"].update(seed=seed, device=str(device), run_id=f"{mode}_seed{seed}")
    cfg["pattern_generator"].update(mode=mode, seed=seed, binarization="ste", superpixel_factor=1)
    cfg["training"].update(loader_seed=seed + 1000, learn_patterns=mode.startswith("learnable"))
    cfg["pattern_generator"]["hadamard_tile_size"] = cfg["forward_model"]["downscale_factor"]
    if smoke:
        cfg["dataset"]["image_size"] = 16
        cfg["forward_model"]["downscale_factor"] = 4
        cfg["pattern_generator"]["hadamard_tile_size"] = 4
        cfg["inverse_model"]["reconstruction"]["hidden_channels"] = [4, 4, 4, 4, 4, 1]
        cfg["training"].update(max_steps=8, warmup_steps=2, log_every=4, batch_size=4)
    cfg = sync_derived_config_fields(cfg)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite/reuse {output}; choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(cfg, indent=2))
    set_seed(seed)
    model = DifferentiableMicroscope.from_run_config(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    loaders = smoke_loaders(cfg) if smoke else tuple(build_dataloader(cfg, s) for s in ("train", "val", "test"))
    train_loader, val_loader, test_loader = loaders
    data = {"val": dataset_evidence(val_loader), "test": dataset_evidence(test_loader)}
    initial = {"inverse_state_sha256": state_hash(model.inverse_model),
               "binary_patterns_sha256": tensor_hash(model.pattern_generator().detach()),
               "data": data}
    train_iter = repeat_dataloader(train_loader)
    steps, warmup = int(cfg["training"]["max_steps"]), int(cfg["training"]["warmup_steps"])
    if not 0 <= warmup < steps:
        raise ValueError("warmup_steps must be smaller than max_steps")
    best_score = float("inf")
    best_state = None
    history = []
    for step in range(1, steps + 1):
        # Binary masks in every forward; m affects only the STE surrogate slope.
        m = 1.0 if step <= warmup else min(8.0, 1.0 + 7.0*(step - warmup)/(steps - warmup))
        model.pattern_generator.sigmoid_m = m
        model.set_illumination_trainable(mode.startswith("learnable") and step > warmup)
        model.train()
        x = next(train_iter)
        x = x.to(device) if torch.is_tensor(x) else x[0].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x, sigmoid_m=m)
        loss = torch.nn.functional.l1_loss(outputs["x_recon"], x)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        loss.backward()
        for param in model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise RuntimeError(f"Nonfinite gradient at step {step}")
        optimizer.step()
        if step % int(cfg["training"]["log_every"]) == 0 or step == steps:
            val = evaluate_controlled(model, val_loader, device, sigmoid_m=m, noise_seed=201,
                                      noise_mode="exact_poisson_plus_read")
            history.append({"step": step, "train_l1": float(loss.detach()), "val_mse": val["mse"], "m": m})
            print(f"{mode} seed={seed} step={step}/{steps} val_mse={val['mse']:.6g}", flush=True)
            if val["mse"] < best_score:
                best_score = val["mse"]
                best_state = {"model_state_dict": copy.deepcopy(model.state_dict()),
                              "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                              "sigmoid_m": m, "step": step, "config": cfg,
                              "selection": "minimum validation MSE, binary masks, exact Poisson plus read noise",
                              "resume_exact": False, "checkpoint_schema": 2}
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg, "step": steps, "sigmoid_m": m, "resume_exact": False}, output / "last.pt")
    torch.save(best_state, output / "best.pt")
    model.load_state_dict(best_state["model_state_dict"], strict=True)
    model.eval()
    m = best_state["sigmoid_m"]
    model.pattern_generator.sigmoid_m = m
    draws = []
    for noise_seed in (301, 302, 303):
        draws.append(evaluate_controlled(model, test_loader, device, sigmoid_m=m, noise_seed=noise_seed,
                                         noise_mode="exact_poisson_plus_read"))
    approximation = evaluate_controlled(model, test_loader, device, sigmoid_m=m, noise_seed=301,
                                       noise_mode="differentiable_poisson_plus_read")
    patterns = model.pattern_generator().detach()
    acquisition = acquisition_metadata(patterns, downscale=cfg["forward_model"]["downscale_factor"],
                                       photon_scale=cfg["detector_noise"]["photon_count"],
                                       dose_mode=cfg["forward_model"]["dose_mode"],
                                       sequence_dose=cfg["forward_model"]["sequence_dose"])
    torch.save(patterns.cpu(), output / "evaluated_patterns.pt")
    metrics = {k: sum(d[k] for d in draws) / len(draws) for k in ("mse", "ssim", "psnr_mean_per_image")}
    result = {"mode": mode, "seed": seed, "smoke_only": smoke, "steps_executed": steps,
              "selected_step": best_state["step"], "sigmoid_m": m, "test": metrics,
              "test_noise_draws": draws, "gaussian_approximation_diagnostic": approximation,
              "initial": initial, "acquisition": acquisition, "history": history,
              "git_commit": get_git_commit_hash(), "torch_version": torch.__version__,
              "implementation_sha256": implementation_hash(),
              "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
              "test_inference_scope": "Held out from gradient training; confirmatory status additionally requires a test set not previously used for design decisions.",
              "checkpoint_sha256": hashlib.sha256((output / "best.pt").read_bytes()).hexdigest(),
              "numerical_claim": "Synthetic software smoke test only" if smoke else "Prospective controlled experiment; not a reproduction of unavailable U2OS data"}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/audit/controlled_patchmnist.yaml"))
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--downscale", type=int)
    p.add_argument("--num-patterns", type=int)
    p.add_argument("--photon-count", type=float)
    p.add_argument("--sigma-read", type=float)
    p.add_argument("--dose-mode", choices=["fixed_peak", "equal_mean"])
    p.add_argument("--upsampling", choices=["locality_aware", "transpose_conv", "original_locality", "original_transpose"])
    args = p.parse_args()
    cfg = load_experiment_config(args.config)
    for section, key, value in (
        ("forward_model", "downscale_factor", args.downscale),
        ("pattern_generator", "num_patterns", args.num_patterns),
        ("detector_noise", "photon_count", args.photon_count),
        ("detector_noise", "sigma_read", args.sigma_read),
        ("forward_model", "dose_mode", args.dose_mode),
    ):
        if value is not None:
            cfg[section][key] = value
    if args.upsampling is not None:
        cfg["inverse_model"]["upsampling"]["mode"] = args.upsampling
    results = []
    for seed in args.seeds:
        for mode in args.modes:
            results.append(run_one(cfg, mode=mode, seed=seed,
                                   output=Path(args.output) / f"{mode}_seed{seed}",
                                   device=torch.device(args.device), smoke=args.smoke))
    # No cross-seed standard deviation is a confidence interval; keep raw paired
    # results and evaluation-noise draws available separately.
    (Path(args.output) / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
