"""Reproducible image-weighted evaluation for the audited protocol."""

from __future__ import annotations

import copy
import torch

from evaluation.metrics import mse, ssim, psnr


@torch.no_grad()
def evaluate_controlled(model, loader, device, *, sigmoid_m, noise_seed,
                        noise_mode=None, ssim_padding="reflect"):
    modes = [(m, m.training) for m in model.modules()]
    noise_config = copy.deepcopy(model.detector_noise.config)
    generator_state = loader.generator.get_state() if loader.generator is not None else None
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    scores = []
    try:
        model.eval()
        if noise_mode is not None:
            model.detector_noise.config.mode = noise_mode
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(noise_seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(noise_seed)
            for batch in loader:
                x = batch.to(device) if torch.is_tensor(batch) else batch[0].to(device)
                result = model(x, sigmoid_m=sigmoid_m)
                for pred, target in zip(result["x_recon"], x):
                    p, t = pred[None], target[None]
                    scores.append([float(mse(p, t)), float(ssim(p, t, padding_mode=ssim_padding)),
                                   float(psnr(p, t))])
    finally:
        model.detector_noise.config = noise_config
        for module, training in modes:
            module.training = training
        if generator_state is not None:
            loader.generator.set_state(generator_state)
    if not scores:
        raise ValueError("Cannot evaluate an empty dataset")
    mean = torch.tensor(scores, dtype=torch.float64).mean(dim=0).tolist()
    return {"mse": mean[0], "ssim": mean[1], "psnr_mean_per_image": mean[2],
            "num_images": len(scores), "noise_seed": noise_seed,
            "ssim_padding": ssim_padding, "data_range": 1.0,
            "aggregation": "mean over images", "per_image": scores}
