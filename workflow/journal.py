"""Scientific decisions for the journal campaign, applied after legacy templates.

The final resolved YAML is saved with each job. No test score chooses a recipe.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

from .catalog import protocol
from .common import read_json, write_json


def configure(cfg, job, dependencies):
    p = protocol()
    stage = job["stage"].removesuffix("_tune")
    family = job.get("family", p["architecture"])
    cfg["workflow"].update(protocol=p["id"], architecture_family=family,
                           tuning=bool(job.get("tuning")), claims=p["claims"])
    cfg["deviations_from_paper"] = [
        "Prospective journal_v1; see docs/JOURNAL_PROTOCOL.md for all scientific choices.",
        f"Declared {family} architecture family with repaired optimization/evaluation; original U2OS data unavailable.",
    ]
    inv = cfg.get("inverse_model")
    if inv is not None:
        up = inv.setdefault("upsampling", {})
        if family == "original":
            up["mode"] = "original_transpose" if up.get("mode") in {"transpose_conv", "original_transpose"} else "original_locality"
            inv.setdefault("reconstruction", {})["hidden_channels"] = list(p["decoder_widths"])
    pg = cfg.get("pattern_generator")
    if pg:
        pg["random_fixed_m"] = 8.0
    tr = cfg.get("training", {})
    ds = cfg.get("dataset", {})
    if stage in {"content", "ablation", "segmentation"}:
        ds.update(name="bbbc022_preproc_ablation", preproc_mode=p["bbbc022"]["preprocessing"],
                  q_low=p["bbbc022"]["q_low"], q_high=p["bbbc022"]["q_high"],
                  epoch_varying_train_crops=True)
        # These are simulated photon units on a declared normalized specimen.
        cfg["detector_noise"].update(mode="differentiable_poisson_plus_read", noise_normalization="paper_v3",
                                     apply_noise=True, **{k: p["bbbc022"][k] for k in ("photon_count", "gamma", "sigma_read")})
    if stage in {"patchmnist", "ablation"}:
        cfg["detector_noise"].update(mode="differentiable_poisson_plus_read", noise_normalization="paper_v3",
                                     apply_noise=True, photon_count=10000., gamma=10., sigma_read=0.)
        cfg["architecture_family"] = family
        cfg["evaluation"] = {"validation_only": bool(job.get("tuning")),
                             "noise_seed": 8675309, "selection_final_phase_only": True}
        cfg["journal_phases"] = copy.deepcopy(p["ablation_phases"])
        tr["refresh_loader_each_pass"] = True
        tr["gradient_clip_norm"] = None  # raw parameter norms differ across FFT coordinates
        if job.get("tuning"):
            tr["illumination_lr"] = float(job["lr"])
        elif dependencies:
            selection = read_json(Path(next(iter(dependencies.values()))) / "selection.json")
            mode = cfg["pattern_generator"]["mode"]
            tr["illumination_lr"] = selection["selected"][mode]["learning_rate"]
            cfg["workflow"]["lr_selection"] = selection
    if stage == "patchmnist_content":
        ds.update(image_size=256, num_train=3000, num_val=375, num_test=375)
        pg.update(num_patterns=4, mode=job["mode"])
        cfg["forward_model"]["downscale_factor"] = job["downscale"]
        cfg["detector_noise"].update(mode="differentiable_poisson_plus_read", noise_normalization="paper_v3",
                                     apply_noise=True, photon_count=10000., gamma=10., sigma_read=0.)
        tr.update(learn_patterns=job["mode"].startswith("learnable"), max_steps=10100,
                  use_staged_hardening=job["mode"].startswith("learnable"), fixed_sigmoid_m=8.,
                  log_every=200, batch_size=32, illumination_lr=1., inverse_lr=.001,
                  staged_hardening={"inverse_warmup_steps": 1500, "joint_soft_steps": 5600,
                                    "harden_m_values": [2, 4, 8], "harden_steps_per_m": 1000})
    if stage in {"content", "patchmnist_content", "noise"}:
        # Fixed and learned arms use the same phase boundaries, optimizer-state
        # restoration and final-phase checkpoint eligibility as well as budget.
        tr["use_staged_hardening"] = True
        tr["validation_noise_seed"] = 105013
        tr["test_noise_seeds"] = [101, 202, 303, 404, 505]
    if stage == "segmentation":
        ds.update(return_mask=True, canonical_mask_mode="minimal_percentile", mask_before_crop=True,
                  mask_threshold=p["segmentation"]["threshold"], mask_closing_kernel=p["segmentation"]["closing_kernel"])
        for key in ("preprocessing_mode", "bias", "clip_max", "mask_mode", "mask_raw_threshold",
                    "mask_smooth_interval", "mask_dp_epsilon"):
            ds.pop(key, None)
        task = tr["task_aware"]
        task.update(eval_sigmoid_m=8., finetune_sigmoid_m=1.)
        task["stage1"]["matched_phases"] = True
        tr["validation_noise_seed"] = 105013
    if stage == "sr":
        cfg["model"]["upsampling_mode"] = "original_locality"
        cfg["eval"].update(metric_scope="image", include_borders=True, stitched=True)
    if stage == "mcf7":
        m = p["mcf7"]
        cfg["algorithm1"] = {k: m[k] for k in ("epochs", "epoch_baseline", "epoch_cutoff", "epoch_step", "m_init")}
        cfg["algorithm1"].update(schedule="cumulative", m_values=[2, 3, 4, 5])
        tr.update(eval_sigmoid_m=m["eval_m"], selection_metric=m["selection_metric"],
                  selection_start_epoch=m["selection_start_epoch"], inverse_lr=m["inverse_lr"],
                  swinir_lr=m["inverse_lr"], gan_warmup_epochs=57)
        cfg["matched_loss"] = True
    if stage == "controlled" and dependencies:
        selection = read_json(Path(next(iter(dependencies.values()))) / "selection.json")
        tr["illumination_lr"] = selection["selected"][job["mode"]]["learning_rate"]
        cfg["workflow"]["lr_source"] = "Transferred from soft-mask PatchMNIST validation search; binary coordinate ranking is secondary."
    return cfg


def repeated_noise_evaluation(cfg, out):
    """Reload selected weights and publish the mean of five held-out draws."""
    import statistics
    import torch
    from evaluation.checkpoint_protocol import checkpoint_sigmoid_m
    from evaluation.eval_reconstruction import evaluate_reconstruction
    from models.microscope import DifferentiableMicroscope
    from training.dataloaders import build_dataloader
    from utils.evaluation_mode import evaluation_rng

    out = Path(out)
    device = torch.device(cfg["experiment"]["device"])
    saved = torch.load(out / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    model = DifferentiableMicroscope.from_run_config(cfg).to(device)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    m = checkpoint_sigmoid_m(saved)
    loader = build_dataloader(cfg, "test")
    rows = []
    for seed in cfg["training"]["test_noise_seeds"]:
        with evaluation_rng(seed, device):
            mse, ssim = evaluate_reconstruction(model, loader, device, apply_noise=True, sigmoid_m=m)
        rows.append({"noise_seed": seed, "mse": mse, "ssim": ssim})
    report = {"sigmoid_m": m, "draws": rows, "aggregation": "mean across declared detector-noise draws",
              "mean_mse": statistics.mean(r["mse"] for r in rows),
              "mean_ssim": statistics.mean(r["ssim"] for r in rows),
              "noise_sd_mse": statistics.stdev(r["mse"] for r in rows),
              "noise_sd_ssim": statistics.stdev(r["ssim"] for r in rows)}
    write_json(out / "metrics/test_noise_draws.json", report)
    summary_path = out / "metrics/run_summary.json"
    summary = read_json(summary_path)
    summary["single_draw_diagnostic"] = {"mse": summary["test_mse"], "ssim": summary["test_ssim"]}
    summary.update(test_mse=report["mean_mse"], test_ssim=report["mean_ssim"],
                   test_noise_seeds=cfg["training"]["test_noise_seeds"], test_aggregation=report["aggregation"])
    write_json(summary_path, summary)


def select_learning_rates(job, dependencies, out):
    """Select finite validation scores and preserve the complete search evidence."""
    p = protocol()
    candidates = {mode: [] for mode in p["learning_rates"]}
    for name, directory in dependencies.items():
        root = Path(directory)
        import yaml
        cfg = yaml.safe_load((root / "workflow_config.yaml").read_text())
        result = read_json(root / "metrics/run_summary.json")
        if not cfg.get("evaluation", {}).get("validation_only"):
            raise ValueError(f"LR candidate was not validation-only: {name}")
        if any(result.get(k) is not None for k in ("test_mse", "test_ssim")):
            raise ValueError(f"LR candidate contains forbidden test evaluation: {name}")
        mode = cfg["pattern_generator"]["mode"]
        score = float(result["best_val_mse"])
        if not math.isfinite(score):
            raise ValueError(f"Nonfinite validation score: {name}")
        candidates[mode].append({"job": name, "learning_rate": float(cfg["training"]["illumination_lr"]),
                                 "validation_mse": score})
    selected = {}
    for mode, rows in candidates.items():
        grid = p["learning_rates"][mode]
        if sorted(r["learning_rate"] for r in rows) != sorted(grid):
            raise ValueError(f"Incomplete or duplicate learning-rate grid for {mode}")
        winner = min(rows, key=lambda r: (r["validation_mse"], r["learning_rate"]))
        selected[mode] = {**winner, "at_grid_boundary": winner["learning_rate"] in (min(grid), max(grid))}
    result = {"protocol": p["id"], "architecture_family": job["family"], "tuning_seed": p["tuning_seed"],
              "criterion": "minimum validation MSE; ties choose smaller LR; no test access",
              "selected": selected, "candidates": candidates}
    write_json(Path(out) / "selection.json", result)
    return result
