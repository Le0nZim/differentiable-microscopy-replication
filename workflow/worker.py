"""Run exactly one job in a fresh process, using a fully materialized config."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

from .common import ROOT, module, read_json, write_json


def materialize(job, settings, prepared, out, dependencies):
    import yaml
    sys.path.insert(0, str(ROOT / "src")) if str(ROOT / "src") not in sys.path else None
    from utils.experiment_config import load_experiment_config, sync_derived_config_fields

    stage, seed = job["stage"], job["seed"]
    device = settings["run"]["device"]
    templates = {
        "patchmnist": "figure10_ablation_patchmnist/ablation.yaml",
        "ablation": "figure10_ablation/ablation.yaml",
        "upsampling": "figure05_upsampling/base.yaml",
        "noise": "table01_noise_robustness/noise_table.yaml",
        "segmentation": "figure04_segmentation/task_aware.yaml",
        "sr": "table02_swinir_sr/full.yaml",
        "mcf7": "figure08_mcf7/swinir_fix.yaml",
        "content_swinir": "figure03_content_aware/paper_faithful_pixel_perceptual_gan.yaml",
        "controlled": "audit/controlled_patchmnist.yaml",
    }
    if stage == "content":
        cfg = module("scripts/figure03_content_aware/train_base.py").build_config(job["comp"], job["downscale"], 4, job["mode"], seed, device)
    elif stage in {"sr", "mcf7", "content_swinir", "controlled"}:
        cfg = yaml.safe_load((ROOT / "configs" / templates[stage]).read_text())
    else:
        cfg = load_experiment_config(ROOT / "configs" / templates[stage])

    # Only these machine paths and declared controls override experiment recipes.
    cfg.setdefault("experiment", {}).update(seed=seed, device=device, run_id=job["id"],
        output_dir=str(out), results_csv=str(out / "results.csv"))
    cfg["workflow"] = {"job": job, "dataset_identity": "BBBC022 substitute" if stage in {"content", "content_swinir", "segmentation", "ablation"} else stage,
                       "protocol": "binary_equal_mean_audit" if stage == "controlled" else "audited_legacy_soft_mask",
                       "note": "Fresh rerun; not evidence that previous numerical results are valid."}
    ds = cfg.get("dataset", {})
    ds["seed"] = settings["run"]["data_seed"]
    if "pattern_generator" in cfg:
        cfg["pattern_generator"]["seed"] = seed
    if "training" in cfg:
        cfg["training"]["loader_seed"] = seed + 1000

    if stage in {"patchmnist", "upsampling", "noise", "controlled"}:
        ds.update(data_root=prepared["mnist_root"], download=False, disjoint_val_test=True,
                  cache_dir=str(Path(settings["run"]["cache_root"]) / "patchmnist"), source_sha256=prepared["mnist_sha256"])
    if stage in {"content", "ablation"}:
        ds.update(data_root=settings["data"]["bbbc022"], repo_root=str(ROOT),
                  split_path=prepared["bbbc022_large"])
    if stage in {"patchmnist", "ablation"}:
        cfg = module("scripts/table03_ablation/run.py").apply_variant(cfg, job["variant"])
        cfg["training"]["refresh_loader_each_pass"] = True
    elif stage == "upsampling":
        ds.update(image_size=job["size"], num_train=job["count"])
        cfg["pattern_generator"].update(mode="random_fixed", num_patterns=8)
        cfg["inverse_model"]["upsampling"]["mode"] = job["up"]
        cfg["training"].update(learn_patterns=False, use_staged_hardening=False,
                               fixed_sigmoid_m=10., max_steps=4000, log_every=200)
    elif stage == "noise":
        cfg["pattern_generator"]["mode"] = job["mode"]
        cfg["detector_noise"].update(photon_count=job["photons"], sigma_read=job["sigma"])
        cfg = module("scripts/table01_noise_robustness/run.py")._configure_run_kind(cfg)
    elif stage == "segmentation":
        cfg["dataset"].update(data_root=settings["data"]["bbbc022"], split_path=prepared["bbbc022_segmentation"])
        cfg["pattern_generator"]["mode"] = job["mode"]
        cfg["forward_model"]["downscale_factor"] = job["downscale"]
        cfg["training"]["learn_patterns"] = job["mode"].startswith("learnable")
        task = cfg["training"]["task_aware"]
        task.update(stage1_mode="train", content_aware_checkpoint=None)
        phases = task["stage1"]["learnable"]
        task["stage1"]["fixed_steps"] = phases["inverse_warmup_steps"] + phases["joint_soft_steps"] + len(phases["harden_m_values"]) * phases["harden_steps_per_m"]
        cfg["experiment"]["study_label"] = "Fresh three-stage segmentation; BBBC022 pseudo-labels, nested well split"
    elif stage == "content_swinir":
        cfg.update(seed=seed, base_exp_root=str(out.parent), base_run_dir=dependencies[job["requires"][0]])
    elif stage == "sr":
        cfg["data"].update(train_roots=list(settings["data"]["sr_train"].values()),
                           split_manifest=prepared["sr_split"], test_roots=prepared["sr_test_roots"])
        # Forking after model/PyTorch initialization can deadlock worker thread pools.
        cfg["training"]["multiprocessing_context"] = "spawn"
        cfg["training"]["num_workers"] = settings["run"]["num_workers"]
        cfg["training"]["eval_num_workers"] = min(2, settings["run"]["num_workers"])
    elif stage == "mcf7":
        cfg["dataset"].update(data_root=settings["data"]["mcf7_images"], images_dir=".",
                               manifest_csv=prepared["mcf7_manifest"], bbbc021_channel="tubulin")
        cfg["data_seed"] = settings["run"]["data_seed"]
    if stage not in {"sr", "mcf7", "content_swinir", "controlled"}:
        cfg = sync_derived_config_fields(cfg)
    return cfg


def _invoke(relative, args):
    subprocess.run([sys.executable, str(ROOT / relative), *map(str, args)], cwd=ROOT, check=True)


def execute(spec_path):
    import torch
    import yaml
    spec = read_json(spec_path)
    job, cfg = spec["job"], spec["config"]
    out = Path(spec["output"])
    device = torch.device(spec["device"])
    sys.path.insert(0, str(ROOT / "src")) if str(ROOT / "src") not in sys.path else None
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "workflow_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    engine = job["engine"]
    if engine == "cnn":
        from training.train_reconstruction import train
        from training.staged_hardening_train import train_staged_hardening
        fn = train_staged_hardening if cfg["training"].get("use_staged_hardening") else train
        fn(cfg, str(out))
        artifacts = ["metrics/run_summary.json", "checkpoints/best.pt", "config.yaml"]
    elif engine == "ablation":
        impl = module("scripts/table03_ablation/run.py")
        impl.run_one(cfg, out, letter=job["variant"], phases=impl.default_phases(), seed=job["seed"],
                     log_every=cfg["training"]["log_every"], save_config_yaml=True)
        artifacts = ["metrics/run_summary.json", "checkpoints_best.pt", "figures/qualitative_panel.png"]
    elif engine == "segmentation":
        from training.train_task_aware_segmentation import train_task_aware_segmentation
        train_task_aware_segmentation(cfg, out)
        artifacts = ["metrics/run_summary.json", "checkpoints/best.pt"]
    elif engine == "refiner":
        impl = module("scripts/figure03_content_aware/train_swinir.py")
        impl.train_cell(cfg, job["comp"], job["mode"], device, out)
        prefix = f"{job['comp']}/{job['mode']}"
        artifacts = [prefix + "/metrics.json", prefix + "/checkpoints/best.pt", prefix + "/grids/test_grid.png"]
    elif engine == "sr":
        impl = module("scripts/table02_swinir_sr/run.py")
        condition = next(c for c in cfg["training"]["conditions"] if c["name"] == job["condition"])
        impl.train_condition(cfg, condition, device, out, resume=False, iterations_override=None)
        artifacts = [f"{job['condition']}/result.json", f"{job['condition']}/checkpoints/best.pt"]
    elif engine == "mcf7":
        impl = module("scripts/figure08_mcf7/train.py")
        schedule = cfg["algorithm1"]
        impl.train(cfg, job["condition"], device, epochs=schedule["epochs"], baseline=schedule["epoch_baseline"],
                   step=schedule["epoch_step"], max_steps_per_epoch=None, seed=job["seed"], out_dir=out,
                   n_train=cfg["dataset"]["num_train"], n_val=cfg["dataset"]["num_val"], n_test=cfg["dataset"]["num_test"],
                   val_subset=None, n_examples=8)
        artifacts = ["result.json", "checkpoints/best.pt", "examples/pair_00.pt"]
    elif engine == "mcf7_figures":
        view = out / "models"
        view.mkdir()
        for name, source in spec["dependencies"].items():
            condition = name.split("/")[1].rsplit("_seed", 1)[0]
            (view / condition).symlink_to(source, target_is_directory=True)
        for figure in (8, 9):
            template = yaml.safe_load((ROOT / f"configs/figure08_mcf7/reproduce_fig{figure}_tubulin.yaml").read_text())
            for key in ("dataset", "pattern_generator", "forward_model", "swinir", "training", "data_seed"):
                template[key] = copy.deepcopy(cfg[key])
            template["reproduce"]["runs_dir"] = str(view)
            p = out / f"figure{figure}.yaml"
            p.write_text(yaml.safe_dump(template))
            extra = ["--indices", "0", "1", "2"] if figure == 8 else []
            _invoke(f"scripts/figure08_mcf7/reproduce_fig{figure}.py", ["--config", p, "--device", device, "--out", out, *extra])
        artifacts = ["figure8_paper_style.png", "figure9_paper_style.png", "figure8_metrics.json", "figure9_metrics.json"]
    elif engine == "controlled":
        _invoke("scripts/audit/run_controlled.py", ["--config", config_path, "--device", device,
                "--seeds", job["seed"], "--modes", job["mode"], "--output", out / "controlled"])
        # Audit driver refuses pre-existing mode directories and writes per-model evidence.
        prefix = f"controlled/{job['mode']}_seed{job['seed']}"
        artifacts = ["controlled/results.json", prefix + "/result.json", prefix + "/best.pt"]
    else:
        raise ValueError(f"Unknown worker engine {engine}")
    if not artifacts:
        raise RuntimeError(f"No completion artifacts for {job['id']}")
    for relative in artifacts:
        p = out / relative
        if not p.is_file() or p.stat().st_size == 0:
            raise RuntimeError(f"Worker returned without required artifact: {p}")
    write_json(out / "worker_complete.json", {"artifacts": artifacts})


if __name__ == "__main__":
    execute(sys.argv[1])
