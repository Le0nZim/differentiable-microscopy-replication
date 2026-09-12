"""Smoke test for the training loop."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.train_reconstruction import train
from utils.experiment_config import load_experiment_config, sync_derived_config_fields


def test_training_smoke(tmp_path: Path, monkeypatch):
    # Exercise the real epoch/checkpoint/export path without 3,000 MNIST images,
    # a CUDA dependency, or missing derived configuration fields.
    import training.train_reconstruction as trainer
    config = load_experiment_config("configs/_shared/base_patchmnist.yaml")
    config["experiment"]["device"] = "cpu"
    config["experiment"]["run_id"] = "audit_epoch_smoke"
    config["dataset"]["image_size"] = 16
    config["pattern_generator"]["num_patterns"] = 2
    config["forward_model"]["downscale_factor"] = 4
    config["inverse_model"]["reconstruction"]["hidden_channels"] = [4, 4, 4, 4, 4, 1]
    config["training"]["batch_size"] = 2
    config = sync_derived_config_fields(config)
    g = torch.Generator().manual_seed(413)
    arrays = {s: torch.rand(n, 1, 16, 16, generator=g) for s, n in (("train", 6), ("val", 3), ("test", 3))}
    monkeypatch.setattr(trainer, "build_dataloader", lambda c, s: DataLoader(arrays[s], batch_size=2))
    config["experiment"]["output_dir"] = str(tmp_path / "run")
    config["experiment"]["results_csv"] = str(tmp_path / "results.csv")
    config["training"]["num_epochs"] = 1

    summary = train(config, config["experiment"]["output_dir"])

    run_dir = Path(summary["run_dir"])
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "metrics" / "training_history.json").exists()
    assert (run_dir / "figures" / "reconstruction_batch0.png").exists()
    assert (run_dir / "learned_patterns" / "patterns.png").exists()
    assert Path(config["experiment"]["results_csv"]).exists()
