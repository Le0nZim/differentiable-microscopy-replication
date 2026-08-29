"""Smoke test for the training loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from training.train_reconstruction import train
from utils.config import load_yaml_config


@pytest.mark.slow
def test_training_smoke(tmp_path: Path):
    config = load_yaml_config("configs/_shared/base_patchmnist.yaml")
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
