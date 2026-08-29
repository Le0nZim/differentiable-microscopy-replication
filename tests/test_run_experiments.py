"""Tests for experiment matrix runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.run_experiments import run_matrix


def test_run_matrix_dry_run():
    summaries = run_matrix("configs/_shared/patchmnist_x8_matrix_debug.yaml", dry_run=True)
    assert len(summaries) == 2
    assert summaries[0]["compression"] == 16.0
    assert "pattern_mode" in summaries[0]


@pytest.mark.slow
def test_run_matrix_smoke(tmp_path: Path):
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        Path("configs/_shared/patchmnist_x8_matrix_debug.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    base_path = tmp_path / "base_patchmnist.yaml"
    base_path.write_text(
        Path("configs/_shared/base_patchmnist.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    matrix_content = matrix_path.read_text(encoding="utf-8").replace(
        "results_smoke.csv",
        str(tmp_path / "results.csv"),
    )
    matrix_path.write_text(matrix_content, encoding="utf-8")

    summaries = run_matrix(matrix_path, dry_run=False)
    assert len(summaries) == 2
    assert Path(summaries[0]["run_dir"]).exists()
    print(json.dumps(summaries, indent=2))
