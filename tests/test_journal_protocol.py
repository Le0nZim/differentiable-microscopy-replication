"""Scientific gates for the finalized, prospective journal recipes."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from workflow.catalog import jobs, protocol
from workflow.common import module, read_json, write_json
from workflow.data import defaults
from workflow.journal import select_learning_rates
from workflow.worker import execute, materialize


def resolved_job(tmp_path, stage, **criteria):
    settings = defaults(tmp_path / "data")
    settings["run"].update(device="cpu", cache_root=str(tmp_path / "cache"))
    prepared = {"mnist_root": str(tmp_path), "mnist_sha256": "synthetic",
                "bbbc022_large": str(tmp_path / "large.json"),
                "bbbc022_segmentation": str(tmp_path / "small.json"), "bbbc039": str(tmp_path / "manual.json"),
                "mcf7_manifest": str(tmp_path / "mcf.csv")}
    job = next(j for j in jobs([42], [stage]) if j["stage"] == stage and all(j.get(k) == v for k, v in criteria.items()))
    deps = {}
    for name in job["requires"]:
        directory = tmp_path / name
        write_json(directory / "selection.json", {"selected": {
            "learnable_frequency": {"learning_rate": .1}, "learnable_spatial": {"learning_rate": .01}}})
        deps[name] = str(directory)
    return job, materialize(job, settings, prepared, tmp_path / "out", deps)


@pytest.mark.parametrize("family", ["original", "rewritten"])
def test_fourier_spatial_arms_start_with_identical_masks_and_inverse_weights(tmp_path, family):
    from models.microscope import DifferentiableMicroscope
    from utils.experiment_config import sync_derived_config_fields
    models = []
    for variant in ("C", "D"):
        _, cfg = resolved_job(tmp_path, "patchmnist", family=family, variant=variant)
        cfg["dataset"]["image_size"] = 16
        cfg["pattern_generator"]["num_patterns"] = 2
        cfg["forward_model"]["downscale_factor"] = 4
        cfg = sync_derived_config_fields(cfg)
        torch.manual_seed(42)
        models.append(DifferentiableMicroscope.from_run_config(cfg))
    a, b = models
    torch.testing.assert_close(a.pattern_generator(sigmoid_m=1.), b.pattern_generator(sigmoid_m=1.), atol=2e-7, rtol=1e-6)
    assert a.inverse_model.state_dict().keys() == b.inverse_model.state_dict().keys()
    for key, value in a.inverse_model.state_dict().items():
        torch.testing.assert_close(value, b.inverse_model.state_dict()[key], rtol=0, atol=0)


def test_validation_search_trains_without_ever_constructing_test_data(tmp_path, monkeypatch):
    import training.dataloaders as data
    from utils.experiment_config import sync_derived_config_fields
    job, cfg = resolved_job(tmp_path, "patchmnist_tune", family="original", variant="C")
    cfg["dataset"]["image_size"] = 16
    cfg["pattern_generator"]["num_patterns"] = 2
    cfg["forward_model"]["downscale_factor"] = 4
    cfg["training"].update(batch_size=2, log_every=1)
    cfg["journal_phases"] = [dict(name="soft", m=1., steps=1, freeze_illum=True),
                             dict(name="final", m=2., steps=2, freeze_illum=False)]
    cfg = sync_derived_config_fields(cfg)
    touched = []
    arrays = {s: torch.rand(3, 1, 16, 16) for s in ("train", "val")}
    def dataset(_cfg, split):
        touched.append(split)
        assert split != "test", "Tuning must never read the test split"
        return arrays[split]
    monkeypatch.setattr(data, "build_dataset", dataset)
    impl = module("scripts/table03_ablation/run.py")
    original_evaluate = impl._evaluate
    validation_calls = []
    def evaluate(model, loader, *args, **kwargs):
        if loader.dataset is arrays["val"]:
            validation_calls.append(1)
            # The ineligible soft phase deliberately has the lowest score.
            return float(len(validation_calls) - 1), .5
        return original_evaluate(model, loader, *args, **kwargs)
    monkeypatch.setattr(impl, "_evaluate", evaluate)
    out = tmp_path / "candidate"
    spec = tmp_path / "job.json"
    write_json(spec, {"job": job, "config": cfg, "output": str(out), "device": "cpu", "dependencies": {}})
    execute(spec)
    score = read_json(out / "metrics/run_summary.json")
    assert score["test_mse"] is None and score["test_ssim"] is None
    assert score["best_m"] == 2.
    assert score["best_val_mse"] == 1.
    assert set(touched) == {"train", "val"}


def test_lr_selection_uses_only_a_complete_validation_grid(tmp_path):
    p = protocol()
    dependencies = {}
    for mode, grid in p["learning_rates"].items():
        for i, lr in enumerate(grid):
            directory = tmp_path / f"{mode}_{i}"
            directory.mkdir()
            (directory / "workflow_config.yaml").write_text(yaml.safe_dump({
                "evaluation": {"validation_only": True}, "pattern_generator": {"mode": mode},
                "training": {"illumination_lr": lr}}))
            write_json(directory / "metrics/run_summary.json", {"best_val_mse": float((i - 1) ** 2),
                                                                  "test_mse": None, "test_ssim": None})
            dependencies[directory.name] = str(directory)
    selected = select_learning_rates({"family": "original"}, dependencies, tmp_path / "selection")
    for mode, row in selected["selected"].items():
        assert row["learning_rate"] == p["learning_rates"][mode][1]
        assert not row["at_grid_boundary"]
    one = Path(next(iter(dependencies.values())))
    write_json(one / "metrics/run_summary.json", {"best_val_mse": 0., "test_mse": .2})
    with pytest.raises(ValueError, match="forbidden test"):
        select_learning_rates({"family": "original"}, dependencies, tmp_path / "invalid")


@pytest.mark.parametrize("kernel", [2, 3, 10, 11])
def test_pseudo_mask_uses_exact_kernel_and_zero_background(kernel):
    from scipy.ndimage import binary_closing
    from datasets.bbbc022_hoechst import make_pseudo_mask
    rng = np.random.default_rng(419)
    image = (rng.random((37, 41)) > .8).astype(np.float32)
    actual = make_pseudo_mask(torch.from_numpy(image)[None], .3, kernel)[0].numpy()
    expected = binary_closing(image >= .3, structure=np.ones((kernel, kernel)), border_value=0)
    np.testing.assert_array_equal(actual, expected)


def test_mcf7_schedule_and_backbone_comparison_are_matched(tmp_path):
    impl = module("scripts/figure08_mcf7/train.py")
    _, cfg = resolved_job(tmp_path, "mcf7", condition="wswinir")
    for epoch, expected in [(0, 1), (149, 1), (150, 1), (159, 2), (179, 3), (199, 4), (219, 5), (229, 5)]:
        m, unfrozen = impl._schedule_m(epoch, 150, [2, 3, 4, 5], 20, cfg["algorithm1"])
        assert m == expected and unfrozen == (epoch >= 150)
    for condition in ("wswinir", "wcnn256", "wcnn64"):
        _, _, up, full_loss = impl.condition_spec(cfg, condition)
        assert up == "original_locality" and full_loss
    assert cfg["training"]["inverse_lr"] == cfg["training"]["swinir_lr"]
    assert cfg["training"]["selection_metric"] == "mse"
    assert cfg["training"]["eval_sigmoid_m"] == 5.


@pytest.mark.parametrize("condition", ["wswinir", "wcnn256"])
def test_matched_mcf7_models_train_and_reload_through_renderer(tmp_path, monkeypatch, condition):
    from torch.utils.data import DataLoader
    impl = module("scripts/figure08_mcf7/train.py")
    _, cfg = resolved_job(tmp_path, "mcf7", condition=condition)
    backbone, _, up, full_loss = impl.CONDITIONS[condition]
    monkeypatch.setitem(impl.CONDITIONS, condition, (backbone, 16, up, full_loss))
    cfg["forward_model"]["downscale_factor"] = 4
    cfg["swinir"].update(embed_dim=12, depths=[1], num_heads=[3], window_size=4, use_checkpoint=False)
    cfg["training"].update(micro_batch_size=2, grad_accum=2, selection_start_epoch=0, amp_dtype="none")
    arrays = {s: torch.rand(5, 1, 16, 16) for s in ("train", "val", "test")}
    monkeypatch.setattr(impl, "_loaders", lambda *a: {s: DataLoader(x, batch_size=2) for s, x in arrays.items()})
    def stack(*args):
        return {"pixel_weight": 1., "pixel_kind": "l1", "perceptual_weight": .2,
                "perceptual": lambda a, b: (a - b).square().mean(),
                "discriminator": torch.nn.Conv2d(1, 1, 1), "gan_weight": .1,
                "gan_loss": lambda logits, real: (logits - float(real)).square().mean()}
    monkeypatch.setattr(impl, "build_loss_stack", stack)
    out = tmp_path / "models" / condition
    result = impl.train(cfg, condition, torch.device("cpu"), epochs=1, baseline=0, step=1,
                        max_steps_per_epoch=None, seed=81, out_dir=out, n_train=5, n_val=5,
                        n_test=5, val_subset=None, n_examples=1, gan_warmup_epochs=0)
    assert result["history"][0]["train_images"] == 5 and result["full_loss"]
    assert result["checkpoint_selection"]["selection"] == "min_val_mse"
    renderer = module("scripts/figure08_mcf7/reproduce_fig8.py")
    monkeypatch.setattr(renderer, "TR", impl)
    model, size, _ = renderer._load_model(condition, cfg, out.parent, torch.device("cpu"))
    assert size == 16
    assert torch.isfinite(renderer._recon(model, arrays["test"][:2], cfg["training"]["eval_sigmoid_m"])).all()


def test_sr_metrics_cover_borders_weight_images_and_count_acquisitions(tmp_path):
    from baselines.swinir.am4_table2 import eval_dataset_fair
    Image.fromarray(np.full((17, 19), 51, np.uint8)).save(tmp_path / "small.png")
    Image.fromarray(np.full((31, 35), 204, np.uint8)).save(tmp_path / "large.png")
    class Black(torch.nn.Module):
        num_patterns, downscale = 4, 4
        def forward(self, x, **kwargs):
            return {"x_recon": torch.zeros_like(x)}
    result = eval_dataset_fair(Black(), tmp_path, patch_size=16, device=torch.device("cpu"),
                               learnable=False, eval_sigmoid_m=8, metric_scope="image", include_borders=True)
    assert result["mse"] == pytest.approx((.2**2 + .8**2) / 2, abs=1e-6)
    assert result["images"] == 2 and result["tiles"] == 10
    for row in result["per_image"]:
        assert row["evaluated_pixels"] == row["height"] * row["width"]
        assert row["scalar_measurements"] == row["tiles"] * 64
        assert row["effective_compression"] == row["evaluated_pixels"] / row["scalar_measurements"]


def test_segmentation_defaults_to_manual_annotations_without_pseudo_parameters(tmp_path):
    _, cfg = resolved_job(tmp_path, "segmentation", mode="learnable_frequency")
    ds = cfg["dataset"]
    assert ds["name"] == "bbbc039" and ds["label_source"] == "BBBC039 manual nucleus annotations"
    assert ds["num_train_images"] == 100 and ds["num_val_images"] == ds["num_test_images"] == 50
    assert not any(k.startswith("mask_") for k in ds)
    assert not {"bias", "clip_max", "mask_raw_threshold"} & ds.keys()
    assert cfg["training"]["task_aware"]["stage1_mode"] == "train"


def test_evaluation_rng_repeats_draws_and_preserves_training_stream():
    from utils.evaluation_mode import evaluation_rng
    torch.manual_seed(735)
    before = torch.get_rng_state()
    with evaluation_rng(11, "cpu"):
        a = torch.rand(20)
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
    with evaluation_rng(11, "cpu"):
        torch.testing.assert_close(a, torch.rand(20), rtol=0, atol=0)
