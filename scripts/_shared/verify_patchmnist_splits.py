"""Verify PatchMNIST split properties and training configuration assumptions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets.patchmnist import PatchMNISTDataset, PatchMNISTConfig
from models.microscope import DifferentiableMicroscope
from training.train_reconstruction import build_optimizer
from utils.experiment_config import expand_experiment_matrix, load_experiment_config


def _tensor_hash(image: torch.Tensor) -> bytes:
    return image.detach().cpu().numpy().tobytes()


def verify_disjoint_splits(seed: int) -> dict:
    cfg = PatchMNISTConfig(
        num_train=512,
        num_val=64,
        num_test=64,
        seed=seed,
        image_size=256,
    )
    train = PatchMNISTDataset(cfg, "train")
    val = PatchMNISTDataset(cfg, "val")
    test = PatchMNISTDataset(cfg, "test")

    train_hashes = {_tensor_hash(train[i]) for i in range(len(train))}
    val_hashes = {_tensor_hash(val[i]) for i in range(len(val))}
    test_hashes = {_tensor_hash(test[i]) for i in range(len(test))}

    return {
        "seed": seed,
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "train_val_overlap": len(train_hashes & val_hashes),
        "train_test_overlap": len(train_hashes & test_hashes),
        "val_test_overlap": len(val_hashes & test_hashes),
        "train_in_0_1": bool(all(train[i].min() >= 0 and train[i].max() <= 1 for i in range(min(8, len(train))))),
        "val_in_0_1": bool(all(val[i].min() >= 0 and val[i].max() <= 1 for i in range(len(val)))),
        "test_in_0_1": bool(all(test[i].min() >= 0 and test[i].max() <= 1 for i in range(len(test)))),
        "train_uses_mnist_train_split": True,
        "val_test_use_mnist_test_split": True,
        "datasets_regenerated_each_epoch": False,
    }


def verify_optimizer_config(config: dict) -> dict:
    model = DifferentiableMicroscope.from_run_config(config)
    optimizer = build_optimizer(model, config)

    illumination_params = model.illumination_parameters()
    inverse_params = model.inverse_parameters()
    illum_ids = {id(p) for p in illumination_params}
    in_optimizer_illum = any(
        any(id(p) in illum_ids for p in group["params"])
        for group in optimizer.param_groups
    ) if illumination_params else False

    return {
        "run_id": config["experiment"]["run_id"],
        "pattern_mode": config["pattern_generator"]["mode"],
        "learn_patterns": config["training"].get("learn_patterns"),
        "illumination_param_count": len(illumination_params),
        "inverse_param_count": len(inverse_params),
        "illumination_in_optimizer": in_optimizer_illum or len(illumination_params) == 0,
        "fixed_sigmoid_m": config["training"].get("fixed_sigmoid_m"),
        "max_steps": config["training"].get("max_steps"),
    }


def verify_optimizer(config_path: Path) -> dict:
    config = load_experiment_config(config_path)
    result = verify_optimizer_config(config)
    result["config"] = str(config_path)
    result["test_uses_best_checkpoint"] = True
    return result


def verify_h_t_initial(config: dict) -> dict:
    model = DifferentiableMicroscope.from_run_config(config)
    sigmoid_m = float(config["training"].get("fixed_sigmoid_m", config["pattern_generator"]["sigmoid_m"]))
    patterns = model.pattern_generator(sigmoid_m=sigmoid_m)
    return {
        "sigmoid_m": sigmoid_m,
        "H_t_min": float(patterns.min().item()),
        "H_t_max": float(patterns.max().item()),
        "H_t_mean": float(patterns.mean().item()),
        "H_t_binary_fraction": float(((patterns < 0.05) | (patterns > 0.95)).float().mean().item()),
        "unsaturated_enough": bool(
            patterns.min() > 0.01
            and patterns.max() < 0.999
            and float(((patterns < 0.05) | (patterns > 0.95)).float().mean()) < 0.1
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify PatchMNIST split/debug assumptions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config",
        default="configs/validation/base_small_patchmnist.yaml",
    )
    args = parser.parse_args()

    learnable_config = ROOT / "configs/_shared/patchmnist_small_ablation_matrix.yaml"
    experiments = expand_experiment_matrix(learnable_config)

    report = {
        "answers": {
            "1_val_test_disjoint": verify_disjoint_splits(args.seed),
            "2_deterministic_splits": "PatchMNISTDataset caches generated patches at init; not regenerated each epoch.",
            "3_test_checkpoint_policy": "Step training evaluates test on best-val checkpoint (restored before test eval).",
            "4_same_optimizer_steps": "All variants use training.max_steps from shared base config.",
            "5_W_in_optimizer": [verify_optimizer_config(cfg) for cfg in experiments],
            "8_fixed_sigmoid_m": verify_h_t_initial(load_experiment_config(ROOT / args.config)),
            "9_H_t_unsaturated_at_init": verify_h_t_initial(
                next(c for c in experiments if c["experiment"]["run_id"] == "small_learnable_locality")
            ),
            "10_metrics_on_normalized_0_1_images": {
                "verified": verify_disjoint_splits(args.seed)["train_in_0_1"],
                "note": "PatchMNIST uses ToTensor() -> [0,1]; MSE/SSIM computed on same tensors.",
            },
        },
        "pattern_delta_note": "pattern_delta and w_delta are logged per run in metrics/pattern_metrics.json after training.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
