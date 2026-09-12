#!/usr/bin/env python3
"""Execute narrowly selected upstream definitions on tiny diagnostic examples.

Requires the reviewed upstream commit, not microscopy data or its dependencies.
No upstream module-level dataset downloads or training notebooks are executed.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.locality_upsampling import OriginalLocalityUpsampling

UPSTREAM = "f8cc74b51847bd31b320a7f153dcf3eedecc3e7a"


def definition(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class Mask(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = nn.Parameter(torch.tensor(0.5))
    def forward(self, m):
        return self.h


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.d = nn.Parameter(torch.tensor(0.5))
    def forward(self, y, **kwargs):
        return y * self.d


class Forward:
    def compute_yt(self, x, h):
        return x*h, x*h


class Upsampler(nn.Module):
    def __init__(self, bn=False):
        super().__init__()
        self.net = nn.BatchNorm2d(1) if bn else nn.Identity()
    def forward(self, y, **kwargs):
        return self.net(y)


def run(original):
    commit = subprocess.check_output(["git", "-C", str(original), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM:
        raise ValueError(f"Expected reviewed commit {UPSTREAM}, got {commit}")
    metric = lambda a, b, **kw: float(((a-b)**2).mean().detach())
    loop_path = original / "modules/train_utils.py"
    loop = definition(loop_path, "loop", {"torch": torch, "np": np,
                      "ssim_ignite": metric, "mse_distance": metric})
    decoder, mask, up = Decoder(), Mask(), Upsampler()
    opt = (torch.optim.SGD(decoder.parameters(), lr=0.1), torch.optim.SGD(mask.parameters(), lr=0.1))
    loader = [(torch.ones(2, 1, 4, 4), torch.zeros(2))]
    with contextlib.redirect_stdout(io.StringIO()):
        loop("cpu", loader, decoder, up, Forward(), mask, nn.MSELoss(), opt,
             type_="train", losses=[], epoch=1, train_model_iter=1, train_H_iter=1,
             metrics={k: [] for k in ("ssim11", "ssim5", "mse")},
             connect_forward_inverse=lambda *a: 0)
    first_grad = 2*(0.5*0.5-1)*0.5
    new_decoder = 0.5 - 0.1*first_grad
    second_grad = 2*(new_decoder*0.5-1)*new_decoder
    expected_accumulated = 0.5 - 0.1*(first_grad + second_grad)
    expected_alternating = 0.5 - 0.1*second_grad
    assert math.isclose(float(mask.h.detach()), expected_accumulated, abs_tol=1e-6)
    up_bn = Upsampler(bn=True).train()
    before = copy.deepcopy(up_bn.state_dict())
    with contextlib.redirect_stdout(io.StringIO()):
        loop("cpu", loader, Decoder(), up_bn, Forward(), Mask(), nn.MSELoss(), opt,
             type_="test", losses=[], epoch=1,
             metrics={k: [] for k in ("ssim11", "ssim5", "mse")},
             connect_forward_inverse=lambda *a: 0)
    assert up_bn.net.num_batches_tracked == before["net.num_batches_tracked"] + 1

    # Independent tensor-layout equivalence check against actual custom_v2.
    path = original / "modules/models/decoder_upsampling_nets.py"
    upstream_cls = definition(path, "custom_v2", {"torch": torch, "nn": nn, "math": math})
    torch.manual_seed(713)
    old = upstream_cls(lambda_scale_factor=3, T=4, recon_img_size=16,
                       init_method="xavier_normal", custom_upsampling_bias=True,
                       upsample_postproc_block=nn.Identity())
    new = OriginalLocalityUpsampling(4, 4, 4, 4)
    with torch.no_grad():
        new.weights.copy_(old.weights)
        new.bias.copy_(old.biases[:, 0])
    x = torch.randn(3, 4, 4, 4)
    torch.testing.assert_close(new.project(x), old(x))

    # Repeated source-index shuffles, exactly as save_grids resets its RNG.
    source = original / "create_patchMNIST_dataset.py"
    save_node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "save_grids")
    # Execute only the two real statements that seed and reorder source indices.
    nodes = [n for n in save_node.body if isinstance(n, (ast.Expr, ast.Assign))
             and ("np.random.seed" in ast.unparse(n) or "data[np.random.choice" in ast.unparse(n))]
    def order():
        env = {"np": np, "data": np.arange(60000)}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), env)
        return env["data"]
    assert np.array_equal(order(), order())
    return {
        "upstream_commit": commit,
        "source_sha256": {str(p.relative_to(original)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (loop_path, path, source)},
        "gradient_accumulation": {"observed_H_after_step": float(mask.h.detach()),
                                  "sum_of_two_gradients_prediction": expected_accumulated,
                                  "proper_alternating_prediction": expected_alternating},
        "validation_batchnorm": {"upsampler_still_training": up_bn.training,
                                 "num_batches_tracked_increment": 1,
                                 "running_mean_after": float(up_bn.net.running_mean)},
        "custom_v2_projection_matches_new_original_locality": True,
        "repeated_training_source_order_identical": True,
        "limitation": "Tiny deterministic diagnostics, not an estimate of published performance bias.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    result = run(args.original_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
