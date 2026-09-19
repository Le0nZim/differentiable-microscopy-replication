#!/usr/bin/env python3
"""Compare the original architecture ports to the pinned released definitions.

No datasets, training runs or checkpoints are read. This checks forward maps
and input gradients at copied weights, not initialization streams or training.
Run with the replication environment and an existing original source checkout.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn

ORIGINAL_SHA = "f8cc74b51847bd31b320a7f153dcf3eedecc3e7a"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def copy_layers(original, port):
    types = (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)
    old = [m for m in original.modules() if isinstance(m, types)]
    new = [m for m in port.modules() if isinstance(m, types)]
    assert len(old) == len(new)
    for source, target in zip(old, new):
        assert type(source) is type(target)
        target.load_state_dict(source.state_dict(), strict=True)


def compare(original, port, inputs):
    for training in (False, True):
        original.train(training)
        port.train(training)
        xa = inputs.detach().clone().requires_grad_()
        xb = inputs.detach().clone().requires_grad_()
        ya, yb = original(xa), port(xb)
        weights = torch.linspace(-1, 1, ya.numel()).reshape_as(ya)
        ga = torch.autograd.grad((ya * weights).sum(), xa)[0]
        gb = torch.autograd.grad((yb * weights).sum(), xb)[0]
        torch.testing.assert_close(ya, yb, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(ga, gb, rtol=2e-5, atol=2e-6)
        yield {"training": training,
               "output_max_abs": float((ya - yb).detach().abs().max()),
               "input_gradient_max_abs": float((ga - gb).abs().max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sha = subprocess.check_output(["git", "-C", str(args.original_root), "rev-parse", "HEAD"], text=True).strip()
    if sha != ORIGINAL_SHA:
        parser.error(f"Expected original commit {ORIGINAL_SHA}, found {sha}")
    dirty = subprocess.check_output(["git", "-C", str(args.original_root), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty.strip():
        parser.error("Original checkout has tracked modifications")
    sys.path.insert(0, str(args.original_root.resolve()))
    from modules.models.decoder_upsampling_nets import custom_v2, learnable_transpose_conv
    from modules.models.decoder_support_blocks import conv_bn_block
    from modules.models.decoder import genv1
    from models.locality_upsampling import (
        OriginalLocalityUpsampling, OriginalTransposeUpsampling,
        LocalityAwareUpsampling, TransposeConvUpsampling,
    )
    from models.recon_cnn import ReconCNN, ReconCNNConfig

    rows = []
    for t, d in ((1, 8), (4, 8), (8, 8), (4, 4)):
        h, hd, q = 16, 16 // d, d.bit_length()
        torch.manual_seed(812 + t + d)
        post = nn.Sequential(conv_bn_block(1, max(1, t // 2), 3, 1, 1),
                             conv_bn_block(max(1, t // 2), t, 3, 1, 1))
        old = custom_v2(lambda_scale_factor=q, T=t, recon_img_size=h,
                        init_method="xavier_normal", custom_upsampling_bias=True,
                        upsample_postproc_block=post)
        new = OriginalLocalityUpsampling(t, hd, hd, d)
        with torch.no_grad():
            new.weights.copy_(old.weights)
            new.bias.copy_(old.biases[:, 0])
        copy_layers(old, new)
        rows.extend({"module": "original_locality", "T": t, "d": d, **row}
                    for row in compare(old, new, torch.randn(2, t, hd, hd)))
        old = learnable_transpose_conv(lambda_scale_factor=q, T=t)
        new = OriginalTransposeUpsampling(t, d)
        copy_layers(old, new)
        rows.extend({"module": "original_transpose", "T": t, "d": d, **row}
                    for row in compare(old, new, torch.randn(2, t, hd, hd)))
        old = genv1(t, h, 1, [24, 12, 8, 4, 2], "sigmoid")
        new = ReconCNN(ReconCNNConfig(in_channels=t, hidden_channels=[24, 12, 8, 4, 2, 1]))
        copy_layers(old, new)
        rows.extend({"module": "original_decoder_widths", "T": t, "d": d, **row}
                    for row in compare(old, new, torch.randn(2, t, h, h)))

    models = {
        "original_locality": OriginalLocalityUpsampling(4, 32, 32, 8),
        "rewritten_locality": LocalityAwareUpsampling(4, 32, 32, 8),
        "original_transpose": OriginalTransposeUpsampling(4, 8),
        "rewritten_transpose": TransposeConvUpsampling(4, 8),
        "original_width_decoder": ReconCNN(ReconCNNConfig(in_channels=4, hidden_channels=[24, 12, 8, 4, 2, 1])),
        "rewritten_width_decoder": ReconCNN(ReconCNNConfig(in_channels=4)),
    }
    result = {
        "original_commit": sha, "torch_version": torch.__version__, "device": "cpu",
        "scope": "Copied-weight forward and input-gradient checks; not training equivalence or paper-result replay",
        "cases": len(rows),
        "max_forward_error": max(r["output_max_abs"] for r in rows),
        "max_input_gradient_error": max(r["input_gradient_max_abs"] for r in rows),
        "parameter_counts_H256_T4_d8": {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()},
        "rows": rows,
    }
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
