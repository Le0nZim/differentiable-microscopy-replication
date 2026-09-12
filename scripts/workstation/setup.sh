#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
if [[ ! -e .venv/bin/python ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -e '.[dev]'
swinir_commit=6545850fbf8df298df73d81f3e8cba638787c8bd
if [[ ! -e SwinIR ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/JingyunLiang/SwinIR.git SwinIR
  git -C SwinIR checkout --detach "$swinir_commit"
elif [[ "$(git -C SwinIR rev-parse HEAD)" != "$swinir_commit" ]]; then
  echo "Existing SwinIR has a different revision; expected $swinir_commit. Move it aside, then rerun setup." >&2
  exit 1
fi
.venv/bin/python -c 'import torch, torchvision, timm; print("PyTorch:", torch.__version__, "CUDA devices:", torch.cuda.device_count())'
echo "Setup complete. Run: source .venv/bin/activate"
echo "Then: python paper.py download-weights"
