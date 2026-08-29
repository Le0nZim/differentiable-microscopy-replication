#!/usr/bin/env python3
"""MCF7 LI+SwinIR launcher (tmux-safe).

The old offline-refinement runner was removed during repo cleanup.
Supported path: scripts/run_mcf7_li_swinir_paper_direct.py

By default re-launches in a detached tmux session (survives closing Cursor).
Use --foreground to run in the current shell.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIRECT = ROOT / "scripts/run_mcf7_li_swinir_paper_direct.py"
DEFAULT_LOG = ROOT / "experiments/swinir_or_highres/mcf7_paper_direct_full/run.log"
LAUNCHER = ROOT / "scripts/launch_in_tmux.sh"


def _in_tmux() -> bool:
    return os.environ.get("TMUX") is not None


def _gpu_index(device: str) -> str:
    if device.rsplit(":", 1)[-1] == "1":
        return "1"
    return "0"


def _build_paper_direct_cmd(args: argparse.Namespace, *, device: str) -> list[str]:
    cmd = [sys.executable, "-u", str(PAPER_DIRECT), "--device", device, "--seed", str(args.seed)]
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.full_budget:
        cmd.append("--full-budget")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="MCF7 LI+SwinIR launcher (defaults to detached tmux)")
    parser.add_argument("--foreground", action="store_true", help="Run in current shell (no tmux)")
    parser.add_argument("--tmux-session", default="mcf7_li_swinir", help="tmux session name")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--full-budget", action="store_true")
    parser.add_argument("--launch-full-approved", action="store_true")
    parser.add_argument("--offline-refinement", action="store_true")
    parser.add_argument("--refinement-mode", default=None)
    parser.add_argument("--swinir-steps", type=int, default=None)
    parser.add_argument("--swinir-lr", type=float, default=None)
    parser.add_argument("--swinir-loss", default=None)
    args, extra = parser.parse_known_args()

    legacy = any(
        [
            args.launch_full_approved,
            args.offline_refinement,
            args.refinement_mode is not None,
            args.swinir_steps is not None,
            args.swinir_lr is not None,
            args.swinir_loss is not None,
        ]
    )
    if legacy:
        print(
            "ERROR: offline-refinement flags are no longer supported (scripts removed).\n"
            f"Use paper-direct: python {PAPER_DIRECT} --device {args.device}\n"
            f"Or: python scripts/run_mcf7_li_swinir.py --device {args.device}",
            file=sys.stderr,
        )
        sys.exit(2)

    cmd = _build_paper_direct_cmd(args, device=args.device)
    cmd.extend(extra)

    if args.foreground or _in_tmux():
        os.execv(cmd[0], cmd)

    gpu = _gpu_index(args.device)
    # Inside tmux only one GPU is visible; always use cuda:0 there.
    tmux_cmd = _build_paper_direct_cmd(args, device="cuda:0")
    tmux_cmd.extend(extra)

    launch = [str(LAUNCHER), args.tmux_session, gpu, str(args.log_file), *tmux_cmd]
    subprocess.run(launch, check=True)


if __name__ == "__main__":
    main()
