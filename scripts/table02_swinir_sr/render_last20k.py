#!/usr/bin/env python3
"""Render Fig. 7 from the clean-split *last* checkpoints (step 20,000).

The paper/figure07_swinir_sr folder used best.pt, which for
with-LI is the 15,000-step val checkpoint. This writes a sibling folder from
last.pt (20,000 steps for both conditions).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REPL = ROOT
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SWIN_OUT = ROOT / "experiments/table02_swinir_sr/full"
EVAL_OUT = SWIN_OUT / "full_image_eval_last20k"
FIG_DIR = ROOT / "paper/figure07_swinir_sr_last20k"
BUILD = ROOT / "paper/_build_components.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render() -> dict:
    mod = _load("fig7_full", ROOT / "scripts/table02_swinir_sr/render_full_image_eval.py")
    mod.CONFIG = REPL / "configs/table02_swinir_sr/full.yaml"
    mod.RUN = SWIN_OUT
    mod.OUT = EVAL_OUT
    mod.CONDITIONS = [
        ("swinir_wo_li", False, SWIN_OUT / "swinir_wo_li/checkpoints/last.pt", "woLI", "SwinIR w/o LI"),
        ("swinir_with_li", True, SWIN_OUT / "swinir_with_li/checkpoints/last.pt", "withLI", "SwinIR with LI"),
    ]
    mod.main()
    meta = json.loads((EVAL_OUT / "metadata.json").read_text())
    meta["checkpoint"] = "last.pt (step 20000) per condition"
    (EVAL_OUT / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def write_folder(B, meta: dict) -> None:
    d = FIG_DIR
    for ds, entry in meta["datasets"].items():
        for suffix, cond in (("GT", "ground_truth"), ("woLI", "swinir_without_li"), ("withLI", "swinir_with_li")):
            img = Image.open(EVAL_OUT / f"{ds}_{suffix}.png").convert("L")
            B.image_svg(img, d / f"images/{ds}/{cond}.svg", source=str((EVAL_OUT / f"{ds}_{suffix}.png").relative_to(ROOT)))
        for key, lab in (("swinir_wo_li", "without_LI"), ("swinir_with_li", "with_LI")):
            m = entry["conditions"][key]["metrics_vs_gt"]
            B.text_svg(
                d / f"metric_labels/{ds}_{lab}.svg",
                f"PSNR {m['psnr']:.2f} / SSIM {m['ssim']:.3f}",
                width=420, height=52, size=20, anchor="middle",
            )
    for lab in ("Ground Truth", "SwinIR w/o LI", "SwinIR with LI"):
        B.text_svg(
            d / f"labels/{lab.lower().replace(' ', '_').replace('/', '')}.svg",
            lab, width=420, height=54, size=24, weight="bold", anchor="middle",
        )
    B.border_svg(d / "symbols/with_li_green_border.svg", "#00a65a")
    B.border_svg(d / "symbols/without_li_red_border.svg", "#d62728")
    B.line_svg(d / "symbols/with_li_green_dashed.svg", "#00a65a", dashed=True)
    B.line_svg(d / "symbols/without_li_red_dashed.svg", "#d62728", dashed=True)
    B.scale_bars(d)
    (d / "CHECKPOINT.md").write_text(
        "Clean-split run, last.pt at 20,000 iterations (both conditions).\n"
        "Sibling folder paper/figure07_swinir_sr used best.pt, which for "
        "with-LI is the 15,000-step validation checkpoint.\n"
        "w/o-LI best.pt and last.pt are the same 20,000-step weights.\n"
        "Split: Flickr2K HR-only (no x2/x3/x4) and scene-level 2% val holdout.\n",
        encoding="utf-8",
    )
    rows = ["# Displayed-image metrics at last.pt (step 20,000)\n",
            "\n| dataset | w/o LI PSNR | w/o LI SSIM | with LI PSNR | with LI SSIM |\n",
            "| --- | --- | --- | --- | --- |\n"]
    for ds, entry in meta["datasets"].items():
        wo = entry["conditions"]["swinir_wo_li"]["metrics_vs_gt"]
        wi = entry["conditions"]["swinir_with_li"]["metrics_vs_gt"]
        rows.append(f"| {ds} | {wo['psnr']:.2f} | {wo['ssim']:.3f} | {wi['psnr']:.2f} | {wi['ssim']:.3f} |\n")
    (d / "METRICS.md").write_text("".join(rows), encoding="utf-8")
    print(f"Wrote {d}", flush=True)


def main() -> None:
    meta = render()
    B = _load("build_components", BUILD)
    write_folder(B, meta)


if __name__ == "__main__":
    main()
