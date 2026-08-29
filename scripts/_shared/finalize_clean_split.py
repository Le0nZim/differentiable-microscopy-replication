#!/usr/bin/env python3
"""Aggregate clean-split reruns and write *_fixed figure/table folders."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REPL = ROOT
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from models.microscope import DifferentiableMicroscope  # noqa: E402
from training.dataloaders import build_dataloader  # noqa: E402
from utils.experiment_config import load_experiment_config  # noqa: E402
from utils.logging import save_measurement_grid  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402

BUILD = ROOT / "paper/_build_components.py"
NOISE_OUT = ROOT / "experiments/table01_noise_robustness"
SWIN_OUT = ROOT / "experiments/table02_swinir_sr/full"
FIG_DIR = ROOT / "paper"
TAB_DIR = ROOT / "paper"
PY = sys.executable

PAPER_T1 = {
    (10.0, 0.0): {"learnable": 0.0025, "fixed": 0.0108},
    (10.0, 2.7): {"learnable": 0.0024, "fixed": 0.0107},
    (10.0, 2.0): {"learnable": 0.0024, "fixed": 0.0107},
    (10.0, 6.0): {"learnable": 0.0024, "fixed": 0.0107},
    (10000.0, 0.0): {"learnable": 0.0059, "fixed": 0.0214},
    (10000.0, 2.7): {"learnable": 0.0058, "fixed": 0.0210},
    (10000.0, 2.0): {"learnable": 0.0061, "fixed": 0.0213},
    (10000.0, 6.0): {"learnable": 0.0069, "fixed": 0.0235},
}
PAPER_T2 = {
    "Set5": {"wo": (14.03, 0.3079), "wi": (26.74, 0.8113)},
    "Set14": {"wo": (13.64, 0.2258), "wi": (23.60, 0.6930)},
    "BSD100": {"wo": (14.28, 0.2094), "wi": (22.90, 0.6317)},
    "Urban100": {"wo": (13.51, 0.2146), "wi": (21.51, 0.6402)},
    "Manga109": {"wo": (12.09, 0.1952), "wi": (20.18, 0.6652)},
}


def _load_build():
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_components", BUILD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _status(rel: float) -> str:
    if rel is None:
        return "mismatch"
    a = abs(rel)
    if a <= 0.05:
        return "aligned"
    if a <= 1.0:
        return "close"
    return "mismatch"


def render_fig06_test_panels() -> Path:
    """Save first TEST-batch GT / fixed / learnable panels for the extreme cell."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fixed_id = "patchmnist_noise_random_fixed_pc10_sr6p0_seed42"
    learn_id = "patchmnist_noise_learnable_frequency_pc10_sr6p0_seed42"
    fig_dir = NOISE_OUT / "fig06_test_panels"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def load_model(run_id: str):
        cfg = load_experiment_config(NOISE_OUT / run_id / "config.yaml")
        model = DifferentiableMicroscope.from_run_config(cfg).to(device)
        img = int(cfg["dataset"]["image_size"])
        with torch.no_grad():
            model.forward_model._ensure_psfs(torch.zeros(1, 1, img, img, device=device))
        ckpt = torch.load(NOISE_OUT / run_id / "checkpoints" / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, cfg

    fixed, fcfg = load_model(fixed_id)
    learn, lcfg = load_model(learn_id)
    loader = build_dataloader(fcfg, "test")
    batch = next(iter(loader)).to(device)
    set_seed(7)
    with torch.no_grad():
        y_fixed = fixed(batch, sigmoid_m=fcfg["training"].get("fixed_sigmoid_m"), apply_noise=True)
        y_learn = learn(batch, sigmoid_m=float(lcfg["training"].get("sharpen_eval_m", 10.0)), apply_noise=True)
    save_measurement_grid(batch, fig_dir / "ground_truth.png")
    save_measurement_grid(y_fixed["x_recon"], fig_dir / "reconstruction_fixed.png")
    save_measurement_grid(y_learn["x_recon"], fig_dir / "reconstruction_learnable.png")
    return fig_dir


def write_fig06(B) -> None:
    d = FIG_DIR / "figure06_noise_robustness"
    src_dir = render_fig06_test_panels()
    srcs = {
        "A_ground_truth": src_dir / "ground_truth.png",
        "B_fixed_random": src_dir / "reconstruction_fixed.png",
        "C_our_method": src_dir / "reconstruction_learnable.png",
    }
    for name, src in srcs.items():
        gray = np.asarray(Image.open(src).convert("L").crop((2, 2, 258, 258)), dtype=np.float32)
        lo, hi = float(gray.min()), float(gray.max())
        B.image_svg(
            B.to_rgb((gray - lo) / max(hi - lo, 1e-6), "viridis", 0, 1),
            d / f"images/{name}.svg",
            source=str(src.relative_to(ROOT)),
            note="First TEST-split PatchMNIST sample; viridis display; clean val/test digit pools.",
        )
    for lab in ("A", "B", "C"):
        B.text_svg(d / f"labels/panel_{lab}.svg", lab, width=80, height=80, size=54, weight="bold", anchor="middle")
    B.scale_bars(d)
    (d / "SPLIT.md").write_text(
        "Clean split: MNIST train for train; disjoint halves of MNIST test for val vs test.\n"
        "Qualitative panels are the first TEST batch (not val).\n",
        encoding="utf-8",
    )


def write_fig07(B, meta: dict) -> None:
    d = FIG_DIR / "figure07_swinir_sr"
    base = SWIN_OUT / "full_image_eval"
    for ds, entry in meta["datasets"].items():
        for suffix, cond in (("GT", "ground_truth"), ("woLI", "swinir_without_li"), ("withLI", "swinir_with_li")):
            img = Image.open(base / f"{ds}_{suffix}.png").convert("L")
            B.image_svg(img, d / f"images/{ds}/{cond}.svg", source=str((base / f"{ds}_{suffix}.png").relative_to(ROOT)))
        for key, lab in (("swinir_wo_li", "without_LI"), ("swinir_with_li", "with_LI")):
            m = entry["conditions"][key]["metrics_vs_gt"]
            B.text_svg(
                d / f"metric_labels/{ds}_{lab}.svg",
                f"PSNR {m['psnr']:.2f} / SSIM {m['ssim']:.3f}",
                width=420, height=52, size=20, anchor="middle",
            )
    for lab, _color in (("Ground Truth", "#555555"), ("SwinIR w/o LI", "#d62728"), ("SwinIR with LI", "#00a65a")):
        B.text_svg(
            d / f"labels/{lab.lower().replace(' ', '_').replace('/', '')}.svg",
            lab, width=420, height=54, size=24, weight="bold", anchor="middle",
        )
    B.border_svg(d / "symbols/with_li_green_border.svg", "#00a65a")
    B.border_svg(d / "symbols/without_li_red_border.svg", "#d62728")
    B.line_svg(d / "symbols/with_li_green_dashed.svg", "#00a65a", dashed=True)
    B.line_svg(d / "symbols/without_li_red_dashed.svg", "#d62728", dashed=True)
    B.scale_bars(d)
    (d / "SPLIT.md").write_text(
        "Clean split: Flickr2K HR-only (no x2/x3/x4) and scene-level 2% val holdout.\n",
        encoding="utf-8",
    )


def write_table01() -> None:
    payload = json.loads((NOISE_OUT / "table1_v3_results.json").read_text())
    table = payload["table_seed42"]
    d = TAB_DIR / "table01_noise_robustness"
    (d / "rendered").mkdir(parents=True, exist_ok=True)
    (d / "components").mkdir(parents=True, exist_ok=True)
    shutil.copy2(NOISE_OUT / "table1_v3_results.md", d / "components/ours_table1_v3_results.md")

    our_rows = []
    cmp_rows = []
    md = [
        "# Table 1 comparison — Robustness to Poisson and read noise (PatchMNIST, clean split)\n",
        "\nStatus: **clean-split rerun**\n",
        "\n| Metric | Paper | Ours | Abs diff | Rel diff | Status | Note |\n",
        "|---|---|---|---|---|---|---|\n",
    ]
    for cell in table:
        pc, sr = float(cell["photon_count"]), float(cell["sigma_read"])
        for method, key, ours_key in (
            ("learnable (Our)", "learnable", "learnable_mse"),
            ("fixed_random", "fixed", "random_mse"),
        ):
            ours = float(cell[ours_key])
            paper = PAPER_T1[(pc, sr)][key]
            adiff = ours - paper
            rdiff = adiff / paper if paper else None
            st = _status(rdiff)
            our_rows.append({"method": method, "photon_count": pc, "sigma_read": sr, "our_mse": f"{ours:.4f}"})
            metric = f"MSE {method.split()[0]} pc={pc:g} sigma={sr}"
            cmp_rows.append({
                "metric": metric,
                "paper_value": paper,
                "our_value": f"{ours:.4f}",
                "absolute_difference": f"{adiff:.4f}",
                "relative_difference": f"{rdiff:.4f}" if rdiff is not None else "",
                "status": st,
                "note": "clean val/test digit pools",
            })
            rel = f"{rdiff:.4f}" if rdiff is not None else ""
            md.append(
                f"| {metric} | {paper} | {ours:.4f} | {adiff:.4f} | "
                f"{rel} | {st} | clean split |\n"
            )
    wins = all(c["learnable_wins"] for c in table)
    md.append(
        f"| Claim N6: learnable beats fixed at every cell | yes | "
        f"{'yes (all 8 cells)' if wins else 'NO'} |  |  | "
        f"{'aligned' if wins else 'mismatch'} | Clean-split rerun. |\n"
    )
    with (d / "our_values.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["method", "photon_count", "sigma_read", "our_mse"])
        w.writeheader()
        w.writerows(our_rows)
    with (d / "comparison.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(cmp_rows[0].keys()))
        w.writeheader()
        w.writerows(cmp_rows)
    src_paper = TAB_DIR / "table01_noise_robustness/paper_values.csv"
    if src_paper.exists() and src_paper.resolve() != (d / "paper_values.csv").resolve():
        shutil.copy2(src_paper, d / "paper_values.csv")
    (d / "rendered/table01_comparison.md").write_text("".join(md), encoding="utf-8")
    (d / "comparison.md").write_text("".join(md), encoding="utf-8")
    (d / "README.md").write_text(
        "# Table 1 (clean split)\n\n"
        "Same noise recipe as `table01_noise_robustness` with disjoint MNIST-test "
        "digit pools for val vs test.\n",
        encoding="utf-8",
    )


def write_table02(summary: dict) -> None:
    d = TAB_DIR / "table02_swinir_sr"
    (d / "rendered").mkdir(parents=True, exist_ok=True)
    (d / "components").mkdir(parents=True, exist_ok=True)
    src_report = SWIN_OUT / "full_run_table.md"
    if src_report.exists():
        shutil.copy2(src_report, d / "components/ours_AM4_resolution_report.md")
    our_rows = []
    cmp_rows = []
    md = [
        "# Table 2 comparison — Learnable illumination + SwinIR (clean split)\n",
        "\nStatus: **clean-split rerun**\n",
        "\n| Metric | Paper | Ours | Abs diff | Rel diff | Status | Note |\n",
        "|---|---|---|---|---|---|---|\n",
    ]
    for ds, c in summary["comparison"].items():
        pairs = [
            ("PSNR w/o-LI", PAPER_T2[ds]["wo"][0], c["wo_li_psnr"], ".2f"),
            ("PSNR with-LI", PAPER_T2[ds]["wi"][0], c["with_li_psnr"], ".2f"),
            ("SSIM w/o-LI", PAPER_T2[ds]["wo"][1], c["wo_li_ssim"], ".4f"),
            ("SSIM with-LI", PAPER_T2[ds]["wi"][1], c["with_li_ssim"], ".4f"),
        ]
        our_rows.append({"dataset": ds, "condition": "SwinIR w/o LI", "psnr": f"{c['wo_li_psnr']:.2f}", "ssim": f"{c['wo_li_ssim']:.4f}"})
        our_rows.append({"dataset": ds, "condition": "SwinIR with LI", "psnr": f"{c['with_li_psnr']:.2f}", "ssim": f"{c['with_li_ssim']:.4f}"})
        for name, paper, ours, fmt in pairs:
            adiff = ours - paper
            rdiff = adiff / paper if paper else None
            st = _status(rdiff)
            metric = f"{name} {ds}"
            cmp_rows.append({
                "metric": metric, "paper_value": paper, "our_value": format(ours, fmt),
                "absolute_difference": f"{adiff:.4f}",
                "relative_difference": f"{rdiff:.4f}" if rdiff is not None else "",
                "status": st, "note": "HR-only Flickr2K + scene val split",
            })
            rel = f"{rdiff:.4f}" if rdiff is not None else ""
            md.append(
                f"| {metric} | {paper} | {format(ours, fmt)} | {adiff:.4f} | "
                f"{rel} | {st} | clean split |\n"
            )
    with (d / "our_values.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["dataset", "condition", "psnr", "ssim"])
        w.writeheader()
        w.writerows(our_rows)
    with (d / "comparison.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(cmp_rows[0].keys()))
        w.writeheader()
        w.writerows(cmp_rows)
    src_paper = TAB_DIR / "table02_swinir_sr/paper_values.csv"
    if src_paper.exists() and src_paper.resolve() != (d / "paper_values.csv").resolve():
        shutil.copy2(src_paper, d / "paper_values.csv")
    (d / "rendered/table02_comparison.md").write_text("".join(md), encoding="utf-8")
    (d / "comparison.md").write_text("".join(md), encoding="utf-8")
    (d / "README.md").write_text(
        "# Table 2 (clean split)\n\n"
        "Same AM-4 SwinIR recipe as `table02_swinir_sr` with Flickr2K HR-only "
        "files and a scene-level validation holdout.\n",
        encoding="utf-8",
    )


def render_fig07_full_images() -> dict:
    """Run tiled full-image eval into the SwinIR run dir, then return metadata."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fig7_full",
        ROOT / "scripts/table02_swinir_sr/render_full_image_eval.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # Point the renderer at the clean-split run.
    spec.loader.exec_module(mod)
    mod.CONFIG = REPL / "configs/table02_swinir_sr/full.yaml"
    mod.RUN = SWIN_OUT
    mod.OUT = SWIN_OUT / "full_image_eval"
    mod.CONDITIONS = [
        ("swinir_wo_li", False, SWIN_OUT / "swinir_wo_li/checkpoints/best.pt", "woLI", "SwinIR w/o LI"),
        ("swinir_with_li", True, SWIN_OUT / "swinir_with_li/checkpoints/best.pt", "withLI", "SwinIR with LI"),
    ]
    mod.main()
    return json.loads((mod.OUT / "metadata.json").read_text())


def main() -> None:
    print("Finalizing clean-split Fig.6 / Fig.7 / Table 1 / Table 2 ...", flush=True)
    B = _load_build()
    write_fig06(B)
    print("Wrote figure06_noise_robustness", flush=True)
    write_table01()
    print("Wrote table01_noise_robustness", flush=True)
    meta = render_fig07_full_images()
    write_fig07(B, meta)
    print("Wrote figure07_swinir_sr", flush=True)
    summary = json.loads((SWIN_OUT / "aggregate_summary.json").read_text())
    write_table02(summary)
    print("Wrote table02_swinir_sr", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
