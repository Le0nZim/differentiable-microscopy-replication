#!/usr/bin/env python3
"""Evaluate + figures + metrics for the Fig-3 SwinIR-fix experiment.

Consumes the per-cell outputs written by ``fig3_swinir_fix_train.py`` (a
``<out-root>/<comp>/<pattern>/{metrics.json,checkpoints/best.pt}`` tree) and
produces:

  * metrics_summary.csv    (required columns; base vs SwinIR, deltas, paths)
  * figures/swinir_diagnostic_<name>.png     GT | rand base | rand+SwinIR |
                                             learn base | learn+SwinIR  (locked viridis [0,1])
  * figures/swinir_random_fields_<name>.png  same, several random test fields
  * figures/fig3_full_<name>.png             paper-style Fig-3 (A..F) with the
                                             NEW +SwinIR columns

Robust to partially-trained sweeps: cells without a checkpoint are skipped (CSV)
or fall back to the base reconstruction (full panel), with a printed warning.

Usage:
  python scripts/fig3_swinir_fix_report.py --name paper_faithful_l1_ssim --device cuda:0
  python scripts/fig3_swinir_fix_report.py --name paper_faithful_l1_ssim --device cuda:0 --full-panel
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from baselines.swinir import fig3_refine_stage as S  # noqa: E402
from baselines.swinir.refinement_model import OfflineSwinIRRefinement  # noqa: E402
from evaluation.metrics import ssim as ssim_metric  # noqa: E402

EXP = ROOT / "experiments/figure3_bbbc022_swinir_fix_v1"
BASE_ROOT = ROOT / "experiments/ablations/bbbc022_content_aware_v2"
COMPS = ["x16", "x64", "x256", "x1024"]
DOWNSCALE = {"x16": 8, "x64": 16, "x256": 32, "x1024": 64}
NUM_PATTERNS = 4
ILLUM = {"random_fixed": "pseudo_random", "learnable_frequency": "learnable"}


def load_refiner(ckpt_path: Path, device: torch.device) -> OfflineSwinIRRefinement | None:
    if not ckpt_path.exists():
        return None
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    swinir_cfg = payload["swinir_cfg"]
    ref = OfflineSwinIRRefinement(swinir_cfg, {"mode": "direct"}).to(device)
    ref.load_state_dict(payload["refiner_state_dict"])
    ref.eval()
    return ref


# ---------------------------------------------------------------------------
# metrics_summary.csv
# ---------------------------------------------------------------------------
def build_metrics_csv(out_root: Path, csv_path: Path) -> list[dict]:
    rows = []
    for comp in COMPS:
        for pattern in ("random_fixed", "learnable_frequency"):
            mp = out_root / comp / pattern / "metrics.json"
            if not mp.exists():
                continue
            m = json.loads(mp.read_text())
            t = m["test"]
            rows.append({
                "compression": comp, "downscale": DOWNSCALE[comp], "num_patterns": NUM_PATTERNS,
                "illumination": ILLUM[pattern],
                "base_ssim": round(t["base_ssim"], 4), "swinir_ssim": round(t["ref_ssim"], 4),
                "delta_ssim": round(t["ref_ssim"] - t["base_ssim"], 4),
                "base_mse": round(t["base_mse"], 6), "swinir_mse": round(t["ref_mse"], 6),
                "delta_mse": round(t["ref_mse"] - t["base_mse"], 6),
                "iterations_reached": m.get("iterations_reached"),
                "best_step": m.get("best_step"),
                "checkpoint_path": m.get("checkpoint_path", ""),
                "config_path": str((out_root / comp / pattern / "config.yaml").resolve()),
            })
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        cols = ["compression", "downscale", "num_patterns", "illumination", "base_ssim",
                "swinir_ssim", "delta_ssim", "base_mse", "swinir_mse", "delta_mse",
                "iterations_reached", "best_step", "checkpoint_path", "config_path"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] wrote {csv_path} ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['compression']:>6} {r['illumination']:<13} "
              f"SSIM {r['base_ssim']:.4f}->{r['swinir_ssim']:.4f} ({r['delta_ssim']:+.4f}) | "
              f"MSE {r['base_mse']:.5f}->{r['swinir_mse']:.5f} ({r['delta_mse']:+.6f})")
    return rows


# ---------------------------------------------------------------------------
# SwinIR diagnostic panels (locked viridis [0,1], fixed field + random fields)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _fields_for_cell(out_root: Path, comp: str, pattern: str, device, indices: list[int]):
    """Return list of (gt, base, refined) numpy arrays for the given test indices."""
    base_model, run_cfg, eval_m = S.load_frozen_base(BASE_ROOT, comp, pattern, device)
    cache = S.build_pair_cache(base_model, run_cfg, "test", device, eval_m, crop_size=256, base_batch=8, seed=42)
    del base_model
    torch.cuda.empty_cache()
    ref = load_refiner(out_root / comp / pattern / "checkpoints" / "best.pt", device)
    has_ref = ref is not None
    fields = []
    for idx in indices:
        idx = min(idx, len(cache) - 1)
        gt = cache.gt[idx:idx+1].to(device, torch.float32)
        xb = cache.x_base[idx:idx+1].to(device, torch.float32)
        rec = ref(xb).clamp(0, 1) if ref is not None else xb.clamp(0, 1)
        fields.append((gt[0, 0].cpu().numpy(), xb.clamp(0, 1)[0, 0].cpu().numpy(), rec[0, 0].cpu().numpy()))
    del ref
    torch.cuda.empty_cache()
    return fields, has_ref


def render_diagnostic(out_root: Path, name: str, device, fixed_index: int, out_dir: Path):
    """rows=comps, cols=[GT, rand base, rand+SwinIR, learn base, learn+SwinIR]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["GT", "pseudo-random\nbase", "pseudo-random\n+ SwinIR", "learnable\nbase", "learnable\n+ SwinIR"]
    fig, axes = plt.subplots(len(COMPS), 5, figsize=(2.15 * 5, 2.15 * len(COMPS)))
    have = {}
    for r, comp in enumerate(COMPS):
        rf, has_r = _fields_for_cell(out_root, comp, "random_fixed", device, [fixed_index])
        lf, has_l = _fields_for_cell(out_root, comp, "learnable_frequency", device, [fixed_index])
        have[comp] = (has_r, has_l)
        gt, rb, rr = rf[0]
        _, lb, lr = lf[0]
        panels = [gt, rb, rr, lb, lr]
        for c, img in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(comp, fontsize=12, weight="bold")
    fig.suptitle(f"Fig-3 SwinIR diagnostic ({name}) — locked viridis [0,1], test field #{fixed_index}", fontsize=12)
    fig.tight_layout()
    p = out_dir / f"swinir_diagnostic_{name}.png"
    fig.savefig(p, dpi=150, facecolor="white"); plt.close(fig)
    print(f"[diagnostic] wrote {p}  (refiner present per comp: {have})")
    return p


def render_random_fields(out_root: Path, name: str, device, comp: str, indices: list[int], out_dir: Path):
    """For one compression, several random test fields x {rand,learn} base/SwinIR."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rf, _ = _fields_for_cell(out_root, comp, "random_fixed", device, indices)
    lf, _ = _fields_for_cell(out_root, comp, "learnable_frequency", device, indices)
    n = len(indices)
    rows = ["GT", "rand base", "rand+SwinIR", "learn base", "learn+SwinIR"]
    fig, axes = plt.subplots(5, n, figsize=(2.1 * n, 2.1 * 5))
    if n == 1:
        axes = axes.reshape(5, 1)
    for j in range(n):
        gt, rb, rr = rf[j]
        _, lb, lr = lf[j]
        for r, img in enumerate([gt, rb, rr, lb, lr]):
            ax = axes[r, j]
            ax.imshow(img, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(rows[r], fontsize=10)
            if r == 0:
                ax.set_title(f"test #{indices[j]}", fontsize=9)
    fig.suptitle(f"Fig-3 SwinIR random test fields ({name}, {comp}) — locked viridis [0,1]", fontsize=12)
    fig.tight_layout()
    p = out_dir / f"swinir_random_fields_{name}_{comp}.png"
    fig.savefig(p, dpi=150, facecolor="white"); plt.close(fig)
    print(f"[random_fields] wrote {p}")
    return p


# ---------------------------------------------------------------------------
# Full paper-style Fig 3 (A..F) with the NEW +SwinIR columns
# ---------------------------------------------------------------------------
def render_full_panel(out_root: Path, name: str, device, fixed_index: int, out_dir: Path):
    import render_fig03_content_aware as R  # reuse the validated panel renderers
    from models.microscope import DifferentiableMicroscope
    from utils.experiment_config import load_experiment_config
    from training.dataloaders import build_dataloader

    comp_dir = out_dir / f"components_{name}"; comp_dir.mkdir(parents=True, exist_ok=True)
    # common test specimen (fixed index) via the base x16 hadamard config
    ref_cfg = load_experiment_config(R.run_dir(BASE_ROOT, "x16", "hadamard_fixed") / "config.yaml")
    ref_cfg["experiment"]["device"] = str(device)
    image_size = ref_cfg["dataset"]["image_size"]
    loader = build_dataloader(ref_cfg, "test")
    specimen, chosen = R.select_test_specimen(loader, 65.0, fixed_index)
    specimen = specimen.to(device)
    print(f"[full] common test specimen index={chosen}")

    metrics = R.read_metrics(BASE_ROOT)  # base 4-method metrics for D/E
    swinir_metrics = {}
    recon_grid, detect_grid = {}, {}
    learned_patterns, fixed_patterns = {}, {}
    for comp, _ds in R.COMPS:
        for mkey, _lab, _mk, _c in R.METHODS:
            cfg = load_experiment_config(R.run_dir(BASE_ROOT, comp, mkey) / "config.yaml")
            cfg["experiment"]["device"] = str(device)
            model = R.load_model(cfg, R.run_dir(BASE_ROOT, comp, mkey) / "checkpoints" / "best.pt", device, image_size)
            with torch.no_grad():
                out = model(specimen, sigmoid_m=10.0, apply_noise=False)
            recon = out["x_recon"][0, 0].clamp(0, 1).cpu().numpy()
            s = float(ssim_metric(out["x_recon"].clamp(0, 1), specimen).item())
            recon_grid[(comp, mkey)] = {"img": recon, "ssim": s}
            detect_grid[(comp, mkey)] = {"img": out["y_down"][0].mean(dim=0).cpu().numpy()}
            if mkey == "learnable_frequency":
                p_all = out["patterns"][:, 0].cpu().numpy()
                learned_patterns[comp] = p_all[int(np.argmax(p_all.reshape(p_all.shape[0], -1).var(axis=1)))]
            if comp == "x16" and mkey in ("hadamard_fixed", "uniform_all_ones", "random_fixed"):
                p_all = out["patterns"][:, 0].cpu().numpy()
                fixed_patterns[mkey] = p_all[int(np.argmax(p_all.reshape(p_all.shape[0], -1).var(axis=1)))] if mkey == "hadamard_fixed" else p_all[0]
            if mkey in ("random_fixed", "learnable_frequency"):
                eval_m = S.EVAL_M[mkey]
                ref = load_refiner(out_root / comp / mkey / "checkpoints" / "best.pt", device)
                with torch.no_grad():
                    base_out = model(specimen, sigmoid_m=eval_m, apply_noise=False)["x_recon"]
                    rrec_t = ref(base_out).clamp(0, 1) if ref is not None else base_out.clamp(0, 1)
                s_sw = float(ssim_metric(rrec_t, specimen).item())
                recon_grid[(comp, f"{mkey}_sw")] = {"img": rrec_t[0, 0].cpu().numpy(), "ssim": s_sw}
                mm = out_root / comp / mkey / "metrics.json"
                if mm.exists():
                    md = json.loads(mm.read_text())["test"]
                    swinir_metrics[(comp, mkey)] = {"ssim": md["ref_ssim"], "mse": md["ref_mse"]}
                if ref is None:
                    print(f"[full] WARNING missing refiner {comp}/{mkey}; +SwinIR col = base")
                else:
                    del ref
            del model
            torch.cuda.empty_cache()

    gt = specimen[0, 0].cpu().numpy()
    R.render_grid(recon_grid, comp_dir / "panel_A_reconstructions.png", columns=R.PANEL_A_COLS,
                  title=f"A  Reconstructions (same field; last 2 cols = +SwinIR, {name})",
                  normalize_each=False, annotate_ssim=True)
    R.render_grid(detect_grid, comp_dir / "panel_B_detections.png",
                  columns=[(k, lab, mk, c) for k, lab, mk, c in R.METHODS],
                  title="B  Detections (normalized by max of field)", normalize_each=True)
    R.render_patterns([fixed_patterns["hadamard_fixed"], fixed_patterns["uniform_all_ones"], fixed_patterns["random_fixed"]],
                      ["Hadamard", "All ones", "Pseudo-random"], comp_dir / "panel_C1_fixed_patterns.png",
                      title="C1  Fixed illumination patterns", vertical=False, fixed_range=True)
    R.render_patterns([learned_patterns[c] for c, _ in R.COMPS],
                      [f"downscale {DOWNSCALE[c]}x{DOWNSCALE[c]}" for c, _ in R.COMPS],
                      comp_dir / "panel_C2_learned_patterns.png", title="C2  Learned patterns", vertical=True)
    R.render_scatter(metrics, comp_dir / "panel_D_ssim.png", key="ssim", ylabel="SSIM",
                     title="D  SSIM vs compression", swinir_metrics=swinir_metrics)
    R.render_scatter(metrics, comp_dir / "panel_E_mse.png", key="mse", ylabel="MSE",
                     title="E  MSE vs compression", swinir_metrics=swinir_metrics)
    R.render_single(gt, comp_dir / "panel_F_ground_truth.png", title="F  Ground truth")
    R.render_legend(comp_dir / "panel_legend.png")
    p = out_dir / f"fig3_full_{name}.png"
    R.compose(comp_dir, p)
    print(f"[full] wrote {p}")
    return p


def update_report_table(rows: list[dict], name: str, report_path: Path):
    """Fill the LIVE RESULTS block of REPORT.md from the CSV rows + acceptance verdicts."""
    if not report_path.exists() or not rows:
        # never clobber the in-progress placeholder when no cells have finished yet
        if not rows:
            print("[report] no trained cells yet; leaving REPORT.md LIVE RESULTS placeholder intact")
        return
    order = {"x16": 0, "x64": 1, "x256": 2, "x1024": 3}
    rows = sorted(rows, key=lambda r: (order.get(r["compression"], 9), r["illumination"]))
    lines = [f"_Source: `{name}` — regenerated from `metrics_summary.csv`. n(test)=60/cell._", "",
             "| compression | illumination | base SSIM | SwinIR SSIM | ΔSSIM | base MSE | SwinIR MSE | ΔMSE | iters |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['compression']} | {r['illumination']} | {r['base_ssim']:.4f} | "
                     f"{r['swinir_ssim']:.4f} | {r['delta_ssim']:+.4f} | {r['base_mse']:.6f} | "
                     f"{r['swinir_mse']:.6f} | {r['delta_mse']:+.6f} | {r.get('iterations_reached','?')} |")
    # acceptance verdicts (only over cells present so far)
    by = {(r["compression"], r["illumination"]): r for r in rows}
    n = len(rows)
    ssim_up = sum(1 for r in rows if r["delta_ssim"] > 0)
    mse_dn = sum(1 for r in rows if r["delta_mse"] < 0)
    lines += ["", f"**Acceptance (over {n}/8 cells trained so far):**",
              f"1. SwinIR improves SSIM over base: **{ssim_up}/{n}** cells.",
              f"2. SwinIR reduces MSE over base: **{mse_dn}/{n}** cells."]
    # #3 learnable+SwinIR >= pseudo+SwinIR per compression
    comp_cmp = []
    for comp in COMPS:
        pr, ln = by.get((comp, "pseudo_random")), by.get((comp, "learnable"))
        if pr and ln:
            comp_cmp.append(f"{comp}: learn {ln['swinir_ssim']:.4f} {'>=' if ln['swinir_ssim'] >= pr['swinir_ssim'] else '<'} rand {pr['swinir_ssim']:.4f}")
    if comp_cmp:
        lines.append("3. learnable+SwinIR vs pseudo-random+SwinIR (SSIM): " + "; ".join(comp_cmp) + ".")
    # random+SwinIR vs learnable base (the paper's striking claim)
    striking = []
    for comp in COMPS:
        pr = by.get((comp, "pseudo_random"))
        ln = by.get((comp, "learnable"))
        if pr and ln:
            mark = ">" if pr["swinir_ssim"] > ln["base_ssim"] else "<="
            striking.append(f"{comp}: rand+SwinIR {pr['swinir_ssim']:.4f} {mark} learn-base {ln['base_ssim']:.4f}")
    if striking:
        lines.append("   Paper's claim (random+SwinIR vs learnable-CNN base): " + "; ".join(striking) + ".")
    # #4 x1024 sanity
    for illum in ("pseudo_random", "learnable"):
        r = by.get(("x1024", illum))
        if r:
            ok = r["delta_ssim"] > 0 and r["delta_mse"] < 0
            lines.append(f"4. x1024 {illum}: ΔSSIM {r['delta_ssim']:+.4f}, ΔMSE {r['delta_mse']:+.6f} "
                         f"→ {'genuine metric gain' if ok else 'FLAG: not a clear metric win (check for smoothing/hallucination)'}.")
    block = "\n".join(lines)
    text = report_path.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS_TABLE_START -->", "<!-- RESULTS_TABLE_END -->"
    if start in text and end in text:
        pre = text.split(start)[0]
        post = text.split(end)[1]
        text = pre + start + "\n" + block + "\n" + end + post
        report_path.write_text(text, encoding="utf-8")
        print(f"[report] updated LIVE RESULTS block in {report_path.name} ({n} cells)")


def write_gan_comparison(l1_csv: Path, other_csv: Path, other_name: str, out_md: Path):
    """Side-by-side of the canonical l1_ssim run vs a supplementary run (e.g. GAN),
    over the cells the supplementary run actually trained. Shows the metric-vs-
    perceptual trade-off explicitly (no hiding)."""
    def _load(p):
        with open(p) as fh:
            return {(r["compression"], r["illumination"]): r for r in csv.DictReader(fh)}
    l1 = _load(l1_csv)
    ot = _load(other_csv)
    order = {"x16": 0, "x64": 1, "x256": 2, "x1024": 3}
    keys = sorted(ot.keys(), key=lambda k: (order.get(k[0], 9), k[1]))
    lines = [f"# l1_ssim vs {other_name} — shared cells", "",
             "SwinIR-refinement SSIM/MSE by loss recipe (test split, n=60/cell). "
             "`base` is the frozen content-aware reconstruction (identical for both).", "",
             "| comp | illum | base SSIM | l1_ssim SSIM | " + f"{other_name} SSIM | base MSE | l1_ssim MSE | {other_name} MSE |",
             "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        o = ot[k]
        b = l1.get(k)
        bl_ssim = f"{float(b['swinir_ssim']):.4f}" if b else "—"
        bl_mse = f"{float(b['base_mse']):.6f}" if b else f"{float(o['base_mse']):.6f}"
        l1_mse = f"{float(b['swinir_mse']):.6f}" if b else "—"
        lines.append(f"| {k[0]} | {k[1]} | {float(o['base_ssim']):.4f} | {bl_ssim} | "
                     f"{float(o['swinir_ssim']):.4f} | {bl_mse} | {l1_mse} | {float(o['swinir_mse']):.6f} |")
    lines += ["", "_Note: l1_ssim optimises SSIM+MSE directly; the pixel+perceptual+GAN recipe "
              "optimises perceptual/texture realism and may trade a little pixel MSE for "
              "sharper, more paper-like restorations._", ""]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[compare] wrote {out_md} ({len(keys)} shared cells)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="loss-config name = <EXP>/<name> output root")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fixed-index", type=int, default=7, help="fixed test field for the main grids")
    ap.add_argument("--random-indices", default="3,11,19,27,41", help="test indices for the random-fields panel")
    ap.add_argument("--random-comp", default="x256")
    ap.add_argument("--full-panel", action="store_true")
    args = ap.parse_args()

    out_root = EXP / args.name
    device = torch.device(args.device)
    fig_dir = EXP / "figures"

    # The canonical metrics_summary.csv belongs to the l1_ssim sweep (task Step 7.3);
    # any other loss config writes a name-suffixed CSV so it never clobbers it.
    is_canonical = (args.name == "paper_faithful_l1_ssim")
    csv_path = EXP / ("metrics_summary.csv" if is_canonical else f"metrics_summary_{args.name}.csv")
    rows = build_metrics_csv(out_root, csv_path)
    if is_canonical:
        update_report_table(rows, args.name, EXP / "REPORT.md")
    else:
        # supplementary run (e.g. GAN): if the canonical l1_ssim CSV exists, emit a
        # side-by-side comparison of the shared cells (metric vs perceptual trade-off).
        l1_csv = EXP / "metrics_summary.csv"
        if l1_csv.exists() and rows:
            write_gan_comparison(l1_csv, csv_path, args.name, EXP / f"COMPARE_l1ssim_vs_{args.name}.md")
    render_diagnostic(out_root, args.name, device, args.fixed_index, fig_dir)
    idxs = [int(x) for x in args.random_indices.split(",") if x.strip()]
    render_random_fields(out_root, args.name, device, args.random_comp, idxs, fig_dir)
    if args.full_panel:
        render_full_panel(out_root, args.name, device, args.fixed_index, fig_dir)


if __name__ == "__main__":
    main()
