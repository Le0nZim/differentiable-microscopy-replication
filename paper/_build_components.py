#!/usr/bin/env python3
"""Build PowerPoint-ready *atomic* constituent SVGs for manuscript Figures 3-10.

Design rules (per user request):
  * Every microscopy / reconstruction / mask / pattern tile is rendered CLEANLY
    from the source checkpoints or saved tensors - a bare image frame with NO
    overlaid markers, NO SSIM/PSNR text, NO titles, NO panel borders, and NO
    clipping. Each tile is its own SVG (embedded lossless PNG at native pixels).
  * Nothing is a composite. Legends are NOT emitted as grouped graphics; instead
    every legend glyph (shape / line / colour swatch) is a separate symbol SVG.
  * Quantitative graphs (scatter / line / bar) are editable vector SVGs in Arial.
  * Scale-bar templates are unlabeled (no physical calibration exists in source).

Run (GPU used automatically when available):
    python paper/_build_components.py
"""
from __future__ import annotations

import base64
import csv
import importlib.util
import io
import math
import shutil
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
REPL = ROOT
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for p in (str(SRC), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none", "axes.unicode_minus": False})

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MANIFEST: list[dict[str, str]] = []


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# low-level writers
# ---------------------------------------------------------------------------
def add(path: Path, kind: str, source: str, note: str = "") -> None:
    MANIFEST.append({
        "figure": path.relative_to(OUT).parts[0],
        "component": str(path.relative_to(OUT)),
        "kind": kind,
        "source": source,
        "note": note,
    })


def to_rgb(arr, cmap: str = "viridis", vmin: float | None = None, vmax: float | None = None) -> Image.Image:
    a = np.asarray(arr, dtype=np.float32)
    vmin = float(a.min()) if vmin is None else vmin
    vmax = float(a.max()) if vmax is None else vmax
    if vmax <= vmin:
        vmax = vmin + 1e-6
    n = np.clip((a - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = (matplotlib.colormaps[cmap](n)[..., :3] * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(rgb)


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGBA").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def image_svg(img: Image.Image, out: Path, *, source: str, note: str = "", overlay: str = "") -> None:
    im = img.convert("RGBA")
    payload = _png_b64(im)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{im.width}" height="{im.height}" viewBox="0 0 {im.width} {im.height}" '
        f'shape-rendering="crispEdges"><title>{escape(out.stem)}</title>'
        f'<image width="{im.width}" height="{im.height}" image-rendering="pixelated" '
        f'href="data:image/png;base64,{payload}"/>{overlay}</svg>\n',
        encoding="utf-8",
    )
    add(out, "clean image (embedded PNG)", source, note)


def image_with_scalebar_svg(img: Image.Image, out: Path, *, source: str, note: str = "",
                            frac: float = 0.2, color: str = "#ffffff") -> None:
    w, h = img.size
    bar_w = int(round(w * frac))
    bar_h = max(3, int(round(h * 0.028)))
    margin = int(round(w * 0.06))
    x, y = w - bar_w - margin, h - bar_h - margin
    overlay = f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" fill="{color}"/>'
    image_svg(img, out, source=source,
              note=note or f"White scale bar burned in ({bar_w}px of {w}px; uncalibrated).", overlay=overlay)


def colorbar_svg(out: Path, cmap: str, *, width: int = 48, height: int = 420, n: int = 128, note: str = "") -> None:
    stops = []
    for i in range(n + 1):
        t = i / n  # SVG offset 0 = top -> map to value 1.0
        r, g, b, _ = matplotlib.colormaps[cmap](1.0 - t)
        stops.append(f'<stop offset="{t:.4f}" stop-color="rgb({int(r*255+0.5)},{int(g*255+0.5)},{int(b*255+0.5)})"/>')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<defs><linearGradient id="cb" x1="0" y1="0" x2="0" y2="1">{"".join(stops)}</linearGradient></defs>'
        f'<rect x="1" y="1" width="{width-2}" height="{height-2}" fill="url(#cb)" stroke="#000000" stroke-width="1"/></svg>\n',
        encoding="utf-8",
    )
    add(out, "vector colorbar (0.0-1.0, unlabeled)", "colormap",
        note or "Vertical 0.0 (bottom) -> 1.0 (top); matches GT display normalization.")


def save_plot(fig: plt.Figure, out: Path, *, source: str, note: str = "") -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()  # keeps the requested figure size (unlike bbox_inches='tight')
    fig.savefig(out, format="svg", facecolor="white")
    plt.close(fig)
    add(out, "editable vector graph (Arial)", source, note)


def text_svg(out: Path, text: str, *, width=420, height=70, size=28, weight="normal", color="#000000", anchor="start") -> None:
    x = 0 if anchor == "start" else width / 2 if anchor == "middle" else width
    lines = text.split("\n")
    dy = size * 1.2
    y0 = (height - dy * (len(lines) - 1)) / 2
    spans = "".join(f'<tspan x="{x}" y="{y0 + i * dy:.1f}">{escape(t)}</tspan>' for i, t in enumerate(lines))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<text font-family="Arial" font-size="{size}" font-weight="{weight}" fill="{color}" '
        f'text-anchor="{anchor}">{spans}</text></svg>\n',
        encoding="utf-8",
    )
    add(out, "vector text label (Arial)", "recreated from figure annotation")


def symbol_svg(out: Path, shape: str, *, stroke: str, fill="none", size=64, sw=4) -> None:
    c, r = size / 2, size * 0.30
    if shape == "circle":
        body = f'<circle cx="{c}" cy="{c}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    elif shape == "square":
        body = f'<rect x="{c-r}" y="{c-r}" width="{2*r}" height="{2*r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    elif shape == "triangle":
        body = f'<polygon points="{c},{c-r-2} {c-r-2},{c+r} {c+r+2},{c+r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"/>'
    elif shape == "diamond":
        body = f'<polygon points="{c},{c-r} {c+r},{c} {c},{c+r} {c-r},{c}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"/>'
    else:
        raise ValueError(shape)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">{body}</svg>\n',
        encoding="utf-8",
    )
    add(out, "vector symbol", "recreated from figure legend")


def line_svg(out: Path, color: str, *, dashed=False, marker: str | None = None) -> None:
    dash = ' stroke-dasharray="18 12"' if dashed else ""
    mk = ""
    if marker == "circle":
        mk = f'<circle cx="240" cy="20" r="10" fill="{color}"/>'
    elif marker == "square":
        mk = f'<rect x="230" y="10" width="20" height="20" fill="{color}"/>'
    elif marker == "triangle":
        mk = f'<polygon points="240,8 228,32 252,32" fill="{color}"/>'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="480" height="40" viewBox="0 0 480 40">'
        f'<line x1="8" y1="20" x2="472" y2="20" stroke="{color}" stroke-width="6"{dash}/>{mk}</svg>\n',
        encoding="utf-8",
    )
    add(out, "vector line swatch", "recreated from figure legend")


def border_svg(out: Path, color: str, *, dashed=False, width=500, height=260) -> None:
    dash = ' stroke-dasharray="18 12"' if dashed else ""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect x="4" y="4" width="{width-8}" height="{height-8}" fill="none" stroke="{color}" stroke-width="8"{dash}/></svg>\n',
        encoding="utf-8",
    )
    add(out, "vector overlay (border)", "recreated from figure convention")


def scale_bars(fig_dir: Path) -> None:
    for color, name in (("#ffffff", "white"), ("#000000", "black")):
        p = fig_dir / "scale_bars" / f"scale_bar_{name}_unlabeled.svg"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="240" height="28" viewBox="0 0 240 28">'
            f'<rect x="10" y="9" width="220" height="10" fill="{color}"/></svg>\n',
            encoding="utf-8",
        )
        add(p, "vector scale-bar template", "template",
            "Unlabeled: source figures carry no physical pixel calibration.")


# ===========================================================================
# Figure 3 - content-aware reconstruction (+ SwinIR), BBBC022 substitute
# ===========================================================================
def build_fig3() -> None:
    d = OUT / "figure03_content_aware"
    R = _load_module("render_fig03", SCRIPTS / "figure03_content_aware" / "render.py")
    F3 = _load_module("fig3_report", SCRIPTS / "figure03_content_aware" / "report.py")
    S = F3.S
    base_root = F3.BASE_ROOT
    out_root = F3.EXP / "paper_faithful_pixel_perceptual_gan"

    ref_cfg = R.load_experiment_config(R.run_dir(base_root, "x16", "hadamard_fixed") / "config.yaml")
    ref_cfg["experiment"]["device"] = str(DEVICE)
    image_size = ref_cfg["dataset"]["image_size"]
    loader = R.build_dataloader(ref_cfg, "test")
    specimen, chosen = R.select_test_specimen(loader, 65.0, 7)
    specimen = specimen.to(DEVICE)
    print(f"[fig3] specimen index={chosen} device={DEVICE}", flush=True)

    gt_rgb = to_rgb(specimen[0, 0].cpu().numpy(), "viridis", 0, 1)
    image_svg(gt_rgb, d / "images/ground_truth.svg", source="figure03_content_aware/base (GT test field)")
    image_with_scalebar_svg(gt_rgb, d / "images/ground_truth_with_scalebar.svg",
                            source="figure03_content_aware/base (GT test field)")
    colorbar_svg(d / "colorbars/viridis_0to1.svg", "viridis",
                 note="Viridis 0.0->1.0; matches GT / reconstruction / detection display normalization.")
    colorbar_svg(d / "colorbars/gray_0to1.svg", "gray",
                 note="Grayscale 0.0->1.0; matches illumination-pattern (C1/C2) display normalization.")

    col_name = {"hadamard_fixed": "hadamard", "uniform_all_ones": "all_ones",
                "random_fixed": "pseudo_random", "learnable_frequency": "learnable"}
    for comp, _ds in R.COMPS:
        for mkey, _lab, _mk, _c in R.METHODS:
            cfg = R.load_experiment_config(R.run_dir(base_root, comp, mkey) / "config.yaml")
            cfg["experiment"]["device"] = str(DEVICE)
            model = R.load_model(cfg, R.run_dir(base_root, comp, mkey) / "checkpoints/best.pt", DEVICE, image_size)
            with torch.no_grad():
                out = model(specimen, sigmoid_m=10.0, apply_noise=False)
            recon = out["x_recon"][0, 0].clamp(0, 1).cpu().numpy()
            image_svg(to_rgb(recon, "viridis", 0, 1),
                      d / f"images/reconstructions/{comp}/{col_name[mkey]}.svg",
                      source=f"base recon {comp}/{mkey}")
            det = out["y_down"][0].mean(dim=0).cpu().numpy()
            image_svg(to_rgb(det, "viridis", 0, float(det.max())),
                      d / f"images/detections/{comp}/{col_name[mkey]}.svg",
                      source=f"detection y_down {comp}/{mkey}", note="Normalized by field max.")
            p_all = out["patterns"][:, 0].cpu().numpy()
            if mkey == "learnable_frequency":
                ds = f"downscale_{R.DOWNSCALE[comp]}x{R.DOWNSCALE[comp]}"
                with torch.no_grad():
                    soft = model(specimen, sigmoid_m=1.0, apply_noise=False)["patterns"][:, 0].cpu().numpy()
                    hard = model(specimen, sigmoid_m=8.0, apply_noise=False)["patterns"][:, 0].cpu().numpy()
                ch = int(np.argmax(soft.reshape(soft.shape[0], -1).var(axis=1)))
                lo, hi = R.robust_norm(soft[ch], 1, 99)
                image_svg(to_rgb(soft[ch], "gray", lo, hi), d / f"patterns/learned/{ds}.svg",
                          source=f"learned H_t {comp} (soft sigmoid m=1)",
                          note="Soft transmittance (m=1): grayscale, matches paper Fig.3 C2 appearance.")
                image_svg(to_rgb(hard[ch], "gray", 0, 1), d / f"patterns/learned_deployed_binary/{ds}.svg",
                          source=f"learned H_t {comp} (deployed sigmoid m=8)",
                          note="Near-binary mask physically projected at eval sharpness m=8.")
            if comp == "x16" and mkey in ("hadamard_fixed", "uniform_all_ones", "random_fixed"):
                if mkey == "hadamard_fixed":
                    pat = p_all[int(np.argmax(p_all.reshape(p_all.shape[0], -1).var(axis=1)))]
                else:
                    pat = p_all[0]
                image_svg(to_rgb(pat, "gray", 0, float(pat.max()) or 1.0),
                          d / f"patterns/fixed/{col_name[mkey]}.svg", source=f"fixed pattern {mkey}")

            if mkey in ("random_fixed", "learnable_frequency"):
                ref = F3.load_refiner(out_root / comp / mkey / "checkpoints/best.pt", DEVICE)
                with torch.no_grad():
                    base_out = model(specimen, sigmoid_m=S.EVAL_M[mkey], apply_noise=False)["x_recon"]
                    rrec = (ref(base_out) if ref is not None else base_out).clamp(0, 1)
                image_svg(to_rgb(rrec[0, 0].cpu().numpy(), "viridis", 0, 1),
                          d / f"images/reconstructions/{comp}/{col_name[mkey]}_swinir.svg",
                          source=f"+SwinIR recon {comp}/{mkey}")
                del ref
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    # graphs D / E (no legend on the scatter - paper keeps the legend separate)
    base = {}
    with (base_root / "results.csv").open() as fh:
        for r in csv.DictReader(fh):
            base[(r["compression"], r["pattern"])] = {"ssim": float(r["test_ssim"]), "mse": float(r["test_mse"])}
    swin = {}
    with (F3.EXP / "metrics_summary_paper_faithful_pixel_perceptual_gan.csv").open() as fh:
        for r in csv.DictReader(fh):
            k = (r["compression"], "random_fixed" if r["illumination"] == "pseudo_random" else "learnable_frequency")
            swin[k] = {"ssim": float(r["swinir_ssim"]), "mse": float(r["swinir_mse"])}
    order = ["x1024", "x256", "x64", "x16"]
    styles = [("hadamard_fixed", "s", "#1f77b4"), ("uniform_all_ones", "^", "#1f77b4"),
              ("random_fixed", "o", "#d62728"), ("learnable_frequency", "o", "#2ca02c")]
    szs = {"x1024": 180, "x256": 130, "x64": 90, "x16": 55}
    for metric, ylabel in (("ssim", "SSIM"), ("mse", "MSE")):
        fig, ax = plt.subplots(figsize=(5.39, 3.61))  # width x height inches (user spec)
        for x in range(4):
            ax.axvline(x, color="#9ecae1", ls="--", lw=0.8, alpha=0.7)
        for key, marker, color in styles:
            ax.scatter(range(4), [base[(c, key)][metric] for c in order],
                       s=[szs[c] for c in order], marker=marker, facecolors="none", edgecolors=color, linewidths=1.6)
        for key, color in (("random_fixed", "#d62728"), ("learnable_frequency", "#2ca02c")):
            ax.scatter(range(4), [swin[(c, key)][metric] for c in order],
                       s=[szs[c] for c in order], marker="o", facecolors=color, edgecolors="black", linewidths=0.7)
        ax.set_xticks(range(4), order, fontsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.grid(axis="y", ls=":", alpha=0.4)
        save_plot(fig, d / f"plots/{metric}_vs_compression.svg",
                  source="results.csv + GAN metrics CSV",
                  note="Figure 5.39in W x 3.61in H, Arial 16pt.")

    # individual legend glyphs
    symbol_svg(d / "symbols/hadamard_blue_square.svg", "square", stroke="#1f77b4")
    symbol_svg(d / "symbols/all_ones_blue_triangle.svg", "triangle", stroke="#1f77b4")
    symbol_svg(d / "symbols/pseudo_random_red_open_circle.svg", "circle", stroke="#d62728")
    symbol_svg(d / "symbols/learnable_green_open_circle.svg", "circle", stroke="#2ca02c")
    symbol_svg(d / "symbols/pseudo_random_swinir_red_filled_circle.svg", "circle", stroke="#000000", fill="#d62728")
    symbol_svg(d / "symbols/learnable_swinir_green_filled_circle.svg", "circle", stroke="#000000", fill="#2ca02c")
    for ds_k, px in {"8x8": 34, "16x16": 46, "32x32": 58, "64x64": 72}.items():
        symbol_svg(d / f"symbols/marker_size_downscale_{ds_k}.svg", "circle", stroke="#000000", size=px, sw=3)
    for lab in ("Hadamard", "All ones", "Pseudo-random", "Learnable", "Pseudo-random + SwinIR", "Learnable + SwinIR"):
        text_svg(d / f"labels/col_{lab.lower().replace(' ', '_').replace('+', 'plus')}.svg", lab, width=460, height=54, size=22)
    for comp in ("x16", "x64", "x256", "x1024"):
        text_svg(d / f"labels/row_{comp}.svg", comp, width=180, height=54, size=24, weight="bold", anchor="middle")
    scale_bars(d)


# ===========================================================================
# Figure 4 - task-aware segmentation, BBBC022 substitute
# ===========================================================================
def build_fig4() -> None:
    d = OUT / "figure04_segmentation"
    F4 = _load_module("fig4_report", SCRIPTS / "figure04_segmentation" / "report.py")
    seed, k = 42, 8
    imgs, masks = F4._test_examples(seed, k)
    k = len(imgs)
    x = torch.stack(imgs).to(DEVICE)
    print(f"[fig4] {k} test fields device={DEVICE}", flush=True)

    # Paper-like display: re-load the SAME test fields with percentile (calibrated)
    # normalization. The model trains on `paper_strict` (bias-subtract + clip[0,500]
    # + min-max), which saturates the uniformly-bright Hoechst nuclei to solid yellow
    # (they have no punctate foci to occupy the top of the range like the paper's data).
    # `bbbc022_calibrated` (percentile bg-subtract + p99.9 clip) reveals nuclear texture.
    cfgd = dict(F4.fig4run.load_experiment_config(F4.fig4run.CONFIG_PATH)["dataset"])
    cfgd.update(seed=seed, return_mask=False, preprocessing_mode="bbbc022_calibrated")
    ds_cal = F4.BBBC022HoechstDataset(F4.BBBC022HoechstConfig.from_dict(cfgd), split="test")
    calib = [ds_cal[i] for i in range(k)]

    for j in range(k):
        image_svg(to_rgb(calib[j][0].numpy(), "viridis", 0, 1),
                  d / f"images/A_gt/field_{j+1:02d}.svg", source="BBBC022 test field (GT, percentile display)",
                  note="Display: bbbc022_calibrated (percentile bg-subtract + p99.9 clip); paper-like, non-saturated.")
        image_svg(to_rgb(imgs[j][0].numpy(), "viridis", 0, 1),
                  d / f"images/A_gt_model_input_paper_strict/field_{j+1:02d}.svg",
                  source="BBBC022 test field (GT, exact model input)",
                  note="paper_strict (bias-subtract+clip)+min-max: the array the model sees; nuclei saturate to yellow.")
        image_svg(to_rgb(masks[j][0].numpy(), "viridis", 0, 1),
                  d / f"images/B_pseudo_gt_mask/field_{j+1:02d}.svg", source="pseudo-GT mask (thr0.3+closing)")

    row_dir = {("x64", "random_fixed"): "C1_x64_pseudo_random", ("x64", "learnable_frequency"): "C2_x64_learnable",
               ("x256", "random_fixed"): "D1_x256_pseudo_random", ("x256", "learnable_frequency"): "D2_x256_learnable",
               ("x1024", "random_fixed"): "E1_x1024_pseudo_random", ("x1024", "learnable_frequency"): "E2_x1024_learnable"}
    have_pat: dict[tuple[str, str], np.ndarray] = {}
    for comp, pattern, _ in F4.ROW_ORDER:
        summ = F4._summary(comp, pattern, seed)
        model = F4._load_model(comp, pattern, pattern == "learnable_frequency", seed, DEVICE)
        thr = float(summ.get("selected_threshold", 0.5))
        with torch.no_grad():
            out = model(x, sigmoid_m=F4.EVAL_M, apply_noise=False)
            pmask = (out["seg_prob"] > thr).float().cpu().numpy()[:, 0]
            disp_m = 1.0 if pattern == "learnable_frequency" else F4.EVAL_M  # soft gray for learned (paper-like)
            pat = model.microscope.pattern_generator(sigmoid_m=disp_m).detach().cpu().numpy()[0, 0]
        have_pat[(comp, pattern)] = pat
        for j in range(k):
            image_svg(to_rgb(pmask[j], "viridis", 0, 1),
                      d / f"images/{row_dir[(comp, pattern)]}/field_{j+1:02d}.svg",
                      source=f"predicted mask {comp}/{pattern}")
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    # all six illumination tiles (both illuminations x three compressions)
    pat_name = {("x64", "random_fixed"): "x64_pseudo_random", ("x64", "learnable_frequency"): "x64_learnable",
                ("x256", "random_fixed"): "x256_pseudo_random", ("x256", "learnable_frequency"): "x256_learnable",
                ("x1024", "random_fixed"): "x1024_pseudo_random", ("x1024", "learnable_frequency"): "x1024_learnable"}
    for key, name in pat_name.items():
        pat = have_pat[key]
        lo, hi = float(pat.min()), float(pat.max())
        note = ("Fixed pseudo-random mask (seed 42, full-res): identical across compressions (superpixel_factor=1)."
                if key[1] == "random_fixed" else "Learned H_t, soft display (sigmoid m=1) to match paper grayscale.")
        image_svg(to_rgb(pat, "gray", lo, hi), d / f"patterns/{name}.svg",
                  source=f"illumination H_t {key[0]}/{key[1]}", note=note)

    vals: dict[str, dict[str, tuple[float, float]]] = {}
    with (F4.METRICS / "fig4_metrics.csv").open() as fh:
        for r in csv.DictReader(fh):
            vals.setdefault(r["compression"], {})[r["illumination"]] = (float(r["test_dice"]), float(r["test_iou"]))
    comps = ["x64", "x256", "x1024"]
    for mi, metric in enumerate(("Dice", "IoU")):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        xx = np.arange(3)
        w = 0.34
        ax.bar(xx - w / 2, [vals[c]["pseudo_random"][mi] for c in comps], w, color="#9e9e9e", label="Pseudo-random")
        ax.bar(xx + w / 2, [vals[c]["learnable"][mi] for c in comps], w, color="#1f77b4", label="Learnable")
        ax.set_xticks(xx, comps)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False)
        save_plot(fig, d / f"plots/{metric.lower()}_bars.svg", source="metrics/fig4_metrics.csv")
    symbol_svg(d / "symbols/pseudo_random_gray_square.svg", "square", stroke="#9e9e9e", fill="#9e9e9e")
    symbol_svg(d / "symbols/learnable_blue_square.svg", "square", stroke="#1f77b4", fill="#1f77b4")
    for lab in row_dir.values():
        text_svg(d / f"labels/{lab}.svg", lab.split('_', 1)[0] + " " + lab.split('_', 1)[1].replace('_', ' '), width=440, height=54, size=20)
    text_svg(d / "labels/A_gt.svg", "A  GT image", width=300, height=54, size=22)
    text_svg(d / "labels/B_pseudo_gt.svg", "B  pseudo-GT mask", width=340, height=54, size=22)
    scale_bars(d)


# ===========================================================================
# Figure 4 (new preprocessing) - retrained end-to-end on bbbc022_calibrated data.
# Same component format as figure04_segmentation; GT here IS the model input
# (calibrated), so no separate paper_strict variant is emitted.
# ===========================================================================
def build_fig4_new_pre() -> None:
    d = OUT / "figure04_segmentation_new_pre"
    F4 = _load_module("fig4_cal_report", SCRIPTS / "figure04_segmentation" / "calibrated_report.py")
    seed, k = 42, 8
    imgs, masks = F4._test_examples(seed, k)
    k = len(imgs)
    x = torch.stack(imgs).to(DEVICE)
    print(f"[fig4_new_pre] {k} test fields device={DEVICE}", flush=True)

    for j in range(k):
        image_svg(to_rgb(imgs[j][0].numpy(), "viridis", 0, 1),
                  d / f"images/A_gt/field_{j+1:02d}.svg",
                  source="BBBC022 test field (GT = calibrated model input)",
                  note="Trained AND displayed on bbbc022_calibrated preprocessing (percentile bg-subtract + p99.9 clip).")
        image_svg(to_rgb(masks[j][0].numpy(), "viridis", 0, 1),
                  d / f"images/B_pseudo_gt_mask/field_{j+1:02d}.svg",
                  source="pseudo-GT mask (thr0.3+closing) on calibrated data")

    row_dir = {("x64", "random_fixed"): "C1_x64_pseudo_random", ("x64", "learnable_frequency"): "C2_x64_learnable",
               ("x256", "random_fixed"): "D1_x256_pseudo_random", ("x256", "learnable_frequency"): "D2_x256_learnable",
               ("x1024", "random_fixed"): "E1_x1024_pseudo_random", ("x1024", "learnable_frequency"): "E2_x1024_learnable"}
    have_pat: dict[tuple[str, str], np.ndarray] = {}
    for comp, pattern, _ in F4.ROW_ORDER:
        summ = F4._summary(comp, pattern, seed)
        model = F4._load_model(comp, pattern, pattern == "learnable_frequency", seed, DEVICE)
        thr = float(summ.get("selected_threshold", 0.5))
        with torch.no_grad():
            out = model(x, sigmoid_m=F4.EVAL_M, apply_noise=False)
            pmask = (out["seg_prob"] > thr).float().cpu().numpy()[:, 0]
            disp_m = 1.0 if pattern == "learnable_frequency" else F4.EVAL_M
            pat = model.microscope.pattern_generator(sigmoid_m=disp_m).detach().cpu().numpy()[0, 0]
        have_pat[(comp, pattern)] = pat
        for j in range(k):
            image_svg(to_rgb(pmask[j], "viridis", 0, 1),
                      d / f"images/{row_dir[(comp, pattern)]}/field_{j+1:02d}.svg",
                      source=f"predicted mask {comp}/{pattern} (calibrated)")
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    pat_name = {("x64", "random_fixed"): "x64_pseudo_random", ("x64", "learnable_frequency"): "x64_learnable",
                ("x256", "random_fixed"): "x256_pseudo_random", ("x256", "learnable_frequency"): "x256_learnable",
                ("x1024", "random_fixed"): "x1024_pseudo_random", ("x1024", "learnable_frequency"): "x1024_learnable"}
    for key, name in pat_name.items():
        pat = have_pat[key]
        lo, hi = float(pat.min()), float(pat.max())
        note = ("Fixed pseudo-random mask (seed 42, full-res): identical across compressions (superpixel_factor=1)."
                if key[1] == "random_fixed" else "Learned H_t, soft display (sigmoid m=1) to match paper grayscale.")
        image_svg(to_rgb(pat, "gray", lo, hi), d / f"patterns/{name}.svg",
                  source=f"illumination H_t {key[0]}/{key[1]}", note=note)

    vals: dict[str, dict[str, tuple[float, float]]] = {}
    with (F4.METRICS / "fig4_metrics.csv").open() as fh:
        for r in csv.DictReader(fh):
            vals.setdefault(r["compression"], {})[r["illumination"]] = (float(r["test_dice"]), float(r["test_iou"]))
    comps = ["x64", "x256", "x1024"]
    for mi, metric in enumerate(("Dice", "IoU")):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        xx = np.arange(3)
        w = 0.34
        ax.bar(xx - w / 2, [vals[c]["pseudo_random"][mi] for c in comps], w, color="#9e9e9e", label="Pseudo-random")
        ax.bar(xx + w / 2, [vals[c]["learnable"][mi] for c in comps], w, color="#1f77b4", label="Learnable")
        ax.set_xticks(xx, comps)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False)
        save_plot(fig, d / f"plots/{metric.lower()}_bars.svg", source="calibrated metrics/fig4_metrics.csv")
    symbol_svg(d / "symbols/pseudo_random_gray_square.svg", "square", stroke="#9e9e9e", fill="#9e9e9e")
    symbol_svg(d / "symbols/learnable_blue_square.svg", "square", stroke="#1f77b4", fill="#1f77b4")
    for lab in row_dir.values():
        text_svg(d / f"labels/{lab}.svg", lab.split('_', 1)[0] + " " + lab.split('_', 1)[1].replace('_', ' '), width=440, height=54, size=20)
    text_svg(d / "labels/A_gt.svg", "A  GT image", width=300, height=54, size=22)
    text_svg(d / "labels/B_pseudo_gt.svg", "B  pseudo-GT mask", width=340, height=54, size=22)
    scale_bars(d)


# ===========================================================================
# Figure 5 - upsampling analysis (plot only)
# ===========================================================================
def build_fig5() -> None:
    d = OUT / "figure05_upsampling"
    data: dict[tuple[int, str], dict[int, float]] = {}
    src = REPL / "experiments/figure05_upsampling/results.csv"
    with src.open() as fh:
        for r in csv.DictReader(fh):
            data.setdefault((int(r["image_size"]), r["upsampling"]), {})[int(r["num_train"])] = float(r["test_mse"])
    markers = {64: "D", 128: "o", 256: "v", 512: "s"}
    shapes = {64: "diamond", 128: "circle", 256: "triangle", 512: "square"}
    colors = {"locality_aware": "#2ca02c", "transpose_conv": "#d62728"}
    labels = {"locality_aware": "OurUpSampling", "transpose_conv": "TransposeConv"}
    sizes = sorted({k[0] for k in data})
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for size in sizes:
        for method in ("locality_aware", "transpose_conv"):
            pts = data[(size, method)]
            ns = sorted(pts)
            ax.plot(ns, [math.log(pts[n]) for n in ns], marker=markers.get(size, "o"), color=colors[method],
                    linewidth=1.8, markersize=7, label=f"img size: {size}, {labels[method]}")
    ax.set_xlim(0, 6300)
    ax.set_xlabel("# Training Images")
    ax.set_ylabel("log(MSE)")
    ax.legend(fontsize=8, framealpha=0.9, loc="upper right")
    ax.grid(True, ls="--", alpha=0.35)
    save_plot(fig, d / "plots/log_mse_vs_training_images.svg", source=str(src.relative_to(ROOT)))
    for size in sizes:
        for method, color in colors.items():
            symbol_svg(d / f"symbols/{shapes.get(size, 'circle')}_{labels[method].lower()}_{size}.svg",
                       shapes.get(size, "circle"), stroke=color, fill=color)
    line_svg(d / "symbols/our_upsampling_green_line.svg", "#2ca02c")
    line_svg(d / "symbols/transpose_conv_red_line.svg", "#d62728")
    scale_bars(d)


# ===========================================================================
# Figure 6 - noise robustness (clean PatchMNIST samples, viridis)
# ===========================================================================
def build_fig6() -> None:
    d = OUT / "figure06_noise_robustness"
    base = REPL / "experiments/table01_noise_robustness"
    srcs = {
        "A_ground_truth": base / "patchmnist_noise_random_fixed_pc10_sr6p0_seed42/figures/ground_truth.png",
        "B_fixed_random": base / "patchmnist_noise_random_fixed_pc10_sr6p0_seed42/figures/reconstruction.png",
        "C_our_method": base / "patchmnist_noise_learnable_frequency_pc10_sr6p0_seed42/eval/final/figures/reconstruction_soft.png",
    }
    for name, src in srcs.items():
        gray = np.asarray(Image.open(src).convert("L").crop((2, 2, 258, 258)), dtype=np.float32)
        lo, hi = float(gray.min()), float(gray.max())
        image_svg(to_rgb((gray - lo) / max(hi - lo, 1e-6), "viridis", 0, 1),
                  d / f"images/{name}.svg", source=str(src.relative_to(ROOT)),
                  note="First PatchMNIST sample; viridis display normalization.")
    for lab in ("A", "B", "C"):
        text_svg(d / f"labels/panel_{lab}.svg", lab, width=80, height=80, size=54, weight="bold", anchor="middle")
    scale_bars(d)


# ===========================================================================
# Figure 7 - SwinIR standard SR (clean full images already exist)
# ===========================================================================
def build_fig7() -> None:
    import json
    d = OUT / "figure07_swinir_sr"
    base = REPL / "experiments/table02_swinir_sr/full/full_image_eval"
    meta = json.loads((base / "metadata.json").read_text())
    for ds, entry in meta["datasets"].items():
        for suffix, cond in (("GT", "ground_truth"), ("woLI", "swinir_without_li"), ("withLI", "swinir_with_li")):
            img = Image.open(base / f"{ds}_{suffix}.png").convert("L")
            image_svg(img, d / f"images/{ds}/{cond}.svg", source=str((base / f"{ds}_{suffix}.png").relative_to(ROOT)))
        for key, lab in (("swinir_wo_li", "without_LI"), ("swinir_with_li", "with_LI")):
            m = entry["conditions"][key]["metrics_vs_gt"]
            text_svg(d / f"metric_labels/{ds}_{lab}.svg", f"PSNR {m['psnr']:.2f} / SSIM {m['ssim']:.3f}", width=420, height=52, size=20, anchor="middle")
    # individual learned illumination pattern tiles (from the fig7 pattern panel source arrays)
    F7 = None
    for lab, color in (("Ground Truth", "#555555"), ("SwinIR w/o LI", "#d62728"), ("SwinIR with LI", "#00a65a")):
        text_svg(d / f"labels/{lab.lower().replace(' ', '_').replace('/', '')}.svg", lab, width=420, height=54, size=24, weight="bold", anchor="middle")
    border_svg(d / "symbols/with_li_green_border.svg", "#00a65a")
    border_svg(d / "symbols/without_li_red_border.svg", "#d62728")
    line_svg(d / "symbols/with_li_green_dashed.svg", "#00a65a", dashed=True)
    line_svg(d / "symbols/without_li_red_dashed.svg", "#d62728", dashed=True)
    scale_bars(d)


# ===========================================================================
# Figure 8 - MCF7 SwinIR vs transpose-conv (3x3, clean tiles from models)
# ===========================================================================
def build_fig8() -> None:
    d = OUT / "figure08_mcf7_swinir"
    F8 = _load_module("reproduce_fig8", SCRIPTS / "figure08_mcf7" / "reproduce_fig8.py")
    cfg = F8._load_yaml(REPL / "configs/figure08_mcf7/reproduce_fig8_tubulin.yaml")
    runs = REPL / "experiments/figure08_mcf7/runs"
    eval_m = float(cfg["reproduce"].get("eval_sigmoid_m", 8.0))
    q_model, _, _ = F8._load_model("wswinir", cfg, runs, DEVICE)
    r_model, _, _ = F8._load_model("transpose256", cfg, runs, DEVICE)
    ds_cfg = dict(cfg["dataset"]); ds_cfg.update(seed=42, patch_size=256, image_size=256)
    ds = F8.MCF7Channel2Dataset.from_dict(ds_cfg, split="test")
    indices = [12, 6, 4]  # match published figure8_metrics.json
    print(f"[fig8] indices={indices} device={DEVICE}", flush=True)
    rows = {"P_ground_truth": "gt", "Q_with_swinir": "Q", "R_without_swinir": "R"}
    for col, idx in enumerate(indices, start=1):
        x = ds[idx].unsqueeze(0).to(DEVICE)
        gt = x.squeeze().cpu().numpy()
        Q = F8._recon(q_model, x, eval_m).squeeze().cpu().numpy()
        Rr = F8._recon(r_model, x, eval_m).squeeze().cpu().numpy()
        lo, hi = float(np.percentile(gt, 1.0)), float(np.percentile(gt, 99.5))
        for row, arr in (("P_ground_truth", gt), ("Q_with_swinir", Q), ("R_without_swinir", Rr)):
            image_svg(to_rgb(arr, "viridis", lo, hi), d / f"images/{row}/field_{col}.svg",
                      source=f"MCF7 tubulin test idx {idx} ({rows[row]})",
                      note="Per-column viridis normalization from GT p1/p99.5.")
    del q_model, r_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    for row, lab in (("P_ground_truth", "P: Ground Truth"), ("Q_with_swinir", "Q: with SwinIR"), ("R_without_swinir", "R: w/O SwinIR")):
        text_svg(d / f"labels/{row}.svg", lab, width=430, height=60, size=28, weight="bold")
    scale_bars(d)


# ===========================================================================
# Figure 9 - MCF7 wide-field wSwinIR vs wCNN (clean tiles + patterns)
# ===========================================================================
def build_fig9() -> None:
    d = OUT / "figure09_mcf7_widefield"
    F9 = _load_module("reproduce_fig9", SCRIPTS / "figure08_mcf7" / "reproduce_fig9.py")
    cfg = F9._load_yaml(REPL / "configs/figure08_mcf7/reproduce_fig9_tubulin.yaml")
    rep = cfg["reproduce"]
    runs = REPL / "experiments/figure08_mcf7/runs"
    eval_m = float(rep.get("eval_sigmoid_m", 8.0))
    overlap = float(rep.get("tile_overlap_frac", 0.25))
    swin, _ = F9._load_model("wswinir", cfg, runs, DEVICE)
    cnn, _ = F9._load_model("wcnn64", cfg, runs, DEVICE)
    ds_cfg = dict(cfg["dataset"]); ds_cfg.update(seed=42, patch_size=256, image_size=256)
    ds = F9.MCF7Channel2Dataset.from_dict(ds_cfg, split="test")
    top, left, height, width = int(rep["fig9_top"]), int(rep["fig9_left"]), int(rep["fig9_height"]), int(rep["fig9_width"])
    src_path = Path(ds.specs[int(rep.get("fig9_src_index", 0))][0])
    full = F9._preprocess(F9._load_tiff(src_path), F9.MCF7Channel2Config.from_dict(ds_cfg))
    _, Hs, Ws = full.shape
    top = min(top, max(0, Hs - height)); left = min(left, max(0, Ws - width))
    field = full[:, top:top + height, left:left + width].unsqueeze(0)
    gt = field[0, 0].numpy()
    print(f"[fig9] field {height}x{width} device={DEVICE}", flush=True)
    rec_swin = F9.overlap_tiled_recon(swin, field, DEVICE, eval_m, 256, max(16, int(256 * overlap)))
    rec_cnn = F9.overlap_tiled_recon(cnn, field, DEVICE, eval_m, 64, max(16, int(64 * overlap)))
    lo, hi = float(np.percentile(gt, 1.0)), float(np.percentile(gt, 99.5))
    image_svg(to_rgb(gt, "viridis", lo, hi), d / "images/ground_truth.svg", source="MCF7 tubulin wide field (GT)")
    image_svg(to_rgb(rec_swin, "viridis", lo, hi), d / "images/with_swinir.svg", source="wSwinIR overlap-add reconstruction")
    image_svg(to_rgb(rec_cnn, "viridis", lo, hi), d / "images/wcnn.svg", source="wCNN overlap-add reconstruction")
    for cond, name in (("wswinir", "with_swinir"), ("wcnn64", "wcnn")):
        pats = torch.load(runs / cond / "illumination/patterns.pt", map_location="cpu")
        soft = pats.squeeze(1).clamp(0, 1).numpy()
        for t in range(soft.shape[0]):
            binar = (soft[t] > 0.5).astype(np.float32)
            image_svg(to_rgb(binar, "gray", 0, 1), d / f"patterns/{name}/pattern_{t+1}.svg",
                      source=f"learned H_t {cond} (binarized)")
    del swin, cnn
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    for lab in ("Ground Truth", "wSwinIR", "wCNN"):
        text_svg(d / f"labels/{lab.lower().replace(' ', '_')}.svg", lab, width=300, height=58, size=26, weight="bold", anchor="middle")
    border_svg(d / "symbols/wswinir_green_border.svg", "#28c85a", width=900, height=200)
    border_svg(d / "symbols/wcnn_red_border.svg", "#dc3c3c", width=900, height=200)
    scale_bars(d)


# ===========================================================================
# Figure 10 - ablation A/B/C/D (clean tiles from saved tensors)
# ===========================================================================
def build_fig10() -> None:
    d = OUT / "figure10_ablation"
    F10 = _load_module("reproduce_fig10", SCRIPTS / "figure10_ablation" / "reproduce.py")
    runs = F10.EXP / "runs"
    data = {L: F10._load(runs, L, 42) for L in F10.LETTERS}
    i0, i1 = F10._pick_two_samples(data["A"]["gt"])
    print(f"[fig10] recon samples i0={i0} i1={i1}", flush=True)
    for L in F10.LETTERS:
        rec = data[L]["recon"]
        ht = data[L]["H_t"]
        n = rec.shape[0]
        image_svg(to_rgb(rec[min(i0, n - 1), 0].numpy(), "viridis", 0, 1),
                  d / f"images/a_reconstruction_1/variant_{L}.svg", source=f"ablation {L} recon #1")
        image_svg(to_rgb(ht[0, 0].numpy(), "gray", 0, 1),
                  d / f"images/b_pattern/variant_{L}.svg", source=f"ablation {L} learned H_t")
        image_svg(to_rgb(rec[min(i1, n - 1), 0].numpy(), "viridis", 0, 1),
                  d / f"images/c_reconstruction_2/variant_{L}.svg", source=f"ablation {L} recon #2")

    import json
    summary = json.loads((F10.EXP / "runs/aggregate_summary.json").read_text())
    letters = list("ABCD")
    for metric, ylabel, pk in (("test_ssim", "SSIM", "ssim"), ("test_mse", "MSE", "mse")):
        ours = [summary["aggregate"][x][metric]["mean"] for x in letters]
        paper = [summary["paper_table3_u2os"][x][pk] for x in letters]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        xx = np.arange(4)
        w = 0.38
        ax.bar(xx - w / 2, ours, w, label="Ours (BBBC022 substitute)", color="#4a90d9")
        ax.bar(xx + w / 2, paper, w, label="Paper (U2OS)", color="#bbbbbb")
        ax.set_xticks(xx, letters)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        save_plot(fig, d / f"plots/{pk}_ours_vs_paper.svg", source="runs/aggregate_summary.json")
    border_svg(d / "symbols/proposed_variant_C_green_highlight.svg", "#00b050", width=330, height=970)
    for lab in "abc":
        text_svg(d / f"labels/row_{lab}.svg", lab, width=70, height=70, size=38, weight="bold", anchor="middle")
    for lab, desc in (("A", "fixed Ht + Tr.Conv. + freq"), ("B", "learnable Ht + Tr.Conv. + freq"),
                      ("C", "learnable Ht + locality + freq (proposed)"), ("D", "learnable Ht + locality, no freq")):
        text_svg(d / f"labels/variant_{lab}.svg", f"{lab}\n{desc}", width=380, height=90, size=18, anchor="middle")
    scale_bars(d)


def write_readme() -> None:
    counts: dict[str, int] = {}
    for row in MANIFEST:
        counts[row["figure"]] = counts.get(row["figure"], 0) + 1
    lines = [
        "# Constituent components - Figures 3-10 (atomic parts only)",
        "",
        "Every item here is a single smallest part - never a composite.",
        "",
        "- `images/`   : clean image tiles rendered straight from the source",
        "  checkpoints/tensors - NO overlaid markers, SSIM/PSNR text, titles,",
        "  borders, or clipping. Each is an SVG wrapping a lossless PNG at native",
        "  pixel resolution (viridis for fluorescence, gray for masks/patterns).",
        "- `patterns/` : individual illumination pattern tiles (grayscale).",
        "- `plots/`    : editable vector graphs (Arial text).",
        "- `symbols/`  : each legend glyph separately (shapes, lines, borders).",
        "- `labels/`   : each text label separately (Arial).",
        "- `scale_bars/`: unlabeled bar templates (no physical calibration exists).",
        "",
        "Assemble the figures yourself in PowerPoint from these parts.",
        "",
        "Asset counts:",
    ]
    lines += [f"- `{n}`: {c}" for n, c in sorted(counts.items())]
    lines += ["", "Rebuild:", "", "```bash",
              "python paper/_build_components.py",
              "```", "", "See `manifest.csv` for the source of every asset."]
    (OUT / "FIGURE_COMPONENTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


BUILDERS = {
    "fig3": build_fig3, "fig4": build_fig4, "fig4_new_pre": build_fig4_new_pre,
    "fig5": build_fig5, "fig6": build_fig6, "fig7": build_fig7,
    "fig8": build_fig8, "fig9": build_fig9, "fig10": build_fig10,
}
FIGDIR = {
    "fig3": "figure03_content_aware", "fig4": "figure04_segmentation",
    "fig4_new_pre": "figure04_segmentation_new_pre", "fig5": "figure05_upsampling",
    "fig6": "figure06_noise_robustness", "fig7": "figure07_swinir_sr",
    "fig8": "figure08_mcf7_swinir", "fig9": "figure09_mcf7_widefield", "fig10": "figure10_ablation",
}
# Full default build excludes fig4_new_pre (it depends on a separately-trained experiment).
DEFAULT_ORDER = ["fig3", "fig4", "fig5", "fig6", "fig7", "fig8", "fig9", "fig10"]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Build constituent SVG components")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Subset of builders to run (e.g. fig4_new_pre). Only their target dirs are wiped.")
    args = ap.parse_args()

    targets = args.only if args.only else DEFAULT_ORDER
    for name in targets:
        fig_dir = OUT / FIGDIR[name]
        if fig_dir.exists():
            shutil.rmtree(fig_dir)

    for name in targets:
        BUILDERS[name]()

    fields = ["figure", "component", "kind", "source", "note"]
    manifest_path = OUT / "manifest.csv"
    if not args.only:
        with manifest_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(MANIFEST)
        write_readme()
    else:
        # Merge the freshly built figures' rows into the shared top-level manifest
        # (same convention as a full build; figures never carry their own manifest).
        rebuilt = {FIGDIR[name] for name in targets}
        existing: list[dict] = []
        if manifest_path.exists():
            with manifest_path.open(newline="", encoding="utf-8") as fh:
                existing = [r for r in csv.DictReader(fh) if r["figure"] not in rebuilt]
        merged = existing + [r for r in MANIFEST if r["figure"] in rebuilt]
        with manifest_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(merged)
    print(f"Wrote {len(MANIFEST)} atomic SVG assets to {OUT}")


if __name__ == "__main__":
    main()
