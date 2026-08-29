#!/usr/bin/env python3
"""Build the figure05_upsampling_fixed constituent component from the clean-split grid.

Main plot uses the published Fig. 5 grid exactly: image sizes {128, 256, 512}
and train counts {600, 3000, 6000}. A supplementary plot adds our extra size 64.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPL = ROOT
FIXED_CSV = ROOT / "experiments/upsampling_ablation/patchmnist_upsampling_analysis_fixed/results.csv"
OUT = ROOT / "paper/figures/figure05_upsampling_fixed"
ORIG_CSV = OUT / "original_leaky_split_results.csv"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})
FIGSIZE = (5.33, 4.57)  # inches; PowerPoint placeholder size
# Matplotlib viewBox is in points (72 user units / inch). At 5.33in × 4.57in,
# 16 user units = 16pt. Do NOT use CSS `pt`: in SVG, 1pt = 1.25px, so 16pt
# becomes 20px and PowerPoint measures ~20pt.
FONT_USER_UNITS = 16


PAPER_SIZES = [128, 256, 512]
PAPER_COUNTS = [600, 3000, 6000]
MARKERS = {64: "D", 128: "o", 256: "v", 512: "s"}
SHAPES = {64: "diamond", 128: "circle", 256: "triangle_down", 512: "square"}
COLORS = {"locality_aware": "#2ca02c", "transpose_conv": "#d62728"}
LABELS = {"locality_aware": "OurUpSampling", "transpose_conv": "TransposeConv"}


def load(csv_path: Path) -> dict[tuple[int, str], dict[int, float]]:
    data: dict[tuple[int, str], dict[int, float]] = {}
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            key = (int(r["image_size"]), r["upsampling"])
            data.setdefault(key, {})[int(r["num_train"])] = float(r["test_mse"])
    return data


def _write_svg(fig: plt.Figure, out: Path) -> None:
    """Save an SVG that measures 16pt Arial at 5.33in × 4.57in in PowerPoint."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", facecolor="white")
    text = out.read_text(encoding="utf-8")
    w_in, h_in = FIGSIZE
    text = re.sub(
        r'(<svg\b[^>]*?)\bwidth="[^"]+"',
        rf'\1width="{w_in}in"',
        text,
        count=1,
    )
    text = re.sub(
        r'(<svg\b[^>]*?)\bheight="[^"]+"',
        rf'\1height="{h_in}in"',
        text,
        count=1,
    )

    def _rewrite_text_tag(match: re.Match[str]) -> str:
        style = match.group("style") or ""
        rest = match.group("rest")
        anchor = "start"
        found = re.search(r"text-anchor:\s*([^;]+)", style)
        if found:
            anchor = found.group(1).strip()
        # Presentation attributes (PowerPoint ignores CSS). 16px = 16 viewBox
        # points = 16pt at 5.33in × 4.57in. CSS `16pt` would be 20px (~20pt).
        return (
            f'<text font-family="Arial" font-size="{FONT_USER_UNITS}px" '
            f'text-anchor="{anchor}"{rest}>'
        )

    text = re.sub(
        r'<text(?P<pre>[^>]*?)\sstyle="(?P<style>[^"]*)"(?P<rest>[^>]*)>',
        _rewrite_text_tag,
        text,
    )
    out.write_text(text, encoding="utf-8")


def plot(data, sizes: list[int], counts: list[int], out: Path, *, legend: bool = True) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for size in sizes:
        for method in ("locality_aware", "transpose_conv"):
            pts = data.get((size, method), {})
            ns = [n for n in counts if n in pts]
            if not ns:
                continue
            ax.plot(
                ns,
                [math.log(pts[n]) for n in ns],
                marker=MARKERS[size],
                color=COLORS[method],
                linewidth=2.0,
                markersize=8,
                label=f"img size: {size}, {LABELS[method]}",
            )
    ax.set_xlim(300, max(counts) + 300)
    ax.set_xlabel("# Training Images")
    ax.set_ylabel("log(MSE)")
    ax.tick_params(labelsize=16)
    ax.grid(True, ls="--", alpha=0.35)
    if legend:
        ax.legend(fontsize=16, loc="upper right", framealpha=0.92, handlelength=1.6)
    fig.tight_layout(pad=0.35)
    _write_svg(fig, out)
    plt.close(fig)


def symbol_svg(out: Path, shape: str, color: str, size: int = 64, sw: int = 4) -> None:
    c, r = size / 2, size * 0.30
    if shape == "circle":
        body = f'<circle cx="{c}" cy="{c}" r="{r}" fill="{color}" stroke="{color}" stroke-width="{sw}"/>'
    elif shape == "square":
        body = (f'<rect x="{c-r}" y="{c-r}" width="{2*r}" height="{2*r}" '
                f'fill="{color}" stroke="{color}" stroke-width="{sw}"/>')
    elif shape == "triangle_down":
        body = (f'<polygon points="{c-r-2},{c-r} {c+r+2},{c-r} {c},{c+r+2}" '
                f'fill="{color}" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"/>')
    elif shape == "diamond":
        body = (f'<polygon points="{c},{c-r} {c+r},{c} {c},{c+r} {c-r},{c}" '
                f'fill="{color}" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"/>')
    else:
        raise ValueError(shape)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}"><title>{escape(out.stem)}</title>{body}</svg>\n',
        encoding="utf-8",
    )


def handle_svg(out: Path, shape: str, color: str, *, width: int = 160, height: int = 40) -> None:
    """One legend handle per series: the colored line with its marker on top."""
    cy = height / 2
    cx = width / 2
    r = 9.0
    line = f'<line x1="8" y1="{cy}" x2="{width - 8}" y2="{cy}" stroke="{color}" stroke-width="3.2"/>'
    if shape == "circle":
        mark = f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" stroke="{color}" stroke-width="2"/>'
    elif shape == "square":
        mark = (f'<rect x="{cx - r}" y="{cy - r}" width="{2 * r}" height="{2 * r}" '
                f'fill="{color}" stroke="{color}" stroke-width="2"/>')
    elif shape == "triangle_down":
        mark = (f'<polygon points="{cx - r - 1},{cy - r} {cx + r + 1},{cy - r} {cx},{cy + r + 1}" '
                f'fill="{color}" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>')
    elif shape == "diamond":
        mark = (f'<polygon points="{cx},{cy - r} {cx + r},{cy} {cx},{cy + r} {cx - r},{cy}" '
                f'fill="{color}" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>')
    else:
        raise ValueError(shape)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><title>{escape(out.stem)}</title>{line}{mark}</svg>\n',
        encoding="utf-8",
    )


def line_svg(out: Path, color: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="480" height="40" viewBox="0 0 480 40">'
        f'<line x1="8" y1="20" x2="472" y2="20" stroke="{color}" stroke-width="6"/></svg>\n',
        encoding="utf-8",
    )


def scale_bars(d: Path) -> None:
    for color, name in (("#ffffff", "white"), ("#000000", "black")):
        p = d / "scale_bars" / f"scale_bar_{name}_unlabeled.svg"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="240" height="28" viewBox="0 0 240 28">'
            f'<rect x="10" y="9" width="220" height="10" fill="{color}"/></svg>\n',
            encoding="utf-8",
        )


def comparison(fixed, orig) -> str:
    lines = [
        "# Figure 5 - original vs clean-split test MSE\n\n",
        "Clean split = val and test drawn from disjoint halves of the MNIST test digit pool.\n",
        "Train digits come from the MNIST train pool in both versions, so neither has train/test leakage.\n\n",
        "| img size | # train | method | original MSE | clean-split MSE | delta |\n",
        "| --- | --- | --- | --- | --- | --- |\n",
    ]
    for size in [64] + PAPER_SIZES:
        for n in PAPER_COUNTS:
            for method in ("locality_aware", "transpose_conv"):
                a = orig.get((size, method), {}).get(n)
                b = fixed.get((size, method), {}).get(n)
                if b is None:
                    continue
                delta = "n/a" if a is None else f"{b - a:+.5f}"
                a_txt = "n/a" if a is None else f"{a:.5f}"
                lines.append(f"| {size} | {n} | {LABELS[method]} | {a_txt} | {b:.5f} | {delta} |\n")
    flips = []
    for size in [64] + PAPER_SIZES:
        for n in PAPER_COUNTS:
            loc = fixed.get((size, "locality_aware"), {}).get(n)
            tr = fixed.get((size, "transpose_conv"), {}).get(n)
            if loc is not None and tr is not None and loc >= tr:
                flips.append(f"size {size}, n={n}")
    lines.append(
        "\n**Winner check (clean split):** "
        + ("locality-aware wins every cell of the paper grid.\n"
           if not flips else "transpose wins at " + "; ".join(flips) + ".\n")
    )
    return "".join(lines)


def main() -> None:
    fixed = load(FIXED_CSV)
    all_sizes = sorted({k[0] for k in fixed})

    plot(fixed, PAPER_SIZES, PAPER_COUNTS, OUT / "plots/log_mse_vs_training_images.svg", legend=True)
    plot(fixed, PAPER_SIZES, PAPER_COUNTS, OUT / "plots/log_mse_vs_training_images_nolegend.svg", legend=False)
    plot(fixed, all_sizes, PAPER_COUNTS, OUT / "plots/log_mse_vs_training_images_with_is64.svg", legend=True)
    plot(fixed, all_sizes, PAPER_COUNTS, OUT / "plots/log_mse_vs_training_images_with_is64_nolegend.svg", legend=False)

    symbols = OUT / "symbols"
    for size in all_sizes:
        shape = SHAPES[size]
        for method, color in COLORS.items():
            tag = LABELS[method].lower()
            symbol_svg(symbols / f"marker_is{size}_{tag}.svg", shape, color)
            handle_svg(symbols / f"handle_is{size}_{tag}.svg", shape, color)
            symbol_svg(symbols / f"{shape}_{tag}_{size}.svg", shape, color)
        symbol_svg(symbols / f"marker_is{size}_black.svg", shape, "#000000")
        handle_svg(symbols / f"handle_is{size}_black.svg", shape, "#000000")
        symbol_svg(symbols / f"{shape}_black_{size}.svg", shape, "#000000")
    line_svg(symbols / "our_upsampling_green_line.svg", COLORS["locality_aware"])
    line_svg(symbols / "transpose_conv_red_line.svg", COLORS["transpose_conv"])
    line_svg(symbols / "black_line.svg", "#000000")
    scale_bars(OUT)

    (OUT / "results.csv").write_bytes(FIXED_CSV.read_bytes())
    (OUT / "SPLIT.md").write_text(
        "Clean split: MNIST train pool for train; disjoint halves of the MNIST test pool for val vs test.\n"
        "Grid matches the published Fig. 5: image sizes 128/256/512 and 600/3000/6000 training images.\n"
        "The supplementary plot additionally shows our extra image size 64.\n"
        "Fixed at x8 compression (d=8, T=8), random_fixed patterns, noise-free, 4000 steps, seed 42.\n",
        encoding="utf-8",
    )
    if ORIG_CSV.exists():
        orig = load(ORIG_CSV)
        (OUT / "COMPARISON.md").write_text(comparison(fixed, orig), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
