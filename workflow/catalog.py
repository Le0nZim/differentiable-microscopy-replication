"""Explicit paper coverage and deterministic single-process job order.

No filesystem discovery, training, historical checkpoints or hidden downloads here.
"""
from itertools import product

STAGES = {
    "patchmnist": ("PatchMNIST A–D diagnostic (additional to the paper)", ["mnist"]),
    "upsampling": ("Figure 5: upsamplers, image sizes and training set sizes", ["mnist"]),
    "noise": ("Table 1 / Figure 6: Poisson and read-noise grid", ["mnist"]),
    "content": ("Figure 3: content-aware compression (BBBC022 substitute)", ["bbbc022"]),
    "content_swinir": ("Figure 3: frozen-base SwinIR refinement (BBBC022 substitute)", ["bbbc022", "swinir", "vgg"]),
    "segmentation": ("Figure 4: all three training stages (BBBC022 pseudo-labels)", ["bbbc022"]),
    "sr": ("Table 2 / Figure 7: natural-image SwinIR", ["sr", "swinir", "vgg"]),
    "mcf7": ("Figures 8–9: tubulin SwinIR and CNN models", ["mcf7", "swinir", "vgg"]),
    "ablation": ("Table 3 / Figure 10: A–D (BBBC022 substitute)", ["bbbc022"]),
    "controlled": ("Additional audit protocol: binary masks / equal mean dose", ["mnist"]),
}
PAPER_STAGES = list(STAGES)[:-1]
COMPS = [("x16", 8), ("x64", 16), ("x256", 32), ("x1024", 64)]
MODES = ["uniform_all_ones", "random_fixed", "hadamard_fixed", "learnable_frequency"]
VARIANTS = ["A", "B", "C", "D"]
SR_SETS = ["Set5", "Set14", "BSD100", "Urban100", "Manga109"]


def select_stages(stages=None):
    selected = list(stages or PAPER_STAGES)
    unknown = set(selected) - set(STAGES)
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}. Choose from {', '.join(STAGES)}")
    if "content_swinir" in selected and "content" not in selected:
        selected.append("content")
    return [s for s in STAGES if s in selected]


def jobs(seeds, stages=None):
    result = []
    def add(stage, seed, key, **kwargs):
        job = {"id": f"{stage}/{key}_seed{seed}", "stage": stage, "seed": seed,
               "key": key, "requires": [], **kwargs}
        result.append(job)
        return job
    for stage in select_stages(stages):
        for seed in seeds:
            if stage in {"patchmnist", "ablation"}:
                for variant in VARIANTS:
                    add(stage, seed, variant, engine="ablation", variant=variant)
            elif stage == "upsampling":
                for size, count, up in product([128, 256, 512], [600, 3000, 6000], ["transpose_conv", "locality_aware"]):
                    add(stage, seed, f"s{size}_n{count}_{up}", engine="cnn", size=size, count=count, up=up)
            elif stage == "noise":
                for photons, sigma, mode in product([10, 10000], [0, 2.7, 2, 6], ["random_fixed", "learnable_frequency"]):
                    add(stage, seed, f"k{photons}_sigma{sigma}_{mode}", engine="cnn", photons=photons, sigma=sigma, mode=mode)
            elif stage == "content":
                for (comp, d), mode in product(COMPS, MODES):
                    add(stage, seed, f"bbbc022_{comp}_{mode}", engine="cnn", comp=comp, downscale=d, mode=mode)
            elif stage == "content_swinir":
                for (comp, _), mode in product(COMPS, ["random_fixed", "learnable_frequency"]):
                    job = add(stage, seed, f"{comp}_{mode}", engine="refiner", comp=comp, mode=mode)
                    job["requires"] = [f"content/bbbc022_{comp}_{mode}_seed{seed}"]
            elif stage == "segmentation":
                for (comp, d), mode in product(COMPS[1:], ["random_fixed", "learnable_frequency"]):
                    add(stage, seed, f"{comp}_{mode}", engine="segmentation", comp=comp, downscale=d, mode=mode)
            elif stage == "sr":
                for condition in ["swinir_wo_li", "swinir_with_li"]:
                    add(stage, seed, condition, engine="sr", condition=condition)
            elif stage == "mcf7":
                for condition in ["wswinir", "transpose256", "wcnn64"]:
                    add(stage, seed, condition, engine="mcf7", condition=condition)
                job = add(stage, seed, "figures", engine="mcf7_figures")
                job["requires"] = [f"mcf7/{c}_seed{seed}" for c in ["wswinir", "transpose256", "wcnn64"]]
            elif stage == "controlled":
                for mode in MODES + ["learnable_spatial"]:
                    add(stage, seed, mode, engine="controlled", mode=mode)
    return result


def requirements(stages):
    return sorted({d for s in select_stages(stages) for d in STAGES[s][1]})
