"""Prospective journal matrix, with explicit validation-only prerequisites."""
from itertools import product

STAGES = {
    "patchmnist_tune": ("PatchMNIST: validation-only Fourier/spatial LR search", ["mnist"]),
    "patchmnist": ("PatchMNIST A-D and architecture sensitivity", ["mnist"]),
    "ablation_tune": ("BBBC022: validation-only Fourier/spatial LR search", ["bbbc022"]),
    "ablation": ("Table 3 / Figure 10: A-D and architecture sensitivity (substitute)", ["bbbc022"]),
    "upsampling": ("Figure 5: original upsamplers, image sizes and training set sizes", ["mnist"]),
    "noise": ("Table 1 / Figure 6: photon and read-noise grid", ["mnist"]),
    "patchmnist_content": ("Supplement Figure S3: PatchMNIST compression sweep", ["mnist"]),
    "content": ("Figure 3: compression and acquisition geometry (BBBC022 substitute)", ["bbbc022"]),
    "content_swinir": ("Figure 3 / Table S1: frozen-base SwinIR refinement (substitute)", ["bbbc022", "swinir", "vgg"]),
    "segmentation": ("Figure 4: three-stage segmentation (BBBC039 manual masks by default)", ["bbbc039"]),
    "sr": ("Table 2 / Figure 7: natural-image SwinIR", ["sr", "swinir", "vgg"]),
    "mcf7": ("Figures 8-9: matched-loss tubulin SwinIR and CNN models", ["mcf7", "swinir", "vgg"]),
    "controlled": ("Additional binary-mask / equal-mean-dose controls", ["mnist"]),
}
PAPER_STAGES = list(STAGES)
COMPS = [("x16", 8), ("x64", 16), ("x256", 32), ("x1024", 64)]
MODES = ["uniform_all_ones", "random_fixed", "hadamard_fixed", "learnable_frequency"]
VARIANTS = ["A", "B", "C", "D"]
SR_SETS = ["Set5", "Set14", "BSD100", "Urban100", "Manga109"]


def protocol():
    import yaml
    from .common import ROOT
    return yaml.safe_load((ROOT / "configs/journal/protocol.yaml").read_text())


def select_stages(stages=None):
    selected = set(stages or PAPER_STAGES)
    unknown = selected - set(STAGES)
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}. Choose from {', '.join(STAGES)}")
    for stage, prerequisite in [("patchmnist", "patchmnist_tune"), ("ablation", "ablation_tune"),
                                ("controlled", "patchmnist_tune"), ("content_swinir", "content")]:
        if stage in selected:
            selected.add(prerequisite)
    return [s for s in STAGES if s in selected]


def selection_id(stage, family):
    return f"{stage}/{family}_selection_seed{protocol()['tuning_seed']}"


def jobs(seeds, stages=None, *, segmentation_labels="bbbc039"):
    p = protocol()
    if p["tuning_seed"] in seeds:
        raise ValueError("The tuning seed must be separate from final model seeds")
    result = []

    def add(stage, seed, key, **kwargs):
        job = {"id": f"{stage}/{key}_seed{seed}", "stage": stage, "seed": seed,
               "key": key, "requires": [], "protocol": p["id"], **kwargs}
        result.append(job)
        return job

    for stage in select_stages(stages):
        if stage.endswith("_tune"):
            for family in ("original", "rewritten"):
                candidates = []
                for variant, mode in (("C", "learnable_frequency"), ("D", "learnable_spatial")):
                    for index, lr in enumerate(p["learning_rates"][mode]):
                        job = add(stage, p["tuning_seed"], f"{family}_{variant}_lr{index}",
                                  engine="ablation", variant=variant, family=family, lr=lr, tuning=True)
                        candidates.append(job["id"])
                add(stage, p["tuning_seed"], f"{family}_selection", engine="select_lr",
                    family=family, requires=candidates, tuning=True)
            continue
        for seed in seeds:
            if stage in {"patchmnist", "ablation"}:
                for family, variants in (("original", VARIANTS), ("rewritten", ["C", "D"])):
                    for variant in variants:
                        key = variant if family == "original" else f"rewritten_{variant}"
                        requires = [selection_id(stage + "_tune", family)] if variant != "A" else []
                        add(stage, seed, key, engine="ablation", variant=variant, family=family, requires=requires)
            elif stage == "upsampling":
                for size, count, up in product([128, 256, 512], [600, 3000, 6000], ["original_transpose", "original_locality"]):
                    add(stage, seed, f"s{size}_n{count}_{up}", engine="cnn", size=size, count=count, up=up)
            elif stage == "noise":
                for photons, sigma, mode in product([10, 10000], [0, 2.7, 2, 6], ["random_fixed", "learnable_frequency"]):
                    add(stage, seed, f"k{photons}_sigma{sigma}_{mode}", engine="cnn", photons=photons, sigma=sigma, mode=mode)
            elif stage in {"content", "patchmnist_content"}:
                for (comp, d), mode in product(COMPS, MODES if stage == "content" else ["random_fixed", "learnable_frequency"]):
                    prefix = "bbbc022" if stage == "content" else "patchmnist"
                    add(stage, seed, f"{prefix}_{comp}_{mode}", engine="cnn", comp=comp, downscale=d, patterns=4, mode=mode)
                if stage == "content":
                    # Keep each geometry separate at a fixed nominal compression.
                    for (comp, d), t, mode in product(COMPS, [1, 16], ["random_fixed", "learnable_frequency"]):
                        add(stage, seed, f"bbbc022_{comp}_T{t}_{mode}", engine="cnn", comp=comp,
                            downscale=d // 2 if t == 1 else d * 2, patterns=t, mode=mode)
            elif stage == "content_swinir":
                for (comp, _), mode in product(COMPS, ["random_fixed", "learnable_frequency"]):
                    add(stage, seed, f"{comp}_{mode}", engine="refiner", comp=comp, mode=mode,
                        requires=[f"content/bbbc022_{comp}_{mode}_seed{seed}"])
            elif stage == "segmentation":
                for (comp, d), mode in product(COMPS[1:], ["random_fixed", "learnable_frequency"]):
                    add(stage, seed, f"{comp}_{mode}", engine="segmentation", comp=comp, downscale=d, mode=mode,
                        label_source=segmentation_labels)
            elif stage == "sr":
                for condition in ["swinir_wo_li", "swinir_with_li"]:
                    add(stage, seed, condition, engine="sr", condition=condition)
            elif stage == "mcf7":
                conditions = p["mcf7"]["conditions"]
                for condition in conditions:
                    add(stage, seed, condition, engine="mcf7", condition=condition)
                add(stage, seed, "figures", engine="mcf7_figures",
                    requires=[f"mcf7/{c}_seed{seed}" for c in conditions])
            elif stage == "controlled":
                for mode in MODES + ["learnable_spatial"]:
                    deps = [selection_id("patchmnist_tune", "original")] if mode.startswith("learnable") else []
                    add(stage, seed, mode, engine="controlled", mode=mode, requires=deps)
    return result


def requirements(stages, *, segmentation_labels="bbbc039"):
    needed = {d for s in select_stages(stages) for d in STAGES[s][1]}
    if "bbbc039" in needed and segmentation_labels == "pseudo_trackmate":
        needed.remove("bbbc039")
        needed.add("bbbc022")
    return sorted(needed)
