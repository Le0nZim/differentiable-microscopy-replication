"""Summaries from this campaign's verified jobs only; never glob historical runs."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .catalog import STAGES
from .common import read_json, write_json


def _metrics(obj, prefix=""):
    result = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = prefix + key
            if isinstance(value, (float, int)) and not isinstance(value, bool) and any(m in key.lower() for m in ("mse", "ssim", "psnr", "dice", "iou")):
                result[path] = value
            elif isinstance(value, dict) and key not in {"history", "paper_table2", "config"}:
                result.update(_metrics(value, path + "."))
    return result


def build_report(root):
    from .cli import verify_completed
    root = Path(root)
    if not (root / "plan.json").exists():
        raise ValueError(f"No campaign plan at {root}; run an experiment first.")
    plan, state = read_json(root / "plan.json"), read_json(root / "state.json")
    out = root / "report"
    out.mkdir(exist_ok=True)
    rows, coverage, links, tuning_rows = [], [], [], []
    groups = defaultdict(list)
    for job in plan:
        entry = state.get(job["id"], {})
        complete = verify_completed(entry)
        coverage.append({"job": job["id"], "paper_item": STAGES[job["stage"]][0],
                         "status": "completed" if complete else ("invalid_artifacts" if entry.get("status") == "completed" else entry.get("status", "pending")),
                         "protocol": job.get("protocol", "audited_legacy_soft_mask")})
        if not complete:
            continue
        result_dir = Path(entry["output"])
        if job.get("tuning"):
            if job["engine"] == "select_lr":
                selection = read_json(result_dir / "selection.json")
                for mode, candidates in selection["candidates"].items():
                    for candidate in candidates:
                        tuning_rows.append({"stage": job["stage"], "architecture": job["family"], "mode": mode,
                                            **candidate, "selected": candidate["job"] == selection["selected"][mode]["job"],
                                            "selected_at_grid_boundary": selection["selected"][mode]["at_grid_boundary"]})
            continue
        for artifact in entry["artifacts"]:
            p = result_dir / artifact["path"]
            if p.suffix != ".json" or p.name in {"config.json", "results.json"}:
                continue
            obj = read_json(p)
            for metric, value in _metrics(obj).items():
                rows.append({"job": job["id"], "stage": job["stage"], "condition": job["key"],
                             "seed": job["seed"], "metric": metric, "value": value,
                             "source": str(p.relative_to(root))})
                groups[job["stage"], job["key"], metric].append(value)
        import os
        relative = Path(os.path.relpath(result_dir, out)).as_posix()
        links.append(f"- `{job['id']}`: [artifacts and configuration]({relative}/)")
    aggregate = [{"stage": stage, "condition": condition, "metric": metric,
                  "n_seeds": len(values), "mean": statistics.mean(values),
                  "std": statistics.stdev(values) if len(values) > 1 else ""}
                 for (stage, condition, metric), values in groups.items()]
    paired = paired_differences(rows)
    for filename, values, fields in [
        ("metrics_by_seed.csv", rows, ["job", "stage", "condition", "seed", "metric", "value", "source"]),
        ("aggregate.csv", aggregate, ["stage", "condition", "metric", "n_seeds", "mean", "std"]),
        ("coverage.csv", coverage, ["job", "paper_item", "status", "protocol"]),
        ("paired_differences.csv", paired, ["stage", "comparison", "metric", "seed", "candidate", "reference", "difference"]),
        ("learning_rate_search.csv", tuning_rows, ["stage", "architecture", "mode", "job", "learning_rate", "validation_mse", "selected", "selected_at_grid_boundary"]),
    ]:
        with (out / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    # Compact comparison plots, with no historical scores or desired winner gates.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = []
    for stage in dict.fromkeys(j["stage"] for j in plan):
        metric = "test_dice" if stage == "segmentation" else "test_mse"
        selected = [r for r in aggregate if r["stage"] == stage and r["metric"] == metric]
        if not selected:
            continue
        fig, ax = plt.subplots(figsize=(10, max(3, len(selected) * .3)))
        ax.barh([r["condition"] for r in selected], [r["mean"] for r in selected],
                xerr=[r["std"] or 0 for r in selected], color="#216b91")
        ax.set_xlabel(metric + " (mean ± sample SD across completed training seeds)")
        ax.set_title(STAGES[stage][0])
        fig.tight_layout()
        fig.savefig(out / (stage + ".png"), dpi=140)
        plt.close(fig)
        figures.append(f"![{stage}]({stage}.png)")
    completed = sum(c["status"] == "completed" for c in coverage)
    write_json(out / "summary.json", {"completed": completed, "planned": len(plan), "coverage": coverage})
    body = ["# Fresh campaign results", f"\n{completed}/{len(plan)} planned jobs are verified complete. Missing jobs are listed in coverage.csv.",
            "\n[Per-seed metrics](metrics_by_seed.csv) · [Aggregates](aggregate.csv) · [Paired differences](paired_differences.csv) · [Validation search](learning_rate_search.csv) · [Paper coverage](coverage.csv)",
            "\nBBBC022 substitutes for the missing original U2OS confocal data. Segmentation uses generated pseudo-labels. "
            "The journal workflow uses the original CNN architectures for primary comparisons. Rewritten-architecture C/D results are sensitivity analyses. "
            "Binary/equal-dose controls are distinct from the soft-mask experiments. Validation-search trials are excluded from final result aggregates. "
            "Code checks do not establish numerical reproduction or a preferred method ranking.",
            "\nThe LR search has a fixed, equal candidate budget. A boundary winner does not establish an optimum outside that grid. "
            "Paired differences are candidate minus reference, within the same training seed. Seed SD describes training variability, not uncertainty over populations of specimens.",
            "\nFigure 9's primary panel uses nonoverlapping acquisitions. Its overlap diagnostic reports additional measurements.",
            "\n## Comparisons\n", *figures, "\n## Completed job artifacts\n", *links]
    (out / "README.md").write_text("\n".join(body) + "\n")
    print(out / "README.md")


def paired_differences(rows):
    """Compare only matched conditions and final held-out metrics."""
    lookup = {(r["stage"], r["condition"], r["metric"], r["seed"]): r["value"] for r in rows}
    result = []
    for (stage, condition, metric, seed), value in lookup.items():
        if not (metric.startswith("test") or metric.startswith("per_dataset_best.")):
            continue
        references = []
        if stage in {"patchmnist", "ablation"}:
            references = {"B": ["A"], "C": ["B", "D"], "rewritten_C": ["rewritten_D"]}.get(condition, [])
        elif stage in {"content", "patchmnist_content", "noise", "segmentation"} and condition.endswith("learnable_frequency"):
            references = [condition.removesuffix("learnable_frequency") + "random_fixed"]
        elif stage == "upsampling" and condition.endswith("original_locality"):
            references = [condition.removesuffix("original_locality") + "original_transpose"]
        elif stage == "sr" and condition == "swinir_with_li":
            references = ["swinir_wo_li"]
        elif stage == "mcf7" and condition == "wswinir":
            references = ["wcnn256"]
        for reference in references:
            key = (stage, reference, metric, seed)
            if key in lookup:
                result.append({"stage": stage, "comparison": f"{condition}_minus_{reference}", "metric": metric,
                               "seed": seed, "candidate": value, "reference": lookup[key], "difference": value - lookup[key]})
    return result
