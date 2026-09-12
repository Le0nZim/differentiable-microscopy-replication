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
    rows, coverage, links = [], [], []
    groups = defaultdict(list)
    for job in plan:
        entry = state.get(job["id"], {})
        complete = verify_completed(entry)
        coverage.append({"job": job["id"], "paper_item": STAGES[job["stage"]][0],
                         "status": "completed" if complete else ("invalid_artifacts" if entry.get("status") == "completed" else entry.get("status", "pending")),
                         "protocol": "binary_equal_mean_audit" if job["stage"] == "controlled" else "audited_legacy_soft_mask"})
        if not complete:
            continue
        result_dir = Path(entry["output"])
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
    for filename, values, fields in [
        ("metrics_by_seed.csv", rows, ["job", "stage", "condition", "seed", "metric", "value", "source"]),
        ("aggregate.csv", aggregate, ["stage", "condition", "metric", "n_seeds", "mean", "std"]),
        ("coverage.csv", coverage, ["job", "paper_item", "status", "protocol"]),
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
            "\n[Per-seed metrics](metrics_by_seed.csv) · [Aggregates](aggregate.csv) · [Paper coverage](coverage.csv)",
            "\nBBBC022 substitutes for the missing original U2OS confocal data. Segmentation uses generated pseudo-labels. "
            "The paper workflow retains the documented soft-mask recipes and budgets; it is distinct from the optional binary/equal-dose controlled stage. "
            "Code checks do not establish numerical reproduction or a preferred method ranking.",
            "\nFigure 9's primary panel uses nonoverlapping acquisitions. Its overlap diagnostic reports additional measurements.",
            "\n## Comparisons\n", *figures, "\n## Completed job artifacts\n", *links]
    (out / "README.md").write_text("\n".join(body) + "\n")
    print(out / "README.md")
