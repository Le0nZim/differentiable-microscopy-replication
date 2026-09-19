#!/usr/bin/env python3
"""Read a relocated campaign without changing its ledger or downloading weights.

Hashes establish artifact integrity, not scientific validity. Matching Git LFS
pointers are reported separately: their checkpoint bytes have NOT been verified.
This inspector never makes a snapshot eligible for training resume.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value {value} in {path}")
    return json.loads(path.read_text(), parse_constant=reject)


def local_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes campaign: {relative}")
    return path


def inspect_artifact(path, expected):
    if not path.is_file():
        return "missing"
    with path.open("rb") as handle:
        prefix = handle.read(1024)
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            fields = dict(line.split(" ", 1) for line in prefix.decode().splitlines())
            matches = (fields.get("oid") == "sha256:" + expected["sha256"]
                       and fields.get("size") == str(expected["bytes"]))
            return "lfs_pointer_only" if matches else "mismatch"
        if path.stat().st_size != expected["bytes"]:
            return "mismatch"
        digest = hashlib.sha256(prefix)
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "bytes_verified" if digest.hexdigest() == expected["sha256"] else "mismatch"


def inspect_campaign(root):
    root = Path(root).resolve()
    campaign = read_json(root / "campaign.json")
    state, plan = read_json(root / "state.json"), read_json(root / "plan.json")
    counts, per_stage = Counter(), defaultdict(Counter)
    artifacts, problems, completed, verified_json = Counter(), [], [], {}
    for job, entry in state.items():
        counts[entry["status"]] += 1
        per_stage[job.split("/")[0]][entry["status"]] += 1
        if entry["status"] != "completed":
            continue
        completed.append(job)
        out = local_path(root, Path("jobs") / job / f"attempt_{entry['attempt']:03d}")
        for item in entry.get("artifacts", []):
            path = local_path(out, item["path"])
            status = inspect_artifact(path, item)
            artifacts[status] += 1
            if status in {"missing", "mismatch"}:
                problems.append({"job": job, "artifact": item["path"], "status": status})
            if status == "bytes_verified" and path.suffix == ".json":
                verified_json[path] = read_json(path)
        if not entry.get("artifacts"):
            problems.append({"job": job, "status": "no_recorded_artifacts"})

    # Cross-check the exported tables against the verified result JSONs.
    rows = list(csv.DictReader((root / "report/metrics_by_seed.csv").open()))
    groups, metric_errors, seen = defaultdict(list), [], set()
    for row in rows:
        source = local_path(root, row["source"])
        key = (row["stage"], row["condition"], row["metric"])
        identity = (*key, row["seed"])
        value = float(row["value"])
        obj = verified_json.get(source)
        try:
            for part in row["metric"].split("."):
                obj = obj[part]
            valid = math.isfinite(value) and float(obj) == value and identity not in seen
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            metric_errors.append({"job": row["job"], "metric": row["metric"], "source": row["source"]})
        seen.add(identity)
        groups[key].append(value)
    aggregate = list(csv.DictReader((root / "report/aggregate.csv").open()))
    aggregate_errors, aggregate_keys = [], set()
    for row in aggregate:
        key = (row["stage"], row["condition"], row["metric"])
        values = groups.get(key, [])
        valid = bool(values) and len(values) == int(row["n_seeds"]) and key not in aggregate_keys
        if valid:
            valid = math.isclose(statistics.mean(values), float(row["mean"]), rel_tol=1e-12, abs_tol=1e-12)
            valid &= (row["std"] == "" if len(values) == 1 else
                      math.isclose(statistics.stdev(values), float(row["std"]), rel_tol=1e-12, abs_tol=1e-12))
        if not valid:
            aggregate_errors.append(list(key))
        aggregate_keys.add(key)
    aggregate_errors.extend(list(key) for key in groups.keys() - aggregate_keys)
    return {
        "campaign": root.name, "git_commit_recorded": campaign["git_commit"],
        "source_sha256_recorded": campaign["source_sha256"], "planned_jobs": len(plan),
        "recorded_status": dict(counts), "per_stage": dict(per_stage),
        "artifact_integrity": dict(artifacts), "artifact_problems": problems,
        "metric_rows_checked": len(rows), "metric_problems": metric_errors,
        "aggregate_rows_checked": len(aggregate), "aggregate_problems": aggregate_errors,
        "completed_job_ids": completed,
        "incomplete_records": {j: e["status"] for j, e in state.items() if e["status"] != "completed"},
        "checkpoint_replay_performed": False,
        "scientific_validity": "Not established by artifact hashes or completed status",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", action="append", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [inspect_campaign(p) for p in args.campaign]
    ids = Counter(j for report in reports for j in report["completed_job_ids"])
    result = {"campaigns": reports, "unique_completed_jobs": len(ids),
              "duplicated_completed_job_ids": [j for j, count in ids.items() if count > 1]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{len(ids)} distinct completed job IDs; report: {args.output}")
    return int(any(r[k] for r in reports for k in ("artifact_problems", "metric_problems", "aggregate_problems")))


if __name__ == "__main__":
    raise SystemExit(main())
