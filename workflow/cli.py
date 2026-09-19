from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .catalog import STAGES, jobs, requirements, select_stages
from .common import ROOT, digest, file_sha, git_commit, lock, read_json, source_fingerprint, write_json
from .data import defaults, inspect, load_settings, prepare


def now():
    return datetime.now(timezone.utc).isoformat()


def print_plan(plan):
    counts = Counter(j["stage"] for j in plan)
    for i, (stage, count) in enumerate(counts.items(), 1):
        print(f"{i:02d}. {stage:18} {count:3} jobs  {STAGES[stage][0]}")
    print(f"Total: {len(plan)} sequential jobs. No training has started.")


def verify_completed(entry, *, full_hash=False):
    if entry.get("status") != "completed" or not entry.get("artifacts"):
        return False
    for item in entry["artifacts"]:
        p = Path(entry["output"]) / item["path"]
        if not p.is_file() or p.stat().st_size != item["bytes"]:
            return False
        # Hash small summaries each time; large checkpoints are rehashed if modified.
        if full_hash or p.stat().st_size < 1_000_000 or p.stat().st_mtime_ns != item["mtime_ns"]:
            if file_sha(p) != item["sha256"]:
                return False
    return True


def check_environment(cfg, *, needs_workers=False):
    import torch
    from utils.device import resolve_device
    requested = cfg["run"]["device"]
    if requested == "auto":
        raise ValueError("Set run.device explicitly to cuda:1 (or your chosen GPU); auto is not allowed for full campaigns.")
    device = torch.device(requested)
    if device.type == "cuda":
        index = device.index or 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(f"Requested {requested}, but only {torch.cuda.device_count()} CUDA devices are visible. Set run.device explicitly; check CUDA_VISIBLE_DEVICES.")
        torch.zeros(1, device=device).add_(1)
    elif device.type != "cpu":
        raise ValueError("This workstation workflow supports explicit cpu or cuda:N devices.")
    if needs_workers and cfg["run"]["num_workers"]:
        import socket
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM):
                pass
        except OSError as exc:
            raise ValueError("This environment blocks dataloader worker IPC. Set run.num_workers: 0 before starting.") from exc
    return {"python": sys.version, "torch": torch.__version__, "device": str(device),
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}


def free_space(cfg):
    for key in ("output_root", "cache_root"):
        p = Path(cfg["run"][key])
        while not p.exists():
            p = p.parent
        gb = shutil.disk_usage(p).free / 1024**3
        if gb < cfg["run"]["min_free_gb"]:
            raise ValueError(f"Only {gb:.1f} GiB free for {key}; configured floor is {cfg['run']['min_free_gb']} GiB. Free space or choose another disk.")


def validate(cfg, stages, *, environment=False):
    result = inspect(cfg, stages)
    for key, detail in result["evidence"].items():
        print(f"OK    {key}: {detail}")
    if environment:
        try:
            result["environment"] = check_environment(cfg, needs_workers="sr" in stages)
            free_space(cfg)
            print(f"OK    device: {result['environment']['device']}")
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            result["errors"].append(str(exc))
    if result["errors"]:
        raise ValueError("Preflight failed; no training started:\n  - " + "\n  - ".join(result["errors"]))
    return result


def freeze_campaign(cfg, stages, inspection):
    root = Path(cfg["run"]["output_root"])
    path = root / "campaign.json"
    fingerprint = {"settings": cfg, "source_sha256": source_fingerprint()}
    if path.exists():
        saved = read_json(path)
        if saved["fingerprint"] != digest(fingerprint):
            raise ValueError("This output directory belongs to a different config/source revision. Restore it or set run.output_root to a new campaign; completed jobs will not be silently reused.")
    else:
        if any(p.name != ".campaign.lock" for p in root.iterdir()):
            raise ValueError(f"New campaign requires an empty dedicated directory: {root}")
        from .catalog import protocol
        write_json(path, {**fingerprint, "fingerprint": digest(fingerprint), "git_commit": git_commit(),
                          "scientific_protocol": protocol(), "created": now()})
    inv_path = root / "input_inventory.json"
    old = read_json(inv_path) if inv_path.exists() else {}
    for key, inventory in inspection["inventories"].items():
        if key in old and old[key] != inventory:
            raise ValueError(f"{key} inputs changed since preparation. Use a new output_root; existing results have a different input inventory.")
    prepared_path = root / "prepared.json"
    previous = read_json(prepared_path) if prepared_path.exists() else {"paths": {}, "hashes": {}}
    for filename, sha in previous["hashes"].items():
        if not Path(filename).is_file() or file_sha(filename) != sha:
            raise ValueError(f"Prepared manifest was changed or removed: {filename}. Restore it or use a new campaign.")
    # Preparation is deterministic and idempotent; stages may be added later.
    additions = prepare(cfg, stages, inspection)
    previous["paths"].update(additions)
    for name in ("bbbc022_large", "bbbc022_segmentation", "mcf7_manifest", "sr_split"):
        if name in previous["paths"]:
            p = previous["paths"][name]
            previous["hashes"][p] = file_sha(p)
    write_json(prepared_path, previous)
    old.update(inspection["inventories"])
    write_json(inv_path, old)
    write_json(root / "preflight.json", {"checked": now(), "evidence": inspection["evidence"], "environment": inspection.get("environment")})
    return previous["paths"]


def record_artifacts(out):
    result = []
    for name in read_json(out / "worker_complete.json")["artifacts"]:
        path = out / name
        if path.suffix == ".json":
            # Reject NaN/Infinity instead of marking unusable metrics complete.
            payload = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON value {value} in {path}")))
            if isinstance(payload, dict):
                if payload.get("loss_finite") is False:
                    raise ValueError(f"Trainer recorded nonfinite losses: {path}")
                target = payload.get("iterations_target", payload.get("iterations"))
                if target is not None and payload.get("iterations_reached", target) != target:
                    raise ValueError(f"Trainer did not reach its prescribed iterations: {path}")
        result.append({"path": name, "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns, "sha256": file_sha(path)})
    return result


def run_campaign(cfg, stages, *, dry_run=False):
    plan = jobs(cfg["run"]["seeds"], stages)
    if dry_run:
        print_plan(plan)
        for job in plan:
            print(job["id"])
        return
    root = Path(cfg["run"]["output_root"])
    from .worker import materialize
    with lock(root / ".campaign.lock"):
        inspection = validate(cfg, stages, environment=True)
        prepared = freeze_campaign(cfg, stages, inspection)
        state_path = root / "state.json"
        state = read_json(state_path) if state_path.exists() else {}
        known_path = root / "plan.json"
        known = {j["id"]: j for j in read_json(known_path)} if known_path.exists() else {}
        known.update({j["id"]: j for j in plan})
        write_json(known_path, list(known.values()))
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("MKL_NUM_THREADS", "4")
        for i, job in enumerate(plan, 1):
            previous = state.get(job["id"], {})
            if verify_completed(previous):
                print(f"[{i}/{len(plan)}] completed: {job['id']}", flush=True)
                continue
            if previous.get("status") == "completed":
                # Changing an ancestor after a dependent succeeded invalidates that dependent.
                raise ValueError(f"Completion artifacts changed/missing for {job['id']}. Restore the artifacts or use a new campaign; refusing stale dependent results.")
            dependencies = {}
            for dep in job["requires"]:
                if not verify_completed(state.get(dep, {})):
                    raise ValueError(f"Dependency is not verified complete: {dep}")
                dependencies[dep] = state[dep]["output"]
            free_space(cfg)
            job_root = root / "jobs" / job["id"]
            attempt = int(previous.get("attempt", 0)) + 1
            while (job_root / f"attempt_{attempt:03}").exists():
                attempt += 1
            out = job_root / f"attempt_{attempt:03}"
            out.mkdir(parents=True)
            resolved = materialize(job, cfg, prepared, out, dependencies)
            spec = {"job": job, "config": resolved, "output": str(out), "device": cfg["run"]["device"], "dependencies": dependencies}
            write_json(out / "job.json", spec)
            state[job["id"]] = {"status": "running", "attempt": attempt, "output": str(out), "started": now()}
            write_json(state_path, state)
            print(f"[{i}/{len(plan)}] starting {job['id']} (attempt {attempt})\nLog: {out / 'console.log'}", flush=True)
            child = None
            try:
                with (out / "console.log").open("w") as log:
                    child = subprocess.Popen([sys.executable, "-m", "workflow.worker", str(out / "job.json")],
                                             cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    while child.poll() is None:
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            print(f"  running {job['id']} — {datetime.now().strftime('%H:%M:%S')}; log: {out / 'console.log'}", flush=True)
                    if child.returncode:
                        raise RuntimeError(f"Job failed with exit code {child.returncode}: {job['id']}. Read {out / 'console.log'}; rerun the same command after fixing the cause.")
                state[job["id"]].update(status="completed", finished=now(), artifacts=record_artifacts(out))
            except BaseException as exc:
                if child is not None and child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                state[job["id"]].update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", finished=now(), error=str(exc))
                write_json(state_path, state)
                raise
            write_json(state_path, state)
            print(f"  completed {job['id']}", flush=True)
        from .report import build_report
        build_report(root)
        print(f"Selected stages complete. Report: {root / 'report/README.md'}")


def status(cfg):
    root = Path(cfg["run"]["output_root"])
    if not (root / "state.json").exists():
        print("No jobs have started.")
        return
    state = read_json(root / "state.json")
    counts = Counter("invalid_artifacts" if e["status"] == "completed" and not verify_completed(e) else e["status"] for e in state.values())
    print(", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    for name, entry in state.items():
        if entry["status"] != "completed":
            print(f"{entry['status']}: {name}\n  {Path(entry['output']) / 'console.log'}")


def main(argv=None):
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description="Prepare datasets and run every paper experiment sequentially. Start with START_HERE.md.")
    parser.add_argument("--config", default=str(ROOT / "workstation.yaml"), help="Machine path config (default: workstation.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create one editable config using your existing data folder")
    init.add_argument("--data-root", required=True)
    init.add_argument("--output-root")
    for name in ("check", "prepare", "plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--stages", nargs="+", choices=list(STAGES))
        if name == "check":
            p.add_argument("--data-only", action="store_true", help="Check data/dependencies without requiring the configured CUDA device")
        if name == "run":
            p.add_argument("--dry-run", action="store_true", help="Print the exact job order without writing or training")
    sub.add_parser("status")
    sub.add_parser("report")
    sub.add_parser("download-weights", help="Download and verify torchvision VGG19 weights once; requires internet")
    smoke = sub.add_parser("smoke", help="Small synthetic CPU smoke; does not require datasets")
    smoke.add_argument("--output", default=str(ROOT / "runs" / ("smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S"))))
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT / "src")) if str(ROOT / "src") not in sys.path else None
    try:
        if args.command == "init":
            import yaml
            path = Path(args.config)
            if path.exists():
                raise ValueError(f"Config already exists: {path}; edit it or select a new --config path.")
            cfg = defaults(args.data_root)
            if args.output_root:
                cfg["run"]["output_root"] = str(Path(args.output_root).expanduser().resolve())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Machine paths only. Recipes live in configs/.\n" + yaml.safe_dump(cfg, sort_keys=False))
            print(f"Created {path.resolve()}. Next: python paper.py check")
            return 0
        if args.command == "download-weights":
            import torch
            from torchvision.models import VGG19_Weights
            torch.hub.load_state_dict_from_url(VGG19_Weights.IMAGENET1K_V1.url, progress=True, check_hash=True, map_location="cpu")
            print("VGG19 is cached. Subsequent training can run offline.")
            return 0
        if args.command == "smoke":
            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", "2")
            env.setdefault("MKL_NUM_THREADS", "2")
            subprocess.run([sys.executable, str(ROOT / "scripts/audit/run_controlled.py"), "--config",
                            str(ROOT / "configs/audit/controlled_patchmnist.yaml"), "--smoke", "--device", "cpu",
                            "--seeds", "42", "--output", str(Path(args.output).resolve())], cwd=ROOT, env=env, check=True)
            return 0
        cfg = load_settings(args.config)
        stages = select_stages(getattr(args, "stages", None))
        if args.command == "plan":
            print_plan(jobs(cfg["run"]["seeds"], stages))
        elif args.command == "check":
            validate(cfg, stages, environment=not args.data_only)
        elif args.command == "prepare":
            with lock(Path(cfg["run"]["output_root"]) / ".campaign.lock"):
                inspection = validate(cfg, stages)
                freeze_campaign(cfg, stages, inspection)
            print("Data manifests prepared. PatchMNIST is generated and cached automatically when first needed.")
        elif args.command == "run":
            run_campaign(cfg, stages, dry_run=args.dry_run)
        elif args.command == "status":
            status(cfg)
        elif args.command == "report":
            from .report import build_report
            build_report(Path(cfg["run"]["output_root"]))
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to continue with verified completed jobs skipped.", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
