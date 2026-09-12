#!/usr/bin/env python3
"""Inventory saved configurations without loading historical model checkpoints."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("table01_noise_robustness", "table02_swinir_sr", "table03_ablation",
            "figure03_content_aware", "figure04_segmentation", "figure05_upsampling",
            "figure08_mcf7", "figure10_ablation_patchmnist",
            "figure10_ablation_patchmnist_udith_schedule")


def main():
    records = []
    for family in FAMILIES:
        for path in sorted((ROOT / "experiments" / family).rglob("config.yaml")):
            c = yaml.safe_load(path.read_text())
            if not isinstance(c, dict):
                continue
            p, f, n, tr, ds = (c.get(k, {}) for k in ("pattern_generator", "forward_model", "detector_noise", "training", "dataset"))
            learned = p.get("mode", "").startswith("learnable")
            phases = tr.get("staged_hardening", {})
            if learned and tr.get("use_staged_hardening") and phases:
                budget = (phases.get("inverse_warmup_steps", 0) + phases.get("joint_soft_steps", 0)
                          + len(phases.get("harden_m_values", [])) * phases.get("harden_steps_per_m", 0))
            else:
                budget = tr.get("max_steps")
            d, t = f.get("downscale_factor"), p.get("num_patterns")
            records.append({"family": family, "config": str(path.relative_to(ROOT)),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "mode": p.get("mode"), "seed": c.get("experiment", {}).get("seed"),
                            "dataset": ds.get("name"), "data_seed": ds.get("seed"),
                            "split_path": ds.get("split_path"), "preproc_mode": ds.get("preproc_mode"),
                            "binarization": p.get("binarization", "soft"),
                            "superpixel_factor": p.get("superpixel_factor", 1),
                            "noise": n, "downscale": d, "patterns": t,
                            "compression_from_config": d*d/t if d and t else None,
                            "configured_updates": budget,
                            "upsampling": c.get("inverse_model", {}).get("upsampling")})
    out = ROOT / "docs/audit/historical_inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source_commit": "b71d8441e7e321c54df89f814cebbd587d4bc9b2",
                               "scope": "Saved config.yaml files in listed experiment families; not checkpoint replay or exhaustive artifact validation.",
                               "records": records}, indent=2) + "\n")
    print(json.dumps({"configs": len(records), "families": {
        f: len([r for r in records if r["family"] == f]) for f in FAMILIES}}, indent=2))


if __name__ == "__main__":
    main()
