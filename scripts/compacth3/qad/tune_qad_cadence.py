#!/usr/bin/env python3
"""Make the QAD configs emit frequent short checkpoints so we can find the
EARLIEST checkpoint that reduces grain without raising temporal anomalies.

Decision rule (lead): select the earliest checkpoint that meaningfully reduces
grain without increasing temporal anomalies. Do NOT optimise until QAD
reproduces BF16 exactly -- BF16 has more temporal failures on some cases.
"""
import pathlib, yaml
M = pathlib.Path("/mnt/nfs/vlm-aryan/fasth3-h3-serve-cookbook-eval-20260831/repo-main-3d8ac9d1")
base = M / "examples/train/configs/distribution_matching/minimax_h3"

for tag in ("r768", "r16"):
    p = base / f"qad_nvfp4_4call_{tag}.yaml"
    if not p.exists():
        print(f"MISSING {p.name}"); continue
    d = yaml.safe_load(p.read_text())
    ck = d["training"]["checkpoint"]
    ck["training_state_checkpointing_steps"] = 25      # frequent -> finer sweet-spot search
    ck["checkpoints_total_limit"] = 24                 # keep them all
    d["training"]["loop"]["max_train_steps"] = 200
    # validation cadence: cheap gate signal during the run
    v = d.get("callbacks", {}).get("validation")
    if isinstance(v, dict):
        v["every_steps"] = 50
        v["run_at_start"] = False
    d["training"]["tracker"]["run_name"] = f"qad-nvfp4-4call-{tag}-v2-shortckpt"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    print(f"{p.name}: ckpt every {ck['training_state_checkpointing_steps']} x{ck['checkpoints_total_limit']}, "
          f"max {d['training']['loop']['max_train_steps']} steps, val every {v.get('every_steps')}")
