# SPDX-License-Identifier: Apache-2.0
"""Training-free FP4 attention variants on the QAD held-out rows (torchrun; same config as training).

NVFP4 linears off, FP4 VSA-128 attention on, one numeric mode at a time: per-forward x0 error and teacher
alignment (the student's own 8-step sample vs the teacher's, same seed). Independent of the linear
calibration, so it can run before Stage A's amax exists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VARIANTS = {
    "single": dict(two_level_p=False, smooth_k=False),
    "two_level": dict(two_level_p=True, smooth_k=False),
    "single_smoothk": dict(two_level_p=False, smooth_k=True),
    "two_level_smoothk": dict(two_level_p=True, smooth_k=True),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args, overrides = parser.parse_known_args()

    from fastvideo.attention.backends.fp4_vsa_qat import FP4AttentionNumerics, install_fp4_vsa_attention
    from fastvideo.distributed import get_world_group, maybe_init_distributed_environment_and_model_parallel
    from fastvideo.train.utils.builder import build_from_config
    from fastvideo.train.utils.config import load_run_config

    cfg = load_run_config(args.config, overrides=overrides or None)
    maybe_init_distributed_environment_and_model_parallel(cfg.training.distributed.tp_size,
                                                          cfg.training.distributed.sp_size)
    _, method, _, _ = build_from_config(cfg)
    rank0 = get_world_group().rank == 0
    run = None
    if rank0:
        import wandb
        run = wandb.init(project="fasth3-omniref-nvfp4", name=Path(args.out).name, config={"variants": args.variants})
    rows = method._rank_rows()
    results = {}
    for name in args.variants:
        numerics = FP4AttentionNumerics(quantize=True, **VARIANTS[name])
        method.student.fp4_numerics = numerics
        install_fp4_vsa_attention(method.student.transformer, numerics)
        summary = method._evaluate(rows, rollout=True, linears=False)
        results[name] = summary
        if rank0:
            keys = ("score", "x0_rel_l2/video/mean", "x0_rel_l2/audio/mean", "endpoint/video", "endpoint/audio",
                    "keyframe/vs_teacher/first", "keyframe/vs_teacher/last")
            print("VARIANT " + json.dumps({"variant": name, **{k: round(summary[k], 5) for k in keys if k in summary}}),
                  flush=True)
            run.log({f"{name}/{k}": v for k, v in summary.items()})
    if rank0:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "variants.json").write_text(json.dumps(results, indent=1))
        run.finish()


if __name__ == "__main__":
    main()
