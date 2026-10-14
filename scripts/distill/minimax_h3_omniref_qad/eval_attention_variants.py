# SPDX-License-Identifier: Apache-2.0
"""Training-free FP4 attention variants on the QAD held-out rows (torchrun; same config as training).

NVFP4 linears off, FP4 VSA-128 attention on, one numeric mode at a time: per-forward x0 error and teacher
alignment (the student's own 8-step sample vs the teacher's, same seed). Independent of the linear
calibration, so it can run before Stage A's amax exists.

``--linears`` keeps the student's NVFP4 linears on (the QAD operating point; needed to compare calibrated vs QAD
weights), ``--no-rollout`` reports only the teacher-forced metrics. The ``qk*`` variants are Q/K-only FP4 (BF16 P.V)
with the error-reduction options of ``fp4_vsa_attn_qat`` (Hadamard rotation, Q smoothing with the exact per-key
correction, BF16/FP8 dense-prefix query rows).
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
    "qk": dict(quantize_pv=False, smooth_k=True),
    "qk_had": dict(quantize_pv=False, smooth_k=True, hadamard=True),
    "qk_sq": dict(quantize_pv=False, smooth_k=True, smooth_q="global"),
    "qk_sqseg": dict(quantize_pv=False, smooth_k=True, smooth_q="segment"),
    "qk_mix16": dict(quantize_pv=False, smooth_k=True, prefix_precision="bf16"),
    "qk_mix8": dict(quantize_pv=False, smooth_k=True, prefix_precision="fp8"),
    "qk_had_mix16": dict(quantize_pv=False, smooth_k=True, hadamard=True, prefix_precision="bf16"),
    "qk_had_mix8": dict(quantize_pv=False, smooth_k=True, hadamard=True, prefix_precision="fp8"),
    "qk_had_sq": dict(quantize_pv=False, smooth_k=True, hadamard=True, smooth_q="global"),
    "qk_had_sqseg": dict(quantize_pv=False, smooth_k=True, hadamard=True, smooth_q="segment"),
    "qk_had_sq_mix16": dict(quantize_pv=False, smooth_k=True, hadamard=True, smooth_q="global",
                            prefix_precision="bf16"),
    "qk_had_sq_mix8": dict(quantize_pv=False, smooth_k=True, hadamard=True, smooth_q="global",
                           prefix_precision="fp8"),
}


def cache_teacher_forwards(method) -> None:
    """Teacher-forced evals follow the teacher's own trajectory, so its forwards repeat across variants: replay them.

    Keyed by (row spec, call index within the row); only valid without rollouts (whose calls depend on the student).
    """
    state = {"key": None, "call": 0}
    cache: dict = {}
    row, forward = method._row, method.teacher.forward_rung

    def cached_row(spec):
        state["key"], state["call"] = id(spec), 0
        return row(spec)

    def cached_forward(prepared, video, audio, row_timesteps):
        key = (state["key"], state["call"])
        state["call"] += 1
        if key not in cache:
            cache[key] = tuple(t.detach().cpu() for t in forward(prepared, video, audio, row_timesteps))
        return tuple(t.to(video.device, non_blocking=True) for t in cache[key])

    method._row = cached_row
    method.teacher.forward_rung = cached_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--linears", action="store_true", help="keep the NVFP4 QAD linears on")
    parser.add_argument("--no-rollout", dest="rollout", action="store_false", help="teacher-forced metrics only")
    args, overrides = parser.parse_known_args()

    import time

    from fastvideo.attention.backends.fp4_vsa_qat import (FP4AttentionNumerics, install_fp4_vsa_attention,
                                                          vsa_impls)
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
    if not args.rollout:
        cache_teacher_forwards(method)
    results = {}
    for name in args.variants:
        numerics = FP4AttentionNumerics(quantize=True, **VARIANTS[name])
        method.student.fp4_numerics = numerics
        install_fp4_vsa_attention(method.student.transformer, numerics)
        impl = vsa_impls(method.student.transformer)[0].fine_attention_override  # shared by every layer
        start = time.time()
        summary = method._evaluate(rows, rollout=args.rollout, linears=args.linears)
        if numerics.prefix_precision is not None and impl.total_blocks:
            # This rank's share of listed key blocks visited by high-precision (dense-prefix) query tiles.
            summary["hp_block_share"] = impl.prefix_blocks / impl.total_blocks
        summary["eval_seconds"] = time.time() - start
        results[name] = summary
        if rank0:
            keys = ("score", "hp_block_share", "eval_seconds", "x0_rel_l2/video/mean", "x0_rel_l2/audio/mean", "endpoint/video", "endpoint/audio",
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
