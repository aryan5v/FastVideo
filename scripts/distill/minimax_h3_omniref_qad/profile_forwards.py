# SPDX-License-Identifier: Apache-2.0
"""Time one OmniRef row's PDD forwards under each numeric arm (run with torchrun, same config as training).

Arms: teacher (bf16, sm_100a VSA-128), student with NVFP4 linears only (bf16 attention), student fully
quantized (FP4 attention emulator), and one student forward+backward. Prints seconds per forward and the
x0 rel-L2 per arm, which feeds the cost estimate of the full run.
"""
from __future__ import annotations

import argparse
import json
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--rungs", type=int, nargs="+", default=[0, 7])
    args, overrides = parser.parse_known_args()

    from fastvideo.distributed import get_world_group, maybe_init_distributed_environment_and_model_parallel
    from fastvideo.train.methods.knowledge_distillation.pdd_qad_metrics import x0_pair
    from fastvideo.train.utils.builder import build_from_config
    from fastvideo.train.utils.config import load_run_config

    cfg = load_run_config(args.config, overrides=overrides or None)
    maybe_init_distributed_environment_and_model_parallel(cfg.training.distributed.tp_size,
                                                          cfg.training.distributed.sp_size)
    _, method, _, _ = build_from_config(cfg)
    student, teacher = method.student, method.teacher
    method._provisional_calibration()  # no-op when Stage A's table is configured
    spec = student.eval_rows[0]
    prepared = method._row(spec)
    schedulers = student.schedulers()
    timesteps = (schedulers[0].timesteps, schedulers[1].timesteps)
    rank0 = get_world_group().rank == 0

    def timed(model, rung, *, grad=False, **quant):
        torch.cuda.synchronize()
        start = time.perf_counter()
        context = student.quantization(**quant) if model is student else torch.no_grad()
        with context, torch.set_grad_enabled(grad), model.rung_scope(prepared, rung, timesteps) as rt:
            out = model.forward_rung(prepared, prepared.video, prepared.audio, rt)
            if grad:
                (out[0].float().pow(2).mean() + out[1].float().pow(2).mean()).backward()
        torch.cuda.synchronize()
        return out, time.perf_counter() - start

    report = {"row": spec.id, "tokens": int(prepared.layout.sequence_length)}
    for rung in args.rungs:
        for warm in (True, False):
            t_out, t_teacher = timed(teacher, rung)
            lin_out, t_lin = timed(student, rung, linears=True, attention=False)
            full_out, t_full = timed(student, rung)
            _, t_train = timed(student, rung, grad=True)
            student.transformer.zero_grad(set_to_none=True)
        pair = x0_pair(prepared, schedulers, rung, prepared.video, prepared.audio, {
            "student": full_out, "teacher": t_out})
        pair_lin = x0_pair(prepared, schedulers, rung, prepared.video, prepared.audio, {
            "student": lin_out, "teacher": t_out})
        rel = {f"{name}/{m}": float((p[m]["student"] - p[m]["teacher"]).norm() / p[m]["teacher"].norm())
               for name, p in (("joint", pair), ("linears", pair_lin)) for m in ("video", "audio")}
        report[f"rung{rung}"] = {"s_teacher": round(t_teacher, 2), "s_student_linears": round(t_lin, 2),
                                 "s_student_joint": round(t_full, 2), "s_student_fwd_bwd": round(t_train, 2),
                                 **{k: round(v, 5) for k, v in rel.items()}}
    if rank0:
        print("PROFILE " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
