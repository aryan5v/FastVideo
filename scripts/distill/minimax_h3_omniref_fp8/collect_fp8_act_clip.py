# SPDX-License-Identifier: Apache-2.0
"""MSE-optimal static FP8 activation clip per DiT linear input, from teacher-forced OmniRef training rows.

For every main-block linear input (``b<idx>.attn.to_q`` - shared by q/k/v and the bf16 gate -,
``attn.to_out``, ``ff.fc_in``, ``ff.fc_out``) the script walks the bf16 student (== teacher) along its own
PDD ladder on ``--rows`` training rows and accumulates, for each candidate clip ``fraction * amax`` of the
1000-row max-calibration table, the squared error of rounding the input to float8_e4m3fn with the static
scale ``clip / 448`` (values beyond the clip saturate). It also accumulates the error of the deployed
per-token dynamic scale, so the two can be compared. The chosen clip per key is the candidate with the least
pooled squared error (the NVIDIA Model-Optimizer ``mse`` calibrator's criterion, on this activation set).

Output JSON: ``{key: {"amax_max": table amax, "amax_mse": chosen clip, "fraction": f, "rel_mse": {f: ...},
"rel_mse_dynamic": ...}}``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
FRACTIONS = (1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.015, 0.007)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-table", required=True, help="JSON {key: amax} from the 1000-row calibration")
    parser.add_argument("--rows", type=int, default=48)
    args, overrides = parser.parse_known_args()

    import sys
    sys.path.insert(0, str(HERE))
    from fp8_student import FP8_MAX, FP8_MIN_SCALE, fake_quant_activation, named_fp8_linears

    from fastvideo.distributed import get_world_group, maybe_init_distributed_environment_and_model_parallel
    from fastvideo.train.methods.knowledge_distillation.pdd_qad_metrics import step_packed
    from fastvideo.train.utils.builder import build_from_config
    from fastvideo.train.utils.config import load_run_config

    cfg = load_run_config(args.config, overrides=overrides or None)
    maybe_init_distributed_environment_and_model_parallel(cfg.training.distributed.tp_size,
                                                          cfg.training.distributed.sp_size)
    _, method, _, _ = build_from_config(cfg)
    table = json.loads(Path(args.max_table).read_text())
    device = method.student.device

    err = {k: torch.zeros(len(FRACTIONS), dtype=torch.float64, device=device) for k in table}
    err_dyn = {k: torch.zeros((), dtype=torch.float64, device=device) for k in table}
    energy = {k: torch.zeros((), dtype=torch.float64, device=device) for k in table}
    seen_amax = {k: torch.zeros((), dtype=torch.float32, device=device) for k in table}

    def make_hook(key: str):

        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            x = inputs[0].reshape(-1, inputs[0].shape[-1])
            xf = x.float()
            energy[key] += xf.square().sum(dtype=torch.float64)
            seen_amax[key] = torch.maximum(seen_amax[key], xf.abs().max())
            err_dyn[key] += (fake_quant_activation(x).float() - xf).square().sum(dtype=torch.float64)
            for i, fraction in enumerate(FRACTIONS):
                scale = max(float(table[key]) * fraction / FP8_MAX, FP8_MIN_SCALE)
                q = (xf * (1.0 / scale)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * scale
                err[key][i] += (q - xf).square().sum(dtype=torch.float64)

        return hook

    handles, seen = [], set()
    for _, key, module in named_fp8_linears(method.student.transformer, include_refiner=False):
        if key in table and key not in seen:  # q/k/v share one entry: hook the to_q module only
            seen.add(key)
            handles.append(module.register_forward_pre_hook(make_hook(key)))
    missing = sorted(set(table) - seen)
    if missing:
        raise KeyError(f"calibration keys without a hooked module: {missing[:6]}")

    student = method.student
    specs = student.train_rows
    stride = max(len(specs) // args.rows, 1)
    chosen = specs[::stride][:args.rows]
    dp_rank, dp_size = method.dp_rank, method.dp_size
    with torch.no_grad():
        for position, spec in enumerate(chosen):
            if position % dp_size != dp_rank:
                continue
            prepared = method._row(spec)
            schedulers = student.schedulers()
            timesteps = (schedulers[0].timesteps, schedulers[1].timesteps)
            video, audio = prepared.video, prepared.audio
            for rung in range(student.num_rungs):
                with student.rung_scope(prepared, rung, timesteps) as rt:
                    out = student.forward_rung(prepared, video, audio, rt)
                video, audio = step_packed(schedulers, prepared, rung, video, audio, *out)
            if get_world_group().rank == 0:
                print(f"row {position + 1}/{len(chosen)} {spec.id}", flush=True)
    for handle in handles:
        handle.remove()

    keys = sorted(table)
    flat = torch.stack([torch.cat([err[k], err_dyn[k][None], energy[k][None]]) for k in keys])
    amax_flat = torch.stack([seen_amax[k] for k in keys])
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(flat)
        torch.distributed.all_reduce(amax_flat, op=torch.distributed.ReduceOp.MAX)
    if get_world_group().rank == 0:
        result = {}
        for row, key, observed in zip(flat.tolist(), keys, amax_flat.tolist(), strict=True):
            static, dynamic, total = row[:len(FRACTIONS)], row[len(FRACTIONS)], row[len(FRACTIONS) + 1]
            best = min(range(len(FRACTIONS)), key=lambda i: static[i])
            result[key] = {
                "amax_max": float(table[key]),
                "amax_mse": float(table[key]) * FRACTIONS[best],
                "fraction": FRACTIONS[best],
                "observed_amax_subset": observed,
                "rel_mse": {str(f): s / total for f, s in zip(FRACTIONS, static, strict=True)},
                "rel_mse_dynamic": dynamic / total,
            }
        Path(args.out).write_text(json.dumps(result, indent=1))
        print(f"wrote {args.out}: {len(result)} keys over {len(chosen)} rows", flush=True)


if __name__ == "__main__":
    main()
