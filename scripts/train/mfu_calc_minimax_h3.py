#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MFU calculator for MiniMax-H3 DMD2 / SFT training (GB200, packed T2VA).

Turns measured trainer step times (W&B ``step_time_sec``, split by the
``update_student`` flag for DMD2) into Model FLOPs Utilization under two
conventions:

  dense-equiv   attention FLOPs counted as if every layer were dense and the
                student had no VSA gate projection — the model-work convention
                used when quoting sparse-attention speedups.
  actual        attention FLOPs at the realized keep fraction of the VSA-H3
                block mask (exempt mode) plus the student's extra
                ``to_gate_compress`` GEMM — honest hardware utilization.

Everything is analytic and CPU-only; no fastvideo import.

Architecture (verified against fastvideo/configs/models/dits/minimax_h3.py and
fastvideo/models/dits/minimax_h3.py):
  hidden 5376, 50 blocks + 2 text-refiner blocks, 56 heads x head_dim 128
  (attention inner dim 7168 != hidden), SwiGLU ffn_dim 14336 (fc_in emits
  2*ffn), bias-free block GEMMs, per-block AdaLN 2688 -> 6*5376*3 (applied per
  timestep row, not per token -> negligible FLOPs, huge params).

FLOPs accounting (per model forward over one packed sequence of S tokens):
  linears     2 * P_active * S, where P_active is the sum of per-token GEMM
              weight elements (block QKV/out + SwiGLU; NOT the AdaLN tables).
  attention   4 * n_pairs * inner_dim  (2 for QK^T + 2 for PV), n_pairs = S^2
              dense; under VSA-H3 exempt mode: prefix (text+audio) queries stay
              dense, video queries attend prefix keys plus the top-k kept video
              tiles, keep = ceil((1-sparsity)*n_video_tiles)/n_video_tiles.
  backward    2x forward.
  recompute   full activation checkpointing replays 1x forward inside every
              backward, so a grad-mode "unit" costs fwd + recompute + bwd = 4x
              forward FLOPs; a no-grad forward costs 1x.
  ignored     RMSNorm/RoPE/softmax/elementwise, the pooled tile-score matmuls
              (~0.5 GFLOP/layer), and kernel padding of partial 256-token tiles
              (mask keep-frac 0.139 at tile level vs 0.133 counted here).

DMD2 per-sequence step recipes (verified against
fastvideo/train/methods/distribution_matching/dmd2.py, simulate rollout with a
3-step ladder and real_score_guidance_scale=1.0):
  critic step   3 no-grad student fwd (rollout) + critic grad unit
              = 3*F_s + 4*F_d
  student step  2 no-grad student fwd + student grad unit + 1 no-grad critic
                fwd + 1 no-grad teacher fwd  = 6*F_s + 2*F_d
  sft step      student grad unit = 4*F_s
Multiply by gradient-accumulation rounds; every DP rank runs its own sequence.
Teacher/critic are always dense; the student is dense (v6) or VSA-H3 (v7+).

MFU = accum * dp * F_step / (num_gpus * step_time * peak),  dp = gpus / sp.

Peak: default 2.25e15 FLOP/s = NVIDIA's dense (non-sparse) BF16 peak per
B200/GB200 GPU. MFU scales inversely with this; pass --peak to change it.

Examples:
  # v7 VSA-90 production (Triton or CuTe — just remeasure step times):
  python mfu_calc_minimax_h3.py --step-type dmd-critic  --step-time 83.65 \
      --gpus 32 --accum 2 --student-backend vsa --sparsity 0.9
  python mfu_calc_minimax_h3.py --step-type dmd-student --step-time 71.27 \
      --gpus 32 --accum 2 --student-backend vsa --sparsity 0.9
  # blended 4:1 cadence in one shot:
  python mfu_calc_minimax_h3.py --step-type dmd-blend \
      --step-time-critic 83.65 --step-time-student 71.27 \
      --gpus 32 --accum 2 --student-backend vsa --sparsity 0.9
  # v6 dense baseline:
  python mfu_calc_minimax_h3.py --step-type dmd-blend \
      --step-time-critic 51.2 --step-time-student 57.75 \
      --gpus 32 --accum 1 --student-backend dense
  # SFT overfit probe (4 GPUs, SP=4):
  python mfu_calc_minimax_h3.py --step-type sft --step-time 3.823 \
      --gpus 4 --sp 4 --student-backend vsa --sparsity 0.9
"""

from __future__ import annotations

import argparse
import math

# ---------------------------------------------------------------- architecture
HIDDEN = 5376
HEADS = 56
HEAD_DIM = 128
INNER = HEADS * HEAD_DIM  # 7168 — attention width != hidden
FFN = 14336
LAYERS = 50
REFINER_LAYERS = 2
TEXT_DIM = 5120
TIME_EMBED_DIM = 2688
TIME_EMBED_HIDDEN = 5376
FREQ_DIM = 256
VIDEO_PATCH_DIM = 24 * 1 * 2 * 2  # in_channels * prod(patch_size)
AUDIO_IN = 32
MODALITIES = 3

# Per-token GEMM weight elements in one transformer block (bias-free).
BLOCK_ATTN_GEMM = 3 * HIDDEN * INNER + INNER * HIDDEN  # q,k,v,out
BLOCK_FFN_GEMM = HIDDEN * (2 * FFN) + FFN * HIDDEN  # SwiGLU fc_in/fc_out
BLOCK_GEMM = BLOCK_ATTN_GEMM + BLOCK_FFN_GEMM  # 385,351,680
GATE_GEMM = HIDDEN * INNER  # to_gate_compress (VSA student only)
ADALN_GEMM = TIME_EMBED_DIM * 6 * HIDDEN * MODALITIES  # per timestep row

# ------------------------------------------------------------ default sequence
TEXT_TOKENS = 300  # varies per prompt; ~300 for the VidProM/synth mix
AUDIO_TOKENS = 414  # 207 audio latents x 2 channel rows @ 124 frames
VIDEO_GRID = (37, 24, 42)  # latent (37,48,84) patched (1,2,2) @ 768x1344x124
TILE = (4, 8, 8)  # VSA-H3 256-token video tiles

PEAK_BF16_DENSE = 2.25e15  # NVIDIA B200/GB200 dense BF16 FLOP/s per GPU


def param_count() -> dict[str, float]:
    """Analytic parameter count of the dense H3 DiT (for reconciliation)."""
    block = BLOCK_GEMM + 2 * HEAD_DIM + ADALN_GEMM + 6 * HIDDEN * MODALITIES + 2 * HIDDEN
    refiner_block = BLOCK_GEMM + 2 * HEAD_DIM + 2 * HIDDEN
    other = ((VIDEO_PATCH_DIM + 1) * HIDDEN + (AUDIO_IN + 1) * HIDDEN + (TEXT_DIM + 1) * HIDDEN +
             (FREQ_DIM + 1) * TIME_EMBED_HIDDEN + (TIME_EMBED_HIDDEN + 1) * TIME_EMBED_DIM + HIDDEN +
             (HIDDEN + TIME_EMBED_DIM * 2 * HIDDEN + 2 * HIDDEN) + (HIDDEN + 1) * VIDEO_PATCH_DIM +
             (HIDDEN + 1) * AUDIO_IN)
    total = LAYERS * block + REFINER_LAYERS * refiner_block + other
    return {
        "total_dense": total,
        "per_block": block,
        "vsa_gate_extra": LAYERS * GATE_GEMM,
        "active_gemm_per_token": LAYERS * BLOCK_GEMM,
    }


def _video_tiles(grid: tuple[int, int, int]) -> int:
    return math.prod(math.ceil(g / t) for g, t in zip(grid, TILE))


def attention_pairs(text: int, audio: int, video: int, sparsity: float) -> float:
    """Attended (query, key) token pairs per layer under VSA-H3 exempt mode."""
    s = text + audio + video
    if sparsity <= 0.0:
        return float(s) * s
    n_vid_tiles = _video_tiles(VIDEO_GRID)
    keep_tiles = max(1, min(math.ceil((1.0 - sparsity) * n_vid_tiles), n_vid_tiles))
    keep_frac = keep_tiles / n_vid_tiles
    prefix = text + audio
    # prefix queries are always dense; video queries see prefix keys (exempt)
    # plus keep_frac of the video keys (top-k tiles, sizes ~uniform on average)
    return float(prefix) * s + float(video) * (prefix + keep_frac * video)


def forward_flops(
    *,
    text: int = TEXT_TOKENS,
    audio: int = AUDIO_TOKENS,
    video: int | None = None,
    vsa: bool = False,
    sparsity: float = 0.9,
    gate_active: bool = True,
) -> dict[str, float]:
    """FLOPs of ONE transformer forward over one packed sequence."""
    if video is None:
        video = math.prod(VIDEO_GRID)
    s = text + audio + video
    linear = 2.0 * s * (LAYERS * BLOCK_GEMM)
    # text refiner (2 plain blocks over the text stream only)
    refiner = REFINER_LAYERS * (2.0 * text * BLOCK_GEMM + 4.0 * text * text * INNER)
    # io projections + embedders + AdaLN tables (~2 timestep rows) — tiny:
    # proj_out/audio_proj_out run over the whole packed sequence, proj_in and
    # audio_proj_in over their own modality rows only.
    io = 2.0 * (text * TEXT_DIM * HIDDEN + (video + s) * VIDEO_PATCH_DIM * HIDDEN + (audio + s) * AUDIO_IN * HIDDEN +
                2 * (FREQ_DIM * TIME_EMBED_HIDDEN + TIME_EMBED_HIDDEN * TIME_EMBED_DIM) + 2 *
                (LAYERS * ADALN_GEMM + TIME_EMBED_DIM * 2 * HIDDEN))
    pairs = attention_pairs(text, audio, video, sparsity if vsa else 0.0)
    attn = LAYERS * 4.0 * INNER * pairs
    gate = LAYERS * 2.0 * s * GATE_GEMM if (vsa and gate_active) else 0.0
    total = linear + refiner + io + attn + gate
    return {"linear": linear, "attention": attn, "gate": gate, "refiner+io": refiner + io, "total": total}


def step_flops(
    step_type: str,
    *,
    student_vsa: bool,
    sparsity: float,
    convention: str,
    gate_active: bool = True,
    teacher_forwards: int = 1,
    text: int = TEXT_TOKENS,
) -> dict[str, float]:
    """FLOPs of one micro-round (one sequence) for a given step type."""
    dense = forward_flops(text=text, vsa=False)["total"]
    if convention == "dense-equiv" or not student_vsa:
        f_s = dense
    else:
        f_s = forward_flops(text=text, vsa=True, sparsity=sparsity, gate_active=gate_active)["total"]
    if step_type == "dmd-critic":
        # 3 no-grad student rollout fwd + critic grad unit (fwd+recompute+bwd)
        parts = {"student_fwd(no-grad)": 3 * f_s, "critic_grad_unit": 4 * dense}
    elif step_type == "dmd-student":
        # rollout (2 no-grad + 1 grad fwd) + student bwd + recompute
        # + 1 no-grad critic fwd + 1 no-grad teacher fwd (guidance scale 1)
        parts = {
            "student_fwd(no-grad)": 2 * f_s,
            "student_grad_unit": 4 * f_s,
            "critic_fwd(no-grad)": dense,
            "teacher_fwd(no-grad)": teacher_forwards * dense,
        }
    elif step_type == "sft":
        parts = {"student_grad_unit": 4 * f_s}
    else:
        raise ValueError(f"unknown step type {step_type!r}")
    parts["total"] = sum(parts.values())
    return parts


def mfu(step_total_flops: float, step_time: float, gpus: int, sp: int, accum: int, peak: float) -> float:
    dp = gpus // sp
    job_flops = accum * dp * step_total_flops
    return job_flops / (gpus * step_time * peak)


def _report_one(args: argparse.Namespace, step_type: str, step_time: float) -> None:
    print(f"--- {step_type}  (t = {step_time:.3f} s/step, {args.gpus} GPUs, sp={args.sp}, "
          f"accum={args.accum}, student={args.student_backend}"
          f"{f', sparsity={args.sparsity}' if args.student_backend == 'vsa' else ''})")
    for conv in ("dense-equiv", "actual"):
        parts = step_flops(
            step_type,
            student_vsa=args.student_backend == "vsa",
            sparsity=args.sparsity,
            convention=conv,
            gate_active=not args.no_gate,
            teacher_forwards=args.teacher_forwards,
            text=args.text_tokens,
        )
        value = mfu(parts["total"], step_time, args.gpus, args.sp, args.accum, args.peak)
        detail = "  ".join(f"{k}={v / 1e15:.2f}" for k, v in parts.items() if k != "total")
        print(f"  {conv:<12} MFU = {value * 100:6.2f} %   "
              f"({parts['total'] / 1e15:.2f} PFLOP/seq-step: {detail})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step-type",
                        choices=["dmd-critic", "dmd-student", "dmd-blend", "sft"],
                        required=True,
                        help="dmd-blend needs --step-time-critic/--step-time-student and "
                        "reports the 4:1 cadence blend as well as each step type")
    parser.add_argument("--step-time", type=float, help="measured s/step (W&B step_time_sec)")
    parser.add_argument("--step-time-critic", type=float, help="s/step of critic-only steps")
    parser.add_argument("--step-time-student", type=float, help="s/step of student-update steps")
    parser.add_argument("--gpus", type=int, default=32)
    parser.add_argument("--sp", type=int, default=1, help="sequence-parallel size (SFT probes use 4)")
    parser.add_argument("--accum", type=int, default=2, help="gradient accumulation rounds per step")
    parser.add_argument("--sparsity", type=float, default=0.9)
    parser.add_argument("--student-backend", choices=["vsa", "dense"], default="vsa")
    parser.add_argument("--no-gate", action="store_true", help="exclude the to_gate_compress GEMM from 'actual'")
    parser.add_argument("--teacher-forwards", type=int, default=1, help="2 if real_score_guidance_scale != 1")
    parser.add_argument("--text-tokens", type=int, default=TEXT_TOKENS)
    parser.add_argument("--peak", type=float, default=PEAK_BF16_DENSE, help="per-GPU dense BF16 peak FLOP/s")
    parser.add_argument("--params", action="store_true", help="print the analytic parameter reconciliation")
    args = parser.parse_args()

    if args.params:
        p = param_count()
        print(f"dense H3 DiT params        : {p['total_dense'] / 1e9:.3f} B "
              f"(bf16 checkpoint {2 * p['total_dense'] / 2**30:.1f} GiB)")
        print(f"VSA gate extra (student)   : {p['vsa_gate_extra'] / 1e9:.3f} B")
        print(f"per-token GEMM params      : {p['active_gemm_per_token'] / 1e9:.3f} B (50 blocks)")
        print(f"DCP fp32 params+moments    : student+critic dense = "
              f"{24 * p['total_dense'] / 2**30:.0f} GiB (measured 741 GiB)")
        print()

    if args.gpus % args.sp:
        raise SystemExit("--gpus must be divisible by --sp")
    print(f"peak assumed: {args.peak / 1e15:.2f} PFLOP/s dense BF16 per GPU "
          f"(MFU scales as 1/peak; sequence S = {args.text_tokens + AUDIO_TOKENS + math.prod(VIDEO_GRID)})")

    if args.step_type == "dmd-blend":
        if args.step_time_critic is None or args.step_time_student is None:
            raise SystemExit("dmd-blend requires --step-time-critic and --step-time-student")
        _report_one(args, "dmd-critic", args.step_time_critic)
        _report_one(args, "dmd-student", args.step_time_student)
        blend_time = 4 * args.step_time_critic + args.step_time_student
        print(f"--- dmd 4:1 blend  (5 steps = {blend_time:.1f} s, "
              f"{5 * args.accum * (args.gpus // args.sp)} sequences)")
        for conv in ("dense-equiv", "actual"):
            kw = dict(
                student_vsa=args.student_backend == "vsa",
                sparsity=args.sparsity,
                convention=conv,
                gate_active=not args.no_gate,
                teacher_forwards=args.teacher_forwards,
                text=args.text_tokens,
            )
            flops5 = 4 * step_flops("dmd-critic", **kw)["total"] + step_flops("dmd-student", **kw)["total"]
            value = mfu(flops5, blend_time, args.gpus, args.sp, args.accum, args.peak)
            print(f"  {conv:<12} MFU = {value * 100:6.2f} %")
    else:
        if args.step_time is None:
            raise SystemExit(f"{args.step_type} requires --step-time")
        _report_one(args, args.step_type, args.step_time)


if __name__ == "__main__":
    main()
