# SPDX-License-Identifier: Apache-2.0
"""Benchmark MiniMax-H3 few-step (DMD-style) inference: VSA-H3 vs dense attention.

Runs the same prompts (same seeds) through the full T2VA pipeline once per
attention mode and reports per-request end-to-end latency plus the
denoising-stage time (``FASTVIDEO_STAGE_LOGGING=1``):

- ``dense``: FLASH_ATTN with the FA4 CuTe kernels (``FASTVIDEO_FA4=1``).
- ``vsa``: VIDEO_SPARSE_ATTN_H3 at ``--sparsity`` (default 0.9). The
  sparsity is applied at generator boot via ``FastVideoArgs.VSA_sparsity``
  (``pipeline.experimental``); the H3 denoising stage builds per-step VSA
  metadata from it. ``--vsa-kernel cutedsl`` (default) opts into the FA4
  CuTe 256-tile forward; ``triton`` uses the 256-to-64 expansion fallback.
- ``microbench``: model-free attention-layer microbenchmark on the exact
  packed H3 sequence geometry of the requested video shape. Times
  ``block_sparse_attn_256_bshd`` (Triton and, when importable, the FA4 CuTe
  path) through the real ``MiniMaxH3VSAImpl`` tile/pool/top-k/untile path
  against dense flash attention and torch SDPA. Use it as the speedup proxy
  when the full VSA pipeline leg is unavailable.

The attention backend is resolved at generator boot, so each mode runs in a
fresh subprocess (one generator boot per mode). This also isolates the modes
from each other's CUDA state: a crash in one leg still leaves the other
legs' numbers and the final table intact.

Example (one 4-GPU node):

    FASTVIDEO_FA4=1 python examples/inference/minimax_h3/h3_vsa_dmd.py \\
        --model-path /path/to/MiniMax-H3 \\
        --prompts-json validation.json --num-prompts 4 \\
        --output-dir outputs/h3_vsa_dmd --modes dense,vsa,microbench

Caveat — sparse-trained students: with the base (dense-trained) checkpoint
this benchmark measures SPEED only. The base model was never trained under
VSA top-k masks, so at 90% sparsity output-quality parity is not expected;
the per-mode videos are written for eyeballing, but judge quality with a
VSA-trained DMD student checkpoint. Likewise the 3-step DMD ladder applied
to the base checkpoint is a latency proxy for a distilled student, not a
quality reference.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

GENERATION_MODES = ("dense", "vsa")
ALL_MODES = GENERATION_MODES + ("microbench",)

# Environment applied in the worker subprocess BEFORE importing fastvideo.
# The backend env var is folded into FastVideoArgs at generator boot.
MODE_ENV: dict[str, dict[str, str]] = {
    "dense": {
        "FASTVIDEO_ATTENTION_BACKEND": "FLASH_ATTN",
        "FASTVIDEO_FA4": "1",
    },
    "vsa": {
        # Layers that do not support VSA-H3 (e.g. the token refiner) fall
        # back to flash attention, so FA4 stays enabled here too.
        "FASTVIDEO_ATTENTION_BACKEND": "VIDEO_SPARSE_ATTN_H3",
        "FASTVIDEO_FA4": "1",
    },
    "microbench": {
        "FASTVIDEO_FA4": "1",
    },
}

DEFAULT_PROMPTS = [
    "A cinematic drone shot over coastal cliffs at sunrise, golden light, gentle ocean waves, ultra detailed.",
    "A barista pours latte art in a warm cafe, steam rising, shallow depth of field, soft morning light.",
    "A red fox trots across fresh snow between pine trees, breath visible in the cold air, tracking shot.",
    "Neon-lit rain-soaked city street at night, reflections on wet asphalt, pedestrians with umbrellas.",
]

WARMUP_SEED = 999
FIRST_SEED = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True, help="Full modular MiniMax-H3 pipeline directory")
    parser.add_argument("--prompts-json", default=None, help='Optional {"data": [{"caption": ...}]} prompt file')
    parser.add_argument("--num-prompts", type=int, default=4, help="Timed requests per mode")
    parser.add_argument("--sparsity", type=float, default=0.9, help="VSA sparsity for the vsa/microbench modes")
    parser.add_argument("--output-dir", default="outputs/h3_vsa_dmd")
    parser.add_argument("--modes", default="dense,vsa", help=f"Comma-separated subset of {ALL_MODES}")
    parser.add_argument("--dmd-steps", default="1000,667,333", help="FASTVIDEO_DMD_DENOISING_STEPS ladder")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--warmup", type=int, default=1, help="Untimed warm-up requests per mode")
    parser.add_argument("--vsa-kernel", choices=("cutedsl", "triton"), default="cutedsl")
    parser.add_argument("--mode-timeout", type=int, default=5400, help="Hard per-mode timeout in seconds")
    parser.add_argument("--microbench-text-tokens", type=int, default=300, help="Assumed text prefix length")
    parser.add_argument("--microbench-heads", default="14,56", help="Per-GPU head counts to microbench")
    parser.add_argument("--_worker", choices=ALL_MODES, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts_json:
        with open(args.prompts_json) as handle:
            rows = json.load(handle)["data"]
        prompts = [row["caption"] for row in rows[:args.num_prompts]]
    else:
        prompts = [DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)] for i in range(args.num_prompts)]
    if len(prompts) < args.num_prompts:
        raise ValueError(f"Requested {args.num_prompts} prompts but only {len(prompts)} available.")
    return prompts


def apply_worker_env(mode: str, args: argparse.Namespace) -> None:
    """Set the mode's environment. Must run before any fastvideo import."""
    env = dict(MODE_ENV[mode])
    env["FASTVIDEO_DMD_DENOISING_STEPS"] = args.dmd_steps
    env["FASTVIDEO_STAGE_LOGGING"] = "1"  # per-stage timings on the result object
    env["FASTVIDEO_VSA_CUTEDSL"] = "1" if args.vsa_kernel == "cutedsl" else "0"
    os.environ.update(env)


def denoise_seconds(result) -> float | None:
    stages = getattr(getattr(result, "logging_info", None), "stages", None)
    if not stages:
        return None
    for stage_name, metrics in stages.items():
        if "denois" in stage_name.lower():
            execution_time = metrics.get("execution_time")
            if execution_time is not None:
                return float(execution_time)
    return None


def run_generation_worker(args: argparse.Namespace) -> int:
    mode = args._worker
    apply_worker_env(mode, args)
    mode_dir = Path(args.output_dir) / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args)
    dmd_steps = [int(step) for step in args.dmd_steps.split(",") if step.strip()]

    from fastvideo import VideoGenerator
    from fastvideo.api import (
        EngineConfig,
        GenerationRequest,
        GeneratorConfig,
        OffloadConfig,
        OutputConfig,
        ParallelismConfig,
        PipelineSelection,
        SamplingConfig,
    )

    experimental: dict[str, float] = {}
    if mode == "vsa":
        # Boot-time run-level sparsity: the H3 denoising stage reads
        # fastvideo_args.VSA_sparsity (mirrored onto ForwardBatch.VSA_sparsity
        # per request) when building the per-step VSA metadata.
        experimental["VSA_sparsity"] = args.sparsity

    print(f"[{mode}] booting generator (backend={os.environ['FASTVIDEO_ATTENTION_BACKEND']}, "
          f"sparsity={experimental.get('VSA_sparsity', 0.0)}, dmd_steps={dmd_steps})",
          flush=True)
    boot_start = time.perf_counter()
    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path=args.model_path,
            engine=EngineConfig(
                num_gpus=args.num_gpus,
                use_fsdp_inference=args.num_gpus > 1,
                parallelism=ParallelismConfig(tp_size=1, sp_size=args.num_gpus),
                offload=OffloadConfig(
                    dit=False,
                    dit_layerwise=False,
                    text_encoder=True,
                    vae=True,
                    pin_cpu_memory=False,
                ),
            ),
            pipeline=PipelineSelection(experimental=experimental),
        ))
    load_time = time.perf_counter() - boot_start
    print(f"[{mode}] generator ready in {load_time:.1f}s (model load, excluded from timings)", flush=True)

    def build_request(prompt: str, seed: int, output_name: str) -> GenerationRequest:
        return GenerationRequest(
            prompt=prompt,
            negative_prompt="",
            sampling=SamplingConfig(
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                fps=24,
                num_inference_steps=len(dmd_steps),
                guidance_scale=1.0,
                batch_cfg=False,
                seed=seed,
            ),
            output=OutputConfig(
                output_path=str(mode_dir / output_name),
                save_video=True,
                return_frames=False,
            ),
        )

    records: list[dict] = []
    try:
        for warmup_index in range(args.warmup):
            print(f"[{mode}] warm-up {warmup_index} (untimed; absorbs kernel JIT/autotune)", flush=True)
            warmup_start = time.perf_counter()
            generator.generate(build_request(prompts[0], WARMUP_SEED, f"warmup{warmup_index}.mp4"))
            print(f"[{mode}] warm-up {warmup_index} done in {time.perf_counter() - warmup_start:.1f}s", flush=True)

        for index, prompt in enumerate(prompts):
            request = build_request(prompt, FIRST_SEED + index, f"prompt{index:02d}.mp4")
            request_start = time.perf_counter()
            result = generator.generate(request)
            e2e_seconds = time.perf_counter() - request_start
            record = {
                "prompt_index": index,
                "seed": FIRST_SEED + index,
                "e2e_seconds": e2e_seconds,
                "generation_seconds": result.generation_time,
                "denoise_seconds": denoise_seconds(result),
                "video_path": result.video_path,
                "peak_memory_mb": result.peak_memory_mb,
            }
            records.append(record)
            parts = [f"[{mode}] {index:02d} e2e={e2e_seconds:.1f}s"]
            if record["generation_seconds"] is not None:
                parts.append(f"gen={record['generation_seconds']:.1f}s")
            if record["denoise_seconds"] is not None:
                parts.append(f"denoise={record['denoise_seconds']:.1f}s")
            parts.append(f"-> {result.video_path}")
            print(" ".join(parts), flush=True)
    finally:
        payload = {
            "mode": mode,
            "sparsity": args.sparsity if mode == "vsa" else 0.0,
            "vsa_kernel": args.vsa_kernel if mode == "vsa" else None,
            "dmd_steps": dmd_steps,
            "shape": [args.height, args.width, args.num_frames],
            "num_gpus": args.num_gpus,
            "load_seconds": load_time,
            "requests": records,
        }
        (mode_dir / "results.json").write_text(json.dumps(payload, indent=2))
        generator.shutdown()
    return 0


def _time_cuda_call(fn, warmup: int = 3, iters: int = 10) -> float:
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def run_microbench_worker(args: argparse.Namespace) -> int:
    """Per-attention-layer proxy on the exact packed H3 geometry, single GPU.

    Measures one self-attention layer's compute (no sequence-parallel
    all-to-all, which is identical across backends): dense flash attention
    and SDPA on the true packed length vs the full VSA-H3 path
    (tile scatter + fp32 tile pooling + top-k mask + block-sparse kernel +
    untile) on the 256-padded tile buffer.
    """
    mode = args._worker
    apply_worker_env(mode, args)

    import torch

    from fastvideo.attention.backends.video_sparse_attn_h3 import (MiniMaxH3VSAImpl, MiniMaxH3VSAMetadataBuilder)
    from fastvideo.pipelines.basic.minimax_h3.packing import (MINIMAX_H3_AUDIO_CHANNELS, audio_latent_num_frames,
                                                              video_latent_num_frames)

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    head_dim = 128
    patch_size = (1, 2, 2)
    spatial_ratio = 16  # H3 video VAE spatial compression
    latent_frames = video_latent_num_frames(args.num_frames)
    latent_height, latent_width = args.height // spatial_ratio, args.width // spatial_ratio
    n_text = args.microbench_text_tokens
    n_cond = 0  # T2V: no keyframe conditioning rows
    n_audio = audio_latent_num_frames(args.num_frames) * MINIMAX_H3_AUDIO_CHANNELS
    n_video = ((latent_frames // patch_size[0]) * (latent_height // patch_size[1]) * (latent_width // patch_size[2]))
    seq_len = n_text + n_cond + n_audio + n_video
    print(f"[microbench] packed H3 sequence for {args.height}x{args.width}x{args.num_frames}: "
          f"text={n_text} (assumed) + cond={n_cond} + audio={n_audio} + video={n_video} = {seq_len} rows, "
          f"head_dim={head_dim}",
          flush=True)

    builder = MiniMaxH3VSAMetadataBuilder()
    metadata_by_sparsity = {
        sparsity: builder.build(
            current_timestep=0,
            raw_latent_shape=(latent_frames, latent_height, latent_width),
            patch_size=patch_size,
            VSA_sparsity=sparsity,
            prefix_segments=(n_text, n_cond, n_audio),
            device=device,
            exempt=True,
        )
        for sparsity in (args.sparsity, 0.0)
    }
    reference_metadata = metadata_by_sparsity[args.sparsity]
    print(f"[microbench] tiles: prefix={reference_metadata.num_prefix_tiles} "
          f"video={reference_metadata.num_video_tiles} "
          f"padded_len={int(reference_metadata.variable_block_sizes.numel()) * 256}",
          flush=True)
    impl = MiniMaxH3VSAImpl(num_heads=0, head_size=head_dim, causal=False, softmax_scale=1.0, prefix="blocks.0.attn")

    flash_attn_func = None
    fa_version = None
    try:
        from fastvideo.attention.utils import flash_attn_default
        flash_attn_func = flash_attn_default.flash_attn_func
        fa_version = flash_attn_default.fa_version
    except Exception as error:  # noqa: BLE001 - report and continue with SDPA only
        print(f"[microbench] flash attention unavailable ({error}); dense rows fall back to SDPA only", flush=True)

    rows: list[dict] = []
    head_counts = [int(h) for h in args.microbench_heads.split(",") if h.strip()]
    for num_heads in head_counts:
        note = "per-rank slice of the sp=4 run" if num_heads == 14 else "full model on one GPU"
        print(f"[microbench] heads={num_heads} ({note})", flush=True)
        qkv = torch.randn(3, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        query, key, value = (t.contiguous() for t in qkv.unbind(0))

        def record(name: str, milliseconds: float, num_heads: int = num_heads) -> None:
            rows.append({"heads": num_heads, "name": name, "ms": milliseconds})
            print(f"[microbench]   {name:<34} {milliseconds:9.3f} ms/layer-call", flush=True)

        if flash_attn_func is not None:
            def dense_flash(q=query[None], k=key[None], v=value[None]):
                out = flash_attn_func(q, k, v)
                return out[0] if isinstance(out, tuple) else out

            record(f"dense flash (FA{fa_version})", _time_cuda_call(dense_flash) * 1e3)

        def dense_sdpa(q=query[None], k=key[None], v=value[None]):
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))

        record("dense torch SDPA", _time_cuda_call(dense_sdpa) * 1e3)

        kernel_choices = ["triton"]
        if args.vsa_kernel == "cutedsl":
            kernel_choices.insert(0, "cutedsl")
        for kernel in kernel_choices:
            os.environ["FASTVIDEO_VSA_CUTEDSL"] = "1" if kernel == "cutedsl" else "0"
            for sparsity, metadata in metadata_by_sparsity.items():
                def vsa_layer(qkv=qkv, metadata=metadata):
                    tiled = impl.preprocess_qkv(qkv, metadata)
                    q, k, v = tiled.chunk(3, dim=0)
                    out = impl.forward(q, k, v, None, metadata)
                    return impl.postprocess_output(out, metadata)

                label = f"VSA-H3 {kernel} sparsity={sparsity:.2f}"
                try:
                    record(label, _time_cuda_call(vsa_layer) * 1e3)
                except Exception as error:  # noqa: BLE001 - a kernel path may be uninstalled
                    print(f"[microbench]   {label:<34} FAILED: {error}", flush=True)
                    rows.append({"heads": num_heads, "name": label, "ms": None, "error": str(error)})

    mode_dir = Path(args.output_dir) / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "sparsity": args.sparsity,
        "geometry": {
            "seq_len": seq_len,
            "text": n_text,
            "cond": n_cond,
            "audio": n_audio,
            "video": n_video,
            "head_dim": head_dim,
        },
        "note": ("per-layer self-attention compute only, single GPU, excludes the sequence-parallel "
                 "all-to-all (identical across backends); heads=14 matches one rank of the 4-GPU sp run"),
        "rows": rows,
    }
    (mode_dir / "results.json").write_text(json.dumps(payload, indent=2))
    return 0


def run_mode_subprocess(mode: str, args: argparse.Namespace) -> dict:
    """Run one mode in a fresh interpreter; stream output and survive crashes."""
    mode_dir = Path(args.output_dir) / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, os.path.abspath(__file__), "--_worker", mode]
    for key, value in vars(args).items():
        if key in ("_worker",) or value is None:
            continue
        command.extend([f"--{key.replace('_', '-')}", str(value)])
    child_env = dict(os.environ, PYTHONUNBUFFERED="1")

    print(f"\n=== mode {mode}: launching worker ===", flush=True)
    start = time.perf_counter()
    process = subprocess.Popen(
        command,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    def kill_group() -> None:
        print(f"=== mode {mode}: timeout after {args.mode_timeout}s, killing process group ===", flush=True)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    watchdog = threading.Timer(args.mode_timeout, kill_group)
    watchdog.start()
    tail: deque[str] = deque(maxlen=60)
    log_path = mode_dir / "worker.log"
    with open(log_path, "w") as log_file:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            tail.append(line.rstrip("\n"))
    return_code = process.wait()
    watchdog.cancel()
    elapsed = time.perf_counter() - start

    results_path = mode_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else None
    status = {"mode": mode, "return_code": return_code, "elapsed_seconds": elapsed, "results": results}
    if return_code != 0 or results is None:
        signature = [line for line in tail if any(token in line for token in
                                                  ("Error", "error", "Traceback", "CUDA", "NCCL", "Signal",
                                                   "Segmentation", "terminate", "Killed"))]
        status["crash_signature"] = signature[-15:] or list(tail)[-15:]
        (mode_dir / "crash_signature.txt").write_text("\n".join(status["crash_signature"]) + "\n")
        print(f"=== mode {mode}: FAILED (rc={return_code}); signature saved to {mode_dir / 'crash_signature.txt'} ===",
              flush=True)
    else:
        print(f"=== mode {mode}: completed in {elapsed:.0f}s ===", flush=True)
    return status


def _mode_stats(status: dict) -> dict | None:
    results = status.get("results")
    if not results or not results.get("requests"):
        return None
    requests = results["requests"]
    generation = [r["generation_seconds"] for r in requests if r.get("generation_seconds") is not None]
    denoise = [r["denoise_seconds"] for r in requests if r.get("denoise_seconds") is not None]
    return {
        "n": len(requests),
        "load": results.get("load_seconds"),
        "e2e": statistics.mean(r["e2e_seconds"] for r in requests),
        "gen": statistics.mean(generation) if generation else None,
        "denoise": statistics.mean(denoise) if denoise else None,
    }


def _fmt(value: float | None, width: int) -> str:
    return f"{value:>{width}.1f}" if value is not None else f"{'-':>{width}}"


def _speedup(dense: dict | None, row: dict, metric: str) -> str:
    if dense is None or dense.get(metric) is None or row.get(metric) in (None, 0):
        return "-"
    return f"{dense[metric] / row[metric]:.2f}x"


def summarize(statuses: list[dict], args: argparse.Namespace) -> None:
    stats = {status["mode"]: _mode_stats(status) for status in statuses if status["mode"] in GENERATION_MODES}
    dense = stats.get("dense")

    print("\n================ H3 DMD 3-step inference: attention backend benchmark ================")
    print(f"shape={args.height}x{args.width}x{args.num_frames}  gpus={args.num_gpus}  "
          f"dmd_steps={args.dmd_steps}  vsa sparsity={args.sparsity} ({args.vsa_kernel})")
    header = (f"{'mode':<12} {'n':>3} {'load(s)':>9} {'mean e2e(s)':>12} {'mean gen(s)':>12} "
              f"{'mean denoise(s)':>16} {'e2e speedup':>12} {'denoise speedup':>16}")
    print(header)
    print("-" * len(header))
    for mode in ("dense", "vsa"):
        label = f"vsa@{args.sparsity:.2f}" if mode == "vsa" else mode
        row = stats.get(mode)
        if row is None:
            if any(status["mode"] == mode for status in statuses):
                print(f"{label:<12} {'-':>3} {'-':>9} {'CRASHED':>12} {'-':>12} {'-':>16} {'-':>12} {'-':>16}")
            continue
        print(f"{label:<12} {row['n']:>3} {_fmt(row['load'], 9)} {_fmt(row['e2e'], 12)} {_fmt(row['gen'], 12)} "
              f"{_fmt(row['denoise'], 16)} {_speedup(dense, row, 'e2e'):>12} "
              f"{_speedup(dense, row, 'denoise'):>16}")

    micro = next((status for status in statuses if status["mode"] == "microbench"), None)
    if micro and micro.get("results"):
        results = micro["results"]
        geometry = results["geometry"]
        print(f"\nAttention-layer microbench (seq={geometry['seq_len']} rows: text {geometry['text']} + "
              f"audio {geometry['audio']} + video {geometry['video']}; {results['note']}):")
        for row in results["rows"]:
            timing = f"{row['ms']:.3f} ms" if row.get("ms") is not None else f"FAILED: {row.get('error')}"
            print(f"  heads={row['heads']:>2}  {row['name']:<34} {timing}")
    print("\nNote: base checkpoint is dense-trained; the vsa leg measures speed, not quality parity.")


def main() -> None:
    args = parse_args()
    if args._worker in GENERATION_MODES:
        sys.exit(run_generation_worker(args))
    if args._worker == "microbench":
        sys.exit(run_microbench_worker(args))

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown = sorted(set(modes) - set(ALL_MODES))
    if unknown:
        raise ValueError(f"Unknown modes {unknown}; choose from {ALL_MODES}.")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args)
    (out_root / "prompts.txt").write_text("\n\n".join(f"[{i:02d}] {p}" for i, p in enumerate(prompts)))

    statuses = [run_mode_subprocess(mode, args) for mode in modes]
    summarize(statuses, args)
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(statuses, indent=2))
    print(f"\nPer-mode outputs and summary under: {out_root}")
    sys.exit(0 if all(status["return_code"] == 0 and status["results"] is not None for status in statuses) else 2)


if __name__ == "__main__":
    main()
