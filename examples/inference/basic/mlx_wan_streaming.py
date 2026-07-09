# SPDX-License-Identifier: Apache-2.0
"""Streaming (block-autoregressive) causal Wan generation on Apple Silicon (MLX).

Track C, rung 5 demo + rung 6 benchmark. Loads the released Self-Forcing causal
checkpoint into ``MLXCausalWanDiT`` and generates block-by-block, printing each
block's latency as it finalizes — the streaming behaviour a live preview would
show. Reports the headline latency numbers: time-to-first-block, steady per-block
latency, and peak memory.

With ``--decode``, each finalized latent block is decoded via TAEHV and the
growing frame list is rewritten to an MP4 so frames appear live. With a positive
``--local-attn-size``, the rolling KV cache is bounded (long-video path).

The denoise-side metrics are content-independent, so ``--random-prompt`` (default
when no text encoder is present in the model root) uses random text embeddings —
enough to measure latency exactly. Supply a full pipeline root (transformer +
text_encoder + tokenizer [+ vae]) to encode a real prompt and decode frames.

Example (transformer-only root, latency benchmark):

    python examples/inference/basic/mlx_wan_streaming.py \
      --model-root ~/models/sfwan_t2v_1.3b \
      --height 480 --width 832 --num-frames 21

Example (bounded window + live TAEHV decode):

    python examples/inference/basic/mlx_wan_streaming.py \
      --model-root ~/models/sfwan_t2v_1.3b \
      --local-attn-size 6 --sink-size 1 \
      --decode --output-video video_samples/stream_long.mp4 \
      --num-frames 24
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np


def _mx_dtype(name: str):
    import mlx.core as mx

    return {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}[name]


def _try_load_taehv(*, checkpoint_path: Path | None, source_path: Path | None, device, dtype):
    """Load TAEHV for per-block decode, or return None with a printed note."""
    try:
        from fastvideo.mlx_runtime.taehv_decode import ensure_taew2_1_checkpoint
    except Exception as exc:  # noqa: BLE001 - optional path.
        print(f"[decode] TAEHV helpers unavailable ({type(exc).__name__}: {exc}); skipping decode.")
        return None

    try:
        ckpt = ensure_taew2_1_checkpoint(checkpoint_path)
    except FileNotFoundError as exc:
        print(f"[decode] TAEHV checkpoint not found ({exc}); skipping decode. "
              "Pass --taehv-checkpoint-path taew2_1.pth once you have the weights.")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[decode] could not resolve TAEHV checkpoint ({type(exc).__name__}: {exc}); skipping decode.")
        return None

    try:
        if source_path is not None:
            import importlib.util

            spec = importlib.util.spec_from_file_location("fastvideo_external_taehv", source_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load TAEHV source from {source_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            TAEHV = module.TAEHV
        else:
            from fastvideo.third_party.taehv import TAEHV

        taehv = TAEHV(str(ckpt)).to(device=device, dtype=dtype)
        taehv.eval()
        print(f"[decode] loaded TAEHV from {ckpt}")
        return taehv
    except Exception as exc:  # noqa: BLE001
        print(f"[decode] failed to construct TAEHV ({type(exc).__name__}: {exc}); skipping decode. "
              "Note: imageio-ffmpeg is required for MP4 export.")
        return None


def _decode_block_frames(taehv, latents_np: np.ndarray, *, device, dtype, parallel: bool) -> np.ndarray:
    """Decode one latent block ``[1,C,T,H,W]`` → pixel frames ``[Tp,H,W,3]`` float in [0,1]-ish."""
    import torch

    latents = torch.from_numpy(latents_np).to(device=device, dtype=dtype)
    with torch.no_grad():
        video_ntchw = taehv.decode_video(
            latents.transpose(1, 2),
            parallel=parallel,
            show_progress_bar=False,
        )
    video = video_ntchw.transpose(1, 2)  # N C T H W
    return video[0].permute(1, 2, 3, 0).float().cpu().numpy()


def _write_mp4(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    from diffusers.utils import export_to_video

    if not frames:
        return
    video_np = np.concatenate(frames, axis=0)
    # Clamp for export_to_video (expects float in ~[0,1]).
    video_np = np.clip(video_np, 0.0, 1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(video_np, str(output_path), fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX causal Wan streaming demo + latency benchmark.")
    parser.add_argument("--model-root", type=Path, required=True,
                        help="Root with transformer/ (and optionally text_encoder/, tokenizer/, vae/).")
    parser.add_argument("--prompt", default="A fox runs through a mossy forest.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=21, help="Latent frames (== blocks when nfb=1).")
    parser.add_argument("--dmd-denoising-steps", default="1000,750,500,250")
    parser.add_argument("--flow-shift", type=float, default=8.0)
    parser.add_argument("--num-frames-per-block", type=int, default=1)
    parser.add_argument(
        "--local-attn-size",
        type=int,
        default=-1,
        help="Rolling KV window in latent frames (-1 = global compat window of 21). "
        "Positive values bound memory for long video.",
    )
    parser.add_argument(
        "--sink-size",
        type=int,
        default=0,
        help="Number of latent frames kept as attention sinks under eviction.",
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        default=False,
        help="Decode each finalized latent block with TAEHV and rewrite a growing MP4.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("video_samples/mlx_wan_streaming.mp4"),
        help="MP4 path used when --decode is set (rewritten after each block).",
    )
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--taehv-checkpoint-path", type=Path, default=None, help="Path to taew2_1.pth.")
    parser.add_argument("--taehv-source-path", type=Path, default=None, help="Optional local TAEHV .py override.")
    parser.add_argument("--taehv-parallel", action="store_true", default=False)
    parser.add_argument("--mlx-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--mlx-quantization", choices=("none", "int8", "int4"), default="none")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--metrics-out", type=Path, default=None)
    args = parser.parse_args()

    import mlx.core as mx

    from fastvideo.mlx_runtime.causal_dit import mlx_causal_dit_from_diffusers_safetensors
    from fastvideo.mlx_runtime.causal_sampler import build_dmd_schedule, stream_causal_latents

    dtype = _mx_dtype(args.mlx_dtype)
    quantization = None if args.mlx_quantization == "none" else args.mlx_quantization
    config = json.loads((args.model_root / "transformer" / "config.json").read_text())
    latent_h, latent_w = args.height // 8, args.width // 8
    patch = tuple(config["patch_size"])
    frame_seqlen = (latent_h // patch[1]) * (latent_w // patch[2])

    mx.clear_cache()
    mx.reset_peak_memory()
    load_start = time.perf_counter()
    model = mlx_causal_dit_from_diffusers_safetensors(
        args.model_root / "transformer" / "diffusion_pytorch_model.safetensors",
        args.model_root / "transformer" / "config.json",
        dtype=args.mlx_dtype,
        quantization=quantization,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        num_frames_per_block=args.num_frames_per_block,
    )
    load_s = time.perf_counter() - load_start
    print(f"loaded causal DiT ({config['num_layers']} layers, {args.mlx_quantization}, "
          f"local_attn_size={args.local_attn_size}, sink_size={args.sink_size}) in {load_s:.1f}s")

    # Optional live decode (TAEHV). Cleanly skip if weights / deps are missing.
    taehv = None
    torch_device = None
    torch_dtype = None
    if args.decode:
        import torch

        torch_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        torch_dtype = torch.float16 if args.mlx_dtype in ("fp16", "bf16") else torch.float32
        taehv = _try_load_taehv(
            checkpoint_path=args.taehv_checkpoint_path,
            source_path=args.taehv_source_path,
            device=torch_device,
            dtype=torch_dtype,
        )
        if taehv is None:
            print("[decode] continuing in latency-only mode (no per-block frames).")

    # Prompt embeddings: encode if a text encoder is present, else random (latency is content-independent).
    text_len = int(config.get("text_len", 512))
    if (args.model_root / "text_encoder").exists() and (args.model_root / "tokenizer").exists():
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from examples.inference.basic.mlx_wan_prompt_to_video import encode_prompt
        embeds = encode_prompt(model_root=args.model_root, prompt=args.prompt,
                               max_sequence_length=text_len, device_arg="auto", dtype_arg="fp16")
        text = mx.array(embeds.numpy()).astype(dtype)
        print(f"encoded prompt -> {tuple(text.shape)}")
    else:
        rng = np.random.default_rng(args.seed)
        text = mx.array((rng.standard_normal((1, text_len, int(config["text_dim"]))) * 0.1).astype(np.float16)).astype(
            dtype)
        print(f"no text encoder under {args.model_root}; using random prompt embeds {tuple(text.shape)} (latency only)")

    # Rotary tables for the whole clip (frame-major), and pure-noise latents.
    from examples.inference.basic.mlx_wan_prompt_to_video import make_rotary_embeddings
    cos_full, sin_full = make_rotary_embeddings(
        config, latent_frames=args.num_frames, latent_height=latent_h, latent_width=latent_w)
    gen = np.random.default_rng(args.seed + 1)
    noise = mx.array(
        gen.standard_normal((1, int(config["in_channels"]), args.num_frames, latent_h, latent_w)).astype(
            np.float32)).astype(dtype)

    dmd_steps = [int(s) for s in args.dmd_denoising_steps.split(",") if s.strip()]
    schedule, timesteps = build_dmd_schedule(dmd_steps, flow_shift=args.flow_shift, warp_denoising_step=True)

    window_tokens = (21 if args.local_attn_size == -1 else args.local_attn_size) * frame_seqlen
    print(f"streaming {args.num_frames} frames x {args.height}x{args.width} "
          f"({len(timesteps)}-step DMD, {frame_seqlen} tokens/frame, "
          f"kv_window≈{window_tokens} tokens)...")
    mx.reset_peak_memory()
    stream_start = time.perf_counter()
    block_latencies: list[float] = []
    peak_gib_by_block: list[float] = []
    blocks: list[np.ndarray] = []
    decoded_frame_chunks: list[np.ndarray] = []
    prev = stream_start
    time_to_first = None
    for block_index, latent in stream_causal_latents(
            model, text, noise, cos_full, sin_full, schedule, timesteps,
            frame_seqlen=frame_seqlen, seed=args.seed):
        now = time.perf_counter()
        block_latencies.append(now - prev)
        prev = now
        if time_to_first is None:
            time_to_first = now - stream_start
        peak_gib_by_block.append(mx.get_peak_memory() / (1024**3))
        blocks.append(np.array(latent.astype(mx.float32)))
        print(f"  block {block_index:2d} ready  (+{block_latencies[-1]:.2f}s, "
              f"peak={peak_gib_by_block[-1]:.3f} GiB)")

        if taehv is not None:
            try:
                frames = _decode_block_frames(
                    taehv,
                    blocks[-1],
                    device=torch_device,
                    dtype=torch_dtype,
                    parallel=args.taehv_parallel,
                )
                decoded_frame_chunks.append(frames)
                _write_mp4(decoded_frame_chunks, args.output_video, args.fps)
                print(f"    decoded → {args.output_video} "
                      f"({sum(c.shape[0] for c in decoded_frame_chunks)} pixel frames so far)")
            except Exception as exc:  # noqa: BLE001 - keep streaming if one decode fails.
                print(f"    [decode] block {block_index} failed ({type(exc).__name__}: {exc}); "
                      "continuing without further decode.")
                taehv = None

    total_s = time.perf_counter() - stream_start
    peak_gib = mx.get_peak_memory() / (1024**3)
    steady = statistics.median(block_latencies[1:]) if len(block_latencies) > 1 else block_latencies[0]
    metrics = {
        "num_frames": args.num_frames,
        "resolution": f"{args.height}x{args.width}",
        "dmd_steps": len(timesteps),
        "quantization": args.mlx_quantization,
        "local_attn_size": args.local_attn_size,
        "sink_size": args.sink_size,
        "frame_seqlen": frame_seqlen,
        "kv_window_tokens": window_tokens,
        "load_s": round(load_s, 2),
        "time_to_first_block_s": round(time_to_first, 3),
        "block_latency_first_s": round(block_latencies[0], 3),
        "block_latency_steady_s": round(steady, 3),
        "block_latencies_s": [round(x, 3) for x in block_latencies],
        "peak_gib_by_block": [round(x, 3) for x in peak_gib_by_block],
        "total_stream_s": round(total_s, 2),
        "peak_gib": round(peak_gib, 3),
        "decode": bool(args.decode and decoded_frame_chunks),
        "output_video": str(args.output_video) if decoded_frame_chunks else None,
    }
    print("\n=== streaming metrics ===")
    print(json.dumps(metrics, indent=2))
    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(metrics, indent=2))
        print(f"wrote {args.metrics_out}")


if __name__ == "__main__":
    main()
