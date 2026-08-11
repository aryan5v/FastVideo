# SPDX-License-Identifier: Apache-2.0
"""FastH3 S1: depth-prune the 33B H3 DiT to a smaller student.

Two steps:
1. **Sensitivity sweep**: for each transformer block, run one calibration
   forward with that block's residual contribution removed (identity
   ablation via forward hooks) and measure the output drift against the full
   model. Blocks with the smallest drift are the most prunable.
2. **Prune**: keep the first/last blocks unconditionally, then the least-
   sensitive middle blocks until ``--keep-blocks`` remain (default 22 = 14B).
   Write a diffusers-layout checkpoint directory (``config.json`` with the
   reduced ``num_layers`` + a single safetensors of the retained weights) so
   the training loader can ``init_from`` it directly.

Usage (cluster, 8 GPUs):
  $PY -m torch.distributed.run --nproc_per_node=8 scripts/fasth3/prune_h3_depth.py \
      --model-root <H3 snapshot dir> --out <pruned dir> --keep-blocks 22
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, help="H3 diffusers snapshot (local)")
    parser.add_argument("--out", required=True, help="output pruned checkpoint dir")
    parser.add_argument("--keep-blocks", type=int, default=22)
    parser.add_argument("--sweep-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _loader_args(model_root: str) -> object:
    from fastvideo.fastvideo_args import FastVideoArgs

    return FastVideoArgs.from_kwargs(
        model_path=model_root,
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=False,
        training_mode=False,
        inference_mode=True,
        trust_remote_code=False,
        revision="main",
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
    )


def _load_transformer(model_root: str) -> torch.nn.Module:
    from fastvideo.models.loader.component_loader import TransformerLoader

    transformer = TransformerLoader().load(f"{model_root}/transformer", _loader_args(model_root))
    return transformer.to("cuda:0").eval()


def _synthetic_inputs(transformer: torch.nn.Module, device: torch.device, seed: int = 42):
    """A small packed sequence exercising all three modalities (no real data)."""
    from fastvideo.pipelines.basic.minimax_h3.packing import (
        build_packed_sequence,
        build_row_timesteps,
        patchify_video_latents,
    )

    g = torch.Generator("cpu").manual_seed(seed)
    num_latent_frames, latent_height, latent_width = 9, 16, 16
    num_audio_latents = 60
    text_len = 24
    patch = tuple(int(v) for v in transformer.patch_size)
    channels = transformer.num_channels_latents
    audio_channels = transformer.audio_in_channels

    layout = build_packed_sequence(
        torch.full((text_len, ), 1, dtype=torch.long),
        num_latent_frames,
        latent_height,
        latent_width,
        num_audio_latents,
        patch,
    )
    video = torch.randn(1, channels, num_latent_frames, latent_height, latent_width, generator=g)
    video_rows = patchify_video_latents(video, patch).to(device)
    audio_rows = torch.randn(1, num_audio_latents * 2, audio_channels, generator=g).to(device)
    text_rows = torch.randn(1, text_len, transformer.text_dim, generator=g).to(device)
    unique, inverse = build_row_timesteps(layout, 0.5, 0.4)

    kwargs = dict(
        hidden_states=video_rows,
        audio_hidden_states=audio_rows,
        encoder_hidden_states=text_rows,
        timestep=unique.to(device, dtype=torch.float32),
        timestep_indices=inverse.to(device, dtype=torch.long),
        token_tags=layout.token_tags.to(device, dtype=torch.long),
        position_ids=layout.position_ids.to(device, dtype=torch.float32),
        video_indices=layout.video_indices.to(device, dtype=torch.long),
        audio_indices=layout.audio_indices.to(device, dtype=torch.long),
        text_indices=layout.text_indices.to(device, dtype=torch.long),
    )
    return kwargs


def _ablation_drift(transformer: torch.nn.Module, inputs: dict, block_index: int) -> float:
    """Output drift when block ``block_index``'s residual contribution is removed."""
    from fastvideo.forward_context import set_forward_context

    block = transformer.transformer_blocks[block_index]
    original_forward = block.forward

    def identity_forward(hidden_states, *args, **kwargs):
        return hidden_states  # no residual addition, no processing

    block.forward = identity_forward  # type: ignore[method-assign]
    try:
        with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
            video_out, audio_out = transformer(**inputs)
        ref_video, ref_audio = _REFERENCE
        drift_v = (video_out.float() - ref_video.float()).square().mean().item()
        drift_a = (audio_out.float() - ref_audio.float()).square().mean().item()
        return drift_v + drift_a
    finally:
        block.forward = original_forward  # type: ignore[method-assign]


_REFERENCE: tuple[torch.Tensor, torch.Tensor] | None = None


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    transformer = _load_transformer(args.model_root)
    inputs = _synthetic_inputs(transformer, device, seed=args.seed)

    global _REFERENCE
    with torch.no_grad():
        from fastvideo.forward_context import set_forward_context

        with set_forward_context(current_timestep=0, attn_metadata=None):
            ref_video, ref_audio = transformer(**inputs)
    _REFERENCE = (ref_video, ref_audio)
    if rank == 0:
        print(f"reference forward OK: video {tuple(ref_video.shape)} audio {tuple(ref_audio.shape)}", flush=True)

    if rank == 0:
        # Single-rank sweep: 50 ablations x ~4s is a few minutes; no rank
        # gathering needed.
        num_blocks = len(transformer.transformer_blocks)
        results: list[tuple[int, float]] = []
        for block_index in range(num_blocks):
            drift = _ablation_drift(transformer, inputs, block_index)
            results.append((block_index, drift))
            print(f"block {block_index}: drift {drift:.6e}", flush=True)

        results.sort(key=lambda item: item[1])
        print("sensitivity (ascending drift = most prunable):", flush=True)
        for block_index, drift in results:
            print(f"  block {block_index}: {drift:.6e}", flush=True)

        keep = sorted(int(b) for b, _ in results[: args.keep_blocks])
        # guarantee first/last blocks survive
        for boundary in (0, num_blocks - 1):
            if boundary not in keep:
                keep.pop()
                keep.append(boundary)
        keep = sorted(keep)
        print(f"keeping {len(keep)} blocks: {keep}", flush=True)
        if args.sweep_only:
            print("SWEEP DONE", flush=True)
            return

        _write_pruned_checkpoint(args.model_root, args.out, transformer, keep)


def _write_pruned_checkpoint(model_root: str, out_dir: str, transformer: torch.nn.Module,
                             keep: list[int]) -> None:
    import safetensors.torch

    root = Path(model_root) / "transformer"
    config = json.loads((root / "config.json").read_text())
    config["num_layers"] = len(keep)

    state: dict[str, torch.Tensor] = {}
    for name, param in transformer.state_dict().items():
        if name.startswith("transformer_blocks."):
            index = int(name.split(".")[1])
            if index not in keep:
                continue
            new_name = name.replace(f"transformer_blocks.{index}",
                                    f"transformer_blocks.{keep.index(index)}")
            state[new_name] = param.detach().float()
        else:
            state[name] = param.detach().float()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    safetensors.torch.save_file(state, str(out / "diffusion_pytorch_model.safetensors"))
    print(f"PRUNED CHECKPOINT WRITTEN: {out} ({len(keep)} blocks, {len(state)} tensors)", flush=True)


if __name__ == "__main__":
    main()
