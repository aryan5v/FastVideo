# SPDX-License-Identifier: Apache-2.0
"""Generate MiniMax-H3 at low resolution, then upscale 2x: latent upscalers vs bicubic (evaluation script).

For each prompt: encode, denoise at ``--width x --height``, decode the native
low-resolution clip, then produce 2x clips with

* ``bicubic``: bicubic resize of the decoded low-resolution frames;
* ``lbh``: the LBH-123-AI "Minimax_h3_latent_Upscaler" 3D latent upscaler (Apache-2.0
  weights, MIT ComfyUI node) applied to the clean latent, decoded at 2x;
* ``mamad``: Mamad8's "H3-Latent-Upscaler-2x" clean-latent upscaler, decoded at 2x.

Upscaler code is not vendored: pass the cloned ComfyUI node repositories
(``--lbh-repo``, ``--mamad-repo``) and the checkpoints. Every clip keeps the
low-resolution audio. Prints one JSON line per prompt and output with
denoise/upscale/decode/save times.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minimax_h3_two_node_pipeline as two_node  # noqa: E402


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def load_lbh(repo: Path, checkpoint: Path, device: torch.device):
    """The LBH 3D latent upscaler: ComfyUI-space (normalized) H3 latent -> 2x spatial."""
    folder_paths = types.ModuleType("folder_paths")  # the node registers a ComfyUI model folder at import
    folder_paths.folder_names_and_paths = {}
    folder_paths.models_dir = str(checkpoint.parent)
    folder_paths.add_model_folder_path = lambda *args, **kwargs: None
    sys.modules.setdefault("folder_paths", folder_paths)
    module = _load_module("lbh_upscaler_3d", repo / "nodes" / "minimax_h3_latent_upscaler_3d.py")
    state = module._extract_upscaler_sd(module._load_raw_sd(str(checkpoint)))
    cfg = module._detect_arch(state)
    model = module.LatentResizer3D(**{key: cfg[key] for key in ("in_channels", "in_blocks", "out_blocks", "channels",
                                                                 "dropout", "attn", "temporal_every",
                                                                 "temporal_kernel")})
    model.load_state_dict(state, strict=True)
    model = model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    mean, std = module._make_norm_tensors(device, torch.bfloat16)

    @torch.no_grad()
    def upscale(z: torch.Tensor) -> torch.Tensor:
        t, h, w = z.shape[-3:]
        out = model((z.to(torch.bfloat16) - mean) / std, scale=2.0, target_size=(t, 2 * h, 2 * w))
        return (out * std + mean).float()

    return upscale


def load_mamad(repo: Path, checkpoint: Path, device: torch.device):
    """Mamad8's clean-latent 2x upscaler: ComfyUI-space (normalized) H3 latent -> 2x spatial."""
    from safetensors.torch import load_file

    module = _load_module("mamad_upscaler", repo / "upscaler.py")
    model = module.build_upscaler(load_file(str(checkpoint)), module.read_checkpoint_info(checkpoint)).to(device)
    dtype = next(model.parameters()).dtype

    @torch.no_grad()
    def upscale(z: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype in (torch.float16, torch.bfloat16)):
            return model(z.to(dtype)).float()

    return upscale


def _timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, round(time.perf_counter() - start, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts", required=True, help="JSON list or {id: prompt}")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--lbh-repo", type=Path)
    parser.add_argument("--lbh-checkpoint", type=Path)
    parser.add_argument("--mamad-repo", type=Path)
    parser.add_argument("--mamad-checkpoint", type=Path)
    parser.add_argument("--local-port", type=int, default=29651)
    parser.add_argument("--output-dir", default="outputs/minimax_h3_lowres_upscale")
    args = parser.parse_args()
    args.compile_vae = False
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(args.prompts).read_text())
    prompts = prompts if isinstance(prompts, dict) else {f"p{i}": p for i, p in enumerate(prompts)}

    from fastvideo.entrypoints.video_generator import VideoGenerator
    from fastvideo.pipelines.basic.minimax_h3.packing import h3_dit_patch_size, unpatchify_video_tokens

    io_pipeline, request, io_args = two_node._build("io", args)
    gen_pipeline, _, _ = two_node._build("gen", args)
    device = torch.device("cuda")
    vae = io_pipeline.get_module("vae")
    upscalers = {}
    if args.lbh_repo:
        upscalers["lbh"] = load_lbh(args.lbh_repo, args.lbh_checkpoint, device)
    if args.mamad_repo:
        upscalers["mamad"] = load_mamad(args.mamad_repo, args.mamad_checkpoint, device)

    def save(tag: str, name: str, video_u8: torch.Tensor, audio, sample_rate: int, fps: int) -> float:
        start = time.perf_counter()
        frames = [frame.numpy() for frame in video_u8.permute(1, 2, 3, 0).contiguous()]
        VideoGenerator._save_video_with_audio_single_pass(output_path=str(out_dir / f"{tag}__{name}.mp4"),
                                                          frames=frames,
                                                          fps=fps,
                                                          audio=audio,
                                                          sample_rate=sample_rate)
        return round(time.perf_counter() - start, 3)

    buffers: dict[tuple[int, ...], torch.Tensor] = {}

    def decode_raw(z: torch.Tensor) -> torch.Tensor:
        shape = tuple(vae.decoded_pixel_shape(z.shape))
        if shape not in buffers:  # reuse pre-touched pages: fresh pageable pages fault slowly on unified memory
            buffers[shape] = torch.zeros(shape, dtype=torch.float32)
        out = buffers[shape]
        with torch.no_grad(), torch.autocast("cuda", dtype=vae.decode_autocast_dtype):
            vae.decode_to_pixels(z, out)
        return out[0].mul(255).clamp_(0, 255).to(torch.uint8)

    for index, (tag, prompt) in enumerate([next(iter(prompts.items()))] + list(prompts.items())):
        warmup = index == 0
        message, (layout, raw_shape, fps) = two_node._encode(io_pipeline, request, io_args, prompt, args)
        batch, denoise_s = _timed(lambda message=message: two_node.denoise(gen_pipeline, message, device))
        _, channels, frames, height, width = raw_shape
        z = vae.denormalize_latents(
            unpatchify_video_tokens(batch.latents[layout.num_condition_video_rows:], frames, height, width, channels,
                                    h3_dit_patch_size(io_args)).float())
        decoded = two_node._decode_and_save(io_pipeline, {
            "video_latents": batch.latents,
            "audio_latents": batch.audio_latents
        }, (layout, raw_shape, fps), out_dir / f"{tag}__native_{args.width}x{args.height}.mp4")
        audio_batch = io_pipeline.run(("audio_decoding_stage", ), _audio_batch(batch, layout, raw_shape))
        audio, rate = audio_batch.extra["audio"], int(audio_batch.extra["audio_sample_rate"])
        rows = {"prompt": tag, "warmup": warmup, "denoise_s": round(denoise_s, 3), "native": decoded}
        low, low_s = _timed(lambda z=z: decode_raw(z))
        rows["low_decode_s"] = low_s
        up, up_s = _timed(lambda low=low: _bicubic(low))
        rows["bicubic"] = {"upscale_s": up_s, "save_s": save(tag, "bicubic2x", up, audio, rate, fps)}
        for name, upscale in upscalers.items():
            # ComfyUI hands these nodes H3 latents in the normalized (DiT) space; the LBH node then applies its own
            # per-channel normalization on top, as in its training.
            z2, ups_s = _timed(lambda z=z, upscale=upscale: vae.denormalize_latents(upscale(vae.normalize_latents(z))))
            pixels, dec_s = _timed(lambda z2=z2: decode_raw(z2))
            rows[name] = {"upscale_s": ups_s, "decode_2x_s": dec_s, "save_s": save(tag, f"{name}2x", pixels, audio,
                                                                                    rate, fps)}
            del z2, pixels
        print(json.dumps(rows), flush=True)


def _audio_batch(batch, layout, raw_shape):
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    return ForwardBatch(data_type="video",
                        latents=batch.latents,
                        audio_latents=batch.audio_latents,
                        raw_latent_shape=raw_shape,
                        extra={MINIMAX_H3_LAYOUT_KEY: layout})


@torch.no_grad()
def _bicubic(video_u8: torch.Tensor) -> torch.Tensor:
    """[3, T, H, W] uint8 -> 2x bicubic, frame by frame on the GPU."""
    frames = video_u8.permute(1, 0, 2, 3).cuda().float()
    out = torch.cat([
        F.interpolate(chunk, scale_factor=2, mode="bicubic", align_corners=False) for chunk in frames.split(16)
    ])
    return out.round_().clamp_(0, 255).to(torch.uint8).permute(1, 0, 2, 3).cpu()


if __name__ == "__main__":
    main()
