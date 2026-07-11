# SPDX-License-Identifier: Apache-2.0
"""Rough image-to-video smoke for Wan2.2-TI2V-5B on MLX.

Proves the I2V path works end-to-end *before* we spend training on it: take an
input image, VAE-encode it (torch Wan VAE — the MLX runtime has decode only),
lock it as latent frame 0 (``build_i2v_inputs``: first-frame replace + per-token
timestep with frame-0 at t=0), run the MLX 5B DMD denoise, and decode a clip.

    python -m fastvideo.benchmarks.smoke_5b_i2v --image path/to.jpg \
        --prompt "the fox turns its head, snow falling" --output ~/Desktop/i2v.mp4

With no --image, uses frame 0 of the fox 5B demo on the Desktop.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np


def _default_vae_dir() -> Path:
    hits = glob.glob(str(Path.home() / ".cache/huggingface/hub/"
                        "models--FastVideo--FastWan2.2-TI2V-5B-FullAttn-Diffusers/snapshots/*/vae"))
    if not hits:
        raise SystemExit("Wan2.2 VAE dir not found; pass --vae-dir")
    return Path(hits[0])


def _load_image_chw(image_path: Path, height: int, width: int) -> np.ndarray:
    """Return a [1,3,1,H,W] float32 tensor in [-1, 1]."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB").resize((width, height))
    arr = (np.asarray(img, dtype=np.float32) / 127.5) - 1.0   # [H,W,3] in [-1,1]
    chw = arr.transpose(2, 0, 1)[None, :, None, :, :]         # [1,3,1,H,W]
    return chw.astype(np.float32)


def _extract_first_frame(video_path: Path, out_png: Path) -> Path:
    import imageio.v3 as iio

    frame = iio.imread(video_path, index=0)
    iio.imwrite(out_png, frame)
    return out_png


def encode_image_to_dit_latent(image_path: Path, *, height: int, width: int, vae_dir: Path) -> np.ndarray:
    """VAE-encode the image and normalize into DiT latent space: (z - mean)/std."""
    import torch
    from diffusers import AutoencoderKLWan

    cfg = json.loads((vae_dir / "config.json").read_text())
    mean = np.asarray(cfg["latents_mean"], dtype=np.float32).reshape(1, -1, 1, 1, 1)
    std = np.asarray(cfg["latents_std"], dtype=np.float32).reshape(1, -1, 1, 1, 1)

    vae = AutoencoderKLWan.from_pretrained(vae_dir, torch_dtype=torch.float32)
    vae.eval()
    img = torch.from_numpy(_load_image_chw(image_path, height, width))
    with torch.no_grad():
        posterior = vae.encode(img).latent_dist
        z = posterior.mode().cpu().numpy().astype(np.float32)   # [1,48,1,lh,lw]
    return (z - mean) / std


def main() -> None:
    ap = argparse.ArgumentParser(description="Wan2.2-5B I2V smoke (MLX).")
    ap.add_argument("--image", type=Path, default=None)
    ap.add_argument("--prompt", default="a red fox turns its head as snow falls, cinematic slow motion")
    ap.add_argument("--output", type=Path, default=Path.home() / "Desktop/fastvideo_demos/fox_5b_i2v.mp4")
    ap.add_argument("--model-root", type=Path, default=Path.home() / "models/fastwan22_5b")
    ap.add_argument("--vae-dir", type=Path, default=None)
    ap.add_argument("--text-encoder-root", type=Path, default=None)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num-frames", type=int, default=81)
    ap.add_argument("--dmd-denoising-steps", default="1000,757,522")
    ap.add_argument("--flow-shift", type=float, default=5.0)
    ap.add_argument("--quant", default="fp16", help="fp16 | int8 | mxfp4 ...")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import mlx.core as mx

    from examples.inference.basic.mlx_wan_prompt_to_video import encode_prompt, make_rotary_embeddings
    from fastvideo.mlx_runtime.sampling import dmd_step, pred_noise_to_pred_video
    from fastvideo.mlx_runtime.wan22 import mlx_wan22_dit_from_diffusers_safetensors
    from fastvideo.mlx_runtime.wan22_i2v import build_i2v_inputs, replace_first_latent_frame
    from fastvideo.mlx_runtime.wan22_sample import build_wan22_dmd_schedule
    from fastvideo.mlx_runtime.wan_vae import decode_latents_to_video

    vae_dir = args.vae_dir or _default_vae_dir()
    te_root = args.text_encoder_root or Path(glob.glob(str(
        Path.home() / ".cache/huggingface/hub/"
        "models--FastVideo--FastWan2.1-T2V-1.3B-Diffusers/snapshots/*"))[0])

    image = args.image
    if image is None:
        fox = Path.home() / "Desktop/fastvideo_demos/fox_5b_fp16_taehv.mp4"
        if not fox.exists():
            raise SystemExit("no --image and no fox demo video to extract from; pass --image")
        image = _extract_first_frame(fox, Path("/tmp/i2v_input.png"))
        print(f"[i2v] using extracted frame: {image}")

    cfg = json.loads((args.model_root / "transformer" / "config.json").read_text())
    lat_h, lat_w = args.height // 16, args.width // 16
    lat_t = (args.num_frames - 1) // 4 + 1
    in_ch = int(cfg["in_channels"])

    t0 = time.perf_counter()
    img_latent = encode_image_to_dit_latent(image, height=args.height, width=args.width, vae_dir=vae_dir)
    print(f"[i2v] image encoded -> {img_latent.shape} in {time.perf_counter()-t0:.1f}s")
    if img_latent.shape[-2:] != (lat_h, lat_w):
        raise SystemExit(f"encoded latent {img_latent.shape[-2:]} != expected ({lat_h},{lat_w})")

    embeds = encode_prompt(model_root=te_root, prompt=args.prompt, max_sequence_length=512,
                           device_arg="auto", dtype_arg="fp16")
    ehs = mx.array(embeds.numpy()).astype(mx.float16)

    quant = None if args.quant == "fp16" else args.quant
    dit = mlx_wan22_dit_from_diffusers_safetensors(
        args.model_root / "transformer" / "diffusion_pytorch_model.safetensors",
        args.model_root / "transformer" / "config.json", dtype="fp16", quantization=quant)
    freqs = make_rotary_embeddings(cfg, latent_frames=lat_t, latent_height=lat_h, latent_width=lat_w)

    img_mx = mx.array(img_latent.astype(np.float32)).astype(mx.float16)
    rng = np.random.default_rng(args.seed)
    current = mx.array(rng.standard_normal((1, in_ch, lat_t, lat_h, lat_w)).astype(np.float32)).astype(mx.float16)

    steps = [int(s) for s in args.dmd_denoising_steps.split(",") if s.strip()]
    schedule, timesteps = build_wan22_dmd_schedule(steps, flow_shift=args.flow_shift, warp_denoising_step=True)
    renoise_rng = np.random.default_rng(args.seed + 1)
    last = len(timesteps) - 1

    t1 = time.perf_counter()
    mx.reset_peak_memory()
    for i, t in enumerate(timesteps):
        latents_in, ts_pt = build_i2v_inputs(current, img_mx, video_timestep=float(t),
                                             patch_size=tuple(dit.patch_size))
        pred = dit(latents_in.astype(mx.float16), ehs, mx.array(ts_pt).astype(mx.float32), freqs)
        ni = latents_in.astype(mx.float32)
        pn = pred.astype(mx.float32)
        if i < last:
            renoise = mx.array(renoise_rng.standard_normal(tuple(current.shape)).astype(np.float32))
            current = dmd_step(latents=ni, noise_input_latent=ni, pred_noise=pn, schedule=schedule,
                               timestep=float(t), next_timestep=float(timesteps[i + 1]),
                               noise=renoise).astype(mx.float16)
        else:
            current = pred_noise_to_pred_video(pn, ni, schedule.sigma_for(float(t))).astype(mx.float16)
        mx.eval(current)
    final = replace_first_latent_frame(current, img_mx)
    peak = mx.get_peak_memory() / 1024**3
    print(f"[i2v] denoise {len(timesteps)} steps in {time.perf_counter()-t1:.1f}s, peak {peak:.2f} GiB")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    m = decode_latents_to_video(np.array(final.astype(mx.float32)), args.output, fps=16,
                                backend="taehv", z_dim=in_ch)
    print(f"[i2v] decoded via {m['backend']} in {m['decode_s']:.1f}s -> {args.output}")


if __name__ == "__main__":
    main()
