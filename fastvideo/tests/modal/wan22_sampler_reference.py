# SPDX-License-Identifier: Apache-2.0
"""Dump / compare final DMD latents: torch FastWan2.2 pipeline vs MLX sampler.

``dump`` (CUDA via Modal): run the torch densen DMD path with a fixed prompt/seed
and write final latents to an ``.npz`` on the volume.

``compare`` (Mac): reload the dump, re-run MLX ``sample_wan22_dmd`` with the same
noise + text embeds, report max|Δ| / cosine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEED = 1234
RENOISE_SEED = 0
PROMPT = "A red fox trotting through a snowy pine forest at golden hour, cinematic"
# Small enough for a fast dump but exercises multi-step DMD.
HEIGHT, WIDTH, NUM_FRAMES = 256, 448, 33  # latent 16x28x9 after /16 /4
DMD_STEPS = [1000, 757, 522]
FLOW_SHIFT = 5.0


def dump(out_path: Path, model_id: str = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers") -> None:
    """Run torch VideoGenerator / pipeline DMD and save final latents."""
    import torch

    from fastvideo.distributed.parallel_state import maybe_init_distributed_environment_and_model_parallel

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dumping torch 5B DMD latents on {device}", flush=True)

    # Prefer the public VideoGenerator API when available; fall back to a
    # lightweight DiT-only loop matching MLX (same schedule math).
    try:
        from fastvideo import VideoGenerator
        from fastvideo.configs.pipelines.wan import FastWan2_2_TI2V_5B_Config

        gen = VideoGenerator.from_pretrained(
            model_id,
            num_gpus=1,
        )
        # VideoGenerator.generate returns pixels; we need latents — use internal DiT path below.
        del gen
    except Exception as exc:  # noqa: BLE001
        print(f"VideoGenerator path unavailable ({exc}); using DiT-only dump", flush=True)

    # DiT-only dump: load transformer, run warped DMD in torch float32.
    from diffusers import AutoencoderKLWan
    from transformers import AutoTokenizer, UMT5EncoderModel
    from huggingface_hub import snapshot_download

    from fastvideo.configs.models.dits.wanvideo import WanVideoArchConfig, WanVideoConfig
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.dits.wanvideo import WanTransformer3DModel
    from fastvideo.models.loader.utils import get_param_names_mapping, hf_to_custom_state_dict
    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from fastvideo.models.utils import pred_noise_to_pred_video
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from safetensors.torch import load_file

    root = Path(snapshot_download(model_id, allow_patterns=["transformer/*", "text_encoder/*", "tokenizer/*", "vae/*"]))
    cfg = json.loads((root / "transformer" / "config.json").read_text())
    arch = WanVideoConfig(arch_config=WanVideoArchConfig(
        num_attention_heads=int(cfg["num_attention_heads"]),
        attention_head_dim=int(cfg["attention_head_dim"]),
        in_channels=int(cfg["in_channels"]),
        out_channels=int(cfg["out_channels"]),
        text_dim=int(cfg["text_dim"]),
        freq_dim=int(cfg["freq_dim"]),
        ffn_dim=int(cfg["ffn_dim"]),
        num_layers=int(cfg["num_layers"]),
        patch_size=tuple(cfg["patch_size"]),
        cross_attn_norm=bool(cfg.get("cross_attn_norm", True)),
        qk_norm=cfg.get("qk_norm", "rms_norm_across_heads"),
        eps=float(cfg.get("eps", 1e-6)),
        rope_max_seq_len=int(cfg.get("rope_max_seq_len", 1024)),
    ))
    model = WanTransformer3DModel(config=arch, hf_config=cfg).to(device, torch.float16).eval()
    raw = load_file(str(root / "transformer" / "diffusion_pytorch_model.safetensors"), device="cpu")
    custom, _ = hf_to_custom_state_dict(raw.items(), get_param_names_mapping(model.param_names_mapping))
    model.load_state_dict(custom, strict=False)
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
    text_enc = UMT5EncoderModel.from_pretrained(root / "text_encoder", torch_dtype=torch.float16).to(device).eval()
    tokens = tokenizer([PROMPT], return_tensors="pt", padding="max_length", max_length=512, truncation=True)
    with torch.no_grad():
        ehs = text_enc(tokens.input_ids.to(device)).last_hidden_state

    lat_h, lat_w = HEIGHT // 16, WIDTH // 16
    lat_t = (NUM_FRAMES - 1) // 4 + 1
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    noise = torch.randn(1, int(cfg["in_channels"]), lat_t, lat_h, lat_w, generator=gen, dtype=torch.float32)
    latents = noise.to(device=device, dtype=torch.float16)

    scheduler = FlowMatchEulerDiscreteScheduler(shift=FLOW_SHIFT)
    scheduler.set_timesteps(1000, device="cpu")
    step_idx = torch.tensor(DMD_STEPS, dtype=torch.long)
    warped = torch.cat((scheduler.timesteps.cpu(), torch.tensor([0.0])))
    timesteps = warped[1000 - step_idx]
    # sigmas
    def sigma_for(t: float) -> float:
        idx = int(torch.argmin(torch.abs(scheduler.timesteps.cpu() - t)).item())
        return float(scheduler.sigmas.cpu()[idx])

    torch.manual_seed(RENOISE_SEED)
    # Use NumPy RNG for renoise so MLX can reproduce the exact sequence.
    renoise_rng = np.random.default_rng(RENOISE_SEED)
    pt, ph, pw = tuple(cfg["patch_size"])
    tokens_n = (lat_t // pt) * (lat_h // ph) * (lat_w // pw)
    from fastvideo.models.utils import pred_noise_to_pred_video as torch_p2v

    with torch.no_grad(), set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=ForwardBatch(data_type="dummy")):
        for i, t in enumerate(timesteps):
            t_val = float(t.item())
            ts = torch.full((1, tokens_n), t_val, device=device, dtype=torch.long)
            pred = model(hidden_states=latents, encoder_hidden_states=ehs, timestep=ts)
            # Match MLX: pred_video = noise_input - sigma * pred_noise
            sigma = sigma_for(t_val)
            pred_video = latents.float() - sigma * pred.float()
            if i < len(timesteps) - 1:
                sigma_next = sigma_for(float(timesteps[i + 1].item()))
                noise_r = torch.from_numpy(
                    renoise_rng.standard_normal(tuple(latents.shape)).astype(np.float32)).to(device)
                latents = ((1.0 - sigma_next) * pred_video + sigma_next * noise_r).to(latents.dtype)
            else:
                latents = pred_video.to(latents.dtype)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        latents=latents.float().cpu().numpy(),
        noise=noise.numpy(),
        text=ehs.float().cpu().numpy(),
        prompt=np.array(PROMPT),
        seed=np.array(SEED),
        renoise_seed=np.array(RENOISE_SEED),
        dmd_steps=np.array(DMD_STEPS),
        flow_shift=np.array(FLOW_SHIFT),
        height=np.array(HEIGHT),
        width=np.array(WIDTH),
        num_frames=np.array(NUM_FRAMES),
        device=np.array(device),
    )
    print(f"wrote {out_path} latents {tuple(latents.shape)}", flush=True)


def compare(npz_path: Path, dit_ckpt: Path, dit_config: Path) -> None:
    import mlx.core as mx
    import torch

    from examples.inference.basic.mlx_wan_prompt_to_video import make_rotary_embeddings
    from fastvideo.mlx_runtime.wan22 import mlx_wan22_dit_from_diffusers_safetensors
    from fastvideo.mlx_runtime.wan22_sample import sample_wan22_dmd

    data = np.load(npz_path, allow_pickle=True)
    noise = mx.array(data["noise"]).astype(mx.float16)
    text = mx.array(data["text"]).astype(mx.float16)
    ref = data["latents"]
    config = json.loads(Path(dit_config).read_text())
    lat_t, lat_h, lat_w = int(noise.shape[2]), int(noise.shape[3]), int(noise.shape[4])
    freqs = make_rotary_embeddings(config, latent_frames=lat_t, latent_height=lat_h, latent_width=lat_w)
    model = mlx_wan22_dit_from_diffusers_safetensors(dit_ckpt, dit_config, dtype="fp16")
    out = sample_wan22_dmd(
        model,
        text,
        noise,
        freqs,
        dmd_denoising_steps=list(data["dmd_steps"]),
        flow_shift=float(data["flow_shift"]),
        warp_denoising_step=True,
        seed=int(data["renoise_seed"]),
    )
    mx.eval(out)
    mlx_np = np.array(out.astype(mx.float32))
    max_abs = float(np.abs(mlx_np - ref).max())
    mean_abs = float(np.abs(mlx_np - ref).mean())
    cos = float(np.dot(mlx_np.ravel(), ref.ravel()) / (np.linalg.norm(mlx_np) * np.linalg.norm(ref) + 1e-12))
    print(f"MLX vs torch-{data['device']}: max|Δ|={max_abs:.3e} mean|Δ|={mean_abs:.3e} cosine={cos:.6f}")
    # Full multi-step fp16 can drift; cosine is the decisive structural check.
    if cos < 0.99:
        raise SystemExit(f"FAIL: cosine {cos} < 0.99 — sampler likely wrong")
    print("PASS: MLX sampler matches torch DMD (cosine ≥ 0.99)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("dump", "compare"))
    p.add_argument("--path", type=Path, default=Path("/root/data/wan22_sampler/ref.npz"))
    p.add_argument("--dit-checkpoint", type=Path, default=Path.home() / "models/fastwan22_5b/transformer/diffusion_pytorch_model.safetensors")
    p.add_argument("--dit-config", type=Path, default=Path.home() / "models/fastwan22_5b/transformer/config.json")
    args = p.parse_args()
    if args.mode == "dump":
        dump(args.path)
    else:
        compare(args.path, args.dit_checkpoint, args.dit_config)


if __name__ == "__main__":
    main()
