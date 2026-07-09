# SPDX-License-Identifier: Apache-2.0
"""CUDA reference dump/compare for the MLX Wan2.2 (per-token timestep) port.

Two modes:

``dump`` (CUDA via Modal L40S): builds a tiny dense Wan model, runs a single
forward with a 2-D per-token timestep (frame 0 at t=0, rest at t=500), and
writes weights + inputs + output to an ``.npz`` on the Modal volume.

``compare`` (Mac): rebuilds the torch model from dumped weights, converts to
``MLXWan22DiT``, replays the same forward on Metal, and asserts a match.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Tiny config shared by dump and compare (mirrors test_mlx_wan22_parity).
NUM_HEADS, HEAD_DIM, NUM_LAYERS = 4, 16, 2
NUM_FRAMES, HEIGHT, WIDTH, TEXT_DIM, TEXT_LEN = 4, 8, 8, 64, 8
ARCH = dict(
    num_attention_heads=NUM_HEADS,
    attention_head_dim=HEAD_DIM,
    in_channels=16,
    out_channels=16,
    text_dim=TEXT_DIM,
    freq_dim=64,
    ffn_dim=128,
    num_layers=NUM_LAYERS,
    patch_size=(1, 2, 2),
    rope_max_seq_len=64,
)
HF = dict(
    num_attention_heads=NUM_HEADS,
    attention_head_dim=HEAD_DIM,
    in_channels=16,
    out_channels=16,
    text_dim=TEXT_DIM,
    freq_dim=64,
    ffn_dim=128,
    num_layers=NUM_LAYERS,
    patch_size=(1, 2, 2),
    text_len=TEXT_LEN,
    rope_max_seq_len=64,
    eps=1e-6,
)
SEED = 2026


def _build_torch_model():
    import torch

    from fastvideo.configs.models.dits.wanvideo import WanVideoArchConfig, WanVideoConfig
    from fastvideo.models.dits.wanvideo import WanTransformer3DModel

    cfg = WanVideoConfig(arch_config=WanVideoArchConfig(**ARCH))
    model = WanTransformer3DModel(config=cfg, hf_config=HF).eval()
    torch.manual_seed(SEED + 3)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim <= 1:
                if name.endswith("weight") and "norm" in name:
                    param.fill_(1.0)
                else:
                    param.normal_(0.0, 0.02)
            else:
                torch.nn.init.xavier_uniform_(param)
    return model


def _deterministic_inputs():
    rng = np.random.default_rng(SEED + 1)
    latents = rng.standard_normal((1, ARCH["in_channels"], NUM_FRAMES, HEIGHT, WIDTH)).astype(np.float32)
    text = rng.standard_normal((1, TEXT_LEN, TEXT_DIM)).astype(np.float32)
    p_t, p_h, p_w = ARCH["patch_size"]
    tokens_per_frame = (HEIGHT // p_h) * (WIDTH // p_w)
    num_tokens = (NUM_FRAMES // p_t) * tokens_per_frame
    per_frame = [0] + [500] * (NUM_FRAMES // p_t - 1)
    timestep = np.array([[per_frame[i // tokens_per_frame] for i in range(num_tokens)]], dtype=np.int64)
    return latents, text, timestep


def dump(out_path: Path) -> None:
    import torch

    from fastvideo.distributed.parallel_state import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dumping wan22 reference on device={device}", flush=True)
    model = _build_torch_model().to(device, torch.float32)
    latents_np, text_np, timestep_np = _deterministic_inputs()
    latents = torch.from_numpy(latents_np).to(device)
    text = torch.from_numpy(text_np).to(device)
    timestep = torch.from_numpy(timestep_np).to(device)

    with torch.no_grad(), set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=ForwardBatch(data_type="dummy")):
        out = model(hidden_states=latents, encoder_hidden_states=text, timestep=timestep)
        out_np = out.detach().float().cpu().numpy()

    weights = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        device=np.array(device),
        latents=latents_np,
        text=text_np,
        timestep=timestep_np,
        output=out_np,
        **{f"w::{k}": v for k, v in weights.items()},
    )
    print(f"wrote {out_path} ({len(weights)} weight tensors, device={device}, out_shape={out_np.shape})", flush=True)


def compare(npz_path: Path, *, atol: float = 5e-3, rtol: float = 5e-3) -> None:
    import mlx.core as mx
    import torch

    from fastvideo.distributed.parallel_state import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.mlx_runtime.fastwan import mlx_block_weights_from_torch
    from fastvideo.mlx_runtime.wan22 import MLXWan22DiT, MLXWan22TransformerBlock
    from fastvideo.tests.mlx.tiny_wan import TOP_LEVEL_KEY_MAP, mlx_rotary_embeddings

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    data = np.load(npz_path)
    ref_device = str(data["device"])
    ref_output = data["output"]
    weights = {k[len("w::"):]: data[k] for k in data.files if k.startswith("w::")}

    model = _build_torch_model()
    model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})

    state = {name: value.detach().float() for name, value in model.state_dict().items()}
    inner_dim = NUM_HEADS * HEAD_DIM
    mlx_weights = {}
    for mlx_name, torch_name in TOP_LEVEL_KEY_MAP.items():
        tensor = state[torch_name]
        if mlx_name == "patch_embedding.weight":
            tensor = tensor.reshape(inner_dim, -1)
        mlx_weights[mlx_name] = mx.array(tensor.numpy())
    blocks = [
        MLXWan22TransformerBlock(
            mlx_block_weights_from_torch(tb),
            dim=inner_dim,
            ffn_dim=ARCH["ffn_dim"],
            num_heads=NUM_HEADS,
            eps=1e-6,
        ) for tb in model.blocks
    ]
    mlx_model = MLXWan22DiT(mlx_weights, blocks, dict(HF))

    latents = mx.array(data["latents"])
    text = mx.array(data["text"])
    timestep = mx.array(data["timestep"].astype(np.float32))
    freqs_cis = mlx_rotary_embeddings(torch.from_numpy(data["latents"]))
    out = mlx_model(latents, text, timestep, freqs_cis)
    mx.eval(out)
    mlx_output = np.array(out.astype(mx.float32))

    max_abs = float(np.abs(mlx_output - ref_output).max())
    print(f"MLX-Metal vs torch-{ref_device}: max|Δ|={max_abs:.3e} (atol={atol})")
    np.testing.assert_allclose(mlx_output, ref_output, atol=atol, rtol=rtol)
    print("PASS: MLX Wan2.2 per-token-timestep port matches the CUDA reference.")


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA reference dump/compare for MLX Wan2.2 DiT.")
    parser.add_argument("mode", choices=("dump", "compare"))
    parser.add_argument("--path", type=Path, default=Path("/root/data/wan22_ref/ref.npz"))
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    args = parser.parse_args()
    if args.mode == "dump":
        dump(args.path)
    else:
        compare(args.path, atol=args.atol, rtol=args.rtol)


if __name__ == "__main__":
    main()

