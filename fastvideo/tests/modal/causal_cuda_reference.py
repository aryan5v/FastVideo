# SPDX-License-Identifier: Apache-2.0
"""CUDA reference dump for the MLX causal-DiT port (Track C verification).

Two modes:

``dump`` (run on a CUDA GPU, e.g. via ``launch_l40s_job.py``): builds the tiny
causal Wan model, runs ``_forward_inference`` chunk-by-chunk on the GPU with
deterministic (NumPy-seeded, platform-independent) inputs, and writes the model
weights + inputs + per-chunk outputs to an ``.npz`` on the Modal volume.

``compare`` (run on the Mac): loads that ``.npz``, rebuilds the torch model from
the dumped weights, converts it to ``MLXCausalWanDiT``, replays the same chunked
inference on Metal, and asserts the MLX outputs match the CUDA reference. This
closes the one Track-C gate the Mac session could not: MLX-Metal vs real CUDA.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Tiny config shared by dump and compare (mirrors test_mlx_causal_dit_parity).
NUM_HEADS, HEAD_DIM, NUM_LAYERS = 4, 16, 2
NUM_FRAMES, HEIGHT, WIDTH, TEXT_DIM = 4, 8, 8, 64
ARCH = dict(
    num_attention_heads=NUM_HEADS, attention_head_dim=HEAD_DIM, in_channels=16, out_channels=16,
    text_dim=TEXT_DIM, freq_dim=64, ffn_dim=128, num_layers=NUM_LAYERS, patch_size=(1, 2, 2),
    rope_max_seq_len=64, local_attn_size=-1, sink_size=0, num_frames_per_block=1)
HF = dict(
    num_attention_heads=NUM_HEADS, attention_head_dim=HEAD_DIM, in_channels=16, out_channels=16,
    text_dim=TEXT_DIM, freq_dim=64, ffn_dim=128, num_layers=NUM_LAYERS, patch_size=(1, 2, 2), text_len=512,
    rope_max_seq_len=64, eps=1e-6)
SEED = 2026


def _build_torch_model():
    import torch

    from fastvideo.configs.models.dits.wanvideo import WanVideoArchConfig, WanVideoConfig
    from fastvideo.models.dits.causal_wanvideo import CausalWanTransformer3DModel

    cfg = WanVideoConfig(arch_config=WanVideoArchConfig(**ARCH))
    model = CausalWanTransformer3DModel(config=cfg, hf_config=HF).eval()
    torch.manual_seed(SEED + 3)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim <= 1:
                param.fill_(1.0) if (name.endswith("weight") and "norm" in name) else param.normal_(0.0, 0.02)
            else:
                torch.nn.init.xavier_uniform_(param)
    return model


def _deterministic_inputs():
    rng = np.random.default_rng(SEED + 1)
    latents = rng.standard_normal((1, ARCH["in_channels"], NUM_FRAMES, HEIGHT, WIDTH)).astype(np.float32)
    text = rng.standard_normal((1, 24, TEXT_DIM)).astype(np.float32)
    return latents, text


def _rotary():
    import torch

    from fastvideo.layers.rotary_embedding import get_rotary_pos_embed

    d = HEAD_DIM
    rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
    cos, sin = get_rotary_pos_embed(
        (NUM_FRAMES, HEIGHT // 2, WIDTH // 2), NUM_HEADS * HEAD_DIM, NUM_HEADS, rope_dim_list,
        dtype=torch.float32, rope_theta=10000)
    return cos.float().numpy(), sin.float().numpy()


def dump(out_path: Path) -> None:
    import torch

    from fastvideo.distributed.parallel_state import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dumping on device={device}", flush=True)
    model = _build_torch_model().to(device, torch.float32)
    latents_np, text_np = _deterministic_inputs()
    frame_seqlen = (HEIGHT // 2) * (WIDTH // 2)
    window = 21 * frame_seqlen

    latents = torch.from_numpy(latents_np).to(device)
    text = torch.from_numpy(text_np).to(device)
    kv = [{
        "k": torch.zeros(1, window, NUM_HEADS, HEAD_DIM, device=device),
        "v": torch.zeros(1, window, NUM_HEADS, HEAD_DIM, device=device),
        "global_end_index": torch.tensor([0], device=device), "local_end_index": torch.tensor([0], device=device)
    } for _ in range(NUM_LAYERS)]
    cx = [{
        "k": torch.zeros(1, 512, NUM_HEADS, HEAD_DIM, device=device),
        "v": torch.zeros(1, 512, NUM_HEADS, HEAD_DIM, device=device), "is_init": False
    } for _ in range(NUM_LAYERS)]

    outs = []
    with torch.no_grad(), set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=ForwardBatch(data_type="dummy")):
        for i in range(NUM_FRAMES):
            out = model(
                hidden_states=latents[:, :, i:i + 1], encoder_hidden_states=text,
                timestep=torch.tensor([[10]], device=device), kv_cache=kv, crossattn_cache=cx,
                current_start=i * frame_seqlen, cache_start=0, start_frame=i)
            outs.append(out.detach().float().cpu().numpy())

    weights = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path, device=np.array(device), latents=latents_np, text=text_np,
        outputs=np.concatenate(outs, axis=2), **{f"w::{k}": v for k, v in weights.items()})
    print(f"wrote {out_path} ({len(weights)} weight tensors, device={device})", flush=True)


def compare(npz_path: Path, *, atol: float = 5e-3, rtol: float = 5e-3) -> None:
    import mlx.core as mx
    import torch

    from fastvideo.distributed.parallel_state import maybe_init_distributed_environment_and_model_parallel

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    data = np.load(npz_path)
    ref_device = str(data["device"])
    ref_outputs = data["outputs"]
    weights = {k[len("w::"):]: data[k] for k in data.files if k.startswith("w::")}

    # Rebuild the torch model with the *dumped* weights, then convert to MLX.
    model = _build_torch_model()
    model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
    from fastvideo.tests.mlx.test_mlx_causal_dit_parity import _mlx_from_torch

    mlx_model = _mlx_from_torch(model)

    cos_np, sin_np = _rotary()
    cos, sin = mx.array(cos_np), mx.array(sin_np)
    frame_seqlen = (HEIGHT // 2) * (WIDTH // 2)
    kv_caches, crossattn_caches = mlx_model.allocate_caches(batch=1, frame_seqlen=frame_seqlen, dtype=mx.float32)

    latents = data["latents"]
    text = mx.array(data["text"])
    outs = []
    for i in range(NUM_FRAMES):
        out = mlx_model.forward_chunk(
            mx.array(latents[:, :, i:i + 1]), text, mx.array([[10.0]]),
            cos[i * frame_seqlen:(i + 1) * frame_seqlen], sin[i * frame_seqlen:(i + 1) * frame_seqlen],
            kv_caches, crossattn_caches, current_start=i * frame_seqlen)
        mx.eval(out)
        outs.append(np.array(out.astype(mx.float32)))
    mlx_outputs = np.concatenate(outs, axis=2)

    max_abs = float(np.abs(mlx_outputs - ref_outputs).max())
    print(f"MLX-Metal vs torch-{ref_device}: max|Δ|={max_abs:.3e} (atol={atol})")
    np.testing.assert_allclose(mlx_outputs, ref_outputs, atol=atol, rtol=rtol)
    print("PASS: MLX causal port matches the CUDA reference.")


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA reference dump / compare for the MLX causal DiT.")
    parser.add_argument("mode", choices=("dump", "compare"))
    parser.add_argument("--path", type=Path, default=Path("/root/data/causal_ref/causal_cuda_ref.npz"))
    args = parser.parse_args()
    if args.mode == "dump":
        dump(args.path)
    else:
        compare(args.path)


if __name__ == "__main__":
    main()

