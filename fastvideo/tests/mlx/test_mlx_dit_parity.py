# SPDX-License-Identifier: Apache-2.0
"""Full-DiT parity: the MLX Wan runtime vs the PyTorch reference model.

This is the M1 "trustworthy baseline" gate for the Apple Silicon path: a tiny
random-weight ``WanTransformer3DModel`` is run end to end (patch embed ->
condition -> transformer blocks -> unpatchify) in PyTorch and in
``fastvideo.mlx_runtime.fastwan.MLXWanDiT`` with identical weights, and the
outputs must match within pinned fp32 tolerances.

Runs anywhere MLX is installed: on Apple Silicon it exercises the Metal
device, on Linux/CI it runs on MLX's CPU backend (``pip install 'mlx[cpu]'``)
-- the graph is identical, so CPU parity is the CI-friendly golden variant.

    pytest fastvideo/tests/mlx/test_mlx_dit_parity.py -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29513")

mx = pytest.importorskip("mlx.core", reason="MLX is required for DiT parity tests")

from fastvideo.configs.models.dits.wanvideo import (  # noqa: E402
    WanVideoArchConfig,
    WanVideoConfig,
)
from fastvideo.forward_context import set_forward_context  # noqa: E402
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed  # noqa: E402
from fastvideo.mlx_runtime.fastwan import (  # noqa: E402
    MLXQuantizationSpec,
    MLXWanDiT,
    MLXWanTransformerBlock,
    mlx_block_weights_from_torch,
    quantize_matrix,
)
from fastvideo.models.dits.wanvideo import WanTransformer3DModel  # noqa: E402
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch  # noqa: E402

SEED = 2026

# Pinned fp32 tolerances for the full forward (patch embed through unpatchify),
# matching the single-block parity example. Measured max_abs_diff on the MLX
# CPU backend is ~1.7e-6, so 2e-4 keeps ~100x headroom for accumulation-order
# differences across MLX backends while still catching real math/layout bugs.
FP32_ATOL = 2e-4
FP32_RTOL = 2e-4

# Quantized inference is lossy by design; gate it on signal-to-noise vs the
# fp32 MLX output instead of elementwise closeness. int8 (group size 64) on
# this tiny model measures ~43 dB; 20 dB leaves headroom without letting a
# broken dequant path (which lands near 0 dB) slip through.
INT8_MIN_SNR_DB = 20.0

# All matmul dims must be multiples of the int8 group size (64) so the
# quantized variant of the test can reuse the same tiny model.
TINY_ARCH = dict(
    num_attention_heads=4,
    attention_head_dim=16,
    in_channels=16,
    out_channels=16,
    text_dim=64,
    freq_dim=64,
    ffn_dim=128,
    num_layers=2,
    patch_size=(1, 2, 2),
    rope_max_seq_len=64,
)


def _build_tiny_wan_config() -> WanVideoConfig:
    return WanVideoConfig(arch_config=WanVideoArchConfig(**TINY_ARCH))


def _build_hf_config(config: WanVideoConfig) -> dict[str, object]:
    return {
        "num_attention_heads": config.num_attention_heads,
        "attention_head_dim": config.attention_head_dim,
        "in_channels": config.in_channels,
        "out_channels": config.out_channels,
        "text_dim": config.text_dim,
        "freq_dim": config.freq_dim,
        "ffn_dim": config.ffn_dim,
        "num_layers": config.num_layers,
        "patch_size": config.patch_size,
        "text_len": config.text_len,
        "rope_max_seq_len": config.rope_max_seq_len,
        "eps": 1e-6,
    }


def _initialize_model_parameters(model: torch.nn.Module) -> None:
    # ReplicatedLinear parameters are allocated with torch.empty and need an
    # explicit initialization in tests to avoid undefined values.
    torch.manual_seed(SEED + 3)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim <= 1:
                if name.endswith("weight") and "norm" in name:
                    param.fill_(1.0)
                else:
                    param.normal_(mean=0.0, std=0.02)
                continue
            torch.nn.init.xavier_uniform_(param)


def _build_torch_model() -> WanTransformer3DModel:
    config = _build_tiny_wan_config()
    model = WanTransformer3DModel(config=config, hf_config=_build_hf_config(config))
    model = model.to(device="cpu", dtype=torch.float32)
    _initialize_model_parameters(model)
    model.eval()
    return model


def _build_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    hidden_states = torch.randn(1, TINY_ARCH["in_channels"], 4, 8, 8, generator=generator, dtype=torch.float32)
    encoder_hidden_states = torch.randn(1, 8, TINY_ARCH["text_dim"], generator=generator, dtype=torch.float32)
    timestep = torch.tensor([10], dtype=torch.long)
    return hidden_states, encoder_hidden_states, timestep


# Top-level weights: the MLX runtime uses the Diffusers key layout; the torch
# model uses FastVideo's module names (see WanVideoConfig.param_names_mapping).
_TOP_LEVEL_KEY_MAP = {
    "patch_embedding.weight": "patch_embedding.proj.weight",
    "patch_embedding.bias": "patch_embedding.proj.bias",
    "condition_embedder.time_embedder.linear_1.weight": "condition_embedder.time_embedder.mlp.fc_in.weight",
    "condition_embedder.time_embedder.linear_1.bias": "condition_embedder.time_embedder.mlp.fc_in.bias",
    "condition_embedder.time_embedder.linear_2.weight": "condition_embedder.time_embedder.mlp.fc_out.weight",
    "condition_embedder.time_embedder.linear_2.bias": "condition_embedder.time_embedder.mlp.fc_out.bias",
    "condition_embedder.time_proj.weight": "condition_embedder.time_modulation.linear.weight",
    "condition_embedder.time_proj.bias": "condition_embedder.time_modulation.linear.bias",
    "condition_embedder.text_embedder.linear_1.weight": "condition_embedder.text_embedder.fc_in.weight",
    "condition_embedder.text_embedder.linear_1.bias": "condition_embedder.text_embedder.fc_in.bias",
    "condition_embedder.text_embedder.linear_2.weight": "condition_embedder.text_embedder.fc_out.weight",
    "condition_embedder.text_embedder.linear_2.bias": "condition_embedder.text_embedder.fc_out.bias",
    "scale_shift_table": "scale_shift_table",
    "proj_out.weight": "proj_out.weight",
    "proj_out.bias": "proj_out.bias",
}


def _mlx_dit_from_torch_model(
    model: WanTransformer3DModel,
    hf_config: dict[str, object],
    *,
    quantization: MLXQuantizationSpec | None = None,
) -> MLXWanDiT:
    state = {name: value.detach().float() for name, value in model.state_dict().items()}
    inner_dim = int(hf_config["num_attention_heads"]) * int(hf_config["attention_head_dim"])  # type: ignore[arg-type]

    weights = {}
    for mlx_name, torch_name in _TOP_LEVEL_KEY_MAP.items():
        tensor = state[torch_name]
        if mlx_name == "patch_embedding.weight":
            tensor = tensor.reshape(inner_dim, -1)
        array = mx.array(tensor.numpy())
        if quantization is not None and mlx_name.endswith(".weight") and mlx_name != "scale_shift_table":
            weights[mlx_name] = quantize_matrix(array, quantization)
        else:
            weights[mlx_name] = array

    blocks = []
    for torch_block in model.blocks:
        block_weights = mlx_block_weights_from_torch(torch_block)
        if quantization is not None:
            block_weights = {
                name: (quantize_matrix(value, quantization)
                       if name.endswith(".weight") and "norm" not in name and len(value.shape) >= 2 else value)
                for name, value in block_weights.items()
            }
        blocks.append(
            MLXWanTransformerBlock(
                block_weights,
                dim=inner_dim,
                ffn_dim=int(hf_config["ffn_dim"]),  # type: ignore[arg-type]
                num_heads=int(hf_config["num_attention_heads"]),  # type: ignore[arg-type]
                eps=float(hf_config["eps"]),  # type: ignore[arg-type]
            ))
    return MLXWanDiT(weights, blocks, dict(hf_config))


def _mlx_rotary_embeddings(hidden_states: torch.Tensor) -> tuple["mx.array", "mx.array"]:
    """The rotary table the torch model builds internally, converted to MLX."""
    _, _, frames, height, width = hidden_states.shape
    p_t, p_h, p_w = TINY_ARCH["patch_size"]
    head_dim = TINY_ARCH["attention_head_dim"]
    hidden_size = TINY_ARCH["num_attention_heads"] * head_dim
    rope_dim_list = [head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6)]
    freqs_cos, freqs_sin = get_rotary_pos_embed(
        (frames // p_t, height // p_h, width // p_w),
        hidden_size,
        TINY_ARCH["num_attention_heads"],
        rope_dim_list,
        dtype=torch.float64,
        rope_theta=10000,
    )
    return (
        mx.array(freqs_cos.float().numpy()).astype(mx.float32),
        mx.array(freqs_sin.float().numpy()).astype(mx.float32),
    )


def _torch_reference_output(model, hidden_states, encoder_hidden_states, timestep) -> np.ndarray:
    with torch.no_grad(), set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=ForwardBatch(data_type="dummy"),
    ):
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
        )
    return output.detach().float().cpu().numpy()


def _mlx_output(dit, hidden_states, encoder_hidden_states, timestep, freqs_cis) -> np.ndarray:
    out = dit(
        mx.array(hidden_states.numpy()),
        mx.array(encoder_hidden_states.numpy()),
        mx.array(timestep.float().numpy()),
        freqs_cis,
    )
    mx.eval(out)
    return np.array(out.astype(mx.float32))


def test_full_dit_forward_matches_torch_reference(distributed_setup) -> None:
    model = _build_torch_model()
    hidden_states, encoder_hidden_states, timestep = _build_inputs()

    torch_out = _torch_reference_output(model, hidden_states, encoder_hidden_states, timestep)

    dit = _mlx_dit_from_torch_model(model, _build_hf_config(_build_tiny_wan_config()))
    mlx_out = _mlx_output(dit, hidden_states, encoder_hidden_states, timestep, _mlx_rotary_embeddings(hidden_states))

    assert mlx_out.shape == torch_out.shape
    assert np.isfinite(mlx_out).all()
    max_abs = float(np.abs(torch_out - mlx_out).max())
    assert np.allclose(torch_out, mlx_out, atol=FP32_ATOL, rtol=FP32_RTOL), (
        f"MLX full-DiT forward diverged from the torch reference: max_abs_diff={max_abs:.3e} "
        f"(atol={FP32_ATOL}, rtol={FP32_RTOL})")


def test_full_dit_forward_int8_stays_close_to_fp32(distributed_setup) -> None:
    model = _build_torch_model()
    hidden_states, encoder_hidden_states, timestep = _build_inputs()
    hf_config = _build_hf_config(_build_tiny_wan_config())
    freqs_cis = _mlx_rotary_embeddings(hidden_states)

    fp32_out = _mlx_output(
        _mlx_dit_from_torch_model(model, hf_config), hidden_states, encoder_hidden_states, timestep, freqs_cis)
    int8_out = _mlx_output(
        _mlx_dit_from_torch_model(model, hf_config, quantization=MLXQuantizationSpec.from_name("int8")),
        hidden_states, encoder_hidden_states, timestep, freqs_cis)

    assert np.isfinite(int8_out).all()
    noise = float(np.mean(np.square(int8_out - fp32_out)))
    signal = float(np.mean(np.square(fp32_out)))
    snr_db = 10.0 * np.log10(signal / noise) if noise > 0 else float("inf")
    assert snr_db >= INT8_MIN_SNR_DB, (
        f"int8-quantized MLX DiT output is too far from fp32: SNR {snr_db:.1f} dB < {INT8_MIN_SNR_DB} dB")
