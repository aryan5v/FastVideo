# SPDX-License-Identifier: Apache-2.0
"""The M4 Phase A gate: torch fake-quant must bit-match MLX's quantizer.

The roadmap requires this parity to hold BEFORE any GPU is spent on a
quantization-aware training run: if train-time fake-quantization differs from
``mx.quantize``/``mx.dequantize`` at deploy time, the QAT gains evaporate.

These tests compare the pure-torch reference in
``fastvideo/layers/quantization/mlx_affine_qat.py`` against the real MLX
implementation, code-for-code (unpacked integer codes) and value-for-value
(dequantized weights, scales, biases). Runs on any MLX backend (Metal or
``mlx[cpu]``) — no GPU or distributed setup needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core", reason="MLX is required for QAT numerics parity tests")

from fastvideo.layers.quantization.mlx_affine_qat import (  # noqa: E402
    fake_quantize_mlx_affine,
    mlx_affine_dequantize_reference,
    mlx_affine_quantize_reference,
)


def _unpack_uint32_codes(packed: np.ndarray, *, bits: int, out_cols: int) -> np.ndarray:
    """Unpack MLX's little-endian uint32 words into per-element integer codes."""
    el_per_word = 32 // bits
    bitmask = (1 << bits) - 1
    words = packed.astype(np.uint64)
    codes = np.zeros((*packed.shape[:-1], packed.shape[-1] * el_per_word), dtype=np.int32)
    for k in range(el_per_word):
        codes[..., k::el_per_word] = ((words >> (k * bits)) & bitmask).astype(np.int32)
    return codes[..., :out_cols]


def _mlx_quantize(w_np: np.ndarray, *, group_size: int, bits: int):
    q, scales, biases = mx.quantize(mx.array(w_np), group_size=group_size, bits=bits, mode="affine")
    deq = mx.dequantize(q, scales, biases, group_size=group_size, bits=bits, mode="affine")
    mx.eval(q, scales, biases, deq)
    return (np.array(q), np.array(scales), np.array(biases), np.array(deq))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("shape", [(4, 128), (3, 64), (16, 256)])
def test_codes_scales_biases_bitmatch_mlx(dtype: torch.dtype, bits: int, shape: tuple[int, int]) -> None:
    torch.manual_seed(1234)
    w = torch.randn(*shape, dtype=torch.float32).to(dtype) * 0.05

    codes, scales, biases = mlx_affine_quantize_reference(w, group_size=64, bits=bits)

    w_np = w.float().numpy() if dtype == torch.float32 else w.numpy()
    q_mlx, scales_mlx, biases_mlx, _ = _mlx_quantize(w_np, group_size=64, bits=bits)

    codes_mlx = _unpack_uint32_codes(q_mlx, bits=bits, out_cols=shape[-1])
    codes_flat = codes.reshape(shape[0], -1).numpy()

    np.testing.assert_array_equal(codes_flat, codes_mlx)
    np.testing.assert_array_equal(scales.float().numpy(), np.asarray(scales_mlx, dtype=np.float32))
    np.testing.assert_array_equal(biases.float().numpy(), np.asarray(biases_mlx, dtype=np.float32))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_dequantized_weights_bitmatch_mlx(dtype: torch.dtype) -> None:
    torch.manual_seed(7)
    w = torch.randn(8, 192, dtype=torch.float32).to(dtype) * 0.05

    codes, scales, biases = mlx_affine_quantize_reference(w, group_size=64, bits=8)
    deq = mlx_affine_dequantize_reference(codes, scales, biases, out_shape=w.shape)

    w_np = w.float().numpy() if dtype == torch.float32 else w.numpy()
    _, _, _, deq_mlx = _mlx_quantize(w_np, group_size=64, bits=8)

    np.testing.assert_array_equal(deq.float().numpy(), np.asarray(deq_mlx, dtype=np.float32))


def test_fake_quantize_matches_deploy_pipeline_and_passes_gradients() -> None:
    torch.manual_seed(99)
    # bf16 master weights, exactly like a QAT training run.
    w = (torch.randn(4, 128, dtype=torch.float32) * 0.05).to(torch.bfloat16).requires_grad_(True)

    fq = fake_quantize_mlx_affine(w, group_size=64, bits=8, simulate_dtype=torch.float16)

    # Forward: identical to quantizing the fp16-cast weight with MLX (the
    # loader casts checkpoints to fp16 before mx.quantize).
    w_fp16 = w.detach().to(torch.float16)
    _, _, _, deq_mlx = _mlx_quantize(w_fp16.numpy(), group_size=64, bits=8)
    np.testing.assert_array_equal(
        fq.detach().to(torch.float16).float().numpy(),
        np.asarray(deq_mlx, dtype=np.float32),
    )

    # Backward: straight-through — gradients reach the master weight unchanged.
    fq.sum().backward()
    assert w.grad is not None
    np.testing.assert_array_equal(w.grad.float().numpy(), np.ones_like(w.grad.float().numpy()))


def test_quantized_matmul_close_to_fake_quant_linear() -> None:
    torch.manual_seed(3)
    w = (torch.randn(128, 256, dtype=torch.float32) * 0.05).to(torch.float16)
    x = (torch.randn(2, 256, dtype=torch.float32) * 0.5).to(torch.float16)

    codes, scales, biases = mlx_affine_quantize_reference(w, group_size=64, bits=8)
    deq = mlx_affine_dequantize_reference(codes, scales, biases, out_shape=w.shape)
    y_torch = (x.float() @ deq.float().T)

    q, s, b = mx.quantize(mx.array(w.numpy()), group_size=64, bits=8, mode="affine")
    y_mlx = mx.quantized_matmul(
        mx.array(x.numpy()), q, s, b, transpose=True, group_size=64, bits=8, mode="affine")
    mx.eval(y_mlx)

    # Accumulation order differs between frameworks, so this is a tolerance
    # check (the weight values themselves are pinned bitwise above).
    np.testing.assert_allclose(y_torch.numpy(), np.array(y_mlx).astype(np.float32), atol=2e-2, rtol=2e-2)


def test_indivisible_group_size_raises() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        mlx_affine_quantize_reference(torch.randn(4, 100), group_size=64, bits=8)
