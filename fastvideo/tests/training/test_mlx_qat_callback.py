# SPDX-License-Identifier: Apache-2.0
"""The MLX QAT callback: fake-quantized forwards, straight-through training.

Torch-only (no MLX needed): the underlying quantizer numerics are pinned
bitwise against MLX in ``test_mlx_affine_qat_parity.py``; these tests cover
the callback mechanics — module targeting, the forward-scoped weight swap,
gradient flow to the real weights, and requantization after optimizer steps.

The forward-scoped swap (rather than ``torch.nn.utils.parametrize``) is
deliberate: under FSDP2 the parameters are sharded DTensors outside forwards
and unsharded plain tensors inside them, so the module must look completely
vanilla except during its own forward call.
"""

from __future__ import annotations

import types

import pytest
import torch
# fastvideo.dataset.preprocessing_datasets references
# torch.distributed.checkpoint.stateful without importing the submodule and
# relies on an earlier import having loaded it; make that explicit here so
# this test does not depend on import order elsewhere in the suite.
import torch.distributed.checkpoint.stateful  # noqa: F401

from fastvideo.layers.quantization.mlx_affine_qat import fake_quantize_mlx_affine
from fastvideo.train.callbacks.mlx_qat import MLXQuantizationAwareCallback


class _TinyStudentTransformer(torch.nn.Module):
    """Names mirror the Wan layout the exclude patterns are written against."""

    def __init__(self) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(128, 128, bias=False)
        self.ffn_fc_in = torch.nn.Linear(128, 256, bias=True)
        self.norm_q = torch.nn.Linear(128, 128, bias=False)  # excluded by name
        self.tiny = torch.nn.Linear(100, 32, bias=False)  # indivisible -> skipped
        self.patch_embedding = torch.nn.Conv3d(16, 128, kernel_size=(1, 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn_fc_in(self.to_q(x))


def _method_with_student(transformer: torch.nn.Module):
    student = types.SimpleNamespace(transformer=transformer)
    return types.SimpleNamespace(student=student)


def _expected_fq(weight: torch.Tensor) -> torch.Tensor:
    flat = weight.reshape(weight.shape[0], -1) if weight.dim() > 2 else weight
    fq = fake_quantize_mlx_affine(flat, group_size=64, bits=8)
    return fq.reshape(weight.shape).to(weight.dtype)


def test_callback_targets_matrix_weights_and_skips_norms_and_indivisible() -> None:
    transformer = _TinyStudentTransformer()
    callback = MLXQuantizationAwareCallback(group_size=64, bits=8)

    callback.on_train_start(_method_with_student(transformer))

    assert set(callback.quantized_module_names) == {"to_q", "ffn_fc_in", "patch_embedding"}
    assert getattr(transformer.to_q, "_mlx_qat_wrapped", False)
    assert not getattr(transformer.norm_q, "_mlx_qat_wrapped", False)
    assert not getattr(transformer.tiny, "_mlx_qat_wrapped", False)


def test_module_stays_vanilla_outside_forward() -> None:
    transformer = _TinyStudentTransformer()
    MLXQuantizationAwareCallback(group_size=64, bits=8).on_train_start(_method_with_student(transformer))

    # Outside a forward call the module must look untouched: `weight` is the
    # real Parameter in _parameters (what FSDP, optimizers, checkpointing,
    # and dtype sniffing all see).
    assert isinstance(transformer.to_q.weight, torch.nn.Parameter)
    assert "weight" in transformer.to_q._parameters
    assert "weight" not in transformer.to_q.__dict__
    assert next(transformer.parameters()).dtype == torch.float32


def test_forward_computes_with_the_deploy_grid() -> None:
    torch.manual_seed(2)
    transformer = _TinyStudentTransformer()
    MLXQuantizationAwareCallback(group_size=64, bits=8).on_train_start(_method_with_student(transformer))

    x = torch.randn(4, 128)
    out = transformer.to_q(x)
    expected = x @ _expected_fq(transformer.to_q.weight.detach()).T
    torch.testing.assert_close(out, expected, atol=0, rtol=0)

    # Conv weights quantize over the flattened non-output dims.
    latent = torch.randn(1, 16, 4, 8, 8)
    conv = transformer.patch_embedding
    out_conv = conv(latent)
    expected_conv = torch.nn.functional.conv3d(
        latent, _expected_fq(conv.weight.detach()), conv.bias, stride=conv.stride)
    torch.testing.assert_close(out_conv, expected_conv, atol=0, rtol=0)


def test_gradients_flow_to_real_weights_and_requantize_after_step() -> None:
    torch.manual_seed(11)
    transformer = _TinyStudentTransformer()
    MLXQuantizationAwareCallback(group_size=64, bits=8).on_train_start(_method_with_student(transformer))

    weight = transformer.to_q.weight
    optimizer = torch.optim.SGD(transformer.parameters(), lr=0.5)

    out = transformer(torch.randn(4, 128))
    out.square().mean().backward()
    assert weight.grad is not None and weight.grad.abs().sum() > 0

    before_weight = weight.detach().clone()
    optimizer.step()
    assert not torch.equal(weight.detach(), before_weight)

    # The next forward requantizes from the updated weight and stays on the
    # deploy grid.
    x = torch.randn(2, 128)
    torch.testing.assert_close(
        transformer.to_q(x), x @ _expected_fq(weight.detach()).T, atol=0, rtol=0)


def test_wrapping_is_idempotent() -> None:
    transformer = _TinyStudentTransformer()
    method = _method_with_student(transformer)
    first = MLXQuantizationAwareCallback(group_size=64, bits=8)
    first.on_train_start(method)
    with pytest.raises(ValueError, match="matched no weights"):
        # A second callback finds nothing left to wrap instead of
        # double-quantizing.
        MLXQuantizationAwareCallback(group_size=64, bits=8).on_train_start(method)


def test_no_matching_weights_raises() -> None:
    with pytest.raises(ValueError, match="matched no weights"):
        MLXQuantizationAwareCallback().on_train_start(_method_with_student(torch.nn.Module()))


def test_missing_student_raises() -> None:
    with pytest.raises(ValueError, match="No student transformer"):
        MLXQuantizationAwareCallback().on_train_start(types.SimpleNamespace(student=None))


def test_weight_is_restored_even_when_forward_raises() -> None:
    transformer = _TinyStudentTransformer()
    MLXQuantizationAwareCallback(group_size=64, bits=8).on_train_start(_method_with_student(transformer))

    with pytest.raises(RuntimeError):
        transformer.to_q(torch.randn(2, 64))  # wrong input dim -> matmul error

    assert isinstance(transformer.to_q.weight, torch.nn.Parameter)
    assert "weight" in transformer.to_q._parameters
    assert "weight" not in transformer.to_q.__dict__
