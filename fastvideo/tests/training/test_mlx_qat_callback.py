# SPDX-License-Identifier: Apache-2.0
"""The MLX QAT callback: fake-quantized forwards, straight-through training.

Torch-only (no MLX needed): the underlying quantizer numerics are pinned
bitwise against MLX in ``test_mlx_affine_qat_parity.py``; these tests cover
the callback mechanics — module targeting, parametrization behavior, gradient
flow to master weights, and re-quantization after optimizer steps.
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


def test_callback_targets_matrix_weights_and_skips_norms_and_indivisible() -> None:
    transformer = _TinyStudentTransformer()
    callback = MLXQuantizationAwareCallback(group_size=64, bits=8)

    callback.on_train_start(_method_with_student(transformer))

    assert set(callback.quantized_module_names) == {"to_q", "ffn_fc_in", "patch_embedding"}
    assert torch.nn.utils.parametrize.is_parametrized(transformer.to_q, "weight")
    assert not torch.nn.utils.parametrize.is_parametrized(transformer.norm_q, "weight")
    assert not torch.nn.utils.parametrize.is_parametrized(transformer.tiny, "weight")


def test_forward_sees_the_deploy_grid_and_conv_weights_quantize_flattened() -> None:
    transformer = _TinyStudentTransformer()
    master_linear = transformer.to_q.weight.detach().clone()
    master_conv = transformer.patch_embedding.weight.detach().clone()

    MLXQuantizationAwareCallback(group_size=64, bits=8).on_train_start(_method_with_student(transformer))

    expected_linear = fake_quantize_mlx_affine(master_linear, group_size=64, bits=8).to(master_linear.dtype)
    torch.testing.assert_close(transformer.to_q.weight, expected_linear, atol=0, rtol=0)

    flattened = master_conv.reshape(master_conv.shape[0], -1)
    expected_conv = fake_quantize_mlx_affine(flattened, group_size=64,
                                             bits=8).reshape(master_conv.shape).to(master_conv.dtype)
    torch.testing.assert_close(transformer.patch_embedding.weight, expected_conv, atol=0, rtol=0)


def test_gradients_flow_to_master_weights_and_requantize_after_step() -> None:
    torch.manual_seed(11)
    transformer = _TinyStudentTransformer()
    MLXQuantizationAwareCallback(group_size=64, bits=8).on_train_start(_method_with_student(transformer))

    master = transformer.to_q.parametrizations.weight.original
    optimizer = torch.optim.SGD(transformer.parameters(), lr=0.5)

    out = transformer(torch.randn(4, 128))
    out.square().mean().backward()
    assert master.grad is not None and master.grad.abs().sum() > 0

    before_master = master.detach().clone()
    before_effective = transformer.to_q.weight.detach().clone()
    optimizer.step()

    assert not torch.equal(master.detach(), before_master)
    after_effective = transformer.to_q.weight.detach()
    # The effective weight is re-quantized from the updated master and stays
    # on the deploy grid.
    expected = fake_quantize_mlx_affine(master.detach(), group_size=64, bits=8).to(after_effective.dtype)
    torch.testing.assert_close(after_effective, expected, atol=0, rtol=0)
    assert not torch.equal(after_effective, before_effective)


def test_no_matching_weights_raises() -> None:
    with pytest.raises(ValueError, match="matched no weights"):
        MLXQuantizationAwareCallback().on_train_start(_method_with_student(torch.nn.Module()))


def test_missing_student_raises() -> None:
    with pytest.raises(ValueError, match="No student transformer"):
        MLXQuantizationAwareCallback().on_train_start(types.SimpleNamespace(student=None))
