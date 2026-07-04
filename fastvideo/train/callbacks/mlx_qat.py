# SPDX-License-Identifier: Apache-2.0
"""Quantization-aware training callback targeting the Apple/MLX runtime.

Registers a weight parametrization on the student transformer's linear
weights so every forward sees MLX-affine fake-quantized weights (the exact
deploy-time grid of ``mx.quantize``/``mx.dequantize``; see
``fastvideo/layers/quantization/mlx_affine_qat.py`` and its bitwise parity
tests), while gradients flow straight-through to the master weights. Composes
with any ``TrainingMethod`` (DMD2, KD, fine-tune) via YAML:

.. code-block:: yaml

    callbacks:
      mlx_qat:
        group_size: 64
        bits: 8

Targeting mirrors ``mlx_dit_from_diffusers_safetensors``: 2-D (or reshapable
conv) ``.weight`` tensors whose grouped dim divides ``group_size``, excluding
norms and modulation tables. Note that under bf16 autocast the fake-quantized
values are rounded once more to bf16 for the matmul — the quantization
*decisions* (codes, scales, biases) still bit-match deploy time, which is
what QAT needs to learn the deploy grid.

FSDP caveat: parametrizations must be registered before sharding wraps the
modules; this callback runs in ``on_train_start`` after model setup, so the
first multi-GPU smoke run should verify parametrized weights shard cleanly.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
import torch.nn.utils.parametrize as parametrize

from fastvideo.layers.quantization.mlx_affine_qat import fake_quantize_mlx_affine
from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)

# Weights the MLX loader never quantizes: norms and modulation tables.
DEFAULT_EXCLUDE_PATTERNS = (r"norm", r"scale_shift_table")

_SIMULATE_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


class _MLXAffineFakeQuantParametrization(torch.nn.Module):
    """Weight parametrization: master weight -> MLX-affine dequant grid (STE)."""

    def __init__(self, *, group_size: int, bits: int, simulate_dtype: torch.dtype,
                 compute_dtype: torch.dtype | None) -> None:
        super().__init__()
        self.group_size = group_size
        self.bits = bits
        self.simulate_dtype = simulate_dtype
        self.compute_dtype = compute_dtype

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        original_shape = weight.shape
        # Conv-style weights (e.g. Wan's patch embedding) quantize over the
        # flattened non-output dims, matching the MLX loader's (out, -1) reshape.
        weight2d = weight.reshape(original_shape[0], -1) if weight.dim() > 2 else weight
        fq = fake_quantize_mlx_affine(
            weight2d,
            group_size=self.group_size,
            bits=self.bits,
            simulate_dtype=self.simulate_dtype,
        )
        # Present the weight in the model's compute dtype. Under FSDP/HSDP
        # mixed precision the master (`parametrizations.weight.original`) is
        # stored fp32 and only cast at forward time; without an explicit
        # compute_dtype, code that reads `module.weight` outside autocast
        # (validation pipelines, dtype sniffing) would see fp32 weights next
        # to bf16 buffers/biases and crash with a dtype mismatch.
        return fq.reshape(original_shape).to(self.compute_dtype or weight.dtype)


class MLXQuantizationAwareCallback(Callback):
    """Apply MLX-affine fake quantization to the student's weights.

    Args (all YAML-configurable):
        group_size: quantization group size along the input dim (MLX default 64).
        bits: 8 (deploy target) or 4 (evaluated after INT8 proves out).
        simulate_dtype: precision the deploy path casts weights to before
            quantizing ("fp16" matches the MLX loader).
        compute_dtype: dtype `module.weight` presents to forwards and to any
            code that introspects it. Set this to the training precision
            (e.g. "bf16") whenever the trainer keeps fp32 master weights
            (FSDP/HSDP mixed precision) — otherwise validation and other
            non-autocast readers see fp32 weights against bf16 biases.
            ``None`` presents the master's own dtype.
        exclude_patterns: regex fragments; a weight is skipped when any
            matches its module name.
    """

    def __init__(
        self,
        *,
        group_size: int = 64,
        bits: int = 8,
        simulate_dtype: str = "fp16",
        compute_dtype: str | None = None,
        exclude_patterns: tuple[str, ...] | list[str] = DEFAULT_EXCLUDE_PATTERNS,
    ) -> None:
        self._group_size = int(group_size)
        self._bits = int(bits)
        self._simulate_dtype = _SIMULATE_DTYPES[simulate_dtype]
        self._compute_dtype = None if compute_dtype is None else _SIMULATE_DTYPES[compute_dtype]
        self._exclude = [re.compile(pattern) for pattern in exclude_patterns]
        self.quantized_module_names: list[str] = []

    def _is_target(self, module_name: str, module: torch.nn.Module) -> bool:
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.nn.Parameter) or weight.dim() < 2:
            return False
        if any(pattern.search(module_name) for pattern in self._exclude):
            return False
        grouped_dim = weight.shape[1:].numel()
        if grouped_dim % self._group_size != 0:
            logger.warning(
                "mlx_qat: skipping %s — grouped dim %d is not divisible by group_size %d "
                "(the MLX runtime could not quantize this weight either).",
                module_name, grouped_dim, self._group_size)
            return False
        return True

    def on_train_start(self, method: TrainingMethod, iteration: int = 0) -> None:
        student = getattr(method, "student", None)
        if student is None or student.transformer is None:
            raise ValueError("No student transformer found on method; cannot apply MLX QAT")

        self.quantized_module_names = []
        for module_name, module in student.transformer.named_modules():
            if not self._is_target(module_name, module):
                continue
            parametrize.register_parametrization(
                module,
                "weight",
                _MLXAffineFakeQuantParametrization(
                    group_size=self._group_size,
                    bits=self._bits,
                    simulate_dtype=self._simulate_dtype,
                    compute_dtype=self._compute_dtype,
                ),
                unsafe=True,  # the parametrized dtype may differ from the master's.
            )
            self.quantized_module_names.append(module_name)

        if not self.quantized_module_names:
            raise ValueError(
                "mlx_qat matched no weights on the student transformer — check exclude_patterns "
                "and group_size against the model architecture.")
        logger.info(
            "mlx_qat: fake-quantizing %d weights (int%d, group_size=%d, simulate=%s), e.g. %s",
            len(self.quantized_module_names), self._bits, self._group_size, self._simulate_dtype,
            self.quantized_module_names[:3],
        )
