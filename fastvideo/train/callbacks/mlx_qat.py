# SPDX-License-Identifier: Apache-2.0
"""Quantization-aware training callback targeting the Apple/MLX runtime.

Wraps the student transformer's target modules so every forward computes with
MLX fake-quantized weights (the exact deploy-time grid of ``mx.quantize`` /
``mx.dequantize``), while gradients flow straight-through to the real weights.
The default ``mode: affine`` is the INT8 path used by the first Mac QAD runs;
``mode: mxfp4`` switches to the fixed MXFP4 grid (group size 32, no ``bits``).
Composes with any ``TrainingMethod`` (DMD2, KD, fine-tune) via YAML:

.. code-block:: yaml

    callbacks:
      mlx_qat:
        mode: affine
        group_size: 64
        bits: 8

.. code-block:: yaml

    callbacks:
      mlx_qat:
        mode: mxfp4
        simulate_dtype: fp16

Mechanism: a forward-scoped weight swap, NOT ``torch.nn.utils.parametrize``.
Under FSDP2/HSDP the parameters are sharded DTensors *outside* module
forwards and unsharded plain tensors (already cast to the compute dtype)
*inside* them. Parametrizations restructure the parameter into a submodule
and compute from the raw master, which breaks both worlds: dtype sniffing
sees fp32 masters, and the forward mixes sharded DTensors with plain tensors
(``aten.convolution.default: got mixed torch.Tensor and DTensor``, observed
on a DGX B200). Swapping the weight for its fake-quantized version only for
the duration of each wrapped ``forward`` call means: outside forwards the
module is untouched (FSDP, optimizers, checkpointing, and export see vanilla
parameters), and inside forwards the fake-quant operates on exactly the
unsharded compute-dtype weight the matmul would have used.

Targeting mirrors ``mlx_dit_from_diffusers_safetensors``: 2-D (or reshapable
conv) ``.weight`` tensors whose grouped dim divides ``group_size``, excluding
norms and modulation tables. The quantization *decisions* (codes, scales,
biases) bit-match deploy time; under bf16 compute the dequantized values are
rounded once more to bf16 for the matmul, like any bf16 arithmetic.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from fastvideo.layers.quantization.mlx_affine_qat import fake_quantize_mlx_affine
from fastvideo.layers.quantization.mlx_mxfp4_qat import (
    DEFAULT_GROUP_SIZE as MXFP4_GROUP_SIZE,
    fake_quantize_mlx_mxfp4,
)
from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)

# Weights the MLX loader never quantizes: norms and modulation tables.
DEFAULT_EXCLUDE_PATTERNS = (r"norm", r"scale_shift_table")

_SIMULATE_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

_WRAPPED_MARKER = "_mlx_qat_wrapped"


def _fake_quantize_weight(weight: torch.Tensor, *, mode: str, group_size: int, bits: int,
                          simulate_dtype: torch.dtype) -> torch.Tensor:
    original_shape = weight.shape
    # Conv-style weights (e.g. Wan's patch embedding) quantize over the
    # flattened non-output dims, matching the MLX loader's (out, -1) reshape.
    weight2d = weight.reshape(original_shape[0], -1) if weight.dim() > 2 else weight
    if mode == "affine":
        fq = fake_quantize_mlx_affine(weight2d, group_size=group_size, bits=bits, simulate_dtype=simulate_dtype)
    elif mode == "mxfp4":
        fq = fake_quantize_mlx_mxfp4(weight2d, simulate_dtype=simulate_dtype)
    else:
        raise ValueError(f"Unknown MLX QAT mode: {mode!r}")
    # Keep the dtype the module was about to compute with (inside FSDP
    # forwards that is the unsharded compute-dtype weight).
    return fq.reshape(original_shape).to(weight.dtype)


def _install_qat_forward(module: torch.nn.Module, *, mode: str, group_size: int, bits: int,
                         simulate_dtype: torch.dtype) -> None:
    inner_forward = module.forward  # bound method of this instance

    def qat_forward(*args, **kwargs):
        original = module._parameters.pop("weight")
        try:
            # Plain-attribute shadow: getattr finds it before _parameters.
            module.weight = _fake_quantize_weight(original,
                                                  mode=mode,
                                                  group_size=group_size,
                                                  bits=bits,
                                                  simulate_dtype=simulate_dtype)
            return inner_forward(*args, **kwargs)
        finally:
            if "weight" in module.__dict__:
                del module.weight
            module._parameters["weight"] = original

    module.forward = qat_forward
    setattr(module, _WRAPPED_MARKER, True)


class MLXQuantizationAwareCallback(Callback):
    """Apply MLX fake quantization to the student's weights.

    Args (all YAML-configurable):
        mode: ``"affine"`` for affine INT8 (default) or ``"mxfp4"`` for the
            fixed MXFP4 grid.
        group_size: quantization group size along the input dim for affine
            mode (MLX default 64). MXFP4 always uses 32 and ignores this arg.
        bits: affine bit width. MXFP4 ignores this arg.
        simulate_dtype: precision the deploy path casts weights to before
            quantizing ("fp16" matches the MLX loader).
        exclude_patterns: regex fragments; a weight is skipped when any
            matches its module name.
    """

    def __init__(
        self,
        *,
        mode: str = "affine",
        group_size: int = 64,
        bits: int = 8,
        simulate_dtype: str = "fp16",
        exclude_patterns: tuple[str, ...] | list[str] = DEFAULT_EXCLUDE_PATTERNS,
    ) -> None:
        self._mode = mode.strip().lower()
        if self._mode not in {"affine", "mxfp4"}:
            raise ValueError("mlx_qat.mode must be one of {'affine', 'mxfp4'}")
        self._group_size = MXFP4_GROUP_SIZE if self._mode == "mxfp4" else int(group_size)
        self._bits = int(bits)
        self._simulate_dtype = _SIMULATE_DTYPES[simulate_dtype]
        self._exclude = [re.compile(pattern) for pattern in exclude_patterns]
        self.quantized_module_names: list[str] = []

    def _is_target(self, module_name: str, module: torch.nn.Module) -> bool:
        if getattr(module, _WRAPPED_MARKER, False):
            return False
        weight = module._parameters.get("weight")
        if weight is None or weight.dim() < 2:
            return False
        if any(pattern.search(module_name) for pattern in self._exclude):
            return False
        grouped_dim = weight.shape[1:].numel()
        if grouped_dim % self._group_size != 0:
            logger.warning(
                "mlx_qat: skipping %s — grouped dim %d is not divisible by group_size %d "
                "(the MLX runtime could not quantize this weight either).", module_name, grouped_dim, self._group_size)
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
            _install_qat_forward(
                module,
                mode=self._mode,
                group_size=self._group_size,
                bits=self._bits,
                simulate_dtype=self._simulate_dtype,
            )
            self.quantized_module_names.append(module_name)

        if not self.quantized_module_names:
            raise ValueError("mlx_qat matched no weights on the student transformer — check exclude_patterns "
                             "and group_size against the model architecture.")
        logger.info(
            "mlx_qat: fake-quantizing %d weights (mode=%s, int%d, group_size=%d, simulate=%s), e.g. %s",
            len(self.quantized_module_names),
            self._mode,
            self._bits,
            self._group_size,
            self._simulate_dtype,
            self.quantized_module_names[:3],
        )
