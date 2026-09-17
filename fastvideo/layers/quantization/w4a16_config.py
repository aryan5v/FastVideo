# SPDX-License-Identifier: Apache-2.0
"""W4A16 — 4-bit weight, 16-bit activation quantization for CUDA DiT inference.

W4A16 means exactly what it says: the *weights* are stored as 4-bit integers
with a per-group scale (and zero point), and the *activations* stay in a
16-bit float (bf16/fp16). Nothing here quantizes activations, and there is no
fused W4A16 GEMM in this environment either (see "Kernel status" below). The
compute path is: dequantize the stored 4-bit codes back to the activation
dtype, then run an ordinary dense GEMM.

Why this lane exists
--------------------
W4A16 is a **primary deployment path for Ada-generation consumer GPUs**
(RTX 4090 24 GB, RTX 6000 Ada 48 GB). H3's 20B DiT does not fit those cards
in BF16, and Ada has no native NVFP4 tensor-core path — so the low-bit
options that actually apply there are INT8, FP8 and W4A16. W4A16 is the one
that shrinks the weight *storage* the most (5.0 bits/weight with
``group_size=64`` — 4 for the codes plus two fp32 per-group constants — versus
8 for INT8/FP8 and 16 for BF16), which is what
matters when the bottleneck is "does the checkpoint fit", not "how fast is
one GEMM".

Kernel status — read this before quoting a number
-------------------------------------------------
**There is no W4A16 GEMM kernel in this repository or in its installed
dependencies.** ``fastvideo-kernel`` ships an INT8 GEMM
(``csrc/turbodiffusion/gemm/gemm.cu`` is ``int8_gemm`` over
``cutlass::NumericConverter<int8_t, float>``), FP4 attention for sm_100/sm_120,
and block-sparse attention — no 4-bit weight GEMM on any architecture. The
``int4`` matches in ``csrc/turbodiffusion`` are the CUDA 16-byte vector type,
not 4-bit quantization. ``autoawq`` / ``auto_gptq`` / ``gptqmodel`` /
``marlin`` / ``bitsandbytes`` are not installed.

So what this module provides is a **correctness reference**: a faithful
quantize/dequantize pair and a layer method that runs it end-to-end. Its
steady-state weight storage really is 4-bit, but every forward pays a full
dequantize plus a dense 16-bit GEMM, so it is *slower* than BF16 and its
per-forward peak memory transiently includes one dense weight. It is the
schema and the wiring a real kernel would consume — not a speed path. Do not
benchmark it against BF16 and call the result "W4A16 on Ada".

Relationship to the other precision lanes
-----------------------------------------
- ``NVFP4`` (``nvfp4_config.py``) — Blackwell (sm_100+) block-scaled FP4 with
  a real FlashInfer GEMM. Not an Ada path.
- ``INT8Affine`` (``int8_affine_config.py``) — group-64 affine INT8, Ada-viable,
  same "dequantize then dense GEMM" reference shape.
- ``FP8`` / ``AbsMaxFP8`` — also Ada-viable (sm_89 supports FP8 tensor cores).
- **This module** — half the weight storage of INT8/FP8, unconditionally
  weight-only, no fused kernel.

Design notes carried over from the neighbouring configs
-------------------------------------------------------
- **Load-time conversion, dense allocation.** ``W4A16QuantizeMethod.create_weights``
  allocates the same dense Parameter an unquantized linear would, so a plain
  BF16 checkpoint loads unchanged; the 4-bit codes arrive afterwards from
  :func:`convert_model_to_w4a16` (or lazily on first forward). Same shape as
  ``NVFP4QuantizeMethod`` / ``INT8AffineQuantizeMethod``.
- **Layer selection is a constructor field, not a hardcoded model list.**
  ``target_layers`` is an explicit allowlist of full module paths;
  ``layer_suffixes`` is a generic suffix rule for models without an enumerable
  list. ``NVFP4Config``'s hardcoded LTX-2 frozenset is deliberately not
  repeated here.
- **A fail-closed deny list.** ``attn.to_gate_compress`` is H3's VSA
  sparse-attention compression gate. Quantizing it perturbs a *discrete*
  routing decision, so an error there is not a small output perturbation — it
  changes which tiles sparse attention attends to. The constructor can only
  ever *add* exclusions, never remove one, so no caller can quantize it.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from fastvideo.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from fastvideo.models.utils import set_weight_attrs

logger = logging.getLogger(__name__)

DEFAULT_GROUP_SIZE = 64
DEFAULT_BITS = 4
_EPS = 1e-8
_LAZY_CONVERSION_WARNED = False



def _group(w: torch.Tensor, group_size: int) -> torch.Tensor:
    """View the last dim as ``(num_groups, group_size)``.

    Grouping is along the last (input/contraction) axis, matching
    ``int8_affine_config._group`` and MLX's affine quantizer.
    """
    if w.shape[-1] % group_size != 0:
        raise ValueError(f"Last dim {w.shape[-1]} is not divisible by group_size {group_size}.")
    return w.reshape(*w.shape[:-1], w.shape[-1] // group_size, group_size)


def w4a16_quantize(
    w: torch.Tensor,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    bits: int = DEFAULT_BITS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group-wise affine quantization of ``w`` to ``bits``-bit unsigned codes.

    Per group of ``group_size`` contiguous values along the last axis, this
    solves ``w ~= (code - zero) * scale`` with a **min/max affine** fit:
    ``scale = (max - min) / (2**bits - 1)`` and ``zero`` the code that
    reproduces ``min`` exactly, so both endpoints of the group round-trip
    exactly. Codes are ``rint``-rounded (round-half-to-even) and clamped to
    ``[0, 2**bits - 1]``.

    The fit is done in fp32 regardless of ``w.dtype``, so a bf16 checkpoint
    value (which converts to fp32 exactly) yields a higher-precision scale
    store than solving in bf16 would.

    Returns ``(codes, scales, zeros)``:

    - ``codes`` — ``torch.uint8``, shape ``w.shape[:-1] + (K // 2,)``, **two
      4-bit codes packed per byte**. For ``bits=4`` only; ``bits=8`` returns
      one code per byte at shape ``w.shape``.
    - ``scales`` — ``torch.float32``, shape ``w.shape[:-1] + (K // group_size,)``.
    - ``zeros`` — ``torch.float32``, same shape as ``scales``.

    Packing convention (ours — no kernel consumes it yet): the **low nibble is
    the lower K index**, i.e. ``packed[..., j] = codes[..., 2j] | codes[..., 2j+1] << 4``.
    A future kernel has to be written against this layout; it does not match
    AWQ's interleaved layout, GPTQ's ``g_idx`` layout, or bitsandbytes' order.
    """
    if bits != 4 and bits != 8:
        raise ValueError(f"W4A16 stores codes as uint8; bits must be 4 or 8, got {bits}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    w32 = w.detach().float().nan_to_num()
    max_code = float((1 << bits) - 1)
    grouped = _group(w32, group_size)

    w_min = grouped.amin(dim=-1)
    w_max = grouped.amax(dim=-1)
    scales = (w_max - w_min).clamp_min(_EPS) / max_code
    zeros = torch.round(-w_min / scales).clamp_(0.0, max_code)

    codes = torch.round(grouped / scales.unsqueeze(-1) + zeros.unsqueeze(-1))
    codes = codes.clamp_(0.0, max_code).to(torch.uint8).reshape(w32.shape)

    if bits == 8:
        return codes, scales, zeros
    return _pack_4bit(codes), scales, zeros


def _pack_4bit(codes: torch.Tensor) -> torch.Tensor:
    """Pack ``[0, 15]`` codes two-per-byte along the last axis.

    Low nibble = lower K index. The last dim must be even, which every H3
    linear input dim is (5376, 7168, 14336, 2688 are all even).
    """
    if codes.shape[-1] % 2:
        raise ValueError(f"4-bit packing needs an even last dim, got {codes.shape[-1]}")
    codes = codes.to(torch.uint8).reshape(*codes.shape[:-1], codes.shape[-1] // 2, 2)
    low = codes[..., 0]
    high = codes[..., 1] & 0x0F
    return (low | (high << 4)).contiguous()


def _unpack_4bit(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_pack_4bit`; returns uint8 codes at ``2 * packed.shape[-1]``."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def w4a16_dequantize(
    codes: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    bits: int = DEFAULT_BITS,
    out_shape: tuple[int, ...] | torch.Size | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Reconstruct the dense weight from 4-bit codes, group scales and zeros.

    ``codes`` is the packed tensor :func:`w4a16_quantize` returned (or, for
    ``bits=8``, the unpacked one). Pass ``out_shape`` to name the logical
    weight shape; otherwise the reconstruction keeps the code layout's own
    last dim (``2 * packed.shape[-1]`` for 4-bit).

    The arithmetic runs in fp32 and is cast to ``out_dtype`` at the end —
    ``(code - zero) * scale`` in fp32 is the more accurate side of the split,
    same choice ``int8_affine_config`` makes.
    """
    if bits == 4:
        codes = _unpack_4bit(codes)
    if out_shape is not None and tuple(codes.shape) != tuple(out_shape):
        codes = codes.reshape(out_shape)
    grouped = _group(codes.float(), group_size)
    dense = (grouped - zeros.unsqueeze(-1)) * scales.unsqueeze(-1)
    dense = dense.reshape(codes.shape)
    return dense if out_dtype is None else dense.to(out_dtype)


MINIMAX_H3_PREFIX = "minimax_h3"
MINIMAX_H3_NUM_LAYERS = 50
MINIMAX_H3_NUM_REFINER_LAYERS = 2
MINIMAX_H3_BLOCK_SCOPES: tuple[str, ...] = (
    "transformer_blocks",
    "token_refiner.refiner_blocks",
)
MINIMAX_H3_BLOCK_LINEAR_SUFFIXES: tuple[str, ...] = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out",
    "ff.fc_in",
    "ff.fc_out",
)
MINIMAX_H3_MAIN_STACK_LINEAR_SUFFIXES: tuple[str, ...] = MINIMAX_H3_BLOCK_LINEAR_SUFFIXES + ("adaln_proj.linear", )

_GENERIC_LINEAR_SUFFIXES: tuple[str, ...] = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out",
    "ff.fc_in",
    "ff.fc_out",
)

_NEVER_QUANTIZE_SUBSTRINGS: tuple[str, ...] = ("attn.to_gate_compress", )

_H3_FP32_KEPT_SUBSTRINGS: tuple[str, ...] = (
    "proj_in",
    "audio_proj_in",
    "proj_out",
    "audio_proj_out",
    "time_embedder",
)


def _matches_linear_suffix(prefix: str, suffixes: Iterable[str]) -> bool:
    """True when *prefix* is one of *suffixes* or ends at a dot boundary.

    The dot boundary keeps ``"ff.fc_in"`` from matching a hypothetical
    ``"cross_ff.fc_in"``.
    """
    return any(prefix == suffix or prefix.endswith("." + suffix) for suffix in suffixes)


def minimax_h3_w4a16_prefixes(
    *,
    prefix: str = MINIMAX_H3_PREFIX,
    num_layers: int = MINIMAX_H3_NUM_LAYERS,
    num_refiner_layers: int = MINIMAX_H3_NUM_REFINER_LAYERS,
) -> frozenset[str]:
    """Enumerate the exact H3 linear prefixes the H3 W4A16 profile targets.

    Built from H3's real module names as constructed in
    ``fastvideo/models/dits/minimax_h3.py``: ``MiniMaxH3TransformerBlock``
    builds ``{prefix}.transformer_blocks.{i}.attn`` / ``.ff`` / ``.adaln_proj``;
    ``MiniMaxH3TokenRefiner`` builds ``{prefix}.token_refiner.refiner_blocks.{i}.attn`` / ``.ff``.

    50 main blocks x 7 linears + 2 refiner blocks x 6 linears = **362 linears**.

    This is the allowlist :meth:`W4A16Config.for_minimax_h3` hands to
    ``target_layers``. It is a plain function so a caller (or a test) can
    regenerate it from the architecture constants rather than trusting a
    literal — the H3 profile is *derived*, not hardcoded.
    """
    prefixes: set[str] = set()
    for index in range(num_layers):
        for suffix in MINIMAX_H3_MAIN_STACK_LINEAR_SUFFIXES:
            prefixes.add(f"{prefix}.transformer_blocks.{index}.{suffix}")
    for index in range(num_refiner_layers):
        for suffix in MINIMAX_H3_BLOCK_LINEAR_SUFFIXES:
            prefixes.add(f"{prefix}.token_refiner.refiner_blocks.{index}.{suffix}")
    return frozenset(prefixes)


class W4A16Config(QuantizationConfig):
    """Weight-only 4-bit (group-wise affine) quantization with 16-bit activations.

    Layer selection is a constructor field, not a hardcoded model list:
    ``target_layers`` is an explicit allowlist of full module paths and takes
    precedence when given; otherwise ``layer_suffixes`` is matched with a
    dot-boundary suffix rule. Both are subject to the fail-closed deny list
    (``_NEVER_QUANTIZE_SUBSTRINGS``), which the constructor can only widen.

    Weight-only by construction: there is no activation quantizer and
    :class:`W4A16QuantizeMethod` runs a dense 16-bit GEMM over a dequantized
    weight. Use :meth:`for_minimax_h3` for the H3 profile.
    """

    def __init__(
        self,
        group_size: int = DEFAULT_GROUP_SIZE,
        bits: int = DEFAULT_BITS,
        target_layers: Iterable[str] | None = None,
        layer_suffixes: Iterable[str] | None = None,
        exclude_substrings: Iterable[str] | None = None,
        retain_original_weight: bool = True,
    ) -> None:
        super().__init__()
        if bits not in (4, 8):
            raise ValueError(f"W4A16 stores codes as uint8; bits must be 4 or 8, got {bits}")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        self.group_size = group_size
        self.bits = bits
        self.target_layers: frozenset[str] | None = (None if target_layers is None else frozenset(target_layers))
        self.layer_suffixes: tuple[str, ...] = (tuple(_GENERIC_LINEAR_SUFFIXES)
                                                if layer_suffixes is None else tuple(layer_suffixes))
        self.exclude_substrings: tuple[str, ...] = tuple(_NEVER_QUANTIZE_SUBSTRINGS) + tuple(exclude_substrings or ())
        self.retain_original_weight = retain_original_weight

    def get_name(self) -> str:
        return "W4A16"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        """Turing (75).

        The compute path is a plain 16-bit GEMM over a dequantized weight, so
        no 4-bit tensor-core class is required and the config stays loadable
        wherever the other reference paths are. **This is not a claim that
        4-bit runs fast there.** The deployment target for this lane is Ada
        (sm_89, RTX 4090 / RTX 6000 Ada); making it a *fast* path needs a
        fused W4A16 GEMM that does not exist in this repository yet.
        """
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> W4A16Config:
        return cls(
            group_size=config.get("group_size", DEFAULT_GROUP_SIZE),
            bits=config.get("bits", DEFAULT_BITS),
            target_layers=config.get("target_layers"),
            layer_suffixes=config.get("layer_suffixes"),
            exclude_substrings=config.get("exclude_substrings"),
            retain_original_weight=config.get("retain_original_weight", True),
        )

    @classmethod
    def for_minimax_h3(
        cls,
        *,
        group_size: int = DEFAULT_GROUP_SIZE,
        bits: int = DEFAULT_BITS,
        retain_original_weight: bool = True,
    ) -> W4A16Config:
        """The MiniMax-H3 profile: 362 attention / FFN / AdaLN linears.

        The allowlist comes from :func:`minimax_h3_w4a16_prefixes`, and the
        deny list additionally carries H3's fp32-pinned modules (``proj_in``,
        ``audio_proj_in``, ``time_embedder``, ``proj_out``, ``audio_proj_out``)
        — those are excluded both by construction (they are not in the
        allowlist) and by name, so a later widening of the allowlist cannot
        silently reach them.

        ``group_size`` must divide the input dim of every targeted linear:
        with the released H3 config (hidden 5376, inner 7168, ffn 14336,
        adaln 2688) that holds for 32, 64 and 128.
        """
        return cls(
            group_size=group_size,
            bits=bits,
            target_layers=minimax_h3_w4a16_prefixes(),
            exclude_substrings=_H3_FP32_KEPT_SUBSTRINGS,
            retain_original_weight=retain_original_weight,
        )

    def is_target_layer(self, prefix: str) -> bool:
        """Whether ``prefix`` is quantized under this config.

        Deny list first (fail-closed), then ``target_layers`` if supplied,
        else suffix matching. Non-``LinearBase`` layers are filtered by
        :meth:`get_quant_method`, not here, so this is safe to call on any
        module name.
        """
        for banned in self.exclude_substrings:
            if banned in prefix:
                return False
        if self.target_layers is not None:
            return prefix in self.target_layers
        return _matches_linear_suffix(prefix, self.layer_suffixes)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from fastvideo.layers.linear import LinearBase

        if not isinstance(layer, LinearBase) or not self.is_target_layer(prefix):
            return None
        input_size = getattr(layer, "input_size", None)
        if input_size is not None and input_size % self.group_size:
            logger.warning(
                "W4A16: skipping layer %r — input dim %d is not divisible by group_size %d. "
                "The layer runs dense.", prefix, input_size, self.group_size)
            return None
        if self.bits == 4 and input_size is not None and input_size % 2:
            logger.warning(
                "W4A16: skipping layer %r — input dim %d is odd, so 4-bit codes cannot be "
                "packed two-per-byte. The layer runs dense.", prefix, input_size)
            return None
        return W4A16QuantizeMethod(
            layer_prefix=prefix,
            group_size=self.group_size,
            bits=self.bits,
            retain_original_weight=self.retain_original_weight,
        )


class W4A16QuantizeMethod(QuantizeMethodBase):
    """Linear method for weight-only 4-bit affine quantization.

    ``create_weights`` allocates the same dense Parameter an unquantized linear
    would (so a BF16 checkpoint loads unchanged), and the 4-bit codes, group
    scales and zeros arrive later as non-persistent buffers from
    :func:`convert_model_to_w4a16` — conversion happens at *load* time, not
    construction time, exactly mirroring ``NVFP4QuantizeMethod`` and
    ``INT8AffineQuantizeMethod``.

    ``apply`` is the **reference path**: dequantize the whole weight to the
    activation dtype, then ``F.linear``. It is bit-for-bit a dense 16-bit GEMM
    over an approximation of the original weight, which is what makes it
    useful as a correctness oracle — and it is why it is not a performance
    path (see the module docstring's "Kernel status").
    """

    def __init__(
        self,
        layer_prefix: str = "",
        group_size: int = DEFAULT_GROUP_SIZE,
        bits: int = DEFAULT_BITS,
        retain_original_weight: bool = True,
    ) -> None:
        super().__init__()
        self.layer_prefix = layer_prefix
        self.group_size = group_size
        self.bits = bits
        self.retain_original_weight = retain_original_weight

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def _ensure_quantized(self, layer: torch.nn.Module) -> bool:
        """Convert on first use if the loader hook never ran.

        Returns False when the layer is intentionally left dense (grad-enabled
        forward: a training step must see the master weight, not a frozen
        dequantized copy). The loader path is ``_maybe_quantize_model`` ->
        :func:`convert_model_to_w4a16`; this fallback exists so the config is
        still *correct* if that dispatch is missing, but it warns because
        reaching it means the loader hook did not fire.
        """
        if getattr(layer, "_w4a16_codes", None) is not None:
            return True
        weight = getattr(layer, "weight", None)
        if weight is None:
            raise RuntimeError(f"W4A16 layer {self.layer_prefix!r} has no weight and no quantized buffers.")
        if torch.is_grad_enabled():
            return False
        global _LAZY_CONVERSION_WARNED
        if not _LAZY_CONVERSION_WARNED:
            _LAZY_CONVERSION_WARNED = True
            logger.warning(
                "W4A16: layer %r reached apply() unconverted; converting lazily (this message is logged once "
                "per process, not once per layer). The loader hook (_maybe_quantize_model) did not dispatch to "
                "convert_model_to_w4a16 — check its isinstance chain in "
                "fastvideo/models/loader/fsdp_load.py.", self.layer_prefix)
        _quantize_layer_weight(layer, weight, group_size=self.group_size, bits=self.bits)
        return True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._ensure_quantized(layer):
            weight = layer.weight
            if weight is None:
                raise RuntimeError(
                    f"W4A16 layer {self.layer_prefix!r} is in the dense (grad-enabled) branch, but its "
                    "original bf16 weight was purged (W4A16Config(retain_original_weight=False)). Training "
                    "needs the master weight; load with retain_original_weight left at its default for any "
                    "run that takes gradients.")
            return F.linear(x, weight.to(x.dtype) if weight.dtype != x.dtype else weight, bias)

        weight = w4a16_dequantize(
            layer._w4a16_codes,
            layer._w4a16_scales,
            layer._w4a16_zeros,
            group_size=self.group_size,
            bits=self.bits,
            out_shape=layer._w4a16_weight_shape,
            out_dtype=x.dtype,
        )
        return F.linear(x, weight, bias)




def _quantize_layer_weight(
    mod: torch.nn.Module,
    weight: torch.Tensor,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    bits: int = DEFAULT_BITS,
) -> None:
    """Quantize one linear's weight in place into non-persistent buffers."""
    from torch.distributed.tensor import DTensor  # type: ignore

    weight_local = weight.to_local() if isinstance(weight, DTensor) else weight  # type: ignore[arg-type]
    if weight_local.shape[-1] % group_size:
        raise ValueError(f"W4A16 layer {mod!r}: input dim {weight_local.shape[-1]} is not divisible by "
                         f"group_size {group_size}.")
    codes, scales, zeros = w4a16_quantize(weight_local, group_size=group_size, bits=bits)
    mod.register_buffer("_w4a16_codes", codes.contiguous(), persistent=False)
    mod.register_buffer("_w4a16_scales", scales.to(torch.float32).contiguous(), persistent=False)
    mod.register_buffer("_w4a16_zeros", zeros.to(torch.float32).contiguous(), persistent=False)
    mod._w4a16_weight_shape = tuple(weight_local.shape)


def convert_model_to_w4a16(model: torch.nn.Module) -> None:
    """Quantize every W4A16-tagged linear in-place after weights load.

    Mirrors ``convert_model_to_nvfp4`` / ``convert_model_to_int8_affine``: walk
    the module tree once, convert each layer whose ``quant_method`` is a
    :class:`W4A16QuantizeMethod`, and register the 4-bit codes plus per-group
    scales/zeros as non-persistent buffers (so they are not written back into
    ``state_dict``/checkpoints).

    Callers: the loader hook ``_maybe_quantize_model`` in
    ``fastvideo/models/loader/fsdp_load.py``. *That hook is not edited by this
    module* — it dispatches on an explicit ``isinstance`` chain, so it needs a
    matching branch (see the module report). Without it,
    :meth:`W4A16QuantizeMethod.apply` converts lazily on first forward and logs
    a warning, so inference is still correct, just later and noisier.
    """
    converted = 0
    purged = 0
    schemes: set[tuple[int, int]] = set()
    for mod in model.modules():
        qm = getattr(mod, "quant_method", None)
        if not isinstance(qm, W4A16QuantizeMethod):
            continue
        weight = getattr(mod, "weight", None)
        if weight is None:
            continue
        _quantize_layer_weight(mod, weight, group_size=qm.group_size, bits=qm.bits)
        converted += 1
        schemes.add((qm.group_size, qm.bits))
        if not qm.retain_original_weight:
            original = mod._parameters.get("weight")
            if original is not None:
                original.grad = None
            mod.register_parameter("weight", None)
            purged += 1

    if converted:
        logger.info(
            "W4A16 conversion receipt: quantized %d linear layers (%s, reference dequantize-then-GEMM "
            "path); purged %d original bf16 weight tensors.", converted,
            ", ".join(f"group_size={g}, bits={b}" for g, b in sorted(schemes)), purged)
        logger.info("W4A16: no fused 4-bit GEMM is available in this build, so forward compute is a dense "
                    "16-bit GEMM over a dequantized weight. Expect BF16-comparable VRAM for the transient "
                    "dense weight and slower-than-BF16 step times.")



W4A16_SIDECAR_SUFFIX = ".w4a16.safetensors"
W4A16_DIR_SIDECAR_NAME = "w4a16.safetensors"
_W4A16_SIDECAR_FORMAT = "fastvideo.w4a16"
_W4A16_SIDECAR_VERSION = 1
_W4A16_SIDECAR_METADATA_KEY = "fastvideo_w4a16"
_W4A16_SIDECAR_KEY_SEP = "::"
_W4A16_SIDECAR_BUFFERS = (
    "_w4a16_codes",
    "_w4a16_scales",
    "_w4a16_zeros",
)
_W4A16_SIDECAR_DTYPES = {
    "_w4a16_codes": torch.uint8,
    "_w4a16_scales": torch.float32,
    "_w4a16_zeros": torch.float32,
}


def _sidecar_key(module_fqn: str, buffer_name: str) -> str:
    return f"{module_fqn}{_W4A16_SIDECAR_KEY_SEP}{buffer_name}"


def _w4a16_tagged_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module, W4A16QuantizeMethod]]:
    tagged = []
    for fqn, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        if isinstance(qm, W4A16QuantizeMethod):
            tagged.append((fqn, mod, qm))
    return tagged


def _is_dtensor(tensor: torch.Tensor) -> bool:
    try:
        from torch.distributed.tensor import DTensor  # type: ignore
    except ImportError:  # pragma: no cover - depends on the torch build
        return False
    return isinstance(tensor, DTensor)


def w4a16_sidecar_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Collect the quantized tensors of every W4A16 linear in *model*.

    Keys are ``"<module fqn>::<buffer name>"`` and values are detached CPU
    copies. Modules whose buffers are missing (never converted) are skipped;
    the returned mapping is what :func:`save_w4a16_checkpoint` writes.

    ``_w4a16_weight_shape`` is not a tensor and so is not collected here; it
    travels in the manifest instead (see :func:`save_w4a16_checkpoint`).

    FSDP note: a DTensor buffer is saved as this rank's local shard, so a
    sharded save is only reloadable into an identically sharded model.
    """
    state: dict[str, torch.Tensor] = {}
    for fqn, mod, _ in _w4a16_tagged_modules(model):
        for name in _W4A16_SIDECAR_BUFFERS:
            tensor = getattr(mod, name, None)
            if tensor is None:
                continue
            if _is_dtensor(tensor):
                tensor = tensor.to_local()  # type: ignore[attr-defined]
            state[_sidecar_key(fqn, name)] = tensor.detach().to("cpu", copy=True).contiguous()
    return state


def save_w4a16_checkpoint(
    model: torch.nn.Module,
    path: str | os.PathLike[str],
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the model's W4A16 tensors to a compact sidecar safetensors file.

    The file is roughly the 4-bit size (5.0 bits/weight at ``group_size=64``:
    4 for the packed codes plus two fp32 per group) instead of the dense bf16
    size. Returns a receipt dict (also logged) with the module count and both
    sizes. Raises ``RuntimeError`` when the model has no W4A16 linears, which
    usually means the config's layer selection did not cover the model's layer
    paths, and when the tagged layers carry no buffers (never converted).

    The manifest under the ``fastvideo_w4a16`` metadata key carries the format
    name/version, the scheme (``group_size``/``bits``), the per-layer weight and
    buffer shapes, and the quantized module fqns (the keys of ``layers``), so a
    loader can validate a sidecar against a model without materializing the
    tensors. The per-layer ``weight_shape`` is what a load uses to restore
    ``_w4a16_weight_shape``, which the packed codes cannot express.
    """
    from safetensors.torch import save_file

    state = w4a16_sidecar_state_dict(model)
    tagged = _w4a16_tagged_modules(model)
    if not tagged:
        raise RuntimeError("No W4A16 linear layers found in this model; nothing to serialize. Check that the "
                           "model was built with a W4A16Config whose layer selection covers its layer paths "
                           "(e.g. W4A16Config.for_minimax_h3()).")
    if not state:
        raise RuntimeError(f"Found {len(tagged)} W4A16-tagged linear layers but none carry quantized buffers. "
                           "Call convert_model_to_w4a16(model) before saving a sidecar.")

    layers: dict[str, dict[str, Any]] = {}
    quant_prefixes: dict[str, str] = {}
    dense_bytes = 0
    group_sizes: set[int] = set()
    bit_widths: set[int] = set()
    for fqn, mod, qm in tagged:
        codes = getattr(mod, "_w4a16_codes", None)
        weight = getattr(mod, "weight", None)
        if codes is None and weight is None:
            continue
        if weight is not None:
            weight_shape = [int(dim) for dim in weight.shape]
        elif getattr(mod, "_w4a16_weight_shape", None) is not None:
            weight_shape = [int(dim) for dim in mod._w4a16_weight_shape]
        else:
            raise RuntimeError(f"W4A16 layer {fqn!r} has no dense weight and no recorded "
                               "_w4a16_weight_shape; its logical weight shape cannot be recovered from the "
                               "packed codes. Re-run convert_model_to_w4a16(model) before saving.")
        tensors = {
            name: [int(dim) for dim in getattr(mod, name).shape]
            for name in _W4A16_SIDECAR_BUFFERS if getattr(mod, name, None) is not None
        }
        layers[fqn] = {
            "weight_shape": weight_shape,
            "group_size": int(qm.group_size),
            "bits": int(qm.bits),
            "tensors": tensors,
        }
        quant_prefixes[fqn] = getattr(qm, "layer_prefix", "") or ""
        dense_bytes += weight_shape[0] * weight_shape[1] * 2
        group_sizes.add(int(qm.group_size))
        bit_widths.add(int(qm.bits))

    metadata: dict[str, Any] = {
        "format": _W4A16_SIDECAR_FORMAT,
        "version": _W4A16_SIDECAR_VERSION,
        "group_size": group_sizes.pop() if len(group_sizes) == 1 else None,
        "bits": bit_widths.pop() if len(bit_widths) == 1 else None,
        "num_layers": len(layers),
        "layers": layers,
        "quant_prefixes": quant_prefixes,
        "model_class": type(model).__name__,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    payload = dict(state)
    serialized_bytes = sum(t.numel() * t.element_size() for t in payload.values())
    save_file(payload, os.fspath(path), metadata={_W4A16_SIDECAR_METADATA_KEY: json.dumps(metadata)})

    receipt = {
        "path": os.fspath(path),
        "num_layers": len(layers),
        "num_tensors": len(payload),
        "quantized_bytes": serialized_bytes,
        "dense_bfloat16_bytes": dense_bytes,
        "compression_ratio": (dense_bytes / serialized_bytes) if serialized_bytes else 0.0,
    }
    logger.info(
        "W4A16 sidecar: wrote %d quantized modules / %d tensors (%d bytes) to %s "
        "(%.2f GiB quantized vs %.2f GiB dense bf16, %.2fx smaller).",
        receipt["num_layers"],
        receipt["num_tensors"],
        serialized_bytes,
        receipt["path"],
        serialized_bytes / (1 << 30),
        dense_bytes / (1 << 30),
        receipt["compression_ratio"],
    )
    return receipt


def read_w4a16_sidecar_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return the manifest of a sidecar file without materializing its tensors."""
    from safetensors import safe_open

    with safe_open(os.fspath(path), framework="pt", device="cpu") as handle:
        raw = handle.metadata() or {}
    if _W4A16_SIDECAR_METADATA_KEY not in raw:
        raise ValueError(f"{os.fspath(path)} is not a FastVideo W4A16 sidecar "
                         f"(no {_W4A16_SIDECAR_METADATA_KEY!r} metadata).")
    return json.loads(raw[_W4A16_SIDECAR_METADATA_KEY])


def w4a16_sidecar_path_for(checkpoint_path: str | os.PathLike[str]) -> str:
    """Conventional sidecar path for a transformer checkpoint or directory.

    ``.../transformer.safetensors`` -> ``.../transformer.w4a16.safetensors``;
    a directory -> ``<dir>/w4a16.safetensors``.
    """
    raw = os.fspath(checkpoint_path)
    if os.path.isdir(raw):
        return os.path.join(raw, W4A16_DIR_SIDECAR_NAME)
    if raw.endswith(".safetensors"):
        return raw[:-len(".safetensors")] + W4A16_SIDECAR_SUFFIX
    return raw + W4A16_SIDECAR_SUFFIX


def _sidecar_target_device(mod: torch.nn.Module, name: str) -> torch.device | None:
    """Device the restored buffer should live on.

    Mirrors ``_quantize_layer_weight``, which registers the buffers on the
    (local) weight's device; falls back to an existing buffer, then to the
    module's parameter device so a purge-then-restore still lands on GPU.
    """
    weight = getattr(mod, "weight", None)
    if weight is not None and weight.device.type != "meta":
        return weight.device
    existing = getattr(mod, name, None)
    if existing is not None and existing.device.type != "meta":
        return existing.device
    for param in mod.parameters(recurse=False):
        if param.device.type != "meta":
            return param.device
    return None


def _expected_sidecar_shapes(name: str, weight_shape: tuple[int, int], group_size: int,
                             bits: int) -> tuple[tuple[int, ...], ...]:
    """The one shape a fresh conversion would produce for *name*.

    Unlike the NVFP4 sidecar there is no padded variant to accept: the
    quantizer groups along the last axis and ``_group`` refuses a K that is
    not divisible by ``group_size``, and the packed code interpretation is
    fixed by ``bits``. A sidecar that disagrees would not merely be unusual —
    reading it back would unpack the wrong nibble order or the wrong number of
    groups, which is silent corruption rather than an error.
    """
    out_dim, in_dim = weight_shape
    if name == "_w4a16_codes":
        if bits == 8:
            return ((out_dim, in_dim), )
        if in_dim % 2:
            raise ValueError(f"Sidecar declares a ({out_dim}, {in_dim}) weight at bits={bits}, but 4-bit codes "
                             "pack two per byte and need an even input dim.")
        return ((out_dim, in_dim // 2), )
    if in_dim % group_size:
        raise ValueError(f"Sidecar declares a ({out_dim}, {in_dim}) weight with group_size {group_size}, "
                         "which does not divide the input dim; this layout cannot be dequantized.")
    return ((out_dim, in_dim // group_size), )


def load_w4a16_checkpoint(
    model: torch.nn.Module,
    path: str | os.PathLike[str],
    *,
    strict: bool = True,
) -> int:
    """Restore W4A16 tensors from a sidecar, skipping ``convert_model_to_w4a16``.

    Registers ``_w4a16_codes`` / ``_w4a16_scales`` / ``_w4a16_zeros`` on every
    W4A16-tagged linear from the sidecar, byte-for-byte as the conversion would
    have produced them, and restores the ``_w4a16_weight_shape`` attribute that
    :meth:`W4A16QuantizeMethod.apply` needs (it is not a buffer, so nothing else
    would). The dense bf16 weights are never touched (they may be absent
    entirely), and nothing here needs a GPU or a 4-bit kernel — the
    dequantize-then-GEMM reference path in ``apply`` is pure PyTorch, so a
    pre-quantized checkpoint loads on any host.

    ``strict`` raises on any layer-set mismatch (a sidecar that does not
    describe this model); with ``strict=False`` those are logged and skipped,
    leaving those layers unconverted. Scheme (``group_size``/``bits``) and
    per-tensor shape/dtype mismatches are **never** downgraded: mis-read or
    mis-unpacked codes produce garbage output with no error, so those always
    raise.

    Returns the number of layers restored.
    """
    from safetensors import safe_open

    tagged = _w4a16_tagged_modules(model)
    if not tagged:
        raise RuntimeError("No W4A16 linear layers are attached to this model, so a sidecar cannot be restored. "
                           "This is the silent-dense failure mode: the model's W4A16Config layer selection "
                           "does not cover its layer paths (for MiniMax-H3 use "
                           "W4A16Config.for_minimax_h3()).")

    manifest = read_w4a16_sidecar_metadata(path)
    if manifest.get("format") != _W4A16_SIDECAR_FORMAT:
        raise ValueError(f"Unsupported W4A16 sidecar format {manifest.get('format')!r} in {os.fspath(path)}.")
    if int(manifest.get("version", -1)) != _W4A16_SIDECAR_VERSION:
        raise ValueError(f"Unsupported W4A16 sidecar version {manifest.get('version')!r} in {os.fspath(path)} "
                         f"(this build reads version {_W4A16_SIDECAR_VERSION}).")

    saved_layers: dict[str, dict[str, Any]] = manifest.get("layers", {})
    model_fqns = {fqn for fqn, _, _ in tagged}
    missing = sorted(model_fqns - set(saved_layers))
    extra = sorted(set(saved_layers) - model_fqns)
    if missing or extra:
        message = (f"W4A16 sidecar {os.fspath(path)} does not match this model: {len(missing)} layers missing "
                   f"from the sidecar, {len(extra)} layers not in the model. First missing={missing[:3]}, "
                   f"first extra={extra[:3]}.")
        if strict:
            raise ValueError(message)
        logger.warning("%s Restoring the intersection only.", message)

    restored = 0
    with safe_open(os.fspath(path), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        for fqn, mod, qm in tagged:
            if fqn not in saved_layers:
                continue
            entry = saved_layers[fqn]
            try:
                weight_shape = tuple(int(dim) for dim in entry["weight_shape"])
                group_size = int(entry["group_size"])
                bits = int(entry["bits"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"W4A16 sidecar {os.fspath(path)} entry for {fqn!r} is malformed: {entry!r} "
                                 "does not carry an integer weight_shape/group_size/bits.") from exc
            if len(weight_shape) != 2:
                raise ValueError(f"W4A16 sidecar entry {fqn!r} declares weight shape {list(weight_shape)}; "
                                 "a linear weight is 2-D.")
            if group_size != qm.group_size or bits != qm.bits:
                raise ValueError(f"W4A16 sidecar {os.fspath(path)} was written for {fqn!r} with "
                                 f"group_size={group_size}, bits={bits}, but this model quantizes it with "
                                 f"group_size={qm.group_size}, bits={qm.bits}.")
            weight = getattr(mod, "weight", None)
            if weight is not None and tuple(int(dim) for dim in weight.shape) != weight_shape:
                raise ValueError(f"W4A16 sidecar entry {fqn!r} describes a {list(weight_shape)} weight, but "
                                 f"this model's layer has shape {list(weight.shape)}.")
            tensors: dict[str, torch.Tensor] = {}
            for name in _W4A16_SIDECAR_BUFFERS:
                key = _sidecar_key(fqn, name)
                if key not in available:
                    continue
                tensor = handle.get_tensor(key)
                expected_dtype = _W4A16_SIDECAR_DTYPES[name]
                if tensor.dtype != expected_dtype:
                    raise ValueError(f"W4A16 sidecar tensor {key} has dtype {tensor.dtype}, expected "
                                     f"{expected_dtype}. Codes are uint8 bit patterns; a cast would silently "
                                     "unpack to different 4-bit values.")
                expected = _expected_sidecar_shapes(name, weight_shape, group_size, bits)
                if tuple(tensor.shape) not in expected:
                    raise ValueError(f"W4A16 sidecar tensor {key} has shape {tuple(tensor.shape)}, expected "
                                     f"one of {list(expected)} for a {list(weight_shape)} linear with "
                                     f"group_size={group_size}, bits={bits}.")
                device = _sidecar_target_device(mod, name)
                if device is not None:
                    tensor = tensor.to(device=device, non_blocking=True)
                tensors[name] = tensor
            if set(tensors) != set(_W4A16_SIDECAR_BUFFERS):
                message = (f"W4A16 sidecar entry for {fqn!r} is incomplete (has {sorted(tensors)}); "
                           f"all of {list(_W4A16_SIDECAR_BUFFERS)} are required.")
                if strict:
                    raise ValueError(message)
                logger.warning("%s Skipping this layer.", message)
                continue
            for name, tensor in tensors.items():
                mod.register_buffer(name, tensor, persistent=False)
            mod._w4a16_weight_shape = weight_shape
            restored += 1

    logger.info("W4A16 sidecar: restored %d quantized modules from %s (dense weights untouched).", restored,
                os.fspath(path))
    return restored


__all__ = [
    "DEFAULT_BITS",
    "DEFAULT_GROUP_SIZE",
    "MINIMAX_H3_BLOCK_LINEAR_SUFFIXES",
    "MINIMAX_H3_BLOCK_SCOPES",
    "MINIMAX_H3_NUM_LAYERS",
    "MINIMAX_H3_NUM_REFINER_LAYERS",
    "MINIMAX_H3_PREFIX",
    "W4A16Config",
    "W4A16QuantizeMethod",
    "W4A16_DIR_SIDECAR_NAME",
    "W4A16_SIDECAR_SUFFIX",
    "convert_model_to_w4a16",
    "load_w4a16_checkpoint",
    "minimax_h3_w4a16_prefixes",
    "read_w4a16_sidecar_metadata",
    "save_w4a16_checkpoint",
    "w4a16_dequantize",
    "w4a16_quantize",
    "w4a16_sidecar_path_for",
    "w4a16_sidecar_state_dict",
]
