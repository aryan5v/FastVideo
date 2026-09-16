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

import logging
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
# Affine codes span [0, 2**bits - 1]; for bits=4 that is [0, 15], stored as
# torch.uint8 (two codes per byte, see ``w4a16_quantize``).
# Guard against a degenerate group (all-equal values) producing a zero scale
# and a division by zero in the zero-point solve.
_EPS = 1e-8
# A model with 362 targeted linears would otherwise emit 362 copies of the
# "loader hook did not fire" warning on its first forward.
_LAZY_CONVERSION_WARNED = False

# ---------------------------------------------------------------------------
# Group-wise affine 4-bit quantizer
# ---------------------------------------------------------------------------


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


# --- MiniMax-H3 layer set -------------------------------------------------
#
# H3's DiT (``fastvideo/models/dits/minimax_h3.py``) names its linears
# ``{prefix}.{scope}.{i}.{suffix}`` with ``prefix="minimax_h3"``,
# ``num_layers=50``, ``num_refiner_layers=2``
# (``fastvideo/configs/models/dits/minimax_h3.py``). Both block stacks hold the
# same ``MiniMaxH3Attention`` / ``MiniMaxH3FeedForward``; only the main stack
# has ``adaln_proj``.
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
# The main stack additionally carries the per-block AdaLN modulation GEMM.
MINIMAX_H3_MAIN_STACK_LINEAR_SUFFIXES: tuple[str, ...] = MINIMAX_H3_BLOCK_LINEAR_SUFFIXES + ("adaln_proj.linear", )

# Suffix rules for a generic (non-H3) transformer, so the config is usable
# before a model has an enumerable prefix list.
_GENERIC_LINEAR_SUFFIXES: tuple[str, ...] = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out",
    "ff.fc_in",
    "ff.fc_out",
)

# Linears that must NEVER be quantized, whatever a caller passes in
# ``target_layers``. ``attn.to_gate_compress`` is H3's VSA sparse-attention
# compression gate: its output steers a *discrete* tile-selection decision, so
# quantizing it does not merely perturb the output, it can change which tiles
# the sparse attention reads. H3 also probes the loaded gate structurally
# (``MiniMaxH3Attention._gate_active`` tests ``weight != 0`` once to skip a
# guaranteed-zero branch), which a dequantized weight would break. It matches
# no generic "norm"/"embedder" exclusion heuristic, so it is named explicitly.
_NEVER_QUANTIZE_SUBSTRINGS: tuple[str, ...] = ("attn.to_gate_compress", )

# Modules H3 already pins to fp32 (``MiniMaxH3Transformer3DModel._keep_in_fp32_modules``).
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
        # Fail-closed deny list. The hard exclusions are always present;
        # ``exclude_substrings`` can only add, never remove -- this is what
        # keeps ``attn.to_gate_compress`` unquantizable no matter what a
        # caller passes as ``target_layers``.
        self.exclude_substrings: tuple[str, ...] = tuple(_NEVER_QUANTIZE_SUBSTRINGS) + tuple(exclude_substrings or ())
        # Keep the dense bf16 ``layer.weight`` Parameter after conversion.
        # Default True and it matters here: ``MiniMaxH3AdaLayerNormModulation.forward``
        # reads ``self.linear.weight.dtype`` to cast its input, so purging the
        # weight raises AttributeError. Setting False frees the bf16 copy at
        # the cost of requiring every caller to stop touching ``.weight``.
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
        # A group must sit entirely inside one weight row. Skipping (rather
        # than raising) keeps a config change from hard-failing model
        # construction; the warning is what makes the skip visible.
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

        # Reference path: one dense dequantize per forward, then a normal
        # 16-bit GEMM. No cached dense copy -- caching it would hold both the
        # 4-bit codes and a full bf16 weight on the device, which is exactly
        # the memory this lane exists to avoid.
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


# ---------------------------------------------------------------------------
# Load-time conversion
# ---------------------------------------------------------------------------


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
    # The logical weight shape is recorded rather than inferred at dequantize
    # time: the packed code layout is (..., K // 2) and the orthogonal shape
    # is not recoverable from it alone.
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
            # register_parameter(None) (as convert_model_to_nvfp4 does) rather
            # than popping the key: `layer.weight` then reads as None instead
            # of raising AttributeError.
            original = mod._parameters.get("weight")
            if original is not None:
                original.grad = None
            mod.register_parameter("weight", None)
            purged += 1

    if converted:
        # Say plainly what was produced. This is the 4-bit *storage* receipt,
        # not a throughput claim: apply() still runs a dense 16-bit GEMM.
        logger.info(
            "W4A16 conversion receipt: quantized %d linear layers (%s, reference dequantize-then-GEMM "
            "path); purged %d original bf16 weight tensors.", converted,
            ", ".join(f"group_size={g}, bits={b}" for g, b in sorted(schemes)), purged)
        logger.info("W4A16: no fused 4-bit GEMM is available in this build, so forward compute is a dense "
                    "16-bit GEMM over a dequantized weight. Expect BF16-comparable VRAM for the transient "
                    "dense weight and slower-than-BF16 step times.")


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
    "convert_model_to_w4a16",
    "minimax_h3_w4a16_prefixes",
    "w4a16_dequantize",
    "w4a16_quantize",
]
