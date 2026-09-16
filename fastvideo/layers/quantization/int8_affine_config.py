# SPDX-License-Identifier: Apache-2.0
"""Weight-only affine INT8 quantization (group size 64) for CUDA inference.

This is the CUDA-side counterpart of the affine INT8 scheme the Apple
Silicon (MLX) deployment path already validates: per-group min/max affine
quantization with ``group_size=64`` and ``bits=8``, applied to the linear
*weights* only. Activations stay in bf16 — there is no activation
quantizer here and there should not be one until a fused INT8 GEMM lands.

The quantizer math is transcribed from
``fastvideo/layers/quantization/mlx_affine_qat.py`` (itself a transcription
of MLX's CPU ``quantized.cpp::quantize`` at v0.31.2), so the integer codes,
per-group scales, and per-group biases this config produces are the same
decisions the MLX runtime's quantizer makes. ``int8_affine_quantize`` and
``int8_affine_dequantize`` were verified bit-identical to
``mlx_affine_quantize_reference`` / ``mlx_affine_dequantize_reference`` on
fp32, fp16, and bf16 inputs (codes, scales, biases, and the dequantized
tensor all exactly equal) — the only representation change is that codes are
stored as ``uint8`` rather than ``int32``.

Design notes, deliberately different from ``nvfp4_config.py``:

- **No hardcoded single-model layer list.** ``NVFP4Config`` hardcodes the
  LTX-2 prefix set and its own docstring flags that as a wart. Here the
  selection rule is a constructor field (``target_layers`` /
  ``layer_suffixes``) with a model-agnostic default, and the MiniMax-H3 set
  is built from H3's real module names by
  :func:`minimax_h3_int8_affine_prefixes` / ``INT8AffineConfig.for_minimax_h3``.
- **A fail-closed deny list.** ``attn.to_gate_compress`` is H3's
  sparse-attention (VSA) gate. Quantizing it perturbs a *discrete* routing
  decision, so an error there is not a small output perturbation — it flips
  which tiles the sparse attention attends to. H3's own deploy path leaves
  it alone. The name matches none of the usual "norm"/"scale_shift_table"
  exclusion heuristics, so it is excluded by an explicit deny list that the
  constructor cannot widen away.

Like ``nvfp4_config.py``, this module allocates a *dense bf16* weight and
converts to the low-precision form at load time (``convert_model_to_int8_affine``),
so a plain BF16 checkpoint quantizes with no pre-quantized weights.
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
DEFAULT_BITS = 8
_EPS = 1e-7
# Affine codes span [0, 2**bits - 1]; for bits=8 that is [0, 255], which does
# NOT fit torch.int8. Codes are therefore stored as torch.uint8 (see
# ``int8_affine_quantize``) — the *scheme* is int8 affine, the container is
# unsigned because the zero-point convention is free-floating.
_MAX_UINT8_CODE = 255


# ---------------------------------------------------------------------------
# Affine quantizer — MLX-parity math
# ---------------------------------------------------------------------------


def _group(w: torch.Tensor, group_size: int) -> torch.Tensor:
    """View the last dim as ``(num_groups, group_size)``.

    Identical to ``mlx_affine_qat._group``: MLX groups along the last
    (input) axis, which for a linear weight is the contraction dimension.
    """
    if w.shape[-1] % group_size != 0:
        raise ValueError(f"Last dim {w.shape[-1]} is not divisible by group_size {group_size}; "
                         "MLX affine quantization groups along the last axis.")
    return w.reshape(*w.shape[:-1], w.shape[-1] // group_size, group_size)


def int8_affine_quantize(
    w: torch.Tensor,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    bits: int = DEFAULT_BITS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize like ``mx.quantize(..., mode="affine")``.

    Transcribed from ``mlx_affine_qat.mlx_affine_quantize_reference``. The
    decisions reproduced exactly are: per-group min/max in the input's
    arithmetic, a *negative* scale when ``|w_max| >= |w_min|`` (the
    quantizer anchors at the endpoint with the larger magnitude), the anchor
    re-expressed as an exact integer multiple of the scale so the extreme
    value round-trips exactly, ``rint`` (round-half-to-even) rounding, and
    codes clamped to ``[0, 2**bits - 1]``.

    Bit-identical to ``mlx_affine_quantize_reference`` for a given input
    dtype; the only representation change is that codes are ``torch.uint8``
    rather than ``torch.int32``, because ``2**bits - 1 == 255`` does not fit
    a signed byte. Callers must not cast these to ``int8`` — 255 would wrap
    to -1 and the dequantized weight would be wrong.

    ``scales``/``biases`` come back in ``w.dtype``, as in the reference.
    Since the codes are decided *before* that cast, calling this on an fp32
    ``w`` costs nothing in code fidelity and avoids rounding the stored
    scales to bf16 — which is what ``_quantize_layer_weight`` does (a bf16
    checkpoint value converts to fp32 exactly, so this is lossless input
    with a higher-precision scale store).

    Returns ``(codes, scales, biases)``. Mirroring the reference, ``codes``
    comes back in the *grouped* shape ``w.shape[:-1] + (K // group_size, group_size)``
    (not ``w.shape``) and ``scales``/``biases`` in
    ``w.shape[:-1] + (K // group_size,)``; pass ``out_shape=w.shape`` to
    ``int8_affine_dequantize`` to flatten the grouping back out.
    """
    n_bins = float((1 << bits) - 1)
    grouped = _group(w, group_size).float()

    w_min = grouped.min(dim=-1).values
    w_max = grouped.max(dim=-1).values
    mask = w_min.abs() > w_max.abs()
    scale = ((w_max - w_min) / n_bins).clamp_min(_EPS)
    scale = torch.where(mask, scale, -scale)
    edge = torch.where(mask, w_min, w_max)
    q0 = torch.round(edge / scale)
    nonzero_q0 = q0 != 0
    scale = torch.where(nonzero_q0, edge / torch.where(nonzero_q0, q0, torch.ones_like(q0)), scale)
    bias = torch.where(nonzero_q0, edge, torch.zeros_like(edge))

    codes = torch.round((grouped - bias.unsqueeze(-1)) / scale.unsqueeze(-1))
    codes = codes.clamp(min=0.0, max=n_bins)
    if codes.max().item() > _MAX_UINT8_CODE:
        raise ValueError(f"bits={bits} produced a code above {_MAX_UINT8_CODE}; only bits<=8 fits uint8 storage.")
    return codes.to(torch.uint8), scale.to(w.dtype), bias.to(w.dtype)


def int8_affine_dequantize(
    codes: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    *,
    out_shape: torch.Size | None = None,
) -> torch.Tensor:
    """``code * scale + bias`` per group — the inverse of ``int8_affine_quantize``.

    Mirrors ``mlx_affine_qat.mlx_affine_dequantize_reference`` (which itself
    matches MLX's *CPU* kernel). ``codes`` may be ``uint8``; the multiply is
    done in the scales' dtype, so pass fp32 scales to get the fp32 stream.

    ``codes`` is accepted in either shape the quantizer's callers use: the
    grouped ``(*, K // group_size, group_size)`` the reference returns, or the
    flattened ``(*, K)`` a stored weight buffer naturally has. The two are
    distinguished by rank (grouped codes are one rank above ``scales``).
    """
    dtype = scales.dtype
    if codes.dim() == scales.dim():
        codes = codes.reshape(*scales.shape, codes.shape[-1] // scales.shape[-1])
    deq = codes.to(dtype) * scales.unsqueeze(-1) + biases.unsqueeze(-1)
    if out_shape is not None:
        deq = deq.reshape(out_shape)
    return deq


# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------

# Model-agnostic default: the transformer-block GEMMs every DiT in this
# repo names this way (H3, LTX-2's `attn1/attn2`, ...). Suffix matching is
# used rather than a literal prefix set so depth/prefix variations cannot
# silently drop layers.
_GENERIC_LINEAR_SUFFIXES: tuple[str, ...] = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out",
    "ff.fc_in",
    "ff.fc_out",
)

# Names that are NEVER quantized by this config, whatever else is configured.
# Checked before the allowlist, and unioned with (never replaced by) any
# caller-supplied list, so this is fail-closed: there is no constructor
# argument that re-enables them.
_NEVER_QUANTIZE_SUBSTRINGS: tuple[str, ...] = (
    # H3's VSA sparse-attention compression gate. Quantizing it perturbs a
    # discrete routing decision and the checkpoint's gate is zero-initialized
    # (the branch is exactly disabled until finetuned). H3's deploy path
    # ignores it, so we must too.
    "to_gate_compress",
    # Global timestep-basis projector; feeds every block's modulation.
    "adaln_basis",
)

# Modules H3 already pins to fp32 (MiniMaxH3Transformer3DModel._keep_in_fp32_modules).
# They must not be targeted even if a suffix rule would otherwise reach them.
_H3_FP32_KEPT_SUBSTRINGS: tuple[str, ...] = (
    "proj_in",
    "audio_proj_in",
    "proj_out",
    "audio_proj_out",
    "time_embedder",
)

# `context_embedder` is H3's text input projection. It is NOT in the model's
# fp32 keep set, but it is the same kind of module as `proj_in` /
# `audio_proj_in` (an input projection), and H3's deploy keeps input
# projections in fp32. Quantizing the text conditioning stream while leaving
# the video/audio input streams in fp32 is an unvalidated asymmetry, so it is
# excluded by default. ``include_context_embedder=True`` opts in.
_H3_INPUT_PROJECTION_SUBSTRINGS: tuple[str, ...] = ("context_embedder", )

MINIMAX_H3_PREFIX = "minimax_h3"
MINIMAX_H3_NUM_LAYERS = 50
MINIMAX_H3_NUM_REFINER_LAYERS = 2
# Both block stacks hold the same `MiniMaxH3Attention` / `MiniMaxH3FeedForward`
# modules; the refiner stack simply has no `adaln_proj`.
MINIMAX_H3_BLOCK_SCOPES: tuple[str, ...] = (
    "transformer_blocks",
    "token_refiner.refiner_blocks",
)
# The H3 profile equals the generic set plus the per-block AdaLN modulation
# projection (`minimax_h3.transformer_blocks.{i}.adaln_proj.linear`), which is
# a real per-block GEMM in H3 and is listed in H3's linear inventory.
MINIMAX_H3_INT8_AFFINE_SUFFIXES: tuple[str, ...] = _GENERIC_LINEAR_SUFFIXES + ("adaln_proj.linear", )
MINIMAX_H3_INT8_AFFINE_EXCLUSIONS: tuple[str, ...] = (
    _NEVER_QUANTIZE_SUBSTRINGS + _H3_FP32_KEPT_SUBSTRINGS + _H3_INPUT_PROJECTION_SUBSTRINGS)


def minimax_h3_int8_affine_prefixes(
    *,
    prefix: str = MINIMAX_H3_PREFIX,
    num_layers: int = MINIMAX_H3_NUM_LAYERS,
    num_refiner_layers: int = MINIMAX_H3_NUM_REFINER_LAYERS,
    suffixes: Iterable[str] = MINIMAX_H3_INT8_AFFINE_SUFFIXES,
) -> frozenset[str]:
    """Enumerate the exact H3 linear prefixes this config targets.

    Built from H3's real module names as constructed in
    ``fastvideo/models/dits/minimax_h3.py``:
    ``MiniMaxH3TransformerBlock`` builds ``{prefix}.transformer_blocks.{i}.attn``,
    ``.ff``, ``.adaln_proj``; ``MiniMaxH3TokenRefiner`` builds
    ``{prefix}.token_refiner.refiner_blocks.{i}.attn`` / ``.ff``. Defaults
    match ``MiniMaxH3ArchConfig`` (``prefix="minimax_h3"``, ``num_layers=50``,
    ``num_refiner_layers=2``).

    The enumerated set is *not* how selection runs at runtime (suffix +
    deny-list matching is, so depth changes cannot silently drop layers);
    it exists to be asserted against in tests and to give callers who want
    a literal set one place to get it.
    """
    suffixes = tuple(suffixes)
    prefixes: set[str] = set()
    for index in range(num_layers):
        for suffix in suffixes:
            prefixes.add(f"{prefix}.transformer_blocks.{index}.{suffix}")
    for index in range(num_refiner_layers):
        for suffix in suffixes:
            # The refiner stack has no `adaln_proj`.
            if suffix.startswith("adaln_proj"):
                continue
            prefixes.add(f"{prefix}.token_refiner.refiner_blocks.{index}.{suffix}")
    return frozenset(prefixes)


class INT8AffineConfig(QuantizationConfig):
    """Weight-only affine INT8 (group-64) quantization for CUDA DiT inference.

    Layer selection is a constructor field, not a hardcoded model list:
    ``target_layers`` (explicit full prefixes) takes precedence when given,
    otherwise ``layer_suffixes`` is matched with ``str.endswith``. Both are
    subject to a fail-closed deny list — see ``exclude_substrings``.

    Weight-only: there is no activation quantizer, and ``INT8AffineQuantizeMethod.apply``
    dequantizes the stored codes back to the activation dtype and runs a
    normal bf16/fp32 GEMM. That is the correctness-first path; a fused INT8
    GEMM is a follow-up, not a prerequisite.

    Only the INT8 arithmetic is scheme-specific — nothing here is H3-only.
    Use :meth:`for_minimax_h3` for the H3 profile.
    """

    def __init__(
        self,
        group_size: int = DEFAULT_GROUP_SIZE,
        bits: int = DEFAULT_BITS,
        target_layers: Iterable[str] | None = None,
        layer_suffixes: Iterable[str] | None = None,
        exclude_substrings: Iterable[str] | None = None,
        include_context_embedder: bool = False,
        retain_original_weight: bool = True,
    ) -> None:
        super().__init__()
        if bits < 2 or bits > 8:
            raise ValueError(f"bits must be in [2, 8] (codes are stored as uint8), got {bits}")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        self.group_size = group_size
        self.bits = bits
        self.target_layers: frozenset[str] | None = (None if target_layers is None else frozenset(target_layers))
        self.layer_suffixes: tuple[str, ...] = (tuple(_GENERIC_LINEAR_SUFFIXES)
                                                if layer_suffixes is None else tuple(layer_suffixes))
        # Deny list is always unioned with the hard exclusions: passing a
        # custom list can only ever *add* exclusions, never remove one. This
        # is what keeps `to_gate_compress` unquantizable.
        self.exclude_substrings: tuple[str, ...] = tuple(_NEVER_QUANTIZE_SUBSTRINGS) + tuple(
            exclude_substrings or ())
        self._include_context_embedder = include_context_embedder
        if not include_context_embedder:
            self.exclude_substrings = self.exclude_substrings + _H3_INPUT_PROJECTION_SUBSTRINGS
        # Keep the dense bf16 `layer.weight` Parameter after conversion.
        # Default True: several H3 forwards read `linear.weight.dtype` to
        # cast their input (e.g. `MiniMaxH3AdaLayerNormModulation.forward`),
        # so purging it breaks the model. Setting False frees the bf16 copy
        # at the cost of requiring every caller to stop touching `.weight`.
        self.retain_original_weight = retain_original_weight

    def get_name(self) -> str:
        return "INT8Affine"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        """Turing (75).

        The compute path is a plain bf16/fp32 GEMM over a dequantized weight,
        so no INT8 tensor-core class is required; 75 matches ``AbsMaxFP8Config``
        and keeps the config loadable on the same hosts.
        """
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> INT8AffineConfig:
        return cls(
            group_size=config.get("group_size", DEFAULT_GROUP_SIZE),
            bits=config.get("bits", DEFAULT_BITS),
            target_layers=config.get("target_layers"),
            layer_suffixes=config.get("layer_suffixes"),
            exclude_substrings=config.get("exclude_substrings"),
            include_context_embedder=config.get("include_context_embedder", False),
            retain_original_weight=config.get("retain_original_weight", True),
        )

    @classmethod
    def for_minimax_h3(
        cls,
        *,
        group_size: int = DEFAULT_GROUP_SIZE,
        bits: int = DEFAULT_BITS,
        include_context_embedder: bool = False,
        retain_original_weight: bool = True,
    ) -> INT8AffineConfig:
        """The verified MiniMax-H3 profile: attention + FFN + AdaLN GEMMs.

        Excludes H3's fp32-pinned modules, the VSA gate, and (by default)
        the text input projection.
        """
        return cls(
            group_size=group_size,
            bits=bits,
            layer_suffixes=MINIMAX_H3_INT8_AFFINE_SUFFIXES,
            exclude_substrings=_H3_FP32_KEPT_SUBSTRINGS,
            include_context_embedder=include_context_embedder,
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
        return any(prefix.endswith(suffix) for suffix in self.layer_suffixes)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from fastvideo.layers.linear import LinearBase

        if isinstance(layer, LinearBase) and self.is_target_layer(prefix):
            return INT8AffineQuantizeMethod(
                layer_prefix=prefix,
                group_size=self.group_size,
                bits=self.bits,
                retain_original_weight=self.retain_original_weight,
            )
        return None


class INT8AffineQuantizeMethod(QuantizeMethodBase):
    """Linear method for weight-only affine INT8.

    ``create_weights`` allocates the same dense bf16 Parameter an
    unquantized linear would (so the BF16 checkpoint loads unchanged), and
    the INT8 codes/scales/biases arrive later as non-persistent buffers from
    :func:`convert_model_to_int8_affine` — i.e. conversion happens at *load*
    time, not construction time, exactly mirroring ``NVFP4QuantizeMethod``.
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
        :func:`convert_model_to_int8_affine`; this fallback exists so the
        config is still *correct* if that dispatch is missing, but it warns
        because reaching it means the loader hook did not fire.
        """
        if getattr(layer, "_int8_affine_codes", None) is not None:
            return True
        weight = getattr(layer, "weight", None)
        if weight is None:
            raise RuntimeError(f"INT8Affine layer {self.layer_prefix!r} has no weight and no quantized buffers.")
        if torch.is_grad_enabled():
            return False
        logger.warning(
            "INT8Affine: layer %r reached apply() unquantized; converting lazily. The loader hook "
            "(_maybe_quantize_model) did not dispatch to convert_model_to_int8_affine — check its "
            "isinstance chain in fastvideo/models/loader/fsdp_load.py.",
            self.layer_prefix,
        )
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
            return F.linear(x, weight.to(x.dtype) if weight.dtype != x.dtype else weight, bias)

        codes = layer._int8_affine_codes
        # Dequantize in fp32 (the scales' dtype), then match the activation.
        # `code * scale + bias` in fp32 is the more accurate side of the
        # CPU/Metal split documented in mlx_affine_qat.py.
        weight = int8_affine_dequantize(
            codes,
            layer._int8_affine_scales,
            layer._int8_affine_biases,
            out_shape=codes.shape,
        ).to(x.dtype)
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
    # fp32 source keeps the scale/bias solve out of bf16 (see int8_affine_quantize).
    # nan_to_num matches convert_model_to_nvfp4: one NaN would otherwise poison
    # every group it touches.
    w32 = weight_local.detach().float().nan_to_num()
    if w32.shape[-1] % group_size != 0:
        raise ValueError(f"INT8Affine layer {mod!r}: input dim {w32.shape[-1]} is not divisible by "
                         f"group_size {group_size}.")
    codes, scales, biases = int8_affine_quantize(w32, group_size=group_size, bits=bits)
    # Store codes flattened back to the weight shape (the quantizer returns the
    # grouped view) so `apply` can dequantize with out_shape=codes.shape.
    mod.register_buffer("_int8_affine_codes", codes.reshape(w32.shape).contiguous(), persistent=False)
    mod.register_buffer("_int8_affine_scales", scales.to(torch.float32).contiguous(), persistent=False)
    mod.register_buffer("_int8_affine_biases", biases.to(torch.float32).contiguous(), persistent=False)


def convert_model_to_int8_affine(model: torch.nn.Module, ) -> None:
    """Quantize every INT8Affine-tagged linear in-place after weights load.

    Mirrors ``convert_model_to_nvfp4`` / ``convert_model_to_fp8``: walk the
    module tree once, convert each layer whose ``quant_method`` is an
    :class:`INT8AffineQuantizeMethod`, and register the int8 codes plus
    per-group scales/biases as non-persistent buffers (so they are not
    written back into ``state_dict``/checkpoints).

    Callers: the loader hook ``_maybe_quantize_model`` in
    ``fastvideo/models/loader/fsdp_load.py``. *That hook is not edited by
    this module* — it dispatches on an explicit ``isinstance`` chain, so it
    needs a matching branch (see the module report). Without it,
    ``INT8AffineQuantizeMethod.apply`` converts lazily on first forward and
    logs a warning, so inference is still correct, just later and noisier.
    """
    converted = 0
    purged = 0
    shapes: set[tuple[int, int]] = set()
    for mod in model.modules():
        qm = getattr(mod, "quant_method", None)
        if not isinstance(qm, INT8AffineQuantizeMethod):
            continue
        weight = getattr(mod, "weight", None)
        if weight is None:
            continue
        _quantize_layer_weight(mod, weight, group_size=qm.group_size, bits=qm.bits)
        converted += 1
        shapes.add((qm.group_size, qm.bits))
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
        logger.info("INT8Affine conversion receipt: quantized %d linear layers (%s); purged %d original bf16 "
                    "weight tensors.", converted,
                    ", ".join(f"group_size={g}, bits={b}" for g, b in sorted(shapes)), purged)


__all__ = [
    "DEFAULT_BITS",
    "DEFAULT_GROUP_SIZE",
    "INT8AffineConfig",
    "INT8AffineQuantizeMethod",
    "MINIMAX_H3_BLOCK_SCOPES",
    "MINIMAX_H3_INT8_AFFINE_EXCLUSIONS",
    "MINIMAX_H3_INT8_AFFINE_SUFFIXES",
    "MINIMAX_H3_PREFIX",
    "convert_model_to_int8_affine",
    "int8_affine_dequantize",
    "int8_affine_quantize",
    "minimax_h3_int8_affine_prefixes",
]
