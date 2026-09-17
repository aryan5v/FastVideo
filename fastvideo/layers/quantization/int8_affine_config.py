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
DEFAULT_BITS = 8
_EPS = 1e-7
_MAX_UINT8_CODE = 255




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



_GENERIC_LINEAR_SUFFIXES: tuple[str, ...] = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out",
    "ff.fc_in",
    "ff.fc_out",
)

_NEVER_QUANTIZE_SUBSTRINGS: tuple[str, ...] = (
    "to_gate_compress",
    "adaln_basis",
)

_H3_FP32_KEPT_SUBSTRINGS: tuple[str, ...] = (
    "proj_in",
    "audio_proj_in",
    "proj_out",
    "audio_proj_out",
    "time_embedder",
)

_H3_INPUT_PROJECTION_SUBSTRINGS: tuple[str, ...] = ("context_embedder", )

MINIMAX_H3_PREFIX = "minimax_h3"
MINIMAX_H3_NUM_LAYERS = 50
MINIMAX_H3_NUM_REFINER_LAYERS = 2
MINIMAX_H3_BLOCK_SCOPES: tuple[str, ...] = (
    "transformer_blocks",
    "token_refiner.refiner_blocks",
)
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
        self.exclude_substrings: tuple[str, ...] = tuple(_NEVER_QUANTIZE_SUBSTRINGS) + tuple(
            exclude_substrings or ())
        self._include_context_embedder = include_context_embedder
        if not include_context_embedder:
            self.exclude_substrings = self.exclude_substrings + _H3_INPUT_PROJECTION_SUBSTRINGS
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
        weight = int8_affine_dequantize(
            codes,
            layer._int8_affine_scales,
            layer._int8_affine_biases,
            out_shape=codes.shape,
        ).to(x.dtype)
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
    w32 = weight_local.detach().float().nan_to_num()
    if w32.shape[-1] % group_size != 0:
        raise ValueError(f"INT8Affine layer {mod!r}: input dim {w32.shape[-1]} is not divisible by "
                         f"group_size {group_size}.")
    codes, scales, biases = int8_affine_quantize(w32, group_size=group_size, bits=bits)
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
            original = mod._parameters.get("weight")
            if original is not None:
                original.grad = None
            mod.register_parameter("weight", None)
            purged += 1

    if converted:
        logger.info("INT8Affine conversion receipt: quantized %d linear layers (%s); purged %d original bf16 "
                    "weight tensors.", converted,
                    ", ".join(f"group_size={g}, bits={b}" for g, b in sorted(shapes)), purged)



INT8_AFFINE_SIDECAR_SUFFIX = ".int8affine.safetensors"
INT8_AFFINE_DIR_SIDECAR_NAME = "int8_affine.safetensors"
_INT8_AFFINE_SIDECAR_FORMAT = "fastvideo.int8_affine"
_INT8_AFFINE_SIDECAR_VERSION = 1
_INT8_AFFINE_SIDECAR_METADATA_KEY = "fastvideo_int8_affine"
_INT8_AFFINE_SIDECAR_KEY_SEP = "::"
_INT8_AFFINE_SIDECAR_BUFFERS = (
    "_int8_affine_codes",
    "_int8_affine_scales",
    "_int8_affine_biases",
)
_INT8_AFFINE_SIDECAR_DTYPES = {
    "_int8_affine_codes": torch.uint8,
    "_int8_affine_scales": torch.float32,
    "_int8_affine_biases": torch.float32,
}


def _sidecar_key(module_fqn: str, buffer_name: str) -> str:
    return f"{module_fqn}{_INT8_AFFINE_SIDECAR_KEY_SEP}{buffer_name}"


def _int8_affine_tagged_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module, INT8AffineQuantizeMethod]]:
    tagged = []
    for fqn, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        if isinstance(qm, INT8AffineQuantizeMethod):
            tagged.append((fqn, mod, qm))
    return tagged


def _is_dtensor(tensor: torch.Tensor) -> bool:
    try:
        from torch.distributed.tensor import DTensor  # type: ignore
    except ImportError:  # pragma: no cover - depends on the torch build
        return False
    return isinstance(tensor, DTensor)


def int8_affine_sidecar_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Collect the quantized tensors of every INT8 affine linear in *model*.

    Keys are ``"<module fqn>::<buffer name>"`` and values are detached CPU
    copies. Modules whose buffers are missing (never converted) are skipped;
    the returned mapping is what :func:`save_int8_affine_checkpoint` writes.

    FSDP note: a DTensor buffer is saved as this rank's local shard, so a
    sharded save is only reloadable into an identically sharded model.
    """
    state: dict[str, torch.Tensor] = {}
    for fqn, mod, _ in _int8_affine_tagged_modules(model):
        for name in _INT8_AFFINE_SIDECAR_BUFFERS:
            tensor = getattr(mod, name, None)
            if tensor is None:
                continue
            if _is_dtensor(tensor):
                tensor = tensor.to_local()  # type: ignore[attr-defined]
            state[_sidecar_key(fqn, name)] = tensor.detach().to("cpu", copy=True).contiguous()
    return state


def save_int8_affine_checkpoint(
    model: torch.nn.Module,
    path: str | os.PathLike[str],
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the model's INT8 affine tensors to a compact sidecar safetensors file.

    The file is roughly the INT8 size (1 byte/code plus two fp32 per group —
    8.5 bits/weight at ``group_size=64``) instead of the dense bf16 size.
    Returns a receipt dict (also logged) with the module count and both sizes.
    Raises ``RuntimeError`` when the model has no INT8 affine linears, which
    usually means the config's layer selection did not cover the model's layer
    paths, and when the tagged layers carry no buffers (never converted).

    The manifest under the ``fastvideo_int8_affine`` metadata key carries the
    format name/version, the scheme (``group_size``/``bits``), the per-layer
    weight and buffer shapes, and the quantized module fqns (the keys of
    ``layers``), so a loader can validate a sidecar against a model without
    materializing the tensors.
    """
    from safetensors.torch import save_file

    state = int8_affine_sidecar_state_dict(model)
    tagged = _int8_affine_tagged_modules(model)
    if not tagged:
        raise RuntimeError("No INT8 affine linear layers found in this model; nothing to serialize. "
                           "Check that the model was built with an INT8AffineConfig whose layer selection "
                           "covers its layer paths (e.g. INT8AffineConfig.for_minimax_h3()).")
    if not state:
        raise RuntimeError(f"Found {len(tagged)} INT8Affine-tagged linear layers but none carry quantized "
                           "buffers. Call convert_model_to_int8_affine(model) before saving a sidecar.")

    layers: dict[str, dict[str, Any]] = {}
    quant_prefixes: dict[str, str] = {}
    dense_bytes = 0
    group_sizes: set[int] = set()
    bit_widths: set[int] = set()
    for fqn, mod, qm in tagged:
        codes = getattr(mod, "_int8_affine_codes", None)
        weight = getattr(mod, "weight", None)
        if codes is None and weight is None:
            continue
        weight_shape = [int(dim) for dim in (weight if weight is not None else codes).shape]
        tensors = {
            name: [int(dim) for dim in getattr(mod, name).shape]
            for name in _INT8_AFFINE_SIDECAR_BUFFERS if getattr(mod, name, None) is not None
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
        "format": _INT8_AFFINE_SIDECAR_FORMAT,
        "version": _INT8_AFFINE_SIDECAR_VERSION,
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
    save_file(payload, os.fspath(path), metadata={_INT8_AFFINE_SIDECAR_METADATA_KEY: json.dumps(metadata)})

    receipt = {
        "path": os.fspath(path),
        "num_layers": len(layers),
        "num_tensors": len(payload),
        "quantized_bytes": serialized_bytes,
        "dense_bfloat16_bytes": dense_bytes,
        "compression_ratio": (dense_bytes / serialized_bytes) if serialized_bytes else 0.0,
    }
    logger.info(
        "INT8Affine sidecar: wrote %d quantized modules / %d tensors (%d bytes) to %s "
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


def read_int8_affine_sidecar_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return the manifest of a sidecar file without materializing its tensors."""
    from safetensors import safe_open

    with safe_open(os.fspath(path), framework="pt", device="cpu") as handle:
        raw = handle.metadata() or {}
    if _INT8_AFFINE_SIDECAR_METADATA_KEY not in raw:
        raise ValueError(f"{os.fspath(path)} is not a FastVideo INT8 affine sidecar "
                         f"(no {_INT8_AFFINE_SIDECAR_METADATA_KEY!r} metadata).")
    return json.loads(raw[_INT8_AFFINE_SIDECAR_METADATA_KEY])


def int8_affine_sidecar_path_for(checkpoint_path: str | os.PathLike[str]) -> str:
    """Conventional sidecar path for a transformer checkpoint or directory.

    ``.../transformer.safetensors`` -> ``.../transformer.int8affine.safetensors``;
    a directory -> ``<dir>/int8_affine.safetensors``.
    """
    raw = os.fspath(checkpoint_path)
    if os.path.isdir(raw):
        return os.path.join(raw, INT8_AFFINE_DIR_SIDECAR_NAME)
    if raw.endswith(".safetensors"):
        return raw[:-len(".safetensors")] + INT8_AFFINE_SIDECAR_SUFFIX
    return raw + INT8_AFFINE_SIDECAR_SUFFIX


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


def _expected_sidecar_shapes(name: str, weight_shape: tuple[int, int], group_size: int) -> tuple[tuple[int, ...], ...]:
    """The one shape a fresh conversion would produce for *name*.

    Unlike the NVFP4 sidecar there is no padded variant to accept: the
    quantizer groups along the last axis and ``_group`` refuses a K that is
    not divisible by ``group_size``, so an exact divisor is the only
    legitimate layout. A padded scales tensor would not merely be unusual —
    ``int8_affine_dequantize`` recovers the group width as
    ``codes.shape[-1] // scales.shape[-1]``, so extra groups silently regroup
    every code in the row.
    """
    out_dim, in_dim = weight_shape
    if name == "_int8_affine_codes":
        return ((out_dim, in_dim), )
    if in_dim % group_size:
        raise ValueError(f"Sidecar declares a ({out_dim}, {in_dim}) weight with group_size {group_size}, "
                         "which does not divide the input dim; this layout cannot be dequantized.")
    return ((out_dim, in_dim // group_size), )


def load_int8_affine_checkpoint(
    model: torch.nn.Module,
    path: str | os.PathLike[str],
    *,
    strict: bool = True,
) -> int:
    """Restore INT8 affine tensors from a sidecar, skipping ``convert_model_to_int8_affine``.

    Registers ``_int8_affine_codes`` / ``_int8_affine_scales`` /
    ``_int8_affine_biases`` on every INT8Affine-tagged linear from the
    sidecar, byte-for-byte as the conversion would have produced them. The
    dense bf16 weights are never touched (they may be absent entirely), and
    nothing here needs a GPU or a fused INT8 kernel — the dequantize-then-GEMM
    reference path in :meth:`INT8AffineQuantizeMethod.apply` is pure PyTorch,
    so a pre-quantized checkpoint loads on any host.

    ``strict`` raises on any layer-set mismatch (a sidecar that does not
    describe this model); with ``strict=False`` those are logged and skipped,
    leaving those layers unconverted. Scheme (``group_size``/``bits``) and
    per-tensor shape/dtype mismatches are **never** downgraded: a mis-read
    code buffer produces garbage output with no error, so those always raise.

    Returns the number of layers restored.
    """
    from safetensors import safe_open

    tagged = _int8_affine_tagged_modules(model)
    if not tagged:
        raise RuntimeError("No INT8 affine linear layers are attached to this model, so a sidecar cannot be "
                           "restored. This is the silent-dense failure mode: the model's INT8AffineConfig "
                           "layer selection does not cover its layer paths (for MiniMax-H3 use "
                           "INT8AffineConfig.for_minimax_h3()).")

    manifest = read_int8_affine_sidecar_metadata(path)
    if manifest.get("format") != _INT8_AFFINE_SIDECAR_FORMAT:
        raise ValueError(f"Unsupported INT8 affine sidecar format {manifest.get('format')!r} in "
                         f"{os.fspath(path)}.")
    if int(manifest.get("version", -1)) != _INT8_AFFINE_SIDECAR_VERSION:
        raise ValueError(f"Unsupported INT8 affine sidecar version {manifest.get('version')!r} in "
                         f"{os.fspath(path)} (this build reads version {_INT8_AFFINE_SIDECAR_VERSION}).")

    saved_layers: dict[str, dict[str, Any]] = manifest.get("layers", {})
    model_fqns = {fqn for fqn, _, _ in tagged}
    missing = sorted(model_fqns - set(saved_layers))
    extra = sorted(set(saved_layers) - model_fqns)
    if missing or extra:
        message = (f"INT8 affine sidecar {os.fspath(path)} does not match this model: "
                   f"{len(missing)} layers missing from the sidecar, {len(extra)} layers not in the model. "
                   f"First missing={missing[:3]}, first extra={extra[:3]}.")
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
                raise ValueError(f"INT8 affine sidecar {os.fspath(path)} entry for {fqn!r} is malformed: "
                                 f"{entry!r} does not carry an integer weight_shape/group_size/bits.") from exc
            if len(weight_shape) != 2:
                raise ValueError(f"INT8 affine sidecar entry {fqn!r} declares weight shape {list(weight_shape)}; "
                                 "a linear weight is 2-D.")
            if group_size != qm.group_size or bits != qm.bits:
                raise ValueError(f"INT8 affine sidecar {os.fspath(path)} was written for {fqn!r} with "
                                 f"group_size={group_size}, bits={bits}, but this model quantizes it with "
                                 f"group_size={qm.group_size}, bits={qm.bits}.")
            weight = getattr(mod, "weight", None)
            if weight is not None and tuple(int(dim) for dim in weight.shape) != weight_shape:
                raise ValueError(f"INT8 affine sidecar entry {fqn!r} describes a {list(weight_shape)} weight, but "
                                 f"this model's layer has shape {list(weight.shape)}.")
            tensors: dict[str, torch.Tensor] = {}
            for name in _INT8_AFFINE_SIDECAR_BUFFERS:
                key = _sidecar_key(fqn, name)
                if key not in available:
                    continue
                tensor = handle.get_tensor(key)
                expected_dtype = _INT8_AFFINE_SIDECAR_DTYPES[name]
                if tensor.dtype != expected_dtype:
                    raise ValueError(f"INT8 affine sidecar tensor {key} has dtype {tensor.dtype}, expected "
                                     f"{expected_dtype}. Codes are uint8 because a bits=8 code spans [0, 255] "
                                     "and does not fit int8; casting would silently corrupt them.")
                expected = _expected_sidecar_shapes(name, weight_shape, group_size)
                if tuple(tensor.shape) not in expected:
                    raise ValueError(f"INT8 affine sidecar tensor {key} has shape {tuple(tensor.shape)}, expected "
                                     f"one of {list(expected)} for a {list(weight_shape)} linear with "
                                     f"group_size={group_size}.")
                device = _sidecar_target_device(mod, name)
                if device is not None:
                    tensor = tensor.to(device=device, non_blocking=True)
                tensors[name] = tensor
            if set(tensors) != set(_INT8_AFFINE_SIDECAR_BUFFERS):
                message = (f"INT8 affine sidecar entry for {fqn!r} is incomplete (has {sorted(tensors)}); "
                           f"all of {list(_INT8_AFFINE_SIDECAR_BUFFERS)} are required.")
                if strict:
                    raise ValueError(message)
                logger.warning("%s Skipping this layer.", message)
                continue
            for name, tensor in tensors.items():
                mod.register_buffer(name, tensor, persistent=False)
            restored += 1

    logger.info("INT8Affine sidecar: restored %d quantized modules from %s (dense weights untouched).", restored,
                os.fspath(path))
    return restored


__all__ = [
    "DEFAULT_BITS",
    "DEFAULT_GROUP_SIZE",
    "INT8AffineConfig",
    "INT8AffineQuantizeMethod",
    "INT8_AFFINE_DIR_SIDECAR_NAME",
    "INT8_AFFINE_SIDECAR_SUFFIX",
    "MINIMAX_H3_BLOCK_SCOPES",
    "MINIMAX_H3_INT8_AFFINE_EXCLUSIONS",
    "MINIMAX_H3_INT8_AFFINE_SUFFIXES",
    "MINIMAX_H3_PREFIX",
    "convert_model_to_int8_affine",
    "int8_affine_dequantize",
    "int8_affine_quantize",
    "int8_affine_sidecar_path_for",
    "int8_affine_sidecar_state_dict",
    "load_int8_affine_checkpoint",
    "minimax_h3_int8_affine_prefixes",
    "read_int8_affine_sidecar_metadata",
    "save_int8_affine_checkpoint",
]
