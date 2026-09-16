# SPDX-License-Identifier: Apache-2.0
"""NVFP4 quantization (FlashInfer-backed) for LTX-2 and MiniMax-H3.

NVFP4 is NVIDIA's block-scaled FP4 format (e2m1 mantissa, fp32 alpha,
``layout_128x4`` scale layout, group size 16) — distinct from
generic FP4 / OCP-FP4 / MX-FP4. We name the public surface ``NVFP4``
explicitly so downstream callers don't conflate it with other FP4
variants that may land later (e.g. AMD's MX-FP4 or vendor-neutral
e3m0).

Upstreamed from ``FastVideo-internal`` so consumers that load LTX-2
weights with NVFP4 quantization can drive the public package
end-to-end.

The set of quantized linears is **per-model configuration**, not a
hardcoded constant: ``NVFP4Config(layer_prefixes=...)`` selects it, and
``layer_prefixes=None`` keeps the historical LTX-2 set the default so
nothing regresses. A model whose prefix set is not supplied therefore
attaches **no** quant methods and runs dense in silence — see
``is_nvfp4_linear_prefix`` and the module docs in
``docs/quantization/h3_nvfp4.md``.

``NVFP4Config.for_minimax_h3()`` returns the MiniMax-H3 set (300 block
linears), with H3's VSA compression gate ``attn.to_gate_compress``
excluded.

Quantized weights can be written to / restored from a compact sidecar
safetensors file (packed FP4 codes + block scales + global scale, ~4x
smaller than the dense bf16 weights) instead of being re-derived from
dense weights at load time — see ``save_nvfp4_checkpoint`` /
``load_nvfp4_checkpoint``.

`flashinfer` is imported lazily inside the call paths that need it.
This keeps ``import fastvideo`` cheap on hosts where flashinfer is
not installed; only the actual NVFP4 quantize / matmul ops fail at
use time, with a clear error.
"""
from __future__ import annotations

import json
import logging
import os
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


def _require_flashinfer() -> tuple[Any, Any, Any]:
    """Lazy flashinfer import — raised at use time, not import time.

    Returns the bound ``(SfLayout, mm_fp4, nvfp4_quantize)`` triple from
    flashinfer. Raises ``ImportError`` with an actionable hint if the
    package is not available.
    """
    try:
        from flashinfer import (  # type: ignore[import-not-found]
            SfLayout, mm_fp4, nvfp4_quantize,
        )
    except ImportError as exc:  # pragma: no cover - depends on host env
        raise ImportError("NVFP4 quantization requires flashinfer. "
                          "Install with `pip install flashinfer-python`.") from exc
    return SfLayout, mm_fp4, nvfp4_quantize


_LTX2_REFINE_ONLY_SUFFIXES = (
    ".audio_to_video_attn.to_q",
    ".video_to_audio_attn.to_k",
    ".video_to_audio_attn.to_v",
)

_LTX2_NVFP4_BLOCK_LINEAR_SUFFIXES = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out",
    "attn2.to_q",
    "attn2.to_out",
    "audio_to_video_attn.to_q",
    "audio_to_video_attn.to_out",
    "video_to_audio_attn.to_k",
    "video_to_audio_attn.to_v",
    "ffn.fc_in",
    "ffn.fc_out",
)
_LTX2_NVFP4_LINEAR_PREFIXES = frozenset(f"ltx2.blocks.{block_idx}.{suffix}" for block_idx in range(48)
                                        for suffix in _LTX2_NVFP4_BLOCK_LINEAR_SUFFIXES) | frozenset(
                                            ("ltx2.adaln_single.linear", ))

# --- MiniMax-H3 layer set -------------------------------------------------
#
# H3's DiT (``fastvideo/models/dits/minimax_h3.py``) names its block linears
# ``{prefix}.transformer_blocks.{i}.{suffix}`` with the default
# ``prefix="minimax_h3"`` and ``num_layers=50``
# (``fastvideo/configs/models/dits/minimax_h3.py``). The six suffixes below are
# the always-dense block linears — attention QKV/out and the SwiGLU FFN — which
# carry most of H3's parameters. The token refiner, the per-block AdaLN
# modulation (``adaln_proj.linear``), the patch/audio/context projections and
# ``norm_out.linear`` are deliberately left dense: they are small, run once per
# block or once per forward, and quantizing them buys little while adding
# activation-quantize overhead on every call.
MINIMAX_H3_DIT_PREFIX = "minimax_h3"
MINIMAX_H3_NUM_LAYERS = 50
MINIMAX_H3_BLOCK_LINEAR_SUFFIXES = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out",
    "ff.fc_in",
    "ff.fc_out",
)
# 50 blocks x 6 linears = 300 quantized linears.
MINIMAX_H3_NVFP4_LINEAR_PREFIXES = frozenset(f"{MINIMAX_H3_DIT_PREFIX}.transformer_blocks.{block_idx}.{suffix}"
                                             for block_idx in range(MINIMAX_H3_NUM_LAYERS)
                                             for suffix in MINIMAX_H3_BLOCK_LINEAR_SUFFIXES)

# Linears that must NEVER be quantized, whatever a caller passes in
# ``layer_prefixes``. ``attn.to_gate_compress`` is H3's VSA sparse-attention
# compression gate: H3's own deployment path loads it dense, and the
# zero-initialized gate is probed structurally in the forward
# (``MiniMaxH3Attention._gate_active``) to skip a guaranteed-zero branch.
# Quantizing it would perturb the gate's numerics and defeat the exact-zero
# skip that keeps the VSA branch free while it is untrained.
_ALWAYS_EXCLUDED_LINEAR_SUFFIXES = ("attn.to_gate_compress", )
MINIMAX_H3_NVFP4_EXCLUDED_LINEAR_SUFFIXES = _ALWAYS_EXCLUDED_LINEAR_SUFFIXES


def _matches_linear_suffix(prefix: str, suffixes: frozenset[str] | tuple[str, ...]) -> bool:
    """True when *prefix* is one of *suffixes*, or ends at a dot boundary.

    Entries may be a full module path or a trailing suffix
    (``"attn.to_gate_compress"``), so one pattern covers every block of a
    stack. The dot boundary keeps ``"ff.fc_in"`` from matching a hypothetical
    ``"cross_ff.fc_in"``.
    """
    return any(prefix == suffix or prefix.endswith("." + suffix) for suffix in suffixes)


def is_ltx2_nvfp4_linear_prefix(prefix: str) -> bool:
    """Return whether *prefix* belongs to the LTX-2 NVFP4 deployment set.

    Kept as a module-level predicate for callers that predate the
    ``NVFP4Config.layer_prefixes`` field; per-instance configs should use
    :meth:`NVFP4Config.is_nvfp4_linear_prefix` instead.
    """
    return prefix in _LTX2_NVFP4_LINEAR_PREFIXES


def _is_ltx2_refine_only_prefix(prefix: str) -> bool:
    return any(prefix.endswith(suffix) for suffix in _LTX2_REFINE_ONLY_SUFFIXES)


def _get_ltx2_fp4_stage_profile(default: str = "refine") -> str:
    """Read the active stage profile from the forward context.

    Streaming inference flips between ``base`` and ``refine`` between
    segments; the FP4 layer set differs across the two. Falls back to
    ``default`` whenever the context is not available — this keeps the
    op safe to run outside the streaming server (e.g. during eager
    tests).
    """
    try:
        from fastvideo.forward_context import get_forward_context

        forward_ctx = get_forward_context()
        forward_batch = getattr(forward_ctx, "forward_batch", None)
        if forward_batch is None:
            return default
        extra = getattr(forward_batch, "extra", None)
        if not isinstance(extra, dict):
            return default
        profile = extra.get("ltx2_fp4_stage_profile", default)
        if profile in ("base", "refine"):
            return profile
        return default
    except Exception:
        return default


_OPS_REGISTERED = False


def _register_ops_once() -> None:
    """Register the fastvideo_fp4 torch ops on first import that needs
    them. Each op binds to flashinfer at call time; this just sets up
    the dispatcher entries."""
    global _OPS_REGISTERED
    if _OPS_REGISTERED:
        return

    @torch.library.custom_op(
        "fastvideo_fp4::nvfp4_quantize",
        mutates_args=(),
        device_types="cuda",
    )
    def _nvfp4_quantize_op(
        x: torch.Tensor,
        global_sf: torch.Tensor,
        sf_layout: int,
        do_shuffle: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        SfLayout, _, nvfp4_quantize = _require_flashinfer()
        return nvfp4_quantize(x, global_sf, sfLayout=SfLayout(sf_layout), do_shuffle=do_shuffle)

    @_nvfp4_quantize_op.register_fake
    def _nvfp4_quantize_op_fake(
        x: torch.Tensor,
        global_sf: torch.Tensor,
        sf_layout: int,
        do_shuffle: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del global_sf, sf_layout, do_shuffle
        m, k = x.shape
        quantized = torch.empty((m, (k + 1) // 2), device=x.device, dtype=torch.uint8)
        scales = torch.empty((m, (k + 15) // 16), device=x.device, dtype=torch.uint8)
        return quantized, scales

    @torch.library.custom_op(
        "fastvideo_fp4::mm_fp4",
        mutates_args=(),
        device_types="cuda",
    )
    def _mm_fp4_op(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        alpha: torch.Tensor | None,
        out_dtype: torch.dtype = torch.bfloat16,
        out: torch.Tensor | None = None,
        block_size: int = 16,
        use_8x4_sf_layout: bool = False,
        backend: str = "auto",
        use_nvfp4: bool = True,
    ) -> torch.Tensor:
        _, mm_fp4, _ = _require_flashinfer()
        if a.dtype == torch.float4_e2m1fn_x2:
            a = a.view(torch.uint8) if a.is_contiguous() else a.contiguous().view(torch.uint8)
        if b.dtype == torch.float4_e2m1fn_x2:
            b = b.view(torch.uint8) if b.is_contiguous() else b.contiguous().view(torch.uint8)

        return mm_fp4(
            a,
            b,
            a_scale,
            b_scale,
            alpha,
            out_dtype,
            out,
            block_size=block_size,
            use_8x4_sf_layout=use_8x4_sf_layout,
            backend=backend,
            use_nvfp4=use_nvfp4,
        )

    @_mm_fp4_op.register_fake
    def _mm_fp4_op_fake(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        alpha: torch.Tensor | None,
        out_dtype: torch.dtype = torch.bfloat16,
        out: torch.Tensor | None = None,
        block_size: int = 16,
        use_8x4_sf_layout: bool = False,
        backend: str = "auto",
        use_nvfp4: bool = True,
    ) -> torch.Tensor:
        del a_scale, b_scale, alpha, block_size, use_8x4_sf_layout, backend
        del use_nvfp4
        if out is not None:
            return out
        out_shape = (*a.shape[:-1], b.shape[1])
        return torch.empty(out_shape, device=a.device, dtype=out_dtype)

    _OPS_REGISTERED = True


def _nvfp4_quantize(
    x: torch.Tensor,
    global_sf: Any,
    *,
    sfLayout: Any,
    do_shuffle: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    _register_ops_once()
    SfLayout, _, _ = _require_flashinfer()
    if isinstance(sfLayout, SfLayout):
        sf_layout = sfLayout.value
    elif hasattr(sfLayout, "value"):
        sf_layout = int(sfLayout.value)
    else:
        sf_layout = int(sfLayout)
    if not torch.is_tensor(global_sf):
        global_sf = torch.tensor(global_sf, device=x.device, dtype=torch.float32)
    elif global_sf.device != x.device:
        global_sf = global_sf.to(device=x.device)
    if sf_layout == SfLayout.layout_linear.value:
        x_for_quant = x
        logical_rows = x.shape[0]
    else:
        # Sequence-parallel can feed either logical rows or row-padded
        # rows. Normalize to the kernel tile shape for swizzled layouts
        # so both paths share a stable quantization contract.
        row_tile = 8 if sf_layout == SfLayout.layout_8x4.value else 128
        logical_rows = x.shape[0]
        pad_rows = (-logical_rows) % row_tile
        x_for_quant = F.pad(x, (0, 0, 0, pad_rows))

    quantized, scales = torch.ops.fastvideo_fp4.nvfp4_quantize(x_for_quant, global_sf, sf_layout, do_shuffle)
    if sf_layout != SfLayout.layout_linear.value:
        quantized = quantized.narrow(0, 0, logical_rows)
    return quantized, scales


def _mm_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: Any,
    out_dtype: torch.dtype,
    out: torch.Tensor | None,
    **kwargs: Any,
) -> torch.Tensor:
    _register_ops_once()
    block_size = kwargs.pop("block_size", 16)
    use_8x4_sf_layout = kwargs.pop("use_8x4_sf_layout", False)
    backend = kwargs.pop("backend", "auto")
    use_nvfp4 = kwargs.pop("use_nvfp4", True)
    if kwargs:
        raise TypeError(f"Unsupported kwargs for _mm_fp4: {sorted(kwargs)}")
    if alpha is not None and not torch.is_tensor(alpha):
        alpha = torch.tensor(alpha, device=a.device, dtype=torch.float32)
    return torch.ops.fastvideo_fp4.mm_fp4(
        a,
        b,
        a_scale,
        b_scale,
        alpha,
        out_dtype,
        out,
        block_size,
        use_8x4_sf_layout,
        backend,
        use_nvfp4,
    )


def _coerce_fp4_input_dtype(x: torch.Tensor) -> torch.Tensor:
    """Coerce an activation to a dtype the FP4 linear accepts.

    The pre-attention norm can emit fp32 (e.g. in eager mode, without the
    torch.compile fusion that keeps it bf16). The FP4 linear emits bf16
    regardless (see _mm_fp4 out dtype), so cast fp32 -> bf16 rather than
    failing, matching the sibling fastvideo/layers/fp4linear.py. Non-floating
    inputs (e.g. int/bool) are a genuine error and are rejected fast.
    """
    if not x.is_floating_point():
        raise TypeError(f"fp4 linear expects floating-point inputs, got {x.dtype}")
    if x.dtype not in (torch.bfloat16, torch.float16):
        x = x.to(torch.bfloat16)
    return x


class NVFP4QuantizeMethod(QuantizeMethodBase):

    def __init__(self, layer_prefix: str = ""):
        super().__init__()
        self.weight_fp4 = None
        self.weight_scale = None
        self.x_global_sf = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        self.layer_prefix = layer_prefix
        self._is_refine_only_layer = _is_ltx2_refine_only_prefix(layer_prefix)
        # Set from NVFP4Config.retain_original_weights in get_quant_method:
        # True = retain every original bf16 weight; None/False (default) =
        # purge the purgeable set. Refine-only layers are always retained --
        # the base stage profile runs them dense by deployment contract.
        self._retain_original_weights: bool | None = None

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int],
                       input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs):
        weight = Parameter(torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=params_dtype,
        ),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def quantize_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        SfLayout, _, _ = _require_flashinfer()
        x = _coerce_fp4_input_dtype(x)
        x_2d = x.view(-1, x.shape[-1])
        x_fp4, x_scale = _nvfp4_quantize(
            x_2d,
            self.x_global_sf,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
        )
        return x_fp4, x_scale, self.x_global_sf

    def wants_prequantized_input(self) -> bool:
        if not self._is_refine_only_layer:
            return True
        stage_profile = _get_ltx2_fp4_stage_profile(default="refine")
        return stage_profile != "base"

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        pre_quantized: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> torch.Tensor:
        SfLayout, _, _ = _require_flashinfer()
        # The original bf16 weight may have been purged after FP4 conversion
        # (see convert_model_to_nvfp4); the packed FP4 weight keeps the
        # output dim as its first dimension (only K is packed 2-per-byte).
        weight = getattr(layer, "weight", None)
        out_dim = weight.shape[0] if weight is not None else layer._nvfp4_weight.shape[0]
        original_shape = x.shape

        # Stage-aware profile: keep refine-only FP4 layers in dense mode
        # during stage-1 denoising so the base path doesn't pay the
        # quantize/dequantize tax for layers it never touches.
        stage_profile = _get_ltx2_fp4_stage_profile(default="refine")
        if self._is_refine_only_layer and stage_profile == "base":
            if weight is None:
                raise RuntimeError(f"NVFP4 layer {self.layer_prefix!r} hit the stage-profile dense path, "
                                   "but its original weights were purged "
                                   "(NVFP4Config(retain_original_weights=False)). Streaming/two-stage "
                                   "deploys must load with retain_original_weights left unset (auto) or True.")
            out = (F.linear(x, weight, bias) if torch.cuda.is_available() or bias is None else F.linear(
                x, weight, bias.to(x.dtype)))
            return out.view(*original_shape[:-1], out_dim)
        if pre_quantized is not None:
            x_fp4, x_scale, x_global_sf = pre_quantized
            # FlashInfer fused norm+quant APIs may return 3D tensors for
            # 3D inputs. mm_fp4 only accepts 2D tensors, so flatten
            # batch/sequence dims here.
            if x_fp4.dim() > 2:
                x_fp4 = x_fp4.view(-1, x_fp4.shape[-1])
            if x_scale.dim() > 2:
                x_scale = x_scale.view(-1, x_scale.shape[-1])
        else:
            x = _coerce_fp4_input_dtype(x)
            x = x.view(-1, x.shape[-1])
            x_global_sf = self.x_global_sf
            x_fp4, x_scale = _nvfp4_quantize(
                x,
                x_global_sf,
                sfLayout=SfLayout.layout_128x4,
                do_shuffle=False,
            )

        weight_fp4 = layer._nvfp4_weight
        weight_scale = layer._nvfp4_weight_scale
        weight_global_sf = layer._weight_global_sf

        if hasattr(layer, "_nvfp4_alpha"):
            alpha = layer._nvfp4_alpha / x_global_sf
        else:
            alpha = 1.0 / (x_global_sf * weight_global_sf)

        out = _mm_fp4(
            x_fp4,
            weight_fp4.T,
            x_scale,
            weight_scale.T,
            alpha,
            torch.bfloat16,
            None,
            backend='auto',
        )

        if bias is not None:
            out = out + bias
        out = out.view(*original_shape[:-1], out_dim)
        return out


class NVFP4Config(QuantizationConfig):
    """NVFP4 quantization configuration, parameterized by layer paths.

    NVFP4 is NVIDIA's block-scaled FP4 (e2m1 mantissa, fp32 alpha,
    ``layout_128x4`` scale layout, group size 16).

    Which linears get quantized is set by ``layer_prefixes``. The default is
    the historical LTX-2 set, so ``NVFP4Config()`` behaves exactly as it did
    before the field existed. Other models must pass their own set (see
    :meth:`for_minimax_h3`) — a model whose layer paths are not covered
    attaches no quant methods at all and silently runs dense, which is the
    failure mode this field exists to remove.
    """

    def __init__(
        self,
        layer_profile: str = "refine",
        retain_original_weights: bool | None = None,
        layer_prefixes: frozenset[str] | set[str] | list[str] | None = None,
        exclude_prefixes: frozenset[str] | set[str] | list[str] | None = None,
    ):
        super().__init__()
        # ``base``: stage-1 set (no attn2.to_out, no cross-modal AV
        # projections). ``refine``: full stage-2 set. LTX-2 streaming only:
        # other models have no stage split (their ``_is_refine_only_layer`` is
        # always False, so every quantized layer stays on the FP4 path).
        self.layer_profile = layer_profile
        # Original bf16 ``layer.weight`` retention after FP4 conversion.
        # Default (None/False): purge the purgeable originals -- every
        # always-FP4 layer. Refine-only layers (the cross-modal AV
        # projections) are ALWAYS retained: the ``base`` stage profile runs
        # them dense by deployment contract, including the distilled
        # single-stage deploy. True: retain everything (debugging /
        # pre-purge behavior).
        self.retain_original_weights = retain_original_weights
        # Full module paths of the linears to quantize. None -> the LTX-2 set
        # (unchanged default). Frozen so a config instance is hashable and
        # cannot be mutated after it has been handed to a model.
        self.layer_prefixes: frozenset[str] = (frozenset(_LTX2_NVFP4_LINEAR_PREFIXES)
                                               if layer_prefixes is None else frozenset(layer_prefixes))
        # Additional never-quantize patterns (full path or trailing suffix).
        # Applied on top of ``_ALWAYS_EXCLUDED_LINEAR_SUFFIXES``, which no
        # caller can override.
        self.exclude_prefixes: frozenset[str] = frozenset(exclude_prefixes) if exclude_prefixes else frozenset()

    def is_nvfp4_linear_prefix(self, prefix: str) -> bool:
        """Whether *prefix* is quantized under this config.

        Exclusions are checked first, and ``_ALWAYS_EXCLUDED_LINEAR_SUFFIXES``
        is unconditional: a prefix listed in ``layer_prefixes`` by mistake
        (e.g. a glob that swept up ``attn.to_gate_compress``) still comes back
        False here, so the gate cannot be quantized by any caller.
        """
        if _matches_linear_suffix(prefix, _ALWAYS_EXCLUDED_LINEAR_SUFFIXES):
            return False
        if _matches_linear_suffix(prefix, self.exclude_prefixes):
            return False
        return prefix in self.layer_prefixes

    def get_name(self):
        return "nvfp4"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls):
        return 100

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def for_minimax_h3(cls, **kwargs: Any) -> NVFP4Config:
        """The MiniMax-H3 NVFP4 layer set (300 block linears).

        ``attn.to_gate_compress`` (the VSA sparse-attention gate) is excluded;
        see ``MINIMAX_H3_NVFP4_EXCLUDED_LINEAR_SUFFIXES``. Keyword arguments
        (e.g. ``retain_original_weights``) are forwarded to the constructor.
        """
        kwargs.setdefault("layer_prefixes", MINIMAX_H3_NVFP4_LINEAR_PREFIXES)
        kwargs.setdefault("exclude_prefixes", MINIMAX_H3_NVFP4_EXCLUDED_LINEAR_SUFFIXES)
        return cls(**kwargs)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> NVFP4Config:
        return cls(
            layer_profile=config.get("layer_profile", "refine"),
            retain_original_weights=config.get("retain_original_weights"),
            layer_prefixes=config.get("layer_prefixes"),
            exclude_prefixes=config.get("exclude_prefixes"),
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from fastvideo.layers.linear import LinearBase

        # Use the superset at build/load time, then switch active subset
        # dynamically in NVFP4QuantizeMethod.apply based on stage profile.
        if isinstance(layer, LinearBase) and self.is_nvfp4_linear_prefix(prefix):
            method = NVFP4QuantizeMethod(layer_prefix=prefix)
            method._retain_original_weights = self.retain_original_weights
            return method
        return None


def convert_model_to_nvfp4(model: torch.nn.Module) -> None:
    SfLayout, _, _ = _require_flashinfer()
    from torch.distributed.tensor import DTensor  # type: ignore

    for mod in model.modules():
        qm = getattr(mod, "quant_method", None)
        if isinstance(qm, NVFP4QuantizeMethod):
            weight = getattr(mod, "weight", None)
            if weight is None:
                continue
            weight_local = weight.to_local() if isinstance(weight, DTensor) else weight  # type: ignore[arg-type]
            weight_global_sf = (448 * 6) / weight_local.float().abs().nan_to_num().max()
            fp4_w, fp4_s = _nvfp4_quantize(
                weight_local,
                weight_global_sf,
                sfLayout=SfLayout.layout_128x4,
                do_shuffle=False,
            )
            weight_global_sf_t = torch.as_tensor(
                weight_global_sf,
                device=weight_local.device,
                dtype=torch.float32,
            )
            mod.register_buffer("_nvfp4_weight", fp4_w, persistent=False)
            mod.register_buffer("_nvfp4_weight_scale", fp4_s, persistent=False)
            mod.register_buffer(
                "_weight_global_sf",
                weight_global_sf_t.to(dtype=torch.bfloat16),
                persistent=False,
            )
            mod.register_buffer(
                "_nvfp4_alpha",
                (1.0 / weight_global_sf_t).to(dtype=torch.float32),
                persistent=False,
            )

    _apply_dense_weight_policy(model)


def _apply_dense_weight_policy(model: torch.nn.Module) -> None:
    """Drop the bf16 ``weight`` of every NVFP4 linear that no longer needs it.

    Refine-only layers are NEVER purgeable: the "base" stage profile runs them
    dense by deployment contract (the distilled single-stage deploy included —
    its forward context is the base profile, so e.g. ``audio_to_video_attn``
    routes dense every step). ``retain_original_weights`` therefore only widens
    retention (True = keep everything); it cannot narrow it below the
    dense-capable set.

    Shared by :func:`convert_model_to_nvfp4` and
    :func:`load_nvfp4_checkpoint` so a restored sidecar frees the same memory a
    fresh conversion would.
    """
    from torch.distributed.tensor import DTensor  # type: ignore

    purged = 0
    retained = 0
    purged_bytes = 0
    for mod in model.modules():
        qm = getattr(mod, "quant_method", None)
        if not isinstance(qm, NVFP4QuantizeMethod):
            continue
        weight = getattr(mod, "weight", None)
        if weight is None:
            continue
        retain_flag = getattr(qm, "_retain_original_weights", None)
        retain = getattr(qm, "_is_refine_only_layer", False) or retain_flag is True
        if retain:
            retained += 1
        elif isinstance(weight, DTensor):
            # ponytail: purging FSDP-sharded originals needs per-shard
            # resharding bookkeeping; skip until a sharded deploy needs it.
            retained += 1
        else:
            purged_bytes += weight.numel() * weight.element_size()
            purged += 1
            mod.register_parameter("weight", None)

    if purged or retained:
        logger.info(
            "NVFP4 weight purge receipt: purged %d original bf16 weight tensors "
            "(%.2f GiB freed); retained %d (refine-only dense fallback or "
            "retain_original_weights).",
            purged,
            purged_bytes / (1 << 30),
            retained,
        )


# --- compact NVFP4 checkpoint sidecar -------------------------------------
#
# ``convert_model_to_nvfp4`` registers the packed FP4 tensors with
# ``persistent=False``, so they never enter a ``state_dict`` and a saved
# checkpoint carries dense bf16 weights plus a load-time quantization pass.
# A *sidecar* writes those tensors to their own safetensors file, keyed by
# module path, and restores them without re-running the conversion.
#
# Layout contract (must match ``apply`` / ``quantize_input`` /
# ``convert_model_to_nvfp4`` exactly, or the restored weights are garbage):
#
#   ``_nvfp4_weight``        uint8   ``(out_dim, ceil(in_dim / 2))`` — two e2m1
#                            codes per byte, K packed, N unpacked, exactly as
#                            ``nvfp4_quantize`` returns it (``apply`` passes
#                            ``.T`` to ``mm_fp4``).
#   ``_nvfp4_weight_scale``  uint8   ``(out_dim, ceil(in_dim / 16))`` — e4m3
#                            block-scale bit patterns, ``SfLayout.layout_128x4``
#                            with ``do_shuffle=False`` (row-padded to the
#                            128-row tile by ``_nvfp4_quantize`` and narrowed
#                            back; the padding is not stored).
#   ``_weight_global_sf``    bfloat16 scalar — ``(448 * 6) / max|W|``.
#   ``_nvfp4_alpha``         float32  scalar — ``1 / weight_global_sf`` at fp32
#                            precision, so it is stored separately rather than
#                            recomputed from the bf16-rounded global sf.
#
# Block size is 16 throughout, and the per-row activation scale
# ``NVFP4QuantizeMethod.x_global_sf`` is not persisted: it is a constant
# ``1.0`` on the method (never data-derived), so it must be identical on both
# sides of a save/load.

NVFP4_SIDECAR_SUFFIX = ".nvfp4.safetensors"
# Filename used when the sidecar sits inside a checkpoint *directory*; it does
# not carry the suffix above, which is what a sibling file is named with.
NVFP4_DIR_SIDECAR_NAME = "nvfp4.safetensors"
_NVFP4_SIDECAR_FORMAT = "fastvideo.nvfp4"
_NVFP4_SIDECAR_VERSION = 1
_NVFP4_SIDECAR_METADATA_KEY = "fastvideo_nvfp4"
_NVFP4_SIDECAR_KEY_SEP = "::"
_NVFP4_SIDECAR_SF_LAYOUT = "layout_128x4"
_NVFP4_SIDECAR_DO_SHUFFLE = False
_NVFP4_SIDECAR_BLOCK_SIZE = 16
# Order matters only for the manifest; every buffer is optional on load so a
# future format can add tensors without breaking older readers.
_NVFP4_SIDECAR_BUFFERS = (
    "_nvfp4_weight",
    "_nvfp4_weight_scale",
    "_weight_global_sf",
    "_nvfp4_alpha",
)


def _sidecar_key(module_fqn: str, buffer_name: str) -> str:
    return f"{module_fqn}{_NVFP4_SIDECAR_KEY_SEP}{buffer_name}"


def _split_sidecar_key(key: str) -> tuple[str, str]:
    module_fqn, _, buffer_name = key.rpartition(_NVFP4_SIDECAR_KEY_SEP)
    return module_fqn, buffer_name


def _nvfp4_tagged_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module, NVFP4QuantizeMethod]]:
    tagged = []
    for fqn, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        if isinstance(qm, NVFP4QuantizeMethod):
            tagged.append((fqn, mod, qm))
    return tagged


def _is_dtensor(tensor: torch.Tensor) -> bool:
    # Imported lazily: torch.distributed.tensor is not cheap to import and is
    # absent on some builds.
    try:
        from torch.distributed.tensor import DTensor  # type: ignore
    except ImportError:  # pragma: no cover - depends on the torch build
        return False
    return isinstance(tensor, DTensor)


def nvfp4_sidecar_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Collect the quantized tensors of every NVFP4 linear in *model*.

    Keys are ``"<module fqn>::<buffer name>"`` and values are detached CPU
    copies. Modules whose buffers are missing (never converted) are skipped;
    the returned mapping is what :func:`save_nvfp4_checkpoint` writes.

    FSDP note: a DTensor buffer is saved as this rank's local shard, so a
    sharded save is only reloadable into an identically sharded model.
    """
    state: dict[str, torch.Tensor] = {}
    for fqn, mod, _ in _nvfp4_tagged_modules(model):
        for name in _NVFP4_SIDECAR_BUFFERS:
            tensor = getattr(mod, name, None)
            if tensor is None:
                continue
            if _is_dtensor(tensor):
                tensor = tensor.to_local()  # type: ignore[attr-defined]
            state[_sidecar_key(fqn, name)] = tensor.detach().to("cpu", copy=True).contiguous()
    return state


def save_nvfp4_checkpoint(
    model: torch.nn.Module,
    path: str | os.PathLike[str],
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the model's NVFP4 tensors to a compact sidecar safetensors file.

    The file is roughly the packed-FP4 size (0.5 byte/code plus 1 byte per 16
    codes) instead of the dense bf16 size — about 3.6x smaller for a
    power-of-two K. Returns a receipt dict (also logged) describing the layer
    count and both sizes. Raises ``RuntimeError`` when the model has no NVFP4
    linears, which usually means the config's ``layer_prefixes`` did not cover
    the model's layer paths.
    """
    from safetensors.torch import save_file

    state = nvfp4_sidecar_state_dict(model)
    tagged = _nvfp4_tagged_modules(model)
    if not tagged:
        raise RuntimeError("No NVFP4 linear layers found in this model; nothing to serialize. "
                           "Check that the model was built with an NVFP4Config whose "
                           "layer_prefixes cover its layer paths (e.g. NVFP4Config.for_minimax_h3()).")
    if not state:
        raise RuntimeError(f"Found {len(tagged)} NVFP4-tagged linear layers but none carry quantized "
                           "buffers. Call convert_model_to_nvfp4(model) before saving a sidecar.")

    layers: dict[str, list[int]] = {}
    quant_prefixes: dict[str, str] = {}
    dense_bytes = 0
    for fqn, mod, qm in tagged:
        packed = getattr(mod, "_nvfp4_weight", None)
        weight = getattr(mod, "weight", None)
        if packed is None and weight is None:
            continue
        if weight is not None:
            out_dim, in_dim = int(weight.shape[0]), int(weight.shape[1])
        else:
            # Dense weight already purged: K = 2 codes/byte (H3's dims are
            # even, so the ceil in the packed shape is exact).
            out_dim, in_dim = int(packed.shape[0]), int(packed.shape[1]) * 2
        layers[fqn] = [out_dim, in_dim]
        quant_prefixes[fqn] = getattr(qm, "layer_prefix", "") or ""
        dense_bytes += out_dim * in_dim * 2

    metadata = {
        "format": _NVFP4_SIDECAR_FORMAT,
        "version": _NVFP4_SIDECAR_VERSION,
        "sf_layout": _NVFP4_SIDECAR_SF_LAYOUT,
        "do_shuffle": _NVFP4_SIDECAR_DO_SHUFFLE,
        "block_size": _NVFP4_SIDECAR_BLOCK_SIZE,
        "num_layers": len(layers),
        "layers": layers,
        "quant_prefixes": quant_prefixes,
        "model_class": type(model).__name__,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    payload = dict(state)
    serialized_bytes = sum(t.numel() * t.element_size() for t in payload.values())
    save_file(payload, os.fspath(path), metadata={_NVFP4_SIDECAR_METADATA_KEY: json.dumps(metadata)})

    receipt = {
        "path": os.fspath(path),
        "num_layers": len(layers),
        "num_tensors": len(payload),
        "quantized_bytes": serialized_bytes,
        "dense_bfloat16_bytes": dense_bytes,
        "compression_ratio": (dense_bytes / serialized_bytes) if serialized_bytes else 0.0,
    }
    logger.info(
        "NVFP4 sidecar: wrote %d layers / %d tensors to %s (%.2f GiB quantized vs "
        "%.2f GiB dense bf16, %.2fx smaller).",
        receipt["num_layers"],
        receipt["num_tensors"],
        receipt["path"],
        serialized_bytes / (1 << 30),
        dense_bytes / (1 << 30),
        receipt["compression_ratio"],
    )
    return receipt


def read_nvfp4_sidecar_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return the manifest of a sidecar file without materializing its tensors."""
    from safetensors import safe_open

    with safe_open(os.fspath(path), framework="pt", device="cpu") as handle:
        raw = handle.metadata() or {}
    if _NVFP4_SIDECAR_METADATA_KEY not in raw:
        raise ValueError(f"{os.fspath(path)} is not a FastVideo NVFP4 sidecar "
                         f"(no {_NVFP4_SIDECAR_METADATA_KEY!r} metadata).")
    return json.loads(raw[_NVFP4_SIDECAR_METADATA_KEY])


def nvfp4_sidecar_path_for(checkpoint_path: str | os.PathLike[str]) -> str:
    """Conventional sidecar path for a transformer checkpoint or directory.

    ``.../transformer.safetensors`` -> ``.../transformer.nvfp4.safetensors``;
    a directory -> ``<dir>/nvfp4.safetensors``.
    """
    raw = os.fspath(checkpoint_path)
    if os.path.isdir(raw):
        return os.path.join(raw, NVFP4_DIR_SIDECAR_NAME)
    if raw.endswith(".safetensors"):
        return raw[:-len(".safetensors")] + NVFP4_SIDECAR_SUFFIX
    return raw + NVFP4_SIDECAR_SUFFIX


def _sidecar_target_device(mod: torch.nn.Module, name: str) -> torch.device | None:
    """Device the restored buffer should live on.

    Mirrors ``convert_model_to_nvfp4``, which registers the buffers on the
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


def load_nvfp4_checkpoint(
    model: torch.nn.Module,
    path: str | os.PathLike[str],
    *,
    strict: bool = True,
    purge_dense_weights: bool = True,
) -> int:
    """Restore NVFP4 tensors from a sidecar, skipping ``convert_model_to_nvfp4``.

    Registers ``_nvfp4_weight`` / ``_nvfp4_weight_scale`` /
    ``_weight_global_sf`` / ``_nvfp4_alpha`` on every NVFP4-tagged linear from
    the sidecar, byte-for-byte as the conversion would have produced them. The
    dense bf16 weights are never touched (they may be absent entirely), and no
    flashinfer call is made — this works on a host that only needs to *serve* a
    pre-quantized checkpoint.

    ``strict`` raises on any layer-set or shape mismatch (a sidecar that does
    not describe this model); with ``strict=False`` the mismatches are logged
    and skipped, leaving those layers unconverted. ``purge_dense_weights``
    applies the same retention policy as ``convert_model_to_nvfp4``.

    Returns the number of layers restored.
    """
    from safetensors import safe_open

    tagged = _nvfp4_tagged_modules(model)
    if not tagged:
        raise RuntimeError("No NVFP4 linear layers are attached to this model, so a sidecar cannot be "
                           "restored. This is the silent-dense failure mode: the model's NVFP4Config "
                           "layer_prefixes do not cover its layer paths (for MiniMax-H3 use "
                           "NVFP4Config.for_minimax_h3()).")

    manifest = read_nvfp4_sidecar_metadata(path)
    if manifest.get("format") != _NVFP4_SIDECAR_FORMAT:
        raise ValueError(f"Unsupported NVFP4 sidecar format {manifest.get('format')!r} in {os.fspath(path)}.")
    if int(manifest.get("version", -1)) != _NVFP4_SIDECAR_VERSION:
        raise ValueError(f"Unsupported NVFP4 sidecar version {manifest.get('version')!r} in "
                         f"{os.fspath(path)} (this build reads version {_NVFP4_SIDECAR_VERSION}).")
    # A layout mismatch is unrecoverable (the packed nibbles would be read with
    # the wrong swizzle), so it is never downgraded by strict=False.
    expected_layout = {
        "sf_layout": _NVFP4_SIDECAR_SF_LAYOUT,
        "do_shuffle": _NVFP4_SIDECAR_DO_SHUFFLE,
        "block_size": _NVFP4_SIDECAR_BLOCK_SIZE,
    }
    for key, expected in expected_layout.items():
        if manifest.get(key) != expected:
            raise ValueError(f"NVFP4 sidecar {os.fspath(path)} was written with {key}="
                             f"{manifest.get(key)!r}, but this build quantizes with {key}={expected!r}.")

    saved_layers: dict[str, list[int]] = manifest.get("layers", {})
    model_fqns = {fqn for fqn, _, _ in tagged}
    missing = sorted(model_fqns - set(saved_layers))
    extra = sorted(set(saved_layers) - model_fqns)
    if missing or extra:
        message = (f"NVFP4 sidecar {os.fspath(path)} does not match this model: "
                   f"{len(missing)} layers missing from the sidecar, {len(extra)} layers not in the model. "
                   f"First missing={missing[:3]}, first extra={extra[:3]}.")
        if strict:
            raise ValueError(message)
        logger.warning("%s Restoring the intersection only.", message)

    restored = 0
    with safe_open(os.fspath(path), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        for fqn, mod, _ in tagged:
            if fqn not in saved_layers:
                continue
            out_dim, in_dim = (int(value) for value in saved_layers[fqn])
            tensors: dict[str, torch.Tensor] = {}
            for name in _NVFP4_SIDECAR_BUFFERS:
                key = _sidecar_key(fqn, name)
                if key not in available:
                    continue
                tensor = handle.get_tensor(key)
                expected = _expected_sidecar_shapes(name, out_dim, in_dim)
                if tuple(tensor.shape) not in expected:
                    raise ValueError(f"NVFP4 sidecar tensor {key} has shape {tuple(tensor.shape)}, expected one of "
                                     f"{list(expected)} for a ({out_dim}, {in_dim}) linear.")
                device = _sidecar_target_device(mod, name)
                if device is not None:
                    tensor = tensor.to(device=device, non_blocking=True)
                tensors[name] = tensor
            if "_nvfp4_weight" not in tensors or "_nvfp4_weight_scale" not in tensors:
                message = (f"NVFP4 sidecar entry for {fqn!r} is incomplete (has "
                           f"{sorted(tensors)}); the packed weight and its block scales are both required.")
                if strict:
                    raise ValueError(message)
                logger.warning("%s Skipping this layer.", message)
                continue
            for name, tensor in tensors.items():
                mod.register_buffer(name, tensor, persistent=False)
            restored += 1

    if purge_dense_weights:
        _apply_dense_weight_policy(model)

    logger.info("NVFP4 sidecar: restored %d quantized layers from %s (dense weights %s).", restored, os.fspath(path),
                "purged per policy" if purge_dense_weights else "left in place")
    return restored


def _expected_sidecar_shapes(name: str, out_dim: int, in_dim: int) -> tuple[tuple[int, ...], ...]:
    """The shapes a fresh conversion could produce for *name*.

    ``_nvfp4_quantize`` narrows the packed weight back to the logical row count
    but returns the block scales as the kernel emitted them, i.e. still padded
    to the 128-row tile: a layer whose output dim is not a multiple of 128 gets
    a scale tensor with more rows than the weight. Both are accepted so a
    sidecar written by a real flashinfer conversion validates.
    """
    if name == "_nvfp4_weight":
        return ((out_dim, (in_dim + 1) // 2), )
    if name == "_nvfp4_weight_scale":
        padded_rows = ((out_dim + 127) // 128) * 128
        shapes = {(out_dim, (in_dim + 15) // 16), (padded_rows, (in_dim + 15) // 16)}
        return tuple(sorted(shapes))
    # The two global scales are scalars produced by conversion.
    return ((), )


__all__ = [
    "MINIMAX_H3_NVFP4_EXCLUDED_LINEAR_SUFFIXES",
    "MINIMAX_H3_NVFP4_LINEAR_PREFIXES",
    "NVFP4Config",
    "NVFP4QuantizeMethod",
    "NVFP4_SIDECAR_SUFFIX",
    "convert_model_to_nvfp4",
    "is_ltx2_nvfp4_linear_prefix",
    "load_nvfp4_checkpoint",
    "nvfp4_sidecar_path_for",
    "nvfp4_sidecar_state_dict",
    "read_nvfp4_sidecar_metadata",
    "save_nvfp4_checkpoint",
]
