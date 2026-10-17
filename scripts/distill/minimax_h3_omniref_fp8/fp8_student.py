# SPDX-License-Identifier: Apache-2.0
"""FP8 W8A8 fake quantization of an OmniRef PDD student, for the teacher-forced QAD eval harness.

Reproduces the numerics of the deployed FP8 linears (``FP8QuantizeMethod`` with per-channel weights) on
the harness's bf16-compute student, without a second weight format:

* weights: per-output-channel float8_e4m3fn round trip of the dense weight, the same math as
  ``quantize_minimax_h3_dit_fp8.py`` / ``convert_model_to_fp8`` (applied in place on the local FSDP shard);
* activations: a forward-pre-hook rounds each linear input to float8_e4m3fn, either per token with a dynamic
  scale (the deployed kernel, ``fp8_kernels._quantize_rowwise_kernel``) or per tensor with a static scale
  taken from a calibrated amax table (not supported by the deployed kernels; scoring emulation only).

The GEMM itself is bf16 on dequantized operands. Products of two FP8 values are exact in fp32, so this
differs from the real ``_scaled_mm`` only by the extra bf16 rounding of the dequantized operands (~2**-9
relative, against the ~2**-4 FP8 step).
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)
_LINEAR = re.compile(r"(?:^|\.)(?P<scope>transformer_blocks|token_refiner\.refiner_blocks)\.(?P<idx>\d+)\."
                     r"(?P<sub>attn\.to_q|attn\.to_k|attn\.to_v|attn\.to_out|ff\.fc_in|ff\.fc_out)$")
# q/k/v (and the bf16 gate) read one activation tensor: they share the to_q calibration entry.
_ACT_KEY = {"attn.to_k": "attn.to_q", "attn.to_v": "attn.to_q"}


def quantize_weight_per_channel(weight: torch.Tensor) -> torch.Tensor:
    """Dense ``[out, in]`` weight -> its per-output-channel FP8 round trip (float32)."""
    w = weight.to(torch.bfloat16)
    absmax = w.abs().amax(dim=1).nan_to_num().float()
    scale = (absmax / FP8_MAX).clamp(min=FP8_MIN_SCALE)
    q = (w / scale.to(w.dtype).unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q.float() * scale.unsqueeze(1)


def fake_quant_activation(x: torch.Tensor, static_scale: float | None = None) -> torch.Tensor:
    """Round ``x`` to FP8 per token (dynamic) or with one fixed scale; returns ``x``'s dtype."""
    flat = x.reshape(-1, x.shape[-1]).float()
    if static_scale is None:
        scale = (flat.abs().amax(dim=-1, keepdim=True) / FP8_MAX).clamp(min=FP8_MIN_SCALE)
    else:
        scale = torch.full((1, 1), max(static_scale, FP8_MIN_SCALE), device=x.device, dtype=torch.float32)
    q = (flat * (1.0 / scale)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).float() * scale
    return q.to(x.dtype).view_as(x)


def _local(param: torch.Tensor) -> torch.Tensor:
    if hasattr(param, "to_local"):
        placements = getattr(param, "placements", ())
        if any(type(p).__name__ == "Shard" and p.dim != 0 for p in placements):
            raise ValueError(f"FP8 weight quantization needs row (dim 0) sharding, got {placements}")
        return param.to_local()
    return param.data


def named_fp8_linears(transformer: torch.nn.Module, include_refiner: bool) -> Iterable[tuple[str, str, torch.nn.Module]]:
    """Yield ``(module name, calibration key, module)``; the key is ``b<idx>.<sub>`` or ``r<idx>.<sub>``."""
    for name, module in transformer.named_modules():
        match = _LINEAR.search(name.replace("_checkpoint_wrapped_module.", ""))
        if match is None or (match["scope"] != "transformer_blocks" and not include_refiner):
            continue
        prefix = "b" if match["scope"] == "transformer_blocks" else "r"
        sub = _ACT_KEY.get(match["sub"], match["sub"])
        yield name, f"{prefix}{match['idx']}.{sub}", module


def install_fp8(transformer: torch.nn.Module,
                *,
                weights: bool,
                activations: str,
                act_amax: dict[str, float] | None = None,
                include_refiner: bool = True) -> dict[str, Any]:
    """Quantize weights in place and/or hook activations. ``activations``: none | dynamic | static.

    Static scales (``act_amax[key] / 448``) apply to the main blocks; refiner linears stay dynamic.
    """
    if activations not in ("none", "dynamic", "static"):
        raise ValueError(f"activations must be none, dynamic or static, got {activations!r}")
    if activations == "static" and not act_amax:
        raise ValueError("static activation scales need an amax table")
    count, hooks = 0, []
    for name, key, module in named_fp8_linears(transformer, include_refiner):
        if weights:
            param = module.weight
            local = _local(param)
            with torch.no_grad():
                local.copy_(quantize_weight_per_channel(local).to(local.dtype))
        if activations != "none":
            static = None
            if activations == "static" and key.startswith("b"):
                if key not in act_amax:
                    raise KeyError(f"no calibrated amax for {key} ({name})")
                static = float(act_amax[key]) / FP8_MAX
            hooks.append(module.register_forward_pre_hook(_activation_hook(static)))
        count += 1
    if count == 0:
        raise ValueError("no FP8 linears matched the transformer")
    return {"linears": count, "hooks": hooks}


def _activation_hook(static_scale: float | None) -> Callable[[torch.nn.Module, tuple[Any, ...]], tuple[Any, ...]]:

    def hook(module: torch.nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        return (fake_quant_activation(args[0], static_scale), *args[1:])

    return hook
