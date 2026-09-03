# SPDX-License-Identifier: Apache-2.0
"""Helpers for converting a hybrid MiniMax H3 checkpoint into FastVideo names.

The converter script is the user entry point. This module holds the key rewrite
and LoRA-merge arithmetic so tests can cover them without loading 70 GiB of
weights. It does not vendor an external runtime.
"""

from __future__ import annotations

import re
from typing import Any

import torch

from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig
from fastvideo.models.loader.utils import get_param_names_mapping

# Dropout in the wrapped dense ``to_out`` Sequential has no FastVideo parameter.
_SKIP_SUBSTRINGS = (".attn.orig.to_out.1.", ".attn.to_out.1.")
_LORA_INFIX = re.compile(r"\.lora_([AB])\.([^.]+)\.")
_DROPOUT_SUFFIXES = (".attn.orig.to_out.1.weight", ".attn.to_out.1.weight")


def remap_vdn_key(name: str) -> str | None:
    """Map one VDN / diffusers key onto the FastVideo H3 state-dict surface.

    Returns None for tensors FastVideo does not load (dropout).
    """
    if any(part in name for part in _SKIP_SUBSTRINGS) or name.endswith(_DROPOUT_SUFFIXES):
        return None
    mapper = get_param_names_mapping(MiniMaxH3ArchConfig().param_names_mapping)
    mapped, _, _ = mapper(name)
    return mapped


def normalize_lora_key(name: str) -> str:
    """``lora_A.default.weight`` / ``lora_A.turbo.weight`` -> ``lora_A.weight``."""
    return _LORA_INFIX.sub(r".lora_\1.", name)


def merge_lora_pairs(
    weights: dict[str, torch.Tensor],
    lora: dict[str, torch.Tensor],
    scale: float = 1.0,
) -> int:
    """Fold PEFT ``lora_A`` / ``lora_B`` pairs into ``weights`` in place.

    ``W += scale * (B @ A)`` in fp32, stored back in ``W``'s dtype. Targets are
    remapped through :func:`remap_vdn_key` so ``attn.orig.to_q`` lands on
    ``attn.to_q``.
    """
    normalized = {normalize_lora_key(key): tensor for key, tensor in lora.items()}
    merged = 0
    for name, lora_a in normalized.items():
        if ".lora_A." not in name:
            continue
        lora_b = normalized[name.replace(".lora_A.", ".lora_B.")]
        target = remap_vdn_key(name.split(".lora_A.")[0] + ".weight")
        if target is None:
            continue
        if target not in weights:
            raise KeyError(f"LoRA target {target!r} (from {name!r}) is missing from the base weights.")
        delta = (lora_b.to(torch.float32) @ lora_a.to(torch.float32)) * scale
        weights[target] = weights[target] + delta.to(dtype=weights[target].dtype)
        merged += 1
    return merged


def lora_scale_from_adapter_config(config: dict[str, Any], rank: int | None) -> float:
    """``alpha / rank`` when both are known, else 1.0 (alpha == rank)."""
    alpha = config.get("alpha")
    if alpha is None and isinstance(config.get("config"), dict):
        alpha = config["config"].get("alpha")
        rank = rank or config["config"].get("r") or config["config"].get("rank")
    rank = rank or config.get("r") or config.get("rank")
    if alpha is None or rank in (None, 0):
        return 1.0
    return float(alpha) / float(rank)


def hybrid_arch_fields_from_spec(model_spec: dict[str, Any] | None) -> dict[str, Any]:
    """Read window / delta-rule knobs from a VDN ``model_spec.json`` if present."""
    fields: dict[str, Any] = {
        "hybrid_attention": True,
        "hybrid_window_radius": 1,
        "hybrid_window_chunk": 5,
        "hybrid_anchor_frames": "both",
        "hybrid_delta_rule": "vdn_solve",
        "hybrid_enable_softmax_gate": True,
        "hybrid_enable_text_state": True,
        "hybrid_short_conv_targets": ["k", "v"],
        "hybrid_branch_parallel": True,
    }
    if not model_spec:
        return fields
    transforms = model_spec.get("transforms") or []
    blob: Any = transforms[0] if transforms else model_spec
    if isinstance(blob, dict) and "config" in blob:
        blob = blob["config"]
    hybrid = blob.get("hybrid_attention") if isinstance(blob, dict) else None
    if not isinstance(hybrid, dict):
        hybrid = blob if isinstance(blob, dict) else {}
    softmax = hybrid.get("softmax_attention") if isinstance(hybrid.get("softmax_attention"), dict) else {}
    linear = hybrid.get("linear_attention") if isinstance(hybrid.get("linear_attention"), dict) else {}
    if softmax.get("radius") is not None:
        fields["hybrid_window_radius"] = int(softmax["radius"])
    if softmax.get("chunk") is not None:
        fields["hybrid_window_chunk"] = int(softmax["chunk"])
    if hybrid.get("anchor_frames") is not None:
        fields["hybrid_anchor_frames"] = str(hybrid["anchor_frames"])
    if hybrid.get("enable_softmax_gate") is not None:
        fields["hybrid_enable_softmax_gate"] = bool(hybrid["enable_softmax_gate"])
    if linear.get("delta_rule") is not None:
        fields["hybrid_delta_rule"] = str(linear["delta_rule"])
    if linear.get("enable_text_state") is not None:
        fields["hybrid_enable_text_state"] = bool(linear["enable_text_state"])
    short_conv = linear.get("short_conv")
    if isinstance(short_conv, dict) and "targets" in short_conv:
        fields["hybrid_short_conv_targets"] = list(short_conv["targets"])
    elif isinstance(short_conv, (list, tuple)):
        fields["hybrid_short_conv_targets"] = list(short_conv)
    return fields


def stamp_hybrid_config(config: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    """Copy a transformer ``config.json`` and set the hybrid architecture flags."""
    out = dict(config)
    out.update(fields)
    return out
