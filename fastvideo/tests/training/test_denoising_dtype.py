# SPDX-License-Identifier: Apache-2.0
"""Denoising stages must target the transformer's *compute* dtype.

Regression for the DGX training-validation crash: under FSDP/HSDP mixed
precision the raw parameters are fp32 masters cast to bf16 per-forward, so
``next(parameters()).dtype`` said fp32 — leaving validation latents fp32 and
disabling autocast against bf16-cast weights
(``Input type (float) and bias type (c10::BFloat16) should be the same``).
"""

from __future__ import annotations

import torch

from fastvideo.pipelines.stages.denoising import transformer_compute_dtype
from fastvideo.utils import _mixed_precision_state, set_mixed_precision_policy


def _with_policy_cleared():
    previous = getattr(_mixed_precision_state, "state", None)
    if previous is not None:
        del _mixed_precision_state.state
    return previous


def _restore_policy(previous) -> None:
    if previous is not None:
        _mixed_precision_state.state = previous
    elif hasattr(_mixed_precision_state, "state"):
        del _mixed_precision_state.state


def test_policy_param_dtype_wins_over_fp32_masters() -> None:
    previous = _with_policy_cleared()
    try:
        model = torch.nn.Linear(8, 8, dtype=torch.float32)  # fp32 masters, like FSDP storage
        set_mixed_precision_policy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        assert transformer_compute_dtype(model) == torch.bfloat16
    finally:
        _restore_policy(previous)


def test_falls_back_to_parameter_sniffing_without_policy() -> None:
    previous = _with_policy_cleared()
    try:
        assert transformer_compute_dtype(torch.nn.Linear(8, 8).to(torch.float16)) == torch.float16
        assert transformer_compute_dtype(torch.nn.Linear(8, 8)) == torch.float32
    finally:
        _restore_policy(previous)


def test_fp32_policy_defers_to_the_model() -> None:
    previous = _with_policy_cleared()
    try:
        set_mixed_precision_policy(param_dtype=torch.float32, reduce_dtype=torch.float32)
        assert transformer_compute_dtype(torch.nn.Linear(8, 8).to(torch.bfloat16)) == torch.bfloat16
    finally:
        _restore_policy(previous)


def test_unwraps_module_attribute() -> None:
    previous = _with_policy_cleared()
    try:
        wrapper = torch.nn.Module()
        wrapper.module = torch.nn.Linear(8, 8).to(torch.float16)
        assert transformer_compute_dtype(wrapper) == torch.float16
    finally:
        _restore_policy(previous)
