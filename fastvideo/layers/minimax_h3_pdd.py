# SPDX-License-Identifier: Apache-2.0
"""Dense channel-major PDD heads, adapted from FastVideo H3 V23 / FastGen.

Oracle: FastVideo-h3-v23 bf16199660d9, layers/pdd.py. This scoped version
supports H3's channel-major projections, grid32 and explicit per-call windows.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from fastvideo.layers.linear import ReplicatedLinear


def pdd_sigmas(device: torch.device | str, shift: float) -> torch.Tensor:
    base = torch.linspace(.999, 0, 33, dtype=torch.float64, device=device)
    return base * shift * .999 / (base * (shift - 1) + .999)


def integrate_heads(state: torch.Tensor, heads: torch.Tensor, sigma: torch.Tensor, start: int,
                    end: int) -> torch.Tensor:
    """Heads [window,B,...] predict noise-minus-clean; integrate in FP32."""
    if not 0 <= start <= end <= 32 or end - start > len(heads):
        raise ValueError("Invalid PDD integration window")
    delta = (sigma[start + 1:end + 1] - sigma[start:end]).float()
    delta = delta.reshape((-1, ) + (1, ) * state.ndim)
    return state.float() + (heads[:end - start].float() * delta).sum(0)


def project_heads(x: torch.Tensor,
                  weight: torch.Tensor,
                  bias: torch.Tensor | None,
                  start: int,
                  end: int,
                  weights: torch.Tensor | None = None) -> torch.Tensor:
    if not 0 <= start < end <= 32 or weight.shape[0] % 32:
        raise ValueError("Expected grid32 channel-major PDD parameters and valid window")
    w = weight.unflatten(0, (32, -1))[start:end]
    b = None if bias is None else bias.unflatten(0, (32, -1))[start:end]
    if weights is not None:
        if weights.shape != (end - start, ) or not torch.isfinite(weights).all() or weights.sum() == 0:
            raise ValueError("Invalid PDD fusion coefficients")
        coefficients = (weights / weights.sum()).float()
        w = torch.einsum('n,noi->oi', coefficients, w.float()).to(weight.dtype)
        b = None if b is None else torch.einsum('n,no->o', coefficients, b.float()).to(bias.dtype)
    else:
        w = w.flatten(0, 1)
        b = None if b is None else b.flatten()
    return F.linear(x, w, b)


class H3PDDLinear(ReplicatedLinear):
    """Widened weights are converted offline before FSDP loading."""

    def forward(self,
                x: torch.Tensor,
                *,
                start: int,
                end: int,
                weights: torch.Tensor | None = None) -> tuple[torch.Tensor, None]:
        return project_heads(x, self.weight, self.bias, start, end, weights), None
