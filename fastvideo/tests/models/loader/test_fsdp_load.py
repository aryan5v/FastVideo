# SPDX-License-Identifier: Apache-2.0
"""Loader contracts that do not require FSDP or a checkpoint download."""

import torch

from fastvideo.models.loader.fsdp_load import load_model_from_full_model_state_dict


class _TinyModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2, 2, device="meta"))


def test_checkpoint_only_vsa_gate_is_skipped_for_dense_model() -> None:
    model = _TinyModel()
    state_dict = iter((
        ("weight", torch.ones(2, 2)),
        ("blocks.0.to_gate_compress.bias", torch.zeros(2)),
    ))

    load_model_from_full_model_state_dict(
        model,
        state_dict,
        device=torch.device("cpu"),
        param_dtype=torch.float32,
        strict=True,
        param_names_mapping=lambda name: (name, None, None),
    )

    assert torch.equal(model.weight, torch.ones(2, 2))
