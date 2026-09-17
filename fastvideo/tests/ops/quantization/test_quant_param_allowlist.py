# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract tests for the ``fsdp_load`` zero-init allowlist.

A quantization config registers scale tensors that no checkpoint carries
(``AbsMaxFP8`` -> ``scale_weight`` / ``scale_input``; see
``fastvideo/layers/quantization/absmax_fp8.py``). ``fsdp_load`` rejects every
model parameter that is absent from the incoming state dict, so enabling
``engine.quantization.transformer_quant`` used to abort the load with::

    Unsupported new parameter: transformer_blocks.0.attn.to_out.scale_input

These tests pin both halves of that contract, on CPU and without any
distributed process group:

* quant scale parameters are accepted and zero-initialized,
* a parameter that is genuinely missing from the checkpoint still raises, so
  the allowlist is not a blanket exemption.

See ``docs/quantization/loader_quant_params.md`` before adding a quantization
config that registers new parameters.
"""

import unittest

import torch
import torch.nn as nn

from fastvideo.layers.quantization.absmax_fp8 import AbsMaxFP8LinearMethod
from fastvideo.models.loader.fsdp_load import (
    ALLOWED_NEW_PARAM_PATTERNS,
    is_allowed_new_param,
    load_model_from_full_model_state_dict,
)

DTYPE = torch.float32
REPORTED_FQN = "transformer_blocks.0.attn.to_out.scale_input"


class _QuantToOut(nn.Module):
    """A module holding an AbsMaxFP8 linear's parameters under their real names."""

    def __init__(self, in_features: int = 3, out_features: int = 2) -> None:
        super().__init__()
        AbsMaxFP8LinearMethod().create_weights(
            self,
            input_size_per_partition=in_features,
            output_partition_sizes=[out_features],
            input_size=in_features,
            output_size=out_features,
            params_dtype=DTYPE,
        )


class _Block(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attn()


class _Attn(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.to_out = _QuantToOut()


def _model() -> nn.Module:
    model = nn.Module()
    model.add_module("transformer_blocks", nn.ModuleList([_Block()]))
    return model


def _load(model: nn.Module, checkpoint: dict[str, torch.Tensor]):
    return load_model_from_full_model_state_dict(
        model,
        ((name, tensor) for name, tensor in checkpoint.items()),
        device=torch.device("cpu"),
        param_dtype=DTYPE,
        strict=False,
        cpu_offload=False,
        param_names_mapping=lambda name: (name, None, None),
        training_mode=False,
    )


class TestQuantParamAllowlist(unittest.TestCase):

    def test_reported_fqn_is_allowed(self):
        self.assertTrue(is_allowed_new_param(REPORTED_FQN))

    def test_quant_scale_params_are_zero_initialized(self):
        model = _model()
        checkpoint = {"transformer_blocks.0.attn.to_out.weight": torch.ones(2, 3, dtype=DTYPE)}
        _load(model, checkpoint)

        to_out = model.transformer_blocks[0].attn.to_out
        self.assertEqual(to_out.scale_weight.shape, (1, ))
        self.assertEqual(to_out.scale_input.shape, (1, ))
        self.assertTrue(torch.equal(to_out.scale_weight, torch.zeros(1, dtype=DTYPE)))
        self.assertTrue(torch.equal(to_out.scale_input, torch.zeros(1, dtype=DTYPE)))
        self.assertTrue(torch.equal(to_out.weight, torch.ones(2, 3, dtype=DTYPE)))

    def test_missing_real_weight_still_raises(self):
        model = _model()
        with self.assertRaisesRegex(ValueError, "is not supported"):
            _load(model, {})

    def test_bare_scale_param_is_not_admitted(self):
        self.assertFalse(is_allowed_new_param("transformer_blocks.0.attn.to_out.scale"))

    def test_every_absmax_fp8_registered_param_is_allowed(self):
        to_out = _QuantToOut()
        new_params = [name for name, _ in to_out.named_parameters() if name != "weight"]
        self.assertEqual(sorted(new_params), ["scale_input", "scale_weight"])
        for name in new_params:
            fqn = f"transformer_blocks.0.attn.to_out.{name}"
            self.assertTrue(is_allowed_new_param(fqn), f"{fqn} is not in {ALLOWED_NEW_PARAM_PATTERNS}")

    def test_existing_attention_patterns_still_allowed(self):
        for name in ("transformer_blocks.0.attn.to_gate_compress.weight", "blocks.0.attn1.attn_impl.proj_l.weight"):
            self.assertTrue(is_allowed_new_param(name))


if __name__ == "__main__":
    unittest.main()
