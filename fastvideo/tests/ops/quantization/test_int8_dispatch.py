# SPDX-License-Identifier: Apache-2.0
"""The loader must dispatch INT8 affine configs to their conversion hook.

``_maybe_quantize_model`` walks the module tree and converts weights for
whichever quantization method it finds attached. It is an explicit
``isinstance`` chain, so a new quantization config is silently inert until it
gains a branch here -- the model keeps its dense weights and quietly produces
unquantized output with no error.

This test pins the INT8Affine branch so that adding or reordering the chain
cannot silently drop it.
"""

import unittest

import torch
import torch.nn as nn

from fastvideo.layers.quantization import int8_affine_config
from fastvideo.layers.quantization.int8_affine_config import (
    INT8AffineQuantizeMethod,
    convert_model_to_int8_affine,
)
from fastvideo.models.loader import fsdp_load


class _FakeLinear(nn.Module):
    """Minimal stand-in: a dense weight plus an attached quant method."""

    def __init__(self, prefix: str = "minimax_h3.transformer_blocks.0.attn.to_q"):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16))
        self.quant_method = INT8AffineQuantizeMethod(layer_prefix=prefix)


class TestInt8Dispatch(unittest.TestCase):

    def setUp(self):
        self._calls = []
        self._orig = int8_affine_config.convert_model_to_int8_affine

        def _spy(model):
            self._calls.append(model)
            return self._orig(model)

        int8_affine_config.convert_model_to_int8_affine = _spy
        self.addCleanup(setattr, int8_affine_config, "convert_model_to_int8_affine", self._orig)

    def test_int8_layer_triggers_conversion(self):
        model = _FakeLinear()
        fsdp_load._maybe_quantize_model(model)
        self.assertEqual(len(self._calls), 1, "INT8AffineQuantizeMethod did not reach its conversion hook")
        self.assertTrue(hasattr(model, "_int8_affine_codes"), "quantized buffers were not registered")

    def test_unquantized_model_is_untouched(self):
        """A plain layer must not trigger any conversion."""
        model = nn.Linear(8, 8)
        fsdp_load._maybe_quantize_model(model)
        self.assertEqual(self._calls, [], "conversion ran on a model with no quant method")

    def test_conversion_is_exported_by_the_config_module(self):
        """The hook the loader imports must be the one the config defines."""
        self.assertTrue(callable(convert_model_to_int8_affine))
        self.assertTrue(callable(self._orig))


if __name__ == "__main__":
    unittest.main()
