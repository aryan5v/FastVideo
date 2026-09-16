# SPDX-License-Identifier: Apache-2.0
"""Guard against drift between the two copies of the zero-init allowlist.

``fsdp_load`` owns the canonical ``ALLOWED_NEW_PARAM_PATTERNS`` and uses it to
decide which model parameters may be zero-initialized when a checkpoint does not
supply them. ``shard_cache`` keeps a literal mirror of that tuple because
``fsdp_load`` imports ``shard_cache`` -- importing back would be circular.

A mismatch does not fail a run; it silently keeps the weight shard cache cold
for any quantization config that registers new parameters (for example
``AbsMaxFP8``, whose ``scale_weight`` / ``scale_input`` appear in the model but
never in the checkpoint). That is an invisible performance regression, so this
test pins the two tuples together.
"""

import unittest

from fastvideo.models.loader.fsdp_load import ALLOWED_NEW_PARAM_PATTERNS
from fastvideo.models.loader.shard_cache import _ALLOWED_NEW_PARAM_PATTERNS


class TestAllowlistMirror(unittest.TestCase):
    """``shard_cache`` must mirror ``fsdp_load``'s allowlist exactly."""

    def test_mirror_matches_canonical_source(self):
        self.assertEqual(
            _ALLOWED_NEW_PARAM_PATTERNS,
            ALLOWED_NEW_PARAM_PATTERNS,
            "shard_cache._ALLOWED_NEW_PARAM_PATTERNS has drifted from "
            "fsdp_load.ALLOWED_NEW_PARAM_PATTERNS. Update the mirror in "
            "fastvideo/models/loader/shard_cache.py to match.",
        )

    def test_quant_scale_params_are_shared(self):
        """The specific names a quantizing loader depends on are present in both."""
        for name in ("scale_weight", "scale_input"):
            with self.subTest(param=name):
                self.assertIn(name, ALLOWED_NEW_PARAM_PATTERNS)
                self.assertIn(name, _ALLOWED_NEW_PARAM_PATTERNS)


if __name__ == "__main__":
    unittest.main()
