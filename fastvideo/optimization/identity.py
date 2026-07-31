# SPDX-License-Identifier: Apache-2.0
"""Graph identity for a single observed module invocation.

Dispatch has to answer one question about a live call: *which captured region
is this?* The answer must be the same value the producer recorded, or no
artifact would ever match.

That is why this module reuses :mod:`fastvideo.optimization.fx_capture`'s own
helpers rather than re-deriving tensor metadata, shape keys or fingerprints.
Those helpers are the canonical implementation; a second implementation here
would be one refactor away from silently disagreeing with the exports it is
supposed to match, and a fingerprint that disagrees is indistinguishable from
"no artifact available".
"""

from __future__ import annotations

from fastvideo.optimization.fx_capture import (
    capture_invocation_identity as graph_identity,
    invocation_input_signatures as input_signatures,
    invocation_output_signatures as output_signatures,
    invocation_shape_key as shape_key_for,
)

__all__ = [
    "graph_identity",
    "input_signatures",
    "output_signatures",
    "shape_key_for",
]
