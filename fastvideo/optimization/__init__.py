# SPDX-License-Identifier: Apache-2.0
"""Model-independent optimization discovery and dispatch helpers."""

from fastvideo.optimization.dispatch import (attach_graph_dispatch, detach_graph_dispatch)
from fastvideo.optimization.profiler import optimization_profile

__all__ = [
    "attach_graph_dispatch",
    "detach_graph_dispatch",
    "optimization_profile",
]
