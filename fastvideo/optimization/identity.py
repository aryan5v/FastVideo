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

from typing import Any

from torch import nn

from fastvideo.optimization.fx_capture import (
    FXCaptureSession,
    _input_metas,
    _output_metas,
    _shape_key,
    _Scope,
)


def input_signatures(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Layout metadata for every tensor reaching a module's forward."""
    return _input_metas(args, kwargs)


def output_signatures(output: Any) -> list[dict[str, Any]]:
    """Layout metadata for every tensor a module's forward returned."""
    return _output_metas(output)


def shape_key_for(metas: list[dict[str, Any]]) -> str:
    """The stable key identifying one observed shape variant."""
    return _shape_key(metas)


def graph_identity(
    module: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    output: Any,
    *,
    scope: str,
    tracer: str = "symbolic",
) -> dict[str, Any]:
    """Trace one observed invocation and return its captured region record.

    This is the capture pipeline applied to a single call, so the resulting
    ``fingerprint`` is produced by exactly the code that produced the
    fingerprints in an exported profile.

    Unlike :meth:`FXCaptureSession.finalize`, a failure is raised rather than
    recorded: the caller needs a decision, not a report.
    """
    session = FXCaptureSession(tracer=tracer, max_scopes=1, max_shape_variants=1)
    record = _Scope(
        scope=scope,
        class_name=type(module).__name__,
        module=module,
    )
    session._scopes[scope] = record
    session._observe(record, args, kwargs, output)
    regions = session._regions_for(record)
    if not regions:
        reasons = [str(item.get("reason", "")) for item in session._graph_breaks]
        raise RuntimeError(reasons[0] if reasons else "trace_produced_no_region")
    return regions[0]
