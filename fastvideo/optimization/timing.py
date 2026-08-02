# SPDX-License-Identifier: Apache-2.0
"""Opt-in per-phase timing for the artifact dispatch path.

Attributing dispatch overhead needs the cost broken down *in situ*: the same
module, the same live tensors, the same stream. A micro-harness that rebuilds
the region separately measures something else.

Everything here is inert unless ``FASTVIDEO_OPTIMIZATION_ARTIFACT_TIMING`` is
set. When off, :func:`phase` returns a shared no-op context manager and the
recording functions return immediately, so the hot path pays one module-level
boolean check.

CUDA is asynchronous, so wall time around a launch measures launch cost, not
kernel cost. Set the env var to ``sync`` to synchronize around each phase and
attribute device time as well; plain ``1`` leaves the stream alone and measures
only host-side cost, which is what dispatch overhead actually is.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

__all__ = ["ENABLED", "SHADOW", "SYNCHRONIZE", "note", "phase", "record", "snapshot", "write_report"]

_SETTING = os.getenv("FASTVIDEO_OPTIMIZATION_ARTIFACT_TIMING", "")
ENABLED = bool(_SETTING)
SYNCHRONIZE = _SETTING.lower() in {"sync", "cuda", "device", "shadow"}
#: Also run the native forward on every dispatched call and time it, so the
#: candidate path can be compared against the path it replaced on identical
#: inputs. Diagnostic only: it roughly doubles the region's cost and the
#: native result is discarded.
SHADOW = _SETTING.lower() == "shadow"

_totals: dict[str, float] = defaultdict(float)
_counts: dict[str, int] = defaultdict(int)
#: Free-form structured notes, e.g. why an acceleration path declined. Recorded
#: here rather than only logged so the reason survives whatever the host's log
#: level happens to be -- a silently declined fast path looks identical to one
#: that was never attempted.
_notes: dict[str, int] = defaultdict(int)


class _NoOp:
    """Shared, allocation-free stand-in used when timing is disabled."""

    __slots__ = ()

    def __enter__(self) -> "_NoOp":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


_NO_OP = _NoOp()


def _sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - timing must never break a run
        pass


@contextmanager
def _timed(name: str):
    if SYNCHRONIZE:
        _sync()
    start = time.perf_counter()
    try:
        yield
    finally:
        if SYNCHRONIZE:
            _sync()
        _totals[name] += time.perf_counter() - start
        _counts[name] += 1


def phase(name: str):
    """Time one named phase, or do nothing when timing is disabled."""
    if not ENABLED:
        return _NO_OP
    return _timed(name)


def note(message: str) -> None:
    """Count one structured observation. Always on, so it survives log config."""
    _notes[str(message)[:200]] += 1


def record(name: str, seconds: float) -> None:
    """Add an externally measured duration to a named phase."""
    if not ENABLED:
        return
    _totals[name] += seconds
    _counts[name] += 1


def snapshot() -> dict[str, Any]:
    """Per-phase totals, call counts and per-call means, slowest first."""
    phases = {
        name: {
            "total_seconds": round(total, 6),
            "calls": _counts[name],
            "mean_ms": round(total / _counts[name] * 1000.0, 4) if _counts[name] else 0.0,
        }
        for name, total in _totals.items()
    }
    return {
        "timing_schema_version": 1,
        "synchronized": SYNCHRONIZE,
        "phases": dict(
            sorted(phases.items(), key=lambda item: -item[1]["total_seconds"])
        ),
        "notes": dict(sorted(_notes.items(), key=lambda item: -item[1])),
    }


def write_report(path: str | Path) -> Path | None:
    """Write the timing snapshot, returning ``None`` if that failed."""
    if not ENABLED:
        return None
    output = Path(path).expanduser()
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(snapshot(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        return None
    return output
