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
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from fastvideo import envs

__all__ = [
    "ENABLED",
    "SHADOW",
    "SYNCHRONIZE",
    "note",
    "phase",
    "record",
    "reset",
    "snapshot",
    "write_report",
]

# One normalization rule for every operator-facing flag: TIMING=0, =false,
# =no and =off all mean off. `bool(_SETTING)` previously enabled timing for
# any non-empty value, including "0".
_SETTING = envs.env_flag_normalized("FASTVIDEO_OPTIMIZATION_ARTIFACT_TIMING")
ENABLED = _SETTING not in envs.ENV_FLAG_OFF_VALUES
# The richer modes are meaningless when timing is off, so they are gated on it
# rather than parsed independently.
SYNCHRONIZE = ENABLED and _SETTING in {"sync", "cuda", "device", "shadow"}
#: Also run the native forward on every dispatched call and time it, so the
#: candidate path can be compared against the path it replaced on identical
#: inputs. Diagnostic only: it roughly doubles the region's cost and the
#: native result is discarded.
SHADOW = ENABLED and _SETTING == "shadow"
# Shadow mode performs an extra native forward per dispatched call. A stateful
# forward that consumes RNG, updates a cache, or mutates module state can
# therefore change generation results even though the shadow output is
# discarded. Never enable it during parity or generation-result comparisons.

_totals: dict[str, float] = defaultdict(float)
_counts: dict[str, int] = defaultdict(int)
#: Free-form structured notes, e.g. why an acceleration path declined. Recorded
#: here rather than only logged so the reason survives whatever the host's log
#: level happens to be -- a silently declined fast path looks identical to one
#: that was never attempted.
_notes: dict[str, int] = defaultdict(int)
_MAX_DISTINCT_NOTES = 128
_OVERFLOW_NOTE = "other_notes_omitted"


class _NoOp:
    """Shared, allocation-free stand-in used when timing is disabled."""

    __slots__ = ()

    def __enter__(self) -> _NoOp:
        return self

    def __exit__(self, *exc: Any) -> Literal[False]:
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
    key = str(message)[:200]
    if key in _notes or len(_notes) < _MAX_DISTINCT_NOTES:
        _notes[key] += 1
    else:
        _notes[_OVERFLOW_NOTE] += 1


def record(name: str, seconds: float) -> None:
    """Add an externally measured duration to a named phase."""
    if not ENABLED:
        return
    _totals[name] += seconds
    _counts[name] += 1


def reset() -> None:
    """Clear all accumulated timing state.

    The counters are process-global, so two dispatch sessions in one process --
    serving two models in sequence, say -- would otherwise contribute to the
    same totals and produce a report attributable to neither. Call this when
    starting a session whose measurements must stand alone.
    """
    _totals.clear()
    _counts.clear()
    _notes.clear()


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
        "phases": dict(sorted(phases.items(), key=lambda item: -item[1]["total_seconds"])),
        "notes": dict(sorted(_notes.items(), key=lambda item: -item[1])),
    }


def write_report(path: str | Path) -> Path | None:
    """Write the timing snapshot, returning ``None`` if that failed."""
    if not ENABLED:
        return None
    output = Path(path).expanduser()
    temporary: Path | None = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output.parent,
                prefix=f".{output.name}.",
                suffix=".tmp",
                delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(snapshot(), indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
        temporary = None
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        return None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output
