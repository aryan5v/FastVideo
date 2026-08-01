# SPDX-License-Identifier: Apache-2.0
"""Worker-side metadata-only profiler export for optimization discovery."""

from __future__ import annotations

import contextlib
import json
import os
import platform
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from fastvideo import envs
from fastvideo.logger import init_logger
from fastvideo.optimization.fx_capture import CAPTURE_SCHEMA_VERSION

logger = init_logger(__name__)

_REGION_RANGE_PREFIX = "motionkernel::"


def _event_number(event: Any, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _output_path(template: str) -> Path:
    rank = os.getenv("RANK", os.getenv("LOCAL_RANK", "0"))
    rendered = template.replace("<pid>", str(os.getpid())).replace("<rank>", rank)
    path = Path(rendered).expanduser()
    if "<rank>" not in template and "<pid>" not in template and rank != "0":
        path = path.with_name(f"{path.stem}.rank-{rank}{path.suffix}")
    return path


def _device_type(event: Any) -> str:
    """Normalize torch profiler device enums without depending on internals."""
    value = getattr(event, "device_type", None)
    if value is None:
        return "unknown"
    text = str(value).rsplit(".", maxsplit=1)[-1].lower()
    return text


def _rows(profiler: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in profiler.key_averages(group_by_input_shape=True):
        device_type = _device_type(event)
        # CUDA activity rows describe the same kernels already attributed to
        # their CPU-side operator rows. Mixing both categories double-counts
        # device time and makes nested/custom-op scopes look independently
        # optimizable. Unknown is retained for compatibility with profiler
        # implementations that do not expose device_type.
        if device_type == "cuda":
            continue
        event_name = str(event.key)
        row: dict[str, Any] = {
            "name":
            event_name,
            "calls":
            int(event.count),
            "cuda_time_us":
            _event_number(
                event,
                "device_time_total",
                "cuda_time_total",
            ),
            "self_cuda_time_us":
            _event_number(
                event,
                "self_device_time_total",
                "self_cuda_time_total",
            ),
            "cpu_time_us":
            _event_number(event, "cpu_time_total"),
            "device_type":
            device_type,
            "input_shapes":
            [list(shape) for shape in (getattr(event, "input_shapes", None) or []) if isinstance(shape, list | tuple)],
        }
        if event_name.startswith(_REGION_RANGE_PREFIX):
            region_name = event_name.removeprefix(_REGION_RANGE_PREFIX)
            row["name"] = region_name
            row["parent_module"] = region_name.rsplit(".", maxsplit=1)[0]
            row["scope_kind"] = "fx_region"
        rows.append(row)
    return rows


def _write_export(profiler: Any, output: Path, capture: dict[str, Any] | None = None) -> None:
    from fastvideo.version import __version__

    rows = _rows(profiler)
    total_cuda_time_us = sum(max(float(row["self_cuda_time_us"]), 0.0) for row in rows)
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        environment.update({
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(index),
            "gpu_capability": ".".join(map(str, torch.cuda.get_device_capability(index))),
        })
    payload = {
        "schema_version": 1,
        "producer": {
            "name": "fastvideo",
            "version": __version__
        },
        "workload": {
            "workload_id": envs.FASTVIDEO_OPTIMIZATION_PROFILE_WORKLOAD_ID,
            "model_id": envs.FASTVIDEO_OPTIMIZATION_PROFILE_MODEL_ID,
            "task": envs.FASTVIDEO_OPTIMIZATION_PROFILE_TASK,
        },
        "environment": environment,
        "total_cuda_time_us": total_cuda_time_us,
        "rows": rows,
    }
    if capture is not None:
        # Additive, optional keys: readers that only understand ``rows`` are
        # unaffected, and the capture block carries its own schema version.
        payload.update(capture)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def _start_capture(modules: dict[str, Any] | None) -> Any:
    """Attach FX capture hooks, or return None if disabled/unavailable."""
    if not envs.FASTVIDEO_OPTIMIZATION_PROFILE_CAPTURE_FX:
        return None
    try:
        from fastvideo.optimization.fx_capture import FXCaptureSession

        session = FXCaptureSession(
            tracer=envs.FASTVIDEO_OPTIMIZATION_PROFILE_FX_TRACER,
            max_scopes=envs.FASTVIDEO_OPTIMIZATION_PROFILE_FX_MAX_SCOPES,
            max_shape_variants=envs.FASTVIDEO_OPTIMIZATION_PROFILE_FX_MAX_SHAPES,
        )
        hooked = session.attach_modules(modules)
    except Exception:
        # Capture is best-effort telemetry; generation must be unaffected.
        logger.warning("FX capture could not be started; continuing without it", exc_info=True)
        return None
    logger.info("FX capture attached to %d module(s)", hooked)
    return session


def _finish_capture(session: Any) -> dict[str, Any] | None:
    """Detach hooks and build the capture payload, recording any failure."""
    if session is None:
        return None
    try:
        return session.finalize()
    except Exception as exc:
        logger.warning("FX capture finalize failed; exporting failure record", exc_info=True)
        try:
            session.detach()
        except Exception:
            logger.debug("FX capture detach failed", exc_info=True)
        return {
            "capture": {
                "capture_schema_version": CAPTURE_SCHEMA_VERSION,
                # Exception text can include source snippets or argument reprs.
                "errors": [f"finalize_failed:{type(exc).__name__}"],
            },
            "regions": [],
            "graph_breaks": [],
            "unsupported": [],
        }


@contextlib.contextmanager
def optimization_profile(call_index: int, modules: dict[str, Any] | None = None) -> Iterator[None]:
    """Profile exactly the configured pipeline call inside the GPU worker.

    ``modules`` is the pipeline's module mapping. When FX capture is requested
    it is scanned generically for repeated block stacks — no architecture is
    named here or in the capture module.
    """
    template = envs.FASTVIDEO_OPTIMIZATION_PROFILE_OUTPUT
    target_call = envs.FASTVIDEO_OPTIMIZATION_PROFILE_SKIP_RUNS
    if not template or call_index != target_call:
        yield
        return

    session = _start_capture(modules)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize()
    try:
        with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
        ) as profiler:
            yield
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    finally:
        # Tracing happens strictly after the profiler window closes so captured
        # graphs never land in the exported timings.
        capture = _finish_capture(session)
    _write_export(profiler, _output_path(template), capture)
