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


def _rows(profiler: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in profiler.key_averages(group_by_input_shape=True):
        rows.append({
            "name":
            str(event.key),
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
            "input_shapes":
            [list(shape) for shape in (getattr(event, "input_shapes", None) or []) if isinstance(shape, list | tuple)],
        })
    return rows


def _write_export(profiler: Any, output: Path) -> None:
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
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


@contextlib.contextmanager
def optimization_profile(call_index: int) -> Iterator[None]:
    """Profile exactly the configured pipeline call inside the GPU worker."""
    template = envs.FASTVIDEO_OPTIMIZATION_PROFILE_OUTPUT
    target_call = envs.FASTVIDEO_OPTIMIZATION_PROFILE_SKIP_RUNS
    if not template or call_index != target_call:
        yield
        return

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize()
    with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
    ) as profiler:
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    _write_export(profiler, _output_path(template))
