# SPDX-License-Identifier: Apache-2.0
"""Metadata-only capture for AutoKernel-compatible optimization campaigns."""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import platform
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

_SCHEMA_VERSION = 1
_ACTIVE_SESSION: contextvars.ContextVar[_CaptureSession | None] = (contextvars.ContextVar(
    "fastvideo_optimization_session", default=None))


def _tensor_signature(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device_type": tensor.device.type,
        "requires_grad": tensor.requires_grad,
    }


def _signature_key(inputs: Sequence[dict[str, Any]]) -> str:
    return json.dumps(inputs, sort_keys=True, separators=(",", ":"))


def _safe_identity(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _output_path(template: str) -> Path:
    rank = os.getenv("RANK", os.getenv("LOCAL_RANK", "0"))
    rendered = template.replace("<pid>", str(os.getpid())).replace("<rank>", rank)
    path = Path(rendered).expanduser()
    if "<rank>" not in template and "<pid>" not in template and rank != "0":
        path = path.with_name(f"{path.stem}.rank-{rank}{path.suffix}")
    return path


def _environment_identity() -> dict[str, str]:
    device = "cpu"
    capability = ""
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        device = torch.cuda.get_device_name(index)
        capability = ".".join(map(str, torch.cuda.get_device_capability(index)))
    hardware = f"{device}:{capability or platform.machine()}"
    software = (f"torch={torch.__version__};cuda={torch.version.cuda or 'none'};"
                f"python={platform.python_version()}")
    return {
        "hardware_profile_id": hardware,
        "software_profile_id": software,
    }


@dataclass
class _Observation:
    inputs: list[dict[str, Any]]
    count: int = 0
    total_device_time_us: float = 0.0
    tags: tuple[str, ...] = ()


@dataclass
class _Target:
    name: str
    operation: str
    kind: str
    spec_locator: str | None
    requires_backward: bool
    attributes: dict[str, Any]
    calls: int = 0
    total_device_time_us: float = 0.0
    observations: dict[str, _Observation] = field(default_factory=dict)
    pending_timers: list[tuple[_Observation, _Timer]] = field(default_factory=list)


@dataclass
class _Timer:
    started_ns: int
    start_event: torch.cuda.Event | None
    end_event: torch.cuda.Event | None
    ended_ns: int | None = None


class _CaptureSession:

    def __init__(
        self,
        output: Path,
        *,
        workload_id: str,
        model_id: str,
        task: str,
        variant_id: str,
    ) -> None:
        self.output = output
        self.workload = {
            "workload_id": workload_id,
            "model_id": model_id,
            "task": task,
            "variant_id": variant_id,
            "profile_scope": "fastvideo_pipeline",
        }
        self.targets: dict[str, _Target] = {}
        self.root_timer: _Timer | None = None
        self.total_profiled_device_time_us = 0.0

    @staticmethod
    def _start_timer(use_cuda: bool) -> _Timer:
        start_event = None
        end_event = None
        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        return _Timer(time.perf_counter_ns(), start_event, end_event)

    @staticmethod
    def _end_timer(timer: _Timer) -> None:
        timer.ended_ns = time.perf_counter_ns()
        if timer.end_event is not None:
            timer.end_event.record()

    @staticmethod
    def _elapsed_us(timer: _Timer) -> float:
        if timer.start_event is not None and timer.end_event is not None:
            return float(timer.start_event.elapsed_time(timer.end_event) * 1000)
        ended_ns = timer.ended_ns or time.perf_counter_ns()
        return float((ended_ns - timer.started_ns) / 1000)

    def start_workload(self) -> None:
        self.root_timer = self._start_timer(torch.cuda.is_available())

    def finish_workload(self) -> None:
        if self.root_timer is None:
            return
        self._end_timer(self.root_timer)
        if self.root_timer.end_event is not None:
            torch.cuda.synchronize()
        self.total_profiled_device_time_us = self._elapsed_us(self.root_timer)

    def start_target(
        self,
        *,
        name: str,
        operation: str,
        tensors: Mapping[str, torch.Tensor],
        kind: str,
        spec_locator: str | None,
        attributes: Mapping[str, Any],
        tags: Sequence[str],
    ) -> tuple[_Target, _Observation, _Timer]:
        target = self.targets.get(name)
        requires_backward = torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors.values())
        if target is None:
            target = _Target(
                name=name,
                operation=operation,
                kind=kind,
                spec_locator=spec_locator,
                requires_backward=requires_backward,
                attributes=dict(attributes),
            )
            self.targets[name] = target
        elif (target.operation != operation or target.kind != kind or target.spec_locator != spec_locator):
            raise RuntimeError(f"Optimization target {name!r} changed identity during capture")
        target.requires_backward |= requires_backward
        inputs = [_tensor_signature(tensor_name, tensor) for tensor_name, tensor in tensors.items()]
        key = _signature_key(inputs)
        observation = target.observations.get(key)
        if observation is None:
            observation = _Observation(inputs=inputs, tags=tuple(tags))
            target.observations[key] = observation
        target.calls += 1
        observation.count += 1
        use_cuda = bool(tensors) and all(tensor.is_cuda for tensor in tensors.values())
        return target, observation, self._start_timer(use_cuda)

    def finish_target(
        self,
        target: _Target,
        observation: _Observation,
        timer: _Timer,
    ) -> None:
        self._end_timer(timer)
        if timer.end_event is not None:
            # Workload finalization synchronizes once before reading all events.
            target.pending_timers.append((observation, timer))
        else:
            elapsed_us = self._elapsed_us(timer)
            target.total_device_time_us += elapsed_us
            observation.total_device_time_us += elapsed_us

    def write(self) -> None:
        targets = []
        for target in self.targets.values():
            for observation, timer in target.pending_timers:
                elapsed_us = self._elapsed_us(timer)
                target.total_device_time_us += elapsed_us
                observation.total_device_time_us += elapsed_us
            observations = []
            for index, observation in enumerate(target.observations.values(), start=1):
                observations.append({
                    "name": f"shape_{index}",
                    "count": observation.count,
                    "total_device_time_us": observation.total_device_time_us,
                    "inputs": observation.inputs,
                    "tags": list(observation.tags),
                })
            targets.append({
                "name": target.name,
                "operation": target.operation,
                "kind": target.kind,
                "spec_locator": target.spec_locator,
                "total_device_time_us": target.total_device_time_us,
                "self_device_time_us": target.total_device_time_us,
                "calls": target.calls,
                "requires_backward": target.requires_backward,
                "observations": observations,
                "attributes": target.attributes,
            })
        if not targets:
            return
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "producer": {
                "name": "fastvideo",
                "version": _fastvideo_version(),
            },
            "workload": self.workload,
            "environment": _environment_identity(),
            "total_profiled_device_time_us": self.total_profiled_device_time_us,
            "targets": targets,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.output.parent,
                prefix=f".{self.output.name}.",
                suffix=".tmp",
                delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(self.output)


def _fastvideo_version() -> str:
    from fastvideo.version import __version__

    return __version__


@contextlib.contextmanager
def optimization_workload(
    *,
    workload_id: str,
    model_id: str,
    task: str = "video_generation",
    variant_id: str = "default",
) -> Iterator[None]:
    """Capture one pipeline run when FASTVIDEO_OPTIMIZATION_CAPTURE is set."""
    template = os.getenv("FASTVIDEO_OPTIMIZATION_CAPTURE", "").strip()
    if not template or _ACTIVE_SESSION.get() is not None:
        yield
        return
    session = _CaptureSession(
        _output_path(template),
        workload_id=_safe_identity(workload_id, "fastvideo-workload"),
        model_id=_safe_identity(model_id, "unknown-model"),
        task=_safe_identity(task, "video_generation"),
        variant_id=_safe_identity(variant_id, "default"),
    )
    token = _ACTIVE_SESSION.set(session)
    session.start_workload()
    try:
        yield
    finally:
        session.finish_workload()
        session.write()
        _ACTIVE_SESSION.reset(token)


@contextlib.contextmanager
def optimization_target(
        *,
        name: str,
        operation: str,
        tensors: Mapping[str, torch.Tensor],
        kind: str = "fusion",
        spec_locator: str | None = None,
        attributes: Mapping[str, Any] | None = None,
        tags: Sequence[str] = (),
) -> Iterator[None]:
    """Record shape/layout metadata and aggregate timing for one candidate."""
    session = _ACTIVE_SESSION.get()
    if session is None:
        yield
        return
    target, observation, timer = session.start_target(
        name=name,
        operation=operation,
        tensors=tensors,
        kind=kind,
        spec_locator=spec_locator,
        attributes=attributes or {},
        tags=tags,
    )
    with torch.profiler.record_function(f"fastvideo.optimization::{name}"):
        try:
            yield
        finally:
            session.finish_target(target, observation, timer)
