# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for worker-side optimization profiler export."""

from __future__ import annotations

import json

import pytest

from fastvideo.optimization import profiler as optimization_profiler


class _Event:
    key = "aten::add"
    count = 3
    device_time_total = 12.5
    self_device_time_total = 9.5
    cpu_time_total = 20.0
    input_shapes = [[1, 4], [1, 4]]


class _FakeProfiler:

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def key_averages(self, *, group_by_input_shape):
        assert group_by_input_shape is True
        return [_Event()]


def test_worker_profile_exports_metadata_only(tmp_path, monkeypatch):
    output = tmp_path / "profile.json"
    monkeypatch.setenv(
        "FASTVIDEO_OPTIMIZATION_PROFILE_OUTPUT",
        str(output),
    )
    monkeypatch.setenv("FASTVIDEO_OPTIMIZATION_PROFILE_SKIP_RUNS", "1")
    monkeypatch.setenv("FASTVIDEO_OPTIMIZATION_PROFILE_WORKLOAD_ID", "unit")
    monkeypatch.setenv("FASTVIDEO_OPTIMIZATION_PROFILE_MODEL_ID", "model")
    monkeypatch.setenv("FASTVIDEO_OPTIMIZATION_PROFILE_TASK", "t2v")
    monkeypatch.setattr(
        optimization_profiler.torch.profiler,
        "profile",
        lambda **kwargs: _FakeProfiler(),
    )
    monkeypatch.setattr(
        optimization_profiler.torch.cuda,
        "is_available",
        lambda: False,
    )

    with optimization_profiler.optimization_profile(0):
        pass
    assert not output.exists()

    with optimization_profiler.optimization_profile(1):
        pass
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["workload"]["workload_id"] == "unit"
    assert payload["workload"]["model_id"] == "model"
    assert payload["total_cuda_time_us"] == 9.5
    assert payload["rows"][0]["input_shapes"] == [[1, 4], [1, 4]]
    assert "values" not in payload["rows"][0]
def test_capture_detached_when_pre_profiler_step_fails(tmp_path, monkeypatch):
    """A failure before the profiler window opens must not leak hooks."""
    output = tmp_path / "profile.json"
    monkeypatch.setenv("FASTVIDEO_OPTIMIZATION_PROFILE_OUTPUT", str(output))
    monkeypatch.setenv("FASTVIDEO_OPTIMIZATION_PROFILE_SKIP_RUNS", "0")

    events = {"finalize": 0}

    class _Session:
        tracer = "auto"

        def finalize(self):
            events["finalize"] += 1
            return {
                "capture": {},
                "regions": [],
                "graph_breaks": [],
                "unsupported": [],
            }

        def detach(self):
            pass

    monkeypatch.setattr(
        optimization_profiler,
        "_start_capture",
        lambda modules: _Session(),
    )
    monkeypatch.setattr(
        optimization_profiler.torch.cuda,
        "is_available",
        lambda: True,
    )

    def _boom():
        raise RuntimeError("synthetic synchronize failure")

    monkeypatch.setattr(optimization_profiler.torch.cuda, "synchronize", _boom)

    with (
        pytest.raises(RuntimeError, match="synthetic synchronize failure"),
        optimization_profiler.optimization_profile(0),
    ):
        pass

    assert events["finalize"] == 1
    assert not output.exists()
