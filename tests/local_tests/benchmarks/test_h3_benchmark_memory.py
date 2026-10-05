"""Benchmark capacity sampling supports both pod cgroup versions, without CUDA."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/benchmarks/minimax_h3_4090/bench_pod.py"
SPEC = importlib.util.spec_from_file_location("h3_benchmark_memory", SCRIPT)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class OneSample:
    def __init__(self):
        self.finished = False

    def is_set(self):
        return self.finished

    def wait(self, _seconds):
        self.finished = True


def sample(root, monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", SimpleNamespace(
        nvmlInit=lambda: None, nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda _index: None,
        nvmlDeviceGetMemoryInfo=lambda _handle: SimpleNamespace(used=8192)))
    peak = BENCH.HostMemoryPeak(root)
    peak.stop = OneSample()
    peak._sample()
    return peak


@pytest.mark.parametrize("version", [1, 2])
def test_cgroup_memory_excludes_cached_files_from_anonymous_peak(tmp_path, monkeypatch, version):
    if version == 2:
        (tmp_path / "memory.current").write_text("4096")
        (tmp_path / "memory.stat").write_text("anon 1024\nfile 3072\n")
    else:
        root = tmp_path / "memory"
        root.mkdir()
        (root / "memory.usage_in_bytes").write_text("4096")
        (root / "memory.stat").write_text("rss 512\ntotal_rss 1024\ntotal_cache 3072\n")
    peak = sample(tmp_path, monkeypatch)
    assert peak.host_error is None
    assert peak.peak_bytes == 4096
    assert peak.peak_anon_bytes == 1024
    assert peak.peak_gpu_bytes == 8192


def test_missing_host_counters_do_not_disable_gpu_capacity_measurement(tmp_path, monkeypatch):
    peak = sample(tmp_path, monkeypatch)
    assert peak.host_error is not None
    assert peak.peak_gpu_bytes == 8192


def test_malformed_host_counters_do_not_report_a_measured_zero(tmp_path, monkeypatch):
    (tmp_path / "memory.current").write_text("invalid")
    peak = sample(tmp_path, monkeypatch)
    assert peak.host_error is not None
    assert peak.peak_gpu_bytes == 8192
    assert peak.metrics()["peak_host_cgroup_gib"] is None
    assert peak.metrics()["peak_host_anon_gib"] is None


@pytest.mark.parametrize("fail,once", [(True, False), (False, False), (False, True)])
def test_generation_receipts_preserve_memory_and_once_never_adds_clips(tmp_path, monkeypatch, fail, once):
    model = tmp_path / "model"
    model.mkdir()
    (model / "fastvideo_inference.json").write_text("{}")
    prompts = tmp_path / "prompts.json"
    prompts.write_text('{"ceramics": "test"}')
    shutdown = []
    requests = []

    class FailingGenerator:
        def generate(self, _request):
            requests.append(_request)
            if fail:
                raise RuntimeError("CUDA out of memory")

        def shutdown(self):
            shutdown.append(True)

    class Memory:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def metrics(self):
            return {"peak_gpu_used_gib": 11.9, "peak_host_cgroup_gib": 28.3,
                    "peak_host_anon_gib": 20.1, "host_memory_error": None}

    fake_torch = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda="test"),
                                 cuda=SimpleNamespace(get_device_name=lambda _index: "test GPU"))
    fake_video = SimpleNamespace(VideoGenerator=SimpleNamespace(from_config=lambda _config: FailingGenerator()))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "fastvideo", fake_video)
    monkeypatch.setattr(BENCH, "HostMemoryPeak", Memory)
    monkeypatch.setattr(BENCH.subprocess, "check_output", lambda *_args, **_kwargs: "test GPU")
    argv = [str(SCRIPT), "failure", str(model), "fp8", "--prompt-file", str(prompts),
            "--output-root", str(tmp_path / "outputs")]
    if once:
        argv.append("--once")
    monkeypatch.setattr(sys, "argv", argv)
    if fail:
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            BENCH.main()
    else:
        BENCH.main()
    raw = json.loads((tmp_path / "outputs/failure/results.json").read_text())
    if fail:
        assert raw["runs"] == []
        assert len(raw["failed_runs"]) == 1
        failed = raw["failed_runs"][0]
        assert failed["peak_gpu_used_gib"] == 11.9
        assert failed["peak_host_cgroup_gib"] == 28.3
        assert failed["warmup"]
        assert "CUDA out of memory" in failed["error"]
    else:
        assert len(requests) == len(raw["runs"]) == (1 if once else 3)
        assert raw["runs"][0]["warmup"]
        assert all(run["peak_gpu_used_gib"] == 11.9 for run in raw["runs"])
        if once:
            assert "median_e2e_s" not in raw
            assert "mean_e2e_s" not in raw
        else:
            assert not any(run["warmup"] for run in raw["runs"][1:])
            assert "mean_e2e_s" in raw
    assert shutdown == [True]
