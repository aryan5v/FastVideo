"""Benchmark capacity sampling supports both pod cgroup versions, without CUDA."""
import importlib.util
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
