# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the workload-driven generation launcher parser."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_LAUNCHER = os.path.join(
    _REPO_ROOT,
    "examples",
    "inference",
    "optimizations",
    "generation_launcher.py",
)


def _load_launcher():
    """Load the launcher script by path without permanently mutating sys.path."""
    # Use a unique module name and no sys.path rewrite — file location is enough.
    spec = importlib.util.spec_from_file_location(
        "generation_launcher_under_test", _LAUNCHER
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def launcher():
    return _load_launcher()


def _minimal_workload(**overrides):
    payload = {
        "schema_version": 1,
        "workload_id": "unit-t2v",
        "model": {"model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"},
        "task": "t2v",
        "prompt": "a raccoon in sunflowers",
        "sampling": {
            "height": 480,
            "width": 832,
            "num_frames": 49,
            "num_inference_steps": 4,
            "guidance_scale": 5.0,
            "seed": 1024,
            "dtype": "bfloat16",
        },
        "runtime": {
            "num_gpus": 1,
            "text_encoder_cpu_offload": True,
        },
        "measurement": {"warmups": 1, "runs": 2, "save_frames": True},
        "mode_env": {
            "native": {"FASTVIDEO_WAN_FUSIONS": "0"},
            "optimized": {"FASTVIDEO_WAN_FUSIONS": "1"},
            "fused": {
                "FASTVIDEO_WAN_FUSIONS": "1",
                "EXTRA_FUSE_FLAG": "1",
            },
        },
    }
    payload.update(overrides)
    return payload


def test_load_workload_dict_roundtrip(tmp_path: Path, launcher):
    path = tmp_path / "w.json"
    path.write_text(json.dumps(_minimal_workload()), encoding="utf-8")
    loaded = launcher.load_workload_dict(path)
    assert loaded["workload_id"] == "unit-t2v"
    request = launcher.build_request(loaded, base_dir=tmp_path)
    assert request["sampling"]["height"] == 480
    assert "dtype" not in request["sampling"]
    model_id, kwargs = launcher.build_generator_kwargs(loaded)
    assert model_id.endswith("1.3B-Diffusers")
    assert kwargs["num_gpus"] == 1
    assert kwargs["text_encoder_cpu_offload"] is True


def test_prompt_file(tmp_path: Path, launcher):
    prompt = tmp_path / "p.txt"
    prompt.write_text("from file\n", encoding="utf-8")
    payload = _minimal_workload()
    del payload["prompt"]
    payload["prompt_file"] = "p.txt"
    path = tmp_path / "w.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = launcher.load_workload_dict(path)
    assert launcher.resolve_prompt(loaded, base_dir=tmp_path) == "from file"


def test_rejects_bad_schema(tmp_path: Path, launcher):
    path = tmp_path / "w.json"
    path.write_text(
        json.dumps(_minimal_workload(schema_version=99)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="schema_version"):
        launcher.load_workload_dict(path)


def test_dry_run_cli_does_not_mutate_environ(tmp_path: Path, launcher, capsys, monkeypatch):
    path = tmp_path / "w.json"
    path.write_text(json.dumps(_minimal_workload()), encoding="utf-8")
    monkeypatch.delenv("FASTVIDEO_WAN_FUSIONS", raising=False)
    code = launcher.main(
        [
            "--workload",
            str(path),
            "--mode",
            "native",
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    assert code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["workload_id"] == "unit-t2v"
    assert plan["mode"] == "native"
    assert plan["mode_env"]["FASTVIDEO_WAN_FUSIONS"] == "0"
    assert "FASTVIDEO_WAN_FUSIONS" not in os.environ


def test_resolve_mode_env_prefers_exact_fused_key(launcher):
    workload = _minimal_workload()
    resolved = launcher.resolve_mode_env(workload, "fused")
    assert resolved["FASTVIDEO_WAN_FUSIONS"] == "1"
    assert resolved["EXTRA_FUSE_FLAG"] == "1"


def test_candidate_keeps_dedicated_result_identity(launcher):
    assert launcher._normalize_mode("optimized") == "optimized"
    assert launcher._normalize_mode("fused") == "optimized"
    assert launcher._normalize_mode("candidate") == "candidate"


def test_profiler_rows_are_metadata_only(launcher):
    class _Event:
        key = "aten::add"
        count = 3
        device_time_total = 12.5
        self_device_time_total = 9.5
        cpu_time_total = 20.0
        input_shapes = [[1, 4], [1, 4]]

    class _Profiler:
        def key_averages(self, *, group_by_input_shape):
            assert group_by_input_shape is True
            return [_Event()]

    rows = launcher.profiler_rows(_Profiler())
    assert rows == [
        {
            "name": "aten::add",
            "calls": 3,
            "cuda_time_us": 12.5,
            "self_cuda_time_us": 9.5,
            "cpu_time_us": 20.0,
            "input_shapes": [[1, 4], [1, 4]],
        }
    ]
    assert "values" not in rows[0]


def test_non_dry_cli_runs_generation(tmp_path: Path, launcher, capsys, monkeypatch):
    path = tmp_path / "w.json"
    path.write_text(json.dumps(_minimal_workload()), encoding="utf-8")
    calls = []

    def fake_run_generation(**kwargs):
        calls.append(kwargs)
        return {"status": "ok", "mode": "native"}

    monkeypatch.setattr(launcher, "run_generation", fake_run_generation)
    profile = tmp_path / "profile.json"
    code = launcher.main(
        [
            "--workload",
            str(path),
            "--mode",
            "native",
            "--output-dir",
            str(tmp_path / "out"),
            "--profile-output",
            str(profile),
        ]
    )
    assert code == 0
    assert len(calls) == 1
    assert calls[0]["profile_output"] == profile
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
