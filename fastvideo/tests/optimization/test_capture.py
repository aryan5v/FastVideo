# SPDX-License-Identifier: Apache-2.0
"""Tests for metadata-only optimization campaign capture."""

from __future__ import annotations

import json

import torch

from fastvideo.optimization import optimization_target, optimization_workload


def test_capture_is_disabled_without_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("FASTVIDEO_OPTIMIZATION_CAPTURE", raising=False)
    output = tmp_path / "campaign.json"
    with optimization_workload(workload_id="wan", model_id="Wan"):
        with optimization_target(
            name="wan.norm",
            operation="wan_norm",
            tensors={"x": torch.ones(2, 3)},
        ):
            pass
    assert not output.exists()


def test_capture_writes_autokernel_campaign(monkeypatch, tmp_path):
    output = tmp_path / "campaign.json"
    monkeypatch.setenv("FASTVIDEO_OPTIMIZATION_CAPTURE", str(output))
    x = torch.ones(2, 3, requires_grad=True)
    with optimization_workload(
        workload_id="wan-t2v",
        model_id="Wan2.1-T2V-1.3B",
        variant_id="bf16",
    ):
        for _ in range(2):
            with optimization_target(
                name="wan.norm",
                operation="wan_norm",
                tensors={"x": x},
                spec_locator="models/wan.py:SPEC",
                attributes={"model_family": "wan"},
                tags=("denoise",),
            ):
                x + 1

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["workload"]["workload_id"] == "wan-t2v"
    assert payload["total_profiled_device_time_us"] > 0
    assert payload["targets"][0]["calls"] == 2
    assert payload["targets"][0]["requires_backward"] is True
    assert payload["targets"][0]["observations"][0]["count"] == 2
    assert payload["targets"][0]["observations"][0]["inputs"] == [{
        "name": "x",
        "shape": [2, 3],
        "stride": [3, 1],
        "dtype": "float32",
        "device_type": "cpu",
        "requires_grad": True,
    }]
    assert "values" not in json.dumps(payload)


def test_capture_expands_rank_and_pid_placeholders(monkeypatch, tmp_path):
    template = tmp_path / "campaign-<rank>-<pid>.json"
    monkeypatch.setenv("FASTVIDEO_OPTIMIZATION_CAPTURE", str(template))
    monkeypatch.setenv("RANK", "3")
    with optimization_workload(workload_id="wan", model_id="Wan"):
        with optimization_target(
            name="wan.norm",
            operation="wan_norm",
            tensors={"x": torch.ones(1)},
        ):
            pass
    outputs = list(tmp_path.glob("campaign-3-*.json"))
    assert len(outputs) == 1
