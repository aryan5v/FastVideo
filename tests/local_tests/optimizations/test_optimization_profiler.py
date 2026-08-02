# SPDX-License-Identifier: Apache-2.0
"""CPU tests for optimization profiler export accounting."""

from __future__ import annotations

from dataclasses import dataclass

from fastvideo.optimization.profiler import _rows


@dataclass
class _Event:
    key: str
    device_type: object
    device_time_total: float
    self_device_time_total: float
    cpu_time_total: float = 0.0
    count: int = 1
    input_shapes: tuple[tuple[int, ...], ...] = ()


class _DeviceType:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return f"DeviceType.{self.name}"


class _Profiler:
    def key_averages(self, *, group_by_input_shape: bool):
        assert group_by_input_shape
        return (
            _Event("aten::mul", _DeviceType("CPU"), 80.0, 80.0),
            _Event("mul_cuda_kernel", _DeviceType("CUDA"), 80.0, 80.0),
        )


def test_rows_exclude_duplicate_cuda_activity_records():
    rows = _rows(_Profiler())

    assert [row["name"] for row in rows] == ["aten::mul"]
    assert rows[0]["device_type"] == "cpu"
    assert rows[0]["self_cuda_time_us"] == 80.0


def test_rows_export_fx_region_ranges_with_parent_scope():
    class _RegionProfiler:
        def key_averages(self, *, group_by_input_shape: bool):
            assert group_by_input_shape
            return (
                _Event(
                    "motionkernel::transformer.blocks.0123abcd",
                    _DeviceType("CPU"),
                    42.0,
                    0.0,
                ),
            )

    rows = _rows(_RegionProfiler())

    assert rows[0]["name"] == "transformer.blocks.0123abcd"
    assert rows[0]["parent_module"] == "transformer.blocks"
    assert rows[0]["scope_kind"] == "fx_region"
