# SPDX-License-Identifier: Apache-2.0

import pytest

from fastvideo.benchmarks.mlx_fastwan_bench import (
    ALLOWED_MODES,
    _mode_to_dtype_quant,
    _parse_list,
)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("fp16", ("fp16", None)),
        ("bf16", ("bf16", None)),
        ("int8", ("fp16", "int8")),
        ("int4", ("fp16", "int4")),
        ("mxfp8", ("fp16", "mxfp8")),
        ("mxfp4", ("fp16", "mxfp4")),
        ("nvfp4", ("fp16", "nvfp4")),
    ],
)
def test_benchmark_modes_map_to_runtime_quantization(mode: str, expected: tuple[str, str | None]) -> None:
    assert mode in ALLOWED_MODES
    assert _mode_to_dtype_quant(mode) == expected


def test_benchmark_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError, match="Unsupported modes"):
        _parse_list("fp16,not_a_mode", ALLOWED_MODES, "modes")

