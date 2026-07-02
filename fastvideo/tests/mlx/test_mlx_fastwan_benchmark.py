# SPDX-License-Identifier: Apache-2.0

import pytest

from fastvideo.benchmarks.mlx_fastwan_bench import (
    ALLOWED_MODES,
    _gib_to_bytes,
    _html_grid,
    _load_prompt_cases,
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


def test_load_prompt_cases_from_plain_text(tmp_path) -> None:
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("\n# comment\nA fox runs through a forest.\nA raccoon walks in sunflowers.\n")

    cases = _load_prompt_cases("unused", prompt_file)

    assert [case.id for case in cases] == ["prompt-001", "prompt-002"]
    assert [case.prompt for case in cases] == [
        "A fox runs through a forest.",
        "A raccoon walks in sunflowers.",
    ]


def test_load_prompt_cases_from_jsonl(tmp_path) -> None:
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text('{"id": "Fox Forest", "prompt": "A fox runs."}\n{"name": "clock", "caption": "A clock burns."}\n')

    cases = _load_prompt_cases("unused", prompt_file)

    assert [case.id for case in cases] == ["fox-forest", "clock"]
    assert [case.prompt for case in cases] == ["A fox runs.", "A clock burns."]


def test_load_builtin_prompt_set() -> None:
    cases = _load_prompt_cases("unused", None, "motion7")
    assert len(cases) == 7
    assert cases[0].id == "beach-sunset"


def test_gib_to_bytes_rejects_non_positive_values() -> None:
    assert _gib_to_bytes(None) is None
    assert _gib_to_bytes(1.5) == int(1.5 * 1024**3)
    with pytest.raises(ValueError, match="positive"):
        _gib_to_bytes(0)


def test_html_grid_includes_video_and_sync_controls() -> None:
    rendered = _html_grid([
        {
            "prompt_id": "fox",
            "prompt": "A fox runs.",
            "mode": "int8",
            "decoder": "taehv",
            "status": "ok",
            "video_path": "fox/video_int8_taehv.mp4",
            "total_s": 12.3,
            "denoise_s": 10.0,
            "decode_s": 1.0,
            "peak_gib": 4.5,
        }
    ])

    assert "Restart + play all" in rendered
    assert "fox/video_int8_taehv.mp4" in rendered
    assert "A fox runs." in rendered
