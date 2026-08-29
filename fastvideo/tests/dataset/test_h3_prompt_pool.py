# SPDX-License-Identifier: Apache-2.0
"""Prompt-source audit tests for the FastH3 sprint."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.fasth3_sprint.prepare_h3_prompt_pool import prepare_prompt_pool


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_prompt_pool_selects_valid_short_audio_rows(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    _write_jsonl(existing / "manifest_rank00000.jsonl", [{"caption": "Existing prompt"}])
    t2va = tmp_path / "t2va.jsonl"
    common = {
        "validation": {
            "status": "passed"
        },
        "runtime_config": {
            "generate_audio": True
        },
    }
    _write_jsonl(
        t2va,
        [
            {
                **common,
                "id": "short",
                "prompt_compiled": "A short synchronized scene",
                "sampling": {
                    "dimensions": {
                        "duration_bucket": "five_to_six_seconds",
                        "audio_profile": "dialogue_ambience",
                    }
                },
            },
            {
                **common,
                "id": "long",
                "prompt_compiled": "A longer synchronized scene",
                "sampling": {
                    "dimensions": {
                        "duration_bucket": "eleven_to_fifteen_seconds",
                        "audio_profile": "minimal_audio",
                    }
                },
            },
        ],
    )
    h3ext = tmp_path / "h3ext.jsonl"
    _write_jsonl(h3ext, [{"sample_id": "extra", "prompt": "A distinct extended scene"}])

    receipt = prepare_prompt_pool(t2va, h3ext, existing, tmp_path / "out")

    assert receipt["unique_ids"] == 3
    assert receipt["unique_normalized_prompts"] == 3
    assert receipt["overlap_with_existing"] == {"t2va": 0, "h3ext": 0}
    selection = (tmp_path / "out" / "short_teacher_cache_selection.jsonl").read_text().splitlines()
    assert len(selection) == 1
    assert json.loads(selection[0])["id"] == "short"
