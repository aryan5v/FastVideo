# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any


def _load_runner() -> Any:
    path = Path(__file__).resolve().parents[3] / "scripts/fasth3_sprint/run_baseline_matrix.py"
    spec = importlib.util.spec_from_file_location("test_fasth3_sprint_baseline_matrix_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_request_binds_current_prompt(monkeypatch: Any, tmp_path: Path) -> None:
    runner = _load_runner()
    inference_args = argparse.Namespace(prompt="first prompt")
    calls: list[tuple[str, Path, int]] = []

    def fake_build_request(args: argparse.Namespace, output_path: Path, seed: int) -> object:
        calls.append((args.prompt, output_path, seed))
        return object()

    monkeypatch.setattr(runner.basic_fasth3, "build_request", fake_build_request)
    output_path = tmp_path / "second.mp4"

    runner._build_prompt_request(inference_args, output_path, 7, "second prompt")

    assert calls == [("second prompt", output_path, 7)]
