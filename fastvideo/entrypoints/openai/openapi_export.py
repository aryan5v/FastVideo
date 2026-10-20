# SPDX-License-Identifier: Apache-2.0
"""Build and check the checked-in OpenAPI document without starting a server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastvideo.entrypoints.openai.api_server import create_app
from fastvideo.fastvideo_args import FastVideoArgs

SPEC_PATH = Path(__file__).resolve().parents[3] / "docs" / "openapi" / "fastvideo-openai.json"
# The schema is route-derived; this public checkpoint only constructs an empty
# FastVideoArgs object. It is never loaded, downloaded, or executed.
SCHEMA_MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def build_openapi() -> dict[str, Any]:
    """Return the normalized schema without entering the app lifespan."""
    args = FastVideoArgs(model_path=SCHEMA_MODEL_PATH)
    app = create_app(args, output_dir="outputs", served_model_name="fastvideo")
    return app.openapi()


def render_openapi(spec: dict[str, Any]) -> str:
    """Stable, reviewable JSON with sorted keys and a trailing newline."""
    return json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def check_openapi(path: Path = SPEC_PATH) -> bool:
    """Return whether the checked-in document matches the current routes."""
    if not path.exists():
        return False
    return path.read_text(encoding="utf-8") == render_openapi(build_openapi())


def write_openapi(path: Path = SPEC_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_openapi(build_openapi()), encoding="utf-8")


__all__ = ["SPEC_PATH", "build_openapi", "check_openapi", "render_openapi", "write_openapi"]
