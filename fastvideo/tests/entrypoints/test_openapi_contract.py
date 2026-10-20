# SPDX-License-Identifier: Apache-2.0
"""The HTTP schema must describe the server's actual, tested behavior."""

from __future__ import annotations

import json

import pytest

from fastvideo.entrypoints.openai.openapi_export import SPEC_PATH, build_openapi, render_openapi
from fastvideo.version import __version__


@pytest.fixture(scope="module")
def spec() -> dict:
    return build_openapi()


def test_openapi_is_generated_from_mounted_routes(spec: dict) -> None:
    assert spec["openapi"] == "3.1.0"
    assert spec["info"]["version"] == __version__
    assert "/v1/videos" in spec["paths"]
    assert "/v1/images/generations" in spec["paths"]
    assert "/playground/" not in spec["paths"]


def test_request_validation_is_documented_as_http_400(spec: dict) -> None:
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue
            responses = operation.get("responses", {})
            assert "422" not in responses, f"{method.upper()} {path} still documents FastAPI's 422"
    schemas = spec["components"]["schemas"]
    assert "HTTPValidationError" not in schemas
    assert "ValidationError" not in schemas
    assert "FastVideoError" in schemas


def test_video_routes_document_the_json_body_and_binary_media(spec: dict) -> None:
    create = spec["paths"]["/v1/videos"]["post"]
    assert create["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith("/VideoGenerationRequest")
    sync = spec["paths"]["/v1/videos/sync"]["post"]
    assert sync["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith("/VideoGenerationRequest")
    assert "video/mp4" in sync["responses"]["200"]["content"]
    assert "audio/wav" in sync["responses"]["200"]["content"]
    download = spec["paths"]["/v1/videos/{video_id}/content"]["get"]
    assert "video/mp4" in download["responses"]["200"]["content"]
    assert "audio/wav" in download["responses"]["200"]["content"]


def test_image_and_error_responses_are_explicit(spec: dict) -> None:
    image = spec["paths"]["/v1/images/generations"]["post"]
    assert "400" in image["responses"]
    assert "500" in image["responses"]
    content = spec["paths"]["/v1/images/{image_id}/content"]["get"]
    for media_type in ("image/png", "image/jpeg", "image/webp"):
        assert media_type in content["responses"]["200"]["content"]
    health = spec["paths"]["/health"]["get"]
    assert "503" in health["responses"]
    error_schema = spec["components"]["schemas"]["FastVideoError"]
    assert error_schema["properties"]["error"]["required"] == ["message", "type", "param", "code"]


def test_checked_in_spec_is_current(spec: dict) -> None:
    assert SPEC_PATH.exists(), (
        f"Missing {SPEC_PATH}; run `python scripts/export_openapi.py` and commit the result."
    )
    expected = render_openapi(spec)
    actual = SPEC_PATH.read_text(encoding="utf-8")
    assert actual == expected, "OpenAPI spec is stale; run `python scripts/export_openapi.py`."


def test_spec_does_not_publish_host_paths_or_credentials(spec: dict) -> None:
    rendered = render_openapi(spec)
    assert "/Users/" not in rendered
    assert "/home/" not in rendered
    assert "127.0.0.1" not in rendered
    assert "Authorization" not in rendered
    spec_json = json.loads(rendered)
    assert spec_json["info"]["description"].startswith("HTTP contract")
