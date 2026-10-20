# SPDX-License-Identifier: Apache-2.0
"""Normalize the generated FastAPI schema to the HTTP behavior clients see.

FastAPI documents its default 422 validation response even when the server
adapts that failure to the OpenAI-style 400 envelope. This module removes that
stale response, adds the shared error envelope, and documents the video/image
binary responses and the raw ``Request`` video body. It operates on the schema
FastAPI generates from the mounted routes; it is not a second route table.

The generated schema does not load a model or enter the application lifespan.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from fastvideo.entrypoints.openai.protocol import VideoGenerationRequest

ERROR_SCHEMA = "FastVideoError"
ERROR_REF = f"#/components/schemas/{ERROR_SCHEMA}"

_ERROR_RESPONSES = {
    400: "Invalid request",
    404: "Not found",
    500: "Generation failed",
    503: "Generation engine unavailable",
}

_BINARY_RESPONSES: dict[str, dict[str, str]] = {
    "/v1/videos/sync": {
        "video/mp4": "MP4 video",
        "audio/wav": "WAV audio",
    },
    "/v1/videos/{video_id}/content": {
        "video/mp4": "MP4 video",
        "audio/wav": "WAV audio",
    },
    "/v1/images/{image_id}/content": {
        "image/png": "PNG image",
        "image/jpeg": "JPEG image",
        "image/webp": "WebP image",
    },
}

_400_PATHS = {
    "/v1/videos",
    "/v1/videos/sync",
    "/v1/videos/{video_id}/content",
    "/v1/images",
    "/v1/images/generations",
    "/v1/images/edits",
}

_404_PATHS = {
    "/v1/models/{model}",
    "/v1/videos/{video_id}",
    "/v1/videos/{video_id}/content",
    "/v1/images/{image_id}/content",
}

_500_PATHS = {
    "/v1/videos/sync",
    "/v1/images",
    "/v1/images/generations",
    "/v1/images/edits",
}


def _error_response(description: str) -> dict[str, Any]:
    return {
        "description": description,
        "content": {
            "application/json": {
                "schema": {
                    "$ref": ERROR_REF
                }
            }
        },
    }


def _add_error_schema(schema: dict[str, Any]) -> None:
    components = schema.setdefault("components", {})
    schemas = components.setdefault("schemas", {})
    schemas[ERROR_SCHEMA] = {
        "type": "object",
        "required": ["error"],
        "properties": {
            "error": {
                "type": "object",
                "required": ["message", "type", "param", "code"],
                "properties": {
                    "message": {
                        "type": "string"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["invalid_request_error", "server_error"],
                    },
                    "param": {
                        "type": ["string", "null"]
                    },
                    "code": {
                        "type": ["integer", "string", "null"]
                    },
                },
            }
        },
    }


def _add_video_request_schema(schema: dict[str, Any]) -> None:
    components = schema.setdefault("components", {})
    schemas = components.setdefault("schemas", {})
    video_schema = VideoGenerationRequest.model_json_schema(ref_template="#/components/schemas/{model}")
    definitions = video_schema.pop("$defs", {})
    for name, definition in definitions.items():
        schemas.setdefault(name, definition)
    schemas.setdefault("VideoGenerationRequest", video_schema)


def _set_request_body(operation: dict[str, Any]) -> None:
    operation.setdefault(
        "requestBody",
        {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": "#/components/schemas/VideoGenerationRequest"
                    }
                }
            },
        },
    )


def _set_binary_response(operation: dict[str, Any], media_types: dict[str, str]) -> None:
    operation["responses"]["200"] = {
        "description": "Generated media",
        "content": {
            media_type: {
                "schema": {
                    "type": "string",
                    "format": "binary"
                },
            }
            for media_type, description in media_types.items()
        },
    }


def normalize_openapi_contract(schema: dict[str, Any]) -> dict[str, Any]:
    """Apply FastVideo's actual HTTP statuses and media types to ``schema``."""
    _add_error_schema(schema)
    _add_video_request_schema(schema)

    for path, path_item in schema.get("paths", {}).items():
        for method, operation in path_item.items():
            if method not in {"get", "post", "put", "patch", "delete"} or not isinstance(operation, dict):
                continue
            responses = operation.setdefault("responses", {})
            # FastAPI's 422 is never returned by this server; request validation
            # is adapted to the OpenAI-style 400 envelope.
            responses.pop("422", None)
            if path in _400_PATHS:
                responses.setdefault("400", _error_response(_ERROR_RESPONSES[400]))
            if path in _404_PATHS:
                responses.setdefault("404", _error_response(_ERROR_RESPONSES[404]))
            if path in _500_PATHS:
                responses.setdefault("500", _error_response(_ERROR_RESPONSES[500]))
            if path == "/health":
                responses.setdefault("503", _error_response(_ERROR_RESPONSES[503]))
            if method == "post" and path in {"/v1/videos", "/v1/videos/sync"}:
                _set_request_body(operation)
            if path in _BINARY_RESPONSES:
                _set_binary_response(operation, _BINARY_RESPONSES[path])

    schemas = schema.get("components", {}).get("schemas", {})
    schemas.pop("HTTPValidationError", None)
    schemas.pop("ValidationError", None)
    schema.setdefault("info", {})["description"] = (
        "HTTP contract for FastVideo's OpenAI-compatible media routes. This schema "
        "describes the tested FastVideo surface, not full OpenAI API or SDK compatibility. "
        "Model, task and runtime support depends on the loaded serve configuration; see "
        "docs/cookbook/serving-compatibility.md and docs/cookbook/serving-coverage.md. "
        "FastVideo-specific fields include audio_start_in_s, audio_end_in_s, extra_params, "
        "video_reference, image_reference, audio_reference, input_reference, video_path and video_url.")
    return schema


def install_openapi_contract(app: FastAPI) -> None:
    """Make ``GET /openapi.json`` use the normalized schema."""
    original_openapi = app.openapi

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is not None:
            return app.openapi_schema
        schema = normalize_openapi_contract(original_openapi())
        app.openapi_schema = schema
        return schema

    app.openapi = openapi  # type: ignore[method-assign]


__all__ = ["ERROR_REF", "ERROR_SCHEMA", "install_openapi_contract", "normalize_openapi_contract"]
