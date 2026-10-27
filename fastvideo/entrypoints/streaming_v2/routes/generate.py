# SPDX-License-Identifier: Apache-2.0
"""``POST /v1/streamv2/generate``: generate one segment and stream it back as SSE."""
from __future__ import annotations

import asyncio
import base64
import io
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from PIL import Image

from fastvideo.api.compat import explicit_request_updates, legacy_generate_call_to_request
from fastvideo.api.schema import GenerationRequest
from fastvideo.entrypoints.streaming_v2.messages import (
    DoneEvent,
    ErrorEvent,
    GenerateRequest,
    LastFrameEvent,
    LoraSelection,
    VideoChunkEvent,
    VideoStartEvent,
    format_sse,
)
from fastvideo.entrypoints.streaming_v2.video_stream import encode_fmp4, video_mime
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3Reference, validate_references

if TYPE_CHECKING:
    from fastvideo.entrypoints.streaming_v2.server import ServerState

logger = init_logger(__name__)
router = APIRouter()


@router.post("/v1/streamv2/generate")
async def generate_segment(raw_request: Request) -> Response:
    """Validate the request, then stream the segment; an invalid request gets HTTP 400 before any event."""
    received = time.perf_counter()
    state: ServerState = raw_request.app.state.streaming_v2
    try:
        body = GenerateRequest.model_validate(await raw_request.json())
        generation_request = build_generation_request(body, state.default_request)
        # An adapter this server does not allow, or a strength outside its range, is the request's fault.
        state.validate_lora(body.lora)
    except ValueError as error:  # malformed JSON, schema violations, undecodable images, reference limits, LoRA
        error_body = ErrorEvent(code="invalid_request", message=str(error))
        return JSONResponse(status_code=400, content=error_body.model_dump())
    return StreamingResponse(
        stream_segment(state, generation_request, body.return_last_frame, received, body.lora),
        media_type="text/event-stream",
    )


def build_generation_request(body: GenerateRequest, default_request: GenerationRequest | None) -> GenerationRequest:
    """Merge the operator's explicit defaults with the body; the body's prompt, size, frames, seed, and images win.

    Denoising steps and other sampling knobs come from the operator defaults or
    the model preset. Frames and audio are returned in memory, not saved. A body
    without reference images is a text-to-video request, which the served model
    answers when it has no reference transformer; the images are checked as
    references only when the request carries them.
    """
    kwargs: dict[str, Any] = explicit_request_updates(default_request) if default_request is not None else {}
    kwargs.update(width=body.width, height=body.height, num_frames=body.num_frames)
    if body.seed is not None:
        kwargs["seed"] = body.seed
    images = [_decode_image(index, data) for index, data in enumerate(body.reference_images, start=1)]
    kwargs["references"] = (validate_references([MiniMaxH3Reference(source=image) for image in images]) if images else [])
    kwargs.update(save_video=False, return_frames=True)
    return legacy_generate_call_to_request(body.prompt, None, legacy_kwargs=kwargs)


async def stream_segment(
    state: ServerState,
    generation_request: GenerationRequest,
    return_last_frame: bool,
    received: float,
    lora: LoraSelection | None = None,
) -> AsyncIterator[str]:
    """Generate one segment and yield its SSE frames: optional ``last_frame``, ``video_start``, chunks, ``done``.

    A ``ValueError`` from generation is the request's fault and ends the stream
    with an ``invalid_request`` error; any other failure, including one while
    encoding, ends it with ``generation_failed``. A LoRA adapter that fails to
    load is the request's fault too, so it ends the same way.
    """
    try:
        result, generation_ms = await state.generate(generation_request, lora)
    except ValueError as error:
        logger.warning("streaming_v2 rejected a request during generation: %s", error)
        yield format_sse(ErrorEvent(code="invalid_request", message=str(error)))
        return
    except Exception as error:
        logger.exception("streaming_v2 generation failed")
        yield format_sse(ErrorEvent(code="generation_failed", message=str(error)))
        return

    try:
        frames, audio, sample_rate = _unpack_result(result)
        if return_last_frame:
            png = await asyncio.to_thread(_encode_png, frames[-1])
            yield format_sse(LastFrameEvent(data=base64.b64encode(png).decode("ascii")))
        yield format_sse(VideoStartEvent(mime=video_mime(has_audio=audio is not None)))
        async for chunk in encode_fmp4(frames, generation_request.sampling.fps, audio, sample_rate):
            yield format_sse(VideoChunkEvent(data=base64.b64encode(chunk).decode("ascii")))
        timings = {"generation_ms": generation_ms, "e2e_latency_ms": (time.perf_counter() - received) * 1000.0}
        yield format_sse(DoneEvent(timings=timings))
    except Exception as error:
        logger.exception("streaming_v2 failed to stream a generated segment")
        yield format_sse(ErrorEvent(code="generation_failed", message=str(error)))


def _decode_image(index: int, data: str) -> Image.Image:
    """Decode one base64 reference image; raises ``ValueError`` when it is not base64 or not a readable image."""
    try:
        image = Image.open(io.BytesIO(base64.b64decode(data, validate=True)))
        image.load()
    except (ValueError, OSError) as error:  # binascii.Error is a ValueError; PIL raises OSError subclasses
        raise ValueError(
            f"reference_images[{index - 1}] (Picture {index}) is not a decodable image: {error}") from error
    return image


def _unpack_result(result: Any) -> tuple[list[np.ndarray], Any | None, int | None]:
    """Take the decoded frames, audio, and audio sample rate of the request's single result."""
    if isinstance(result, list):
        if len(result) != 1:
            raise RuntimeError(f"Expected one generation result, got {len(result)}.")
        result = result[0]
    if not result.frames:
        raise RuntimeError("The generator returned no frames.")
    return list(result.frames), result.audio, result.audio_sample_rate


def _encode_png(frame: np.ndarray) -> bytes:
    """Encode one HxWx3 uint8 RGB frame as PNG, favoring speed over size."""
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()
