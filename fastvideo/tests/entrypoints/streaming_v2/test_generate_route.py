# SPDX-License-Identifier: Apache-2.0
"""``POST /v1/streamv2/generate`` with a fake generator and a fake encoder.

CPU-only: no model weights and no ffmpeg (``test_video_stream.py`` covers the
real encoder).
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from fastvideo.api.results import GenerationResult
from fastvideo.api.schema import GenerationRequest
from fastvideo.entrypoints.streaming_v2.routes import generate as generate_route
from fastvideo.entrypoints.streaming_v2.server import ServerState, create_app
from fastvideo.entrypoints.streaming_v2.video_stream import video_mime

_FAKE_CHUNKS = [b"init-segment", b"fragment-1", b"fragment-2"]


def _png_base64(width: int = 32, height: int = 24) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 120, 200)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _body(**overrides: Any) -> dict[str, Any]:
    body = {
        "prompt": "a lighthouse at dusk",
        "reference_images": [_png_base64()],
        "width": 64,
        "height": 48,
        "num_frames": 4,
        "seed": 7,
        "return_last_frame": True,
    }
    body.update(overrides)
    return body


class _FakeGenerator:
    """Returns four distinct frames plus audio and records every request."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        frames = [np.full((48, 64, 3), 40 * index, dtype=np.uint8) for index in range(4)]
        return GenerationResult(frames=frames, audio=np.zeros(1600, dtype=np.float32), audio_sample_rate=16000)


async def _fake_encode(frames: list[np.ndarray], fps: int, audio: Any = None, sample_rate: Any = None) -> AsyncIterator[bytes]:
    for chunk in _FAKE_CHUNKS:
        yield chunk


async def _failing_encode(frames: list[np.ndarray], fps: int, audio: Any = None, sample_rate: Any = None) -> AsyncIterator[bytes]:
    yield _FAKE_CHUNKS[0]
    raise RuntimeError("ffmpeg failed: boom")


def _events(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse an SSE body into (event name, JSON data) pairs."""
    events = []
    for frame in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def fake_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate_route, "encode_fmp4", _fake_encode)


def test_generate_renders_a_text_request_without_references(fake_encoder: None) -> None:
    generator = _FakeGenerator()
    with TestClient(create_app(generator)) as client:
        response = client.post("/v1/streamv2/generate", json=_body(reference_images=[]))

    assert response.status_code == 200
    assert [name for name, _ in _events(response.text)] == ["last_frame", "video_start", "video_chunk", "video_chunk", "video_chunk", "done"]
    assert generator.requests[0].inputs.references == []


def test_generate_streams_last_frame_video_and_done_in_order(fake_encoder: None) -> None:
    generator = _FakeGenerator()
    with TestClient(create_app(generator)) as client:
        response = client.post("/v1/streamv2/generate", json=_body())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response.text)
    assert [name for name, _ in events] == ["last_frame", "video_start", "video_chunk", "video_chunk", "video_chunk", "done"]

    last_frame = Image.open(io.BytesIO(base64.b64decode(events[0][1]["data"])))
    assert last_frame.format == "PNG"
    assert np.array_equal(np.asarray(last_frame), np.full((48, 64, 3), 120, dtype=np.uint8))
    assert events[1][1] == {"mime": video_mime(has_audio=True)}
    assert b"".join(base64.b64decode(data["data"]) for name, data in events if name == "video_chunk") == b"".join(_FAKE_CHUNKS)
    assert set(events[-1][1]["timings"]) == {"generation_ms", "e2e_latency_ms"}

    request = generator.requests[0]
    assert request.prompt == "a lighthouse at dusk"
    assert (request.sampling.width, request.sampling.height, request.sampling.num_frames, request.sampling.seed) == (64, 48, 4, 7)
    assert len(request.inputs.references) == 1 and request.inputs.references[0].media_type == "image"
    assert request.output.save_video is False and request.output.return_frames is True


def test_generate_omits_last_frame_unless_requested(fake_encoder: None) -> None:
    with TestClient(create_app(_FakeGenerator())) as client:
        response = client.post("/v1/streamv2/generate", json=_body(return_last_frame=False))

    assert [name for name, _ in _events(response.text)][:2] == ["video_start", "video_chunk"]


@pytest.mark.parametrize(
    "content",
    [
        b"not json",
        json.dumps(_body(width=None)).encode(),
        json.dumps({key: value for key, value in _body().items() if key != "num_frames"}).encode(),
        json.dumps(_body(prompt="")).encode(),
        json.dumps(_body(unknown_field=1)).encode(),
        json.dumps(_body(reference_images=["%%% not base64 %%%"])).encode(),
        json.dumps(_body(reference_images=[base64.b64encode(b"not an image").decode()])).encode(),
        json.dumps(_body(reference_images=[_png_base64()] * 10)).encode(),
    ],
)
def test_generate_rejects_invalid_requests_with_http_400(content: bytes) -> None:
    generator = _FakeGenerator()
    with TestClient(create_app(generator)) as client:
        response = client.post("/v1/streamv2/generate", content=content, headers={"content-type": "application/json"})

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_request"
    assert response.json()["message"]
    assert generator.requests == []


@pytest.mark.parametrize(
    ("error", "code"),
    [(ValueError("reference image outside 1:4..4:1"), "invalid_request"), (RuntimeError("CUDA OOM"), "generation_failed")],
)
def test_generate_maps_generation_failures_to_error_codes(fake_encoder: None, error: Exception, code: str) -> None:
    with TestClient(create_app(_FakeGenerator(error=error))) as client:
        response = client.post("/v1/streamv2/generate", json=_body())

    assert response.status_code == 200
    assert _events(response.text) == [("error", {"code": code, "message": str(error)})]


def test_generate_reports_encoding_failure_after_video_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate_route, "encode_fmp4", _failing_encode)
    with TestClient(create_app(_FakeGenerator())) as client:
        response = client.post("/v1/streamv2/generate", json=_body())

    events = _events(response.text)
    assert [name for name, _ in events] == ["last_frame", "video_start", "video_chunk", "error"]
    assert events[-1][1] == {"code": "generation_failed", "message": "ffmpeg failed: boom"}


class _GatedGenerator:
    """Blocks each ``generate()`` call until the test releases it and records which calls are running."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Semaphore(0)
        self.calls = 0

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        self.started.release()
        self.release.wait(timeout=10)
        return GenerationResult(frames=[np.zeros((8, 8, 3), dtype=np.uint8)])


def test_lock_covers_generation_but_not_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """While request 1 is still streaming, request 2 already generates."""
    streaming = asyncio.Event()
    finish_streaming = asyncio.Event()

    async def blocking_encode(frames: list[np.ndarray], fps: int, audio: Any = None, sample_rate: Any = None) -> AsyncIterator[bytes]:
        streaming.set()
        await finish_streaming.wait()
        yield b"chunk"

    monkeypatch.setattr(generate_route, "encode_fmp4", blocking_encode)
    generator = _GatedGenerator()
    generator.release.set()

    async def scenario() -> None:
        state = ServerState(generator=generator)
        request = GenerationRequest(prompt="p")
        first = generate_route.stream_segment(state, request, False, 0.0)
        assert (await first.__anext__()).startswith("event: video_start")  # request 1 finished generating
        chunk_task = asyncio.ensure_future(first.__anext__())  # request 1 now blocks while streaming
        await asyncio.wait_for(streaming.wait(), timeout=10)
        assert not state.lock.locked()
        second_result, _ = await asyncio.wait_for(state.generate(request), timeout=10)
        assert generator.calls == 2 and second_result.frames is not None
        finish_streaming.set()
        assert (await chunk_task).startswith("event: video_chunk")
        await first.aclose()

    asyncio.run(scenario())


def test_lock_stays_held_until_generation_returns_when_caller_is_cancelled() -> None:
    generator = _GatedGenerator()

    async def scenario() -> None:
        state = ServerState(generator=generator)
        task = asyncio.ensure_future(state.generate(GenerationRequest(prompt="p")))
        await asyncio.to_thread(generator.started.acquire)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state.lock.locked()  # the uninterruptible call still runs
        generator.release.set()
        for _ in range(100):
            if not state.lock.locked():
                break
            await asyncio.sleep(0.05)
        assert not state.lock.locked()

    asyncio.run(scenario())
