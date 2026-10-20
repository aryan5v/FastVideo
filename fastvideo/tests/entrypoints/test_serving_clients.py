# SPDX-License-Identifier: Apache-2.0
"""Run shipped serving clients against a local fake HTTP server."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
AUDIO_CLIENT = ROOT / "examples" / "serving" / "clients" / "audio.py"


@pytest.fixture
def fake_audio_server():
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep pytest output readable
            pass

        def _json(self, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            requests.append(json.loads(self.rfile.read(length)))
            self._json({"id": "audio_test", "status": "completed", "media_type": "audio/wav", "error": None})

        def do_GET(self) -> None:
            if self.path == "/v1/videos/audio_test":
                self._json({"id": "audio_test", "status": "completed", "media_type": "audio/wav", "error": None})
                return
            if self.path == "/v1/videos/audio_test/content":
                body = b"RIFF\x00\x00\x00\x00WAVEfmt "
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_audio_client_uses_media_type_and_downloads_wav(fake_audio_server, tmp_path: Path) -> None:
    base_url, requests = fake_audio_server
    env = {
        **os.environ,
        "FASTVIDEO_BASE_URL": base_url,
        "FASTVIDEO_MODEL": "stable-audio",
        "FASTVIDEO_SECONDS": "6",
        "FASTVIDEO_PROMPT": "rain on a metal roof",
    }
    completed = subprocess.run(
        [sys.executable, str(AUDIO_CLIENT)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert requests == [{
        "model": "stable-audio",
        "prompt": "rain on a metal roof",
        "seconds": 6,
    }]
    assert (tmp_path / "audio_test.wav").read_bytes().startswith(b"RIFF")
    assert "Saved audio_test.wav" in completed.stdout
