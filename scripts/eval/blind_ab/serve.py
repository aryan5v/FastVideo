#!/usr/bin/env python3
"""Local blind A/B voting server for generated videos (standard library only).

Usage:
    python scripts/eval/blind_ab/serve.py --bundle <dir> [--votes <file>] [--port 8765]

Open http://localhost:8765/ to vote and http://localhost:8765/results for results.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "blind_ab"  # noqa: A001

from .app_state import AppState, BallotError  # noqa: E402
from .bundle import BundleError, load_bundle  # noqa: E402
from .stats import summary_to_csv  # noqa: E402
from .votes import VoteError, VoteStore  # noqa: E402

logger = logging.getLogger("blind_ab")

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_FILES = {
    "index.html": "text/html; charset=utf-8",
    "results.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "results.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
}
VIDEO_TYPES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"}
MAX_BODY_BYTES = 16 * 1024
CHUNK_BYTES = 256 * 1024
RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
VIDEO_ROUTE_RE = re.compile(r"^/video/([A-Za-z0-9_-]{1,64})/(left|right)$")


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Return an inclusive (start, end) byte range, or None for the whole file.

    Raises ValueError for unsatisfiable or malformed ranges.
    """
    if not header:
        return None
    match = RANGE_RE.match(header.strip())
    if not match or (not match.group(1) and not match.group(2)):
        raise ValueError(f"unsupported range {header!r}")
    first, last = match.group(1), match.group(2)
    if not first:
        length = int(last)
        if length == 0:
            raise ValueError("empty suffix range")
        return (max(0, size - length), size - 1)
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start >= size or end < start:
        raise ValueError("range not satisfiable")
    return (start, end)


class Handler(BaseHTTPRequestHandler):
    server_version = "BlindAB/1.0"
    protocol_version = "HTTP/1.1"
    state: AppState  # set by make_server

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.debug("%s - %s", self.address_string(), format % args)

    # ---- helpers -------------------------------------------------------
    def _send_bytes(self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, payload: Any, extra: dict[str, str] | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8", extra)

    def _error(self, status: int, message: str) -> None:
        self._send_json(status, {"ok": False, "error": message})

    def _query(self) -> dict[str, str]:
        return {k: v[-1] for k, v in parse_qs(urlparse(self.path).query).items()}

    # ---- routing -------------------------------------------------------
    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            self._route_get(path)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:  # noqa: BLE001 - keep the server alive, log details
            logger.exception("GET %s failed", path)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal error; see server log")

    def _route_get(self, path: str) -> None:
        state, query = self.state, self._query()
        if path in ("/", "/index.html"):
            self._serve_static("index.html")
        elif path in ("/results", "/results.html"):
            self._serve_static("results.html")
        elif path.startswith("/static/"):
            self._serve_static(path[len("/static/"):])
        elif path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif path == "/api/info":
            self._send_json(HTTPStatus.OK, state.info())
        elif path == "/api/ballot":
            self._get_ballot(query.get("voter"))
        elif path == "/api/results":
            self._send_json(HTTPStatus.OK, state.results(query.get("baseline")))
        elif path == "/api/results.json":
            self._send_json(HTTPStatus.OK, state.results(query.get("baseline")),
                            {"Content-Disposition": 'attachment; filename="blind_ab_results.json"'})
        elif path == "/api/results.csv":
            body = summary_to_csv(state.results(query.get("baseline"))).encode("utf-8")
            self._send_bytes(HTTPStatus.OK, body, "text/csv; charset=utf-8",
                             {"Content-Disposition": 'attachment; filename="blind_ab_results.csv"'})
        elif path == "/api/votes.jsonl":
            body = "".join(v.to_json() + "\n" for v in state.votes.all()).encode("utf-8")
            self._send_bytes(HTTPStatus.OK, body, "application/x-ndjson; charset=utf-8",
                             {"Content-Disposition": 'attachment; filename="votes.jsonl"'})
        elif (match := VIDEO_ROUTE_RE.match(path)) is not None:
            self._serve_video(match.group(1), match.group(2))
        else:
            self._error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path != "/api/vote":
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                self._error(HTTPStatus.BAD_REQUEST, "missing or oversized request body")
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise VoteError("expected a JSON object")
            self._send_json(HTTPStatus.OK, self.state.record_vote(payload))
        except (VoteError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except BallotError as exc:
            self._error(HTTPStatus.GONE, str(exc))
        except OSError:
            logger.exception("failed to write vote")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "could not save the vote; see server log")

    # ---- handlers ------------------------------------------------------
    def _get_ballot(self, voter: str | None) -> None:
        try:
            self._send_json(HTTPStatus.OK, self.state.new_ballot(voter))
        except VoteError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def _serve_static(self, name: str) -> None:
        content_type = STATIC_FILES.get(name)
        if content_type is None:
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        self._send_bytes(HTTPStatus.OK, (STATIC_DIR / name).read_bytes(), content_type)

    def _serve_video(self, ballot_id: str, side: str) -> None:
        try:
            video = self.state.video_path(ballot_id, side)
        except (BallotError, KeyError):
            self._error(HTTPStatus.NOT_FOUND, "unknown ballot")
            return
        size = video.stat().st_size
        try:
            byte_range = parse_range(self.headers.get("Range"), size)
        except ValueError:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start, end = byte_range if byte_range else (0, size - 1)
        length = max(0, end - start + 1)
        self.send_response(HTTPStatus.PARTIAL_CONTENT if byte_range else HTTPStatus.OK)
        self.send_header("Content-Type", VIDEO_TYPES.get(video.suffix.lower(), "application/octet-stream"))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "private, max-age=3600")
        if byte_range:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD" or length == 0:
            return
        with video.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(CHUNK_BYTES, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


class QuietServer(ThreadingHTTPServer):
    """Threaded server that ignores clients hanging up mid-request (common with video seeking)."""
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError, TimeoutError)):
            return
        logger.exception("error handling request from %s", client_address)


def make_server(state: AppState, host: str, port: int) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler, ), {"state": state})
    server = QuietServer((host, port), handler)
    return server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True, help="bundle directory (manifest or simple layout)")
    parser.add_argument("--votes", help="votes JSONL path (default: <bundle>/votes.jsonl)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address; use 0.0.0.0 to share on a LAN")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--baseline", help="arm slug used for speedup columns (default: first arm with speed)")
    parser.add_argument("--arms", help="comma-separated arm slugs to compare (default: every arm in arms.json)")
    parser.add_argument("--verbose", action="store_true", help="log every HTTP request")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    try:
        only = [slug.strip() for slug in args.arms.split(",") if slug.strip()] if args.arms else None
        bundle = load_bundle(args.bundle, only)
    except BundleError as exc:
        logger.error("%s", exc)
        return 2
    if args.baseline and args.baseline not in bundle.arm_slugs:
        logger.error("--baseline %r is not one of %s", args.baseline, list(bundle.arm_slugs))
        return 2
    votes_path = Path(args.votes) if args.votes else bundle.root / "votes.jsonl"
    try:
        store = VoteStore(votes_path)
    except OSError as exc:
        logger.error("cannot read votes file %s: %s", votes_path, exc)
        return 2
    state = AppState(bundle, store, baseline=args.baseline)
    try:
        server = make_server(state, args.host, args.port)
    except OSError as exc:
        logger.error("cannot bind %s:%d: %s", args.host, args.port, exc)
        return 2
    logger.info("bundle %s (%s layout): %d arms, %d clips, %d existing votes", bundle.root, bundle.layout,
                len(bundle.arms), len(bundle.clips), len(store.all()))
    logger.info("votes file: %s", store.path)
    shown_host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0", "::") else args.host
    logger.info("vote at http://%s:%d/  results at http://%s:%d/results", shown_host, args.port, shown_host,
                args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
