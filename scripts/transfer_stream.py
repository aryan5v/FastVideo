#!/usr/bin/env python3
"""Send or receive one binary stream over a TCP socket."""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

CHUNK_SIZE = 8 * 1024 * 1024


def send(host: str, port: int) -> None:
    with socket.create_connection((host, port), timeout=60) as sock:
        while chunk := sys.stdin.buffer.read(CHUNK_SIZE):
            sock.sendall(chunk)


def receive(port: int, output: Path) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", port))
        server.listen(1)
        server.settimeout(900)
        connection, _ = server.accept()
        with connection, output.open("wb") as stream:
            while chunk := connection.recv(CHUNK_SIZE):
                stream.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    sender = subparsers.add_parser("send")
    sender.add_argument("--host", required=True)
    sender.add_argument("--port", required=True, type=int)
    receiver = subparsers.add_parser("receive")
    receiver.add_argument("--port", required=True, type=int)
    receiver.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "send":
        send(args.host, args.port)
    else:
        receive(args.port, args.output)


if __name__ == "__main__":
    main()

