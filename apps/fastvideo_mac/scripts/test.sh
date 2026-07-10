#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

swift build
mkdir -p .build/self-test
swiftc \
  Sources/FastVideoMac/Models.swift \
  Sources/FastVideoMac/GenerationLibrary.swift \
  Sources/FastVideoMac/ProcessDriver.swift \
  Tests/CoreSelfTest.swift \
  -o .build/self-test/FastVideoMacCoreSelfTest
.build/self-test/FastVideoMacCoreSelfTest
python3 -m unittest Tests/test_bridge.py
