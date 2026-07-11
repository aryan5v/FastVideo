#!/bin/sh
set -eu

APP_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REPO_ROOT=$(CDPATH= cd -- "$APP_ROOT/../.." && pwd)
APP_NAME="FastWan QAD.app"
DIST_DIR="$APP_ROOT/dist"
APP_DIR="$DIST_DIR/$APP_NAME"
CONTENTS="$APP_DIR/Contents"
RESOURCES="$CONTENTS/Resources"
SOURCE="$RESOURCES/fastvideo-source"

cd "$APP_ROOT"
swift build -c release

rm -rf "$DIST_DIR/FastVideo.app" "$APP_DIR"
mkdir -p "$CONTENTS/MacOS" "$RESOURCES" "$SOURCE"
cp ".build/release/FastVideoMac" "$CONTENTS/MacOS/FastVideoMac"
cp "Resources/Info.plist" "$CONTENTS/Info.plist"
cp "Resources/model-catalog.json" "$RESOURCES/model-catalog.json"
UV_BIN=${FASTWAN_UV_BIN:-$(command -v uv 2>/dev/null || true)}
if [ -n "$UV_BIN" ]; then
    mkdir -p "$RESOURCES/bin"
    cp "$UV_BIN" "$RESOURCES/bin/uv"
    chmod 755 "$RESOURCES/bin/uv"
elif [ "${FASTWAN_RELEASE_BUILD:-0}" = "1" ]; then
    echo "FASTWAN_RELEASE_BUILD requires a uv binary" >&2
    exit 1
fi

if [ "${FASTWAN_RELEASE_BUILD:-0}" = "1" ]; then
    python3 - "$RESOURCES/model-catalog.json" <<'PY'
import json
import sys

catalog = json.load(open(sys.argv[1]))
for name, asset in [("shared", catalog["shared"]), *catalog["variants"].items()]:
    checksum = asset.get("sha256", "")
    if len(checksum) != 64:
        raise SystemExit(f"release asset {name} needs a SHA-256 checksum")
PY
fi

# Bundle the narrow source runtime needed by the managed MLX environment. The
# model weights and Python environment remain outside the app in user-owned
# folders, so rebuilding the UI never duplicates multi-gigabyte artifacts.
cp "$REPO_ROOT/pyproject.toml" "$SOURCE/pyproject.toml"
cp "$REPO_ROOT/README.md" "$SOURCE/README.md"
ditto "$REPO_ROOT/fastvideo" "$SOURCE/fastvideo"
mkdir -p "$SOURCE/examples/inference/basic" "$SOURCE/apps/fastvideo_mac/bridge"
cp "$REPO_ROOT/examples/inference/basic/mlx_wan_prompt_to_video.py" "$SOURCE/examples/inference/basic/"
cp "$APP_ROOT/bridge/fastvideo_mlx_bridge.py" "$SOURCE/apps/fastvideo_mac/bridge/"

codesign --force --deep --sign - "$APP_DIR"
echo "$APP_DIR"
