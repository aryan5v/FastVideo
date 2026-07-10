#!/bin/sh
set -eu

APP_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REPO_ROOT=$(CDPATH= cd -- "$APP_ROOT/../.." && pwd)
APP_NAME=FastVideo.app
DIST_DIR="$APP_ROOT/dist"
APP_DIR="$DIST_DIR/$APP_NAME"
CONTENTS="$APP_DIR/Contents"
RESOURCES="$CONTENTS/Resources"
SOURCE="$RESOURCES/fastvideo-source"

cd "$APP_ROOT"
swift build -c release

rm -rf "$APP_DIR"
mkdir -p "$CONTENTS/MacOS" "$RESOURCES" "$SOURCE"
cp ".build/release/FastVideoMac" "$CONTENTS/MacOS/FastVideoMac"
cp "Resources/Info.plist" "$CONTENTS/Info.plist"

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
