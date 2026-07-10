# FastVideo for Mac

FastVideo for Mac is a native SwiftUI front end for the Apple Silicon
FastWan-QAD 1.3B MLX runtime. It is intentionally not a local web server:
Swift owns setup, generation state, history, playback, Finder export, the macOS
Share Sheet, notifications, and sleep prevention. Python owns one narrow JSONL
bridge to the existing MLX inference entrypoint.

On macOS 26 and newer, navigation and primary controls use SwiftUI's native
Liquid Glass APIs (`glassEffect`, `GlassEffectContainer`, `.glass`, and
`.glassProminent`). Older supported systems retain a material-backed fallback.
The content and media layers stay deliberately dark and quiet so glass remains
a functional hierarchy rather than decoration on every card.

## Product flow

1. A four-step, skippable onboarding introduces local inference, lets people
   choose a starter prompt, demonstrates live x0 preview, and checks the Mac.
   Downloads are deferred so people can explore first; unreleased builds can
   connect directly to model artifacts already on the Mac.
2. The first-run Setup screen creates a Python 3.12 environment with
   `uv pip install -e '.[mlx]'`, discovers imageio's bundled ffmpeg or a system
   ffmpeg, and connects to local shared model files and MLX checkpoints.
3. Create defaults to the release-selected EMA weights while keeping RAW
   available for direct A/B comparison. A candidate is selectable only when its
   MLX checkpoint is present.
4. Generate starts the three-step DMD lane. After every non-final step, the full
   x0 prediction is decoded with TAEHV and atomically published as a rough MP4.
   The native player swaps to that preview while MLX continues denoising.
5. The final MP4 replaces the preview and is stored with prompt, settings, time,
   and metrics in `~/Library/Application Support/FastVideo/Generations`.

The release-validated default remains 832x480, 81 frames, 16 fps, INT8 MLX DiT,
TAEHV decode, and DMD timesteps `1000,757,522`. Smaller frame sizes are exposed
for iteration, but are not presented as release validation evidence.

## Run from source

Only Apple Command Line Tools are needed to compile the UI:

```console
cd apps/fastvideo_mac
swift run FastVideoMac
```

The app will locate the enclosing FastVideo checkout automatically. Use Setup
to choose another checkout, Python executable, model folder, or explicit RAW
and EMA checkpoint directories.

## Build an app bundle

```console
cd apps/fastvideo_mac
./scripts/package_app.sh
open dist/FastVideo.app
```

The packaging script embeds the FastVideo Python source required by this lane,
but not the Python environment or model weights. It applies an ad-hoc signature
for local testing. Public distribution still requires a Developer ID signature,
Hardened Runtime review, notarization, and a release model with final checksums.

## Model layout

The model folder contains the shared Diffusers tokenizer, text encoder, and
transformer config. The app auto-detects either of these variant layouts:

```text
mlx_dit_raw/        mlx_dit_ema/
mlx_dit/raw/        mlx_dit/ema/
raw/mlx_dit/        ema/mlx_dit/
```

Each checkpoint directory must contain `mlx_dit.json` and
`mlx_dit.safetensors`. On this development Mac, the app also discovers
`~/models/qad_int8_v2_ema`, `~/mlx-ckpt-cache-qad-v2-ema/int8`, and
`~/mlx-ckpt-cache-qad-v2/int8`. The earlier v1 EMA export is intentionally
skipped because its generated videos collapse to noise. Explicit paths in Setup
override auto-detection.

## Verify

The test script works with lightweight Apple Command Line Tools and does not
require XCTest:

```console
./scripts/test.sh
```

It compiles the app, runs a Swift Foundation self-test for durable history and
preview-to-final playback selection, and runs bridge tests for checkpoint
detection and live-preview command construction.

## Architecture

```text
SwiftUI app
  ├── Setup + runtime diagnosis
  ├── Generation library (JSON + local MP4s)
  ├── AVKit preview/final player
  └── ProcessDriver
         │ JSON lines
         ▼
fastvideo_mlx_bridge.py
  ├── Hugging Face snapshot download
  └── mlx_wan_prompt_to_video.py
         ├── MPS prompt encode
         ├── MLX INT8 DMD denoise
         ├── per-step x0 → TAEHV preview
         └── final TAEHV MP4
```
