# FastWan QAD for Mac

FastWan QAD is a native SwiftUI application for the Apple silicon FastWan QAD
1.3B MLX runtime. It is not a local web server. Swift owns onboarding,
first-party model installation, generation state, history, playback, Finder
export, the macOS Share Sheet, notifications, and sleep prevention. Python owns
one narrow JSONL bridge to the existing MLX inference entrypoint.

On macOS 26 and newer, navigation and primary controls use SwiftUI Liquid Glass.
Older supported systems retain a material-backed fallback. The workspace stays
dark and quiet so the video remains the visual focus.

## Product flow

1. A four-step onboarding introduces local inference, live x0 preview, privacy,
   and the one-click model setup.
2. The embedded release catalog installs the recommended EMA model by default.
   RAW is an optional second download for direct checkpoint comparisons. Users
   never need to enter a repository, URL, or filesystem path.
3. The packaged app carries `uv`, which prepares the managed Python 3.12, MLX,
   MPS, and video-export runtime before the selected model download begins.
4. Create opens as one centered prompt composer. Generation mode, model, and
   format options stay behind the composer settings menu. Fast generation is
   the recommended default; Full remains available when every frame should be
   generated natively. The video workspace appears only after generation starts.
5. Every non-final DMD x0 prediction is decoded with TAEHV and atomically
   published as a playable preview. The final MP4 replaces it automatically.
6. Library presents local generations as a visual gallery with playback,
   prompt, generation mode, render metadata, Finder export, sharing, and deletion.

The release default is EMA, Fast mode, 832x480, 81 output frames, 16 fps, INT8
MLX DiT, TAEHV decode, and DMD timesteps `1000,757,522`. Fast mode generates 41
source frames, then uses MLX-native RIFE 2× interpolation and light sharpening
to restore the requested 81-frame output. The validated 1.3B recipe is about
2.7× faster at roughly 0.97 reconstruction MS-SSIM.

## First-party model distribution

`Resources/model-catalog.json` is bundled into the application. It points to
four first-party release archives:

- shared tokenizer, text encoder, scheduler, VAE, and transformer config,
- RIFE 4.25 weights installed under `rife/RIFE-4.25` for offline fast
  generation,
- prequantized EMA MLX weights,
- optional prequantized RAW MLX weights.

The installer accepts HTTPS release assets, streams download progress into the
native UI, verifies SHA-256 when present, rejects unsafe archive paths, and
installs through a staging directory. Release packaging is gated by
`FASTWAN_RELEASE_BUILD=1`, which requires all catalog checksums and a bundled
`uv` executable.

Downloaded files live under:

```text
~/Library/Application Support/FastWan QAD/Models/v2/
  Shared/
  EMA/
  RAW/
```

Developer options retain explicit source, Python, shared-model, and checkpoint
overrides. The app also recognizes the validated local v2 development caches;
the broken v1 EMA export is intentionally skipped because it produces noise.

## Run from source

```console
cd apps/fastvideo_mac
swift run FastVideoMac
```

## Build an app bundle

```console
cd apps/fastvideo_mac
./scripts/package_app.sh
open "dist/FastWan QAD.app"
```

The development bundle embeds the FastVideo Python source, model catalog, and a
local `uv` binary when available. It uses an ad-hoc signature. Public
distribution still requires Developer ID signing, Hardened Runtime review,
notarization, published release assets, and final catalog checksums.

## Verify

```console
./scripts/test.sh
```

The script compiles the app, runs the Swift history/process/preview self-test,
and runs Python tests for checkpoint detection, first-party archive installation,
live-preview command construction, and the macOS 27 AVPlayerView contract.

## Architecture

```text
SwiftUI app
  ├── Liquid Glass onboarding + one-click EMA setup
  ├── Prompt-first Create workspace
  ├── Visual local Library
  ├── Models & Runtime
  └── ProcessDriver
         │ JSON lines
         ▼
fastvideo_mlx_bridge.py
  ├── first-party release archive installer
  └── mlx_wan_prompt_to_video.py
         ├── MPS prompt encode
         ├── MLX INT8 DMD denoise
         ├── per-step x0 → TAEHV preview
         ├── final TAEHV decode
         └── optional RIFE 2× interpolation → MP4
```
