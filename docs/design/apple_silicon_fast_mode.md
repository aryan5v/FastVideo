# Fast mode (RIFE) — Apple Silicon

`--fast` makes local generation ~2.7× faster by **generating fewer frames and
interpolating the rest** with an Apple-Silicon-native RIFE model, instead of
denoising every frame. Video-diffusion denoise is dominated by self-attention,
which is O(tokens²); halving the frames cuts the token count ~2× and the denoise
compute ~3.7×, so the wall-clock drops far more than 2×. RIFE (which estimates
its own optical flow — no motion vectors needed) fills the dropped frames back
in for ~1.4 s, and a light unsharp pass counters its softening.

Measured on the 1.3B INT8 QAD model (fox, 480×832×81, M4): generate 41 + RIFE→81
runs in ~35 s of denoise vs ~90 s full, at reconstruction MS-SSIM **0.97**.
Reproduce with `python -m fastvideo.benchmarks.eval_metalfx_rife --mode int8`.

> **Note:** Apple's *MetalFX* frame interpolation is **not** usable here — it
> requires game-engine motion vectors + depth, which diffusion output lacks. We
> use the video-native **`rife-mlx`** model instead (Metal-backed, torch-free).

## Install

```bash
uv pip install -e ".[mlx]"   # RIFE ships vendored; this only needs MLX
```

## Use

```bash
python examples/inference/basic/mlx_wan_prompt_to_video.py \
  --mlx-checkpoint <FastWan2.1-T2V-1.3B-INT8-QAD> \
  --prompt "A red fox trotting through a snowy pine forest at golden hour, cinematic" \
  --num-frames 81 --fast \
  --output-path video_samples/fox_fast.mp4
```

`--num-frames` stays the *target* length; fast mode generates the smallest
VAE-aligned keyframe count that RIFE can interpolate to that target.

| Flag | Default | Meaning |
|---|---|---|
| `--fast` / `--no-fast` | off | enable fast mode |
| `--fast-factor` | 2 | generate 1/factor of the frames (2 = half) |
| `--fast-sharpen` | 0.6 | light unsharp strength to counter RIFE softness (0 disables) |

Fast mode composes with everything else (`--mlx-quantization int8`,
`--mlx-compile`, TAEHV vs `--decode-backend wan-vae`). Keep `--fast-factor` at 2
for quality — larger temporal gaps are where RIFE starts inventing motion.

## Spatial fast mode (`--fast-spatial`)

The spatial twin of `--fast`: instead of dropping frames, drop pixels. Denoise
*and decode* at `height/width // fast-spatial-scale`, then resample the decoded
frames up to the requested size. Self-attention is O(tokens²), so halving each
spatial axis cuts the token count 4× and the denoise time far more than that —
measured on the 1.3B INT8 QAD model at 480×832×81, M4 Max: **86.1 s → 10.3 s**
of denoise. It composes with `--fast`; both together run the same clip in
**4.5 s** of denoise.

```bash
python examples/inference/basic/mlx_wan_prompt_to_video.py \
  --prompt "A red fox trotting through a snowy pine forest at golden hour, cinematic" \
  --height 480 --width 832 --num-frames 81 --fast-spatial \
  --output-path video_samples/fox_fast_spatial.mp4
```

| Flag | Default | Meaning |
|---|---|---|
| `--fast-spatial` / `--no-fast-spatial` | off | enable spatial fast mode |
| `--fast-spatial-scale` | 2 | denoise at 1/scale of each spatial axis |
| `--fast-spatial-upsample-mode` | `lanczos` | pixel interpolation kernel (`lanczos`, `cubic`, `bilinear`, `nearest`) |
| `--fast-spatial-sharpen` | 0.4 | light unsharp strength to counter resampling softness (0 disables) |

### The upsample must happen in pixel space

This is the one thing to get right. The obvious implementation — bilinearly
upsample the finished latents and decode at the target size — **does not work**,
and produces a distinctive failure: correct composition and silhouette under a
smeared, hazy veil, with ringing along strong edges.

A Wan latent cell is a *learned code* for an 8×8 (Wan2.1) or 16×16 (Wan2.2)
pixel block, not a low-pass sample of the image. The average of two adjacent
codes is not the code of the averaged blocks; it is a vector the decoder was
never trained on. Measured on Wan2.1-1.3B at 480×832, a 2× bilinear latent
upsample destroys **62%** of the latent's high-frequency energy while leaving
its overall magnitude intact — exactly the signature of that veil. At Wan2.2-5B
the same operation degrades to black or noise.

Decoded RGB frames have no such problem: an image *is* a sampled 2-D signal, so
Lanczos interpolation is the operation it was defined for. The result is soft —
it carries stage-1's real detail budget and no more — but clean and coherent.

`--refine` gets away with a latent-space upsample only because a second DMD pass
re-denoises the hand-off; spatial fast mode passes the latent straight to the
decoder, so it cannot.

## Refine (`--refine`) stage-2 timesteps

`--refine` hands stage 1 to stage 2 as `(1 - sigma) * upsampled + sigma * noise`,
where `sigma` comes from the *first* stage-2 timestep. FastWan's DMD grid opens
at `t=1000`, which is `sigma == 1` exactly — so a stage-2 grid that starts there
weights the stage-1 result at zero and refine silently degrades into a plain
full-resolution run at twice the cost.

Left unset, `--refine-dmd-denoising-steps` now derives the stage-2 grid from the
stage-1 one with leading full-noise steps dropped (`1000,757,522` → `757,522`).
That keeps the pass on timesteps the distilled student was trained on while
letting stage-1 structure through: hand-off `sigma = 0.757`, stage-1 weight
`0.243`. Passing a grid that starts at full noise is now an error rather than a
silently wasted pass.

The run prints the resolved hand-off so it is visible:

```
[refine] stage-2 hand-off sigma=0.7568 (stage-1 weight 0.2432)
```

There is a trade-off in choosing that grid. Later start = more of the draft
survives, but fewer stage-2 steps. On Wan2.1 the default `757,522` gives weight
0.243 with two steps; `--refine-dmd-denoising-steps 522` gives weight 0.478 with
one. `--refine-sigma` decouples the noise level from the timestep entirely — it
logs a warning, because the DiT is then told a timestep that does not match the
noise it receives.

**Wan2.2-5B has a lower ceiling.** Its warped schedule maps `1000,757,522` to
sigmas `1.000, 0.940, 0.845`, so the best available stage-1 weight is **0.060**
(vs 0.243 at 1.3B). Un-warped (`--no-warp`) the same grid gives `1.000, 0.757,
0.522` and a weight of 0.243 — but warping is what matches the FastVideo
sampling schedule, so turning it off changes the timesteps the distilled student
sees. Which is better at 5B is unresolved and needs a run on real 5B weights.

## Where denoise time actually goes

Measured on M4 Max (32 GPU cores), 1.3B INT8, 480×832×81 — 32,760 tokens after
patching, 30 layers, 12 heads, head_dim 128:

| | per step | share |
|---|---:|---:|
| self-attention (SDPA) | 19.05s | **70%** |
| all linears (q/k/v/out + FFN) | 8.01s | 30% |
| everything else | ~1.6s | — |

The important part is what that leaves room for:

**There is no kernel headroom.** Sustained fp16 GEMM on this machine tops out at
**12.1 TFLOP/s** (measured, saturates by 4096³). Dense SDPA runs at 10.6
TFLOP/s — **88% of the machine's own ceiling** — and int8 `quantized_matmul`
hits 10.9. Nothing here is leaving meaningful performance on the table, so a
better attention kernel is not the lever. Ring Attention and friends do not
apply either: they distribute exact attention across devices to escape memory
limits, and this is one device that is compute-bound, not memory-bound.

**Attention is exactly O(tokens²), so token count is the lever.** Measured:
4× fewer tokens → **16× cheaper** attention. That is precisely what
`--fast-spatial` buys, and it is why that mode — not a kernel rewrite — is the
denoise story: 86.1s → 10.3s, and 4.5s stacked with `--fast`.

**Sparse attention does not work for free.** The `FASTVIDEO_MLX_WINDOW`
sliding-window path is fast (6.6× at a ±3-frame window) but destroys the output
on a checkpoint trained with dense attention — heavy colour-block noise, subject
barely recognisable. Structural agreement with the dense baseline is 0.25 at
±3 frames and 0.03 at ±5. Sparsity of this kind is a *training-time* method
(this is what VSA is); it cannot be switched on at inference. Treat that env var
as a research knob only.

**INT8 is a memory optimisation, not a speed one.** On the same shape, int8
`quantized_matmul` is ~10% *slower* than fp16 (83.1ms vs 75.2ms). End to end at
1.3B: `--mlx-quantization none` denoises in 83.3s vs 86.1s for int8, at 5.11 GiB
peak vs 3.87. INT8 remains the right default — it is what makes 14B fit at all —
but on a small model with headroom, fp16 is marginally faster. It is not a
quality problem either: int8 and unquantized fp16 agree at 0.964 on the same
seed.

**Precision settings are not hurting quality.** `--mlx-dtype bf16` is 6% slower
than the fp16 default (90.2s vs 84.9s) and no better looking; fp16 and bf16
differ at 0.769 agreement purely because rounding perturbs a chaotic 3-step
trajectory, not because either degrades. Keep fp16.

**Model load is now a visible share of fast-spatial.** With denoise at 10.3s,
the 4.2s DiT load matters. Use `--mlx-checkpoint` (a pre-quantized MLX
checkpoint from `--save-mlx-checkpoint`) so the run skips casting and quantizing
the Diffusers weights every time.

**Decoder choice.** `--decode-backend wan-vae` under `--fast-spatial` costs
1.27s → 29.6s of decode for a marginal edge-quality gain. TAEHV is the right
default here.
