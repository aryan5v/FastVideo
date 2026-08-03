# H3 Student Sizing for Apple Silicon at INT8

How small an H3 student needs to be to reach a wide Mac install base, given the
measured result that **affine int8 is the format** (`m5_survey_results.md`).

int8 costs 1.0625 GB per billion parameters. MLX allocator caps are the
project's own measured bands from `hardware_tier.py`.

## What fits

| Mac | MLX cap | activations + decoders | DiT budget | int8 params | int6 params |
|---|---|---|---|---|---|
| 8 GB | 5 GiB | 2.0 GiB | 3.0 GiB | ~2.8B | ~3.7B |
| **16 GB** | **12 GiB** | **3.0 GiB** | **9.0 GiB** | **~8.5B** | ~11.1B |
| 24 GB | 15 GiB | 3.5 GiB | 11.5 GiB | ~10.8B | ~14.2B |
| 32–36 GB | 24 GiB | 4.5 GiB | 19.5 GiB | ~18.4B | ~24.0B |
| 64 GB | 44 GiB | 4.5 GiB | 39.5 GiB | ~37.2B | ~48.6B |

Weight cost of candidate sizes:

| params | int8 | int6 | bf16 |
|---|---|---|---|
| 2B | 2.1 GB | 1.6 GB | 4.0 GB |
| **4B** | **4.2 GB** | **3.2 GB** | 8.0 GB |
| 5B | 5.3 GB | 4.1 GB | 10.0 GB |
| 7B | 7.4 GB | 5.7 GB | 14.0 GB |

## Recommendation: ~4B at int8, with 16 GB as the floor

**16 GB is the right floor for "wide".** It is the base configuration across the
current M-series line — Air, base Pro, mini — so it reaches most machines sold
in the last several years. 8 GB machines exist in volume but are legacy, and
designing for them costs more quality than the extra reach is worth.

At 16 GB the ceiling is ~8.5B, but **~4B is the number to build.** 4.2 GB of
weights against a 9 GiB DiT budget leaves room for encoder sequencing,
longer clips, higher base resolution, and the multi-clip case — none of which a
7–8B student would have. It is also roughly 3× FastWan-1.3B's capacity, so the
quality step should be clearly visible.

A **~2B** variant is the option if 8 GB machines turn out to matter. It fits
everywhere, but it is a severe compression ratio from a frontier teacher and
should be treated as a separate product decision, not a default.

## The encoder is probably the real constraint, not the DiT

H3 is omni-modal, and its text/multimodal encoder is unsized (M005). This
matters more than the student size, because on a 16 GB machine the encoder can
be the peak:

| Encoder | int8 size | Peak with a 4B DiT (sequenced) | 16 GB? |
|---|---|---|---|
| 3B | 3.2 GB | 7.2 GB | fine |
| 5B | 5.3 GB | 7.2 GB | fine |
| 7B | 7.4 GB | 7.4 GB | fine, encoder now sets the peak |
| 10B | 10.6 GB | 10.6 GB | tight against a 12 GiB cap |

Peak is `max(encoder, DiT + activations)` **only if the runtime encodes, frees
the encoder, then denoises**. Without that discipline these are sums and a 16 GB
machine is out of reach immediately. It is a hard requirement, not an
optimization.

Two levers if the encoder is large:

- **Quantize the encoder harder than the DiT.** The encoder runs once; its error
  does not compound across denoising steps. int4 or int6 on an encoder is far
  safer than on a denoiser, and the survey shows affine int4 at ~9% relative
  error — acceptable for a conditioning path, not for a DiT run 3–8 times.
- **Precompute embeddings.** For a text-only path, conditioning can be computed
  once and cached, removing the encoder from the device budget entirely.

## What pushes back against going smaller

**Distillation cost rises as the student shrinks.** Capacity reduction (Track B1)
is already the least predictable line item, and a bigger compression ratio makes
it worse, not better. Going from a 4B target to 2B is not half the work — it is
a harder problem with a less certain outcome.

**There is a quality floor.** FastWan-1.3B is shippable because Wan-1.3B was a
released model at that size, not a compression of something far larger. An H3
student at 2B is a much steeper ratio and may not clear the bar that makes
shipping worthwhile.

**int6 is the lever that buys capacity without shrinking.** Pending measurement,
int6 at ~0.022 relative error puts ~5.5B in the same 4.2 GB an int8 4B occupies
— a bigger, better student at identical memory on the proven grid. If int6
holds, size up rather than down.

## Open

`M001` still gates the whole thing: if H3 is a large MoE, distilling to a dense
4B student is a severe ratio and Track B1 may not be feasible on 36 GPUs at all.
Resolve that before committing to any student size.
