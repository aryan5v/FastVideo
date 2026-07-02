# FastVideo on Apple Silicon

## Statement of purpose

FastVideo should make fast, local video generation practical beyond high-end
NVIDIA systems. Apple Silicon is the next important target: millions of users
already have Macs with capable neural, GPU, and unified-memory hardware, but the
software stack needs to be designed around the constraints and strengths of that
platform.

Our goal is to bring the FastVideo experience to a wide range of Apple Silicon
Macs, from 16 GB unified-memory laptops to higher-memory MacBook Pro, Mac Studio,
and Mac Pro systems. More memory should unlock higher resolution, longer clips,
and better decode quality. Smaller machines should still receive the best
quality and speed that is physically realistic through careful model choice,
quantization, scheduling, caching, and decode strategy.

This is not a one-time port. It is an ongoing optimization track. As Apple
hardware, MLX, model architectures, distillation methods, and quantization
techniques improve, the Mac path should keep improving with them.

## Why now

FastWan-QAD showed what is possible when the model, quantization strategy, and
runtime are co-designed: the recent launch generated a 5-second 480p video in
1.8 seconds on a single RTX 5090 using quantization-aware distillation.

Apple Silicon will not reach that result by simply copying the NVIDIA path.
Blackwell tensor cores, CUDA kernels, and NVFP4-specific execution do not map
directly to Macs. The opportunity is to build the Apple-native equivalent:

- an MLX-first DiT runtime,
- memory-aware prompt encoding and decode,
- Mac-friendly quantization targets,
- a distilled/QAT model that is trained with those targets in mind,
- and benchmarks that make the quality/speed tradeoffs visible.

## Progress so far

We have established a working proof of concept:

- FastWan runs locally on macOS through MPS and an experimental MLX DiT path.
- The path supports FP16 plus MLX quantization experiments including INT8, INT4,
  MXFP-style modes, and NVFP4-style mode simulation.
- TAEHV decode is available for lower-memory, faster video reconstruction.
- Prompt encoding can be isolated so the text encoder is freed before DiT
  denoising, which matters for 16 GB systems.
- INT8 is currently the most reliable quantization target: it meaningfully
  reduces memory while staying much closer to FP16 than INT4-style modes.
- A benchmark harness now measures latency, peak memory, generated artifacts,
  and optional quality metrics across mode and decoder combinations.

The important result is not that the current videos are final quality. They are
not. The important result is that the pipeline is now real enough to measure,
compare, and improve.

## Product target

The Apple Silicon path should eventually expose memory-aware presets rather than
one brittle configuration:

- **16 GB Macs:** accessible local generation with smaller resolution/clip
  length, INT8 or better quantized DiT weights, prompt-cache/freeing, and TAEHV
  decode by default.
- **24-36 GB Macs:** better resolution, longer clips, optional higher-quality
  decode, and more room for FP16/INT8 comparisons.
- **64 GB+ Macs:** higher quality presets, longer clips, more model families,
  stronger decode options, and broader benchmarking coverage.

Every tier should aim for the same principle: use the available unified memory
intelligently, keep the runtime responsive, and avoid hiding quality regressions
behind raw speed numbers.

## Technical direction

The work should proceed on three tracks.

### 1. Runtime

Build a clean MLX runtime for the parts that benefit most from Apple-native
execution, starting with the DiT denoising loop. Keep MPS/PyTorch where it is
still the practical bridge, but move performance-critical and memory-critical
paths toward MLX as they mature.

Near-term runtime priorities:

- stabilize the MLX FastWan DiT path,
- keep DMD sampling on device,
- benchmark `mx.compile` and fused MLX kernels with quality checks,
- make TAEHV and VAE decode choices explicit,
- reduce host/device transfers,
- and keep the runtime easy to test against FP16 reference behavior.

### 2. Model

The long-term quality and speed gains will come from a Mac-specific model path,
not only from runtime optimization.

The likely model strategy is:

- start from Wan/FastWan-compatible weights rather than training from scratch,
- fine-tune or distill with Apple-targeted constraints,
- use quantization-aware training for the quant modes we actually want to serve
  on Macs,
- optimize for a small number of denoising steps,
- and export checkpoints that are friendly to MLX loading and inference.

The first serious target should be a distilled INT8-oriented model, because INT8
currently offers the best quality/memory balance. INT4/MXFP-style modes remain
important, but they likely need QAT/distillation before they become reliable
quality presets.

### 3. Benchmarks and model coverage

We should benchmark the Mac path like a product surface, not a single demo.

Required benchmark coverage:

- FP16, BF16, INT8, INT4, MXFP-style, and NVFP4-style modes where supported,
- TAEHV vs full VAE decode,
- multiple memory tiers,
- multiple prompts with visible motion and physics,
- multiple supported FastVideo model families,
- and both latency and quality metrics.

The benchmark should produce artifacts that are easy to inspect: videos,
side-by-side HTML grids, JSON metrics, and short markdown summaries. This is how
we decide what is actually improving.

## Major milestones

1. **Reliable local baseline**
   - MLX DiT path works consistently on Apple Silicon.
   - TAEHV and VAE decode paths are benchmarked.
   - INT8/FP16 are stable enough for repeatable demos.

2. **Benchmark suite**
   - Standard prompts, resolutions, frame counts, and memory tiers.
   - Side-by-side visual outputs.
   - Latency, memory, and quality metrics collected in one place.
   - Coverage for other FastVideo-supported models.

3. **Runtime hardening**
   - Fewer CPU/GPU transfers.
   - On-device scheduler math.
   - Tested `mx.compile` and fused-kernel paths.
   - Cleaner install and CLI surface for Mac users.

4. **Mac-specific distillation/QAT**
   - Train or distill from a Wan/FastWan base model.
   - Target the Mac runtime and quantization modes directly.
   - Prioritize INT8 first, then evaluate INT4/MXFP-style modes after QAT.

5. **16 GB product preset**
   - A documented configuration that runs reliably on 16 GB unified memory.
   - Clear expectations around resolution, frame count, decode mode, and speed.

6. **Higher-memory quality presets**
   - Better defaults for 24 GB, 32 GB, 64 GB, and larger systems.
   - Higher-quality decode and longer clips where the hardware allows it.

7. **Public Mac support story**
   - A reproducible demo.
   - A benchmark article with honest tradeoffs.
   - Clear setup docs.
   - A path for contributors to add models, prompts, metrics, and Apple-specific
     optimizations.

## Operating principle

We should optimize for honest progress. Fast videos that look broken are not the
goal. Beautiful videos that take too long are also not the goal. The work is to
find the best quality-speed-memory point for each Mac tier, make it reproducible,
and keep pushing that frontier forward.

