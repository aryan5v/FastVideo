# AutoKernel campaign capture

FastVideo can capture a metadata-only optimization campaign during a normal
pipeline run. The campaign hands model-specific bottlenecks to an external
kernel-search loop without exporting prompts, activations, tensor values, or
model weights.

Set one environment variable and run the normal inference command:

```bash
FASTVIDEO_OPTIMIZATION_CAPTURE=/tmp/wan-campaign.json \
  fastvideo generate <your normal Wan arguments>
```

Distributed jobs should include placeholders so each process writes its own
atomic artifact:

```bash
export FASTVIDEO_OPTIMIZATION_CAPTURE=/tmp/wan-<rank>-<pid>.json
```

The capture contains:

- model, workload, FastVideo, PyTorch, CUDA, and hardware identities;
- candidate names and external `KernelSpec` locators;
- tensor shapes, strides, dtypes, and device types;
- call counts and shape frequencies;
- aggregate CUDA-event timings for the pipeline and each candidate.

Capture is disabled by default. When disabled, instrumented layer calls take
their existing path without constructing capture metadata. Enabling capture
adds timing events and performs one CUDA synchronization when the pipeline run
finishes, so use it for optimization discovery rather than production serving.

## Prepare the search

In the AutoKernel checkout, validate before loading any referenced Python:

```bash
uv run campaign.py validate /tmp/wan-campaign.json
uv run campaign.py rank /tmp/wan-campaign.json
```

After reviewing and trusting the campaign and its spec locators, launch the
ranked search directly:

```bash
uv run campaign.py run /tmp/wan-campaign.json --budget-hours 10
```

Use `--dry-run` to review the generated instructions or `--resume` after an
interruption. MotionKernel writes a durable log, terminal receipt, optimized
artifacts, and `workspace/morning_report.md`.

Wan currently registers three boundaries: modulated pre-attention LayerNorm,
the post-self-attention gated residual plus LayerNorm, and the post-MLP gated
residual. Add future model targets with the generic `optimization_target`
context manager; no changes to the campaign schema or MotionKernel ranking
path are required.

## Use promoted kernels

FastVideo ships safe bundled baselines and retains the native PyTorch path for
training, unsupported layouts, or failures. To enable all Wan fusion points:

```bash
export FASTVIDEO_WAN_FUSIONS=1
```

To test verified output from an overnight MotionKernel run without copying
files into FastVideo, explicitly trust its artifact directory:

```bash
export FASTVIDEO_AUTOKERNEL_ARTIFACT_DIR=/path/to/motionkernel/workspace
```

FastVideo matches artifacts by their declared operation, validates
`KERNEL_TYPE`, and falls back to the bundled implementation if an artifact
cannot be loaded. The older `FASTVIDEO_WAN_FUSED_NORM=1` switch remains a
compatibility alias.
