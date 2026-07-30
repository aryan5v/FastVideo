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

After reviewing and trusting the campaign and its spec locators:

```bash
uv run campaign.py prepare /tmp/wan-campaign.json --trust-specs
uv run orchestrate.py plan
```

This creates ranked candidate kernels, the orchestration plan, and a campaign
receipt under `workspace/`. Follow AutoKernel's `program.md` to run the
unattended experiment loop.

Wan's first registered boundary is `wan.self_attn_residual_norm`, backed by
`models/wan_gated_residual_norm.py:SPEC`. Add future model targets with the
generic `optimization_target` context manager; no changes to the campaign
schema or AutoKernel ranking path are required.
