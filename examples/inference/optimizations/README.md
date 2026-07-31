# Optimization Examples

```bash
python examples/inference/optimizations/attention_example.py
```

## Workload-driven generation launcher

`generation_launcher.py` runs a single generation **mode** from a versioned
MotionKernel/FastVideo workload manifest. Baseline and optimized runs should
use separate processes so environment variables do not leak.

```bash
# Native baseline
python examples/inference/optimizations/generation_launcher.py \
  --workload /path/to/motionkernel/workloads/wan_t2v_1.3b_480p.yaml \
  --mode native \
  --output-dir /tmp/wan_ab

# Optimized / fused candidate (mode_env from the workload is applied)
python examples/inference/optimizations/generation_launcher.py \
  --workload /path/to/motionkernel/workloads/wan_t2v_1.3b_480p.yaml \
  --mode optimized \
  --output-dir /tmp/wan_ab

# Validate request construction without loading weights
python examples/inference/optimizations/generation_launcher.py \
  --workload /path/to/motionkernel/workloads/ltx_480p.yaml \
  --mode native \
  --output-dir /tmp/ltx_ab \
  --dry-run

# Capture one dedicated metadata-only operator profile. Profiling is performed
# outside the measured runs so profiler overhead does not affect A/B timing.
python examples/inference/optimizations/generation_launcher.py \
  --workload /path/to/motionkernel/workloads/ltx_480p.yaml \
  --mode native \
  --output-dir /tmp/ltx_profile \
  --profile-output /tmp/ltx_profile/profiler.json
```

From MotionKernel, the same flow is:

```bash
python workload.py run-ab \
  --fastvideo-checkout /path/to/FastVideo \
  --workload workloads/wan_t2v_1.3b_480p.yaml \
  --output workspace/wan_ab
```

Result files use `schema_version: 1` and record wall time, generation time,
peak memory, environment identity, optional frames path, and failure reasons.
`candidate` mode writes `candidate_result.json`, so it cannot overwrite a
previously validated `optimized_result.json`.

## Generic graph dispatch

Once a kernel has been packaged as an artifact bundle, FastVideo can run it
without any model-specific code. Point the runtime at a **trusted** directory
of bundles:

```bash
FASTVIDEO_OPTIMIZATION_ARTIFACT_DIR=/path/to/artifacts \
FASTVIDEO_OPTIMIZATION_ARTIFACT_MODEL_ID=Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
FASTVIDEO_OPTIMIZATION_ARTIFACT_DIAGNOSTICS=/tmp/dispatch.json \
python examples/inference/optimizations/generation_launcher.py \
  --workload /path/to/motionkernel/workloads/wan_t2v_1.3b_480p.yaml \
  --mode candidate \
  --output-dir /tmp/wan_candidate
```

Dispatch attaches to repeated block stacks -- children of an `nn.ModuleList`
that share a class -- exactly as capture does, so no architecture is named
anywhere. Per stack and per observed input signature it runs the first call
natively (which is what reveals the output signature), recomputes the module's
graph fingerprint with the capture tracer, and selects a bundle whose
fingerprint, tensor signatures and declared environment all match. The chosen
entry point is called as `candidate(module, *args, **kwargs)`.

Everything is fail-safe. A missing match, an unloadable bundle, an untraceable
module or an exception raised by the candidate falls back to native execution
and records a structured reason; the candidate is not retried afterwards.

| Variable | Default | Purpose |
| --- | --- | --- |
| `FASTVIDEO_OPTIMIZATION_ARTIFACT_DIR` | `""` | Trusted artifact root. **Unset means nothing is wrapped at all** and generation is byte-for-byte identical to a build without this feature. |
| `FASTVIDEO_OPTIMIZATION_ARTIFACT_TRACER` | `symbolic` | Tracer used to recompute the fingerprint. `symbolic` does not re-execute the module with real inputs; `export` and `dynamo` do. |
| `FASTVIDEO_OPTIMIZATION_ARTIFACT_MODEL_ID` | profile model id | Model identity matched against each bundle. |
| `FASTVIDEO_OPTIMIZATION_ARTIFACT_MODEL_REVISION` | `*` | Revision matched against each bundle. |
| `FASTVIDEO_OPTIMIZATION_ARTIFACT_DISTRIBUTED_MODE` | auto | Sharding mode. Auto-detection only resolves a single-rank run; a multi-rank run must declare its mode or no artifact is selected. |
| `FASTVIDEO_OPTIMIZATION_ARTIFACT_MAX_SCOPES` | `64` | Upper bound on dispatched block stacks. |
| `FASTVIDEO_OPTIMIZATION_ARTIFACT_MAX_SHAPES` | `8` | Upper bound on resolved input signatures per stack. |
| `FASTVIDEO_OPTIMIZATION_ARTIFACT_DIAGNOSTICS` | `""` | Optional path for the structured dispatch/fallback report. |

The diagnostics report is metadata only: it records each scope, shape key,
decision reason, artifact id, rejection codes and call counts, plus the
registry and runtime identity. It never contains tensor or prompt data.

The bundle format, its packager and the matching rules are documented in
MotionKernel's `docs/ARTIFACT_BUNDLE.md`.
