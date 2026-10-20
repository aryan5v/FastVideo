# Run H3 with a server and playground

Start a supported FastVideo model on CUDA, one DGX Spark, or Apple Silicon MLX,
then iterate on prompts in a browser, with cURL, or from your app. The server and
clients can run on the same machine.

The server exposes OpenAI-compatible **video jobs**, image routes, and audio
media jobs. The Python and JavaScript client examples use the video-job
interface, but requests go to FastVideo. You do not need an OpenAI account or
cloud key. This is the tested FastVideo surface, not a claim of full OpenAI API
or SDK compatibility; see [HTTP serving compatibility](serving-compatibility.md).

The walkthrough below uses FastH3 because it has the longest-running playground
and client coverage. For another family, pick a baseline from the
[serving ledger](serving-coverage.md) and use its `fastvideo serve --config`
command. The request, polling and download steps stay the same when that loaded
pipeline supports the requested task.

The [H3 recipe selector](minimax-h3.md) provides the same workflow with runtime
selection. This guide covers FastH3 V1 (four forwards) and FastH3 V2
(eight forwards). Start a server, then iterate in the playground or with the
OpenAI Python client. Other H3 recipes keep their direct Python commands.

CUDA requests reuse one loaded `VideoGenerator`. The Python SDK can do the same
when you reuse the generator across `generate()` calls. MLX keeps one
`MiniMaxH3MLXPipeline` and a prompt-embedding cache, but still loads and releases
components between phases to fit unified memory. MLX serving does not keep all
weights resident or remove phase-loading time. Use a server when you want
separate clients to share the process without restarting scripts.

## Install and start the server

### CUDA

Use a FastVideo clone and an activated Python environment. Complete the
[CUDA installation requirements](../getting_started/installation/gpu.md)
before running these commands:

```bash
UV_TORCH_BACKEND=cu130 uv pip install -e ".[fasth3]"
fastvideo serve --config examples/serving/openai_fasth3.yaml --server.host 127.0.0.1
```

The configuration loads `FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2` and
advertises it as `fasth3`. It configures four CUDA GPUs but does not record
a GPU model or VRAM requirement. This is a source-backed server profile, not
the measured GB200 Python performance profile. Compilation is disabled.

For FastH3 V2, use the same install and the 8-step config (nine sigma
points, eight DiT forwards, VSA sparsity 0.8):

```bash
UV_TORCH_BACKEND=cu130 uv pip install -e ".[fasth3]"
fastvideo serve --config examples/serving/openai_fasth3_8step.yaml --server.host 127.0.0.1
```

Keep the server running. In another terminal, check readiness:

```bash
curl --fail-with-body http://127.0.0.1:8000/health
```

After model loading completes, the response is `{"status":"ok"}`.

### NVIDIA DGX Spark

Complete the [DGX Spark installation](../getting_started/installation/spark.md)
before running these commands. GB10 has no FA4 / ptx_blackwell VSA kernel:

```bash
UV_TORCH_BACKEND=cu130 uv pip install -e .
FASTVIDEO_VSA_PTX_BLACKWELL=0 FASTVIDEO_FA4=0 FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN_H3 \
  fastvideo serve --config examples/serving/openai_fasth3_spark.yaml --server.host 127.0.0.1
```

The configuration loads `FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree`
on one GB10 and advertises it as `fasth3`. Lazy module load still reloads
Qwen3-VL and the DiT between phases of each request. Legal `num_frames` values
are `17n+5`, capped at 362 (15.08 s); a 345-frame request on one Spark can OOM.
There is no cookbook server for two Sparks; use the generate YAML after
[pairing two Sparks](../getting_started/installation/spark_pair.md).

Keep the server running. In another terminal, check readiness:

```bash
curl --fail-with-body http://127.0.0.1:8000/health
```

After model loading completes, the response is `{"status":"ok"}`.

### Apple Silicon MLX

Complete the [MLX install](../getting_started/installation/mlx.md),
including `ffmpeg`. From your FastVideo clone, install the MLX extra:

```bash
uv pip install -e ".[mlx]"
```

Download and convert the weights once. Skip this step if you already have the
snapshot and converted DiT:

```bash
hf download FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2 --local-dir ./FastH3-Preview-v0.2
python scripts/checkpoint_conversion/convert_minimax_h3_mlx.py --model-root ./FastH3-Preview-v0.2/transformer --out ./FastH3-MLX --formats "int6"
```

Edit `generator.model_root` and `generator.mlx_checkpoint` in
`examples/serving/mlx_fasth3.yaml` if your weights are elsewhere. Paths are
relative to the directory where you launch the server. Start it with:

```bash
python -m fastvideo.entrypoints.openai.mlx_server --config examples/serving/mlx_fasth3.yaml
```

This uses the native MLX pipeline, not PyTorch MPS. The default output is
832 × 480, 124 frames at 24 fps, with four DiT forwards and the full H3 VAE.
The HTTP field `num_inference_steps=5` describes five sigma points; the adapter
passes `num_steps=4` to MLX. Temporal/spatial fast modes, VSA, reference inputs,
LoRA selection, and alternate decoders are not exposed by this server adapter.
Unsupported request options return HTTP 400 before a job starts.

The server binds to `127.0.0.1:8000` and advertises `fasth3`, so the playground
and clients below work unchanged. Do not run CUDA, Spark, and MLX servers on the
same port. MLX readiness means the pipeline is initialized; components load during
generation. The first request can take longer than a repeated cached prompt.

The MLX server has no recorded device or unified-memory requirement. The
direct Python recipe's M4 Max measurements are not a server benchmark or a
minimum-memory claim.

For FastH3 V2, convert that checkpoint's transformer with `--include-vsa`
and start the 8-step config. Do not overwrite a V1 export. The
[MiniMax H3 cookbook](minimax-h3.md) has the same download and convert
commands as the Python recipe.

```bash
hf download FastVideo/FastVideo-FastH3-8-Step-V2 --local-dir ./FastH3-8-Step-V2
python scripts/checkpoint_conversion/convert_minimax_h3_mlx.py --model-root ./FastH3-8-Step-V2/transformer --out ./FastH3-8-Step-V2-MLX --formats "int8" --include-vsa
python -m fastvideo.entrypoints.openai.mlx_server --config examples/serving/mlx_fasth3_8step.yaml
```

That adapter passes `num_steps=8` when the HTTP field is `num_inference_steps=9`,
and it enables the trained VSA recipe (sparsity 0.8, 64-token tiles). Reuse the
preview VAE, audio VAE, text encoder, and tokenizer if those directories already
exist; edit the YAML paths if your files live elsewhere.

## Open the playground

Open [the local H3 playground](http://127.0.0.1:8000/playground/) after startup.
Write a prompt, optionally set a seed, then select **Generate video**. The page
checks the job status and shows the completed video with a download link. Edit
the prompt and generate again without restarting the server.

The playground sends requests to `/v1/videos` on the same server that serves
the page. **Use this prompt with cURL** shows the equivalent submission. Recent
jobs include requests from other clients, so a script and the playground can
use the same model process. Opening the page or copying a command does not
start generation.

The page URL includes the active job ID. Reloading that URL resumes status
checks without resubmitting the prompt. A failed connection or a 30-minute
polling timeout does not cancel execution. Select **Check status** to reconnect.
If submission itself is interrupted, check Recent jobs before submitting again.

This first playground supports H3 text-to-video/audio. Reference-media inputs
remain available through the API, not through the playground. It does not start
or manage a GPU server for you.

## Generate with cURL or an SDK

These examples use the server's resolution, frame count, and sampling defaults.
Do not copy Sora-specific durations or resolutions onto H3. V1 configs use
124 frames, 24 fps, and the five-point distilled sigma schedule with four DiT
forwards. V2 configs use nine sigma points and eight DiT forwards. CUDA
and one Spark use 1344 × 768; MLX uses 832 × 480. The OpenAI Python client is
the same for every FastH3 server that advertises `fasth3`.

Each client submits a job, checks for completion or failure, and downloads an
MP4 named after the job ID. Polling stops after 30 minutes; a timeout does not
cancel GPU execution. Keep the printed job ID to retrieve its status later.
Transport retries are disabled to avoid accidental duplicate submissions.

### OpenAI Python

Install the tested client, then run the checked-in example:

```bash
python -m pip install openai==3.6.0
python examples/serving/clients/video.py
```

<include>/_snippets/clients/video.py.md</include>

### OpenAI JavaScript

Use Node.js 22 or later on your computer or in your webapp's backend. Do not put a
private server key in browser code.

```bash
npm ci --prefix examples/serving/clients
node examples/serving/clients/video.mjs
```

<include>/_snippets/clients/video.mjs.md</include>

### cURL

Install `curl` and `jq`, then run:

```bash
bash examples/serving/clients/video.sh
```

<include>/_snippets/clients/video.sh.md</include>

## Images and audio

Image baselines serve `/v1/images/generations`, `/v1/images`, and
`/v1/images/edits`. Start one with its checked-in config, then send the
advertised model alias:

```bash
fastvideo serve --config examples/serving/openai_flux1_dev.yaml
curl --fail-with-body http://127.0.0.1:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"flux1-dev","prompt":"a photo of a cat","size":"1024x1024","response_format":"b64_json"}'
```

SD3.5 Medium is gated; accept its license and authenticate before starting that
server. The FLUX defaults come from the loaded model's sampling-param class, so
omitting `num_inference_steps` or `guidance_scale` uses the model's own defaults,
not a generic image default. The image routes are not mounted by the native MLX
H3/Wan server.

Audio uses the same video-job routes so one polling and deletion contract works:
submit to `POST /v1/videos`, poll `GET /v1/videos/{id}`, then download
`GET /v1/videos/{id}/content`. Check `media_type`; a WAV job returns
`audio/wav`, not `video/mp4`. Set `seconds`, or
`audio_start_in_s`/`audio_end_in_s`; do not send image geometry fields.

```bash
FASTVIDEO_MODEL=stable-audio FASTVIDEO_SECONDS=6 \
  python examples/serving/clients/audio.py
FASTVIDEO_MODEL=mmaudio-large-44k-v2 \
FASTVIDEO_VIDEO_REFERENCE=/path/to/clip.mp4 \
  python examples/serving/clients/audio.py
```

Node and cURL versions are in `examples/serving/clients/audio.mjs` and
`audio.sh`.
MMAudio requires `video_reference`; Stable Audio is text-to-audio. The
[serving compatibility matrix](serving-compatibility.md) lists which families
target each workload, and the [coverage ledger](serving-coverage.md) records
which baselines exist.

## Connect your app

Set `FASTVIDEO_BASE_URL` to the FastVideo endpoint, including `/v1`, and
`FASTVIDEO_MODEL` to its advertised model alias. The examples default to
`http://127.0.0.1:8000/v1` and `fasth3`.

For a remote GPU machine, keep the server bound to loopback and forward the
port. Replace `user@gpu-host` with your SSH destination:

```bash
ssh -N -L 8000:127.0.0.1:8000 user@gpu-host
```

Then open `http://127.0.0.1:8000/playground/` on your computer. The same forwarded
address works for the cURL and SDK clients. If local port 8000 is occupied, use
`-L 8001:127.0.0.1:8000`, open port 8001, and set `FASTVIDEO_BASE_URL` to
`http://127.0.0.1:8001/v1` for the example clients.

Your webapp backend can submit a job and return its ID to the browser. Poll
from the backend, then proxy the completed download or store it in your own
artifact store. Do not hold a browser request open for the entire generation.

The client key `local` is a placeholder required by the SDK, not server
authentication. FastVideo's HTTP server has no built-in API-key check. Before
public deployment, put it behind an authenticated TLS proxy with restricted
origins, request limits, and access controls. If your proxy uses bearer tokens,
set `FASTVIDEO_API_KEY` on your backend. Never reuse an OpenAI cloud key here.

## Compatibility and limits

The pinned clients cover a tested subset of the OpenAI video-job interface.
They do not establish full OpenAI API compatibility. The compatibility tests
send real HTTP requests to the pinned SDKs against a fake generator, so they
establish client and protocol behavior, not GPU output quality, latency, or
memory use. The server configuration still needs a recorded generation run
before it can be marked Verified in the cookbook.

| Operation | Endpoint |
| --- | --- |
| List served models | `GET /v1/models` |
| Create a video or audio job | `POST /v1/videos` |
| Retrieve status, including a failed job | `GET /v1/videos/{id}` |
| List jobs | `GET /v1/videos` |
| Download MP4, WAV, PNG, JPEG, or WebP content | `GET /v1/videos/{id}/content`, `GET /v1/images/{id}/content` |
| Delete a job and its artifact | `DELETE /v1/videos/{id}` |
| Generate an image | `POST /v1/images/generations`, `POST /v1/images` |
| Edit an image | `POST /v1/images/edits` |
| Synchronous media response | `POST /v1/videos/sync` (FastVideo extension) |
| Liveness and engine health | `GET /health` |

Failed async jobs return HTTP 200 with `status: "failed"` and an `error` object,
so SDK polling can stop. Invalid requests return HTTP 400 with the OpenAI-style
error envelope even when FastAPI's validator rejected the body; the OpenAPI
document therefore removes FastAPI's default 422. Missing jobs return 404,
synchronous generation failures return 500, and unhealthy `/health` returns
503. The video download endpoint supports only the `video` variant, not
thumbnails or spritesheets.

Chat Completions, Responses, Embeddings, Files, Batches, fine-tuning, vector
stores, assistants, realtime, and audio transcription/translation/speech routes
are not implemented. Remix, extensions, characters, thumbnail and spritesheet
operations are not implemented. There is no OpenAI streaming video mode or SSE
stream; async jobs are polled. `/v1/videos/sync` is a FastVideo extension and is
not used by the pinned client examples.

The generated HTTP schema is checked in at
[`docs/openapi/fastvideo-openai.json`](../openapi/fastvideo-openai.json); a
running server serves the same document at `/openapi.json`. It is derived from
the mounted FastAPI routes and is not a hand-maintained second route table.
[HTTP serving compatibility](serving-compatibility.md) records the exact tested
client versions, extension fields, capability matrix, and unsupported features.

The server has no built-in API-key check. The examples' `local` key is a
placeholder required by the SDK, not authentication; the default CORS policy is
permissive. For remote access, use authenticated TLS, restrict origins and
request rates, and do not expose an inference server directly to the public
internet. Local and cluster media URLs are a trusted surface; URL downloads are
bounded by `server.media_reference_max_bytes` (default 2 GiB), and server-owned
copies are removed after the job.

The server serializes generation through one loaded pipeline. Job metadata is
in memory and is lost on restart. Async artifacts remain on disk until deleted
through the API; establish a retention policy for production use. Deleting an
in-progress job removes the API resource but does not interrupt an already
running CUDA call.

For model-specific request fields and reference media, see the
[HTTP contract](../design/server_contracts/openai.md). OpenAI marks its hosted
Sora API as deprecated; FastVideo runs its own models, and the
[OpenAI video API reference](https://developers.openai.com/api/reference/python/resources/videos/methods/create)
describes the client interface, not FastVideo model availability.
