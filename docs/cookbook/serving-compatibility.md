# HTTP serving compatibility

This page records what the OpenAI-compatible REST surface actually does. It is
a behavior and capability reference, not a claim that FastVideo implements the
whole OpenAI API. Model availability still depends on the checked-in serving
configuration and its hardware path; the schema cannot prove that a loaded
pipeline supports an operation.

The machine-readable contract is generated from the mounted FastAPI routes:
`docs/openapi/fastvideo-openai.json`. A running server serves the same document
at `/openapi.json`. See [the serving coverage ledger](serving-coverage.md) for
baselines and [the HTTP contract](../design/server_contracts/openai.md) for
request-field details.

## Tested client surface

The checked-in clients and protocol tests cover the following surface. They use
a fake generator, so they establish request/response behavior, not GPU output.

| Client | Pinned version | Covered behavior | Not covered here |
| --- | --- | --- | --- |
| OpenAI Python | `openai==3.6.0` | `videos.create`, `retrieve`, `list`, `downloadContent`, `delete` | Image and audio helpers, streaming, files, batches |
| OpenAI JavaScript | `openai@7.8.0` | `videos.create`, `retrieve`, `downloadContent` | Image and audio helpers, streaming, files, batches |
| cURL + jq | System tools | Submit, poll, download, delete, error handling | SDK-specific retries or typing |

These pins are what the repository examples install. A different SDK release
may change or remove video helpers; check its changelog before treating it as
tested.

## Routes

| Method | Route | Success | Documented failures |
| --- | --- | --- | --- |
| `GET` | `/v1/models` | Model list | None in normal operation |
| `GET` | `/v1/models/{model}` | Model card | 404 unknown |
| `GET` | `/v1/model_info` | Model metadata | None in normal operation |
| `POST` | `/v1/videos` | `queued` video job | 400 invalid or unsupported request |
| `POST` | `/v1/videos/sync` | MP4 or WAV body | 400 invalid request, 500 generation failure |
| `GET` | `/v1/videos` | Job list | 400 invalid `order` or pagination |
| `GET` | `/v1/videos/{id}` | Job status; failed jobs are HTTP 200 | 404 unknown |
| `DELETE` | `/v1/videos/{id}` | Deleted job | 404 unknown |
| `GET` | `/v1/videos/{id}/content` | MP4 or WAV body | 400 unsupported variant, 404 missing or incomplete |
| `POST` | `/v1/images/generations` | Image response | 400 invalid request, 500 generation failure |
| `POST` | `/v1/images` | Image response | 400 invalid request, 500 generation failure |
| `POST` | `/v1/images/edits` | Image response | 400 invalid request or media, 500 generation failure |
| `GET` | `/v1/images/{id}/content` | PNG, JPEG, or WebP body | 404 missing or incomplete |
| `GET` | `/health` | `{"status":"ok"}` | 503 engine unavailable |

`POST /v1/videos/generations` is a hidden compatibility alias for `/v1/videos`.
`POST /v1/videos/sync` is a FastVideo extension. Deleting a queued job removes
it immediately; deleting an in-progress job removes the API resource while the
uninterruptible generation call finishes.

## Capability matrix

`✅` means the checked-in serving baseline targets that workload. It is a
source-backed declaration, not a GPU run. `—` means no maintained baseline.
The loaded pipeline still validates every request at admission.

| Family or baseline | `t2v` | `i2v` | `t2i` | `v2a` | `t2a` |
| --- | --- | --- | --- | --- | --- |
| FastH3 | ✅ (+ audio output) | — | — | — | — |
| Wan2.2 T2V/I2V A14B | ✅ | ✅ | — | — | — |
| Wan2.1 I2V 14B | — | ✅ | — | — | — |
| HunyuanVideo 1.5 | ✅ | ✅ | — | — | — |
| Kandinsky 6 TI2VA | ✅ (+ audio output) | — | — | — | — |
| LTX2 Distilled | ✅ (+ audio output) | — | — | — | — |
| FLUX.1 / FLUX.2 / SD3.5 | — | — | ✅ | — | — |
| MMAudio | — | — | — | ✅ | — |
| Stable Audio | — | — | — | — | ✅ |

FastH3 also accepts its `t2va`, `fl2va` and `ref2va` task spellings. The
[capability ledger](serving-coverage.md) lists families without a baseline,
including Cosmos, GEN3C, Helios, LingBot and MatrixGame. A model appearing in
the support matrix does not automatically make every route or workload above
available.

## FastVideo extensions

These fields and routes are part of FastVideo's contract but are not claimed as
OpenAI API behavior:

- `video_reference`, `image_reference`, `audio_reference`, `input_reference`,
  `reference_url`, `video_path` and `video_url` reference fields.
- `audio_start_in_s` and `audio_end_in_s`; `media_type` on job responses.
- `extra_params`, `video_params`, `lora`, `n`, `num_outputs_per_prompt`,
  `guidance_scale_2`, `boundary_ratio`, `flow_shift`, frame interpolation, and
  the other model-specific sampling fields in the generated schema.
- `POST /v1/videos/sync` and the `/v1/videos/generations` alias.
- `file_path` and `peak_memory_mb` response metadata.

Reference objects with `file_id` are accepted by the transport schema but
return HTTP 400 because FastVideo does not implement an OpenAI Files store.
Unknown top-level fields are rejected; unsupported model-specific fields fail
at admission rather than creating a job that fails later.

## Not implemented

The server does not implement Chat Completions, Responses, Embeddings,
Assistants, Files, Batches, Fine-tuning, Vector Stores, Realtime, audio
transcriptions/translations, or text-to-speech routes. It does not implement
Sora remix, extension, character, thumbnail, or spritesheet operations. There
is no Server-Sent Events stream and no OpenAI streaming video mode. The
asynchronous video job is polled with `GET /v1/videos/{id}`; WebSocket
streaming is a separate FastVideo runtime and is not part of this REST contract.

## Errors, auth, and media

Invalid JSON, Pydantic validation failures, unsupported fields and missing
model references return the OpenAI-style envelope with HTTP 400:

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": null, "code": 400}}
```

Missing resources return 404, synchronous generation failures return 500, and
`/health` returns 503 when the engine is unavailable. An asynchronous job that
fails after admission remains HTTP 200 with `status: "failed"` and a string
error code, so SDK polling stops instead of retrying a transport error.

The server has no built-in API-key check. The examples' `local` key is a
placeholder, and the default CORS policy is permissive. For remote access, put
the server behind authenticated TLS and restrict origins and request rates.
Do not add a public inference endpoint or proxy to make the examples work.

Local and cluster media URLs are a trusted deployment surface. URL downloads
are capped by `server.media_reference_max_bytes` (default 2 GiB), and
server-owned copies are removed after the job. This is not an untrusted-URL
sanitizer.

## Verification boundary

The HTTP and client tests use a fake generator and run without loading weights.
They establish routing, status codes, envelopes, multipart and binary handling,
and client polling. They do not establish GPU generation quality, latency,
memory use, or a hardware fit. The OpenAPI document is route-derived; generated
schema presence is not runtime evidence.
