# SPDX-License-Identifier: Apache-2.0
"""Pipelined two-node MiniMax-H3 inference prototype (e.g. two DGX Sparks over their QSFP link).

The generation node runs only the DiT denoise; the I/O node runs text encoding,
latent preparation, video and audio decoding and MP4 writing. Clips are
pipelined: while the generation node denoises clip N+1, the I/O node decodes
and writes clip N. H3 generates whole clips (no chunked/causal generation), so
the pipelining unit is one clip.

Start the generation node first, then the I/O node::

    # generation node
    python scripts/inference/minimax_h3_two_node_pipeline.py --role gen \
        --config examples/inference/basic/basic_fasth3_spark_v2_nvfp4.yaml --model-path STACK \
        --bind 0.0.0.0:29700
    # I/O node
    python scripts/inference/minimax_h3_two_node_pipeline.py --role io \
        --config examples/inference/basic/basic_fasth3_spark_v2_nvfp4.yaml --model-path STACK \
        --peer GEN_HOST:29700 --prompts prompts.json --clips 6 --output-dir outputs/two_node

The role pipelines follow the encoder/decoder and DiT split proposed for
component-disaggregated H3 inference; transport is a plain length-prefixed TCP
stream of CPU tensors (loaded with ``weights_only=True``). The I/O node prints
one JSON line per clip and a summary with per-clip latency, each node's busy
time and steady-state throughput (seconds of video per wall-clock second).
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import json
import os
import queue
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any

import torch

_HEADER = struct.Struct("!Q")


# --------------------------------------------------------------------------- transport
def _send(sock: socket.socket, payload: dict[str, Any]) -> tuple[int, float]:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    start = time.perf_counter()
    sock.sendall(_HEADER.pack(len(data)) + data)
    return len(data), time.perf_counter() - start


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks, remaining = [], size
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 22))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv(sock: socket.socket) -> tuple[dict[str, Any] | None, float, float]:
    """Return (payload, seconds waiting for the first byte, seconds transferring)."""
    wait_start = time.perf_counter()
    header = sock.recv(_HEADER.size, socket.MSG_WAITALL)
    if not header:
        return None, time.perf_counter() - wait_start, 0.0
    first_byte = time.perf_counter()
    data = _recv_exact(sock, _HEADER.unpack(header)[0])
    done = time.perf_counter()
    return torch.load(io.BytesIO(data), weights_only=True), first_byte - wait_start, done - first_byte


def _cpu(value: Any) -> Any:
    return value.detach().cpu().contiguous() if isinstance(value, torch.Tensor) else value


def _layout_to_wire(layout: Any) -> dict[str, Any]:
    fields = {field.name: _cpu(getattr(layout, field.name)) for field in dataclasses.fields(layout)}
    fields["reference_segments"] = [list(segment) for segment in fields["reference_segments"]]
    return fields


def _layout_from_wire(fields: dict[str, Any], device: torch.device) -> Any:
    from fastvideo.pipelines.basic.minimax_h3.packing import MiniMaxH3PackedLayout

    values = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in fields.items()}
    values["reference_segments"] = tuple((kind, rows, tuple(shape)) for kind, rows, shape in values["reference_segments"])
    return MiniMaxH3PackedLayout(**values)


# --------------------------------------------------------------------------- role pipelines
def _role_pipelines():
    from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import (
        MiniMaxH3BasePipeline,
        _apply_h3_checkpoint_arch_configs,
    )
    from fastvideo.pipelines.basic.minimax_h3.stages import (
        MiniMaxH3AudioDecodingStage,
        MiniMaxH3DenoisingStage,
        MiniMaxH3LatentPreparationStage,
        MiniMaxH3VideoDecodingStage,
    )

    class _Resident(MiniMaxH3BasePipeline):
        _lazy_module_names: tuple[str, ...] = ()

        def _defer_denoise_modules(self, fastvideo_args) -> bool:
            return False

        def initialize_pipeline(self, fastvideo_args) -> None:
            _apply_h3_checkpoint_arch_configs(self.model_path, fastvideo_args, self._extra_config_module_map)

        def run(self, names: tuple[str, ...], batch):
            if not self.post_init_called:
                self.post_init()
            for name in names:
                batch = self._stage_name_mapping[name](batch, self.fastvideo_args)
            return batch

    class IOPipeline(_Resident):
        """Text encoder + video/audio VAEs: encode before denoise, decode after."""

        _required_config_modules = ["text_encoder", "tokenizer", "processor", "vae", "audio_vae", "scheduler"]

        def create_pipeline_stages(self, fastvideo_args) -> None:
            self._add_condition_stages(fastvideo_args, ref2va=False)
            vae, audio_vae = self.get_module("vae"), self.get_module("audio_vae")
            self.add_stage(
                "latent_preparation_stage",
                MiniMaxH3LatentPreparationStage(vae=vae,
                                                audio_vae=audio_vae,
                                                scheduler=self.get_module("scheduler"),
                                                ref2va=False))
            self.add_stage("video_decoding_stage", MiniMaxH3VideoDecodingStage(vae=vae))
            self.add_stage("audio_decoding_stage", MiniMaxH3AudioDecodingStage(audio_vae=audio_vae))

    class GenPipeline(_Resident):
        """DiT only."""

        _required_config_modules = ["transformer", "scheduler", "audio_scheduler"]

        def create_pipeline_stages(self, fastvideo_args) -> None:
            self.add_stage(
                "denoising_stage",
                MiniMaxH3DenoisingStage(transformer=self.get_module("transformer"),
                                        scheduler=self.get_module("scheduler"),
                                        audio_scheduler=self.get_module("audio_scheduler")))

    return IOPipeline, GenPipeline


def _build(role: str, args: argparse.Namespace):
    from fastvideo.api.compat import generator_config_to_fastvideo_args
    from fastvideo.api.parser import load_raw_config, parse_config
    from fastvideo.api.schema import RunConfig
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel

    for name, value in (("LOCAL_RANK", "0"), ("RANK", "0"), ("WORLD_SIZE", "1"), ("MASTER_ADDR", "127.0.0.1"),
                        ("MASTER_PORT", str(args.local_port))):
        os.environ.setdefault(name, value)
    config = parse_config(RunConfig, load_raw_config(Path(args.config)))
    config.generator.model_path = args.model_path
    if role == "io" and not args.compile_vae:
        config.generator.engine.compile.vae_enabled = False
    fastvideo_args = generator_config_to_fastvideo_args(config.generator)
    torch.cuda.set_device(0)
    fastvideo_args.finalize_device_offload_policy(0)
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    io_cls, gen_cls = _role_pipelines()
    pipeline = (io_cls if role == "io" else gen_cls)(args.model_path, fastvideo_args)
    return pipeline, config.request, fastvideo_args


# --------------------------------------------------------------------------- generation node
def run_gen(args: argparse.Namespace) -> None:
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    pipeline, _, fastvideo_args = _build("gen", args)
    device = torch.device("cuda")
    host, port = args.bind.rsplit(":", 1)
    server = socket.create_server((host, int(port)))
    print(json.dumps({"event": "gen_ready", "bind": args.bind}), flush=True)
    conn, _ = server.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    # Drain requests on a thread: a request and a reply larger than the socket buffers would
    # otherwise block both nodes in sendall.
    inbox: queue.Queue = queue.Queue()

    def reader() -> None:
        while True:
            message, wait_s, recv_s = _recv(conn)
            inbox.put((message, recv_s))
            if message is None or message.get("stop"):
                return

    threading.Thread(target=reader, daemon=True).start()
    while True:
        wait_start = time.perf_counter()
        message, recv_s = inbox.get()
        wait_s = time.perf_counter() - wait_start
        if message is None or message.get("stop"):
            break
        start = time.perf_counter()
        batch = ForwardBatch(data_type="video",
                             prompt_embeds=[message["prompt_embeds"].to(device)],
                             latents=message["video_latents"].to(device),
                             audio_latents=message["audio_latents"].to(device),
                             raw_latent_shape=tuple(message["raw_latent_shape"]),
                             num_inference_steps=int(message["num_inference_steps"]),
                             VSA_sparsity=float(message["vsa_sparsity"]),
                             extra={MINIMAX_H3_LAYOUT_KEY: _layout_from_wire(message["layout"], device)})
        batch = pipeline.run(("denoising_stage", ), batch)
        torch.cuda.synchronize()
        denoise_s = time.perf_counter() - start
        reply = {
            "clip": message["clip"],
            "video_latents": _cpu(batch.latents),
            "audio_latents": _cpu(batch.audio_latents),
            "timing": {
                "gen_wait_s": round(wait_s, 3),
                "gen_recv_s": round(recv_s, 4),
                "gen_denoise_s": round(denoise_s, 3)
            },
        }
        size, send_s = _send(conn, reply)
        reply["timing"]["gen_send_s"] = round(send_s, 4)
        print(json.dumps({"event": "denoised", "clip": message["clip"], **reply["timing"], "reply_mb": size / 2**20}),
              flush=True)
    del fastvideo_args
    conn.close()


# --------------------------------------------------------------------------- I/O node
def _encode(pipeline, request, fastvideo_args, prompt: str, args: argparse.Namespace) -> tuple[dict, Any]:
    from copy import deepcopy

    from fastvideo.api.compat import request_to_sampling_param
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from fastvideo.utils import shallow_asdict

    req = deepcopy(request)
    req.prompt = prompt
    req.inputs.prompt_path = None
    req.sampling.width, req.sampling.height, req.sampling.num_frames = args.width, args.height, args.frames
    sampling = request_to_sampling_param(req, model_path=args.model_path)
    sampling.prompt = prompt
    batch = ForwardBatch(**shallow_asdict(sampling), eta=0.0, n_tokens=0, VSA_sparsity=fastvideo_args.VSA_sparsity)
    batch = pipeline.run(("input_preparation_stage", "conditioning_stage", "latent_preparation_stage"), batch)
    layout = batch.extra[MINIMAX_H3_LAYOUT_KEY]
    message = {
        "prompt_embeds": _cpu(batch.prompt_embeds[0]),
        "video_latents": _cpu(batch.latents),
        "audio_latents": _cpu(batch.audio_latents),
        "layout": _layout_to_wire(layout),
        "raw_latent_shape": list(batch.raw_latent_shape),
        "num_inference_steps": int(batch.num_inference_steps),
        "vsa_sparsity": float(batch.VSA_sparsity),
    }
    return message, (layout, tuple(batch.raw_latent_shape), int(sampling.fps))


def _decode_and_save(pipeline, reply: dict, context: Any, path: Path) -> dict[str, float]:
    from fastvideo.entrypoints.video_generator import VideoGenerator
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    layout, raw_latent_shape, fps = context
    device = torch.device("cuda")
    start = time.perf_counter()
    batch = ForwardBatch(data_type="video",
                         latents=reply["video_latents"].to(device),
                         audio_latents=reply["audio_latents"].to(device),
                         raw_latent_shape=raw_latent_shape,
                         extra={MINIMAX_H3_LAYOUT_KEY: layout})
    batch = pipeline.run(("video_decoding_stage", ), batch)
    torch.cuda.synchronize()
    video_s = time.perf_counter() - start
    batch = pipeline.run(("audio_decoding_stage", ), batch)
    audio_s = time.perf_counter() - start - video_s
    save_start = time.perf_counter()
    video = batch.output[0]  # [3, T, H, W] uint8
    frames = [frame.numpy() for frame in video.permute(1, 2, 3, 0).contiguous()]
    VideoGenerator._save_video_with_audio_single_pass(output_path=str(path),
                                                      frames=frames,
                                                      fps=fps,
                                                      audio=batch.extra["audio"],
                                                      sample_rate=int(batch.extra["audio_sample_rate"]))
    return {"io_video_decode_s": round(video_s, 3), "io_audio_decode_s": round(audio_s, 3),
            "io_save_s": round(time.perf_counter() - save_start, 3)}


def run_io(args: argparse.Namespace) -> None:
    prompts = json.loads(Path(args.prompts).read_text())
    prompt_list = list(prompts.values()) if isinstance(prompts, dict) else list(prompts)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline, request, fastvideo_args = _build("io", args)
    host, port = args.peer.rsplit(":", 1)
    for _ in range(120):
        try:
            sock = socket.create_connection((host, int(port)))
            break
        except OSError:
            time.sleep(5)
    else:
        raise ConnectionError(f"could not reach the generation node at {args.peer}")
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    clip_seconds = args.frames / args.fps
    sent: dict[int, dict[str, Any]] = {}
    contexts: dict[int, Any] = {}
    rows = []

    def submit(clip: int) -> None:
        start = time.perf_counter()
        message, contexts[clip] = _encode(pipeline, request, fastvideo_args, prompt_list[clip % len(prompt_list)], args)
        torch.cuda.synchronize()
        encode_s = time.perf_counter() - start
        message["clip"] = clip
        size, send_s = _send(sock, message)
        sent[clip] = {"t_submit": start, "io_encode_s": round(encode_s, 3), "io_send_s": round(send_s, 4),
                      "request_mb": round(size / 2**20, 2)}

    t0 = time.perf_counter()
    for clip in range(min(args.depth, args.clips)):
        submit(clip)
    for clip in range(args.clips):
        reply, wait_s, recv_s = _recv(sock)
        t_reply = time.perf_counter()
        timing = _decode_and_save(pipeline, reply, contexts.pop(clip), out_dir / f"clip{clip:03d}.mp4")
        t_done = time.perf_counter()
        if clip + args.depth < args.clips:
            submit(clip + args.depth)
        row = {"clip": clip, **sent.pop(clip), **reply["timing"], **timing, "io_wait_s": round(wait_s, 3),
               "io_recv_s": round(recv_s, 4), "t_reply": round(t_reply - t0, 3), "t_done": round(t_done - t0, 3)}
        row["latency_s"] = round(t_done - row.pop("t_submit"), 3)
        rows.append(row)
        print(json.dumps(row), flush=True)
    _send(sock, {"stop": True})
    steady = rows[args.warmup:]
    summary: dict[str, Any] = {"clips": len(rows), "clip_video_s": round(clip_seconds, 3)}
    if len(steady) >= 2:
        span = steady[-1]["t_done"] - steady[0]["t_done"]
        summary["steady_s_per_clip"] = round(span / (len(steady) - 1), 3)
        summary["video_s_per_wall_s"] = round(clip_seconds * (len(steady) - 1) / span, 4)
    for key in ("latency_s", "gen_denoise_s", "io_encode_s", "io_video_decode_s", "io_audio_decode_s", "io_save_s",
                "io_send_s", "gen_recv_s", "io_recv_s"):
        values = sorted(row[key] for row in steady) if steady else []
        if values:
            summary[f"median_{key}"] = values[len(values) // 2]
    if steady:
        busy = [row["io_encode_s"] + row["io_video_decode_s"] + row["io_audio_decode_s"] + row["io_save_s"]
                for row in steady]
        summary["median_io_busy_s"] = round(sorted(busy)[len(busy) // 2], 3)
    print(json.dumps({"summary": summary}), flush=True)
    (out_dir / "rows.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=("io", "gen"), required=True)
    parser.add_argument("--config", required=True, help="FastVideo run config (YAML) of the H3 stack")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--bind", default="0.0.0.0:29700", help="gen: HOST:PORT to listen on")
    parser.add_argument("--peer", help="io: generation node HOST:PORT")
    parser.add_argument("--local-port", type=int, default=29650, help="this process's torch.distributed port")
    parser.add_argument("--prompts", help="io: JSON list or {id: prompt} of prompts, cycled over clips")
    parser.add_argument("--clips", type=int, default=6)
    parser.add_argument("--depth", type=int, default=2, help="io: clips in flight (2 keeps the DiT busy)")
    parser.add_argument("--warmup", type=int, default=2, help="io: leading clips excluded from the summary")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--fps", type=float, default=24.0, help="io: output frame rate, for the throughput figure")
    parser.add_argument("--compile-vae", action="store_true", help="io: keep the config's VAE compile")
    parser.add_argument("--output-dir", default="outputs/minimax_h3_two_node")
    args = parser.parse_args()
    if args.role == "io":
        if not args.peer or not args.prompts:
            parser.error("--role io needs --peer and --prompts")
        run_io(args)
    else:
        run_gen(args)


if __name__ == "__main__":
    main()
