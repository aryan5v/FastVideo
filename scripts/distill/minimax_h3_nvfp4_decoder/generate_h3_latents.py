# SPDX-License-Identifier: Apache-2.0
"""Generate FastH3 DiT latents as decoder-distillation data.

Runs a FastH3 run config (generator + request) with ``output_type="latent"``,
one process per GPU, and writes parquet shards in FastVideo's record schema
(``vae_latent_bytes/shape/dtype``, latents normalized with the VAE's latent
mean/std, like the preprocessed H3 datasets) so ``train_qad.py`` reads them with
``--latents-normalized yes``. Prompts are sampled deterministically from JSONL
files; each clip draws a resolution from ``--resolutions``. Re-running skips
clips already written.

Example (one GPU of four)::

    python generate_h3_latents.py --config examples/inference/basic/basic_fasth3_spark_v2_nvfp4.yaml \\
        --tag v2_nvfp4 --prompts '/data/*/prompts/source.jsonl' --num-clips 1000 --shard 0 --num-shards 4 \\
        --output-dir /data/h3_generated_latents
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

FLUSH_EVERY = 8


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        merged[key] = deep_merge(merged[key], value) if isinstance(value, dict) and isinstance(
            merged.get(key), dict) else value
    return merged


def load_prompts(pattern: str) -> list[str]:
    prompts = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            prompts.extend(json.loads(line)["prompt"] for line in handle if line.strip())
    if not prompts:
        raise FileNotFoundError(f"no prompts in {pattern}")
    return prompts


def parse_weighted(spec: str) -> list[tuple[tuple[int, int], float]]:
    """``"832x480:0.5,1344x768:0.5"`` -> [((832, 480), 0.5), ...]."""
    items = []
    for item in spec.split(","):
        size, _, weight = item.partition(":")
        width, height = (int(v) for v in size.lower().split("x"))
        items.append(((width, height), float(weight or 1)))
    return items


def clip_plan(args: argparse.Namespace, prompts: list[str]) -> list[dict[str, Any]]:
    """The full deterministic plan; every shard takes every ``num_shards``-th entry."""
    rng = random.Random(args.seed)
    sizes = parse_weighted(args.resolutions)
    plan = []
    for index in range(args.num_clips):
        (width, height) = rng.choices([s for s, _ in sizes], weights=[w for _, w in sizes])[0]
        plan.append({
            "id": f"{args.tag}-{index:06d}",
            "prompt": prompts[rng.randrange(len(prompts))],
            "seed": rng.randrange(2**31),
            "width": width,
            "height": height,
            "num_frames": args.num_frames,
        })
    return plan[args.shard::args.num_shards]


def existing_ids(data_dir: Path) -> set[str]:
    import pyarrow.parquet as pq

    done: set[str] = set()
    for path in data_dir.glob("*.parquet"):
        done.update(pq.read_table(path, columns=["id"]).column("id").to_pylist())
    return done


def write_part(data_dir: Path, shard: int, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    stamp = int(time.time() * 1000)
    path = data_dir / f"shard{shard:03d}-{stamp}.parquet"
    tmp = path.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(rows), tmp)
    tmp.replace(path)


def build_generator(args: argparse.Namespace):
    from fastvideo.api.compat import generator_config_to_fastvideo_args, normalize_generator_config
    from fastvideo.api.parser import load_raw_config
    from fastvideo.entrypoints.video_generator import VideoGenerator

    raw = load_raw_config(args.config)
    generator_raw = deep_merge(raw["generator"], json.loads(args.generator_override or "{}"))
    request = deep_merge(raw["request"], json.loads(args.request_override or "{}"))
    fastvideo_args = generator_config_to_fastvideo_args(normalize_generator_config(generator_raw))
    fastvideo_args.output_type = "latent"
    return VideoGenerator.from_fastvideo_args(fastvideo_args), request


def latent_stats(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    config = json.loads((Path(args.vae_dir) / "config.json").read_text())
    mean = torch.tensor(config["latents_mean"], dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = torch.tensor(config["latents_std"], dtype=torch.float32).view(1, -1, 1, 1, 1)
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--generator-override", help="JSON deep-merged into the config's generator block")
    parser.add_argument("--request-override", help="JSON deep-merged into the config's request block")
    parser.add_argument("--tag", required=True, help="generator name, used in clip ids and the output folder")
    parser.add_argument("--vae-dir", required=True, help="any H3 vae/ folder; supplies latents_mean/std")
    parser.add_argument("--prompts", required=True, help="glob of JSONL files with a 'prompt' field")
    parser.add_argument("--num-clips", type=int, required=True, help="clips across all shards")
    parser.add_argument("--resolutions", default="832x480:0.45,1344x768:0.35,1024x768:0.2")
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    data_dir = Path(args.output_dir) / args.tag / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    plan = clip_plan(args, load_prompts(args.prompts))
    done = existing_ids(data_dir)
    todo = [clip for clip in plan if clip["id"] not in done]
    print(json.dumps({"tag": args.tag, "shard": args.shard, "planned": len(plan), "todo": len(todo)}), flush=True)
    if not todo:
        return
    mean, std = latent_stats(args)
    generator, request = build_generator(args)
    sampling = dict(request.get("sampling", {}))
    pending: list[dict[str, Any]] = []
    for clip in todo:
        start = time.perf_counter()
        try:
            result = generator.generate(
                request={
                    "prompt": clip["prompt"],
                    "negative_prompt": request.get("negative_prompt", ""),
                    "sampling": {
                        **sampling, "seed": clip["seed"],
                        "width": clip["width"],
                        "height": clip["height"],
                        "num_frames": clip["num_frames"]
                    },
                    "output": {
                        "save_video": False,
                        "return_frames": True
                    },
                })
        except Exception as error:  # noqa: BLE001 - record and continue; one bad prompt must not stop the shard
            print(json.dumps({"id": clip["id"], "error": repr(error)[:500]}), flush=True)
            continue
        latents = result.samples
        if not isinstance(latents, torch.Tensor) or latents.ndim != 5 or not torch.isfinite(latents).all():
            print(json.dumps({"id": clip["id"], "error": f"bad latents {type(latents)}"}), flush=True)
            continue
        normalized = ((latents.float().cpu() - mean) / std)[0].contiguous()
        pending.append({
            **clip,
            "generator": args.tag,
            "vae_latent_bytes": normalized.numpy().tobytes(),
            "vae_latent_shape": list(normalized.shape),
            "vae_latent_dtype": "float32",
        })
        print(json.dumps({
            "id": clip["id"],
            "shape": list(normalized.shape),
            "seconds": round(time.perf_counter() - start, 1)
        }),
              flush=True)
        if len(pending) >= FLUSH_EVERY:
            write_part(data_dir, args.shard, pending)
            pending = []
    if pending:
        write_part(data_dir, args.shard, pending)
    generator.shutdown()


if __name__ == "__main__":
    main()
