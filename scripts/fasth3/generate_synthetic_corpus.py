# SPDX-License-Identifier: Apache-2.0
"""FastH3 synthetic corpus generator: teacher-generated clips for DMD.

DMD2 matches a distribution — the critic needs the teacher's score over
prompts, not just real pairs. FastWan distilled against Wan-Syn's 600k
synthetic samples for exactly this reason. Generating from our own 33B
teacher gives:

- an exactly-matched teacher distribution (coherent AV pairing by
  construction — the teacher generates both),
- prompt coverage we control (prompt set is the corpus design),
- scaling with GPU time instead of dataset licensing.

Flow:
1. prompts.jsonl:  {"id": "...", "prompt": "..."}  (curated set)
2. For each prompt: run the FastVideo H3 pipeline (50-step teacher,
   480p-class, 5s) -> mp4 (video + muxed stereo audio)
3. The output manifest feeds `preprocess_h3_corpus.py` phase A (latents)
   and phase C (text embeddings from the prompts) — the captions ARE the
   prompts, so no caption pass is needed for synthetic clips.

Usage (cluster, 1 node):
  $PY examples/inference/basic/basic_minimax_h3_t2v.py-style loop, or:
  $PY scripts/fasth3/generate_synthetic_corpus.py --prompts prompts.jsonl \
      --out <synth_dir> --num-gpus 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, help="jsonl: id, prompt")
    parser.add_argument("--out", required=True, help="synth corpus dir (videos/ + manifest)")
    parser.add_argument("--model-path", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=124, help="17n+5 window (124 = 5s)")
    parser.add_argument("--steps", type=int, default=50, help="keep the teacher's full schedule")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-clips", type=int, default=0, help="0 = all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from fastvideo import VideoGenerator  # noqa: PLC0415
    from fastvideo.api import (  # noqa: PLC0415
        EngineConfig,
        GenerationRequest,
        GeneratorConfig,
        OffloadConfig,
        OutputConfig,
        ParallelismConfig,
        SamplingConfig,
    )

    prompts = [json.loads(line) for line in open(args.prompts, encoding="utf-8")]
    if args.max_clips:
        prompts = prompts[: args.max_clips]
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    prompts = prompts[rank::world_size]

    out_dir = Path(args.out)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"synthetic_manifest_rank{rank}.jsonl"

    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path=args.model_path,
            engine=EngineConfig(
                num_gpus=args.num_gpus,
                use_fsdp_inference=args.num_gpus > 1,
                parallelism=ParallelismConfig(tp_size=1, sp_size=args.num_gpus),
                offload=OffloadConfig(dit=False, dit_layerwise=False, text_encoder=True, vae=True,
                                      pin_cpu_memory=False),
            ),
        ))
    try:
        with open(manifest_path, "w", encoding="utf-8") as fh:
            for index, item in enumerate(prompts):
                clip_id = item["id"]
                video_path = videos_dir / f"{clip_id}.mp4"
                if video_path.exists():
                    continue
                result = generator.generate(
                    GenerationRequest(
                        prompt=item["prompt"],
                        negative_prompt="",
                        sampling=SamplingConfig(
                            height=args.height,
                            width=args.width,
                            num_frames=args.num_frames,
                            fps=24,
                            num_inference_steps=args.steps,
                            guidance_scale=1.0,
                            batch_cfg=False,
                            seed=args.seed + index,
                        ),
                        output=OutputConfig(
                            output_path=str(video_path),
                            save_video=True,
                            return_frames=False,
                        ),
                    ))
                fh.write(json.dumps({"clip_id": clip_id, "video": str(video_path),
                                     "caption": item["prompt"]}) + "\n")
                fh.flush()
                print(f"[rank {rank}] generated {clip_id} ({index}/{len(prompts)}) -> {result.video_path}",
                      flush=True)
    finally:
        generator.shutdown()
    print(f"[rank {rank}] SYNTH DONE: {len(prompts)} clips", flush=True)


if __name__ == "__main__":
    main()
