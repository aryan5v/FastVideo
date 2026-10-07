# SPDX-License-Identifier: Apache-2.0
"""Generate FastH3 V2 latents (pre-VAE) for decoder comparisons.

Runs a FastH3 run config (e.g. ``examples/inference/basic/basic_fasth3_spark_v2_nvfp4.yaml``)
with ``output_type="latent"`` so the pipeline stops before the video VAE, and
saves each clip's raw (denormalized) NCTHW latent to ``<out>/<name>.pt`` plus a
``manifest.json`` with prompt, seed and geometry. Decode them with
``bench_decoder.py --latents '<out>/*.pt'``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# Hard cases for a decoder: faces, small text, high-frequency texture, water,
# point lights at night, and fast motion.
PROMPTS = {
    "portrait": "Close-up portrait of an elderly woman with freckles and silver hair laughing near a sunlit window, "
    "fine skin texture and individual strands of hair visible.",
    "signage": "A rainy Tokyo street at dusk lined with shop signs and small printed menus, people with umbrellas "
    "walking past, wet reflections on the pavement.",
    "foliage": "A slow dolly through a dense fern forest, sunlight filtering through thousands of small leaves, "
    "dew drops on moss.",
    "ocean": "Waves crashing against black volcanic rocks, white foam and spray in slow motion, overcast sky.",
    "night_city": "A drone shot over a city at night, thousands of window lights and car headlights, light trails "
    "on a highway interchange.",
    "skateboard": "A skateboarder performing a fast kickflip down a set of concrete stairs, handheld camera "
    "following the action.",
    "kitchen": "A chef chopping colorful vegetables quickly on a wooden board, steam rising from pans in the "
    "background.",
    "crowd": "A crowded stadium during a concert, confetti falling, stage lights sweeping across the audience.",
    "fabric": "A slow pan across a stack of knitted sweaters and patterned scarves, detailed wool texture.",
}


def build_generator(config_path: str, model_path: str | None):
    from fastvideo.api.parser import load_raw_config
    from fastvideo.api.compat import generator_config_to_fastvideo_args, normalize_generator_config
    from fastvideo.entrypoints.video_generator import VideoGenerator

    raw = load_raw_config(config_path)
    generator_raw = dict(raw["generator"])
    if model_path:
        generator_raw["model_path"] = model_path
    fastvideo_args = generator_config_to_fastvideo_args(normalize_generator_config(generator_raw))
    fastvideo_args.output_type = "latent"
    return VideoGenerator.from_fastvideo_args(fastvideo_args), raw["request"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="FastH3 run config (generator + request)")
    parser.add_argument("--model-path", help="override generator.model_path (e.g. a local snapshot)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resolutions", default="832x480,1344x768", help="comma list of WxH")
    parser.add_argument("--prompts", default=",".join(PROMPTS), help="comma list of prompt names")
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generator, request = build_generator(args.config, args.model_path)
    sampling = dict(request.get("sampling", {}))
    manifest = []
    for resolution in args.resolutions.split(","):
        width, height = (int(v) for v in resolution.lower().split("x"))
        for name in args.prompts.split(","):
            clip = f"v2_{name}_{width}x{height}"
            result = generator.generate(
                request={
                    "prompt": PROMPTS[name],
                    "negative_prompt": request.get("negative_prompt", ""),
                    "sampling": {
                        **sampling, "seed": args.seed,
                        "height": height,
                        "width": width,
                        "num_frames": args.num_frames
                    },
                    "output": {
                        "save_video": False,
                        "return_frames": True
                    },
                })
            latents = result.samples
            if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
                raise RuntimeError(f"expected NCTHW latents for {clip}, got {type(latents)}")
            torch.save(latents.float().cpu(), out_dir / f"{clip}.pt")
            manifest.append({
                "clip": clip,
                "prompt": PROMPTS[name],
                "seed": args.seed,
                "width": width,
                "height": height,
                "num_frames": args.num_frames,
                "latent_shape": list(latents.shape),
            })
            print(json.dumps(manifest[-1]), flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    generator.shutdown()


if __name__ == "__main__":
    main()
