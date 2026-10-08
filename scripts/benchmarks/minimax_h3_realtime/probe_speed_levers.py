# SPDX-License-Identifier: Apache-2.0
"""Training-free speed/quality probes for a FastH3 T2VA checkpoint.

``generate``: for each (variant, prompt) job assigned to this slot, run the
unmodified pipeline with the variant's levers (``h3_levers.py``: block skips,
middle-block token compression) and DMD rung subset, and save the raw
(decode-stage input) latent plus the DiT denoise time.

``evaluate``: decode every saved latent and its prompt's reference with the
pipeline's video VAE (fp16 autocast) and write LPIPS (AlexNet, every 4th frame)
and PSNR (all frames) per run.

``grid``: labelled side-by-side MP4s for chosen prompts and variants.

Variants are a JSON list of ``{"name", "skip": [...], "compress": {"start",
"end", "mode"}, "steps": [...], "prompts": [...]}``; omitted fields mean the
unmodified model; ``"score": true`` also records the per-block residual-change
score (it syncs per block, so its timing is not representative). Runs one rank per process (external launcher), one process
per GPU; ``--slot``/``--num-slots`` split the jobs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "minimax_h3_vae"))

from generate_v2_latents import PROMPTS  # noqa: E402
from h3_levers import CompressSpec, install_levers  # noqa: E402


def build(args: argparse.Namespace, lazy: bool = False):
    from fastvideo.api.compat import generator_config_to_fastvideo_args, normalize_generator_config
    from fastvideo.api.parser import load_raw_config
    from fastvideo.entrypoints.video_generator import VideoGenerator

    for name, value in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0"), ("MASTER_ADDR", "127.0.0.1"),
                        ("MASTER_PORT", str(29700 + args.slot))):
        os.environ.setdefault(name, value)
    raw = load_raw_config(args.config)
    generator_raw = dict(raw["generator"], model_path=args.model_path)
    generator_raw["engine"] = dict(generator_raw["engine"], execution_backend="external_launcher")
    if lazy:
        generator_raw["engine"]["offload"] = dict(generator_raw["engine"]["offload"], lazy_module_load=True)
    fastvideo_args = generator_config_to_fastvideo_args(normalize_generator_config(generator_raw))
    fastvideo_args.output_type = "latent"
    generator = VideoGenerator.from_fastvideo_args(fastvideo_args)
    worker = generator.executor.worker.worker
    return generator, worker.pipeline, worker.fastvideo_args, raw["request"]


def load_variants(path: str) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text())


def jobs_for(variants: list[dict[str, Any]], prompts: list[str]) -> list[tuple[dict[str, Any], str]]:
    return [(variant, prompt) for variant in variants for prompt in variant.get("prompts", prompts)]


# --------------------------------------------------------------------------- generate
class TimedStage:
    """Wrap a stage's forward to record synchronized wall time."""

    def __init__(self, stage: Any) -> None:
        self.stage = stage
        self.original = stage.forward
        self.seconds = 0.0
        stage.forward = self

    def __call__(self, batch: Any, fastvideo_args: Any) -> Any:
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = self.original(batch, fastvideo_args)
        torch.cuda.synchronize()
        self.seconds = time.perf_counter() - start
        return result


def generate(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    (out / "latents").mkdir(parents=True, exist_ok=True)
    variants = load_variants(args.variants)
    jobs = jobs_for(variants, args.prompts)[args.slot::args.num_slots]
    todo = [(v, p) for v, p in jobs if not (out / "latents" / f"{v['name']}__{p}.pt").exists()]
    print(json.dumps({"slot": args.slot, "jobs": len(jobs), "todo": len(todo)}), flush=True)
    if not todo:
        return
    generator, pipeline, fastvideo_args, request = build(args)
    transformer = pipeline.get_module("transformer")
    trained_steps = list(fastvideo_args.pipeline_config.dmd_denoising_steps)
    sampling = dict(request["sampling"])
    frames = (args.num_frames - 5) // 17 * 5 + 2
    grid = (frames, args.height // 32, args.width // 32)
    # One untimed request: warms kernels and, under deferred loading, creates the denoise stages.
    generator.generate(request={"prompt": PROMPTS[args.prompts[0]], "sampling": {**sampling, "seed": args.seed,
                                "width": args.width, "height": args.height, "num_frames": args.num_frames},
                                "output": {"save_video": False, "return_frames": True}})
    timer = TimedStage(pipeline._stage_name_mapping["denoising_stage"])
    with open(out / f"runs-slot{args.slot}.jsonl", "a") as log:
        for variant, prompt in todo:
            compress = variant.get("compress")
            handle = install_levers(transformer,
                                    skip=tuple(variant.get("skip", ())),
                                    compress=CompressSpec(**compress) if compress else None,
                                    score=bool(variant.get("score", False)))
            handle.set_grid(*grid)
            steps = variant.get("steps", trained_steps)
            fastvideo_args.pipeline_config.dmd_denoising_steps = list(steps)
            try:
                start = time.perf_counter()
                result = generator.generate(
                    request={
                        "prompt": PROMPTS[prompt],
                        "negative_prompt": "",
                        "sampling": {
                            **sampling, "seed": args.seed,
                            "width": args.width,
                            "height": args.height,
                            "num_frames": args.num_frames,
                            "num_inference_steps": len(steps) + 1
                        },
                        "output": {
                            "save_video": False,
                            "return_frames": True
                        },
                    })
                total = time.perf_counter() - start
            finally:
                handle.remove()
                fastvideo_args.pipeline_config.dmd_denoising_steps = trained_steps
            latent = result.samples.float().cpu().contiguous()
            torch.save(latent, out / "latents" / f"{variant['name']}__{prompt}.pt")
            record = {
                "variant": variant["name"],
                "prompt": prompt,
                "dit_seconds": round(timer.seconds, 3),
                "total_seconds": round(total, 2),
                "forwards": len(steps),
                "shape": list(latent.shape)
            }
            if handle.scores:
                record["block_change"] = {str(k): float(np.mean(v)) for k, v in sorted(handle.scores.items())}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps({k: v for k, v in record.items() if k != "block_change"}), flush=True)
    generator.shutdown()


# --------------------------------------------------------------------------- evaluate
class Decoder:

    def __init__(self, args: argparse.Namespace) -> None:
        self.generator, self.pipeline, _, _ = build(args, lazy=True)
        self.vae = self.pipeline.get_module("vae")
        self.device = torch.device("cuda")

    @torch.no_grad()
    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        """Raw ``[1, 24, T, H, W]`` latent -> float pixels ``[T, 3, H, W]`` in [0, 1] on the GPU."""
        from fastvideo.models import pinned_offload

        pinned_offload.load(self.vae, self.device, pin=False)
        latent = latent.to(self.device, torch.float32)
        output = torch.empty(self.vae.decoded_pixel_shape(latent.shape), dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            self.vae.decode_to_pixels(latent, output)
        return output[0].permute(1, 0, 2, 3).clamp(0, 1).to(self.device)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a - b)**2).mean())
    return 99.0 if mse == 0 else 10 * float(np.log10(1.0 / mse))


def evaluate(args: argparse.Namespace) -> None:
    import lpips

    out = Path(args.output_dir)
    prompts = args.prompts[args.slot::args.num_slots]
    decoder = Decoder(args)
    metric = lpips.LPIPS(net="alex", verbose=False).to("cuda").eval()
    done = set()
    metrics_path = out / f"metrics-slot{args.slot}.jsonl"
    if metrics_path.exists():
        done = {(r["variant"], r["prompt"]) for r in map(json.loads, metrics_path.read_text().splitlines())}
    with open(metrics_path, "a") as log:
        for prompt in prompts:
            reference_path = out / "latents" / f"reference__{prompt}.pt"
            if not reference_path.exists():
                continue
            reference = decoder(torch.load(reference_path))
            for path in sorted((out / "latents").glob(f"*__{prompt}.pt")):
                variant = path.name.split("__")[0]
                if variant == "reference" or (variant, prompt) in done:
                    continue
                video = decoder(torch.load(path))
                frames = slice(None, None, 4)
                with torch.no_grad():
                    lp = float(metric(video[frames] * 2 - 1, reference[frames] * 2 - 1).mean())
                record = {"variant": variant, "prompt": prompt, "lpips": round(lp, 4), "psnr": round(psnr(video, reference),
                                                                                                    2)}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
    decoder.generator.shutdown()


# --------------------------------------------------------------------------- grid
def grid(args: argparse.Namespace) -> None:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    out = Path(args.output_dir)
    media = Path(args.media_dir)
    media.mkdir(parents=True, exist_ok=True)
    decoder = Decoder(args)
    panels = [item.split("=", 1) for item in args.panels.split(",")]
    font = ImageFont.load_default(size=18)
    for prompt in args.prompts[args.slot::args.num_slots]:
        videos = []
        for name, label in panels:
            video = decoder(torch.load(out / "latents" / f"{name}__{prompt}.pt"))
            video = torch.nn.functional.interpolate(video, scale_factor=args.scale, mode="bilinear", antialias=True)
            frames = (video.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
            strip = Image.new("RGB", (frames.shape[2], 26), "black")
            ImageDraw.Draw(strip).text((6, 3), label, fill="white", font=font)
            strip_array = np.asarray(strip)[None].repeat(frames.shape[0], 0)
            videos.append(np.concatenate((strip_array, frames), axis=1))
        tiled = np.concatenate(videos, axis=2)
        path = media / f"{args.grid_tag}-{prompt}.mp4"
        imageio.mimsave(path, list(tiled), fps=24, format="mp4", quality=6, macro_block_size=2)
        print(json.dumps({"grid": str(path)}), flush=True)
    decoder.generator.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("generate", "evaluate", "grid"))
    parser.add_argument("--config", required=True, help="FastH3 run config (generator + request)")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variants", help="JSON list of variants (generate)")
    parser.add_argument("--prompts", nargs="+", default=list(PROMPTS)[:8])
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--num-slots", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--warmup", action="store_true", help="accepted for compatibility; generate always warms up")
    parser.add_argument("--media-dir", help="grid output directory")
    parser.add_argument("--panels", default="reference=reference", help="comma list of variant=label (grid)")
    parser.add_argument("--grid-tag", default="grid")
    parser.add_argument("--scale", type=float, default=0.5)
    args = parser.parse_args()
    {"generate": generate, "evaluate": evaluate, "grid": grid}[args.mode](args)


if __name__ == "__main__":
    main()
