"""Run the FastWan 1.3B three-step compatibility baseline on Apple Silicon.

This proof of concept intentionally uses the unquantized FastWan checkpoint:
the FastWan-QAD NVFP4 release depends on NVIDIA-only kernels. FastVideo selects
MPS + Torch SDPA automatically; its MPS compatibility settings force eager FP16
DiT execution and disable CUDA-only autocast and offload paths.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="FastWan MPS proof of concept")
    parser.add_argument("--model", default="FastVideo/FastWan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--output-path", default="video_samples/mps_fastwan_poc.mp4")
    parser.add_argument(
        "--prompt",
        default="A paper boat sails through a shallow stream in a mossy forest.",
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument(
        "--mps-high-watermark-ratio",
        default="2.0",
        help="PyTorch MPS allocation ceiling; use a lower value if macOS reports memory pressure.",
    )
    args = parser.parse_args()

    # Must be set before MPS is initialized. The tiled 33-frame POC needs a
    # small amount of headroom above PyTorch's conservative default ceiling.
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", args.mps_high_watermark_ratio)

    import torch

    from fastvideo import VideoGenerator

    if not torch.backends.mps.is_available():
        raise RuntimeError("This proof of concept requires an Apple Silicon Mac with PyTorch MPS available.")

    generator = VideoGenerator.from_pretrained(args.model, num_gpus=1)
    try:
        generator.generate_video(
            prompt=args.prompt,
            num_inference_steps=3,
            guidance_scale=1.0,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            output_path=args.output_path,
            save_video=True,
        )
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
