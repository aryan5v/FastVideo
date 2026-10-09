# SPDX-License-Identifier: Apache-2.0
"""Decode saved QAD eval samples to side-by-side mp4s: teacher (left) | student (right), same seed.

``--runs`` takes ``name=<eval_save_dir>/stepNNNNN`` pairs; every run must hold the same row ids. One mp4 per row
with one panel per run after the teacher panel. One GPU, the composed OmniRef model's VAE.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="name=dir pairs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-rows", type=int, default=5)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "minimax_h3_nvfp4_decoder"))
    import generate_omniref_latents as gen
    import imageio.v2 as imageio

    from fastvideo.pipelines.basic.minimax_h3.packing import unpatchify_video_tokens

    runs = dict(item.split("=", 1) for item in args.runs)
    first = Path(next(iter(runs.values())))
    driver = gen.OmniRefLatentGenerator(argparse.Namespace(model_path=args.model_path, master_port=29711))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for row_file in sorted(first.glob("*.pt"))[:args.max_rows]:
        panels = []
        for index, (name, run_dir) in enumerate(runs.items()):
            data = torch.load(Path(run_dir) / row_file.name)
            channels, frames, height, width = data["latent_shape"]
            keys = (["teacher_video"] if index == 0 else []) + ["student_video"]
            for key in keys:
                latent = unpatchify_video_tokens(data[key][None], frames, height, width, channels, (1, 2, 2))[0]
                panels.append(driver.decode(latent))
        frames_out = list(np.concatenate(panels, axis=2))
        target = out / f"{data['case']}_{row_file.stem[-24:]}.mp4"
        imageio.mimsave(target, frames_out, fps=24, format="mp4")
        print(f"{target}: teacher | {' | '.join(runs)}", flush=True)
    driver.shutdown()


if __name__ == "__main__":
    main()
