# SPDX-License-Identifier: Apache-2.0
"""Decoded-audio gate for a QAD checkpoint's saved eval samples (one GPU).

Inputs are two ``eval_save_dir`` step directories of the same run: ``--samples`` (the checkpoint's evaluation)
and ``--seed1`` (the step-0 noise-floor evaluation at seed+1, whose ``teacher_audio`` is the bf16 teacher at the
other seed). Every row is decoded with the model's audio VAE; the gate passes when, per statistic, the median
student-vs-teacher gap (same seed) is within the teacher's own seed-to-seed spread. Prints and writes JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--seed1", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--slack", type=float, default=1.0)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "minimax_h3_nvfp4_decoder"))
    import generate_omniref_latents as gen

    from fastvideo.distributed import get_local_torch_device
    from fastvideo.models import pinned_offload
    from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_AUDIO_CHANNELS, unpack_audio_tokens
    from fastvideo.train.methods.knowledge_distillation.pdd_qad_audio_stats import (audio_stats, log_mel_distance,
                                                                                    seed_floor_gate)

    driver = gen.OmniRefLatentGenerator(argparse.Namespace(model_path=args.model_path, master_port=29717))
    vae = driver.pipeline.get_module("audio_vae")
    device = get_local_torch_device()
    pinned_offload.load(vae, device, pin=False)

    @torch.no_grad()
    def decode(rows: torch.Tensor) -> torch.Tensor:
        latents = unpack_audio_tokens(rows.to(device), rows.shape[0] // MINIMAX_H3_AUDIO_CHANNELS)
        wave = vae.decode(vae.denormalize_latents(latents.float())).sample.float()
        return wave[:, 0].transpose(0, 1).cpu()  # [samples, 2]

    rate = int(vae.sampling_rate)
    student, teacher, other, per_row = [], [], [], []
    for path in sorted(Path(args.samples).glob("*.pt")):
        twin = Path(args.seed1) / path.name
        if not twin.is_file():
            continue
        data, data1 = torch.load(path), torch.load(twin)
        s, t, o = decode(data["student_audio"]), decode(data["teacher_audio"]), decode(data1["teacher_audio"])
        student.append(audio_stats(s, rate))
        teacher.append(audio_stats(t, rate))
        other.append(audio_stats(o, rate))
        per_row.append({"row": path.stem, "case": data["case"], "log_mel_student": log_mel_distance(s, t, rate),
                        "log_mel_teacher_seed": log_mel_distance(o, t, rate), "student": student[-1],
                        "teacher": teacher[-1], "teacher_seed1": other[-1]})
    gate = seed_floor_gate(student, teacher, other, slack=args.slack)
    report = {"rows": len(per_row), "gate": gate, "per_row": per_row}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({"rows": len(per_row), **gate}), flush=True)
    driver.shutdown()


if __name__ == "__main__":
    main()
