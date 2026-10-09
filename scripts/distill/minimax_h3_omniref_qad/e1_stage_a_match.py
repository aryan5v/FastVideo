# SPDX-License-Identifier: Apache-2.0
"""E1: does the QAD harness start at the deployed PTQ point? (torchrun, same config machinery as training)

(a) Per held-out row and forward, the step-0 student with NVFP4 linears and bf16 attention (Stage A's protocol),
    teacher-forced, gives v_rel against the bf16 teacher; compared with Stage A's own T1 records for the same rows
    and seeds (``t1-omniref-shard*.jsonl``). Pass: the mean over rows is within 5% per forward and modality.
(b) The same student's own 8-step sample vs Stage A's saved deployed-path NVFP4 sample (``nvfp4/*__nvfp4_s0.pt``,
    packed export through the inference pipeline). Pass: rel-L2 <= 1%. The harness teacher vs Stage A's bf16 sample
    (``heldout_bf16/*__bf16_s0.pt``) is the loader floor. FP4 attention parity is the separate Spark fidelity gate.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage-a", required=True, help="Stage A eval dir (t1-omniref-shard*.jsonl, nvfp4/)")
    parser.add_argument("--bf16-dir", required=True, help="Stage A calib/heldout_bf16")
    parser.add_argument("--out", required=True)
    args, overrides = parser.parse_known_args()

    from fastvideo.distributed import get_world_group, maybe_init_distributed_environment_and_model_parallel
    from fastvideo.pipelines.basic.minimax_h3.packing import unpatchify_video_tokens
    from fastvideo.train.methods.knowledge_distillation.pdd_qad_metrics import step_packed, target_slices, x0_pair
    from fastvideo.train.utils.builder import build_from_config
    from fastvideo.train.utils.config import load_run_config

    cfg = load_run_config(args.config, overrides=overrides or None)
    maybe_init_distributed_environment_and_model_parallel(cfg.training.distributed.tp_size,
                                                          cfg.training.distributed.sp_size)
    _, method, _, _ = build_from_config(cfg)
    student, teacher = method.student, method.teacher
    records = {}
    for path in sorted(glob.glob(f"{args.stage_a}/t1-omniref-shard*.jsonl")):
        for line in open(path):
            record = json.loads(line)
            records[record["id"]] = record
    rank0 = get_world_group().rank == 0
    rows, results = [spec for spec, counted in method._rank_rows() if counted], []
    with torch.no_grad(), student.quantization(linears=True, attention=False):
        for spec in rows:
            if spec.id not in records:
                continue
            prepared = method._row(spec)
            schedulers = student.schedulers()
            timesteps = (schedulers[0].timesteps, schedulers[1].timesteps)
            video, audio = prepared.video, prepared.audio
            per_rung = []
            for rung in range(student.num_rungs):
                outs = {}
                for name, model in (("teacher", teacher), ("student", student)):
                    with model.rung_scope(prepared, rung, timesteps) as rt:
                        outs[name] = model.forward_rung(prepared, video, audio, rt)
                pair = x0_pair(prepared, schedulers, rung, video, audio, outs)
                per_rung.append({m: _rel(pair[m]["student_v"], pair[m]["teacher_v"]) for m in ("video", "audio")})
                video, audio = step_packed(schedulers, prepared, rung, video, audio, *outs["teacher"])
            teacher_video, teacher_audio = video, audio
            video, audio = prepared.video, prepared.audio
            for rung in range(student.num_rungs):
                with student.rung_scope(prepared, rung, timesteps) as rt:
                    out = student.forward_rung(prepared, video, audio, rt)
                video, audio = step_packed(schedulers, prepared, rung, video, audio, *out)
            cut = target_slices(prepared)
            channels, frames, height, width = prepared.latent_shape

            def unpatch(rows_: torch.Tensor) -> torch.Tensor:
                return unpatchify_video_tokens(rows_[cut["video"]][None].float(), frames, height, width, channels,
                                               (1, 2, 2))[0].cpu()

            stem = spec.id
            nvfp4 = torch.load(next(Path(args.stage_a, "nvfp4").glob(f"{stem}__nvfp4_s0.pt")), map_location="cpu")
            bf16 = torch.load(next(Path(args.bf16_dir).glob(f"{stem}__bf16_s0.pt")), map_location="cpu")
            stage_a = records[spec.id]["forwards"]
            results.append({
                "id": spec.id, "case": spec.case,
                "v_rel": per_rung,
                "stage_a_v_rel": [{m: f["nvfp4"][m]["v_rel"] for m in ("video", "audio")} for f in stage_a],
                "endpoint_student_vs_stage_a_nvfp4": {"video": _rel(unpatch(video), nvfp4["video"]),
                                                      "audio": _rel(audio[cut["audio"]].cpu(), nvfp4["audio"])},
                "endpoint_teacher_vs_stage_a_bf16": {"video": _rel(unpatch(teacher_video), bf16["video"]),
                                                     "audio": _rel(teacher_audio[cut["audio"]].cpu(), bf16["audio"])},
                "endpoint_student_vs_harness_teacher": _rel(unpatch(video), unpatch(teacher_video)),
                "stage_a_final_video_rel_vs_bf16": records[spec.id].get("final_video_rel_vs_bf16"),
            })
            if method.is_sp_leader and get_world_group().rank == 0:
                print("E1ROW " + json.dumps(results[-1]), flush=True)
    if rank0 and results:
        summary = {}
        for modality in ("video", "audio"):
            ours = [sum(r["v_rel"][k][modality] for r in results) / len(results) for k in range(8)]
            theirs = [sum(r["stage_a_v_rel"][k][modality] for r in results) / len(results) for k in range(8)]
            summary[f"v_rel_ours/{modality}"] = ours
            summary[f"v_rel_stage_a/{modality}"] = theirs
            summary[f"v_rel_max_rel_dev/{modality}"] = max(abs(o / t - 1) for o, t in zip(ours, theirs, strict=True))
            for key in ("endpoint_student_vs_stage_a_nvfp4", "endpoint_teacher_vs_stage_a_bf16"):
                summary[f"{key}/{modality}/max"] = max(r[key][modality] for r in results)
        summary["pass_a"] = all(summary[f"v_rel_max_rel_dev/{m}"] <= 0.05 for m in ("video", "audio"))
        summary["pass_b"] = all(summary[f"endpoint_student_vs_stage_a_nvfp4/{m}/max"] <= 0.01 for m in ("video",
                                                                                                         "audio"))
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "e1.json").write_text(json.dumps({"summary": summary, "rows": results}, indent=1))
        print("E1SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
