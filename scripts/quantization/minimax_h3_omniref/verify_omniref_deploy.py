# SPDX-License-Identifier: Apache-2.0
"""Run the deployed FastH3 OmniRef inference path on held-out rows and compare with saved QAD harness samples.

torchrun, one process per GPU (the generator runs in-process as an external
launcher, sequence-parallel over the ranks)::

    torchrun --nproc_per_node=4 verify_omniref_deploy.py --model-path <composed dir> \\
        --rows ab-vote/rows.json --limit 4 --samples ab-vote-mix8/samples/step00000_qad120_mix8 \\
        --nvfp4 --attention emulator-mix8 --out <dir>

Each row is generated from its precomputed conditioning (``OmniRefLatentGenerator``:
no text encoder, no reference encode) at the row's seed, through the pipeline's own
denoising stage. ``--nvfp4`` loads the packed NVFP4 DiT in the composed directory
(``NVFP4Config(layer_profile="h3_dit_vsa")``, replicated weights; a mixed export's
bf16 blocks come from its metadata). ``--attention`` picks the fine block-sparse
attention of every VSA-H3 layer:

* ``kernel``: the deployed bf16 sm_100a VSA-128 kernel;
* ``emulator-<mode>``: the QAD FP4 attention emulator (``fp4_vsa_qat``) in the
  given numerics: ``mix8`` (NVFP4 Q/K video rows, per-token FP8 prefix rows, bf16
  P.V, K smoothing), ``qk`` (NVFP4 Q/K everywhere) or ``mixmx`` (MXFP8 prefix rows, the FA4 kernel numerics);
* ``fa4-<mode>``: the FA4 sm_100 kernels (``fastvideo.attention.backends.fa4_vsa_h3``).

Per row the saved target video and audio rows are compared (rel-L2) with the
sample file's ``--compare-key`` entries (``student`` = the QAD harness's own
8-forward sample, ``teacher`` = its bf16 teacher), and written to ``--out`` with
the harness's sample keys (``student_*`` = this run, ``teacher_*`` copied from the saved sample) so
``audio_stats_gate.py`` and the endpoint metrics read it like a harness eval directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "distill" / "minimax_h3_omniref_qad"))

EMULATOR_MODES = {
    "mix8": dict(quantize=True, quantize_pv=False, smooth_k=True, prefix_precision="fp8"),
    "mix16": dict(quantize=True, quantize_pv=False, smooth_k=True, prefix_precision="bf16"),
    "mixmx": dict(quantize=True, quantize_pv=False, smooth_k=True, prefix_precision="mxfp8"),
    "qk": dict(quantize=True, quantize_pv=False, smooth_k=True),
}


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def build_driver(model_path: str, master_port: int, nvfp4: bool):
    """``OmniRefLatentGenerator`` with resident weights and, optionally, the packed NVFP4 DiT."""
    import fastvideo.api as api
    from generate_omniref_latents import OmniRefLatentGenerator

    engine_config = api.EngineConfig

    def patched(**kwargs):
        kwargs = {**kwargs, "use_fsdp_inference": False}
        if nvfp4:
            kwargs["quantization"] = api.QuantizationConfig(transformer_quant="NVFP4", layer_profile="h3_dit_vsa")
        return engine_config(**kwargs)

    api.EngineConfig = patched
    try:
        return OmniRefLatentGenerator(argparse.Namespace(model_path=model_path, master_port=master_port))
    finally:
        api.EngineConfig = engine_config


def install_attention(transformer: torch.nn.Module, attention: str) -> str:
    if attention == "kernel":
        return "kernel"
    kind, _, mode = attention.partition("-")
    if kind == "emulator":
        from fastvideo.attention.backends.fp4_vsa_qat import FP4AttentionNumerics, install_fp4_vsa_attention

        count = install_fp4_vsa_attention(transformer, FP4AttentionNumerics(**EMULATOR_MODES[mode]))
        return f"emulator {mode} on {count} layers"
    if kind == "fa4":
        from fastvideo.attention.backends.fa4_vsa_h3 import install_fa4_vsa_attention

        count = install_fa4_vsa_attention(transformer, mode)
        return f"fa4 {mode} on {count} layers"
    raise ValueError(f"unknown --attention {attention!r}")


@torch.no_grad()
def generate_rows(driver, row: dict, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Target (video rows, audio rows) of the final denoised packed sequence, CPU fp32."""
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY

    batch = driver.denoise.forward(driver._batch(row, seed), driver.fastvideo_args)
    layout = batch.extra[MINIMAX_H3_LAYOUT_KEY]
    video = batch.latents[layout.num_condition_video_rows:].float().cpu()
    audio = batch.audio_latents[layout.num_condition_audio_rows:].float().cpu()
    return video, audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--rows", required=True, help="rows JSON (id, parquet, seed, case, resolution)")
    parser.add_argument("--limit", type=int, default=0, help="first N rows only (0 = all)")
    parser.add_argument("--ids", nargs="*", default=[], help="row id suffixes to run (default: --limit)")
    parser.add_argument("--samples", default="", help="dir of saved harness samples <id>.pt to compare against")
    parser.add_argument("--compare-key", default="student", choices=("student", "teacher"))
    parser.add_argument("--nvfp4", action="store_true")
    parser.add_argument("--attention", default="kernel")
    parser.add_argument("--warmup", action="store_true", help="run the first row once untimed")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from generate_omniref_latents import read_row

    rows = json.loads(Path(args.rows).read_text())
    if args.ids:
        rows = [r for r in rows if any(r["id"].endswith(s) for s in args.ids)]
    elif args.limit:
        rows = rows[:args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", 0))
    driver = build_driver(args.model_path, int(os.environ["MASTER_PORT"]), args.nvfp4)
    wrapper = driver.pipeline.get_module("transformer")
    transformer = wrapper.materialize() if hasattr(wrapper, "materialize") else wrapper
    attention = install_attention(transformer, args.attention)
    if rank == 0:
        print(json.dumps({"VERIFY_SETUP": {"nvfp4": args.nvfp4, "attention": attention, "rows": len(rows)}}),
              flush=True)
    if args.warmup and rows:
        generate_rows(driver, read_row(rows[0]["parquet"]), int(rows[0]["seed"]))
    results = []
    for spec in rows:
        row = read_row(spec["parquet"])
        torch.cuda.synchronize()
        start = time.perf_counter()
        video, audio = generate_rows(driver, row, int(spec["seed"]))
        torch.cuda.synchronize()
        record = {"id": spec["id"], "case": spec["case"], "seconds": round(time.perf_counter() - start, 2)}
        sample_path = Path(args.samples) / f"{spec['id']}.pt" if args.samples else None
        if sample_path is not None and sample_path.is_file():
            saved = torch.load(sample_path, map_location="cpu", weights_only=False)
            if int(saved.get("seed", spec["seed"])) != int(spec["seed"]):
                raise SystemExit(f"{sample_path}: seed {saved['seed']} != row seed {spec['seed']}")
            ref_video, ref_audio = saved[f"{args.compare_key}_video"], saved[f"{args.compare_key}_audio"]
            if ref_video.shape != video.shape or ref_audio.shape != audio.shape:
                raise SystemExit(f"{spec['id']}: shapes {tuple(video.shape)}/{tuple(audio.shape)} vs saved "
                                 f"{tuple(ref_video.shape)}/{tuple(ref_audio.shape)}")
            record.update(video_rel_l2=round(rel_l2(video, ref_video), 5),
                          audio_rel_l2=round(rel_l2(audio, ref_audio), 5),
                          video_bit_exact=bool(torch.equal(video, ref_video.float())),
                          audio_bit_exact=bool(torch.equal(audio, ref_audio.float())))
        if rank == 0:
            sample = {"student_video": video, "student_audio": audio, "seed": int(spec["seed"]), "case": spec["case"]}
            if sample_path is not None and sample_path.is_file():
                sample.update(teacher_video=saved["teacher_video"], teacher_audio=saved["teacher_audio"],
                              latent_shape=saved.get("latent_shape"))
                record.update(video_vs_teacher=round(rel_l2(video, saved["teacher_video"]), 5),
                              audio_vs_teacher=round(rel_l2(audio, saved["teacher_audio"]), 5))
            torch.save(sample, out_dir / f"{spec['id']}.pt")
            print("VERIFY", json.dumps(record), flush=True)
        results.append(record)
    if rank == 0:
        compared = [r for r in results if "video_rel_l2" in r]
        summary = {
            "rows": len(results),
            "compared": len(compared),
            "mean_video_rel_l2": round(sum(r["video_rel_l2"] for r in compared) / max(len(compared), 1), 5),
            "mean_audio_rel_l2": round(sum(r["audio_rel_l2"] for r in compared) / max(len(compared), 1), 5),
            "max_video_rel_l2": max((r["video_rel_l2"] for r in compared), default=None),
            "max_audio_rel_l2": max((r["audio_rel_l2"] for r in compared), default=None),
            "mean_video_vs_teacher": round(sum(r["video_vs_teacher"] for r in compared) / max(len(compared), 1), 5),
            "mean_audio_vs_teacher": round(sum(r["audio_vs_teacher"] for r in compared) / max(len(compared), 1), 5),
        }
        (out_dir / "summary.json").write_text(
            json.dumps({
                "args": vars(args),
                "attention": attention,
                "summary": summary,
                "rows": results
            }, indent=1))
        print("VERIFY_SUMMARY", json.dumps(summary), flush=True)
    driver.shutdown()


if __name__ == "__main__":
    main()
