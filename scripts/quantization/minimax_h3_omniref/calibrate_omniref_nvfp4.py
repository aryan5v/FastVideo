# SPDX-License-Identifier: Apache-2.0
"""Calibrate static NVFP4 FFN activation scales for a FastH3 OmniRef (Ref2VA) checkpoint.

``calibrate``: run calibration rows through the unmodified bf16 pipeline (all PDD forwards,
precomputed Ref2VA conditioning via ``generate_omniref_latents.OmniRefLatentGenerator``: reference
rows, Qwen3-VL vision tokens and reference audio included) with ``H3AmaxCollector`` hooks, then
generate the held-out bf16 references (seed s and the noise-floor seed s+1; hooks off). One process
per GPU; ``--shard``/``--num-shards`` split both plans. Resumable: rows already in the shard file
are skipped.

``merge``: combine the shard files into the runtime/converter JSON (``--act-amax``), a per-layer
report (per token group and case, late ``fc_out`` top-k and percentiles) and the half-vs-full
convergence check.

Calibration rows come from the same deterministic per-(case, resolution) shuffle as the OmniRef
latent plan (``--plan-seed``); held-out rows exclude every source clip of that plan's first
``--exclude-per-group`` entries.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "distill" / "minimax_h3_nvfp4_decoder"))

from h3_amax_collector import (H3AmaxCollector, convergence, layer_report, merge_states,  # noqa: E402
                               runtime_amax_table)

RESOLUTIONS = ("480p", "768p")


# --------------------------------------------------------------------------- plans
def _group_plan(groups: dict[tuple[str, str], list[dict[str, Any]]], seed: int, counts: dict[str, int],
                exclude: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    """Per (case, resolution): the first ``counts[res]`` of the seeded shuffle; interleaved by fraction.

    Interleaving by rank / count keeps every prefix of the plan balanced across cases and
    resolutions, so the first half is a fair convergence sample.
    """
    from generate_omniref_latents import select_clips

    ranked = []
    for key in sorted(groups):
        count = counts.get(key[1], 0)
        if count <= 0:
            continue
        clips = select_clips({key: groups[key]}, seed, count, exclude)
        ranked.extend(((rank + 0.5) / count, key, clip) for rank, clip in enumerate(clips))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [{**clip, "plan_index": index} for index, (_, _, clip) in enumerate(ranked)]


def calibration_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    from generate_omniref_latents import load_manifest

    groups = load_manifest(args.manifest, args.cases, args.max_frames)
    return _group_plan(groups, args.plan_seed, {"480p": args.calib_480p, "768p": args.calib_768p})


def heldout_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    from generate_omniref_latents import load_manifest, select_clips

    groups = load_manifest(args.manifest, args.cases, args.max_frames)
    main_plan = select_clips(groups, args.plan_seed, args.exclude_per_group)
    excluded = frozenset(clip["source"] for clip in main_plan)
    return _group_plan(groups, args.heldout_seed, {"480p": args.heldout_480p, "768p": args.heldout_768p}, excluded)


# --------------------------------------------------------------------------- generation helpers
def real_transformer(driver: Any) -> torch.nn.Module:
    from fastvideo.pipelines.lazy_module import is_lazy_module

    transformer = driver.denoise.transformer
    return transformer.materialize() if is_lazy_module(transformer) else transformer


def current_layout() -> Any:
    from fastvideo.forward_context import get_forward_context
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY

    batch = getattr(get_forward_context(), "forward_batch", None)
    return None if batch is None else batch.extra.get(MINIMAX_H3_LAYOUT_KEY)


@torch.no_grad()
def generate_rows(driver: Any, row: dict[str, Any], seed: int, decode_audio: bool = False) -> dict[str, Any]:
    """Normalized target video latent ``[24, T, H, W]`` and target audio rows (float32, CPU).

    With ``decode_audio`` also the pipeline's decoded waveform (``[samples, channels]`` float16) and rate.
    """
    from fastvideo.pipelines.basic.minimax_h3.packing import h3_dit_patch_size, unpatchify_video_tokens
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY

    batch = driver.denoise.forward(driver._batch(row, seed), driver.fastvideo_args)
    layout = batch.extra[MINIMAX_H3_LAYOUT_KEY]
    _, channels, frames, height, width = batch.raw_latent_shape
    video = unpatchify_video_tokens(batch.latents[layout.num_condition_video_rows:], frames, height, width, channels,
                                    h3_dit_patch_size(driver.fastvideo_args))[0]
    audio = batch.audio_latents[layout.num_condition_audio_rows:]
    result: dict[str, Any] = {"video": video.float().cpu().contiguous(), "audio": audio.float().cpu().contiguous()}
    if decode_audio:
        batch = driver.pipeline._stage_name_mapping["audio_decoding_stage"].forward(batch, driver.fastvideo_args)
        result["waveform"] = batch.extra["audio"].detach().to(torch.float16).cpu().contiguous()
        result["sample_rate"] = int(batch.extra["audio_sample_rate"])
    return result


def save_atomic(obj: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def heldout_name(clip: dict[str, Any], tag: str) -> str:
    return f"{clip['id']}__{tag}.pt"


# --------------------------------------------------------------------------- modes
def calibrate(args: argparse.Namespace) -> None:
    from generate_omniref_latents import OmniRefLatentGenerator, read_row

    out = Path(args.output_dir)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    heldout_dir = out / "heldout_bf16"
    heldout_dir.mkdir(parents=True, exist_ok=True)
    plan = calibration_plan(args)[args.shard::args.num_shards]
    heldout = heldout_plan(args)[args.shard::args.num_shards]
    shard_path = out / "shards" / f"amax-shard{args.shard:02d}.pt"
    previous = torch.load(shard_path, weights_only=False) if shard_path.exists() else None
    done = {row["id"] for row in previous["rows"]} if previous else set()
    todo = [clip for clip in plan if clip["id"] not in done]
    heldout_todo = [(clip, tag, seed) for clip in heldout for tag, seed in (("bf16_s0", clip["seed"]),
                                                                            ("bf16_s1", clip["seed"] + 1))
                    if not (heldout_dir / heldout_name(clip, tag)).exists()]
    print(json.dumps({"shard": args.shard, "calibration": len(plan), "todo": len(todo), "heldout_todo":
                      len(heldout_todo)}), flush=True)
    if not todo and not heldout_todo:
        return
    driver = OmniRefLatentGenerator(args)
    transformer = real_transformer(driver)
    collector = H3AmaxCollector(transformer)
    if previous:
        collector.rows = previous["rows"]
        collector.topk, collector.hist = previous["topk"], previous["hist"]
    restore = collector.wrap_transformer(transformer, current_layout)
    collector.attach()
    for clip in todo:
        start = time.perf_counter()
        row = read_row(clip["parquet"])
        collector.begin_row()
        try:
            generate_rows(driver, row, clip["seed"])
        except Exception as error:  # noqa: BLE001 - one bad row must not stop the shard
            collector._row = None
            print(json.dumps({"id": clip["id"], "error": repr(error)[:500]}), flush=True)
            continue
        record = collector.end_row(**{k: clip[k] for k in ("id", "case", "resolution", "seed", "plan_index")},
                                   num_frames=int(row["num_frames"]))
        save_atomic(collector.state(), shard_path)
        ffn = {k: max(v) for k, v in record["amax"].items() if ".ff." in k}
        print(json.dumps({"id": clip["id"], "seconds": round(time.perf_counter() - start, 1), "ffn_amax_max":
                          round(max(ffn.values()), 1)}), flush=True)
    collector.detach()
    restore()
    for clip, tag, seed in heldout_todo:
        start = time.perf_counter()
        try:
            row = read_row(clip["parquet"])
            result = generate_rows(driver, row, seed, decode_audio=True)
        except Exception as error:  # noqa: BLE001 - record and continue
            print(json.dumps({"heldout": clip["id"], "tag": tag, "error": repr(error)[:500]}), flush=True)
            continue
        meta = {k: clip[k] for k in ("id", "source", "case", "resolution", "parquet", "plan_index")}
        save_atomic({**result, **meta, "seed": seed, "tag": tag}, heldout_dir / heldout_name(clip, tag))
        print(json.dumps({"heldout": clip["id"], "tag": tag, "seconds": round(time.perf_counter() - start, 1)}),
              flush=True)
    driver.shutdown()


def merge(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    states = [torch.load(p, weights_only=False) for p in sorted((out / "shards").glob("amax-shard*.pt"))]
    if not states:
        raise SystemExit(f"no shard files under {out / 'shards'}")
    merged = merge_states(states)
    expected = len(calibration_plan(args))
    table = runtime_amax_table(merged["rows"], margin=args.margin)
    check = convergence(merged["rows"], tolerance=args.tolerance)
    report = layer_report(merged)
    (out / "amax.json").write_text(json.dumps(table, indent=1) + "\n")
    (out / "amax_report.json").write_text(json.dumps({"convergence": check, "layers": report}, indent=1) + "\n")
    torch.save(merged, out / "amax_merged.pt")
    summary = {"rows": len(merged["rows"]), "planned": expected, "ffn_keys": len(table),
               "convergence_passed": check["passed"], "max_gap": round(check["max_gap"], 4),
               "layers_over_tolerance": check["layers_over_tolerance"],
               "fc_out_amax_max": max(v for k, v in table.items() if k.endswith("fc_out")),
               "fc_in_amax_max": max(v for k, v in table.items() if k.endswith("fc_in"))}
    print(json.dumps(summary), flush=True)
    if len(merged["rows"]) < expected:
        print(json.dumps({"warning": f"{expected - len(merged['rows'])} planned rows missing"}), flush=True)


def add_plan_args(parser: argparse.ArgumentParser) -> None:
    from generate_omniref_latents import CASES

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--max-frames", type=int, default=243)
    parser.add_argument("--plan-seed", type=int, default=20261007, help="the OmniRef latent plan's seed")
    parser.add_argument("--calib-480p", type=int, default=80, help="calibration rows per case at 480p")
    parser.add_argument("--calib-768p", type=int, default=20, help="calibration rows per case at 768p")
    parser.add_argument("--heldout-seed", type=int, default=20261008)
    parser.add_argument("--heldout-480p", type=int, default=12)
    parser.add_argument("--heldout-768p", type=int, default=2)
    parser.add_argument("--exclude-per-group", type=int, default=300)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("calibrate", "merge"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", help="composed FastH3 OmniRef model directory (calibrate)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=int(os.environ.get("MASTER_PORT", 29500)))
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=0.05)
    add_plan_args(parser)
    args = parser.parse_args()
    {"calibrate": calibrate, "merge": merge}[args.mode](args)


if __name__ == "__main__":
    main()
