#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Constrained 50-to-N block-map selection for H3 structural pruning.

The audit of the activation-34 map (job 6928 student34) showed all 16 removals
taken from source blocks 4-30 while 31-49 stayed intact: a front-loaded
amputation. This selector keeps ablation scores as *proposals* only and enforces
the structural constraints that the zero-shot and recovery evidence supports:

* keep the first ``keep_prefix`` blocks and the final block,
* never remove more than ``max_contiguous_removed`` consecutive blocks,
* take at least ``min_late_removals`` removals from source >= ``late_start``,
* never remove the ``audio_veto`` blocks with the highest audio-only
  ablation importance (audio has no private capacity outside AdaLN, so the
  blocks that carry it cannot be re-acquired elsewhere).

Usage::

    python scripts/fasth3_sprint/select_h3_block_map.py \
        --partials-dir eval/h6-block-score --keep-blocks 34 \
        --output /tmp/base34-recut-v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ("ablation_video", "ablation_audio", "ablation_total", "residual_video", "residual_audio",
           "cross_modal_change")
CATEGORIES = ("speech", "music", "motion", "sound_event", "multiple_shots")


def load_scores(partials_dir: Path) -> dict[str, list[dict[str, float]]]:
    shards = sorted(partials_dir.glob("partial-*.json"))
    if not shards:
        raise SystemExit(f"No partial-*.json under {partials_dir}")
    totals: dict[str, list[dict[str, float]]] = {}
    count = 0
    for shard in shards:
        data = json.loads(shard.read_text())
        count += 1
        for category, rows in data.get("by_category", {}).items():
            acc = totals.setdefault(category, [dict.fromkeys(METRICS, 0.0) for _ in rows])
            if len(acc) != len(rows):
                raise SystemExit(f"{shard.name}: inconsistent block count for {category}")
            for dst, src in zip(acc, rows, strict=True):
                for metric in METRICS:
                    dst[metric] += float(src[metric])
    return {cat: [{m: v / count for m, v in row.items()} for row in rows] for cat, rows in totals.items()}


def _minmax(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high <= low:
        return [0.5] * len(values)
    return [(v - low) / (high - low) for v in values]


def importance(scores: dict[str, list[dict[str, float]]], audio_weight: float) -> tuple[list[float], list[float]]:
    """Return (blended importance, audio-only importance) per source block."""
    blocks = len(next(iter(scores.values())))
    video_raw: list[float] = [0.0] * blocks
    audio_raw: list[float] = [0.0] * blocks
    weight_sum = 0.0
    for category, rows in scores.items():
        weight = 1.0 if category in CATEGORIES else 0.5
        weight_sum += weight
        for index, row in enumerate(rows):
            video_raw[index] += weight * (0.5 * row["ablation_video"] + 0.3 * row["ablation_total"] +
                                          0.2 * row["cross_modal_change"])
            audio_raw[index] += weight * (0.6 * row["ablation_audio"] + 0.2 * row["ablation_total"] +
                                          0.2 * row["cross_modal_change"])
    video = _minmax([v / weight_sum for v in video_raw])
    audio = _minmax([a / weight_sum for a in audio_raw])
    blended = [(1.0 - audio_weight) * v + audio_weight * a for v, a in zip(video, audio, strict=True)]
    return blended, audio


def select_map(blended: list[float],
               audio: list[float],
               keep: int,
               *,
               keep_prefix: int = 4,
               max_contiguous_removed: int = 2,
               min_late_removals: int = 4,
               late_start: int = 31,
               audio_veto: int = 6,
               base_map: list[int] | None = None) -> list[int]:
    """Select a kept set; with ``base_map`` only remove blocks it still has.

    ``base_map`` re-cuts an already pruned lineage (e.g. 42 -> 34): the pools
    are restricted to blocks the base kept, and the contiguity cap applies to
    *newly* removed source blocks, since the base's own gaps are already
    proven tolerable by its recovered quality.
    """
    blocks = len(blended)
    allowed = set(base_map) if base_map is not None else set(range(blocks))
    removed_count = (len(base_map) if base_map is not None else blocks) - keep
    if removed_count <= 0:
        return sorted(allowed)
    if min_late_removals > removed_count:
        raise ValueError("min_late_removals exceeds the removal budget")
    veto = set(int(i) for i in sorted(allowed, key=lambda i: -audio[i])[:audio_veto])
    protected = veto | (set(range(keep_prefix)) & allowed) | {blocks - 1}

    base_removed = set(range(blocks)) - allowed
    base_cap = 0
    run = 0
    for index in range(blocks):
        run = run + 1 if index in base_removed else 0
        base_cap = max(base_cap, run)
    # A re-cut may not stretch an already proven gap into a longer hole.
    cap = max(max_contiguous_removed, base_cap)

    def contiguous_ok(removed: set[int]) -> bool:
        final = removed | base_removed
        run = 0
        for index in range(blocks):
            run = run + 1 if index in final else 0
            if run > cap:
                return False
        return True

    late_pool = [i for i in sorted(allowed) if late_start <= i < blocks - 1 and i not in protected]
    early_pool = [i for i in sorted(allowed) if keep_prefix <= i < late_start and i not in protected]
    removed: set[int] = set()
    for index in sorted(late_pool, key=lambda i: blended[i])[:min_late_removals * 3]:
        if sum(1 for i in removed if i >= late_start) >= min_late_removals:
            break
        trial = removed | {index}
        if contiguous_ok(trial):
            removed = trial
    for index in sorted(early_pool, key=lambda i: blended[i]):
        if len(removed) >= removed_count:
            break
        trial = removed | {index}
        if contiguous_ok(trial):
            removed = trial
    if len(removed) < removed_count:  # contiguity blocked early picks; backfill from the late pool
        for index in sorted(late_pool, key=lambda i: blended[i]):
            if len(removed) >= removed_count:
                break
            trial = removed | {index}
            if contiguous_ok(trial):
                removed = trial
    if len(removed) != removed_count or not contiguous_ok(removed):
        raise SystemExit("constraint set is infeasible for this keep count")
    if sum(1 for i in removed if i >= late_start) < min_late_removals:
        raise SystemExit("could not place the required late-half removals")
    return [i for i in sorted(allowed) if i not in removed]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partials-dir", type=Path, required=True)
    parser.add_argument("--keep-blocks", type=int, default=34)
    parser.add_argument("--audio-weight", type=float, default=0.5)
    parser.add_argument("--keep-prefix", type=int, default=4)
    parser.add_argument("--max-contiguous-removed", type=int, default=2)
    parser.add_argument("--min-late-removals", type=int, default=4)
    parser.add_argument("--late-start", type=int, default=31)
    parser.add_argument("--audio-veto", type=int, default=6)
    parser.add_argument("--base-map", type=Path, default=None,
                        help="Optional already-pruned kept set; restricts removals to a re-cut lineage")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scores = load_scores(args.partials_dir)
    blended, audio = importance(scores, args.audio_weight)
    base_map = None
    if args.base_map:
        payload_in = json.loads(args.base_map.read_text())
        base_map = payload_in["block_map"] if isinstance(payload_in, dict) else payload_in
    block_map = select_map(blended,
                           audio,
                           args.keep_blocks,
                           keep_prefix=args.keep_prefix,
                           max_contiguous_removed=args.max_contiguous_removed,
                           min_late_removals=args.min_late_removals,
                           late_start=args.late_start,
                           audio_veto=args.audio_veto,
                           base_map=base_map)
    local_keep = None
    if base_map is not None:
        kept = set(block_map)
        local_keep = [local for local, source in enumerate(base_map) if source in kept]
    removed = [i for i in range(len(blended)) if i not in block_map]
    payload = {
        "block_map": block_map,
        "prune_block_map_local": local_keep,
        "removed": removed,
        "source_model": "MiniMaxAI/MiniMax-H3",
        "strategy": "constrained-audio-veto-recut",
        "audio_weight": args.audio_weight,
        "constraints": {
            "keep_prefix": args.keep_prefix,
            "max_contiguous_removed": args.max_contiguous_removed,
            "min_late_removals": args.min_late_removals,
            "late_start": args.late_start,
            "audio_veto": args.audio_veto,
        },
        "blended_importance": [round(v, 5) for v in blended],
        "audio_importance": [round(v, 5) for v in audio],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1))
    if local_keep is not None:
        args.output.with_suffix(".local.json").write_text(
            json.dumps({"block_map": local_keep, "source_block_map": block_map}, indent=1))
    print(json.dumps({"keep": len(block_map), "removed": removed}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
