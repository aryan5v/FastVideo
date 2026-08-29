#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Merge H3 block-score shards and emit activation-selected/uniform maps."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path


METRIC_WEIGHTS = {
    "ablation_video": 0.25,
    "ablation_audio": 0.25,
    "ablation_total": 0.10,
    "residual_video": 0.10,
    "residual_audio": 0.10,
    "cross_modal_change": 0.20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partials", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=16)
    parser.add_argument("--keep-blocks", type=int, default=20)
    parser.add_argument("--wandb-project", default="fasth3-14b-2step-qad-sprint")
    parser.add_argument("--run-id", default="h6-block-score-aggregate")
    return parser.parse_args()


def _add(target: list[dict[str, float]], source: list[dict[str, float]]) -> None:
    for dst, src in zip(target, source, strict=True):
        for metric in METRIC_WEIGHTS:
            dst[metric] += float(src[metric])


def _divide(values: list[dict[str, float]], count: int) -> list[dict[str, float]]:
    if count <= 0:
        raise ValueError("Cannot average an empty score group")
    return [{metric: score[metric] / count for metric in METRIC_WEIGHTS} for score in values]


def _minmax(values: list[float]) -> list[float]:
    low = min(values)
    high = max(values)
    if high == low:
        return [0.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def _uniform_map(source_blocks: int, keep_blocks: int) -> list[int]:
    selected = [round(index * (source_blocks - 1) / (keep_blocks - 1)) for index in range(keep_blocks)]
    if len(set(selected)) != keep_blocks:
        raise RuntimeError(f"Uniform map contains duplicates: {selected}")
    return selected


def main() -> None:
    args = parse_args()
    paths = sorted(args.partials.glob("partial-*.json"))
    if len(paths) != args.expected_shards:
        raise RuntimeError(f"Expected {args.expected_shards} complete partials, found {len(paths)}")
    partials = [json.loads(path.read_text()) for path in paths]
    if any(not partial.get("complete") for partial in partials):
        raise RuntimeError("At least one block-score partial is incomplete")
    num_blocks = int(partials[0]["num_blocks"])
    if any(int(partial["num_blocks"]) != num_blocks for partial in partials):
        raise RuntimeError("Block-score partials disagree on source block count")
    zero = [{metric: 0.0 for metric in METRIC_WEIGHTS} for _ in range(num_blocks)]
    overall = [{**row} for row in zero]
    strata = {name: [{**row} for row in zero] for name in partials[0]["by_stratum"]}
    categories = {name: [{**row} for row in zero] for name in partials[0]["by_category"]}
    sample_ids: list[str] = []
    stratum_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    for partial in partials:
        _add(overall, partial["overall"])
        for name in strata:
            _add(strata[name], partial["by_stratum"][name])
        for name in categories:
            _add(categories[name], partial["by_category"][name])
        sample_ids.extend(str(item) for item in partial["sample_ids"])
        stratum_counts.update({name: int(value) for name, value in partial["stratum_counts"].items()})
        category_counts.update({name: int(value) for name, value in partial["category_counts"].items()})
    if len(sample_ids) != 256 or len(set(sample_ids)) != 256:
        raise RuntimeError(f"Expected 256 unique scored examples, got {len(sample_ids)} / {len(set(sample_ids))}")
    if set(category_counts) != set(categories) or any(count <= 0 for count in category_counts.values()):
        raise RuntimeError(f"Category coverage is incomplete: {dict(category_counts)}")
    if set(stratum_counts) != set(strata) or any(count <= 0 for count in stratum_counts.values()):
        raise RuntimeError(f"Noise-stratum coverage is incomplete: {dict(stratum_counts)}")

    overall_mean = _divide(overall, 256)
    stratum_mean = {name: _divide(values, stratum_counts[name]) for name, values in strata.items()}
    category_mean = {name: _divide(values, category_counts[name]) for name, values in categories.items()}
    metric_norm = {
        metric: _minmax([row[metric] for row in overall_mean])
        for metric in METRIC_WEIGHTS
    }
    overall_importance = [
        sum(METRIC_WEIGHTS[metric] * metric_norm[metric][index] for metric in METRIC_WEIGHTS)
        for index in range(num_blocks)
    ]
    stratum_importance: dict[str, list[float]] = {}
    for name, rows in stratum_mean.items():
        normalized = {
            metric: _minmax([row[metric] for row in rows])
            for metric in METRIC_WEIGHTS
        }
        stratum_importance[name] = [
            sum(METRIC_WEIGHTS[metric] * normalized[metric][index] for metric in METRIC_WEIGHTS)
            for index in range(num_blocks)
        ]
    category_importance: dict[str, list[float]] = {}
    for name, rows in category_mean.items():
        normalized = {
            metric: _minmax([row[metric] for row in rows])
            for metric in METRIC_WEIGHTS
        }
        category_importance[name] = [
            sum(METRIC_WEIGHTS[metric] * normalized[metric][index] for metric in METRIC_WEIGHTS)
            for index in range(num_blocks)
        ]
    importance = []
    for index in range(num_blocks):
        worst_stratum = min(values[index] for values in stratum_importance.values())
        worst_category = min(values[index] for values in category_importance.values())
        importance.append(0.8 * overall_importance[index] + 0.1 * worst_stratum + 0.1 * worst_category)

    interior = sorted(range(1, num_blocks - 1), key=lambda index: importance[index], reverse=True)
    activation_map = sorted([0, num_blocks - 1, *interior[:args.keep_blocks - 2]])
    uniform_map = _uniform_map(num_blocks, args.keep_blocks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    activation_path = args.output_dir / "activation_20_block_map.json"
    uniform_path = args.output_dir / "uniform_20_block_map.json"
    activation_path.write_text(json.dumps({
        "strategy": "activation_ablation_structure_aware",
        "source_num_layers": num_blocks,
        "block_map": activation_map,
    }, indent=2, sort_keys=True) + "\n")
    uniform_path.write_text(json.dumps({
        "strategy": "uniform_depth_control",
        "source_num_layers": num_blocks,
        "block_map": uniform_map,
    }, indent=2, sort_keys=True) + "\n")
    manifest = {
        "format_version": 1,
        "source_revision": partials[0]["source_revision"],
        "source_commit": partials[0]["source_commit"],
        "attention_backend": partials[0]["attention_backend"],
        "calibration_roots": partials[0].get("calibration_roots", []),
        "category_quotas": partials[0].get("category_quotas", {}),
        "sample_count": 256,
        "sample_ids": sorted(sample_ids),
        "category_counts": dict(category_counts),
        "stratum_counts": dict(stratum_counts),
        "metric_weights": METRIC_WEIGHTS,
        "importance_formula": "0.8*overall + 0.1*worst_noise_stratum + 0.1*worst_prompt_category",
        "block_scores": [{
            "source_block": index,
            "importance": importance[index],
            "overall_metrics": overall_mean[index],
            "noise_strata": {name: values[index] for name, values in stratum_mean.items()},
            "prompt_categories": {name: values[index] for name, values in category_mean.items()},
        } for index in range(num_blocks)],
        "activation_block_map": activation_map,
        "uniform_block_map": uniform_map,
        "partials": [str(path) for path in paths],
    }
    manifest_path = args.output_dir / "block_score_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    import wandb

    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required for score aggregation")
    run = wandb.init(
        project=args.wandb_project,
        id=args.run_id,
        name=args.run_id,
        resume="allow",
        job_type="block-score-aggregation",
        config={
            "sample_count": 256,
            "source_revision": manifest["source_revision"],
            "source_commit": manifest["source_commit"],
            "attention_backend": manifest["attention_backend"],
            "metric_weights": METRIC_WEIGHTS,
        },
    )
    for index, value in enumerate(importance):
        run.log({"block/source_index": index, "block/importance": value}, step=index)
    artifact = wandb.Artifact(f"{args.run_id}-maps", type="block-map", metadata=manifest)
    artifact.add_file(str(manifest_path))
    artifact.add_file(str(activation_path))
    artifact.add_file(str(uniform_path))
    run.log_artifact(artifact)
    run.summary.update({
        "persistent_manifest": str(manifest_path),
        "activation_block_map": activation_map,
        "uniform_block_map": uniform_map,
        "sample_count": 256,
    })
    run.finish()


if __name__ == "__main__":
    main()
