"""Compare two dump arms of ``quality_e2e.py`` (same cases and seeds): text conditioning and endpoints.

Per clip: rel-L2 of the Qwen3-VL embeddings per token group (text tokens vs vision/reference tokens, by
``text_token_tags``) and of the final video/audio target rows. ``--samples-out`` also writes harness-format
sample files (``student_*`` = ``--test``, ``teacher_*`` = ``--ref``) so ``audio_stats_gate.py`` can score them
against a ``--ref`` arm run at seed+1 (its ``--seed1`` directory, written with ``--seed1-out``).

  python quality_compare.py --ref dumps/bf16 --test dumps/nvfp4 --samples-out samples/nvfp4 \\
      --ref-seed1 dumps/bf16-seed1 --seed1-out samples/seed1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import torch

TEXT_TAG = 0


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def load(directory: str) -> dict[str, dict]:
    """Dumps keyed by prompt hash (the seed differs between a seed and a seed+1 arm)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.pt"))):
        record = torch.load(path, map_location="cpu", weights_only=False)
        out[Path(path).stem.split("-", 1)[1]] = record
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--ref-seed1", default="")
    parser.add_argument("--samples-out", default="")
    parser.add_argument("--seed1-out", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--tags-from", default="", help="dump arm whose text_token_tags to use when the arms lack them")
    args = parser.parse_args()
    ref, test = load(args.ref), load(args.test)
    seed1 = load(args.ref_seed1) if args.ref_seed1 else {}
    tag_source = load(args.tags_from) if args.tags_from else {}
    rows = []
    for key in sorted(set(ref) & set(test)):
        a, b = test[key], ref[key]
        if a["seed"] != b["seed"]:
            raise SystemExit(f"{key}: seeds differ ({a['seed']} vs {b['seed']})")
        row = {"prompt_hash": key, "seed": a["seed"], "video": round(rel(a["video"], b["video"]), 5),
               "audio": round(rel(a["audio"], b["audio"]), 5)}
        raw_tags = b.get("text_token_tags")
        if raw_tags is None:
            raw_tags = tag_source.get(key, {}).get("text_token_tags")
        if a.get("prompt_embeds") is not None and b.get("prompt_embeds") is not None and raw_tags is not None:
            tags = torch.as_tensor(raw_tags).long()
            ea, eb = a["prompt_embeds"][0], b["prompt_embeds"][0]
            if ea.shape == eb.shape:
                text = tags == TEXT_TAG
                row["embed_all"] = round(rel(ea, eb), 5)
                row["embed_text"] = round(rel(ea[text], eb[text]), 5)
                if bool((~text).any()):
                    row["embed_vision"] = round(rel(ea[~text], eb[~text]), 5)
                row["tokens"] = int(tags.numel())
                row["vision_tokens"] = int((~text).sum())
        if key in seed1:
            row["ref_seed_floor_video"] = round(rel(seed1[key]["video"], b["video"]), 5)
            row["ref_seed_floor_audio"] = round(rel(seed1[key]["audio"], b["audio"]), 5)
        rows.append(row)
        print("COMPARE", json.dumps(row), flush=True)
        if args.samples_out:
            os.makedirs(args.samples_out, exist_ok=True)
            torch.save({"student_video": a["video"], "student_audio": a["audio"], "teacher_video": b["video"],
                        "teacher_audio": b["audio"], "seed": a["seed"], "case": key}, f"{args.samples_out}/{key}.pt")
        if args.seed1_out and key in seed1:
            os.makedirs(args.seed1_out, exist_ok=True)
            torch.save({"teacher_video": seed1[key]["video"], "teacher_audio": seed1[key]["audio"],
                        "student_video": seed1[key]["video"], "student_audio": seed1[key]["audio"],
                        "seed": seed1[key]["seed"], "case": key}, f"{args.seed1_out}/{key}.pt")
    keys = sorted({k for r in rows for k in r if k not in ("prompt_hash", "seed", "tokens", "vision_tokens")})
    summary = {"clips": len(rows), **{k: round(sum(r[k] for r in rows if k in r) / max(1, sum(k in r for r in rows)), 5)
                                       for k in keys}}
    print("COMPARE_SUMMARY", json.dumps(summary), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
