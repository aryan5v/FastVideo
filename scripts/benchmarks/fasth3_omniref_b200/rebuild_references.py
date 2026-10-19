"""Rebuild raw OmniRef cases (prompt + reference images) from precomputed held-out rows, for end-to-end quality runs.

The held-out rows store the reference images only as clean VAE latents. This script decodes every image
reference of each selected row with the model's own video VAE to a PNG, and pairs it with the row's raw prompt
from the dataset's plan manifests, so the full pipeline (text encoder included) can run on held-out content.
The decoded references are VAE reconstructions of the originals, identical for every arm of a comparison.

One GPU::

    python rebuild_references.py --model-path <composed dir> --rows ab-vote/rows.json \\
        --manifests '<omniref-v4-processed>/manifests/*.jsonl' --cases first_frame:4 first_last_frame:3 \\
        storyboard:3 --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "distill" / "minimax_h3_omniref_qad"))


def raw_prompts(pattern: str, wanted: set[tuple[str, str, str]]) -> dict[tuple[str, str, str], dict]:
    """(source id, case, resolution) -> the plan-manifest entry (prompt and reference metadata)."""
    found = {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            for line in handle:
                if "asset-" not in line:
                    continue
                entry = json.loads(line)
                key = (entry["id"].split(":")[-1], entry.get("case"), entry.get("resolution"))
                if key in wanted:
                    found[key] = entry
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--manifests", required=True, help="glob of the dataset plan manifests (raw prompts)")
    parser.add_argument("--cases", nargs="+", required=True, help="case:count, image-reference cases only")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from generate_omniref_latents import OmniRefLatentGenerator, _tensor, prepared_references, read_row
    from PIL import Image

    from fastvideo.pipelines.basic.minimax_h3.packing import unpatchify_video_tokens

    quota = {case: int(count) for case, count in (item.split(":") for item in args.cases)}
    rows = []
    for row in json.loads(Path(args.rows).read_text()):
        if quota.get(row["case"], 0) > 0:
            quota[row["case"]] -= 1
            rows.append(row)
    prompts = raw_prompts(args.manifests, {(r["source"], r["case"], r["resolution"]) for r in rows})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MASTER_PORT", "29733")
    driver = OmniRefLatentGenerator(argparse.Namespace(model_path=args.model_path, master_port=29733))
    cases = []
    for spec in rows:
        raw = prompts.get((spec["source"], spec["case"], spec["resolution"]))
        if raw is None:
            print(f"skip {spec['id']}: no raw prompt", flush=True)
            continue
        row = read_row(spec["parquet"])
        references = prepared_references(row)
        if any(ref.media_type != "image" for ref in references):
            print(f"skip {spec['id']}: non-image reference", flush=True)
            continue
        video_rows = _tensor(row, "reference_video_latent_rows")
        offset, paths = 0, []
        for index, ref in enumerate(references):
            count = ref.num_latent_frames * (ref.latent_height // 2) * (ref.latent_width // 2)
            latent = unpatchify_video_tokens(video_rows[offset:offset + count], ref.num_latent_frames,
                                             ref.latent_height, ref.latent_width, 24, (1, 2, 2))[0]
            offset += count
            frames = driver.decode(latent)
            path = out / f"{spec['id']}-ref{index}.png"
            Image.fromarray(frames[0]).save(path)
            paths.append(str(path))
        if offset != video_rows.shape[0]:
            raise SystemExit(f"{spec['id']}: used {offset} of {video_rows.shape[0]} reference rows")
        cases.append({
            "id": spec["id"],
            "case": spec["case"],
            "resolution": spec["resolution"],
            "seed": int(spec["seed"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
            "num_frames": int(row["num_frames"]),
            "prompt": raw["prompt"],
            "references": [{"kind": "image", "path": p} for p in paths],
        })
        print(json.dumps({"case": spec["id"], "references": len(paths)}), flush=True)
    (out / "cases.json").write_text(json.dumps(cases, indent=1))
    print(f"wrote {len(cases)} cases to {out / 'cases.json'}", flush=True)
    driver.shutdown()


if __name__ == "__main__":
    main()
