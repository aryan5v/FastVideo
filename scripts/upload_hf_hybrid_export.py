#!/usr/bin/env python3
"""Upload a converter-ready hybrid branch export to a private HF model repo."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--repo-name", required=True)
    args = parser.parse_args()

    token = args.token_file.read_text().strip()
    api = HfApi(token=token)
    identity = api.whoami()
    namespace = str(identity.get("name") or identity.get("fullname") or "").strip()
    if not namespace:
        raise RuntimeError("Hugging Face token did not resolve an account name")
    repo_id = f"{namespace}/{args.repo_name}"
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(args.folder),
    )
    readme = """---
license: other
base_model: FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree
library_name: fastvideo
---

# FastH3 Hybrid Preview-v1 — step 260

Converter-ready hybrid-attention branch trained with the Preview-v1 VSA
DataFree checkpoint as both frozen teacher and student backbone. This is a
delta export: use `model_spec.json` and `linear_branch/model.safetensors` with
FastVideo's `convert_vdn_h3_to_fastvideo.py` converter.

Training configuration: 16 GPUs, SP8/TP1, HSDP replicate 2 / shard 8,
5-point sigma grid with four forwards, video/audio shifts 12/3, CFG 0, t2va.
"""
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        commit_message="Add step-260 model card",
    )
    print(f"https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()

