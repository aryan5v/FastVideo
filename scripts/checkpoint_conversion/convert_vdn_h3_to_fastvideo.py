# SPDX-License-Identifier: Apache-2.0
"""Merge a hybrid MiniMax H3 (VDN-H3) checkpoint onto a FastVideo transformer.

Does not vendor the external runtime. The native FastVideo hybrid attention
modules own the target state-dict; this script:

1. Starts from a dense MiniMax H3 / FastH3 ``transformer/`` directory.
2. Overlays the hybrid branch tensors (``linear_attention``, ``to_out_linear``,
   ``softmax_gate``, optional short-conv taps).
3. Merges any PEFT adapters (Stage-B ``default``, optional DMD ``turbo``) into
   the dense QKV/O projections.
4. Rewrites ``attn.orig.*`` onto ``attn.*`` and stamps ``hybrid_attention: true``
   (plus the window / delta-rule knobs) on ``config.json``.

The rest of a FastVideo model dir (VAE, text encoder, schedulers) is unchanged;
point ``--dst`` at a new ``transformer/`` and symlink or copy the other
components from the dense snapshot.

Usage::

    python scripts/checkpoint_conversion/convert_vdn_h3_to_fastvideo.py \\
        --base /path/to/MiniMax-H3/transformer \\
        --hybrid /path/to/vdn-minimax-h3/ckpts/stage-dmd-step-250 \\
        --dst /path/to/FastH3-Hybrid/transformer

    python examples/inference/basic/basic_fasth3.py \\
        --model-path /path/to/FastH3-Hybrid --no-vsa --prompt "..."

Weights stay under the MiniMax H3 Community License of the source checkpoint.
The conversion itself is Apache-2.0.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fastvideo.models.dits.minimax_h3_hybrid.checkpoint import (
    hybrid_arch_fields_from_spec,
    lora_scale_from_adapter_config,
    merge_lora_pairs,
    remap_vdn_key,
    stamp_hybrid_config,
)

INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
WEIGHT_NAME = "diffusion_pytorch_model.safetensors"
SPEC_NAME = "model_spec.json"
BRANCH_DIR = "linear_branch"
ADAPTERS_DIR = "adapters"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, type=Path, help="Dense MiniMax H3 transformer component dir.")
    parser.add_argument("--hybrid",
                        required=True,
                        type=Path,
                        help="VDN exploded checkpoint dir (model_spec.json + linear_branch/).")
    parser.add_argument("--dst", required=True, type=Path, help="Output FastVideo transformer component dir.")
    parser.add_argument("--adapters",
                        default=None,
                        help="Comma-separated adapter names to merge (default: every adapters/ child, "
                        "default then turbo). Pass '' to skip LoRA merge.")
    parser.add_argument("--shard-size-gb", type=float, default=5.0, help="Max shard size when writing an index.")
    return parser.parse_args()


def _safetensors_files(root: Path) -> list[Path]:
    if (root / INDEX_NAME).is_file():
        index = json.loads((root / INDEX_NAME).read_text())
        return [root / shard for shard in sorted(set(index["weight_map"].values()))]
    singles = sorted(root.glob("*.safetensors"))
    if not singles:
        raise FileNotFoundError(f"No safetensors shards under {root}")
    return singles


def load_safetensors_dir(root: Path) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for shard in _safetensors_files(root):
        with safe_open(str(shard), framework="pt") as handle:
            for key in handle.keys():
                out[key] = handle.get_tensor(key)
    return out


def overlay_hybrid_tensors(base: dict[str, torch.Tensor], hybrid: dict[str, torch.Tensor]) -> int:
    """Copy remapped hybrid tensors onto the dense base. Returns tensors written."""
    written = 0
    skipped: list[str] = []
    for source_name, tensor in hybrid.items():
        if ".lora_A." in source_name or ".lora_B." in source_name:
            continue
        target = remap_vdn_key(source_name)
        if target is None:
            skipped.append(source_name)
            continue
        base[target] = tensor
        written += 1
    if skipped:
        print(f"skipped {len(skipped)} dropout/non-parameter keys (e.g. {skipped[0]})")
    return written


def _adapter_dirs(hybrid_root: Path, requested: str | None) -> list[Path]:
    adapters_root = hybrid_root / ADAPTERS_DIR
    if requested == "":
        return []
    if not adapters_root.is_dir():
        return []
    children = sorted(path for path in adapters_root.iterdir() if path.is_dir())
    if requested is None:
        return children
    wanted = [name.strip() for name in requested.split(",") if name.strip()]
    by_name = {path.name: path for path in children}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise FileNotFoundError(f"Requested adapters not found under {adapters_root}: {missing}")
    return [by_name[name] for name in wanted]


def merge_adapters(weights: dict[str, torch.Tensor], hybrid_root: Path, requested: str | None) -> int:
    total = 0
    for adapter_dir in _adapter_dirs(hybrid_root, requested):
        tensors = load_safetensors_dir(adapter_dir)
        rank = None
        for key, tensor in tensors.items():
            if ".lora_A." in key:
                rank = int(tensor.shape[0])
                break
        config_path = adapter_dir / "adapter_config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        scale = lora_scale_from_adapter_config(config, rank)
        merged = merge_lora_pairs(weights, tensors, scale=scale)
        print(f"merged adapter {adapter_dir.name}: {merged} pairs, scale={scale:g}")
        total += merged
    return total


def write_sharded(dst: Path, tensors: dict[str, torch.Tensor], shard_size_gb: float) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    max_bytes = int(shard_size_gb * (1024**3))
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    for key in sorted(tensors):
        tensor = tensors[key].contiguous()
        tensors[key] = tensor
        nbytes = tensor.nbytes
        if current and current_bytes + nbytes > max_bytes:
            shards.append(current)
            current = {}
            current_bytes = 0
        current[key] = tensor
        current_bytes += nbytes
    if current:
        shards.append(current)
    if len(shards) == 1:
        save_file(shards[0], str(dst / WEIGHT_NAME))
        return
    weight_map: dict[str, str] = {}
    for index, shard in enumerate(shards):
        name = f"diffusion_pytorch_model-{index + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(dst / name))
        for key in shard:
            weight_map[key] = name
    (dst / INDEX_NAME).write_text(json.dumps({"metadata": {"total_size": sum(t.nbytes for t in tensors.values())},
                                              "weight_map": weight_map},
                                             indent=2) + "\n")


def main() -> None:
    args = parse_args()
    base, hybrid, dst = args.base, args.hybrid, args.dst
    if not (base / "config.json").is_file():
        raise FileNotFoundError(f"{base} is not a transformer component (missing config.json).")
    if not (hybrid / SPEC_NAME).is_file() and not (hybrid / BRANCH_DIR).is_dir():
        raise FileNotFoundError(f"{hybrid} is not an exploded VDN checkpoint "
                                f"(need {SPEC_NAME} and/or {BRANCH_DIR}/).")

    print(f"loading dense transformer from {base}")
    weights = load_safetensors_dir(base)
    branch_root = hybrid / BRANCH_DIR if (hybrid / BRANCH_DIR).is_dir() else hybrid
    print(f"overlaying hybrid tensors from {branch_root}")
    written = overlay_hybrid_tensors(weights, load_safetensors_dir(branch_root))
    print(f"wrote {written} hybrid tensors")
    merge_adapters(weights, hybrid, args.adapters)

    spec_path = hybrid / SPEC_NAME
    spec = json.loads(spec_path.read_text()) if spec_path.is_file() else {}
    config = stamp_hybrid_config(json.loads((base / "config.json").read_text()), hybrid_arch_fields_from_spec(spec))
    if dst.exists():
        shutil.rmtree(dst)
    write_sharded(dst, weights, args.shard_size_gb)
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote FastVideo hybrid transformer to {dst} ({len(weights)} tensors, "
          f"hybrid_attention={config['hybrid_attention']})")


if __name__ == "__main__":
    main()
