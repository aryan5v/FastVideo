# SPDX-License-Identifier: Apache-2.0
"""Export a MiniMax-H3 DMD2 training checkpoint's student into an inference model dir.

The modular trainer's DCP checkpoints hold every role and optimizer
(``roles.student.transformer.*`` is the piece inference needs, stored as the
fp32 master weights). This script streams just those tensors out of the DCP
shards in bounded memory (no process group, no GPU), renames them from
fastvideo layer names back to the on-disk checkpoint convention (the inverse
of ``MiniMaxH3ArchConfig.param_names_mapping``), casts to bf16, and writes a
sharded safetensors ``transformer/`` next to symlinks into the base model dir
for every other component — so the output loads through the standard
inference pipeline with ~66 GB of new bytes instead of a full copy.

One param per block has no on-disk counterpart: ``attn.to_gate_compress``,
the VSA gate. The base checkpoint default-initializes it; a VSA-trained
student's gate is learned, so it is exported under its fastvideo name (no
mapping rule touches it, and the loader resolves it to the model param
verbatim). Dense-only consumers ignore it.

Usage::

    python scripts/checkpoint_conversion/export_h3_dmd2_student.py \
        --checkpoint /path/to/outputs/<run>/checkpoint-1400 \
        --output-dir /path/to/exports/<run>-step1400 \
        [--base-model /mnt/lustre/vlm-k1kong/models/MiniMax-H3] \
        [--role student] [--dtype bfloat16] [--copy-components]

``--checkpoint latest --run-dir <outputs/run>`` picks the newest
``checkpoint-*`` whose ``dcp/.metadata`` exists (the strict-resume contract).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

INVERSE_PARAM_RULES: tuple[tuple[str, str], ...] = (
    (r"^time_embedder\.fc_in\.(.*)$", r"time_embedder.linear_1.\1"),
    (r"^time_embedder\.fc_out\.(.*)$", r"time_embedder.linear_2.\1"),
    (r"^(.*)\.attn\.to_out\.(weight|bias)$", r"\1.attn.to_out.0.\2"),
    (r"^(.*)\.ff\.fc_in\.(.*)$", r"\1.ff.net.0.proj.\2"),
    (r"^(.*)\.ff\.fc_out\.(.*)$", r"\1.ff.net.2.\2"),
)

EXPECTED_NEW_PARAM_PATTERNS = (re.compile(r"\.attn\.to_gate_compress\."), )

SHARD_BUDGET_BYTES = 5 * 1024**3  # ~5 GB per safetensors shard (bf16)


def to_disk_name(name: str) -> str:
    for pattern, repl in INVERSE_PARAM_RULES:
        new, n = re.subn(pattern, repl, name)
        if n:
            return new
    return name


def find_latest_checkpoint(run_dir: Path) -> Path:
    candidates = sorted(
        (p for p in run_dir.glob("checkpoint-*") if (p / "dcp" / ".metadata").exists()),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"no complete checkpoint-*/dcp/.metadata under {run_dir}")
    return candidates[-1]


def main(args: argparse.Namespace) -> None:
    if args.checkpoint == "latest":
        if args.run_dir is None:
            raise SystemExit("--checkpoint latest requires --run-dir")
        checkpoint = find_latest_checkpoint(args.run_dir)
    else:
        checkpoint = Path(args.checkpoint)
    dcp_dir = checkpoint / "dcp"
    if not (dcp_dir / ".metadata").exists():
        raise FileNotFoundError(f"{dcp_dir}/.metadata missing — incomplete checkpoint, refusing")
    base_model = args.base_model
    if not any((base_model / name).exists() for name in ("model_index.json", "modular_model_index.json")):
        raise FileNotFoundError(f"{base_model} does not look like a model dir "
                                "(no model_index.json or modular_model_index.json)")
    out_dtype = getattr(torch, args.dtype)
    prefix = f"roles.{args.role}.transformer."

    from torch.distributed.checkpoint import FileSystemReader
    import torch.distributed.checkpoint as dcp

    reader = FileSystemReader(str(dcp_dir))
    metadata = reader.read_metadata()
    param_meta = {
        key: meta
        for key, meta in metadata.state_dict_metadata.items()
        if key.startswith(prefix)
    }
    if not param_meta:
        raise SystemExit(f"no keys under {prefix!r} in {dcp_dir}")
    print(f"{checkpoint.name}: {len(param_meta)} tensors under {prefix!r}")

    base_transformer = base_model / "transformer"
    base_index = base_transformer / "diffusion_pytorch_model.safetensors.index.json"
    base_keys: set[str] = set()
    if base_index.exists():
        base_keys = set(json.loads(base_index.read_text())["weight_map"])

    def nbytes(meta) -> int:
        numel = 1
        for dim in meta.size:
            numel *= dim
        return numel * torch.finfo(out_dtype).bits // 8

    ordered = sorted(param_meta)
    shards: list[list[str]] = [[]]
    acc = 0
    for key in ordered:
        size = nbytes(param_meta[key])
        if shards[-1] and acc + size > SHARD_BUDGET_BYTES:
            shards.append([])
            acc = 0
        shards[-1].append(key)
        acc += size

    out_transformer = args.output_dir / "transformer"
    out_transformer.mkdir(parents=True, exist_ok=True)

    weight_map: dict[str, str] = {}
    total_size = 0
    unexpected_new: list[str] = []
    n_shards = len(shards)
    for shard_idx, keys in enumerate(shards, start=1):
        fname = f"diffusion_pytorch_model-{shard_idx:05d}-of-{n_shards:05d}.safetensors"
        state = {
            key: torch.empty(tuple(param_meta[key].size), dtype=param_meta[key].properties.dtype)
            for key in keys
        }
        dcp.load(state, checkpoint_id=str(dcp_dir))
        tensors: dict[str, torch.Tensor] = {}
        for key, tensor in state.items():
            disk_name = to_disk_name(key[len(prefix):])
            if base_keys and disk_name not in base_keys:
                if not any(p.search(disk_name) for p in EXPECTED_NEW_PARAM_PATTERNS):
                    unexpected_new.append(disk_name)
            tensors[disk_name] = tensor.to(out_dtype).contiguous()
            weight_map[disk_name] = fname
            total_size += tensors[disk_name].numel() * tensors[disk_name].element_size()
        save_file(tensors, str(out_transformer / fname))
        del state, tensors
        print(f"  wrote {fname} ({len(keys)} tensors)")

    if unexpected_new:
        raise SystemExit("Export produced keys unknown to the base checkpoint (mapping drift?):\n  " +
                         "\n  ".join(sorted(unexpected_new)[:20]))

    (out_transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2))
    shutil.copy2(base_transformer / "config.json", out_transformer / "config.json")

    for entry in sorted(base_model.iterdir()):
        if entry.name == "transformer":
            continue
        target = args.output_dir / entry.name
        if target.exists() or target.is_symlink():
            continue
        if args.copy_components:
            if entry.is_dir():
                shutil.copytree(entry, target)
            else:
                shutil.copy2(entry, target)
        else:
            target.symlink_to(entry.resolve())

    print(f"Export complete: {args.output_dir}")
    print(f"  transformer: {len(weight_map)} tensors, {total_size / 1024**3:.1f} GiB "
          f"({args.dtype}), {n_shards} shards")
    print(f"  other components {'copied' if args.copy_components else 'symlinked'} from {base_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="checkpoint-N dir, or 'latest' with --run-dir")
    parser.add_argument("--run-dir", type=Path, default=None, help="training output dir for --checkpoint latest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model",
                        type=Path,
                        default=Path("/mnt/lustre/vlm-k1kong/models/MiniMax-H3"),
                        help="base model dir supplying config + non-transformer components")
    parser.add_argument("--role", default="student", help="training role to export (student|critic)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"])
    parser.add_argument("--copy-components",
                        action="store_true",
                        help="copy non-transformer components instead of symlinking")
    main(parser.parse_args())
