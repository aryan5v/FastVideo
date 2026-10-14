# SPDX-License-Identifier: Apache-2.0
"""Write a FastH3 OmniRef QAD student as a bf16 Diffusers ``transformer_ref`` (the NVFP4 converter's input).

A QAD run's tagged checkpoint (``<run>/best/dcp``) holds only the student's
trainable tensors (``roles.student.transformer.transformer_blocks.<i>.{attn.to_q,
to_k, to_v, to_out, to_gate_compress, norm_q, norm_k}, ff.{fc_in, fc_out}``); every
other parameter is the base checkpoint's. This script overlays those tensors on
the base ``transformer_ref`` shards and writes a complete bf16 component.

The overlay is cast to the base dtype (bf16): QAD's forward casts its masters to
bf16 before quantizing (``nvfp4_qad.nvfp4_gemm``), so the bf16 copy is exactly the
weight the trained student ran with.

Then pack it (blocks 42-49 stay bf16, FFN static scales from the pinned table)::

    python scripts/quantization/minimax_h3_omniref/export_omniref_qad.py \\
        --base <model>/transformer_ref --dcp <run>/best/dcp --dst <out>/transformer_ref_bf16
    python scripts/checkpoint_conversion/convert_minimax_h3_modelopt_nvfp4_dit.py \\
        --src <out>/transformer_ref_bf16 --dst <out>/transformer_ref \\
        --quantize-ffn --quantize-attention --quantize-gate \\
        --act-amax amax-mse_bf16_top8.json --skip-blocks 42-49

CPU only; no process group needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

STUDENT_PREFIX = "roles.student.transformer."
# FastVideo module names -> the Diffusers checkpoint names of the same tensor.
_TO_DIFFUSERS = ((re.compile(r"\.attn\.to_out\.weight$"), ".attn.to_out.0.weight"),
                 (re.compile(r"\.ff\.fc_in\.weight$"), ".ff.net.0.proj.weight"),
                 (re.compile(r"\.ff\.fc_out\.weight$"), ".ff.net.2.weight"))


def diffusers_name(fastvideo_name: str) -> str:
    for pattern, replacement in _TO_DIFFUSERS:
        if pattern.search(fastvideo_name):
            return pattern.sub(replacement, fastvideo_name)
    return fastvideo_name


def load_student(dcp_dir: Path) -> dict[str, torch.Tensor]:
    """Every ``roles.student.transformer.*`` tensor of a DCP checkpoint, full size on the CPU."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    reader = dcp.FileSystemReader(str(dcp_dir))
    metadata = reader.read_metadata()
    state: dict[str, torch.Tensor] = {}
    for key, item in metadata.state_dict_metadata.items():
        if not key.startswith(STUDENT_PREFIX):
            continue
        if not isinstance(item, TensorStorageMetadata):
            raise ValueError(f"{dcp_dir}: {key} is not a tensor entry")
        state[key] = torch.empty(item.size, dtype=item.properties.dtype)
    if not state:
        raise SystemExit(f"{dcp_dir} holds no {STUDENT_PREFIX}* tensors")
    dcp.load(state, storage_reader=reader, no_dist=True)
    return {key[len(STUDENT_PREFIX):]: value for key, value in state.items()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, type=Path, help="base transformer_ref (Diffusers shards)")
    parser.add_argument("--dcp", required=True, type=Path, help="QAD tagged checkpoint dir (e.g. <run>/best/dcp)")
    parser.add_argument("--dst", required=True, type=Path)
    args = parser.parse_args()

    index = json.loads((args.base / "diffusion_pytorch_model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    student = {diffusers_name(name): value for name, value in load_student(args.dcp).items()}
    unknown = sorted(set(student) - set(weight_map))
    if unknown:
        raise SystemExit(f"{len(unknown)} student tensors have no base counterpart, e.g. {unknown[:4]}")

    args.dst.mkdir(parents=True, exist_ok=True)
    replaced, changed, max_delta = 0, 0, 0.0
    for shard in sorted(set(weight_map.values())):
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(args.base / shard), framework="pt", device="cpu") as reader:
            for key in reader.keys():  # noqa: SIM118
                base = reader.get_tensor(key)
                if key in student:
                    trained = student[key]
                    if tuple(trained.shape) != tuple(base.shape):
                        raise SystemExit(f"{key}: student shape {tuple(trained.shape)} != base {tuple(base.shape)}")
                    trained = trained.to(base.dtype)
                    delta = (trained.float() - base.float()).abs().max().item()
                    changed += int(delta > 0)
                    max_delta = max(max_delta, delta)
                    replaced += 1
                    base = trained
                tensors[key] = base.contiguous()
        save_file(tensors, str(args.dst / shard), metadata={"format": "pt"})
    if replaced != len(student):
        raise SystemExit(f"replaced {replaced} of {len(student)} student tensors")
    shutil.copy2(args.base / "diffusion_pytorch_model.safetensors.index.json", args.dst)
    for extra in args.base.iterdir():
        if extra.is_file() and extra.suffix != ".safetensors" and extra.name != "diffusion_pytorch_model.safetensors.index.json":
            shutil.copy2(extra, args.dst / extra.name)
    receipt = {
        "base": str(args.base.resolve()),
        "dcp": str(args.dcp.resolve()),
        "dcp_metadata_sha256": sha256_file(args.dcp / ".metadata"),
        "student_tensors": replaced,
        "changed_vs_base": changed,
        "max_abs_delta_vs_base": max_delta,
    }
    (args.dst / "qad_export.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
