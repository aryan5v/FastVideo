# SPDX-License-Identifier: Apache-2.0
"""Quantize a dense MiniMax-H3 / FastH3 OmniRef transformer folder to the FP8 W8A8 release format.

Format (what ``fastvideo.models.loader.fsdp_load._load_prequantized_fp8`` and the
``FastVideo-FastH3-8-Step-V2-FP8`` release use): every attention and MLP linear of the
main blocks (``transformer_blocks.N.attn.{to_q,to_k,to_v,to_out.0}`` and
``ff.net.{0.proj,2}``) is stored as ``float8_e4m3fn`` with one float32 scale per output
channel in ``<name>.weight_scale`` (shape ``[out, 1]``). Activations are quantized per
token at run time (``transformer_quant: FP8``, channel granularity). Everything else
stays in its source dtype: the AdaLN/timestep projections, embeddings, norms, the VSA
compression gates and the token refiner.

The scale and rounding match ``convert_model_to_fp8`` (per-channel path) bit for bit:
``scale = clamp(absmax / 448, min)`` and ``w / scale`` evaluated in the weight dtype.

    python quantize_minimax_h3_dit_fp8.py --src transformer_bf16 --dst transformer_ref
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)  # 448.0
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)
QUANTIZED_MODULES = (r"^transformer_blocks\.\d+\.(attn\.to_q|attn\.to_k|attn\.to_v|attn\.to_out\.0"
                     r"|ff\.net\.0\.proj|ff\.net\.2)\.weight$")
QUANTIZATION_CONFIG = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "weight_scheme": "per_channel",
    "activation_scheme": "dynamic_per_token",
    "quantized_modules": QUANTIZED_MODULES,
}
INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"


def quantize_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(fp8 [out, in], float32 scale [out, 1])`` exactly as ``convert_model_to_fp8`` does."""
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D linear weight, got shape {tuple(weight.shape)}")
    absmax = weight.detach().abs().amax(dim=1).nan_to_num().float()
    scale = (absmax / FP8_MAX).clamp(min=FP8_MIN_SCALE)
    scaled = weight / scale.to(weight.dtype).unsqueeze(1)
    return scaled.clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).contiguous(), scale.unsqueeze(1).contiguous()


def dequantize(weight_fp8: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return weight_fp8.to(dtype) * scale.to(dtype)


def convert_shard(src: Path, dst: Path, pattern: re.Pattern[str], device: str) -> tuple[dict[str, str], int, int]:
    """Write ``dst`` from ``src``; return (weight_map entries, linears quantized, bytes)."""
    tensors: dict[str, torch.Tensor] = {}
    quantized = 0
    with safe_open(str(src), framework="pt", device="cpu") as reader:
        for key in reader.keys():
            tensor = reader.get_tensor(key)
            if pattern.match(key):
                if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                    raise ValueError(f"{key} has dtype {tensor.dtype}; expected a dense floating-point weight")
                weight_fp8, scale = quantize_per_channel(tensor.to(device))
                tensors[key] = weight_fp8.cpu()
                tensors[f"{key}_scale"] = scale.cpu()
                quantized += 1
            else:
                tensors[key] = tensor
    save_file(tensors, str(dst), metadata={"format": "pt"})
    size = sum(t.numel() * t.element_size() for t in tensors.values())
    return {key: dst.name for key in tensors}, quantized, size


def convert(src_dir: Path, dst_dir: Path, device: str) -> None:
    index = json.loads((src_dir / INDEX_NAME).read_text())
    pattern = re.compile(QUANTIZED_MODULES)
    dst_dir.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    total_size = 0
    total_quantized = 0
    for shard in sorted(set(index["weight_map"].values())):
        entries, quantized, size = convert_shard(src_dir / shard, dst_dir / shard, pattern, device)
        weight_map.update(entries)
        total_size += size
        total_quantized += quantized
        print(f"{shard}: {quantized} linears quantized", flush=True)
    expected = sum(1 for key in index["weight_map"] if pattern.match(key))
    if total_quantized != expected or total_quantized == 0:
        raise RuntimeError(f"quantized {total_quantized} linears, expected {expected}")
    out_index = {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(weight_map.items()))}
    (dst_dir / INDEX_NAME).write_text(json.dumps(out_index, indent=2))
    config = json.loads((src_dir / "config.json").read_text())
    config["quantization_config"] = QUANTIZATION_CONFIG
    (dst_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"done: {total_quantized} FP8 linears, {total_size / 2**30:.2f} GiB -> {dst_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="dense transformer folder (bf16 shards + index)")
    parser.add_argument("--dst", type=Path, required=True, help="output FP8 transformer folder")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.dst.resolve() == args.src.resolve():
        parser.error("--dst must differ from --src")
    if (args.dst / INDEX_NAME).exists():
        shutil.rmtree(args.dst)
    convert(args.src, args.dst, args.device)


if __name__ == "__main__":
    main()
