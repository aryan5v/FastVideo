# SPDX-License-Identifier: Apache-2.0
"""Which NVFP4 decoder layers carry the error? Per-block and per-type sensitivity of a QAD checkpoint.

Every configuration decodes the same clips; a "dense" layer runs its trained
master weight in bf16 instead of NVFP4 (``NVFP4DecoderLinear.dense_bypass``), so
no retraining is involved. Reports LPIPS / PSNR vs the original decoder for:
all-NVFP4, all-dense (the precision-free upper bound of this checkpoint), each
single block dense, each linear type dense across blocks, and the greedy
cumulative curve of the most sensitive blocks.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "minimax_h3_vae"))
from bench_decoder import Fidelity, keep_blocks, load_h3_vae  # noqa: E402

from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import convert_decoder_to_nvfp4, nvfp4_linears  # noqa: E402

LINEAR_TYPES = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out", "ff.net.0", "ff.net.2")
FP8_MAX = 448.0


def fp4_weight(layer: torch.nn.Module) -> torch.Tensor:
    """The exact NVFP4 weight the deployed layer multiplies by, expanded to bf16."""
    from fastvideo.layers.quantization.nvfp4_dequant import dequantize_nvfp4_cuda
    from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import _quantized_weight

    packed, inv_scale, global_sf = _quantized_weight(layer.weight.detach().to(torch.bfloat16))
    return dequantize_nvfp4_cuda(packed.view(torch.uint8).contiguous(),
                                 inv_scale.view(torch.uint8).contiguous(), float(global_sf), torch.bfloat16)


def simulate(layer: torch.nn.Module, activation: str) -> None:
    """FP4 weights with ``activation`` in {"bf16", "fp8"} (per-token E4M3 fake quantization)."""
    weight = fp4_weight(layer)
    bias = layer.bias.detach().to(torch.bfloat16) if layer.bias is not None else None

    def forward(x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.bfloat16)
        if activation == "fp8":
            scale = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12) / FP8_MAX
            x = ((x.float() / scale).to(torch.float8_e4m3fn).float() * scale).to(torch.bfloat16)
        return torch.nn.functional.linear(x, weight, bias)

    layer.forward = forward


def decode(vae: torch.nn.Module, latent: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        sample = vae.decode(latent, return_dict=False)[0]
    return vae.denormalize_pixels(sample.float()).clamp(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher-vae-dir", required=True)
    parser.add_argument("--student-vae-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--keep-blocks", type=int, default=0)
    parser.add_argument("--eval-latents", required=True, help="glob of raw NCTHW latents")
    parser.add_argument("--max-clips", type=int, default=5)
    parser.add_argument("--greedy", default="1,2,3,4,6,8,12")
    parser.add_argument("--output", required=True)
    parser.add_argument("--activation-only", action="store_true",
                        help="only the W4A4 / W4A16 / W4A8 comparison (skip per-block and per-type sweeps)")
    args = parser.parse_args()

    device = torch.device("cuda")
    import os
    os.environ.setdefault("FASTVIDEO_H3_VAE_TILE_BATCH", "64")
    teacher = load_h3_vae(args.teacher_vae_dir, device)
    clips = [torch.load(p, map_location="cpu").float().to(device)
             for p in sorted(glob.glob(args.eval_latents))[:args.max_clips]]
    references = [decode(teacher, clip, torch.float16) for clip in clips]
    del teacher
    torch.cuda.empty_cache()

    student = load_h3_vae(args.student_vae_dir, device)
    if args.keep_blocks:
        keep_blocks(student.decoder, args.keep_blocks)
    convert_decoder_to_nvfp4(student.decoder, compute_dtype=torch.bfloat16, act_scale="unit")
    state = torch.load(args.checkpoint, map_location=device)
    student.decoder.load_state_dict(state["decoder"], strict=True)
    student.post_quant_conv.load_state_dict(state["post_quant_conv"], strict=True)
    student.requires_grad_(False).eval()
    fidelity = Fidelity(device)

    named = {name: module for name, module in student.decoder.named_modules() if module in nvfp4_linears(student)}

    def score(dense: set[str]) -> dict[str, float]:
        for name, layer in named.items():
            layer.dense_bypass = name in dense
        values: dict[str, list[float]] = {}
        for clip, reference in zip(clips, references, strict=True):
            for key, value in fidelity(decode(student, clip, torch.bfloat16), reference).items():
                values.setdefault(key, []).append(value)
        return {key: float(np.mean(v)) for key, v in values.items()}

    def block_layers(index: int) -> set[str]:
        return {name for name in named if name.startswith(f"transformer_blocks.{index}.")}

    results: dict[str, object] = {}
    results["all_nvfp4"] = score(set())
    results["all_dense"] = score(set(named))
    print(json.dumps({"all_nvfp4": results["all_nvfp4"], "all_dense": results["all_dense"]}), flush=True)
    fused = student.decoder.fused_blocks_forward
    student.decoder.fused_blocks_forward = None  # the fused stack bypasses per-layer forward
    first = next(iter(named.values()))
    master = first.weight.detach().float()
    print(json.dumps({"fp4_weight_rel_err": float((fp4_weight(first).float() - master).norm() / master.norm())}),
          flush=True)
    for activation in ("bf16", "fp8"):
        for layer in named.values():
            layer.dense_bypass = False
            simulate(layer, activation)
        values: dict[str, list[float]] = {}
        for clip, reference in zip(clips, references, strict=True):
            for key, value in fidelity(decode(student, clip, torch.bfloat16), reference).items():
                values.setdefault(key, []).append(value)
        results[f"w4a{16 if activation == 'bf16' else 8}"] = {key: float(np.mean(v)) for key, v in values.items()}
        print(json.dumps({f"w4a{16 if activation == 'bf16' else 8}": results[f"w4a{16 if activation == 'bf16' else 8}"]}),
              flush=True)
    for layer in named.values():
        del layer.forward  # restore the class forward
    student.decoder.fused_blocks_forward = fused
    if args.activation_only:
        Path(args.output).write_text(json.dumps(results, indent=2))
        return
    num_blocks = len(student.decoder.transformer_blocks)
    per_block = {}
    for index in range(num_blocks):
        per_block[index] = score(block_layers(index))
        print(json.dumps({"block": index, **per_block[index]}), flush=True)
    results["per_block"] = per_block
    per_type = {}
    for kind in LINEAR_TYPES:
        per_type[kind] = score({name for name in named if f".{kind}" in name})
        print(json.dumps({"type": kind, **per_type[kind]}), flush=True)
    results["per_type"] = per_type
    ranked = sorted(per_block, key=lambda i: per_block[i]["lpips"])
    greedy = {}
    for count in (int(v) for v in args.greedy.split(",")):
        if count > num_blocks:
            continue
        dense = set().union(*(block_layers(i) for i in ranked[:count]))
        greedy[count] = {"blocks": ranked[:count], **score(dense)}
        print(json.dumps({"greedy": count, **greedy[count]}), flush=True)
    results["greedy"] = greedy
    Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
