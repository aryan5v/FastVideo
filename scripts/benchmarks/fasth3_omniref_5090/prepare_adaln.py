# SPDX-License-Identifier: Apache-2.0
"""Precompute the checkpoint's fixed PDD AdaLN tables with one projection on GPU at a time."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from fastvideo.layers.mlp import MLP
from fastvideo.layers.pdd import PDDModalitySchedule, build_pdd_sampling_plan
from fastvideo.layers.visual_embedding import Timesteps
from fastvideo.models.dits.minimax_h3 import MiniMaxH3AdaLayerNormModulation
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    component = args.model / "transformer_ref"
    config = json.loads((component / "config.json").read_text())
    if config.get("adaln_rank"):
        raise ValueError("This utility requires the full-rank checkpoint.")
    contract = json.loads((args.model / "fastvideo_inference.json").read_text())
    index = json.loads((component / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]

    def tensor(key, dtype):
        with safe_open(component / index[key], framework="pt", device="cpu") as handle:
            return handle.get_tensor(key).to(device="cuda", dtype=dtype)

    time_proj = Timesteps(config["freq_dim"], flip_sin_to_cos=True, downscale_freq_shift=0).cuda()
    embedder = MLP(config["freq_dim"], config["time_embed_hidden_dim"], config["time_embed_dim"],
                   act_type="silu", dtype=torch.float32).cuda()
    state = {}
    for source, target in (("linear_1", "fc_in"), ("linear_2", "fc_out")):
        for suffix in ("weight", "bias"):
            state[f"{target}.{suffix}"] = tensor(f"time_embedder.{source}.{suffix}", torch.float32)
    embedder.load_state_dict(state)
    del state
    schedules = {name: PDDModalitySchedule(shift=contract[f"{name}_scheduler_shift"])
                 for name in ("video", "audio")}
    plan = build_pdd_sampling_plan(contract["pdd_step_indices"], schedules, device="cuda")
    timesteps = {}
    for name in schedules:
        scheduler = MiniMaxH3Scheduler(shift=schedules[name].shift)
        scheduler.set_timesteps(sigmas=plan.node_sigmas[name].float(), device="cuda")
        timesteps[name] = scheduler.timesteps
    inputs = {}
    for video, audio in zip(timesteps["video"], timesteps["audio"], strict=True):
        for audio_reference in (False, True):
            values = [video.item(), audio.item(), max(video.item(), 0.999)]
            if audio_reference:
                values.append(1.0)
            unique = torch.tensor(values, dtype=torch.float32).unique(sorted=True).cuda()
            temb = embedder(time_proj(unique).float())
            key = (tuple(unique.tolist()), tuple(temb.shape), str(temb.dtype))
            inputs[key] = temb
    tables = {}
    for block in range(config["num_layers"]):
        prefix = f"transformer_blocks.{block}.adaln_proj.linear"
        weight = tensor(f"{prefix}.weight", torch.bfloat16)
        bias = tensor(f"{prefix}.bias", torch.bfloat16)
        with torch.device("meta"):
            layer = MiniMaxH3AdaLayerNormModulation(config["time_embed_dim"], config["hidden_size"])
        layer.linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.linear.bias = torch.nn.Parameter(bias, requires_grad=False)
        table = {}
        for key, temb in inputs.items():
            table[key] = torch.nn.functional.linear(torch.nn.functional.silu(temb).to(weight.dtype),
                                                    weight, bias)
            expected = layer(temb)
            actual = table[key].view(-1, 6 * config["hidden_size"]).chunk(6, dim=-1)
            for left, right in zip(expected, actual, strict=True):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
        layer.enable_host_cache(table)
        for key, temb in inputs.items():
            layer._cache_key = key
            actual = layer(temb)
            expected = table[key].view(-1, 6 * config["hidden_size"]).chunk(6, dim=-1)
            for left, right in zip(expected, actual, strict=True):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
        tables[block] = {repr(key): value.cpu() for key, value in table.items()}
        del layer, table, weight, bias
        print(f"AdaLN {block + 1}/{config['num_layers']}, {len(inputs)} timestep keys", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tables, args.output)
    print(f"Saved {args.output}; checked exact equality with native modulation and cached lookup", flush=True)


if __name__ == "__main__":
    main()
