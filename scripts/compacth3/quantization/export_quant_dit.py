"""Export pre-quantized DiT weights for the FastH3 DMD2 student to safetensors.

Why this does NOT go through ``VideoGenerator``
---------------------------------------------
The DiT is worker-resident: ``VideoGenerator.from_config(...)`` returns an
orchestrator handle whose process never holds an ``nn.Module`` for the
transformer (introspecting it prints "NO nn.Module attribute on the
generator").  So we load the transformer directly with the very loader the
generator uses -- ``PipelineComponentLoader.load_module`` -- and a
``FastVideoArgs`` built through the supported compat adapter
``fastvideo.api.compat.generator_config_to_fastvideo_args``, which is what
pins ``transformer_quant`` onto ``dit_config.quant_config``.

That pin is what makes the linears get built with the quant method attached;
the loader's post-load hook ``_maybe_quantize_model``
(``fastvideo/models/loader/fsdp_load.py``) then dispatches on that method and
calls ``convert_model_to_<scheme>`` to materialize the quantized buffers.

Must be a real file on disk (not stdin): FastVideo workers re-execute
``__main__`` via ``runpy``, and a heredoc has no path.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

SPRINT = "/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829"
M = "/mnt/nfs/vlm-aryan/fasth3-h3-serve-cookbook-eval-20260831/repo-main-3d8ac9d1"
HARNESS = f"{M}/examples/inference/basic/basic_fasth3.py"

# lane -> the registry name resolved by
# fastvideo.layers.quantization.get_quantization_config
LANES = {
    "nvfp4": "NVFP4H3",
    "int8": "INT8Affine",
    "w4a16": "W4A16",
}

# Buffer names carrying the quantized payload, per scheme.  NVFP4 has a real
# serializer/deserializer pair (save_nvfp4_checkpoint / load_nvfp4_checkpoint);
# the other two do not (see the lane note printed by main()).
BUFFERS = {
    "nvfp4": ("_nvfp4_weight", "_nvfp4_weight_scale", "_weight_global_sf", "_nvfp4_alpha"),
    "int8": ("_int8_affine_codes", "_int8_affine_scales", "_int8_affine_biases"),
    "w4a16": ("_w4a16_codes", "_w4a16_scales", "_w4a16_zeros"),
}


def log(msg: str) -> None:
    print(f"[export] {msg}", flush=True)


def build_fastvideo_args(model_path: str, quant_name: str, num_gpus: int):
    """Harness args -> api GeneratorConfig -> FastVideoArgs (supported path)."""
    spec = importlib.util.spec_from_file_location("fasth3_harness", HARNESS)
    harness = importlib.util.module_from_spec(spec)
    sys.modules["fasth3_harness"] = harness
    spec.loader.exec_module(harness)

    # NOTE: argparse treats a passed sequence as the full argument list (it does
    # not strip a program name), so no prog element here.
    argv = [
        "--model-path", model_path,
        "--prompt", "quant-export",
        "--num-gpus", str(num_gpus),
        "--transformer-quant", quant_name,
        "--no-fa4",
        "--no-inference-torch-compile",
        "--steps", "5",
    ]
    args = harness.parse_args(argv)
    args.fa4 = False
    harness.configure_environment(args)

    config = harness.build_generator_config(args)
    log(f"GeneratorConfig built: model_path={config.model_path} "
        f"num_gpus={config.engine.num_gpus} "
        f"transformer_quant={config.engine.quantization.transformer_quant}")

    from fastvideo.api.compat import generator_config_to_fastvideo_args
    fastvideo_args = generator_config_to_fastvideo_args(config)
    log(f"FastVideoArgs built: inference_mode={fastvideo_args.inference_mode} "
        f"training_mode={fastvideo_args.training_mode} "
        f"use_fsdp_inference={fastvideo_args.use_fsdp_inference} "
        f"hsdp_shard_dim={fastvideo_args.hsdp_shard_dim}")
    return fastvideo_args


def load_dit(fastvideo_args, transformer_path: str):
    from fastvideo.models.loader.component_loader import PipelineComponentLoader

    dit_config = fastvideo_args.pipeline_config.dit_config
    quant_config = getattr(dit_config, "quant_config", None)
    if quant_config is None:
        raise RuntimeError(
            "dit_config.quant_config is None after the compat adapter ran -- the "
            "quant config was not pinned, so no linear will be built quantized.")
    log(f"dit_config.quant_config = {type(quant_config).__name__} (name={quant_config.get_name()})")

    log(f"loading transformer from {transformer_path}")
    model = PipelineComponentLoader.load_module(
        module_name="transformer",
        component_model_path=transformer_path,
        transformers_or_diffusers="diffusers",
        fastvideo_args=fastvideo_args,
    )
    log(f"loaded class={type(model).__name__}")
    return model


def scheme_tagged(model, lane: str) -> list[tuple[str, object]]:
    """(fqn, module) pairs whose quant_method belongs to this lane's scheme."""
    from fastvideo.layers.quantization.int8_affine_config import INT8AffineQuantizeMethod
    from fastvideo.layers.quantization.nvfp4_config import NVFP4QuantizeMethod
    from fastvideo.layers.quantization.w4a16_config import W4A16QuantizeMethod

    wanted = {
        "nvfp4": NVFP4QuantizeMethod,
        "int8": INT8AffineQuantizeMethod,
        "w4a16": W4A16QuantizeMethod,
    }[lane]
    return [(fqn, mod) for fqn, mod in model.named_modules()
            if isinstance(getattr(mod, "quant_method", None), wanted)]


def save_int8_or_w4a16(model, lane: str, path: str, tagged) -> dict:
    """No serializer exists for this lane in this checkout.

    ``fastvideo/layers/quantization/{int8_affine,w4a16}_config.py`` export no
    ``save_*_checkpoint`` and no sidecar format -- their ``__all__`` lists only
    the quantizer, the config, the quantize method and the converter.  Writing
    a made-up layout here would produce a file no loader can read, so this
    stops with the evidence instead.
    """
    raise RuntimeError(
        f"lane {lane!r}: no sidecar serializer exists in this checkout. "
        f"grepped 'save_int8_affine_checkpoint' / 'save_w4a16_checkpoint' across the "
        f"whole tree at {M}: zero hits. "
        f"{len(tagged)} layers ARE quantized in memory (conversion receipt above is real) "
        f"and their buffers are {BUFFERS[lane]}, but there is no encoder -- and no "
        f"matching decoder -- so nothing could load the result.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True, choices=sorted(LANES))
    parser.add_argument("--model-path", required=True,
                        help="checkpoint dir, e.g. .../inference/checkpoint-1400")
    parser.add_argument("--out", required=True, help="output directory for the sidecar")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="1 keeps the load whole-model on cuda:0 (no FSDP)")
    args = parser.parse_args()

    lane = args.lane
    quant_name = LANES[lane]
    transformer_path = os.path.join(args.model_path, "transformer")
    if not os.path.isdir(transformer_path):
        raise SystemExit(f"no transformer/ subdir under {args.model_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"lane={lane} quant_name={quant_name}")
    log(f"model_path={args.model_path}")
    log(f"out_dir={out_dir}")

    t0 = time.time()
    fastvideo_args = build_fastvideo_args(args.model_path, quant_name, args.num_gpus)
    model = load_dit(fastvideo_args, transformer_path)
    tagged = scheme_tagged(model, lane)
    log(f"CONVERSION CHECK: {len(tagged)} {quant_name}-tagged linear layers "
        f"present after load (conversion receipt above, from _maybe_quantize_model)")
    if not tagged:
        raise RuntimeError(
            f"no {quant_name}-tagged linears found: the quant config did not cover "
            f"any layer path, so nothing was quantized. Refusing to write an empty sidecar.")

    total_params = sum(p.numel() for p in model.parameters())
    log(f"model parameters: {total_params / 1e9:.2f}B  load+convert took {time.time() - t0:.1f}s")

    if lane == "nvfp4":
        from fastvideo.layers.quantization.nvfp4_config import save_nvfp4_checkpoint
        target = out_dir / "nvfp4_weights.safetensors"
        receipt = save_nvfp4_checkpoint(
            model, target,
            extra_metadata={"model": "FastH3-20B-42block-DMD2", "source": args.model_path})
    else:
        save_int8_or_w4a16(model, lane, str(out_dir), tagged)

    size_bytes = os.path.getsize(receipt["path"])
    log("=" * 72)
    log(f"EXPORT RECEIPT (lane={lane})")
    log(json.dumps(receipt, indent=2))
    log(f"FILE PATH  : {receipt['path']}")
    log(f"FILE SIZE  : {size_bytes / 1e9:.3f} GB ({size_bytes / (1 << 30):.3f} GiB)")
    log(f"MODULE COUNT: {len(tagged)}")
    log("=" * 72)

    with open(os.path.join(out_dir, "export_receipt.json"), "w") as handle:
        json.dump({"lane": lane, "quant_name": quant_name, "file": receipt["path"],
                   "size_bytes": size_bytes, "module_count": len(tagged),
                   "receipt": receipt}, handle, indent=2)
    log(f"wrote {os.path.join(out_dir, 'export_receipt.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
