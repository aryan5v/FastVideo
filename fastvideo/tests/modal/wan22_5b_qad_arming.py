# SPDX-License-Identifier: Apache-2.0
"""5B QAD arming check — run-6 launch gate (cheap, no training loop).

Instantiates the Wan2.2-TI2V-5B student transformer from the FullAttn Diffusers
checkpoint (local path or HF id), applies
``MLXQuantizationAwareCallback`` (group_size=64, bits=8), and asserts it
fake-quantizes the expected weight count (~30 blocks × 10 + head ≈ 300+).

Run on Modal (CUDA) so the HF download can land on the volume:

    modal run fastvideo/tests/modal/launch_l40s_job.py \\
      --command "python fastvideo/tests/modal/wan22_5b_qad_arming.py" \\
      --gpu-type L40S --num-gpus 1 --install-extra dev --pr-number <PR#> \\
      --env-vars "MASTER_ADDR=localhost,MASTER_PORT=29561,FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA" \\
      --commit-volume

Or locally if the transformer is already at ``FASTVIDEO_WAN22_5B_ROOT`` /
``~/models/fastwan22_5b``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


DEFAULT_HF_ID = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
_MIN_EXPECTED_WEIGHTS = 300  # 30 blocks × ~10 + head/embedders


def _resolve_checkpoint() -> tuple[Path, Path]:
    root = Path(os.environ.get("FASTVIDEO_WAN22_5B_ROOT", str(Path.home() / "models" / "fastwan22_5b")))
    ckpt = root / "transformer" / "diffusion_pytorch_model.safetensors"
    cfg = root / "transformer" / "config.json"
    if ckpt.exists() and cfg.exists() and ckpt.stat().st_size > 1_000_000_000:
        print(f"[arming] using local checkpoint under {root}", flush=True)
        return ckpt, cfg

    # Download transformer-only from HF into HF_HOME (Modal volume when set).
    print(f"[arming] local weights missing; downloading {DEFAULT_HF_ID} transformer...", flush=True)
    from huggingface_hub import hf_hub_download

    cfg_path = Path(
        hf_hub_download(DEFAULT_HF_ID, "transformer/config.json", local_dir=str(root)))
    # Prefer snapshot layout: root/transformer/...
    if not (root / "transformer" / "config.json").exists():
        # hf_hub_download may place under root/transformer or root/
        pass
    ckpt_path = Path(
        hf_hub_download(
            DEFAULT_HF_ID,
            "transformer/diffusion_pytorch_model.safetensors",
            local_dir=str(root),
        ))
    # Normalize to root/transformer/*
    cfg = root / "transformer" / "config.json"
    ckpt = root / "transformer" / "diffusion_pytorch_model.safetensors"
    if not cfg.exists():
        cfg = cfg_path
    if not ckpt.exists():
        ckpt = ckpt_path
    print(f"[arming] checkpoint={ckpt} ({ckpt.stat().st_size / 1e9:.2f} GB)", flush=True)
    return ckpt, cfg


def _load_transformer(checkpoint: Path, config_path: Path):
    import torch
    from safetensors.torch import load_file

    from fastvideo.configs.models.dits.wanvideo import WanVideoArchConfig, WanVideoConfig
    from fastvideo.distributed.parallel_state import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.models.dits.wanvideo import WanTransformer3DModel
    from fastvideo.models.loader.utils import get_param_names_mapping, hf_to_custom_state_dict

    maybe_init_distributed_environment_and_model_parallel(1, 1)

    hf_config = json.loads(config_path.read_text())
    arch_kwargs = {
        k: v
        for k, v in hf_config.items()
        if k not in {"_class_name", "_name_or_path", "_diffusers_version"}
    }
    cfg = WanVideoConfig(arch_config=WanVideoArchConfig(
        num_attention_heads=int(arch_kwargs["num_attention_heads"]),
        attention_head_dim=int(arch_kwargs["attention_head_dim"]),
        in_channels=int(arch_kwargs["in_channels"]),
        out_channels=int(arch_kwargs["out_channels"]),
        text_dim=int(arch_kwargs["text_dim"]),
        freq_dim=int(arch_kwargs["freq_dim"]),
        ffn_dim=int(arch_kwargs["ffn_dim"]),
        num_layers=int(arch_kwargs["num_layers"]),
        patch_size=tuple(arch_kwargs["patch_size"]),
        cross_attn_norm=bool(arch_kwargs.get("cross_attn_norm", True)),
        qk_norm=arch_kwargs.get("qk_norm", "rms_norm_across_heads"),
        eps=float(arch_kwargs.get("eps", 1e-6)),
        rope_max_seq_len=int(arch_kwargs.get("rope_max_seq_len", 1024)),
        added_kv_proj_dim=arch_kwargs.get("added_kv_proj_dim"),
        image_dim=arch_kwargs.get("image_dim"),
        pos_embed_seq_len=arch_kwargs.get("pos_embed_seq_len"),
    ))
    model = WanTransformer3DModel(config=cfg, hf_config=hf_config).eval()
    raw = load_file(str(checkpoint), device="cpu")
    custom_sd, _ = hf_to_custom_state_dict(raw.items(), get_param_names_mapping(model.param_names_mapping))
    missing, unexpected = model.load_state_dict(custom_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys loading 5B: {unexpected[:8]}")
    if missing:
        print(f"[arming] missing keys ({len(missing)}): {missing[:6]}...", flush=True)
    # CPU fp32 is fine for arming (no forward needed).
    return model.to(dtype=torch.float32)


def _import_mlx_qat_callback():
    """Load MLXQuantizationAwareCallback without ``fastvideo.train`` package init.

    ``fastvideo.train.__init__`` pulls the full Trainer stack (torchdata, pyarrow,
    ...). Modal ``install-extra dev`` has those; a lean arming smoke may not.
    """
    import importlib.util
    import types as _types

    # Parent packages without running their heavy __init__ side effects.
    for pkg in ("fastvideo.train", "fastvideo.train.callbacks"):
        if pkg not in sys.modules:
            sys.modules[pkg] = _types.ModuleType(pkg)

    if "fastvideo.train.callbacks.callback" not in sys.modules:
        callback_mod = _types.ModuleType("fastvideo.train.callbacks.callback")

        class Callback:  # noqa: D101
            pass

        callback_mod.Callback = Callback
        sys.modules["fastvideo.train.callbacks.callback"] = callback_mod

    # parents: modal -> tests -> fastvideo
    path = Path(__file__).resolve().parents[2] / "train" / "callbacks" / "mlx_qat.py"
    spec = importlib.util.spec_from_file_location("fastvideo.train.callbacks.mlx_qat", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load mlx_qat from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fastvideo.train.callbacks.mlx_qat"] = mod
    spec.loader.exec_module(mod)
    return mod.MLXQuantizationAwareCallback


def main() -> int:
    from types import SimpleNamespace

    try:
        from fastvideo.train.callbacks.mlx_qat import MLXQuantizationAwareCallback
    except Exception as exc:  # noqa: BLE001 - missing optional train deps on Mac
        print(f"[arming] package import failed ({type(exc).__name__}: {exc}); "
              "loading mlx_qat via file stub", flush=True)
        MLXQuantizationAwareCallback = _import_mlx_qat_callback()

    ckpt, cfg = _resolve_checkpoint()
    transformer = _load_transformer(ckpt, cfg)
    n_params = sum(p.numel() for p in transformer.parameters())
    print(f"[arming] transformer params={n_params / 1e9:.2f}B layers={len(transformer.blocks)}", flush=True)

    method = SimpleNamespace(student=SimpleNamespace(transformer=transformer))
    callback = MLXQuantizationAwareCallback(group_size=64, bits=8, simulate_dtype="fp16")
    try:
        callback.on_train_start(method, iteration=0)
    except ValueError as exc:
        # Surface the silent-no-op failure mode explicitly.
        print(f"FAIL: mlx_qat arming raised: {exc}", flush=True)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: unexpected error during mlx_qat arming: {type(exc).__name__}: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        return 3

    n = len(callback.quantized_module_names)
    print(
        f"mlx_qat: fake-quantizing {n} weights (int8, group_size=64, simulate=torch.float16), "
        f"e.g. {callback.quantized_module_names[:3]}",
        flush=True,
    )
    if n < _MIN_EXPECTED_WEIGHTS:
        print(f"FAIL: expected >= {_MIN_EXPECTED_WEIGHTS} quantized weights, got {n}", flush=True)
        return 1
    # Spot-check: modules still look vanilla outside forwards (FSDP safety).
    sample_name = callback.quantized_module_names[0]
    sample = dict(transformer.named_modules())[sample_name]
    if not isinstance(sample.weight, type(next(transformer.parameters()))):
        # Parameter check
        pass
    if "weight" not in sample._parameters:
        print("FAIL: weight missing from _parameters outside forward (parametrization leak?)", flush=True)
        return 4
    print(f"PASS: mlx_qat armed on 5B with {n} weights (>= {_MIN_EXPECTED_WEIGHTS})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
