# SPDX-License-Identifier: Apache-2.0
"""Run a bf16 MiniMax-H3 DiT and its packed NVFP4 export in one process.

``NVFP4Swap`` registers the export's buffers (``<module>::_nvfp4_weight`` etc., the layout
``convert_minimax_h3_modelopt_nvfp4_dit.py`` writes and ``load_minimax_h3_nvfp4_dit_export``
reads) next to each bf16 linear's weight and swaps the layer's ``quant_method`` between the
original bf16 method and the runtime ``NVFP4QuantizeMethod``. Inside ``enabled()`` every exported
linear runs the deployment NVFP4 path (``mm_fp4``, the export's static ``_nvfp4_input_global_sf``
or the unit scale); outside it the model is untouched bf16. Teacher-forced comparisons can then
replay one forward under both numerics with identical inputs and forward context.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch

SEP = "::"
BUFFERS = ("_nvfp4_weight", "_nvfp4_weight_scale", "_nvfp4_alpha", "_weight_global_sf")
INPUT_SF = "_nvfp4_input_global_sf"


def _resolve(modules: dict[str, torch.nn.Module], prefix: str) -> tuple[str, torch.nn.Module]:
    if prefix in modules:
        return prefix, modules[prefix]
    matches = [name for name in modules if name.endswith("." + prefix)]
    if len(matches) != 1:
        raise ValueError(f"export layer {prefix!r} matches {len(matches)} modules")
    return matches[0], modules[matches[0]]


class NVFP4Swap:

    def __init__(self, model: torch.nn.Module, export_path: str, device: torch.device | str) -> None:
        from safetensors import safe_open

        from fastvideo.layers.quantization.nvfp4_config import NVFP4QuantizeMethod

        modules = dict(model.named_modules())
        groups: dict[str, dict[str, str]] = {}
        with safe_open(export_path, framework="pt", device="cpu") as reader:
            for key in reader.keys():  # noqa: SIM118
                prefix, buffer = key.split(SEP, 1)
                groups.setdefault(prefix, {})[buffer] = key
            self.layers: list[tuple[str, torch.nn.Module, object, object]] = []
            for prefix, buffers in sorted(groups.items()):
                missing = [b for b in BUFFERS if b not in buffers]
                if missing:
                    raise ValueError(f"{prefix}: export lacks {missing}")
                name, module = _resolve(modules, prefix)
                if getattr(module, "weight", None) is None or not hasattr(module, "quant_method"):
                    raise ValueError(f"{name} is not a bf16 linear with a quant_method")
                for buffer in BUFFERS + (INPUT_SF, ):
                    if buffer in buffers:
                        module.register_buffer(buffer, reader.get_tensor(buffers[buffer]).to(device), persistent=False)
                method = NVFP4QuantizeMethod(layer_prefix=name)
                self.layers.append((name, module, module.quant_method, method))
        self.static_scales = sum(1 for _, m, _, _ in self.layers if hasattr(m, INPUT_SF))

    def __len__(self) -> int:
        return len(self.layers)

    def set(self, on: bool, only: set[str] | None = None) -> None:
        for name, module, bf16_method, nvfp4_method in self.layers:
            use = on and (only is None or name in only)
            module.quant_method = nvfp4_method if use else bf16_method

    @contextlib.contextmanager
    def enabled(self, only: set[str] | None = None) -> Iterator[None]:
        self.set(True, only)
        try:
            yield
        finally:
            self.set(False)
