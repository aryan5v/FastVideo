# SPDX-License-Identifier: Apache-2.0
"""Teacher/export loading preserves explicit H3 arithmetic and VSA structure."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from fastvideo.attention.selector import (
    _active_component_attention_backend_scope,
    _component_attention_backend_scope,
)
from fastvideo.models.loader import component_loader
from fastvideo.platforms import AttentionBackendEnum as Backend


@pytest.mark.parametrize("requested,expected", [
    (Backend.TORCH_SDPA, Backend.TORCH_SDPA),
    (Backend.FLASH_ATTN, Backend.FLASH_ATTN),
    (Backend.VIDEO_SPARSE_ATTN_H3, Backend.VIDEO_SPARSE_ATTN_H3),
    (Backend.ATTN_QAT_TRAIN, None),
    (Backend.ATTN_QAT_INFER, None),
    (None, None),
])
@pytest.mark.parametrize("fail_load", [False, True])
def test_teacher_transformer_backend(monkeypatch, tmp_path, requested, expected, fail_load):
    (tmp_path / "model.safetensors").touch()
    original_config = SimpleNamespace(
        arch_config=SimpleNamespace(), quant_config="nvfp4_qat",
        update_model_arch=lambda config: None,
    )
    args = SimpleNamespace(
        override_transformer_cls_name=None, model_paths={},
        pipeline_config=SimpleNamespace(dit_config=original_config, dit_precision="bf16"),
        _loading_teacher_critic_model=True,
        hsdp_shard_dim=1, hsdp_replicate_dim=1,
        dit_cpu_offload=False, pin_cpu_memory=False, use_fsdp_inference=False,
        training_mode=False, enable_torch_compile=False, torch_compile_kwargs={},
        inference_torch_compile=False, VSA_tile_size=None, inference_mode=False,
    )
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "ATTN_QAT_TRAIN")
    monkeypatch.setattr(component_loader, "get_diffusers_config", lambda **kwargs: {"_class_name": "FakeH3"})
    monkeypatch.setattr(component_loader.ModelRegistry, "resolve_model_cls", lambda name: (torch.nn.Linear, None))
    monkeypatch.setattr(component_loader, "get_local_torch_device", lambda: torch.device("cpu"))

    def construct(**kwargs):
        scope = _active_component_attention_backend_scope()
        config = kwargs["init_params"]["config"]
        assert scope is not None and scope.backend == expected and not scope.consult_env
        assert config._resolved_attention_backend == expected
        assert config.quant_config is None
        assert kwargs["strict"]
        if fail_load:
            raise RuntimeError("deliberate load failure")
        return torch.nn.Linear(1, 1, dtype=torch.bfloat16)

    monkeypatch.setattr(component_loader, "maybe_load_fsdp_model", construct)
    context = _component_attention_backend_scope(requested, component="transformer") if requested else nullcontext()
    with context:
        previous = _active_component_attention_backend_scope()
        if fail_load:
            with pytest.raises(RuntimeError, match="deliberate load failure"):
                component_loader.TransformerLoader().load(str(tmp_path), args)
        else:
            component_loader.TransformerLoader().load(str(tmp_path), args)
        assert _active_component_attention_backend_scope() is previous
    assert _active_component_attention_backend_scope() is None
    assert original_config.quant_config == "nvfp4_qat"
