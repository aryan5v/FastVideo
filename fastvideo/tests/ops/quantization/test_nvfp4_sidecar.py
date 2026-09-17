# SPDX-License-Identifier: Apache-2.0
"""CPU-only round-trip tests for the compact NVFP4 checkpoint sidecar.

``convert_model_to_nvfp4`` registers its packed tensors with
``persistent=False``, so a saved checkpoint carries dense bf16 weights only and
the FP4 payload is rebuilt at every load. ``save_nvfp4_checkpoint`` /
``load_nvfp4_checkpoint`` write and restore that payload directly.

flashinfer, the FP4 GEMM and ``NVFP4QuantizeMethod.__init__``'s cuda allocation
are all stubbed, so this runs on any host. The conversion, the buffer
registration and the retention policy under test are the real ones.
"""
from __future__ import annotations

import json
import logging

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

import fastvideo.layers.quantization.nvfp4_config as nv
from fastvideo.layers.linear import ReplicatedLinear

_BLOCK = "minimax_h3.transformer_blocks.{idx}"
_IN_DIM = 8
_OUT_DIM = 4


@pytest.fixture(autouse=True)
def _stub_flashinfer(monkeypatch):
    """Patch the three flashinfer touch points the real path would use."""
    import types

    fake_sf_layout = types.SimpleNamespace(layout_128x4=None)
    monkeypatch.setattr(nv, "_require_flashinfer", lambda: (fake_sf_layout, None, None))
    monkeypatch.setattr(nv, "_nvfp4_quantize", _fake_quantize)

    def _init(self, layer_prefix: str = ""):
        self.weight_fp4 = None
        self.weight_scale = None
        self.x_global_sf = torch.tensor(1.0, dtype=torch.float32)
        self.layer_prefix = layer_prefix
        self._is_refine_only_layer = nv._is_ltx2_refine_only_prefix(layer_prefix)
        self._retain_original_weights = None

    monkeypatch.setattr(nv.NVFP4QuantizeMethod, "__init__", _init)


def _fake_quantize(weight, global_sf, sfLayout=None, do_shuffle=False):
    """Deterministic stand-in for ``nvfp4_quantize`` with the real shapes."""
    out_dim, in_dim = weight.shape[0], weight.shape[-1]
    packed = torch.arange(out_dim * ((in_dim + 1) // 2), dtype=torch.uint8).view(out_dim, (in_dim + 1) // 2)
    scales = torch.arange(out_dim * ((in_dim + 15) // 16), dtype=torch.uint8).view(out_dim, (in_dim + 15) // 16)
    return packed, scales


def _module() -> nn.Module:
    """An empty submodule, so FQNs can be assembled explicitly."""
    return nn.Module()


def _build_model(*, num_blocks: int = 2, seed: int = 0, quant_config=None) -> nn.Module:
    """A miniature H3-shaped DiT: ``minimax_h3.transformer_blocks.{i}.{attn.to_q,ff.fc_in}``."""
    config = quant_config if quant_config is not None else nv.NVFP4Config.for_minimax_h3()
    generator = torch.Generator().manual_seed(seed)
    root = _module()
    dit = _module()
    blocks = nn.ModuleList()
    for idx in range(num_blocks):
        block = _module()
        attn = _module()
        attn.to_q = ReplicatedLinear(_IN_DIM,
                                     _OUT_DIM,
                                     bias=False,
                                     quant_config=config,
                                     prefix=f"{_BLOCK.format(idx=idx)}.attn.to_q")
        block.attn = attn
        ff = _module()
        ff.fc_in = ReplicatedLinear(_IN_DIM,
                                    _OUT_DIM,
                                    bias=False,
                                    quant_config=config,
                                    prefix=f"{_BLOCK.format(idx=idx)}.ff.fc_in")
        block.ff = ff
        blocks.append(block)
    dit.transformer_blocks = blocks
    root.minimax_h3 = dit
    # ``create_weights`` allocates uninitialized storage; give it real values so
    # the global scale (and therefore _nvfp4_alpha) is meaningful.
    for _, param in root.named_parameters():
        param.data.copy_(torch.randn(param.shape, generator=generator))
    return root


def _buffers(model: nn.Module) -> dict[str, torch.Tensor]:
    return nv.nvfp4_sidecar_state_dict(model)


def _rewrite_sidecar(src, dst, *, metadata: dict | None = None, tensors: dict | None = None) -> str:
    """Copy a sidecar, optionally replacing its manifest or one tensor."""
    with safe_open(src, framework="pt", device="cpu") as handle:
        payload = {key: handle.get_tensor(key) for key in handle.keys()}
        manifest = json.loads(handle.metadata()[nv._NVFP4_SIDECAR_METADATA_KEY])
    if metadata is not None:
        manifest.update(metadata)
    if tensors is not None:
        payload.update(tensors)
    save_file(payload, str(dst), metadata={nv._NVFP4_SIDECAR_METADATA_KEY: json.dumps(manifest)})
    return str(dst)


def test_convert_purges_weights_and_sidecar_shrinks_the_payload(tmp_path) -> None:
    model = _build_model()
    nv.convert_model_to_nvfp4(model)
    # The always-FP4 layers are purged, so the sidecar is the only copy left.
    assert model.minimax_h3.transformer_blocks[0].attn.to_q.weight is None

    path = tmp_path / "nvfp4.safetensors"
    receipt = nv.save_nvfp4_checkpoint(model, path)
    assert path.exists()
    assert receipt["num_layers"] == 4
    assert receipt["num_tensors"] == 16  # 4 buffers x 4 layers
    assert receipt["quantized_bytes"] == sum(t.numel() * t.element_size() for t in _buffers(model).values())
    assert receipt["quantized_bytes"] < receipt["dense_bfloat16_bytes"]
    assert receipt["compression_ratio"] > 1.0

    manifest = nv.read_nvfp4_sidecar_metadata(path)
    assert manifest["format"] == nv._NVFP4_SIDECAR_FORMAT
    assert manifest["version"] == nv._NVFP4_SIDECAR_VERSION
    assert manifest["sf_layout"] == "layout_128x4"
    assert manifest["do_shuffle"] is False
    assert manifest["block_size"] == 16
    assert manifest["layers"][f"{_BLOCK.format(idx=1)}.ff.fc_in"] == [_OUT_DIM, _IN_DIM]
    assert manifest["quant_prefixes"][f"{_BLOCK.format(idx=0)}.attn.to_q"] == f"{_BLOCK.format(idx=0)}.attn.to_q"


def test_sidecar_state_dict_uses_module_fqn_keys_and_cpu_tensors() -> None:
    model = _build_model()
    nv.convert_model_to_nvfp4(model)
    state = _buffers(model)
    key = f"{_BLOCK.format(idx=0)}.attn.to_q::{nv._NVFP4_SIDECAR_BUFFERS[0]}"
    assert key in state
    assert state[key].device.type == "cpu"
    # Only the four registered buffers are serialized; nothing else leaks in.
    assert {name for _, name in (k.split("::") for k in state)} == set(nv._NVFP4_SIDECAR_BUFFERS)
    assert state[f"{_BLOCK.format(idx=0)}.attn.to_q::_nvfp4_weight"].dtype is torch.uint8


def test_load_restores_buffers_without_reconverting(tmp_path) -> None:
    """The restored tensors must be identical to a fresh conversion's."""
    source = _build_model(seed=1)
    nv.convert_model_to_nvfp4(source)
    expected = {key: value.clone() for key, value in _buffers(source).items()}
    path = tmp_path / "nvfp4.safetensors"
    nv.save_nvfp4_checkpoint(source, path)

    # A different model (different weights) restored purely from the sidecar.
    target = _build_model(seed=2)
    restored = nv.load_nvfp4_checkpoint(target, path)
    assert restored == 4
    actual = _buffers(target)
    assert set(actual) == set(expected)
    for key, value in expected.items():
        assert torch.equal(actual[key], value), key
        assert actual[key].dtype == value.dtype
    # The dense bf16 weights are gone under the default retention policy, so the
    # sidecar really is the only copy of the quantized weights.
    assert target.minimax_h3.transformer_blocks[0].attn.to_q.weight is None


def test_load_does_not_need_flashinfer(monkeypatch, tmp_path) -> None:
    """Serving a pre-quantized checkpoint must not require the FP4 kernels."""
    source = _build_model(seed=3)
    nv.convert_model_to_nvfp4(source)
    expected = {key: value.clone() for key, value in _buffers(source).items()}
    path = tmp_path / "nvfp4.safetensors"
    nv.save_nvfp4_checkpoint(source, path)

    def _boom():
        raise ImportError("NVFP4 quantization requires flashinfer.")

    monkeypatch.setattr(nv, "_require_flashinfer", _boom)
    target = _build_model(seed=4)
    assert nv.load_nvfp4_checkpoint(target, path) == 4
    assert all(torch.equal(_buffers(target)[key], value) for key, value in expected.items())


def test_load_into_a_model_whose_dense_weights_were_never_loaded(tmp_path) -> None:
    """The compact case: the checkpoint carries no bf16 weights at all."""
    source = _build_model(seed=5)
    nv.convert_model_to_nvfp4(source)
    expected = {key: value.clone() for key, value in _buffers(source).items()}
    path = tmp_path / "nvfp4.safetensors"
    nv.save_nvfp4_checkpoint(source, path)

    target = _build_model(seed=6)
    for module in target.modules():
        if getattr(module, "quant_method", None) is not None and hasattr(module, "weight"):
            module.register_parameter("weight", None)
    assert nv.load_nvfp4_checkpoint(target, path) == 4
    assert all(torch.equal(_buffers(target)[key], value) for key, value in expected.items())


def test_load_purges_dense_weights_unless_asked_not_to(tmp_path) -> None:
    source = _build_model(seed=7)
    nv.convert_model_to_nvfp4(source)
    path = tmp_path / "nvfp4.safetensors"
    nv.save_nvfp4_checkpoint(source, path)

    keep = _build_model(seed=8)
    nv.load_nvfp4_checkpoint(keep, path, purge_dense_weights=False)
    assert keep.minimax_h3.transformer_blocks[0].attn.to_q.weight is not None


def _sidecar_of(model: nn.Module, path) -> str:
    nv.convert_model_to_nvfp4(model)
    nv.save_nvfp4_checkpoint(model, path)
    return str(path)


def test_layer_set_mismatch_strict_and_lenient(tmp_path, caplog) -> None:
    source = _build_model(num_blocks=1, seed=10)
    nv.convert_model_to_nvfp4(source)
    path = tmp_path / "nvfp4.safetensors"
    nv.save_nvfp4_checkpoint(source, path)

    wider = _build_model(num_blocks=3, seed=13)
    bigger = _build_model(num_blocks=2, seed=11)
    with pytest.raises(ValueError, match="missing from the sidecar"):
        nv.load_nvfp4_checkpoint(bigger, path)

    with caplog.at_level(logging.WARNING):
        assert nv.load_nvfp4_checkpoint(bigger, path, strict=False) == 2
    assert any("does not match this model" in record.message for record in caplog.records)
    # The unmatched layer keeps whatever it had (nothing) rather than silently
    # becoming a dense layer with a quant_method attached.
    assert getattr(bigger.minimax_h3.transformer_blocks[1].attn.to_q, "_nvfp4_weight", None) is None

    # The other direction: a sidecar with layers this model does not have.
    smaller = _build_model(num_blocks=1, seed=12)
    with pytest.raises(ValueError, match="not in the model"):
        nv.load_nvfp4_checkpoint(smaller, _sidecar_of(wider, tmp_path / "wide.safetensors"))


def test_layout_and_version_mismatches_are_fatal(tmp_path) -> None:
    source = _build_model(seed=13)
    nv.convert_model_to_nvfp4(source)
    path = tmp_path / "nvfp4.safetensors"
    nv.save_nvfp4_checkpoint(source, path)

    target = _build_model(seed=14)
    bad_layout = _rewrite_sidecar(path, tmp_path / "layout.safetensors", metadata={"sf_layout": "layout_linear"})
    with pytest.raises(ValueError, match="sf_layout"):
        nv.load_nvfp4_checkpoint(target, bad_layout)
    # Never downgraded by strict=False: mis-read nibbles are silent corruption.
    with pytest.raises(ValueError, match="sf_layout"):
        nv.load_nvfp4_checkpoint(target, bad_layout, strict=False)

    bad_shuffle = _rewrite_sidecar(path, tmp_path / "shuffle.safetensors", metadata={"do_shuffle": True})
    with pytest.raises(ValueError, match="do_shuffle"):
        nv.load_nvfp4_checkpoint(target, bad_shuffle)

    bad_version = _rewrite_sidecar(path, tmp_path / "version.safetensors", metadata={"version": 99})
    with pytest.raises(ValueError, match="version"):
        nv.load_nvfp4_checkpoint(target, bad_version)

    bad_format = _rewrite_sidecar(path, tmp_path / "format.safetensors", metadata={"format": "something.else"})
    with pytest.raises(ValueError, match="format"):
        nv.load_nvfp4_checkpoint(target, bad_format)


def test_tensor_shape_mismatch_is_rejected(tmp_path) -> None:
    source = _build_model(seed=15)
    nv.convert_model_to_nvfp4(source)
    path = tmp_path / "nvfp4.safetensors"
    nv.save_nvfp4_checkpoint(source, path)

    key = f"{_BLOCK.format(idx=0)}.attn.to_q::_nvfp4_weight"
    bad = _rewrite_sidecar(path, tmp_path / "shape.safetensors", tensors={key: torch.zeros(3, 7, dtype=torch.uint8)})
    with pytest.raises(ValueError, match="expected"):
        nv.load_nvfp4_checkpoint(_build_model(seed=16), bad)


def test_block_scales_padded_to_the_128_row_tile_are_accepted(tmp_path) -> None:
    """``nvfp4_quantize`` returns block scales padded to the 128-row tile.

    ``_nvfp4_quantize`` narrows the packed weight back to the logical row count
    but not the scales, so a model whose output dim is not a multiple of 128
    can legitimately hold a scale tensor with more rows than the weight. A
    sidecar carrying that must load rather than fail shape validation.
    """
    source = _build_model(seed=19)
    nv.convert_model_to_nvfp4(source)
    path = tmp_path / "nvfp4.safetensors"
    nv.save_nvfp4_checkpoint(source, path)

    scale_key = f"{_BLOCK.format(idx=0)}.attn.to_q::_nvfp4_weight_scale"
    padded = _rewrite_sidecar(path,
                              tmp_path / "padded.safetensors",
                              tensors={scale_key: torch.zeros(128, (_IN_DIM + 15) // 16, dtype=torch.uint8)})
    assert nv.load_nvfp4_checkpoint(_build_model(seed=20), padded) == 4


def test_saving_an_unconverted_model_raises(tmp_path) -> None:
    model = _build_model(seed=17)
    with pytest.raises(RuntimeError, match="convert_model_to_nvfp4"):
        nv.save_nvfp4_checkpoint(model, tmp_path / "nvfp4.safetensors")


def test_a_model_with_no_nvfp4_layers_raises_with_the_prefix_hint(tmp_path) -> None:
    """The silent-dense failure mode must be reported, not ignored."""
    empty = _build_model(seed=18, quant_config=nv.NVFP4Config())
    with pytest.raises(RuntimeError, match="for_minimax_h3"):
        nv.save_nvfp4_checkpoint(empty, tmp_path / "nvfp4.safetensors")
    with pytest.raises(RuntimeError, match="for_minimax_h3"):
        nv.load_nvfp4_checkpoint(empty, tmp_path / "nvfp4.safetensors")


def test_sidecar_path_helper(tmp_path) -> None:
    assert nv.nvfp4_sidecar_path_for("/models/h3/transformer.safetensors") == "/models/h3/transformer.nvfp4.safetensors"
    assert nv.nvfp4_sidecar_path_for("/models/h3") == "/models/h3.nvfp4.safetensors"
    assert nv.nvfp4_sidecar_path_for(str(tmp_path)) == str(tmp_path / "nvfp4.safetensors")


def test_read_metadata_rejects_a_foreign_file(tmp_path) -> None:
    from safetensors.torch import save_file as _save

    plain = tmp_path / "plain.safetensors"
    _save({"w": torch.zeros(2, 2)}, str(plain))
    with pytest.raises(ValueError, match="not a FastVideo NVFP4 sidecar"):
        nv.read_nvfp4_sidecar_metadata(plain)
