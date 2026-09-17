# SPDX-License-Identifier: Apache-2.0
"""CPU-only round-trip tests for the INT8 affine and W4A16 checkpoint sidecars.

``convert_model_to_int8_affine`` / ``convert_model_to_w4a16`` register their
quantized tensors with ``persistent=False``, so a saved checkpoint carries
dense bf16 weights only and the low-bit payload is rebuilt at every load — by
re-running a conversion that starts from those dense weights. On the hardware
these lanes target (24-48 GB Ada parts, a 32 GB RTX 5090) H3's bf16 DiT does
not fit, so that rebuild is impossible and a pre-quantized checkpoint is the
only thing that can be served. ``save_*_checkpoint`` / ``load_*_checkpoint``
write and restore that payload directly.

No CUDA, no GPU, no flashinfer: INT8 affine and W4A16 are pure PyTorch, and
every test here runs on a laptop. The quantizers, the buffer registration, the
manifest and the load-time validation under test are the real ones.

The load-bearing assertions are the ones about *silent* corruption:
``torch.equal`` (not ``allclose``) for the round trip, an exact dtype check on
the codes, and a shape check per tensor — a mis-read or mis-unpacked code
buffer produces plausible-looking garbage with no error anywhere.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from typing import Any, NamedTuple

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.quantization import int8_affine_config as i8
from fastvideo.layers.quantization import w4a16_config as w4

IN_DIM = 128
OUT_DIM = 32
GROUP_SIZE = 64
_BLOCK = "minimax_h3.transformer_blocks.{idx}"
_Q = f"{_BLOCK.format(idx=0)}.attn.to_q"
_FF = f"{_BLOCK.format(idx=1)}.ff.fc_in"


class _Scheme(NamedTuple):
    """Everything the two lanes do not share, addressed by name."""

    name: str
    module: Any
    make_config: Callable[..., Any]
    buffers: tuple[str, ...]
    format_name: str
    metadata_key: str
    suffix: str
    dir_name: str
    convert: Callable[[nn.Module], None]
    state_dict: Callable[[nn.Module], dict[str, torch.Tensor]]
    save: Callable[..., dict[str, Any]]
    load: Callable[..., int]
    read_metadata: Callable[[Any], dict[str, Any]]
    path_for: Callable[[Any], str]
    wiring_hint: str


INT8 = _Scheme(
    name="int8_affine",
    module=i8,
    make_config=i8.INT8AffineConfig,
    buffers=i8._INT8_AFFINE_SIDECAR_BUFFERS,
    format_name=i8._INT8_AFFINE_SIDECAR_FORMAT,
    metadata_key=i8._INT8_AFFINE_SIDECAR_METADATA_KEY,
    suffix=i8.INT8_AFFINE_SIDECAR_SUFFIX,
    dir_name=i8.INT8_AFFINE_DIR_SIDECAR_NAME,
    convert=i8.convert_model_to_int8_affine,
    state_dict=i8.int8_affine_sidecar_state_dict,
    save=i8.save_int8_affine_checkpoint,
    load=i8.load_int8_affine_checkpoint,
    read_metadata=i8.read_int8_affine_sidecar_metadata,
    path_for=i8.int8_affine_sidecar_path_for,
    wiring_hint="for_minimax_h3",
)

W4A16 = _Scheme(
    name="w4a16",
    module=w4,
    make_config=w4.W4A16Config,
    buffers=w4._W4A16_SIDECAR_BUFFERS,
    format_name=w4._W4A16_SIDECAR_FORMAT,
    metadata_key=w4._W4A16_SIDECAR_METADATA_KEY,
    suffix=w4.W4A16_SIDECAR_SUFFIX,
    dir_name=w4.W4A16_DIR_SIDECAR_NAME,
    convert=w4.convert_model_to_w4a16,
    state_dict=w4.w4a16_sidecar_state_dict,
    save=w4.save_w4a16_checkpoint,
    load=w4.load_w4a16_checkpoint,
    read_metadata=w4.read_w4a16_sidecar_metadata,
    path_for=w4.w4a16_sidecar_path_for,
    wiring_hint="for_minimax_h3",
)

_ALL = (INT8, W4A16)
_by_name = pytest.mark.parametrize("scheme", _ALL, ids=[s.name for s in _ALL])




def _build(scheme: _Scheme, *, num_blocks: int = 2, seed: int = 0, config=None) -> nn.Module:
    """A miniature H3-shaped DiT: ``minimax_h3.transformer_blocks.{i}.{attn.to_q,ff.fc_in}``."""
    cfg = config if config is not None else scheme.make_config()
    generator = torch.Generator().manual_seed(seed)
    root = nn.Module()
    dit = nn.Module()
    blocks = nn.ModuleList()
    for idx in range(num_blocks):
        block = nn.Module()
        attn = nn.Module()
        attn.to_q = ReplicatedLinear(IN_DIM,
                                     OUT_DIM,
                                     bias=False,
                                     quant_config=cfg,
                                     prefix=f"{_BLOCK.format(idx=idx)}.attn.to_q")
        ff = nn.Module()
        ff.fc_in = ReplicatedLinear(IN_DIM,
                                    OUT_DIM,
                                    bias=False,
                                    quant_config=cfg,
                                    prefix=f"{_BLOCK.format(idx=idx)}.ff.fc_in")
        block.attn = attn
        block.ff = ff
        blocks.append(block)
    dit.transformer_blocks = blocks
    root.minimax_h3 = dit
    for _, param in root.named_parameters():
        param.data.copy_(torch.randn(param.shape, generator=generator))
    return root


def _tagged(model: nn.Module) -> dict[str, nn.Module]:
    return {fqn: mod for fqn, mod in model.named_modules() if getattr(mod, "quant_method", None) is not None}


def _rewrite_sidecar(scheme: _Scheme, src, dst, *, metadata: dict | None = None, tensors: dict | None = None) -> str:
    """Copy a sidecar, optionally replacing its manifest or some tensors."""
    with safe_open(src, framework="pt", device="cpu") as handle:
        payload = {key: handle.get_tensor(key) for key in handle.keys()}
        manifest = json.loads(handle.metadata()[scheme.metadata_key])
    if metadata is not None:
        manifest.update(metadata)
    if tensors is not None:
        payload.update(tensors)
    save_file(payload, str(dst), metadata={scheme.metadata_key: json.dumps(manifest)})
    return str(dst)


def _sidecar_of(scheme: _Scheme, model: nn.Module, path) -> str:
    scheme.convert(model)
    scheme.save(model, path)
    return str(path)




@_by_name
def test_state_dict_does_not_carry_the_quantized_buffers(scheme: _Scheme) -> None:
    """The whole reason for the sidecar: the buffers are non-persistent.

    A plain ``state_dict()`` is the dense bf16 weights and nothing else, so a
    checkpoint written the ordinary way cannot be served on a host that cannot
    hold the dense weights and re-convert.
    """
    model = _build(scheme)
    scheme.convert(model)

    keys = list(model.state_dict())
    assert keys, "the dense weights are persistent, so this must not be empty"
    for key in keys:
        for buffer_name in scheme.buffers:
            assert buffer_name not in key, f"{key} leaked a non-persistent buffer"
    assert set(keys) == {f"{fqn}.weight" for fqn in _tagged(model)}
    for fqn, mod in _tagged(model).items():
        assert getattr(mod, scheme.buffers[0]) is not None, fqn
    assert len(scheme.state_dict(model)) == 4 * len(scheme.buffers)


@_by_name
def test_sidecar_state_dict_uses_module_fqn_keys_and_cpu_tensors(scheme: _Scheme) -> None:
    model = _build(scheme)
    scheme.convert(model)
    state = scheme.state_dict(model)

    assert f"{_Q}::{scheme.buffers[0]}" in state
    assert all(value.device.type == "cpu" for value in state.values())
    assert {key.split("::", 1)[1] for key in state} == set(scheme.buffers)
    assert state[f"{_Q}::{scheme.buffers[0]}"].dtype is torch.uint8
    for name in scheme.buffers[1:]:
        assert state[f"{_Q}::{name}"].dtype is torch.float32


def test_int8_codes_are_uint8_because_they_exceed_the_int8_range() -> None:
    """Affine bits=8 codes span [0, 255]; int8 storage would wrap 255 to -1.

    Pinned on a crafted weight rather than a random one so the assertion is
    about the scheme, not about a lucky seed.
    """
    model = _build(INT8)
    linear = _tagged(model)[_Q]
    weight = torch.full((OUT_DIM, IN_DIM), -1.0)
    weight[:, 0] = 3.0  # one large positive value per row
    linear.weight.data.copy_(weight)
    INT8.convert(model)

    codes = linear._int8_affine_codes
    assert codes.dtype is torch.uint8
    assert codes.max().item() == 255
    as_int8 = codes.to(torch.int8)
    assert as_int8.min().item() == -1
    assert not torch.equal(as_int8.to(torch.int64), codes.to(torch.int64))




@_by_name
def test_save_then_load_into_a_fresh_module_is_bit_identical(scheme: _Scheme, tmp_path) -> None:
    source = _build(scheme, seed=1)
    scheme.convert(source)
    expected = {key: value.clone() for key, value in scheme.state_dict(source).items()}

    path = tmp_path / "sidecar.safetensors"
    scheme.save(source, path)

    target = _build(scheme, seed=2)
    assert scheme.load(target, path) == 4
    actual = scheme.state_dict(target)

    assert set(actual) == set(expected)
    for key, value in expected.items():
        assert actual[key].dtype == value.dtype, key
        assert torch.equal(actual[key], value), f"{key} is not bit-identical (allclose would hide this)"
    assert not torch.equal(target.minimax_h3.transformer_blocks[0].attn.to_q.weight,
                           source.minimax_h3.transformer_blocks[0].attn.to_q.weight)


@_by_name
def test_loaded_buffers_reproduce_the_source_forward(scheme: _Scheme, tmp_path) -> None:
    """A loaded layer must compute what the converted source layer computes.

    This is the end-to-end payoff: for W4A16 it only holds if the load also
    restored ``_w4a16_weight_shape``, which is a plain attribute and not a
    buffer, so nothing else would carry it across.
    """
    source = _build(scheme, seed=3)
    scheme.convert(source)
    path = tmp_path / "sidecar.safetensors"
    scheme.save(source, path)

    target = _build(scheme, seed=4)
    scheme.load(target, path)

    x = torch.randn(2, IN_DIM, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        want, _ = _tagged(source)[_Q](x)
        got, _ = _tagged(target)[_Q](x)
    assert torch.equal(got, want)


def test_w4a16_load_restores_the_logical_weight_shape_attribute(tmp_path) -> None:
    """``_w4a16_weight_shape`` is not a buffer, so the manifest must carry it."""
    source = _build(W4A16, seed=6)
    W4A16.convert(source)
    path = tmp_path / "w4a16.safetensors"
    W4A16.save(source, path)

    target = _build(W4A16, seed=7)
    linear = _tagged(target)[_Q]
    assert getattr(linear, "_w4a16_weight_shape", None) is None
    W4A16.load(target, path)
    assert tuple(linear._w4a16_weight_shape) == (OUT_DIM, IN_DIM)
    assert linear._w4a16_codes.shape == (OUT_DIM, IN_DIM // 2)


@_by_name
def test_save_and_load_with_no_dense_weights_at_all(scheme: _Scheme, tmp_path) -> None:
    """The target-hardware case: the bf16 weights are never present.

    With ``retain_original_weight=False`` the converted model keeps no dense
    copy, and on a card that cannot hold them the loading model has none
    either (``weight`` is ``None``). Save and load must both work from the
    quantized buffers alone — this is the path that exercises the manifest's
    per-layer ``weight_shape`` for the shape it cannot read off a weight.
    """
    config = scheme.make_config(retain_original_weight=False)
    source = _build(scheme, seed=34, config=config)
    scheme.convert(source)
    assert _tagged(source)[_Q].weight is None

    path = tmp_path / "sidecar.safetensors"
    receipt = scheme.save(source, path)
    assert receipt["num_layers"] == 4
    expected = {key: value.clone() for key, value in scheme.state_dict(source).items()}

    target = _build(scheme, seed=35, config=scheme.make_config(retain_original_weight=False))
    for mod in _tagged(target).values():
        mod.register_parameter("weight", None)
    assert scheme.load(target, path) == 4

    actual = scheme.state_dict(target)
    assert set(actual) == set(expected)
    assert all(torch.equal(actual[key], value) for key, value in expected.items())
    x = torch.randn(2, IN_DIM, generator=torch.Generator().manual_seed(36))
    with torch.no_grad():
        assert torch.equal(_tagged(target)[_Q](x)[0], _tagged(source)[_Q](x)[0])


def test_w4a16_bits_8_sidecar_uses_the_unpacked_code_shape(tmp_path) -> None:
    """``bits=8`` stores one code per byte, so the code shape is the weight shape.

    The manifest and the load-time shape check both have to follow ``bits``
    rather than assume the 4-bit packed layout.
    """
    config = W4A16.make_config(bits=8)
    source = _build(W4A16, seed=42, config=config)
    W4A16.convert(source)
    assert _tagged(source)[_Q]._w4a16_codes.shape == (OUT_DIM, IN_DIM)

    path = tmp_path / "w4a16_bits8.safetensors"
    W4A16.save(source, path)
    manifest = W4A16.read_metadata(path)
    assert manifest["bits"] == 8
    assert manifest["layers"][_Q]["tensors"][W4A16.buffers[0]] == [OUT_DIM, IN_DIM]

    target = _build(W4A16, seed=43, config=W4A16.make_config(bits=8))
    assert W4A16.load(target, path) == 4
    assert torch.equal(W4A16.state_dict(target)[f"{_Q}::{W4A16.buffers[0]}"],
                       W4A16.state_dict(source)[f"{_Q}::{W4A16.buffers[0]}"])
    x = torch.randn(2, IN_DIM, generator=torch.Generator().manual_seed(44))
    with torch.no_grad():
        assert torch.equal(_tagged(target)[_Q](x)[0], _tagged(source)[_Q](x)[0])


def test_w4a16_codes_preserve_the_low_nibble_first_packing(tmp_path) -> None:
    """The packed bytes must round-trip the packer's nibble order exactly.

    The convention is the module's own (``_pack_4bit``): the low nibble is the
    lower K index. Re-deriving the expected byte from the source layer's own
    quantizer output is what makes this a check of the *serialized* bytes
    rather than of the packing function.
    """
    source = _build(W4A16, seed=8)
    W4A16.convert(source)
    linear = _tagged(source)[_Q]
    codes = linear._w4a16_codes
    weight = linear.weight.detach().float()
    unpacked = w4._unpack_4bit(codes)
    expected = w4._pack_4bit(unpacked)
    assert torch.equal(codes, expected)
    assert codes.shape == (OUT_DIM, IN_DIM // 2)
    assert unpacked.shape == (OUT_DIM, IN_DIM)
    raw, _, _ = w4.w4a16_quantize(weight, group_size=GROUP_SIZE, bits=4)
    assert torch.equal(unpacked, w4._unpack_4bit(raw))

    path = tmp_path / "w4a16.safetensors"
    W4A16.save(source, path)
    target = _build(W4A16, seed=9)
    W4A16.load(target, path)
    assert torch.equal(_tagged(target)[_Q]._w4a16_codes, codes)




@_by_name
def test_manifest_round_trips_scheme_and_layer_inventory(scheme: _Scheme, tmp_path) -> None:
    model = _build(scheme)
    scheme.convert(model)
    path = tmp_path / "sidecar.safetensors"
    receipt = scheme.save(model, path)

    manifest = scheme.read_metadata(path)
    assert manifest["format"] == scheme.format_name
    assert manifest["version"] == 1
    assert manifest["group_size"] == GROUP_SIZE
    assert manifest["bits"] == (8 if scheme is INT8 else 4)
    assert manifest["num_layers"] == 4
    assert set(manifest["layers"]) == {_Q, _FF, f"{_BLOCK.format(idx=0)}.ff.fc_in",
                                       f"{_BLOCK.format(idx=1)}.attn.to_q"}
    assert manifest["quant_prefixes"][_Q] == _Q
    assert manifest["model_class"] == "Module"

    entry = manifest["layers"][_Q]
    assert entry["weight_shape"] == [OUT_DIM, IN_DIM]
    assert entry["group_size"] == GROUP_SIZE
    assert entry["bits"] == (8 if scheme is INT8 else 4)
    assert set(entry["tensors"]) == set(scheme.buffers)

    with safe_open(path, framework="pt", device="cpu") as handle:
        for fqn, layer in manifest["layers"].items():
            for name, shape in layer["tensors"].items():
                assert list(handle.get_tensor(f"{fqn}::{name}").shape) == shape

    assert receipt["num_layers"] == 4
    assert receipt["num_tensors"] == 4 * len(scheme.buffers)
    assert receipt["quantized_bytes"] == sum(t.numel() * t.element_size() for t in scheme.state_dict(model).values())
    assert receipt["quantized_bytes"] < receipt["dense_bfloat16_bytes"]
    assert receipt["compression_ratio"] > 1.0


@_by_name
def test_save_logs_a_receipt_with_the_module_count_and_bytes(scheme: _Scheme, tmp_path, caplog) -> None:
    model = _build(scheme)
    scheme.convert(model)
    with caplog.at_level(logging.INFO):
        receipt = scheme.save(model, tmp_path / "sidecar.safetensors")
    message = "\n".join(record.message for record in caplog.records)
    assert "4 quantized modules" in message
    assert str(receipt["quantized_bytes"]) in message
    assert "dense bf16" in message


@_by_name
def test_extra_metadata_is_merged_into_the_manifest(scheme: _Scheme, tmp_path) -> None:
    model = _build(scheme)
    scheme.convert(model)
    path = tmp_path / "sidecar.safetensors"
    scheme.save(model, path, extra_metadata={"source_checkpoint": "h3-bf16"})
    assert scheme.read_metadata(path)["source_checkpoint"] == "h3-bf16"




@_by_name
def test_wrong_code_dtype_is_rejected_at_load(scheme: _Scheme, tmp_path) -> None:
    """Codes cast to int8 must fail, not silently dequantize to garbage."""
    source = _build(scheme, seed=10)
    scheme.convert(source)
    path = tmp_path / "sidecar.safetensors"
    scheme.save(source, path)

    key = f"{_Q}::{scheme.buffers[0]}"
    with safe_open(path, framework="pt", device="cpu") as handle:
        bad_codes = handle.get_tensor(key).to(torch.int8)
    bad = _rewrite_sidecar(scheme, path, tmp_path / "int8codes.safetensors", tensors={key: bad_codes})

    with pytest.raises(ValueError, match="dtype"):
        scheme.load(_build(scheme, seed=11), bad)
    with pytest.raises(ValueError, match="dtype"):
        scheme.load(_build(scheme, seed=11), bad, strict=False)


@_by_name
def test_wrong_float_dtype_is_rejected_at_load(scheme: _Scheme, tmp_path) -> None:
    """A bf16 scale store is not a bit-exact restore of the fp32 constants."""
    source = _build(scheme, seed=12)
    scheme.convert(source)
    path = tmp_path / "sidecar.safetensors"
    scheme.save(source, path)

    key = f"{_Q}::{scheme.buffers[1]}"
    with safe_open(path, framework="pt", device="cpu") as handle:
        bad_scales = handle.get_tensor(key).to(torch.bfloat16)
    bad = _rewrite_sidecar(scheme, path, tmp_path / "bf16scales.safetensors", tensors={key: bad_scales})
    with pytest.raises(ValueError, match="dtype"):
        scheme.load(_build(scheme, seed=13), bad)


@_by_name
def test_wrong_shape_is_rejected_at_load(scheme: _Scheme, tmp_path) -> None:
    source = _build(scheme, seed=14)
    scheme.convert(source)
    path = tmp_path / "sidecar.safetensors"
    scheme.save(source, path)

    key = f"{_Q}::{scheme.buffers[0]}"
    bad = _rewrite_sidecar(scheme,
                           path,
                           tmp_path / "shape.safetensors",
                           tensors={key: torch.zeros(OUT_DIM, IN_DIM * 2, dtype=torch.uint8)})
    with pytest.raises(ValueError, match="expected"):
        scheme.load(_build(scheme, seed=15), bad)
    with pytest.raises(ValueError, match="expected"):
        scheme.load(_build(scheme, seed=15), bad, strict=False)


@_by_name
def test_scale_shape_mismatch_is_rejected_at_load(scheme: _Scheme, tmp_path) -> None:
    """A regrouped scales tensor would regroup every code in the row."""
    source = _build(scheme, seed=16)
    scheme.convert(source)
    path = tmp_path / "sidecar.safetensors"
    scheme.save(source, path)

    key = f"{_Q}::{scheme.buffers[1]}"
    bad = _rewrite_sidecar(scheme,
                           path,
                           tmp_path / "scales.safetensors",
                           tensors={key: torch.zeros(OUT_DIM, IN_DIM, dtype=torch.float32)})
    with pytest.raises(ValueError, match="expected"):
        scheme.load(_build(scheme, seed=17), bad)


@_by_name
def test_group_size_and_bits_mismatches_are_fatal(scheme: _Scheme, tmp_path) -> None:
    """A different scheme means different dequantize arithmetic over the bytes."""
    source = _build(scheme, seed=18)
    scheme.convert(source)
    path = tmp_path / "sidecar.safetensors"
    scheme.save(source, path)

    other_group = scheme.make_config(group_size=GROUP_SIZE * 2)
    with pytest.raises(ValueError, match="group_size"):
        scheme.load(_build(scheme, seed=19, config=other_group), path)
    with pytest.raises(ValueError, match="group_size"):
        scheme.load(_build(scheme, seed=19, config=other_group), path, strict=False)

    other_bits = scheme.make_config(bits=4 if scheme is INT8 else 8)
    with pytest.raises(ValueError, match="bits"):
        scheme.load(_build(scheme, seed=19, config=other_bits), path)


@_by_name
def test_weight_shape_that_disagrees_with_the_model_is_fatal(scheme: _Scheme, tmp_path) -> None:
    """A sidecar describing a differently-shaped layer must not load into this one."""
    source = _build(scheme, seed=20)
    scheme.convert(source)
    path = tmp_path / "sidecar.safetensors"
    scheme.save(source, path)

    layers = dict(scheme.read_metadata(path)["layers"])
    layers[_Q] = {**layers[_Q], "weight_shape": [OUT_DIM, IN_DIM * 2]}
    reshaped = _rewrite_sidecar(scheme, path, tmp_path / "ws.safetensors", metadata={"layers": layers})
    with pytest.raises(ValueError, match="shape"):
        scheme.load(_build(scheme, seed=21), reshaped)


@_by_name
def test_malformed_layer_entry_is_reported_not_a_key_error(scheme: _Scheme, tmp_path) -> None:
    """A hand-edited manifest must fail with a message naming the layer."""
    source = _build(scheme, seed=37)
    scheme.convert(source)
    path = tmp_path / "sidecar.safetensors"
    scheme.save(source, path)

    layers = dict(scheme.read_metadata(path)["layers"])
    layers[_Q] = {key: value for key, value in layers[_Q].items() if key != "group_size"}
    malformed = _rewrite_sidecar(scheme, path, tmp_path / "malformed.safetensors", metadata={"layers": layers})
    with pytest.raises(ValueError, match="malformed"):
        scheme.load(_build(scheme, seed=38), malformed)


@_by_name
def test_format_and_version_mismatches_are_fatal(scheme: _Scheme, tmp_path) -> None:
    source = _build(scheme, seed=22)
    scheme.convert(source)
    path = tmp_path / "sidecar.safetensors"
    scheme.save(source, path)

    bad_format = _rewrite_sidecar(scheme, path, tmp_path / "format.safetensors", metadata={"format": "other"})
    with pytest.raises(ValueError, match="format"):
        scheme.load(_build(scheme, seed=23), bad_format)

    bad_version = _rewrite_sidecar(scheme, path, tmp_path / "version.safetensors", metadata={"version": 99})
    with pytest.raises(ValueError, match="version"):
        scheme.load(_build(scheme, seed=23), bad_version)


@_by_name
def test_incomplete_layer_entry_is_rejected(scheme: _Scheme, tmp_path) -> None:
    """A layer missing its code buffer must not silently become dense."""
    source = _build(scheme, seed=24)
    scheme.convert(source)
    path = tmp_path / "sidecar.safetensors"
    scheme.save(source, path)

    with safe_open(path, framework="pt", device="cpu") as handle:
        payload = {key: handle.get_tensor(key) for key in handle.keys()}
        manifest = json.loads(handle.metadata()[scheme.metadata_key])
    del payload[f"{_Q}::{scheme.buffers[0]}"]
    stripped = tmp_path / "stripped.safetensors"
    save_file(payload, str(stripped), metadata={scheme.metadata_key: json.dumps(manifest)})

    with pytest.raises(ValueError, match="incomplete"):
        scheme.load(_build(scheme, seed=25), stripped)


@_by_name
def test_layer_set_mismatch_strict_and_lenient(scheme: _Scheme, tmp_path, caplog) -> None:
    source = _build(scheme, num_blocks=1, seed=26)
    path = tmp_path / "one.safetensors"
    scheme.convert(source)
    scheme.save(source, path)

    bigger = _build(scheme, num_blocks=2, seed=27)
    with pytest.raises(ValueError, match="missing from the sidecar"):
        scheme.load(bigger, path)

    with caplog.at_level(logging.WARNING):
        assert scheme.load(bigger, path, strict=False) == 2
    assert any("does not match this model" in record.message for record in caplog.records)
    assert getattr(bigger.minimax_h3.transformer_blocks[1].attn.to_q, scheme.buffers[0], None) is None

    wider = _build(scheme, num_blocks=3, seed=28)
    with pytest.raises(ValueError, match="not in the model"):
        scheme.load(_build(scheme, num_blocks=1, seed=29), _sidecar_of(scheme, wider,
                                                                       tmp_path / "wide.safetensors"))


@_by_name
def test_read_metadata_rejects_a_foreign_file(scheme: _Scheme, tmp_path) -> None:
    plain = tmp_path / "plain.safetensors"
    save_file({"w": torch.zeros(2, 2)}, str(plain))
    with pytest.raises(ValueError, match="not a FastVideo"):
        scheme.read_metadata(plain)


@_by_name
def test_saving_an_unconverted_model_raises(scheme: _Scheme, tmp_path) -> None:
    model = _build(scheme, seed=30)
    with pytest.raises(RuntimeError, match="convert_model_to"):
        scheme.save(model, tmp_path / "sidecar.safetensors")


@_by_name
def test_a_model_with_no_tagged_layers_raises_with_the_wiring_hint(scheme: _Scheme, tmp_path) -> None:
    """The silent-dense failure mode must be reported, not ignored."""
    empty = _build(scheme, seed=31, config=scheme.make_config(target_layers=["nothing.matches.this"]))
    with pytest.raises(RuntimeError, match=scheme.wiring_hint):
        scheme.save(empty, tmp_path / "sidecar.safetensors")
    with pytest.raises(RuntimeError, match=scheme.wiring_hint):
        scheme.load(empty, tmp_path / "sidecar.safetensors")




@_by_name
def test_load_needs_no_gpu_and_no_flashinfer(scheme: _Scheme, tmp_path, monkeypatch) -> None:
    """Serving a pre-quantized checkpoint on the target host must not import kernels."""
    source = _build(scheme, seed=32)
    scheme.convert(source)
    expected = {key: value.clone() for key, value in scheme.state_dict(source).items()}
    path = tmp_path / "sidecar.safetensors"
    scheme.save(source, path)

    assert "flashinfer" not in sys.modules, "neither lane may pull in flashinfer"
    target = _build(scheme, seed=33)
    assert scheme.load(target, path) == 4
    assert "flashinfer" not in sys.modules
    assert all(torch.equal(scheme.state_dict(target)[key], value) for key, value in expected.items())


def test_sidecar_path_helpers(tmp_path) -> None:
    for scheme in _ALL:
        assert scheme.path_for("/models/h3/transformer.safetensors") == f"/models/h3/transformer{scheme.suffix}"
        assert scheme.path_for("/models/h3") == f"/models/h3{scheme.suffix}"
        assert scheme.path_for(str(tmp_path)) == str(tmp_path / scheme.dir_name)
