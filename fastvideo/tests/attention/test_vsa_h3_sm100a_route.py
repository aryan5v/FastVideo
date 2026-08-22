# SPDX-License-Identifier: Apache-2.0
"""Routing and transport invariants for VSA-H3's tile-64 sm_100a path.

The CUDA kernel assigns a pair of adjacent query tiles to every CTA. H3's
logical tile count is frequently odd, so the no-grad opt-in route appends one
internal zero-valid partner. These CPU tests pin that the partner is visible
only to sm100a transport: trained score/top-k/gate semantics, metadata,
Triton fallbacks, gradients, and returned sequences all remain logical.
"""

import pytest
import torch

import fastvideo.attention.backends.video_sparse_attn_h3 as vsa_h3
from fastvideo.attention.backends.video_sparse_attn_h3 import (VSA_SM100A_ENV, MiniMaxH3VSAImpl,
                                                               MiniMaxH3VSAMetadataBuilder, _sm100a_unavailable_reason)

# Two prefix segments produce three 64-token tiles; the (4,4,8)-token video
# grid produces two more. Five logical tiles exercises the sm100a pair pad.
_SPEC = dict(raw_latent_shape=(4, 8, 16), patch_size=(1, 2, 2), prefix_segments=(70, 30))
_HEADS, _DIM = 2, 128
_CPU = torch.device("cpu")


def _build_meta(
    device: torch.device = _CPU,
    *,
    prefix_segments: tuple[int, ...] = _SPEC["prefix_segments"],
    sparsity: float = 0.0,
    tile_size: int = 64,
):
    return MiniMaxH3VSAMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=_SPEC["raw_latent_shape"],
        patch_size=_SPEC["patch_size"],
        VSA_sparsity=sparsity,
        prefix_segments=prefix_segments,
        device=device,
        tile_size=tile_size,
    )


def _impl():
    return MiniMaxH3VSAImpl(num_heads=_HEADS, head_size=_DIM, causal=False, softmax_scale=_DIM**-0.5)


def _packed_components(meta, count=3, requires_grad=False, device: torch.device = _CPU):
    # bf16 matches the real qkvg buffer and keeps dtype warnings out of route
    # assertions. Dimension zero is 3or4*batch, exactly as Attention.forward.
    return torch.randn(
        count,
        meta.total_seq_length,
        _HEADS,
        _DIM,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=requires_grad,
    )


def _run(impl, meta, *, raw=None, count=3, requires_grad=False):
    if raw is None:
        raw = _packed_components(meta,
                                 count=count,
                                 requires_grad=requires_grad,
                                 device=meta.variable_block_sizes.device)
    tiled = impl.preprocess_qkv(raw, meta)
    components = tiled.chunk(count, dim=0)
    gate = components[3] if count == 4 else None
    output = impl.forward(components[0], components[1], components[2], gate, meta)
    return output, tiled, raw


class _FakeSm100a:
    """CPU stand-in which enforces the real kernel's even-pair contract."""

    def __init__(self, supported=True):
        self.supported = supported
        self.support_calls = []
        self.calls = []

    def is_supported(self, q, variable_block_sizes):
        self.support_calls.append(dict(q=q, vbs=variable_block_sizes))
        n_tiles = variable_block_sizes.numel()
        geometry_supported = n_tiles % 2 == 0 and q.shape[2] == n_tiles * 64
        return self.supported and geometry_supported

    def block_sparse_attn_sm100a(self, q, k, v, q2k_idx, q2k_num, variable_block_sizes, need_lse=True):
        self.calls.append(
            dict(
                q=q,
                k=k,
                v=v,
                q2k_idx=q2k_idx,
                q2k_num=q2k_num,
                vbs=variable_block_sizes,
                need_lse=need_lse,
            ))
        return q.clone(), None


def _fake_map_to_index(block_map):
    """Pure-torch stand-in for Triton's map_to_index (same contract)."""
    batch, heads, query_tiles, key_tiles = block_map.shape
    idx = torch.full(
        (batch, heads, query_tiles, key_tiles),
        -1,
        dtype=torch.int32,
        device=block_map.device,
    )
    num = block_map.sum(dim=-1, dtype=torch.int32)
    for batch_idx in range(batch):
        for head_idx in range(heads):
            for query_idx in range(query_tiles):
                cols = torch.nonzero(block_map[batch_idx, head_idx, query_idx], as_tuple=False).flatten()
                idx[batch_idx, head_idx, query_idx, :cols.numel()] = cols.to(torch.int32)
    return idx, num


class _FakeTriton:

    def __init__(self):
        self.calls = []

    def __call__(self, q, k, v, mask, variable_block_sizes):
        self.calls.append(dict(q=q, k=k, v=v, mask=mask, vbs=variable_block_sizes))
        return q.clone(), None


@pytest.fixture()
def routed(monkeypatch):
    """Install CPU fakes for both kernel routes."""
    fake_sm = _FakeSm100a()
    fake_triton = _FakeTriton()
    monkeypatch.delenv(VSA_SM100A_ENV, raising=False)
    monkeypatch.setattr(vsa_h3, "_sm100a", fake_sm)
    monkeypatch.setattr(vsa_h3, "block_sparse_attn_64_bhsd", fake_triton)
    monkeypatch.setattr(vsa_h3, "map_to_index", _fake_map_to_index)
    return fake_sm, fake_triton


def test_reason_covers_every_precondition():
    query = torch.randn(1, _HEADS, 128, _DIM)
    variable_block_sizes = torch.full((2, ), 64, dtype=torch.long)
    assert "not installed" in _sm100a_unavailable_reason(None, query, variable_block_sizes, grad_mode=False)
    ok = _FakeSm100a(supported=True)
    assert "forward-only" in _sm100a_unavailable_reason(ok, query, variable_block_sizes, grad_mode=True)
    bad = _FakeSm100a(supported=False)
    assert "is_supported" in _sm100a_unavailable_reason(bad, query, variable_block_sizes, grad_mode=False)
    assert _sm100a_unavailable_reason(ok, query, variable_block_sizes, grad_mode=False) is None


@pytest.mark.parametrize(
    ("tile_size", "prefix_segments", "env_enabled", "requires_grad", "extra_tiles"),
    [
        (64, (70, 30), False, False, 0),  # odd, but route is opt-in
        (64, (70, 30), True, False, 1),  # odd no-grad sm100a transport
        (64, (70, 30), True, True, 0),  # every grad path stays logical
        (64, (64, 64), True, False, 0),  # already-even geometry
        (256, (70, 30), True, False, 0),  # pair contract is tile-64 only
    ],
)
def test_preprocess_adds_partner_only_for_odd_tile64_no_grad(
    monkeypatch,
    tile_size,
    prefix_segments,
    env_enabled,
    requires_grad,
    extra_tiles,
):
    if env_enabled:
        monkeypatch.setenv(VSA_SM100A_ENV, "1")
    else:
        monkeypatch.delenv(VSA_SM100A_ENV, raising=False)
    meta = _build_meta(prefix_segments=prefix_segments, tile_size=tile_size)
    raw = _packed_components(meta, requires_grad=requires_grad)
    tiled = _impl().preprocess_qkv(raw, meta)
    logical_tiles = meta.variable_block_sizes.numel()
    logical_seq_len = logical_tiles * tile_size

    assert tiled.shape[1] == (logical_tiles + extra_tiles) * tile_size
    assert torch.equal(tiled[:, meta.untile_combined_index], raw)
    if extra_tiles:
        assert torch.count_nonzero(tiled[:, logical_seq_len:]) == 0


def test_geometry_change_clears_all_padding_and_matches_fresh_sparse_gate_oracle(routed, monkeypatch):
    """An even N buffer reused by odd N+1 must not retain any old valid row."""
    fake_sm, _ = routed
    monkeypatch.setenv(VSA_SM100A_ENV, "1")
    builder = MiniMaxH3VSAMetadataBuilder()
    build_args = dict(
        current_timestep=0,
        raw_latent_shape=_SPEC["raw_latent_shape"],
        patch_size=_SPEC["patch_size"],
        VSA_sparsity=0.5,
        device=_CPU,
        tile_size=64,
    )
    even_meta = builder.build(prefix_segments=(70, 70), **build_args)
    odd_meta = builder.build(prefix_segments=(70, 30), **build_args)
    assert even_meta.variable_block_sizes.numel() == odd_meta.variable_block_sizes.numel() + 1
    impl = _impl()

    even_tiled = impl.preprocess_qkv(_packed_components(even_meta, count=4), even_meta)
    assert torch.count_nonzero(even_tiled[:, -64:]) > 0
    even_ptr = even_tiled.data_ptr()
    odd_raw = _packed_components(odd_meta, count=4)
    assert torch.count_nonzero(odd_raw[3]) > 0
    reused_output, odd_tiled, _ = _run(impl, odd_meta, raw=odd_raw, count=4)
    reused_call = fake_sm.calls[-1]

    assert odd_tiled.data_ptr() == even_ptr, "fixture must exercise shared-buffer reuse"
    is_padding = torch.ones(odd_tiled.shape[1], dtype=torch.bool)
    is_padding[odd_meta.untile_combined_index] = False
    assert torch.count_nonzero(odd_tiled[:, is_padding]) == 0

    fresh_meta = _build_meta(prefix_segments=(70, 30), sparsity=0.5)
    fresh_output, fresh_tiled, _ = _run(impl, fresh_meta, raw=odd_raw, count=4)
    fresh_call = fake_sm.calls[-1]

    assert fresh_tiled.data_ptr() != odd_tiled.data_ptr()
    assert (reused_call["q2k_num"][..., :odd_meta.variable_block_sizes.numel()] <
            odd_meta.variable_block_sizes.numel()).any()
    assert torch.equal(reused_call["q2k_idx"], fresh_call["q2k_idx"])
    assert torch.equal(reused_call["q2k_num"], fresh_call["q2k_num"])
    assert torch.equal(reused_output, fresh_output)


def test_default_off_odd_geometry_routes_unchanged_triton(routed):
    fake_sm, fake_triton = routed
    meta = _build_meta()
    output, tiled, _ = _run(_impl(), meta)
    n_tiles = meta.variable_block_sizes.numel()

    assert n_tiles % 2 == 1
    assert tiled.shape[1] == n_tiles * 64
    assert len(fake_triton.calls) == 1 and fake_sm.calls == []
    call = fake_triton.calls[0]
    assert call["q"].shape[2] == n_tiles * 64
    assert call["mask"].shape[-2:] == (n_tiles, n_tiles)
    assert torch.equal(call["vbs"], meta.variable_block_sizes)
    assert output.shape == (1, n_tiles * 64, _HEADS, _DIM)


def test_env_on_odd_geometry_pads_only_sm100a_transport(routed, monkeypatch):
    fake_sm, fake_triton = routed
    monkeypatch.setenv(VSA_SM100A_ENV, "1")
    meta = _build_meta()
    original_sizes = meta.variable_block_sizes.clone()
    output, tiled, _ = _run(_impl(), meta)
    n_tiles = meta.variable_block_sizes.numel()
    logical_seq_len = n_tiles * 64

    assert n_tiles % 2 == 1
    assert len(fake_sm.calls) == 1 and fake_triton.calls == []
    assert tiled.shape[1] == (n_tiles + 1) * 64
    assert torch.count_nonzero(tiled[:, logical_seq_len:]) == 0
    call = fake_sm.calls[0]
    assert call["q"].shape[2] == (n_tiles + 1) * 64
    assert torch.count_nonzero(call["q"][:, :, logical_seq_len:]) == 0
    assert torch.count_nonzero(call["k"][:, :, logical_seq_len:]) == 0
    assert torch.count_nonzero(call["v"][:, :, logical_seq_len:]) == 0

    # Dense logical rows still attend exactly the N logical keys. The dummy
    # row attends nowhere, and the dummy key never appears in a real row.
    assert call["q2k_num"].dtype == torch.int32
    assert (call["q2k_num"][..., :n_tiles] == n_tiles).all()
    assert (call["q2k_num"][..., n_tiles] == 0).all()
    expected_cols = torch.arange(n_tiles, dtype=torch.int32).expand(1, _HEADS, n_tiles, n_tiles)
    assert torch.equal(call["q2k_idx"][..., :n_tiles, :n_tiles], expected_cols)
    assert (call["q2k_idx"][..., :n_tiles, n_tiles] == -1).all()
    assert (call["q2k_idx"][..., n_tiles, :] == -1).all()
    assert torch.equal(call["vbs"][:-1], original_sizes.to(torch.int32))
    assert call["vbs"][-1].item() == 0
    assert call["need_lse"] is False

    # The transport pad never changes logical metadata or escapes forward.
    assert torch.equal(meta.variable_block_sizes, original_sizes)
    assert int(meta.variable_block_sizes.sum()) == meta.total_seq_length
    assert output.shape == (1, logical_seq_len, _HEADS, _DIM)


def test_env_on_even_geometry_uses_sm100a_without_partner(routed, monkeypatch):
    fake_sm, fake_triton = routed
    monkeypatch.setenv(VSA_SM100A_ENV, "1")
    meta = _build_meta(prefix_segments=(64, 64))
    output, tiled, _ = _run(_impl(), meta)
    n_tiles = meta.variable_block_sizes.numel()

    assert n_tiles % 2 == 0
    assert len(fake_sm.calls) == 1 and fake_triton.calls == []
    assert tiled.shape[1] == n_tiles * 64
    call = fake_sm.calls[0]
    assert call["q"].shape[2] == n_tiles * 64
    assert call["vbs"].numel() == n_tiles and (call["vbs"] > 0).all()
    assert (call["q2k_num"] == n_tiles).all()
    assert output.shape[1] == n_tiles * 64


def test_env_on_grad_inputs_keep_logical_triton_and_backward(routed, monkeypatch):
    fake_sm, fake_triton = routed
    monkeypatch.setenv(VSA_SM100A_ENV, "1")
    meta = _build_meta()
    output, tiled, raw = _run(_impl(), meta, requires_grad=True)
    n_tiles = meta.variable_block_sizes.numel()

    assert tiled.shape[1] == n_tiles * 64
    assert len(fake_triton.calls) == 1 and fake_sm.calls == []
    assert fake_triton.calls[0]["q"].shape[2] == n_tiles * 64
    output.float().sum().backward()
    assert raw.grad is not None and torch.count_nonzero(raw.grad[0]) > 0

    # The same process can route a later non-grad student/critic forward.
    _run(_impl(), meta, raw=raw.detach())
    assert len(fake_sm.calls) == 1


def test_env_on_unsupported_strips_partner_before_triton(routed, monkeypatch):
    fake_sm, fake_triton = routed
    fake_sm.supported = False
    monkeypatch.setenv(VSA_SM100A_ENV, "1")
    warnings = []
    monkeypatch.setattr(vsa_h3.logger, "warning_once", warnings.append)
    meta = _build_meta()
    n_tiles = meta.variable_block_sizes.numel()

    for _ in range(2):
        output, tiled, _ = _run(_impl(), meta)
        assert tiled.shape[1] == (n_tiles + 1) * 64
        assert output.shape[1] == n_tiles * 64

    assert fake_sm.calls == [] and len(fake_triton.calls) == 2
    for call in fake_triton.calls:
        assert call["q"].shape[2] == n_tiles * 64
        assert call["mask"].shape[-2:] == (n_tiles, n_tiles)
        assert torch.equal(call["vbs"], meta.variable_block_sizes)
    # warning_once normally deduplicates; replacing it with list.append lets
    # the test prove both fallback attempts carry the same actionable reason.
    assert len(warnings) == 2 and warnings[0] == warnings[1]
    assert VSA_SM100A_ENV in warnings[0] and "is_supported" in warnings[0]


def test_env_on_missing_module_strips_partner_before_triton(routed, monkeypatch):
    _, fake_triton = routed
    monkeypatch.setattr(vsa_h3, "_sm100a", None)
    monkeypatch.setenv(VSA_SM100A_ENV, "1")
    warnings = []
    monkeypatch.setattr(vsa_h3.logger, "warning_once", warnings.append)
    meta = _build_meta()
    output, tiled, _ = _run(_impl(), meta)
    n_tiles = meta.variable_block_sizes.numel()

    assert tiled.shape[1] == (n_tiles + 1) * 64
    assert output.shape[1] == n_tiles * 64
    assert len(fake_triton.calls) == 1
    assert fake_triton.calls[0]["q"].shape[2] == n_tiles * 64
    assert warnings and "not installed" in warnings[0]


def test_no_grad_context_routes_requires_grad_leaf_through_padded_sm100a(routed, monkeypatch):
    """A requires-grad leaf under torch.no_grad() is a no-grad forward."""
    fake_sm, fake_triton = routed
    monkeypatch.setenv(VSA_SM100A_ENV, "1")
    meta = _build_meta()
    raw = _packed_components(meta, requires_grad=True)

    with torch.no_grad():
        output, tiled, _ = _run(_impl(), meta, raw=raw)

    assert tiled.shape[1] == (meta.variable_block_sizes.numel() + 1) * 64
    assert output.shape[1] == meta.variable_block_sizes.numel() * 64
    assert len(fake_sm.calls) == 1 and fake_triton.calls == []


def test_sparse_topk_and_nonzero_gate_are_bitwise_unchanged(routed, monkeypatch):
    fake_sm, fake_triton = routed
    torch.manual_seed(7)
    meta = _build_meta(sparsity=0.5)
    impl = _impl()
    raw = _packed_components(meta, count=4)
    n_tiles = meta.variable_block_sizes.numel()

    monkeypatch.delenv(VSA_SM100A_ENV, raising=False)
    triton_output, _, _ = _run(impl, meta, raw=raw, count=4)
    logical_mask = fake_triton.calls[-1]["mask"]

    monkeypatch.setenv(VSA_SM100A_ENV, "1")
    sm100a_output, _, _ = _run(impl, meta, raw=raw, count=4)
    call = fake_sm.calls[-1]
    logical_idx, logical_num = _fake_map_to_index(logical_mask)

    # The sm100a mask prefix is exactly the trained logical top-k decision;
    # the only additions are one unselected key and one empty query row.
    assert torch.equal(call["q2k_num"][..., :n_tiles], logical_num)
    assert torch.equal(call["q2k_idx"][..., :n_tiles, :n_tiles], logical_idx)
    assert (call["q2k_idx"][..., :n_tiles, n_tiles] == -1).all()
    assert (call["q2k_num"][..., n_tiles] == 0).all()
    assert torch.equal(sm100a_output, triton_output)


def test_forward_rejects_non_contract_transport_shapes(routed):
    del routed  # kernel fakes only make forward's initial availability check pass
    impl = _impl()
    odd_meta = _build_meta()
    odd_tiles = odd_meta.variable_block_sizes.numel()

    too_many = torch.zeros(1, (odd_tiles + 2) * 64, _HEADS, _DIM, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="tiled query has length"):
        impl.forward(too_many, too_many, too_many, None, odd_meta)

    logical = torch.zeros(1, odd_tiles * 64, _HEADS, _DIM, dtype=torch.bfloat16)
    one_partner = torch.zeros(1, (odd_tiles + 1) * 64, _HEADS, _DIM, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="tiled key length"):
        impl.forward(logical, one_partner, logical, None, odd_meta)
    with pytest.raises(ValueError, match="tiled gate length"):
        impl.forward(logical, logical, logical, one_partner, odd_meta)

    even_meta = _build_meta(prefix_segments=(64, 64))
    even_tiles = even_meta.variable_block_sizes.numel()
    invalid_even_partner = torch.zeros(1, (even_tiles + 1) * 64, _HEADS, _DIM, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="expected the logical length"):
        impl.forward(invalid_even_partner, invalid_even_partner, invalid_even_partner, None, even_meta)


def test_real_sm100a_no_grad_route_receipt(monkeypatch):
    """Exercise the odd-tile semantic pad through the actual GB200 extension.

    The v10 compute-node gate first obtains a Triton-64 oracle with the same
    sparse top-k and nonzero gate, then makes fallback fatal and runs the
    padded sm100a route. This is both a routing receipt and the final numerical
    proof that the zero-valid partner is semantically invisible.
    """
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires a GB200 (sm_100a) compute node")
    if vsa_h3._sm100a is None or not vsa_h3._sm100a._HAS_VSA_SM100A:
        pytest.skip("requires a fastvideo_kernel build containing sm_100a VSA")

    device = torch.device("cuda")
    meta = _build_meta(device=device, sparsity=0.5)
    assert meta.variable_block_sizes.numel() % 2 == 1
    impl = _impl()
    torch.manual_seed(11)
    raw = _packed_components(meta, count=4, device=device)

    with torch.inference_mode():
        monkeypatch.delenv(VSA_SM100A_ENV, raising=False)
        triton_tiled = impl.preprocess_qkv(raw, meta)
        tq, tk, tv, tg = triton_tiled.chunk(4, dim=0)
        triton_output = impl.postprocess_output(impl.forward(tq, tk, tv, tg, meta), meta)

        def reject_triton(*args, **kwargs):
            raise AssertionError("odd no-grad VSA-H3 unexpectedly fell back to Triton-64")

        monkeypatch.setattr(vsa_h3, "block_sparse_attn_64_bhsd", reject_triton)
        monkeypatch.setenv(VSA_SM100A_ENV, "1")
        sm100a_tiled = impl.preprocess_qkv(raw, meta)
        sq, sk, sv, sg = sm100a_tiled.chunk(4, dim=0)
        sm100a_output = impl.postprocess_output(impl.forward(sq, sk, sv, sg, meta), meta)
        torch.cuda.synchronize()

    assert sm100a_tiled.shape[1] == triton_tiled.shape[1] + 64
    assert sm100a_output.shape == triton_output.shape == (1, meta.total_seq_length, _HEADS, _DIM)
    assert torch.isfinite(sm100a_output).all().item()
    max_abs_diff = (sm100a_output.float() - triton_output.float()).abs().max().item()
    torch.testing.assert_close(sm100a_output.float(), triton_output.float(), atol=0.04, rtol=0.02)
    print(f"V10_SM100A_ODD_ROUTE_RECEIPT=max_abs_diff:{max_abs_diff:.6f}")
