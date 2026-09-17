# SPDX-License-Identifier: Apache-2.0
"""CPU checks that the VSA-H3 inference (validation/denoising-stage) metadata
path builds exactly what the training path builds, stays in-bounds, and that
malformed geometry fails synchronously instead of as an async kernel fault.

Shapes mirror the v7 DMD2 validation request that motivated this test:
768x1344, 124 frames -> video latents (37, 48, 84), 207 audio latents,
patch (1, 2, 2), 3-step DMD ladder, 90% sparsity (jobs 2307/2321)."""

import math

import pytest
import torch

from fastvideo.attention.backends.video_sparse_attn_h3 import (_TILE_ELEMS, MiniMaxH3VSAMetadataBuilder,
                                                               _build_block_mask, _h3_tile_geometry,
                                                               _validate_h3_tile_geometry)
from fastvideo.pipelines.basic.minimax_h3.packing import (MINIMAX_H3_TEXT_TAG, audio_latent_num_frames,
                                                          build_packed_sequence, video_latent_num_frames)
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_denoising import _h3_vsa_prefix_segments

_CPU = torch.device("cpu")
_PATCH = (1, 2, 2)
_NUM_FRAMES = 124  # -> 37 latent frames, 207 audio latents
_LATENT = (video_latent_num_frames(_NUM_FRAMES), 768 // 16, 1344 // 16)
_NUM_AUDIO = audio_latent_num_frames(_NUM_FRAMES)
_SPARSITY = 0.9
_DMD_STEPS = 3

_TEXT_LENS = [7, 100, 255, 256, 257, 500]


def _layout(text_len: int, anchors: tuple[str, ...] = ()):
    tags = torch.full((text_len, ), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
    return build_packed_sequence(tags, *_LATENT, _NUM_AUDIO, _PATCH, anchors)


def _inference_metadata(builder, layout, step_index: int, sparsity: float = _SPARSITY):
    """Exactly the MiniMaxH3DenoisingStage.forward calling convention."""
    return builder.build(
        current_timestep=step_index,
        raw_latent_shape=(layout.num_video_latent_frames, layout.latent_height, layout.latent_width),
        patch_size=_PATCH,
        VSA_sparsity=sparsity,
        prefix_segments=_h3_vsa_prefix_segments(layout, _PATCH),
        device=_CPU,
        exempt=True,
        dense_layers=(),
    )


def _training_metadata(builder, layout, sparsity: float = _SPARSITY):
    """Exactly the MiniMaxH3Model._maybe_build_vsa_metadata calling convention."""
    return builder.build(
        current_timestep=0,
        raw_latent_shape=(layout.num_video_latent_frames, layout.latent_height, layout.latent_width),
        patch_size=_PATCH,
        VSA_sparsity=sparsity,
        prefix_segments=_h3_vsa_prefix_segments(layout, _PATCH),
        device=_CPU,
    )


def _assert_in_bounds(meta, layout, tag: str):
    n_tiles = meta.num_prefix_tiles + meta.num_video_tiles
    sizes = meta.variable_block_sizes
    assert sizes.numel() == n_tiles, tag
    assert int(sizes.min()) >= 1 and int(sizes.max()) <= _TILE_ELEMS, tag
    assert int(sizes.sum()) == meta.total_seq_length == layout.sequence_length, tag
    idx = meta.untile_combined_index
    assert idx.numel() == meta.total_seq_length, tag
    assert int(idx.min()) >= 0 and int(idx.max()) < n_tiles * _TILE_ELEMS, tag
    assert idx.unique().numel() == idx.numel(), f"{tag}: untile index must be injective"
    assert bool((idx % _TILE_ELEMS < sizes[idx // _TILE_ELEMS]).all()), tag


@pytest.mark.parametrize("text_len", _TEXT_LENS)
def test_inference_metadata_matches_training(text_len):
    layout = _layout(text_len)
    assert _h3_vsa_prefix_segments(layout, _PATCH) == (text_len, 0, _NUM_AUDIO * 2)

    meta_train = _training_metadata(MiniMaxH3VSAMetadataBuilder(), layout)
    _assert_in_bounds(meta_train, layout, f"train text={text_len}")

    infer_builder = MiniMaxH3VSAMetadataBuilder()  # one builder per denoise loop, as the stage does
    for step in range(_DMD_STEPS):
        meta_inf = _inference_metadata(infer_builder, layout, step)
        _assert_in_bounds(meta_inf, layout, f"infer text={text_len} step={step}")
        assert meta_inf.total_seq_length == meta_train.total_seq_length
        assert meta_inf.num_prefix_tiles == meta_train.num_prefix_tiles
        assert meta_inf.num_video_tiles == meta_train.num_video_tiles
        assert meta_inf.exempt == meta_train.exempt
        assert meta_inf.dense_layers == meta_train.dense_layers
        assert meta_inf.VSA_sparsity == meta_train.VSA_sparsity
        assert torch.equal(meta_inf.variable_block_sizes, meta_train.variable_block_sizes)
        assert torch.equal(meta_inf.untile_combined_index, meta_train.untile_combined_index)


def test_keyframe_conditioned_layout_in_bounds():
    """Image-conditioned validation adds condition keyframe rows to the prefix."""
    layout = _layout(100, anchors=("first", ))
    rows_per_frame = (_LATENT[1] // _PATCH[1]) * (_LATENT[2] // _PATCH[2])
    assert _h3_vsa_prefix_segments(layout, _PATCH) == (100, rows_per_frame, _NUM_AUDIO * 2)
    meta = _inference_metadata(MiniMaxH3VSAMetadataBuilder(), layout, 0)
    _assert_in_bounds(meta, layout, "keyframe-conditioned")


def test_route_a_expansion_in_bounds():
    """The 256->64 route-A remap the Triton fallback consumes stays in-bounds."""
    try:
        from fastvideo_kernel import block_sparse_attn_256
    except Exception as exc:  # triton driver probing raises RuntimeError on GPU-less hosts
        pytest.skip(f"fastvideo_kernel unavailable here: {exc}")
    layout = _layout(100)
    meta = _inference_metadata(MiniMaxH3VSAMetadataBuilder(), layout, 0)
    n_tiles = meta.variable_block_sizes.numel()
    scores = torch.randn(1, 4, n_tiles, n_tiles)
    mask = _build_block_mask(scores, meta.num_prefix_tiles, meta.num_video_tiles, _SPARSITY, exempt=True)
    mask64, sizes64 = block_sparse_attn_256._expand_mask_and_sizes_256_to_64(mask, meta.variable_block_sizes)
    assert mask64.shape[-2:] == (4 * n_tiles, 4 * n_tiles)
    assert sizes64.numel() == 4 * n_tiles
    assert int(sizes64.min()) >= 0 and int(sizes64.max()) <= 64
    assert int(sizes64.sum()) == meta.total_seq_length
    per_tile = sizes64.view(n_tiles, 4)
    assert bool((per_tile[:, 0] > 0).all()), "every logical tile keeps at least one valid 64-block"
    assert bool((per_tile[:, :-1] >= per_tile[:, 1:]).all()), "child sizes must be non-increasing"
    assert torch.equal(per_tile.sum(dim=1), meta.variable_block_sizes.to(per_tile.dtype))


def test_geometry_guard_rejects_corruption():
    """The synchronous guard must catch what would otherwise be an async fault."""
    prefix = (100, _NUM_AUDIO * 2)
    dit_shape = tuple(d // p for d, p in zip(_LATENT, _PATCH, strict=True))
    (_, sizes, untile, _, _) = _h3_tile_geometry(prefix, dit_shape, _CPU)

    with pytest.raises(ValueError, match="tile sizes out of bounds"):
        bad = sizes.clone()
        bad[0] = _TILE_ELEMS + 1
        _validate_h3_tile_geometry(prefix, dit_shape, bad, untile)
    with pytest.raises(ValueError, match="tile sizes out of bounds"):
        bad = sizes.clone()
        bad[-1] += 1  # sum mismatch
        _validate_h3_tile_geometry(prefix, dit_shape, bad, untile)
    with pytest.raises(ValueError, match="untile index"):
        _validate_h3_tile_geometry(prefix, dit_shape, sizes, untile[:-1])
    with pytest.raises(ValueError, match="injective"):
        bad = untile.clone()
        bad[1] = int(bad[0])  # duplicate slot
        _validate_h3_tile_geometry(prefix, dit_shape, sizes, bad)
    with pytest.raises(ValueError, match="injective"):
        bad = untile.clone()
        bad[0] = sizes.numel() * _TILE_ELEMS  # beyond the padded buffer
        _validate_h3_tile_geometry(prefix, dit_shape, sizes, bad)
    partial = int((sizes < _TILE_ELEMS).nonzero()[0])
    with pytest.raises(ValueError, match="injective"):
        bad = untile.clone()
        bad[0] = partial * _TILE_ELEMS + int(sizes[partial])  # first pad slot
        _validate_h3_tile_geometry(prefix, dit_shape, sizes, bad)

    _validate_h3_tile_geometry(prefix, dit_shape, sizes, untile)
    assert int(sizes.sum()) == sum(prefix) + math.prod(dit_shape)


if __name__ == "__main__":
    for _text_len in _TEXT_LENS:
        test_inference_metadata_matches_training(_text_len)
    test_keyframe_conditioned_layout_in_bounds()
    test_route_a_expansion_in_bounds()
    test_geometry_guard_rejects_corruption()
    print("all VSA-H3 inference-metadata parity checks passed")
