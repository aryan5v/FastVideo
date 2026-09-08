import pytest

from dreamverse.session_creation_config import (
    duration_sec_to_segment_cap,
    parse_session_creation_config,
    resolve_frame_size,
    validate_creation_mode_assets,
)


def test_parse_session_creation_config_defaults():
    config = parse_session_creation_config({})
    assert config.model_id == "fast-ltx23"
    assert config.creation_mode == "t2v"
    assert config.aspect_ratio == "16:9"
    assert config.resolution == "720p"
    assert config.duration_sec == 5
    assert config.generation_segment_cap == 1


def test_parse_session_creation_config_maps_duration_to_segment_cap():
    config = parse_session_creation_config(
        {
            "model_id": "fast-ltx2",
            "creation_mode": "ref2av",
            "aspect_ratio": "9:16",
            "resolution": "480p",
            "duration_sec": 15,
        },
    )
    assert config.model_id == "fast-ltx2"
    assert config.creation_mode == "ref2av"
    assert config.generation_segment_cap == 3
    assert config.frame_width >= 480
    assert config.frame_height >= 480


def test_resolve_frame_size_uses_model_default_for_1080p_landscape():
    width, height = resolve_frame_size("16:9", "1080p")
    assert (width, height) == (1920, 1088)


def test_duration_sec_to_segment_cap_respects_global_cap():
    assert duration_sec_to_segment_cap(15, global_cap=2) == 2


def test_validate_creation_mode_assets():
    validate_creation_mode_assets("t2v", has_initial_image=False, has_last_frame_image=False)
    with pytest.raises(ValueError, match="Omni reference"):
        validate_creation_mode_assets("ref2av", has_initial_image=False, has_last_frame_image=False)
    with pytest.raises(ValueError, match="First and last frame"):
        validate_creation_mode_assets("fl2av", has_initial_image=True, has_last_frame_image=False)
