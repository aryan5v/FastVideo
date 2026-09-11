import pytest

from dreamverse.creation_capabilities import (
    LOBBY_CREATION_CAPABILITIES,
    validate_lobby_creation_config,
)


def test_lobby_capabilities_exclude_fl2va_and_4k():
    assert "fl2va" not in LOBBY_CREATION_CAPABILITIES.generation_modes
    assert "4k" not in LOBBY_CREATION_CAPABILITIES.resolutions
    assert LOBBY_CREATION_CAPABILITIES.generation_modes == frozenset({"t2va", "ref2va"})


def test_validate_lobby_creation_config_accepts_supported_t2va():
    validate_lobby_creation_config(
        model_id="fast-ltx23",
        generation_mode="t2va",
        aspect_ratio="16:9",
        resolution="1080p",
        duration_sec=5,
    )


def test_validate_lobby_creation_config_rejects_fl2va():
    with pytest.raises(ValueError, match="FL2VA"):
        validate_lobby_creation_config(
            model_id="fast-ltx23",
            generation_mode="fl2va",
            aspect_ratio="16:9",
            resolution="720p",
            duration_sec=5,
        )


def test_validate_lobby_creation_config_rejects_4k():
    with pytest.raises(ValueError, match="Unsupported resolution"):
        validate_lobby_creation_config(
            model_id="fast-ltx2",
            generation_mode="t2va",
            aspect_ratio="16:9",
            resolution="4k",
            duration_sec=10,
        )
