# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for resolving an MLX checkpoint path or repo id."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastvideo.mlx_runtime.checkpoint import MANIFEST_FILENAME, WEIGHTS_FILENAME
from fastvideo.mlx_runtime import checkpoint_source
from fastvideo.mlx_runtime.checkpoint_source import (
    CHECKPOINT_PATTERNS,
    looks_like_repo_id,
    resolve_mlx_checkpoint,
)


def _checkpoint_dir(root: Path) -> Path:
    """Create a directory that looks like a saved MLX DiT checkpoint."""
    root.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST_FILENAME).write_text(json.dumps({"format_version": 1}))
    (root / WEIGHTS_FILENAME).write_bytes(b"")
    return root


@pytest.mark.parametrize("value", [
    "FastVideo/FastMetal-1.3B-QAD",
    "FastVideo/FastMetal-5B-QAD",
    "owner/name_with.chars-1",
])
def test_repo_ids_are_recognised(value: str) -> None:
    assert looks_like_repo_id(value)


@pytest.mark.parametrize("value", [
    "models/fastmetal-1.3b/extra",  # nested local path
    "/abs/path/to/checkpoint",
    "too/many/slashes",
    "noslash",
    "",
])
def test_non_repo_ids_are_rejected(value: str) -> None:
    assert not looks_like_repo_id(value)


def test_existing_directory_is_returned_unchanged(tmp_path: Path) -> None:
    local = _checkpoint_dir(tmp_path / "ckpt")
    assert resolve_mlx_checkpoint(local) == local


def test_local_directory_wins_over_repo_id_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A local dir shaped like `owner/name` must not trigger a download."""
    local = _checkpoint_dir(tmp_path / "FastVideo" / "FastMetal-1.3B-QAD")
    monkeypatch.chdir(tmp_path)

    import huggingface_hub

    def unexpected_download(*args, **kwargs):
        raise AssertionError("resolve_mlx_checkpoint downloaded an existing local directory")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", unexpected_download)
    assert resolve_mlx_checkpoint("FastVideo/FastMetal-1.3B-QAD") == Path("FastVideo/FastMetal-1.3B-QAD")
    assert local.is_dir()


def test_repo_id_is_downloaded_and_returned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fetched = _checkpoint_dir(tmp_path / "hub" / "FastMetal-1.3B-QAD")
    seen: dict[str, object] = {}

    import huggingface_hub

    def fake_snapshot_download(repo_id, allow_patterns=None, **kwargs):
        seen["repo_id"] = repo_id
        seen["allow_patterns"] = allow_patterns
        return str(fetched)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    assert resolve_mlx_checkpoint("FastVideo/FastMetal-1.3B-QAD") == fetched
    assert seen["repo_id"] == "FastVideo/FastMetal-1.3B-QAD"
    # Only the two checkpoint files, not any README or preview art in the repo.
    assert seen["allow_patterns"] == CHECKPOINT_PATTERNS
    assert set(CHECKPOINT_PATTERNS) == {MANIFEST_FILENAME, WEIGHTS_FILENAME}


def test_missing_local_path_is_not_treated_as_a_repo(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not look"):
        resolve_mlx_checkpoint(tmp_path / "nope")


def test_download_failure_names_the_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    def failing_download(*args, **kwargs):
        raise OSError("401 Client Error")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", failing_download)
    with pytest.raises(RuntimeError, match="FastVideo/FastMetal-1.3B-QAD"):
        resolve_mlx_checkpoint("FastVideo/FastMetal-1.3B-QAD")


def test_repo_without_a_manifest_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A base Diffusers repo is not an MLX checkpoint; say so rather than fail later."""
    empty = tmp_path / "diffusers-repo"
    empty.mkdir()

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(empty))
    with pytest.raises(FileNotFoundError, match="not an MLX DiT checkpoint"):
        resolve_mlx_checkpoint("FastVideo/FastWan2.1-T2V-1.3B-Diffusers")


def test_module_exports_are_stable() -> None:
    assert set(checkpoint_source.__all__) == {
        "CHECKPOINT_PATTERNS",
        "looks_like_repo_id",
        "resolve_mlx_checkpoint",
    }
