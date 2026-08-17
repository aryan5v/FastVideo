# SPDX-License-Identifier: Apache-2.0
"""Resolve an MLX DiT checkpoint from a local path or a Hugging Face repo.

``--mlx-checkpoint`` originally took a local directory only, so running a
published release meant downloading it by hand first. The quantized FastMetal
checkpoints live on the Hub, so accept a repo id there too and fetch it.

Local paths win. A value that names an existing directory is never treated as a
repo id, so a directory called ``org/name`` under the working directory still
resolves to itself and no network call happens.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastvideo.logger import init_logger
from fastvideo.mlx_runtime.checkpoint import MANIFEST_FILENAME, WEIGHTS_FILENAME

logger = init_logger(__name__)

# Hugging Face repo ids are "<owner>/<name>": exactly one slash, and each part
# limited to the characters the Hub allows. Deliberately strict — a mistyped
# local path should fail as a path, not turn into a download attempt.
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

# Only the two files the checkpoint format defines. A release repo may also
# carry a README or preview art; there is no reason to pull those.
CHECKPOINT_PATTERNS = [MANIFEST_FILENAME, WEIGHTS_FILENAME]


def looks_like_repo_id(value: str) -> bool:
    """
    Report whether a string has the shape of a Hugging Face repo id.

    Parameters:
        value (str): The candidate value, for example ``FastVideo/FastMetal-1.3B-QAD``.

    Returns:
        bool: `True` when the value is ``owner/name`` in Hub-legal characters.
    """
    return bool(_REPO_ID_RE.match(value))


def resolve_mlx_checkpoint(value: str | Path) -> Path:
    """
    Resolve an MLX DiT checkpoint reference to a local directory.

    An existing local directory is returned unchanged. Otherwise a value shaped
    like a Hugging Face repo id is downloaded, reusing the local Hub cache when
    it is already present.

    Parameters:
        value (str | Path): A local checkpoint directory or a Hub repo id.

    Returns:
        Path: A local directory holding the checkpoint manifest and weights.

    Raises:
        FileNotFoundError: If the value is neither an existing directory nor a
            usable repo id.
        RuntimeError: If the download fails, with the repo id named so a private
            or misspelled repo is distinguishable from a network problem.
    """
    path = Path(value)
    if path.is_dir():
        return path

    text = str(value)
    if not looks_like_repo_id(text):
        raise FileNotFoundError(f"MLX checkpoint {text!r} is not an existing directory, and does not look "
                                "like a Hugging Face repo id (expected 'owner/name', e.g. "
                                "'FastVideo/FastMetal-1.3B-QAD').")

    from huggingface_hub import snapshot_download

    logger.info("Fetching MLX checkpoint %s from Hugging Face", text)
    try:
        local_dir = snapshot_download(text, allow_patterns=CHECKPOINT_PATTERNS)
    except Exception as exc:  # noqa: BLE001 - surface the repo id with the cause.
        raise RuntimeError(f"Could not fetch MLX checkpoint {text!r} from Hugging Face: {exc}. "
                           "If the repo is private, log in with `huggingface-cli login` or set "
                           "HF_TOKEN.") from exc

    resolved = Path(local_dir)
    if not (resolved / MANIFEST_FILENAME).exists():
        raise FileNotFoundError(f"Hugging Face repo {text!r} does not contain {MANIFEST_FILENAME}, so it is "
                                "not an MLX DiT checkpoint. Pre-quantized checkpoints are produced by "
                                "--save-mlx-checkpoint; a base Diffusers repo will not work here.")
    return resolved


__all__ = ["CHECKPOINT_PATTERNS", "looks_like_repo_id", "resolve_mlx_checkpoint"]
