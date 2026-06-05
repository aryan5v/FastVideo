# SPDX-License-Identifier: Apache-2.0
from pathlib import Path


def test_model_code_does_not_call_rmsnorm_forward_native_directly():
    repo_root = Path(__file__).resolve().parents[3]
    model_roots = [
        repo_root / "fastvideo" / "models",
        repo_root / "fastvideo" / "train" / "models",
    ]

    offenders = []
    for root in model_roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if ".forward_native(" in text:
                offenders.append(path.relative_to(repo_root).as_posix())

    assert offenders == []
