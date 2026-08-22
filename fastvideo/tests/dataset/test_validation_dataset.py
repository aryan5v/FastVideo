# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from fastvideo.dataset.validation_dataset import ValidationDataset


@pytest.mark.parametrize("wrapped", [False, True])
def test_validation_json_accepts_array_and_legacy_data_wrapper(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    wrapped: bool,
) -> None:
    rows = [{
        "caption": "A held-out prompt",
        "source": "source-a",
        "sample_id": "sample-1",
        "width": 128,
        "height": 80,
        "num_frames": 39,
    }]
    document = {"data": rows} if wrapped else rows
    manifest = tmp_path / "validation.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr("fastvideo.dataset.validation_dataset.get_world_rank", lambda: 0)
    monkeypatch.setattr("fastvideo.dataset.validation_dataset.get_world_size", lambda: 1)
    monkeypatch.setattr("fastvideo.dataset.validation_dataset.get_sp_world_size", lambda: 1)

    dataset = ValidationDataset(str(manifest))

    assert dataset.all_samples == rows
    assert list(dataset)[0]["prompt"] == rows[0]["caption"]
