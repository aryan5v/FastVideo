# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from scripts.fasth3_sprint import score_minimax_h3_blocks as scoring


def _write_record(root: Path, sample_id: str, caption: str, **extra: str) -> dict[str, str]:
    text_sha1 = f"text-{sample_id}"
    (root / "latents").mkdir(parents=True, exist_ok=True)
    (root / "text").mkdir(parents=True, exist_ok=True)
    (root / "latents" / f"{sample_id}.safetensors").touch()
    (root / "text" / f"{text_sha1}.safetensors").touch()
    return {
        "id": sample_id,
        "caption": caption,
        "text_sha1": text_sha1,
        **extra,
    }


def _write_manifest(root: Path, records: list[dict[str, str]]) -> None:
    (root / "manifest_rank00000.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
    )


def test_calibration_requires_provenance_for_multishot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = tmp_path / "corpus"
    supplement = tmp_path / "supplement"
    corpus_records = [
        _write_record(corpus, "football", "shot football"),
        _write_record(corpus, "speech", "a woman speaking"),
        _write_record(corpus, "music", "playing violin"),
        _write_record(corpus, "motion", "a bicycle moving"),
        _write_record(corpus, "sound", "an engine revving"),
    ]
    supplement_records = [
        _write_record(
            supplement,
            "multishot",
            "a hard cut between three scenes",
            category="multiple_shots",
            provenance="released_dense_v1_synthetic_multishot",
        ),
    ]
    _write_manifest(corpus, corpus_records)
    _write_manifest(supplement, supplement_records)
    monkeypatch.setattr(scoring, "CATEGORY_QUOTAS", {name: 1 for name in scoring.CATEGORY_QUOTAS})

    records = scoring._calibration_records(corpus, 7, supplement_root=supplement)

    assert len(records) == 5
    assert {record["category"] for record in records} == set(scoring.CATEGORY_QUOTAS)
    assert "football" not in {record["id"] for record in records}
    assert next(record for record in records if record["id"] == "multishot")["_corpus_root"] == str(supplement)


def test_calibration_rejects_missing_multishot_supplement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = tmp_path / "corpus"
    _write_manifest(corpus, [_write_record(corpus, "football", "shot football")])
    monkeypatch.setattr(scoring, "CATEGORY_QUOTAS", {name: 1 for name in scoring.CATEGORY_QUOTAS})

    with pytest.raises(RuntimeError, match="multiple_shots"):
        scoring._calibration_records(corpus, 7)
