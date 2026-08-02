from __future__ import annotations

from collections import UserDict

import pytest

from fastvideo.models.encoders.reason1 import _normalize_chat_template_input_ids


def test_normalizes_batch_encoding_style_mapping() -> None:
    output = UserDict({"input_ids": [[101, 102, 103]]})

    assert _normalize_chat_template_input_ids(output) == [101, 102, 103]


def test_normalizes_legacy_list_output() -> None:
    assert _normalize_chat_template_input_ids([101, 102]) == [101, 102]


@pytest.mark.parametrize(
    "output",
    [UserDict({"attention_mask": [[1]]}), [[1], [2]], [True, 2]],
)
def test_rejects_ambiguous_chat_template_output(output: object) -> None:
    with pytest.raises(RuntimeError):
        _normalize_chat_template_input_ids(output)
