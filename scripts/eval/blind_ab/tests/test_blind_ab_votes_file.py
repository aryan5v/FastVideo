"""Vote JSONL storage and validation. CPU only."""
import json
import threading

import pytest

from blind_ab.votes import Vote, VoteError, VoteStore, clean_vote_fields, vote_from_record


def _vote(i=0, choice="left"):
    return Vote(voter=f"v{i % 3}", timestamp="2026-01-01T00:00:00+00:00", clip_id=f"c{i}", left_arm="a",
                right_arm="b", choice=choice, comment="ok", watch_seconds=1.5, played_seconds=1.0, ballot_id=f"b{i}")


def test_append_and_reload_roundtrip(tmp_path):
    path = tmp_path / "sub" / "votes.jsonl"
    store = VoteStore(path)
    store.append(_vote(0))
    store.append(_vote(1, "tie"))
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    for key in ("voter", "timestamp", "clip_id", "left_arm", "right_arm", "choice", "comment", "watch_seconds"):
        assert key in record
    assert VoteStore(path).all() == (_vote(0), _vote(1, "tie"))


def test_malformed_lines_are_skipped(tmp_path):
    path = tmp_path / "votes.jsonl"
    path.write_text(_vote(0).to_json() + "\nnot json\n{\"voter\": \"x\"}\n\n" + _vote(1).to_json() + "\n")
    assert len(VoteStore(path).all()) == 2


def test_concurrent_appends_produce_whole_lines(tmp_path):
    path = tmp_path / "votes.jsonl"
    store = VoteStore(path)
    threads = [threading.Thread(target=lambda k=k: [store.append(_vote(k * 100 + j)) for j in range(25)])
               for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = path.read_text().splitlines()
    assert len(lines) == 200 and len(store.all()) == 200
    assert all(json.loads(line)["choice"] == "left" for line in lines)


def test_winner_property():
    assert _vote(0, "left").winner == "a"
    assert _vote(0, "right").winner == "b"
    assert _vote(0, "both_bad").winner is None


@pytest.mark.parametrize("payload", [
    {"voter": "x", "choice": "maybe"},
    {"voter": "", "choice": "left"},
    {"voter": "x" * 65, "choice": "left"},
    {"voter": "x", "choice": "left", "comment": "c" * 1001},
    {"voter": "x", "choice": "left", "watch_seconds": -1},
    {"voter": "x", "choice": "left", "watch_seconds": "abc"},
])
def test_invalid_payloads_rejected(payload):
    with pytest.raises(VoteError):
        clean_vote_fields(payload)


def test_clean_fields_strips_and_clamps():
    fields = clean_vote_fields({"voter": "  ann ", "choice": "tie", "comment": " hi ", "watch_seconds": 1e9})
    assert fields["voter"] == "ann" and fields["comment"] == "hi" and fields["watch_seconds"] == 86400.0


def test_vote_from_record_rejects_unknown_choice():
    record = json.loads(_vote(0).to_json())
    record["choice"] = "nope"
    with pytest.raises(VoteError):
        vote_from_record(record)
