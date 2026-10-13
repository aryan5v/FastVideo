"""Optional per-clip reference images (shown with every pair of a clip). CPU only."""
import json

import pytest

from blind_ab.app_state import AppState
from blind_ab.bundle import BundleError, load_bundle
from blind_ab.votes import VoteStore

from test_blind_ab_bundle_parsing import _write

ARMS = ("bf16", "calibrated", "qad120")


def make_bundle(root, reference="refs/c1.jpg", write_image=True):
    _write(root / "arms.json", json.dumps({"arms": [{"slug": a} for a in ARMS]}))
    for clip_id in ("c1", "c2"):
        for arm in ARMS:
            _write(root / "arms" / arm / f"{clip_id}.mp4")
    if write_image:
        _write(root / "refs" / "c1.jpg")
    prompts = {"c1": {"prompt": "p1", "guidance": "faces", "reference": reference}, "c2": "p2"}
    _write(root / "prompts.json", json.dumps(prompts))
    return root


def test_reference_resolves_inside_the_bundle(tmp_path):
    root = make_bundle(tmp_path / "b")
    bundle = load_bundle(root)
    assert bundle.clip("c1").reference == (root / "refs" / "c1.jpg").resolve()
    assert bundle.clip("c2").reference is None


@pytest.mark.parametrize("reference, write_image", [
    ("../outside.jpg", True),
    ("refs/c1.mp4", True),
    ("refs/c1.jpg", False),
    (3, True),
])
def test_bad_reference_is_rejected(tmp_path, reference, write_image):
    (tmp_path / "outside.jpg").write_bytes(b"x")
    with pytest.raises(BundleError, match="reference"):
        load_bundle(make_bundle(tmp_path / "b", reference, write_image))


def test_ballot_carries_reference_url_only_when_present(tmp_path):
    root = make_bundle(tmp_path / "b")
    state = AppState(load_bundle(root), VoteStore(tmp_path / "votes.jsonl"))
    seen = {}
    for _ in range(40):
        ballot = state.new_ballot("ann")
        seen[ballot["clip_id"]] = ballot["reference_url"]
    assert seen == {"c1": "/reference/c1", "c2": ""}
    assert state.reference_path("c1") == (root / "refs" / "c1.jpg").resolve()
    assert state.reference_path("c2") is None and state.reference_path("nope") is None
