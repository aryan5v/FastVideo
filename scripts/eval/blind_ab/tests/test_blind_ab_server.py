"""HTTP server: blinded ballots, range requests, vote round-trip, results export. CPU only, localhost."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from blind_ab.app_state import AppState
from blind_ab.bundle import load_bundle
from blind_ab.serve import make_server, parse_range
from blind_ab.votes import VoteStore

from test_blind_ab_bundle_parsing import make_simple_bundle


@pytest.mark.parametrize("header, expected", [
    (None, None),
    ("bytes=0-", (0, 99)),
    ("bytes=10-19", (10, 19)),
    ("bytes=90-500", (90, 99)),
    ("bytes=-10", (90, 99)),
])
def test_parse_range(header, expected):
    assert parse_range(header, 100) == expected


@pytest.mark.parametrize("header", ["bytes=100-", "bytes=5-2", "items=0-1", "bytes=-", "bytes=-0"])
def test_parse_range_rejects(header):
    with pytest.raises(ValueError):
        parse_range(header, 100)


@pytest.fixture()
def server(tmp_path):
    root = make_simple_bundle(tmp_path / "bundle")
    (root / "arms/base/p0_s1.mp4").write_bytes(bytes(range(256)) * 4)
    store = VoteStore(tmp_path / "votes.jsonl")
    srv = make_server(AppState(load_bundle(root), store, baseline="base"), "127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", store
    srv.shutdown()
    srv.server_close()


def _get(url, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {})) as resp:
        return resp.status, dict(resp.headers), resp.read()


def _post(url, payload):
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_ballot_vote_results_roundtrip(server):
    base, store = server
    status, _, body = _get(f"{base}/api/ballot?voter=ann")
    ballot = json.loads(body)
    assert status == 200 and ballot["clip_id"] in ("p0_s1", "p1_s1")
    blob = json.dumps(ballot)
    assert "base" not in blob.replace(ballot["prompt"], "") and "fp8" not in blob  # blinded

    status, headers, data = _get(base + ballot["left_url"], {"Range": "bytes=2-5"})
    assert status == 206 and len(data) == 4 and headers["Content-Range"].startswith("bytes 2-5/")
    assert headers["Accept-Ranges"] == "bytes" and headers["Content-Type"] == "video/mp4"
    status, _, full = _get(base + ballot["right_url"])
    assert status == 200 and len(full) > 0

    status, reply = _post(f"{base}/api/vote", {"ballot_id": ballot["ballot_id"], "voter": "ann", "choice": "left",
                                               "comment": "sharper", "watch_seconds": 12.3})
    assert status == 200 and {reply["reveal"]["left"]["slug"], reply["reveal"]["right"]["slug"]} == {"base", "fp8"}
    assert _post(f"{base}/api/vote", {"ballot_id": ballot["ballot_id"], "voter": "ann", "choice": "left"})[0] == 410
    assert _post(f"{base}/api/vote", {"ballot_id": "nope", "voter": "ann", "choice": "left"})[0] == 410
    assert _post(f"{base}/api/vote", {"ballot_id": "x", "voter": "ann", "choice": "bogus"})[0] == 400

    (vote, ) = store.all()
    assert vote.comment == "sharper" and vote.watch_seconds == 12.3 and vote.voter == "ann"
    winner = reply["reveal"]["left"]["slug"]
    results = json.loads(_get(f"{base}/api/results")[2])
    assert results["total_votes"] == 1 and results["arms"][0]["slug"] == winner
    assert {r["slug"]: r["speedup_vs_baseline"] for r in results["arms"]}["fp8"] == pytest.approx(70.9 / 50)
    status, headers, csv_body = _get(f"{base}/api/results.csv?baseline=fp8")
    assert status == 200 and "attachment" in headers["Content-Disposition"]
    assert csv_body.decode().startswith("slug,display_name")


def test_static_and_errors(server):
    base, _ = server
    assert b"Which video looks better?" in _get(f"{base}/")[2]
    assert b"results" in _get(f"{base}/results")[2]
    for path in ("/static/../serve.py", "/static/nope.js", "/video/abc/left", "/video/abc/middle", "/api/ballot"):
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(base + path)
        assert err.value.code in (400, 404)
