"""Bundle parsing for both layouts, plus build_bundle.py. Uses placeholder files, CPU only."""
import json

import pytest

from blind_ab.build_bundle import build_bundle, main as build_main, parse_arm_spec
from blind_ab.bundle import BundleError, load_bundle

FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64


def _write(path, data=FAKE_MP4):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data) if isinstance(data, bytes) else path.write_text(data)
    return path


def make_manifest_bundle(root, arms=("v1", "v2", "dense"), rows=((0, "s-000"), (1, "s-001"), (2, "s-002"))):
    _write(root / "arms.json", json.dumps({
        "schema_version": "x",
        "arms": [{"slug": a, "display_name": f"Arm {a}", "checkpoint_step": 10, "extra": {"k": 1}} for a in arms],
    }))
    lines = []
    for index, sample_id in rows:
        entries = {}
        for a in arms:
            rel = f"arms/{a}/videos/{index:03d}_{sample_id}.mp4"
            _write(root / rel)
            entries[a] = {"path": rel, "bytes": len(FAKE_MP4), "sha256": "0" * 64}
        lines.append(json.dumps({"index": index, "sample_id": sample_id, "prompt": f"prompt {index}", "fps": 24,
                                 "arms": entries}))
    _write(root / "manifest.jsonl", "\n".join(lines) + "\n")
    return root


def make_simple_bundle(root):
    _write(root / "arms.json", json.dumps({"arms": [
        {"slug": "base", "display_name": "Base", "speed": {"seconds_per_clip": 70.9, "hardware": "gpu",
                                                           "resolution": "832x480"}},
        {"slug": "fp8", "display_name": "FP8 decoder", "notes": "n", "speed": {"seconds_per_clip": 50}},
        {"slug": "solo", "display_name": "Solo"},
    ]}))
    for arm in ("base", "fp8"):
        for clip in ("p0_s1", "p1_s1"):
            _write(root / "arms" / arm / f"{clip}.mp4")
    _write(root / "arms" / "base" / "only_base.mp4")
    _write(root / "arms" / "base" / "notes.txt", "ignored")
    _write(root / "prompts.json", json.dumps({"p0_s1": "a cat", "p1_s1": "a dog"}))
    return root


def test_manifest_layout(tmp_path):
    bundle = load_bundle(make_manifest_bundle(tmp_path / "b"))
    assert bundle.layout == "manifest"
    assert bundle.arm_slugs == ("v1", "v2", "dense")
    assert bundle.arm("v1").display_name == "Arm v1"
    assert [c.clip_id for c in bundle.clips] == ["000_s-000", "001_s-001", "002_s-002"]
    clip = bundle.clip("001_s-001")
    assert clip.prompt == "prompt 1"
    assert set(clip.videos) == {"v1", "v2", "dense"}
    assert all(p.is_absolute() and p.is_file() for p in clip.videos.values())
    assert bundle.arm("v1").seconds_per_clip is None


def test_manifest_layout_falls_back_to_conventional_path_and_skips_missing(tmp_path):
    root = make_manifest_bundle(tmp_path / "b")
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    rows[0]["arms"]["v1"] = {}  # no path -> arms/v1/videos/000_s-000.mp4
    rows[1]["arms"]["v2"]["path"] = "arms/v2/videos/missing.mp4"
    (root / "arms/v2/videos/001_s-001.mp4").unlink()
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    bundle = load_bundle(root)
    assert "v1" in bundle.clip("000_s-000").videos
    assert set(bundle.clip("001_s-001").videos) == {"v1", "dense"}


def test_simple_layout(tmp_path):
    bundle = load_bundle(make_simple_bundle(tmp_path / "b"))
    assert bundle.layout == "simple"
    assert [c.clip_id for c in bundle.clips] == ["p0_s1", "p1_s1"]  # only_base dropped (one arm)
    assert bundle.clip("p0_s1").prompt == "a cat"
    assert bundle.arm("base").seconds_per_clip == pytest.approx(70.9)
    assert bundle.arm("base").speed["resolution"] == "832x480"
    assert bundle.arm("fp8").notes == "n"


@pytest.mark.parametrize("arms_doc, match", [
    ({"arms": []}, "non-empty"),
    ({"arms": [{"display_name": "x"}]}, "slug"),
    ({"arms": [{"slug": "a"}, {"slug": "a"}]}, "duplicate"),
    ({"arms": [{"slug": "../x"}]}, "invalid"),
    ({"arms": [{"slug": "a", "speed": 3}]}, "speed"),
])
def test_bad_arms_json(tmp_path, arms_doc, match):
    _write(tmp_path / "arms.json", json.dumps(arms_doc))
    with pytest.raises(BundleError, match=match):
        load_bundle(tmp_path)


def test_missing_bundle_and_no_pairs(tmp_path):
    with pytest.raises(BundleError):
        load_bundle(tmp_path / "nope")
    _write(tmp_path / "arms.json", json.dumps({"arms": [{"slug": "a"}, {"slug": "b"}]}))
    _write(tmp_path / "arms/a/x.mp4")
    _write(tmp_path / "arms/b/y.mp4")
    with pytest.raises(BundleError, match="two arms"):
        load_bundle(tmp_path)


@pytest.mark.parametrize("mode", ["symlink", "copy"])
def test_build_bundle_roundtrip(tmp_path, mode):
    src = tmp_path / "src"
    for arm, clips in {"base": ["p0", "p1", "p2"], "fast": ["p0", "p1"], "odd": ["p0", "zz"]}.items():
        for clip in clips:
            _write(src / arm / f"{clip}.mp4")
    out = tmp_path / "bundle"
    speed = {"base": {"seconds_per_clip": 80, "hardware": "gpu", "display_name": "Baseline", "notes": "ref"},
             "fast": {"seconds_per_clip": 20}}
    result = build_bundle(out, [("base", src / "base"), ("fast", src / "fast"), ("odd", src / "odd")], speed,
                          {"p0": "prompt zero", "zz": "unused"}, mode)
    assert result["clips"] == 2  # p0 (3 arms), p1 (2 arms); p2 and zz only in one arm
    bundle = load_bundle(out)
    assert bundle.arm("base").display_name == "Baseline" and bundle.arm("base").notes == "ref"
    assert bundle.arm("base").speed == {"seconds_per_clip": 80, "hardware": "gpu"}
    assert bundle.arm("odd").seconds_per_clip is None
    assert set(bundle.clip("p0").videos) == {"base", "fast", "odd"}
    assert bundle.clip("p0").prompt == "prompt zero"
    assert (out / "arms/base/p0.mp4").is_symlink() == (mode == "symlink")


def test_build_bundle_require_all_and_errors(tmp_path):
    src = tmp_path / "src"
    for arm, clips in {"a": ["p0", "p1"], "b": ["p0"], "c": ["p0", "p1"]}.items():
        for clip in clips:
            _write(src / arm / f"{clip}.mp4")
    arms = [(a, src / a) for a in "abc"]
    assert build_bundle(tmp_path / "o", arms, require_all=True)["clips"] == 1
    with pytest.raises(ValueError):
        build_bundle(tmp_path / "o2", arms[:1])
    with pytest.raises(ValueError, match="seconds_per_clip"):
        build_bundle(tmp_path / "o3", arms, speed={"a": {"seconds_per_clip": -1}})
    with pytest.raises(ValueError):
        parse_arm_spec("no_equals")
    assert build_main(["--out", str(tmp_path / "o4"), "--arm", f"a={src / 'a'}", "--arm", f"b={tmp_path}/none"]) == 2
