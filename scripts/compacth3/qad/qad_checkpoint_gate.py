#!/usr/bin/env python3
"""Select the QAD checkpoint: earliest one that cuts grain without raising anomalies.

Judged as PAIRED DELTAS against the PRE-QAD NVFP4 model (not absolute scores), on the
same prompts+seeds, on the hard-motion set. Criteria from the lead:

    d_grain    < 0      (grain falls)
    d_detail  >= 0      (detail does not regress)
    d_anoms   <= 0      (temporal anomalies do not increase)
    |d_motion| small    (motion magnitude preserved)

Earliest checkpoint satisfying ALL FOUR wins. If NONE does, that is itself the result:
standard QAD is trading away the NVFP4 stability benefit, and only then is a custom
temporal objective worth designing.

Runs over EVERY checkpoint (the run emits one per 25 steps), independent of the
built-in validation cadence (every 50) -- otherwise the sweet spot falls between samples.
"""
import argparse, glob, json, os, re, subprocess, sys

S = "/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829"
MOTION_TOL = 0.15   # |d_motion| within 15% of baseline counts as preserved


def ckpt_step(path):
    m = re.search(r"checkpoint-(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def metrics_for(video_dir):
    """Compute the four decision metrics for one model's generations.

    NOTE: motion magnitude is the mandatory anti-collapse guard. Implementations here
    are deliberately simple and must be swapped for the sweep agent's versions once
    those land, so both sides of every delta use the SAME code path.
    """
    out = {}
    for name, fn in (("motion", _motion_magnitude),
                     ("grain", _hf_energy),
                     ("detail", _sharpness),
                     ("anomalies", _colour_drift)):
        vals = []
        for v in sorted(glob.glob(os.path.join(video_dir, "*.mp4"))):
            try:
                vals.append(fn(v))
            except Exception as e:
                print(f"  WARN {name} failed on {os.path.basename(v)}: {type(e).__name__}", flush=True)
        out[name] = sum(vals) / len(vals) if vals else None
    return out


def _frames(v, n=32):
    import av, numpy as np
    c = av.open(v)
    fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    c.close()
    if len(fr) > n:
        idx = np.linspace(0, len(fr) - 1, n).astype(int)
        fr = [fr[i] for i in idx]
    return fr


def _motion_magnitude(v):
    """Mean optical-flow magnitude. The guard against a model that 'wins' by going static."""
    import cv2, numpy as np
    fr = _frames(v)
    mags = []
    for a, b in zip(fr, fr[1:]):
        ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        fl = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(float(np.sqrt(fl[..., 0] ** 2 + fl[..., 1] ** 2).mean()))
    return sum(mags) / len(mags) if mags else 0.0


def _hf_energy(v):
    """High-frequency energy: proxy for grain. Higher = grainier."""
    import cv2, numpy as np
    vals = []
    for f in _frames(v, 16):
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype("float32")
        vals.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
    return sum(vals) / len(vals) if vals else 0.0


def _sharpness(v):
    """Detail proxy. Deliberately the same family as grain; report both so a
    grain reduction that is really just blurring is visible."""
    import cv2, numpy as np
    vals = []
    for f in _frames(v, 16):
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype("float32")
        vals.append(float(cv2.Sobel(g, cv2.CV_32F, 1, 0).var()))
    return sum(vals) / len(vals) if vals else 0.0


def _colour_drift(v):
    """Flow-warped colour residual between adjacent frames: targets the reported
    'gloves change colour' failure. Motion-compensated, so legitimate motion
    does not count as drift."""
    import cv2, numpy as np
    fr = _frames(v)
    res = []
    for a, b in zip(fr, fr[1:]):
        ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        fl = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = ga.shape
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        wx = (xx + fl[..., 0]).astype(np.float32)
        wy = (yy + fl[..., 1]).astype(np.float32)
        warped = cv2.remap(a, wx, wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        la = cv2.cvtColor(warped, cv2.COLOR_RGB2LAB).astype("float32")
        lb = cv2.cvtColor(b, cv2.COLOR_RGB2LAB).astype("float32")
        res.append(float(np.abs(la - lb).mean()))
    return sum(res) / len(res) if res else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", required=True, help="pre-QAD NVFP4 generations, same prompts+seeds")
    ap.add_argument("--ckpt-root", required=True, help="QAD output_dir containing checkpoint-*")
    ap.add_argument("--gen-dir-template", required=True,
                    help="where per-checkpoint generations live; must contain {step}")
    ap.add_argument("--out", default=f"{S}/adaln_rank_analysis/qad_checkpoint_gate.json")
    a = ap.parse_args()

    base = metrics_for(a.baseline_dir)
    print(f"BASELINE (pre-QAD NVFP4): {json.dumps(base, indent=2)}", flush=True)

    steps = sorted(s for s in (ckpt_step(p) for p in glob.glob(f"{a.ckpt_root}/checkpoint-*")) if s)
    rows, winner = [], None
    for s in steps:
        d = a.gen_dir_template.format(step=s)
        if not os.path.isdir(d):
            print(f"checkpoint-{s}: no generations at {d} -- SKIPPED (must be generated)", flush=True)
            continue
        m = metrics_for(d)
        dl = {k: (m[k] - base[k]) for k in base if m.get(k) is not None and base.get(k)}
        rel = {k: (v / base[k]) for k, v in dl.items() if base.get(k)}
        ok = (rel.get("grain", 1) < 0 and rel.get("detail", -1) >= 0
              and rel.get("anomalies", 1) <= 0 and abs(rel.get("motion", 1)) <= MOTION_TOL)
        rows.append({"step": s, "abs": m, "delta": dl, "rel": rel, "satisfies": ok})
        print(f"checkpoint-{s}: d_grain={rel.get('grain',0):+.3f} d_detail={rel.get('detail',0):+.3f} "
              f"d_anom={rel.get('anomalies',0):+.3f} d_motion={rel.get('motion',0):+.3f}  "
              f"{'<== SATISFIES' if ok else ''}", flush=True)
        if ok and winner is None:
            winner = s

    verdict = ("EARLIEST satisfying checkpoint: %d" % winner) if winner is not None else \
              ("NONE satisfies: standard QAD is trading away the NVFP4 stability benefit "
               "-- a custom temporal objective is now justified")
    print("\nVERDICT:", verdict, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"baseline": base, "rows": rows, "winner": winner, "verdict": verdict},
              open(a.out, "w"), indent=2)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
