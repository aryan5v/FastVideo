from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import cv2

EVAL_DIR = Path(os.environ.get("COMPACTH3_EVAL_ROOT", str(Path(SPRINT_ROOT).parent / "fasth3-eval")))
SCORER = EVAL_DIR / "score_34block.py"

# Working resolution for the frame-pair metrics.  Full-resolution flow on a
# 362-frame 1760x768 clip is the dominant cost and buys nothing for these
# aggregate statistics.
WORK_LONG_SIDE = 512

FPS = 24
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5


def clamp_num_frames(requested: int) -> int:
    n = int(requested)
    k = max(0, (n - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK)
    cand = FRAMES_PER_CHUNK * k + LATENTS_PER_CHUNK
    while cand > 15.0 * FPS and k > 0:
        k -= 1
        cand = FRAMES_PER_CHUNK * k + LATENTS_PER_CHUNK
    while cand < 5.0 * FPS:
        k += 1
        cand = FRAMES_PER_CHUNK * k + LATENTS_PER_CHUNK
    return cand


def _load_scorer():
    spec = importlib.util.spec_from_file_location("h3_score_34block", SCORER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h3_score_34block"] = mod
    spec.loader.exec_module(mod)
    return mod


def _resize(frame, long_side=WORK_LONG_SIDE):
    h, w = frame.shape[:2]
    s = long_side / max(h, w)
    if s >= 1.0:
        return frame
    return cv2.resize(frame, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                      interpolation=cv2.INTER_AREA)


def read_video(path, long_side=WORK_LONG_SIDE):
    """Return (bgr_frames, gray_frames) at reduced resolution."""
    cap = cv2.VideoCapture(str(path))
    bgr, gray = [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fr = _resize(fr, long_side)
        bgr.append(fr)
        gray.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    return bgr, gray


# ---------------------------------------------------------------- 1. motion
def motion_magnitude(gray):
    """Mean optical-flow magnitude per clip -- the anti-collapse guard.

    A rank that appears to win by producing near-static video is caught here.
    """
    mags = []
    for a, b in zip(gray[:-1], gray[1:]):
        f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        m, _ = cv2.cartToPolar(f[..., 0], f[..., 1])
        mags.append(float(m.mean()))
    if not mags:
        return {"flow_mag_mean": None, "flow_mag_p10": None, "n_pairs": 0}
    m = np.asarray(mags, np.float64)
    return {"flow_mag_mean": float(m.mean()), "flow_mag_p10": float(np.percentile(m, 10)),
            "flow_mag_p90": float(np.percentile(m, 90)), "n_pairs": int(m.size)}


# ------------------------------------------------- 2. appearance/colour drift
def colour_drift(bgr):
    """Warp t+1 back to t with estimated flow; residual in CIELAB over
    well-tracked pixels.

    Targets the reported "gloves change colour" failure: a region whose
    geometry tracks fine but whose colour does not.

    Confidence: forward-backward flow consistency.  A pixel counts as
    well-tracked when |flow_fwd(p) + flow_bwd(p + flow_fwd(p))| < fb_tol.
    Residual is the CIELAB Euclidean distance (cv2's L in 0..255, a/b offset
    by 128) between the warped t+1 and the true t.
    """
    FB_TOL = 1.0
    MIN_TRACKED = 0.02          # need >=2% of pixels to report a frame pair

    h, w = bgr[0].shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    per_pair_mean, per_pair_p95, tracked_fracs = [], [], []
    geom_resid = []
    for a, b in zip(bgr[:-1], bgr[1:]):
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        fwd = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        bwd = cv2.calcOpticalFlowFarneback(gb, ga, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # sample the backward flow at the forward-displaced position
        mx = np.clip(gx + fwd[..., 0], 0, w - 1)
        my = np.clip(gy + fwd[..., 1], 0, h - 1)
        bwd_at = cv2.remap(bwd, mx, my, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        fb = np.linalg.norm(fwd + bwd_at, axis=-1)
        tracked = fb < FB_TOL
        tf = float(tracked.mean())
        tracked_fracs.append(tf)
        if tf < MIN_TRACKED:
            continue

        warped = cv2.remap(b, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        lab_a = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_w = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
        d = np.linalg.norm(lab_a - lab_w, axis=-1)[tracked]
        if d.size == 0:
            continue
        per_pair_mean.append(float(d.mean()))
        per_pair_p95.append(float(np.percentile(d, 95)))

        # Geometry stability, separated from colour: the same warp residual but
        # on structural (Sobel gradient magnitude) content, so a clip can be
        # colour-unstable while geometrically stable, or vice versa.
        sa = cv2.magnitude(cv2.Sobel(ga, cv2.CV_32F, 1, 0, ksize=3),
                           cv2.Sobel(ga, cv2.CV_32F, 0, 1, ksize=3))
        sw = cv2.magnitude(cv2.Sobel(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 0, ksize=3),
                           cv2.Sobel(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 0, 1, ksize=3))
        gd = np.abs(sa - sw)[tracked]
        if gd.size:
            geom_resid.append(float(gd.mean()))

    if not per_pair_mean:
        return {"colour_drift_mean": None, "colour_drift_p95": None,
                "geom_residual": None,
                "tracked_frac_mean": (float(np.mean(tracked_fracs)) if tracked_fracs else None),
                "n_pairs": 0}
    pm = np.asarray(per_pair_mean)
    pp = np.asarray(per_pair_p95)
    return {
        "colour_drift_mean": float(pm.mean()),
        "colour_drift_mean_p95overframes": float(np.percentile(pm, 95)),
        "colour_drift_p95": float(pp.mean()),
        "colour_drift_p95_p95overframes": float(np.percentile(pp, 95)),
        "geom_residual": (float(np.mean(geom_resid)) if geom_resid else None),
        "tracked_frac_mean": float(np.mean(tracked_fracs)),
        "n_pairs": int(pm.size),
    }


# --------------------------------------------- 3. HF energy / flicker, 4. detail
def hf_and_detail(gray):
    """High-frequency energy (grain proxy) and Laplacian detail."""
    hf, lap, lum = [], [], []
    for g in gray:
        g8 = g if g.dtype == np.uint8 else g.astype(np.uint8)
        blur = cv2.GaussianBlur(g8, (0, 0), 1.0)
        hf.append(float(((g8.astype(np.float32) - blur.astype(np.float32)) ** 2).mean()))
        lap.append(float(cv2.Laplacian(g8, cv2.CV_64F).var()))
        lum.append(float(g8.mean()))
    if not hf:
        return {}
    hf = np.asarray(hf) / 255.0 ** 2
    lap = np.asarray(lap) / 255.0 ** 2
    lum = np.asarray(lum)
    return {
        "hf_energy": float(hf.mean()),
        "hf_flicker": float(hf.std() / (hf.mean() + 1e-12)),
        "detail_laplacian": float(lap.mean()),
        "luma_flicker": float(np.abs(np.diff(lum)).mean() / (lum.mean() + 1e-9)),
    }


# ------------------------------------------------- 6. crude subject permanence
def subject_permanence(bgr, gray):
    """CRUDE extra-performer detector.

    How crude: a temporal-median background is subtracted per clip, the
    difference is thresholded, morphologically cleaned, and connected
    components above 1% of the frame area are counted.  That counts LARGE
    MOVING BLOBS, not people -- a fast camera arc, a lighting change, or a
    large prop all register.  It is a birth/death alarm, not an identity
    tracker, and it cannot tell an extra performer from a chair.

    Reported per clip: the modal blob count, the number of frames whose count
    exceeds the mode, the number of count transitions, and (when OpenCV ships
    a Haar cascade) the maximum face count over sampled frames.
    """
    if len(gray) < 8:
        return {"blob_count_mode": None, "blob_excess_frames": None,
                "blob_transitions": None, "face_count_max": None}

    stack = np.stack([g.astype(np.float32) for g in gray])
    bg = np.median(stack, axis=0)
    h, w = bg.shape
    min_area = 0.01 * h * w
    area_full = float(h * w)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    counts, largest = [], []
    for g in stack:
        diff = np.abs(g - bg)
        thr = np.clip(diff.std() * 2.5, 8.0, 60.0)
        mask = (diff > thr).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        n, _lab, stats, _c = cv2.connectedComponentsWithStats(mask, 8)
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n)
                 if stats[i, cv2.CC_STAT_AREA] >= min_area]
        counts.append(len(areas))
        largest.append(max(areas) / area_full if areas else 0.0)
    counts = np.asarray(counts)
    largest = np.asarray(largest)
    vals, freqs = np.unique(counts, return_counts=True)
    mode = int(vals[int(np.argmax(freqs))])
    excess = int((counts > mode).sum())
    transitions = int((np.diff(counts) != 0).sum())

    face_max = None
    face_note = "not attempted"
    try:
        if not hasattr(cv2, "CascadeClassifier"):
            face_note = ("UNAVAILABLE: this OpenCV build (cv2 %s) has no CascadeClassifier, "
                         "so the Haar face count could not be run" % cv2.__version__)
        else:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if cascade.empty():
                face_note = "UNAVAILABLE: the Haar frontal-face cascade failed to load"
            else:
                idx = np.linspace(0, len(bgr) - 1, min(25, len(bgr))).astype(int)
                fc = 0
                for i in idx:
                    det = cascade.detectMultiScale(bgr[i], 1.2, 5,
                                                   minSize=(int(0.04 * w), int(0.04 * w)))
                    fc = max(fc, len(det))
                face_max = int(fc)
                face_note = "ok (max faces over 25 sampled frames)"
    except Exception as exc:
        face_max = None
        face_note = f"UNAVAILABLE: {type(exc).__name__}: {exc}"

    return {
        "blob_count_mode": mode,
        "blob_count_max": int(counts.max()),
        "blob_excess_frames": excess,
        "blob_excess_frac": float(excess / max(1, counts.size)),
        "blob_transitions": transitions,
        "blob_transitions_per_100f": float(100.0 * transitions / max(1, counts.size)),
        "subject_persistence_frac": float((counts == mode).mean()),
        "largest_blob_area_frac_mean": float(largest.mean()),
        "largest_blob_area_frac_cv": float(largest.std() / (largest.mean() + 1e-9)),
        "blob_counts_hist": {str(int(v)): int(c) for v, c in zip(vals, freqs)},
        "face_count_max": face_max,
        "face_count_note": face_note,
        "min_blob_area_frac": 0.01,
        "note": "counts large moving blobs, not people; births/deaths are an alarm, not identity",
        "frame_area_px": area_full,
    }


# ---------------------------------------------------------------- per clip
def measure_clip(path, scorer=None):
    bgr, gray = read_video(path)
    rec = {"video": str(path), "n_frames_read": len(bgr)}
    if not bgr:
        rec["error"] = "no frames decoded"
        return rec

    rec.update(motion_magnitude(gray))
    rec.update(colour_drift(bgr))
    rec.update(hf_and_detail(gray))
    rec.update(subject_permanence(bgr, gray))

    small = np.stack([cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                      for g in gray])
    if scorer is not None:
        try:
            rec["scorer_temporal"] = scorer.temporal_metrics(small)
        except Exception as e:
            rec["scorer_temporal_error"] = f"{type(e).__name__}: {e}"
        try:
            rec["scorer_visual"] = scorer.visual_metrics(small, path)
        except Exception as e:
            rec["scorer_visual_error"] = f"{type(e).__name__}: {e}"
        try:
            rec["scorer_flow"] = scorer.flow_metrics(path)
        except Exception as e:
            rec["scorer_flow_error"] = f"{type(e).__name__}: {e}"
        try:
            rec["scorer_audio_health"] = scorer.audio_health(path)
        except Exception as e:
            rec["scorer_audio_health_error"] = f"{type(e).__name__}: {e}"
    return rec


def measure_rank(rank_dir: Path, scorer, cases: dict, prompt_ids: list[str]) -> dict:
    out = {"rank": None, "clips": [], "errors": []}
    results_path = rank_dir / "results.json"
    if results_path.is_file():
        try:
            out["rank"] = json.loads(results_path.read_text()).get("rank")
        except Exception:
            pass

    videos = sorted(rank_dir.glob("*.mp4"))
    for v in videos:
        cid = v.stem.split("__seed")[0]
        rec = measure_clip(v, scorer)
        rec["case"] = cid
        rec["seed"] = v.stem.split("__seed")[1] if "__seed" in v.stem else None
        rec["width"] = cases.get(cid, {}).get("width")
        rec["height"] = cases.get(cid, {}).get("height")
        # The decoded length, not the requested one: four hard-motion cases ask
        # for 362 frames but the model caps at 345 (17k+5 and <=15s at 24fps).
        req = cases.get(cid, {}).get("frames")
        rec["frames_requested"] = req
        rec["frames"] = rec.get("n_frames_read")
        rec["frames_expected"] = clamp_num_frames(req) if req else None
        rec["integrity_ok"] = (rec.get("n_frames_read") == rec["frames_expected"])
        if not rec["integrity_ok"]:
            print(f"[measure] INTEGRITY: {v.name} decoded {rec.get('n_frames_read')} frames, "
                  f"expected {rec['frames_expected']} -- excluded from the aggregate",
                  flush=True)
        out["clips"].append(rec)

    # scorer audio_semantics + prompt_alignment need the prompt text
    for rec in out["clips"]:
        cid = rec["case"]
        ptext = cases.get(cid, {}).get("prompt")
        if not ptext:
            continue
        try:
            rec["scorer_clap_alignment"] = scorer.clap_alignment(rec["video"], ptext)
        except Exception as e:
            rec["scorer_clap_alignment_error"] = f"{type(e).__name__}: {e}"
        try:
            rec["scorer_clip_alignment"] = scorer.clip_alignment(rec["video"], ptext)
        except Exception as e:
            rec["scorer_clip_alignment_error"] = f"{type(e).__name__}: {e}"
    return out


def aggregate(rank_payload: dict) -> dict:
    """Mean over clips of every float metric, plus the per-clip values.

    Nested one level: the reused scorer families come back as dicts
    (scorer_flow, scorer_temporal, scorer_visual, scorer_audio_health), and
    compare_ranks.py reads them as agg[group][sub]["mean"], so those must be
    aggregated too -- a top-level-only pass leaves the whole scorer table empty.
    """
    flat = {}
    nested = {}
    for c in rank_payload["clips"]:
        if c.get("integrity_ok") is False:
            continue
        for k, v in c.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                flat.setdefault(k, []).append(float(v))
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, (int, float)) and not isinstance(v2, bool):
                        nested.setdefault(k, {}).setdefault(k2, []).append(float(v2))

    agg = {}
    for k, vals in flat.items():
        arr = np.asarray(vals, np.float64)
        agg[k] = {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}
    for g, subs in nested.items():
        agg[g] = {s: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                  for s, v in subs.items()}
    rank_payload["aggregate"] = agg
    return rank_payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path,
                    default=Path(SPRINT_ROOT) / "adaln_rank_analysis/sweep")
    ap.add_argument("--cases", type=Path,
                    default=Path(SPRINT_ROOT) / "hardmotion_set.json")
    ap.add_argument("--ranks", default=None)
    args = ap.parse_args()

    cases = {c["id"]: c for c in json.loads(args.cases.read_text())}
    wanted = ([int(x) for x in args.ranks.split(",")] if args.ranks
              else sorted(int(p.name.split("_r")[1]) for p in args.sweep_dir.glob("sweep_r*")))

    scorer = _load_scorer()
    print(f"[measure] scores families: {sorted(scorer.FAMILIES)}", flush=True)
    print(f"[measure] ranks: {wanted}", flush=True)

    for rank in wanted:
        d = args.sweep_dir / f"sweep_r{rank}"
        if not d.is_dir():
            print(f"[measure] rank {rank}: no directory {d}", flush=True)
            continue
        payload = measure_rank(d, scorer, cases, [])
        payload["rank"] = rank
        payload = aggregate(payload)
        out = d / "metrics.json"
        out.write_text(json.dumps(payload, indent=2, default=float))
        n = len(payload["clips"])
        a = payload["aggregate"]
        print(f"[measure] rank {rank}: {n} clips -> {out.name}  "
              f"motion={a.get('flow_mag_mean', {}).get('mean')}  "
              f"drift={a.get('colour_drift_mean', {}).get('mean')}  "
              f"hf={a.get('hf_energy', {}).get('mean')}  "
              f"detail={a.get('detail_laplacian', {}).get('mean')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
