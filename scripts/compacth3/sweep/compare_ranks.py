from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(20260917)

# metric -> (label, better, family)
#   better: "higher" | "lower" | "none" (a diagnostic, no direction)
METRICS = [
    ("flow_mag_mean",              "motion magnitude (flow)",        "higher", "motion"),
    ("flow_mag_p10",               "motion magnitude p10",           "higher", "motion"),
    ("colour_drift_mean",          "colour drift mean (CIELAB)",     "lower",  "appearance"),
    ("colour_drift_p95",           "colour drift p95",               "lower",  "appearance"),
    ("geom_residual",              "geometry warp residual",         "lower",  "appearance"),
    ("tracked_frac_mean",          "tracked fraction",               "higher", "appearance"),
    ("hf_energy",                  "HF energy (grain)",              "none",   "temporal"),
    ("hf_flicker",                 "HF flicker",                     "lower",  "temporal"),
    ("detail_laplacian",           "detail (Laplacian var)",         "none",   "visual"),
    ("blob_excess_frac",           "extra-blob frames fraction",     "lower",  "permanence"),
    ("blob_transitions_per_100f",  "blob births/deaths per 100f",    "lower",  "permanence"),
    ("subject_persistence_frac",   "subject persistence fraction",   "higher", "permanence"),
    ("largest_blob_area_frac_cv",  "largest-blob area CV",           "lower",  "permanence"),
    ("face_count_max",             "face count max (Haar)",          "none",   "permanence"),
]

# Metrics whose verdict must be read against the motion delta: an "improvement"
# here that comes with reduced motion is a collapse, not a win.
ANOMALY_METRICS = {
    "colour_drift_mean", "colour_drift_p95", "geom_residual", "hf_flicker",
    "blob_excess_frac", "blob_transitions_per_100f", "largest_blob_area_frac_cv",
}

SENTINEL_CASE = "t2va-0020260818-008164"

SCORER_KEYS = [
    ("scorer_flow", "flow_magnitude", "motion magnitude (scorer)"),
    ("scorer_flow", "flow_jerk", "flow jerk (scorer)"),
    ("scorer_flow", "flow_direction_instability", "flow dir instability (scorer)"),
    ("scorer_temporal", "motion", "temporal motion (scorer)"),
    ("scorer_temporal", "jerk", "jerk (scorer)"),
    ("scorer_temporal", "flicker", "flicker (scorer)"),
    ("scorer_temporal", "luma_flicker", "luma flicker (scorer)"),
    ("scorer_temporal", "freeze_fraction", "freeze fraction (scorer)"),
    ("scorer_visual", "luma", "luma (scorer)"),
    ("scorer_visual", "contrast", "contrast (scorer)"),
    ("scorer_visual", "tonal_range", "tonal range (scorer)"),
    ("scorer_visual", "sharpness", "sharpness (scorer)"),
    ("scorer_visual", "edge_density", "edge density (scorer)"),
    ("scorer_visual", "black_frame_fraction", "black frame fraction (scorer)"),
    ("scorer_visual", "static_collapse", "static collapse (scorer)"),
    ("scorer_visual", "learned_visual", "learned visual Q-Align/DOVER (scorer)"),
    ("scorer_audio_health", "lufs", "loudness LUFS (scorer)"),
    ("scorer_audio_health", "true_peak_db", "true peak dB (scorer)"),
    ("scorer_audio_health", "silence_ratio", "silence ratio (scorer)"),
    ("scorer_audio_health", "clipping_ratio", "clipping ratio (scorer)"),
    ("scorer_audio_health", "spectral_flatness", "spectral flatness (scorer)"),
    ("scorer_audio_health", "hf_excess_ratio", "HF excess ratio (scorer)"),
]


def get(clip: dict, key: str, sub: str | None = None):
    if sub is None:
        v = clip.get(key)
    else:
        v = (clip.get(key) or {}).get(sub)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _scorer_mean(payloads, aggs, rank, group, sub):
    """Aggregate mean for a reused-scorer sub-metric, with a clips fallback.

    The primary source is metrics.json's aggregate; the fallback recomputes from
    the per-clip records so a missing aggregate block cannot silently empty the
    whole scorer table.
    """
    e = (aggs.get(rank, {}).get(group) or {})
    v = e.get(sub)
    if isinstance(v, dict) and "mean" in v:
        return v["mean"]
    vals = [get(c, group, sub) for c in payloads.get(rank, {}).get("clips", [])]
    vals = [x for x in vals if x is not None]
    return float(np.mean(vals)) if vals else None


def _metric_value(clip, key):
    if key in ("clap_alignment", "prompt_alignment"):
        return None
    if key.startswith("scorer_"):
        return None
    return get(clip, key)


def bootstrap_ci(diffs, n=4000):
    d = np.asarray(diffs, dtype=np.float64)
    if d.size == 0:
        return None, None, None
    idx = RNG.integers(0, d.size, size=(n, d.size))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


MIN_ABS_DZ = 0.8        # paired Cohen's d_z
MIN_REL_EFFECT = 0.01   # 1% of the r16 mean


def effect_stats(diffs, ref_mean):
    d = np.asarray(diffs, dtype=np.float64)
    if d.size < 2:
        return None, None
    sd = float(d.std(ddof=1))
    dz = float(d.mean() / sd) if sd > 0 else float("inf") * (1 if d.mean() > 0 else -1)
    rel = (abs(float(d.mean())) / abs(ref_mean)) if ref_mean else None
    return dz, rel


def sign_test(diffs):
    d = np.asarray([x for x in diffs if x != 0], dtype=np.float64)
    if d.size == 0:
        return None, 0, 0
    pos = int((d > 0).sum())
    n = int(d.size)
    # two-sided exact binomial p under p=0.5
    from math import comb
    k = min(pos, n - pos)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return float(min(1.0, p)), pos, n


def family_table(rank_agg, keys):
    rows = []
    for key, label, better, family in keys:
        e = rank_agg.get(key)
        rows.append((family, label, key, (e["mean"] if e else None),
                     (e["std"] if e else None), better))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path,
                    default=Path(SPRINT_ROOT) / "adaln_rank_analysis/sweep")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    payloads, aggs, clips = {}, {}, {}
    for d in sorted(args.sweep_dir.glob("sweep_r*")):
        rank = int(d.name.split("_r")[1])
        m = d / "metrics.json"
        if not m.is_file():
            print(f"[compare] rank {rank}: missing {m}")
            continue
        p = json.loads(m.read_text())
        payloads[rank] = p
        aggs[rank] = p.get("aggregate", {})
        clips[rank] = {(c["case"], str(c["seed"])): c for c in p.get("clips", [])}

    if not payloads:
        print("[compare] no metrics found")
        return 1
    ranks = sorted(payloads)

    # ---- fold sizes (params / GB) -------------------------------------
    fold_meta = {}
    fm = args.sweep_dir / "folds" / "fold_meta.json"
    if fm.is_file():
        fold_meta = json.loads(fm.read_text()).get("folds", {})

    lines = []
    A = lines.append
    A("# AdaLN rank sweep: behavioral comparison")
    A("")
    A("Model: 4-call DMD2 checkpoint-1400.  Ladder 999/749/500/250.")
    A("Ranks present: " + ", ".join(str(r) for r in ranks))
    A("")
    A("## Per-rank summary")
    A("")
    A("| rank | AdaLN params | AdaLN GB fp16 | total params | total GB fp16 | "
      "motion mag | drift mean | drift p95 | HF energy | detail |")
    A("|---|---|---|---|---|---|---|---|---|---|")

    def a(rank, key):
        e = aggs.get(rank, {}).get(key)
        return e["mean"] if e else None

    # Proven constants from rank_compression_dmd2.json for this checkpoint.
    NON_ADALN_PARAMS = 17_000_721_664
    for r in ranks:
        f = fold_meta.get(str(r), {})
        params = f.get("adaln_params")
        gb = f.get("file_gb")
        tot = (params + NON_ADALN_PARAMS) if params is not None else None
        tot_gb = (tot * 2 / 1e9) if tot is not None else None
        A(f"| {r} | {f'{params:,}' if params is not None else '?'} | "
          f"{f'{gb:.3f}' if gb is not None else '?'} | "
          f"{f'{tot:,}' if tot is not None else '?'} | "
          f"{f'{tot_gb:.2f}' if tot_gb is not None else '?'} | "
          f"{_fmt(a(r,'flow_mag_mean'))} | {_fmt(a(r,'colour_drift_mean'))} | "
          f"{_fmt(a(r,'colour_drift_p95'))} | {_fmt(a(r,'hf_energy'))} | "
          f"{_fmt(a(r,'detail_laplacian'))} |")
    A("")
    A("AdaLN and total are parameter counts of the DiT; GB assumes fp16 AdaLN against a bf16 backbone.")
    A("")

    # ---- scorer families per rank -------------------------------------
    A("## Scorer families (reused from the repo scorer)")
    A("")
    A("The scorer has NO speech and NO anatomy family; neither is measured here.")
    A("")
    hdr = "| metric | " + " | ".join(f"r{r}" for r in ranks) + " |"
    A(hdr)
    A("|---" * (len(ranks) + 1) + "|")
    for group, sub, label in SCORER_KEYS:
        vals = [_scorer_mean(payloads, aggs, r, group, sub) for r in ranks]
        if all(v is None for v in vals):
            continue
        A(f"| {label} | " + " | ".join(_fmt(v) for v in vals) + " |")
    for key, label in (("scorer_clap_alignment", "CLAP audio-semantics alignment"),
                       ("scorer_clip_alignment", "CLIP prompt alignment")):
        vals = []
        for r in ranks:
            vs = [get(c, key) for c in payloads[r].get("clips", [])]
            vs = [v for v in vs if v is not None]
            vals.append(float(np.mean(vs)) if vs else None)
        if all(v is None for v in vals):
            continue
        A(f"| {label} | " + " | ".join(_fmt(v) for v in vals) + " |")
    A("")

    # ---- matched-pair comparison --------------------------------------
    A("## Matched (case, seed) comparison")
    A("")
    pairs = sorted(set(clips.get(ranks[0], {})) & set(clips.get(ranks[-1], {}))
                   if len(ranks) >= 2 else set())
    A(f"matched clips available: {len(pairs)}")
    A("")

    def paired(ra, rb, key, sub=None):
        diffs, ids = [], []
        for cid in pairs:
            ca, cb = clips[ra].get(cid), clips[rb].get(cid)
            if ca is None or cb is None:
                continue
            va = get(ca, key, sub)
            vb = get(cb, key, sub)
            if va is None or vb is None:
                continue
            diffs.append(va - vb)
            ids.append(cid)
        return diffs, ids

    for pair in _interesting_pairs(ranks):
        ra, rb = pair
        A(f"### r{ra} - r{rb}  (positive = r{ra} larger)")
        A("")
        A("Motion delta is carried on EVERY row so a motion-collapse \"win\" is visible on its "
          "face: `motionΔ` is the same matched-pair difference for flow_mag_mean.")
        A("")
        A("| metric | better | mean diff | 95% CI | motionΔ | anomalyΔ/|motionΔ| | sign test p | n | per-seed sign | verdict |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        mdiffs, _ = paired(ra, rb, "flow_mag_mean")
        motion_delta = float(np.mean(mdiffs)) if mdiffs else None
        for key, label, better, family in METRICS:
            diffs, ids = paired(ra, rb, key)
            if not diffs:
                continue
            m, lo, hi = bootstrap_ci(diffs)
            p, pos, n = sign_test(diffs)
            agree = _seed_agreement(diffs, ids, clips[ra])
            ratio = None
            if key in ANOMALY_METRICS and m is not None and motion_delta:
                ratio = abs(m) / abs(motion_delta)
            verdict = "inconclusive"
            if m is not None and lo is not None and (lo > 0 or hi < 0):
                if better == "none":
                    verdict = "differs (no quality direction)"
                else:
                    favours_a = (m > 0) if better == "higher" else (m < 0)
                    verdict = f"favours r{ra}" if favours_a else f"favours r{rb}"
                    if key in ANOMALY_METRICS and favours_a and motion_delta is not None \
                            and ((motion_delta < 0) != (better == "higher")) :
                        verdict += " [MOTION-SUSPECT]"
            A(f"| {label} | {better} | {_fmt(m)} | [{_fmt(lo)}, {_fmt(hi)}] | "
              f"{_fmt(motion_delta)} | {_fmt(ratio)} | "
              f"{_fmt(p)} | {len(diffs)} | {agree} | {verdict} |")
        A("")

    # ---- permanence / stability, aggregate over the whole set ----
    A("## Permanence and stability: aggregate over the 14-case set")
    A("")
    A(f"`{SENTINEL_CASE}` is the unambiguous sentinel (its prompt forbids extra "
      "performers) and is reported separately below.  The VERDICT comes from the "
      "aggregate over all cases, not from the sentinel.")
    A("")
    A("| metric | " + " | ".join(f"r{r}" for r in ranks) + " |")
    A("|---" * (len(ranks) + 1) + "|")
    for key, label, better, family in METRICS:
        if family not in ("permanence", "appearance"):
            continue
        vals = [a(r, key) for r in ranks]
        if all(v is None for v in vals):
            continue
        A(f"| {label} (all cases) | " + " | ".join(_fmt(v) for v in vals) + " |")
    for key, label, better, family in METRICS:
        if family not in ("permanence",):
            continue
        vals = []
        for r in ranks:
            c = clips.get(r, {}).get((SENTINEL_CASE, sorted({s for _, s in clips.get(r, {})})[0])) \
                if clips.get(r) else None
            v = get(c, key) if c else None
            vals.append(v)
        if all(v is None for v in vals):
            continue
        A(f"| {label} (sentinel {SENTINEL_CASE[-6:]}) | " + " | ".join(_fmt(v) for v in vals) + " |")
    A("")

    # ---- decision -----------------------------------------------------
    A("## Decision input")
    A("")
    A("Rule, stated in advance:")
    A("")
    A("```")
    A("r16 ~= r64                              -> prefer r16")
    A("r64 repeatably better AND equal motion  -> prefer r64")
    A("```")
    A("")
    A("\"Repeatably\" = the matched-pair CI excludes zero, the sign agrees across every")
    A("seed used, AND the direction agrees between the paired test and the whole-set")
    A("aggregate.  Motion delta is carried alongside every anomaly delta: a rank showing")
    A("fewer anomalies with reduced motion magnitude has NOT won.")
    A("")
    if 16 in ranks and 64 in ranks:
        A(_decision(payloads, clips, aggs, 16, 64, ranks))
    else:
        A("r16 and r64 are both required for the decision; present ranks: "
          + ", ".join(str(r) for r in ranks))
    A("")

    # ---- motion guard -------------------------------------------------
    A("## Anti-collapse guard (motion magnitude)")
    A("")
    A("Mean optical-flow magnitude per clip -- the mandatory check that no rank "
      "\"wins\" by producing near-static video.")
    A("")
    A("| rank | mean flow mag | p10 | p90 | freeze fraction | vs r768 | flag |")
    A("|---|---|---|---|---|---|---|")
    base = a(768, "flow_mag_mean") if 768 in ranks else None
    if base is None:
        vals = [a(r, "flow_mag_mean") for r in ranks if a(r, "flow_mag_mean") is not None]
        base = max(vals) if vals else None
    flagged_motion = []
    for r in ranks:
        mm = a(r, "flow_mag_mean")
        p10 = a(r, "flow_mag_p10")
        p90 = a(r, "flow_mag_p90")
        ff = ((aggs.get(r, {}) or {}).get("scorer_temporal") or {}).get("freeze_fraction", {})
        ff = ff.get("mean") if isinstance(ff, dict) else None
        ratio = (mm / base) if (mm is not None and base) else None
        flag = ""
        if ratio is not None and ratio < 0.9:
            flag = f"**REDUCED MOTION ({ratio:.2f}x)**"
            flagged_motion.append(r)
        A(f"| r{r} | {_fmt(mm)} | {_fmt(p10)} | {_fmt(p90)} | {_fmt(ff)} | "
          f"{_fmt(ratio)} | {flag} |")
    A("")
    if flagged_motion:
        A(f"**Flagged for reduced motion vs r{768 if 768 in ranks else 'the best rank'}: "
          + ", ".join(f"r{x}" for x in flagged_motion)
          + ". Any apparent quality win for these ranks is suspect and must be read "
            "against this.**")
    else:
        A("No rank shows a >=10% reduction in motion magnitude.")
    A("")

    text = "\n".join(lines) + "\n"
    out = args.out or (args.sweep_dir / "RANK_SWEEP_REPORT.md")
    out.write_text(text)
    print(text)
    print(f"[compare] wrote {out}")
    return 0


def _interesting_pairs(ranks):
    out = []
    for a in ranks:
        for b in ranks:
            if a > b:
                out.append((a, b))
    return out


def _seed_agreement(diffs, ids, clip_map):
    by_seed = {}
    for d, cid in zip(diffs, ids):
        by_seed.setdefault(cid[1], []).append(d)
    parts = []
    for s, ds in sorted(by_seed.items()):
        mean = float(np.mean(ds))
        parts.append(f"seed{s}:{'+' if mean > 0 else '-' if mean < 0 else '0'}")
    return " ".join(parts)


def _decision(payloads, clips, aggs, r16, r64, ranks):
    pairs = sorted(set(clips[r16]) & set(clips[r64]))
    out = [f"matched pairs r16 vs r64: {len(pairs)}", ""]

    def paired(key, sub=None):
        diffs, ids = [], []
        for cid in pairs:
            ca, cb = clips[r64].get(cid), clips[r16].get(cid)
            if ca is None or cb is None:
                continue
            va, vb = get(ca, key, sub), get(cb, key, sub)
            if va is None or vb is None:
                continue
            diffs.append(va - vb)
            ids.append(cid)
        return diffs, ids

    # ---- motion guard, first and loudest ------------------------------
    m16 = aggs.get(r16, {}).get("flow_mag_mean", {}).get("mean")
    m64 = aggs.get(r64, {}).get("flow_mag_mean", {}).get("mean")
    mdiffs, mids = paired("flow_mag_mean")
    md = float(np.mean(mdiffs)) if mdiffs else None
    mlo = mhi = None
    if mdiffs:
        _, mlo, mhi = bootstrap_ci(mdiffs)
    out.append("### Motion guard (checked first)")
    out.append("")
    out.append(f"- aggregate motion magnitude: r16={_fmt(m16)}  r64={_fmt(m64)}  "
               f"ratio r64/r16={_fmt((m64 / m16) if (m16 and m64) else None)}")
    out.append(f"- matched-pair motion delta (r64 - r16): {_fmt(md)} "
               f"CI [{_fmt(mlo)}, {_fmt(mhi)}] over {len(mdiffs)} pairs")
    motion_collapse = bool(mlo is not None and mhi < 0)
    if motion_collapse:
        out.append("")
        out.append("**MOTION COLLAPSE: r64 has significantly LESS motion than r16 on the "
                   "matched pairs. Under the rule stated above, r64 cannot win on reduced "
                   "anomalies while motion is down.**")
    else:
        out.append("")
        out.append("Motion is statistically EQUAL or higher for r64, so anomaly deltas below "
                   "are admissible as wins.")
    out.append("")

    # ---- per-metric, with motion delta attached ----------------------
    n_seeds = len({s for _c, s in pairs})
    out.append("### Anomaly deltas with motion delta attached")
    out.append("")
    out.append(f"matched pairs: {len(pairs)} over {n_seeds} seed(s). "
               + ("With a single seed the 'agrees across seeds' test is vacuous, so the "
                  "admissibility column additionally requires |d_z| >= "
                  f"{MIN_ABS_DZ} and a relative effect >= {MIN_REL_EFFECT:.0%}."
                  if n_seeds < 2 else
                  f"Repeatability is tested across {n_seeds} seeds."))
    out.append("")
    out.append("| metric | mean diff | 95% CI | d_z | rel | motionΔ | anomalyΔ/|motionΔ| | "
               "favours | seed agreement | admissible as r64 win |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    robust = []
    for key, label, better, family in METRICS:
        if better == "none":
            continue
        diffs, ids = paired(key)
        if not diffs:
            continue
        m, lo, hi = bootstrap_ci(diffs)
        if lo is None:
            continue
        ci_excludes_zero = (lo > 0) or (hi < 0)
        favours_64 = (m > 0) if better == "higher" else (m < 0)

        ref16 = aggs.get(r16, {}).get(key, {}).get("mean")
        dz, rel = effect_stats(diffs, ref16)

        by_seed = {}
        for d, cid in zip(diffs, ids):
            by_seed.setdefault(cid[1], []).append(d)
        signs = {s: float(np.mean(ds)) for s, ds in by_seed.items()}
        agree = all((v > 0) == favours_64 for v in signs.values()) if signs else False

        # whole-set aggregate direction must agree with the paired direction
        e16 = ref16
        e64 = aggs.get(r64, {}).get(key, {}).get("mean")
        agg_agrees = None
        if e16 is not None and e64 is not None and e64 != e16:
            agg_agrees = ((e64 > e16) == favours_64)

        ratio = abs(m) / abs(md) if (md and key in ANOMALY_METRICS) else None
        big_enough = (dz is not None and abs(dz) >= MIN_ABS_DZ
                      and rel is not None and rel >= MIN_REL_EFFECT)
        admissible = bool(ci_excludes_zero and favours_64 and agree
                          and agg_agrees is not False and big_enough
                          and not (key in ANOMALY_METRICS and motion_collapse))
        out.append(f"| {label} | {_fmt(m)} | [{_fmt(lo)},{_fmt(hi)}] | {_fmt(dz)} | "
                   f"{_fmt(rel)} | {_fmt(md)} | {_fmt(ratio)} | "
                   f"{'r64' if favours_64 else 'r16'} | {agree} | "
                   f"{'YES' if admissible else 'no'} |")
        if admissible:
            robust.append(label)
    out.append("")

    # ---- permanence aggregate, sentinel separated --------------------
    out.append("### Permanence: aggregate across the 14-case set vs the sentinel alone")
    out.append("")
    for key in ("blob_excess_frac", "blob_transitions_per_100f",
                "subject_persistence_frac", "largest_blob_area_frac_cv"):
        e16 = aggs.get(r16, {}).get(key, {}).get("mean")
        e64 = aggs.get(r64, {}).get(key, {}).get("mean")
        diffs, _ = paired(key)
        s16 = _sentinel_value(clips.get(r16), key)
        s64 = _sentinel_value(clips.get(r64), key)
        out.append(f"- {key}: aggregate r16={_fmt(e16)} r64={_fmt(e64)} "
                   f"(paired delta {_fmt(float(np.mean(diffs)) if diffs else None)}); "
                   f"sentinel {SENTINEL_CASE[-6:]} r16={_fmt(s16)} r64={_fmt(s64)}")
    out.append("")
    out.append("The permanence verdict below is taken from the AGGREGATE rows above, not from "
               "the sentinel case.")
    out.append("")

    if robust:
        out.append(f"**r64 is repeatably better on: {', '.join(robust)}** "
                   f"(CI excludes 0, seed-agreeing, aggregate direction agrees, "
                   f"|d_z| >= {MIN_ABS_DZ}, relative effect >= {MIN_REL_EFFECT:.0%}"
                   f"{', and motion is not reduced' if not motion_collapse else ''}).")
        if motion_collapse:
            out.append("")
            out.append("**BUT the motion guard fired: r64 has reduced motion. Per the rule "
                       "(`r64 repeatably better AND equal motion`), the motion half of the "
                       "conjunct FAILS, so r64 does NOT qualify on these metrics. "
                       "Prefer r16.**")
        else:
            out.append("")
            out.append("**r64 repeatably better AND motion equal -> per the decision rule, "
                       "prefer r64.**")
    else:
        out.append("**No metric shows a bootstrap-significant, seed-agreeing, "
                   "aggregate-confirmed, materially-sized r64 advantage that survives the "
                   "motion guard. Per the decision rule -> r16 and r64 are BEHAVIOURALLY "
                   "INDISTINGUISHABLE, so prefer r16.**")
    return "\n".join(out)


def _sentinel_value(rank_clips, key):
    if not rank_clips:
        return None
    for (cid, _seed), c in rank_clips.items():
        if cid == SENTINEL_CASE:
            return get(c, key)
    return None


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if v != 0 and abs(v) < 1e-3:
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


if __name__ == "__main__":
    raise SystemExit(main())
