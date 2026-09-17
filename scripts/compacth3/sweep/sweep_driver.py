from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

M = os.environ.get("COMPACTH3_CODE_ROOT",
                     str(Path(SPRINT_ROOT).parent / "fasth3-h3-serve-cookbook-eval-20260831" / "repo-main-3d8ac9d1"))
SPRINT = SPRINT_ROOT
SWEEP = Path(SPRINT) / "adaln_rank_analysis" / "sweep"


def _load_basic_fasth3():
    """Load the repo example without colliding with site-packages ``examples``."""
    path = Path(M) / "examples/inference/basic/basic_fasth3.py"
    spec = importlib.util.spec_from_file_location("sweep_basic_fasth3", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sweep_basic_fasth3"] = module
    spec.loader.exec_module(module)
    return module


FPS = 24
MIN_DURATION = 5.0
MAX_DURATION = 15.0
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5


def clamp_num_frames(requested: int) -> int:
    """Nearest VALID MiniMax-H3 frame count at or below the request.

    The pipeline accepts only num_frames of the form 17k+5 (align_num_frames in
    minimax_h3/packing.py) and additionally enforces a 5-15 s duration at 24 fps.
    Four cases in the hard-motion set ask for 362 frames, which is aligned but is
    15.083 s -- over the cap -- so they raise
    "MiniMax-H3 generates 5-15 seconds at 24 fps" and produce nothing.  The
    largest valid count is 17*20+5 = 345 (14.375 s), which is what those four
    become.  Every other case in the set is already valid and is returned
    unchanged.
    """
    n = int(requested)
    k = (n - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK
    if k < 0:
        k = 0
    cand = FRAMES_PER_CHUNK * k + LATENTS_PER_CHUNK
    while cand > MAX_DURATION * FPS and k > 0:
        k -= 1
        cand = FRAMES_PER_CHUNK * k + LATENTS_PER_CHUNK
    while cand < MIN_DURATION * FPS:
        k += 1
        cand = FRAMES_PER_CHUNK * k + LATENTS_PER_CHUNK
    return cand


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--cases", type=Path,
                    default=Path(SPRINT) / "hardmotion_set.json")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--seeds", required=True, help="comma separated")
    ap.add_argument("--num-gpus", type=int, default=4)
    ap.add_argument("--case-ids", default=None, help="comma separated subset")
    ap.add_argument("--video-decode-backend", default="h3-vae")
    ap.add_argument("--profile", default="strict", choices=("all", "strict"))
    ap.add_argument("--attention-backend", default="TORCH_SDPA")
    ap.add_argument("--ladder", default="999,749,500,250")
    return ap.parse_args()


def _inference_args(basic, args, model_path: str) -> argparse.Namespace:
    """Mirror the proven sprint lane's inference configuration exactly.

    attention: DENSE / TORCH_SDPA.  Every sprint log on this box shows
    "Selected backend: TORCH_SDPA" and nothing else -- the sm100a VSA kernel is
    not part of this venv -- and dense attention additionally removes the
    input-dependent tile-selection noise that could otherwise be mistaken for a
    rank effect.  `ia.vsa = False` makes _uses_vsa() false, which also skips the
    sm100a dependency check.
    """
    parser = basic.build_parser()
    ia = parser.parse_args([
        "--model-path", model_path,
        "--prompt", "placeholder",
        "--output", str(args.output_dir),
        "--profile", args.profile,
        "--height", "768",
        "--width", "768",
        "--num-frames", "124",
        "--steps", "5",
        "--seed", "0",
        "--num-gpus", str(args.num_gpus),
        "--repeats", "1",
        "--no-warmup",
        "--replicated-dit",
        "--parallel-vae",
        "--no-compile-vae",
        "--pin-cpu-memory",
        "--no-torch-compile",
        "--no-inference-torch-compile",
        "--no-fa4",
        "--no-lazy-module-load",
        "--no-h3-sequential-load",
        "--video-decode-backend", args.video_decode_backend,
    ])
    ia.vsa = False
    ia.attention = "dense"
    ia.attention_backend = args.attention_backend
    return basic.validate_args(parser, ia)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    basic = _load_basic_fasth3()

    # The ladder must be in the env before the denoising stage reads it.
    os.environ["FASTVIDEO_DMD_DENOISING_STEPS"] = args.ladder
    # rank < 0 is the unpatched reference arm: FASTVIDEO_ADALN_RANK is left
    # unset so the worker applies no fold at all.  Used only for the
    # end-to-end identity gate (patched r=768 vs unpatched must match).
    if args.rank < 0:
        os.environ.pop("FASTVIDEO_ADALN_RANK", None)
    else:
        os.environ["FASTVIDEO_ADALN_RANK"] = str(args.rank)

    ia = _inference_args(basic, args, args.model_path)
    environment = basic.configure_environment(ia)
    basic.validate_profile_dependencies(ia)

    cases = json.loads(Path(args.cases).read_text())
    if args.case_ids:
        wanted = {c.strip() for c in args.case_ids.split(",") if c.strip()}
        cases = [c for c in cases if c["id"] in wanted]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    clamped = {}
    for case in cases:
        want = int(case["frames"])
        got = clamp_num_frames(want)
        if got != want:
            clamped[case["id"]] = {"requested": want, "used": got}
    if clamped:
        print(f"[sweep] clamped frame counts (17k+5 and <=15s @24fps): {clamped}", flush=True)

    print(f"[sweep] rank={args.rank} cases={len(cases)} seeds={seeds} "
          f"gpus={args.num_gpus} profile={args.profile} decode={args.video_decode_backend} "
          f"attention={args.attention_backend}", flush=True)
    print(f"[sweep] ladder={args.ladder}  env={environment}", flush=True)

    config = basic.build_generator_config(ia)
    config.pipeline.experimental["attention_backend"] = args.attention_backend
    started = time.perf_counter()
    generator = basic.VideoGenerator.from_config(config)
    print(f"[sweep] generator ready in {time.perf_counter() - started:.1f}s", flush=True)

    results = []
    try:
        for case in cases:
            for seed in seeds:
                tag = f"{case['id']}__seed{seed}"
                out_path = args.output_dir / f"{tag}.mp4"
                if out_path.is_file() and out_path.stat().st_size > 0:
                    results.append({"case": case["id"], "seed": seed, "status": "skipped",
                                    "video": str(out_path)})
                    print(f"[sweep] skip existing {tag}", flush=True)
                    continue

                ia.prompt = case["prompt"]
                ia.height = int(case["height"])
                ia.width = int(case["width"])
                ia.num_frames = clamp_num_frames(int(case["frames"]))
                req = basic.build_request(ia, out_path, seed)
                req.output.output_path = str(out_path)

                t0 = time.perf_counter()
                try:
                    result = generator.generate(req)
                    wall = time.perf_counter() - t0
                    actual = getattr(result, "video_path", None) or str(out_path)
                    actual = Path(actual)
                    if actual != out_path and actual.is_file():
                        actual.replace(out_path)
                    ok = out_path.is_file() and out_path.stat().st_size > 0
                    results.append({
                        "case": case["id"], "seed": seed,
                        "status": "ok" if ok else "missing_output",
                        "wall_s": wall, "video": str(out_path),
                        "width": ia.width, "height": ia.height, "frames": ia.num_frames,
                    })
                    print(f"[sweep] {tag} wall={wall:.1f}s -> {out_path.name} ok={ok}", flush=True)
                except Exception as exc:  # keep the sweep going
                    wall = time.perf_counter() - t0
                    results.append({"case": case["id"], "seed": seed, "status": "error",
                                    "error": f"{type(exc).__name__}: {exc}", "wall_s": wall})
                    print(f"[sweep] {tag} FAILED after {wall:.1f}s: {type(exc).__name__}: {exc}",
                          flush=True)
                    import traceback
                    traceback.print_exc()
                finally:
                    (args.output_dir / "results.json").write_text(
                        json.dumps({"rank": args.rank, "seeds": seeds, "results": results}, indent=2))
    finally:
        try:
            generator.shutdown()
        except Exception:
            pass

    (args.output_dir / "results.json").write_text(
        json.dumps({"rank": args.rank, "seeds": seeds, "results": results}, indent=2))
    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"[sweep] DONE rank={args.rank} ok={n_ok}/{len(results)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
