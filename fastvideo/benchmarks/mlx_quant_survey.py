# SPDX-License-Identifier: Apache-2.0
"""Survey MLX quantization modes on real DiT weights — support, error, speed.

Answers three questions in one pass on an Apple Silicon machine, with **no
training and no GPU cluster**:

1. **Which modes does this MLX build actually run?** ``mx.quantize`` accepts
   ``affine``/``mxfp8``/``mxfp4``/``nvfp4``, but the newer mode strings need
   recent MLX and may raise.
2. **How much error does each mode introduce on the real weight
   distributions?** This is the cheap predictor of output quality. Ranking
   formats by reconstruction error on the actual checkpoint takes minutes and
   tells you far more than a week of distillation against the wrong format.
3. **Is the mode actually fast here?** A 4-bit mode that is not materially
   faster than int8 is being emulated rather than dispatched to the M5 neural
   accelerators — the memory win still stands, but the throughput case does not.

Usage::

    python -m fastvideo.benchmarks.mlx_quant_survey \\
        --checkpoint ~/models/FastWan2.1-T2V-1.3B-Diffusers/transformer \\
        --modes int8 int4 mxfp8 mxfp4 nvfp4

Requires ``mlx`` and ``safetensors``. Read-only; never writes weights.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

# Modes worth surveying, with the bits/param each costs including scale
# overhead. Used only for the memory column of the report.
MODE_BITS_PER_PARAM = {
    "fp16": 16.0,
    "int8": 8.5,  # 8 + (fp16 scale + fp16 bias) / 64
    "int4": 4.5,  # 4 + (fp16 scale + fp16 bias) / 64
    "mxfp8": 8.25,  # 8 + E8M0 scale / 32
    "mxfp4": 4.25,  # 4 + E8M0 scale / 32
    "nvfp4": 4.5,  # 4 + E4M3 scale / 16 (ignoring the per-tensor fp32 global)
}

# Weights the MLX loader never quantizes; skip them so the error numbers
# describe the layers that actually get quantized.
SKIP_SUBSTRINGS = ("norm", "scale_shift_table", "embedding")


def _spec_for(mode: str) -> dict[str, Any]:
    """Map a mode name onto mx.quantize kwargs."""
    if mode == "int8":
        return {"mode": "affine", "bits": 8, "group_size": 64}
    if mode == "int4":
        return {"mode": "affine", "bits": 4, "group_size": 64}
    if mode in ("mxfp8", "mxfp4"):
        return {"mode": mode, "group_size": 32}
    if mode == "nvfp4":
        return {"mode": "nvfp4", "group_size": 16}
    raise ValueError(f"Unknown mode {mode!r}")


def _roundtrip(mx, w, mode: str):
    """Quantize then dequantize, returning the reconstructed array."""
    spec = _spec_for(mode)
    q = mx.quantize(w, **spec)
    packed, scales = q[0], q[1]
    biases = q[2] if len(q) == 3 else None
    args = [packed, scales] + ([biases] if biases is not None else [])
    return mx.dequantize(*args, **spec)


def probe_modes(mx, modes: list[str]) -> dict[str, str | None]:
    """Return {mode: None if supported else error message}."""
    results: dict[str, str | None] = {}
    for mode in modes:
        try:
            w = mx.zeros((128, 128), dtype=mx.float16)
            mx.eval(_roundtrip(mx, w, mode))
            results[mode] = None
        except Exception as exc:  # noqa: BLE001 — MLX raises varied types per build.
            results[mode] = f"{type(exc).__name__}: {exc}"
    return results


def measure_error(mx, checkpoint: Path, modes: list[str], max_tensors: int) -> dict[str, dict[str, float]]:
    """Per-mode relative L2 and cosine error over real checkpoint weights.

    This is the highest-information-per-minute test in the survey. It ranks
    formats on the weight distributions that actually matter, with no training
    and no inference run.
    """
    from safetensors import safe_open

    files = sorted(checkpoint.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"No .safetensors under {checkpoint}")

    acc: dict[str, dict[str, float]] = {m: {"sq_err": 0.0, "sq_ref": 0.0, "dot": 0.0, "n": 0.0} for m in modes}
    seen = 0

    for path in files:
        with safe_open(str(path), framework="numpy") as handle:
            for key in handle.keys():  # noqa: SIM118 — safetensors handle is not a dict.
                if seen >= max_tensors:
                    break
                if any(s in key.lower() for s in SKIP_SUBSTRINGS):
                    continue
                arr = handle.get_tensor(key)
                if arr.ndim < 2:
                    continue
                w2d = arr.reshape(arr.shape[0], -1)
                # Every surveyed mode groups along the last axis; skip weights
                # whose width is not a multiple of the coarsest group.
                if w2d.shape[-1] % 64 != 0:
                    continue

                w = mx.array(w2d.astype("float16"))
                for mode in modes:
                    try:
                        deq = _roundtrip(mx, w, mode)
                        mx.eval(deq)
                    except Exception:  # noqa: BLE001 — unsupported mode, already reported by probe.
                        continue
                    diff = (deq - w).astype(mx.float32)
                    ref = w.astype(mx.float32)
                    acc[mode]["sq_err"] += float(mx.sum(diff * diff))
                    acc[mode]["sq_ref"] += float(mx.sum(ref * ref))
                    acc[mode]["dot"] += float(mx.sum(deq.astype(mx.float32) * ref))
                    acc[mode]["n"] += 1
                seen += 1

    out: dict[str, dict[str, float]] = {}
    for mode, a in acc.items():
        if a["n"] == 0:
            continue
        rel_l2 = (a["sq_err"] / a["sq_ref"])**0.5 if a["sq_ref"] > 0 else float("nan")
        out[mode] = {"rel_l2": rel_l2, "tensors": a["n"], "bits_per_param": MODE_BITS_PER_PARAM.get(mode, float("nan"))}
    return out


def measure_throughput(mx, modes: list[str], dim: int, tokens: int, iters: int) -> dict[str, float]:
    """Median ms per quantized matmul at DiT-realistic shapes.

    A 4-bit mode that is not clearly faster than int8 is being emulated rather
    than dispatched to the neural accelerators.
    """
    results: dict[str, float] = {}
    x = mx.random.normal((tokens, dim)).astype(mx.float16)

    for mode in ["fp16"] + modes:
        try:
            if mode == "fp16":
                w = mx.random.normal((dim, dim)).astype(mx.float16)

                def run():
                    return x @ w.T
            else:
                spec = _spec_for(mode)
                wq = mx.random.normal((dim, dim)).astype(mx.float16)
                q = mx.quantize(wq, **spec)
                packed, scales = q[0], q[1]
                biases = q[2] if len(q) == 3 else None

                def run(packed=packed, scales=scales, biases=biases, spec=spec):
                    return mx.quantized_matmul(x, packed, scales, biases, transpose=True, **spec)

            for _ in range(3):
                mx.eval(run())
            samples = []
            for _ in range(iters):
                start = time.perf_counter()
                mx.eval(run())
                samples.append((time.perf_counter() - start) * 1000.0)
            samples.sort()
            results[mode] = samples[len(samples) // 2]
        except Exception as exc:  # noqa: BLE001
            results[mode] = float("nan")
            print(f"  {mode}: throughput probe failed — {type(exc).__name__}: {exc}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Diffusers transformer dir with .safetensors")
    parser.add_argument("--modes", nargs="+", default=["int8", "int4", "mxfp8", "mxfp4", "nvfp4"])
    parser.add_argument("--max-tensors", type=int, default=60, help="Weight tensors to sample for the error survey")
    parser.add_argument("--dim", type=int, default=1536, help="Square matmul dim for the throughput probe")
    parser.add_argument("--tokens", type=int, default=8192, help="Sequence length for the throughput probe")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    import mlx.core as mx

    report: dict[str, Any] = {"mlx_version": getattr(mx, "__version__", "unknown")}
    print(f"MLX {report['mlx_version']}\n")

    print("=== 1. Mode support ===")
    support = probe_modes(mx, args.modes)
    report["support"] = support
    for mode, err in support.items():
        print(f"  {mode:<8} {'OK' if err is None else 'UNSUPPORTED — ' + err}")
    usable = [m for m, e in support.items() if e is None]
    if not usable:
        raise SystemExit("\nNo surveyed mode is usable on this MLX build; upgrade mlx and re-run.")

    print("\n=== 2. Throughput (median ms, lower is better) ===")
    print(f"  shape: ({args.tokens}, {args.dim}) x ({args.dim}, {args.dim})")
    tput = measure_throughput(mx, usable, args.dim, args.tokens, args.iters)
    report["throughput_ms"] = tput
    base = tput.get("fp16", float("nan"))
    for mode, ms in tput.items():
        speedup = f"{base / ms:.2f}x vs fp16" if ms == ms and base == base and ms > 0 else "n/a"
        print(f"  {mode:<8} {ms:8.3f} ms   {speedup}")

    if args.checkpoint:
        print("\n=== 3. Reconstruction error on real weights (lower is better) ===")
        err = measure_error(mx, args.checkpoint, usable, args.max_tensors)
        report["error"] = err
        print(f"  {'mode':<8} {'rel L2':>10} {'bits/param':>11} {'tensors':>8}")
        for mode, stats in sorted(err.items(), key=lambda kv: kv[1]["rel_l2"]):
            print(f"  {mode:<8} {stats['rel_l2']:>10.5f} {stats['bits_per_param']:>11.2f} {stats['tensors']:>8.0f}")
        print("\n  Ranking here predicts output-quality ordering. A 4-bit mode whose")
        print("  rel L2 is close to int8's is a genuine candidate; one that is several")
        print("  times worse will show up as visible artifacts after a few denoising steps.")
    else:
        print("\n=== 3. Reconstruction error — SKIPPED (pass --checkpoint) ===")

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
