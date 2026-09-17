from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def read(path):
    import av
    c = av.open(str(path))
    st = next((s for s in c.streams if s.type == "video"), None)
    if st is None:
        c.close()
        return None
    frames = [f.to_ndarray(format="rgb24") for f in c.decode(st)]
    c.close()
    return np.stack(frames).astype(np.int16) if frames else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, type=Path, help="patched r=768 arm")
    ap.add_argument("--b", required=True, type=Path, help="unpatched arm")
    args = ap.parse_args()

    va, vb = read(args.a), read(args.b)
    if va is None or vb is None:
        print(f"IDENTITY GATE: FAILED to decode (a={va is not None} b={vb is not None})")
        return 2
    print(f"shapes: patched {va.shape}  unpatched {vb.shape}")
    if va.shape != vb.shape:
        print("IDENTITY GATE: MISMATCH (different shapes)")
        return 1

    d = np.abs(va - vb)
    n_diff = int((d > 0).sum())
    frac_diff = float((d > 0).mean())
    max_d = int(d.max())
    mse = float((d.astype(np.float64) ** 2).mean())
    psnr = float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))
    identical = n_diff == 0

    print(f"identical={identical}  differing_values={n_diff} ({frac_diff:.8%} of frame values)  "
          f"max_abs_diff={max_d}  mse={mse:.6g}  psnr={psnr:.2f} dB")
    if identical:
        print("IDENTITY GATE: PASS (bit-identical decoded video)")
        return 0
    if psnr > 60:
        print("IDENTITY GATE: PASS (not bit-identical, but PSNR > 60 dB -- consistent with "
              "run-to-run kernel nondeterminism, not a semantic change)")
        return 0
    print("IDENTITY GATE: FAIL -- the identity fold changed the output materially")
    return 1


if __name__ == "__main__":
    sys.exit(main())
