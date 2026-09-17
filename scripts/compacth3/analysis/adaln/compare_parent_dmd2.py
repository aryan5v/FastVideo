#!/usr/bin/env python3
"""Sharper parent-vs-DMD2 comparison from the saved singular spectra."""
import json
from pathlib import Path

OUT = Path("/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/adaln_rank_analysis")
D = {t: json.loads((OUT / f"adaln_rank_{t}.json").read_text()) for t in ("parent", "dmd2")}


def topk_energy(sigma, k):
    s = [float(x) for x in sigma[:k]]
    tot = sum(float(x) ** 2 for x in sigma)
    return sum(x * x for x in s) / tot


print("LEVEL A  (shared coordinate z(t), 4096 x 768, uniform grid, centered)")
print("  model   s1        s2/s1     s3/s1     s4/s1     s5/s1     s6/s1     E@1     E@2     E@3     E@4     E@6")
for t in ("parent", "dmd2"):
    sv = D[t]["level_a_shared_coordinate"]["uniform_grid"]["sigma"]
    r = [sv[i] / sv[0] if i < len(sv) else 0.0 for i in range(6)]
    print("  {:<6s}  {:<9.4f} {:<9.5f} {:<9.5f} {:<9.5f} {:<9.5f} {:<9.5f} ".format(
        t, sv[0], r[1], r[2], r[3], r[4], r[5])
        + " ".join("{:.4f}".format(topk_energy(sv, k)) for k in (1, 2, 3, 4, 6)))

print()
print("LEVEL C  (union over all 42 blocks, 4096 x 4064256, centered, energy-weighted)")
print("  model   E@1     E@2     E@3     E@4     E@6     E@8    stable")
for t in ("parent", "dmd2"):
    c = D[t]["level_c_shared_basis"]["uniform_grid"]["energy_weighted"]["sigma_top"]
    st = D[t]["level_c_shared_basis"]["uniform_grid"]["energy_weighted"]["stats"]["stable_rank_trace_over_smax2"]
    print("  {:<6s}  ".format(t) + " ".join("{:.4f}".format(topk_energy(c, k)) for k in (1, 2, 3, 4, 6, 8))
          + " {:.4f}".format(st))

print()
print("PER-BLOCK (42 blocks, uniform grid)")
hdr = "  model    rank99 mean/median   rank99.9 mean/median   stable mean   E@2 per block min/med/max"
print(hdr)
for t in ("parent", "dmd2"):
    bl = D[t]["level_b_per_block"]["uniform_grid"]["blocks"]
    r99 = sorted(v["stats"]["rank_at_0.990_energy"] for v in bl.values())
    r999 = sorted(v["stats"]["rank_at_0.999_energy"] for v in bl.values())
    st = sorted(v["stats"]["stable_rank_trace_over_smax2"] for v in bl.values())
    cov = D[t]["level_c_shared_basis"]["uniform_grid"]["coverage_min_median_max_fraction_of_block_energy"]["2"]
    n = len(r99)
    print("  {:<7s}  {:.2f}/{:.1f}              {:.2f}/{:.1f}                {:.4f}        {:.4f}/{:.4f}/{:.4f}".format(
        t, sum(r99) / n, r99[n // 2], sum(r999) / n, r999[n // 2], sum(st) / n, cov[0], cov[1], cov[2]))

print()
print("PER-MODALITY PROFILE (level B per-block, energy fraction of each modality table; parent vs dmd2)")
for t in ("parent", "dmd2"):
    bl = D[t]["level_b_per_block"]["uniform_grid"]["blocks"]
    fr = [[v["frobenius_per_modality"][m] for v in bl.values()] for m in range(3)]
    print("  {:<7s} modality frobenius mean: ".format(t)
          + "  ".join("m{}= {:.1f}".format(m, sum(fr[m]) / len(fr[m])) for m in range(3))
          + "   (mean ratio m1/m0={:.4f}, m2/m0={:.4f})".format(
              (sum(fr[1]) / len(fr[1])) / (sum(fr[0]) / len(fr[0])),
              (sum(fr[2]) / len(fr[2])) / (sum(fr[0]) / len(fr[0]))))

print()
print("OPS GRID (191 literal inference timesteps)")
print("  model   A stable  C stable  B rank99 med")
for t in ("parent", "dmd2"):
    a = D[t]["level_a_shared_coordinate"]["operating_points"]["stats"]
    c = D[t]["level_c_shared_basis"]["operating_points"]["energy_weighted"]["stats"]
    b = D[t]["level_b_per_block"]["operating_points"]["summary"]["rank99_min_median_max"]
    print("  {:<7s} {:.4f}    {:.4f}    {}".format(t, a["stable_rank_trace_over_smax2"],
                                                  c["stable_rank_trace_over_smax2"], b))
