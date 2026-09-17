#!/usr/bin/env python3
"""Compact table dump over the adaln_rank_{parent,dmd2}.json spectra.

usage: python3 summarize_adaln_rank.py [tag ...]     (default: parent dmd2)
"""
import json
import sys
from pathlib import Path

OUT = Path("/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/adaln_rank_analysis")

HEAD = ("{:<34s} {:>18s} {:>6s} {:>5s} {:>5s} {:>5s} {:>6s} {:>8s} {:>8s}".format(
    "level", "shape", "numrk", "90%", "95%", "99%", "99.9%", "stable", "entropy"))


def row(label, s):
    return ("{:<34s} {:>18s} {:>6d} {:>5d} {:>5d} {:>5d} {:>6d} {:>8.3f} {:>8.3f}".format(
        label, str(s["shape"]), s["numerical_rank_fp32tol"], s["rank_at_0.900_energy"],
        s["rank_at_0.950_energy"], s["rank_at_0.990_energy"], s["rank_at_0.999_energy"],
        s["stable_rank_trace_over_smax2"], s["entropy_effective_rank"]))


def main():
    tags = sys.argv[1:] or ["parent", "dmd2"]
    for tag in tags:
        d = json.loads((OUT / f"adaln_rank_{tag}.json").read_text())
        print("=" * 130)
        print(f"{tag}   {d['checkpoint']}")
        print(f"  adaln_rank={d['adaln_rank']} hidden={d['hidden_size']} layers={d['num_layers']} "
              f"refiner_layers={d['num_refiner_layers']} refiner_adaln_modules={d['refiner_adaln_modules']}")
        print(f"  shifts={d['scheduler_shifts']}  ops_grid_n={d['grids']['operating_points']['n']}")
        print(HEAD)
        la = d["level_a_shared_coordinate"]
        for key in ("uniform_grid", "operating_points", "control_0_1000", "uniform_grid_bf16_native"):
            if key in la:
                print(row(f"A {key}", la[key]["stats"]))
        for gname, lb in d["level_b_per_block"].items():
            s = lb["summary"]
            print("  B {}: rank90={} rank99={} rank999={} stable={}  (n={} blocks, modulation_dim={})".format(
                gname, s["rank90_min_median_max"], s["rank99_min_median_max"],
                s["rank999_min_median_max"], [round(x, 3) for x in s["stable_rank_min_median_max"]],
                s["n_blocks"], lb["modulation_dim"]))
        for gname, c in d["level_c_shared_basis"].items():
            for key in ("energy_weighted", "block_normalized"):
                if key in c:
                    print(row(f"C {gname} {key}", c[key]["stats"]))
            for pm in c.get("per_modality_uniform_weighted", []):
                print(row(f"C {gname} modality{pm['modality']}", pm["stats"]))
            cov = c.get("coverage_min_median_max_fraction_of_block_energy")
            if cov:
                print(f"  C {gname} shared-basis coverage (min/median/max fraction of a block's energy):")
                print("    " + "  ".join(f"k={k}:{v[0]:.3f}/{v[1]:.3f}/{v[2]:.3f}" for k, v in cov.items()))


if __name__ == "__main__":
    main()
