# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced FP8 score of an OmniRef student against the bf16 teacher (torchrun, QAD harness config).

Same metric as the NVFP4 QAD evals (``OmniRefPDDQADMethod._evaluate``: per-forward x0 relative L2 on the
held-out rows, video and audio separately, ``score`` = their mean), with the student's linears replaced by
FP8 W8A8 fake quantization (see ``fp8_student``). Attention stays bf16 VSA, as on the RTX 4090 profile.
Pass ``--models.student.qad.enabled false`` and ``--models.student.init_from <dir>`` as config overrides.

    --weights fp8|bf16   quantize the student's weights per channel (bf16: leave dense)
    --act dynamic|static|none   per-token dynamic, per-tensor static (--act-table), or unquantized activations
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--weights", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument("--act", choices=("dynamic", "static", "none"), default="dynamic")
    parser.add_argument("--act-table", default=None, help="JSON {b<idx>.<linear>: amax} for --act static")
    parser.add_argument("--act-field", default=None, help="field to read when the table maps keys to dicts")
    parser.add_argument("--max-rows", type=int, default=0, help="evaluate only the first N held-out rows (0: all)")
    parser.add_argument("--no-refiner", action="store_true", help="leave the token-refiner linears in bf16")
    args, overrides = parser.parse_known_args()

    import sys
    sys.path.insert(0, str(HERE))
    from fp8_student import install_fp8

    from fastvideo.distributed import get_world_group, maybe_init_distributed_environment_and_model_parallel
    from fastvideo.train.utils.builder import build_from_config
    from fastvideo.train.utils.config import load_run_config

    cfg = load_run_config(args.config, overrides=overrides or None)
    maybe_init_distributed_environment_and_model_parallel(cfg.training.distributed.tp_size,
                                                          cfg.training.distributed.sp_size)
    _, method, _, _ = build_from_config(cfg)
    table = None
    if args.act == "static":
        raw = json.loads(Path(args.act_table).read_text())
        table = {k: float(v[args.act_field] if isinstance(v, dict) else v) for k, v in raw.items()}
    info = install_fp8(method.student.transformer,
                       weights=args.weights == "fp8",
                       activations=args.act,
                       act_amax=table,
                       include_refiner=not args.no_refiner)
    method.max_eval_rows = args.max_rows
    method._eval_iteration = 0
    rows = method._rank_rows()
    summary = method._evaluate(rows, rollout=False, save_tag=args.tag)
    record = {
        "tag": args.tag,
        "weights": args.weights,
        "act": args.act,
        "act_table": args.act_table,
        "linears": info["linears"],
        "rows": len(method.student.eval_rows),
        "summary": summary
    }
    if get_world_group().rank == 0:
        Path(args.out).write_text(json.dumps(record, indent=1))
        keep = ("score", "x0_rel_l2/video/mean", "x0_rel_l2/audio/mean")
        print(json.dumps({k: summary.get(k) for k in keep} | {"tag": args.tag}), flush=True)


if __name__ == "__main__":
    main()
