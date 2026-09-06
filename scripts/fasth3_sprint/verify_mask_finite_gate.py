# SPDX-License-Identifier: Apache-2.0
"""Issue a continuation receipt only for a completed mask numerical gate.

The method itself enforces nonzero FP32 master/Adam updates and finite gradients
before optimizer steps. This checker binds successful completion to the exact
implementation, saved checkpoint, configuration, and held-out output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

TRAINING_FILES = (
    'fastvideo/models/dits/minimax_h3.py',
    'fastvideo/train/models/minimax_h3/minimax_h3.py',
    'fastvideo/train/methods/knowledge_distillation/minimax_h3_recovery.py',
    'fastvideo/train/methods/knowledge_distillation/minimax_h3_mask_recovery.py',
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--job-id', type=int, required=True)
    parser.add_argument('--gate-code', type=Path, required=True)
    parser.add_argument('--next-code', type=Path, required=True)
    parser.add_argument('--mode', choices=['hard', 'annealed_skip'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    accounting = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', str(args.job_id),
                                          '--format=JobIDRaw,State,ExitCode'], text=True)
    rows = [line.split('|') for line in accounting.splitlines() if line.strip()]
    assert any(row[:3] == [str(args.job_id), 'COMPLETED', '0:0'] for row in rows), accounting
    hashes = {}
    for relative in TRAINING_FILES:
        previous = hashlib.sha256((args.gate_code / relative).read_bytes()).hexdigest()
        assert previous == hashlib.sha256((args.next_code / relative).read_bytes()).hexdigest(), relative
        hashes[relative] = previous
    gate_commit = (args.gate_code / 'CODE_COMMIT').read_text().strip()
    assert (args.run / 'CODE_COMMIT').read_text().strip() == gate_commit
    checkpoint = args.run / 'checkpoint-2'
    assert (checkpoint / 'dcp' / '.metadata').is_file()
    assert len(list(checkpoint.glob('rng_state_rank*.pt'))) == 4
    metadata = json.loads((checkpoint / 'metadata.json').read_text())
    assert metadata['step'] == 2
    config = metadata['config']
    assert config['training']['dit_precision'] == 'fp32'
    assert config['training']['loop']['max_train_steps'] == 2
    assert config['method']['require_fp32_master'] is True
    assert config['method']['pruning_mode'] == args.mode
    assert config['method']['denoising_weight'] == 0
    lines = [json.loads(line) for line in (args.run / 'heldout_metrics.jsonl').read_text().splitlines()]
    assert {line['iteration'] for line in lines} == {0, 2}
    for line in lines:
        assert line['mode'] == args.mode
        assert line['retained_blocks'] == config['method']['retained_blocks']
        errors = {key: value for key, value in line.items() if 'interval' in key or 'endpoint' in key}
        assert len(errors) == 10 and all(math.isfinite(value) for value in errors.values())
    receipt = {'passed': True, 'mode': args.mode, 'job_id': args.job_id,
               'gate_commit': gate_commit, 'training_source_sha256': hashes,
               'run': str(args.run), 'checkpoint': str(checkpoint),
               'fp32_optimizer_update': True, 'finite_backward': True,
               'numerical_evidence': 'Successful execution of fail-closed checks in the hashed method implementation',
               'quality_passed': False, 'heldout': lines}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'receipt': str(args.output), 'passed': True, 'quality_passed': False}))


if __name__ == '__main__':
    main()
