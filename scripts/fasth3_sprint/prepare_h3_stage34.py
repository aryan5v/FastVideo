#!/usr/bin/env python3
"""Prepare the next bounded recovery job only after reviewed stage-42 recovery."""
import argparse
import json
from pathlib import Path
import shlex

import yaml
from h3_stage_promotion import check_review, compose_map

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--review', type=Path, required=True)
p.add_argument('--export', dest='export_path', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
review = json.loads(a.review.read_text())
check_review(review, a.checkpoint)
if Path(review.get('export_path', '')).resolve() != a.export_path.resolve():
    raise ValueError('Review must name the exact recovered export')
config = json.loads((a.export_path / 'transformer/config.json').read_text())
if config['num_layers'] != 42 or config['source_num_layers'] != 50:
    raise ValueError('Expected recovered Base-derived 42-block export')
local, original = compose_map(config['block_map'], 34)
a.output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[2]
cfg = yaml.safe_load((root / 'examples/train/configs/fasth3_base42_recovery.yaml').read_text())
student = a.output.resolve() / 'student34'
cfg['models']['student']['init_from'] = str(student)
cfg['method']['feature_local_block_indices'] = [i for i in range(1, len(original)) if original[i] > original[i-1]+1]
config_path = a.output.resolve() / 'stage34.yaml'
config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
script = (root / 'scripts/fasth3_sprint/slurm_h3_base42_recovery.sbatch').read_text()
script = script.replace('base42', 'base34')
script = script.replace('export STUDENT="${SPRINT_ROOT}/checkpoints/base34-uniform-v1"',
                        'export STUDENT=' + shlex.quote(str(student)))
script = script.replace('test -s "${STUDENT}/transformer/block_map_manifest.json"\n', '')
script = script.replace('${CODE_ROOT}/examples/train/configs/fasth3_base34_recovery.yaml', str(config_path))
# JSON-derived paths are shell quoted; the launcher nests a single-quoted container command.
if any("'" in str(path) for path in (a.export_path, a.output)):
    raise ValueError('Launcher paths must not contain single quotes')
command = ('"${PY}" "${CODE_ROOT}/scripts/checkpoint_conversion/prune_minimax_h3_blocks.py"'
           ' --src ' + shlex.quote(str(a.export_path.resolve() / 'transformer')) +
           ' --dst "${STUDENT}/transformer" --block-map ' + ','.join(map(str, local)) +
           ' --strategy recovered42-to34 --source-model ' + shlex.quote(str(a.export_path.resolve())) +
           ' --source-revision ' + shlex.quote(a.checkpoint.name) + '\n')
anchor = '"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/assemble_h3_stage.py"'
script = script.replace(anchor, command + anchor)
script = script.replace('#SBATCH --nodes=1', '#SBATCH --nodes=1\n#SBATCH --mem=128G')
(a.output / 'stage34.sbatch').write_text(script)
(a.output / 'provenance.json').write_text(json.dumps({'parent_checkpoint': str(a.checkpoint.resolve()),
    'parent_export': str(a.export_path.resolve()), 'local_extraction_map': local,
    'original_base_block_map': original, 'review': review}, indent=2))
print(a.output / 'stage34.sbatch')
