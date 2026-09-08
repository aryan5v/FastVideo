#!/usr/bin/env python3
"""Bind the recovery recipe to the actual selected checkpoint and its seams."""
import argparse
import json
from pathlib import Path
import yaml

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--template', type=Path, required=True)
p.add_argument('--student', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--depth', type=int, choices=[34,42], default=42)
a = p.parse_args()
arch = json.loads((a.student / 'transformer/config.json').read_text())
blocks = arch['block_map']
if len(blocks) != a.depth or arch['source_num_layers'] != 50:
    raise ValueError(f'Expected a Base-derived{a.depth}-block checkpoint')
seams = [i for i in range(1, len(blocks)) if blocks[i] > blocks[i-1]+1]
if not seams:
    raise ValueError('No pruning seams found')
cfg = yaml.safe_load(a.template.read_text())
cfg['models']['student']['init_from'] = str(a.student)
cfg['method']['feature_local_block_indices'] = seams
with a.output.open('x') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(json.dumps({'student': str(a.student), 'seams': seams, 'original_blocks': [blocks[i] for i in seams]}))
