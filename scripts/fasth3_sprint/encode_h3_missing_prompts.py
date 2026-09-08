#!/usr/bin/env python3
"""Encode missing audited prompts with the pinned H3 conditioning path."""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from build_multishot_calibration_supplement import _encode_prompts

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model-root', type=Path, required=True)
p.add_argument('--prompts', type=Path, required=True)
p.add_argument('--receipt', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
missing = set(json.loads(a.receipt.read_text())['missing'])
rows = [json.loads(line) for line in a.prompts.read_text().splitlines()]
rows = [r for r in rows if r['id'] in missing]
assert len(rows) == len(missing) == 32
assert not a.output.exists()
a.output.mkdir(parents=True)
embeds = _encode_prompts(a.model_root, tuple(r['prompt'] for r in rows))
for i, row in enumerate(rows):
    values = embeds[f'multishot_synth_{i:02d}'].numpy()
    assert values.ndim == 2 and values.shape[1] == 5120 and np.isfinite(values).all()
    pq.write_table(pa.Table.from_pylist([{'id': row['id'], 'caption': row['prompt'],
       'text_embedding_shape': list(values.shape), 'text_embedding_dtype': str(values.dtype),
       'text_embedding_bytes': values.tobytes()}]), a.output / f'{i:03d}.parquet')
(a.output / 'encoding_receipt.json').write_text(json.dumps({'count': len(rows),
    'model_root': str(a.model_root), 'source_ids': sorted(missing)}, indent=2))
