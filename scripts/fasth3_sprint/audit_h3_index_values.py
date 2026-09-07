#!/usr/bin/env python3
"""Validate every referenced embedding, reading each Parquet row group only once."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def audit(root):
    receipt = json.loads((root / 'receipt.json').read_text())
    raw = (root / 'prompt_index.jsonl').read_bytes()
    assert receipt['metadata_match_complete'] and hashlib.sha256(raw).hexdigest() == receipt['index_sha256']
    groups = defaultdict(list)
    for line in raw.decode().splitlines():
        row = json.loads(line)
        ref = row['embedding']
        groups[(ref['parquet'], ref['row_group'])].append(row)
    checked = 0
    dtypes = set()
    for (path, group), records in groups.items():
        rows = pq.ParquetFile(path).read_row_group(group).to_pylist()
        for record in records:
            row = rows[record['embedding']['row']]
            assert row['id'] == record['embedding']['cached_id'] and row['caption'] == record['prompt']
            dtype = row['text_embedding_dtype']
            dtypes.add(dtype)
            if dtype in ('bfloat16', 'torch.bfloat16'):
                values = (np.frombuffer(row['text_embedding_bytes'], dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            elif dtype in ('float16', 'float32', 'float64'):
                values = np.frombuffer(row['text_embedding_bytes'], dtype=dtype)
            else:
                raise ValueError(f'Unsupported dtype {dtype}')
            assert values.size == np.prod(row['text_embedding_shape']) and np.isfinite(values).all()
            assert np.any(values != 0), 'All-zero embedding'
            checked += 1
        if checked % 1000 < len(records):
            print(f'Checked {checked}/{receipt["requested"]}', flush=True)
    assert checked == receipt['requested']
    receipt.update(embedding_values_validated=True, validated_embeddings=checked, dtypes=sorted(dtypes),
                   training_ready=False, remaining_gate='GPU native-geometry forward/backward and encoder provenance check')
    (root / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    audit(parser.parse_args().root)
