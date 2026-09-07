#!/usr/bin/env python3
"""Match an audited prompt split to existing embeddings without copying binary data."""
import argparse
import hashlib
import json
from pathlib import Path


def build_index(prompts, cache_roots, output):
    import pyarrow.parquet as pq
    output.mkdir(parents=True, exist_ok=False)
    requested = {}
    for line in prompts.read_text().splitlines():
        row = json.loads(line)
        if row['id'] in requested:
            raise ValueError('duplicate prompt id')
        requested[row['id']] = row
    found, conflicts = {}, []
    for root in cache_roots:
        for path in sorted(root.rglob('*.parquet')):
            parquet = pq.ParquetFile(path)
            for group in range(parquet.num_row_groups):
                rows = parquet.read_row_group(group, columns=['id', 'caption', 'text_embedding_shape',
                                                              'text_embedding_dtype']).to_pylist()
                for offset, cached in enumerate(rows):
                    identifier = cached['id'].split(':', 1)[-1]
                    if identifier not in requested:
                        continue
                    row = requested[identifier]
                    if row['prompt'] != cached['caption']:
                        conflicts.append({'id': identifier, 'reason': 'caption mismatch', 'path': str(path)})
                        continue
                    shape = cached['text_embedding_shape']
                    if len(shape) != 2 or shape[0] < 1 or shape[1] != 5120:
                        raise ValueError(f'invalid embedding shape: {identifier}: {shape}')
                    if identifier in found:
                        raise ValueError(f'ambiguous cached id: {identifier}')
                    found[identifier] = {**row, 'embedding': {'parquet': str(path.resolve()),
                        'row_group': group, 'row': offset, 'cached_id': cached['id'],
                        'shape': shape, 'dtype': cached['text_embedding_dtype']}}
    with (output / 'prompt_index.jsonl').open('w') as stream:
        for identifier in sorted(found):
            stream.write(json.dumps(found[identifier], ensure_ascii=False) + '\n')
    missing = sorted(set(requested) - set(found))
    receipt = {'requested': len(requested), 'matched': len(found), 'missing': missing,
               'conflicts': conflicts, 'source_sha256': hashlib.sha256(prompts.read_bytes()).hexdigest(),
               'index_sha256': hashlib.sha256((output / 'prompt_index.jsonl').read_bytes()).hexdigest(),
               'metadata_match_complete': not missing and not conflicts,
               'embedding_values_validated': False, 'training_ready': False}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('missing', 'conflicts')}), flush=True)
    if missing or conflicts:
        raise ValueError(f'{len(missing)} missing prompts and {len(conflicts)} caption conflicts')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build_index(args.prompts, args.cache_root, args.output)
