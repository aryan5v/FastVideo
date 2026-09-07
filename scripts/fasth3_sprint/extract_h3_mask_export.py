# SPDX-License-Identifier: Apache-2.0
"""Physically extract the exact training mask from a completed full FP32 export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

from scripts.checkpoint_conversion.prune_minimax_h3_blocks import INDEX_NAME, prune_transformer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--full-export', type=Path, required=True)
    parser.add_argument('--compact-export', type=Path, required=True)
    args = parser.parse_args()
    if args.compact_export.exists():
        raise FileExistsError(args.compact_export)
    meta = json.loads((args.checkpoint / 'metadata.json').read_text())
    retained = tuple(meta['config']['method']['retained_blocks'])
    source = args.full_export / 'transformer'
    with (source / 'model.safetensors').open('rb') as stream:
        header_length = struct.unpack('<Q', stream.read(8))[0]
        if header_length > 100_000_000:
            raise ValueError('invalid safetensors header size')
        header = json.loads(stream.read(header_length))
    tensors = {key: value for key, value in header.items() if key != '__metadata__'}
    index = {'metadata': {'total_size': sum(v['data_offsets'][1] - v['data_offsets'][0]
                                          for v in tensors.values())},
             'weight_map': {key: 'model.safetensors' for key in tensors}}
    index_path = source / INDEX_NAME
    if index_path.exists():
        if json.loads(index_path.read_text()) != index:
            raise ValueError('existing source index disagrees with actual export header')
    else:
        index_path.write_text(json.dumps(index, indent=2) + '\n')
    args.compact_export.mkdir(parents=True)
    for item in args.full_export.iterdir():
        if item.name != 'transformer':
            (args.compact_export / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
    receipt = prune_transformer(source, args.compact_export / 'transformer', retained,
                                strategy='trained-hard-mask', source_model=str(args.checkpoint.resolve()),
                                source_revision=(args.checkpoint.parent / 'CODE_COMMIT').read_text().strip())
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
