#!/usr/bin/env python3
"""Attach pinned shared Base components to a completed extracted transformer."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--base', type=Path, required=True)
p.add_argument('--student', type=Path, required=True)
a = p.parse_args()
manifest = json.loads((a.student / 'transformer/block_map_manifest.json').read_text())
index = json.loads((a.student / 'transformer/diffusion_pytorch_model.safetensors.index.json').read_text())
for shard in set(index['weight_map'].values()):
    if not (a.student / 'transformer' / shard).is_file():
        raise RuntimeError(f'Missing extracted shard: {shard}')
for source in a.base.iterdir():
    if source.name in {'transformer', '.cache'}:
        continue
    dest = a.student / source.name
    if dest.is_symlink() and dest.resolve() == source.resolve():
        continue
    if dest.exists() or dest.is_symlink():
        raise RuntimeError(f'Refusing to replace component: {dest}')
    dest.symlink_to(source.resolve(), target_is_directory=source.is_dir())
