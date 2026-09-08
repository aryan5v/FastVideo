#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create a distinct grid32 checkpoint, repeating only joint output heads.

Channel-major (head,C,patch...) matches V23 PDDReplicatedLinear. Every other
weight is copied unchanged, same dtype. Source is never modified.
"""
import argparse
import json
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file


def expand(src: Path, dst: Path):
    if dst.exists():
        raise FileExistsError(dst)
    config = json.loads((src/'transformer/config.json').read_text())
    if config.get('pdd_steps') is not None or config['num_layers'] != 42:
        raise ValueError('Expected single-head recovered42 source')
    folder = dst/'transformer'
    folder.mkdir(parents=True)
    state = {}
    expanded = []
    names = {'proj_out.weight','proj_out.bias','audio_proj_out.weight','audio_proj_out.bias'}
    with safe_open(src/'transformer/model.safetensors', framework='pt', device='cpu') as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if key in names:
                tensor = tensor.repeat((32,) + (1,)*(tensor.ndim-1)).contiguous()
                expanded.append(key)
            state[key] = tensor
    if set(expanded) != names:
        raise ValueError(f'Missing output parameters: {names-set(expanded)}')
    save_file(state, folder/'model.safetensors', metadata={'format':'pt'})
    config['pdd_steps'] = 32
    (folder/'config.json').write_text(json.dumps(config, indent=2))
    for item in src.iterdir():
        if item.name != 'transformer':
            (dst/item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
    (dst/'pdd_expansion_receipt.json').write_text(json.dumps({
        'source': str(src), 'pdd_steps':32,'expanded_keys':expanded,
        'unchanged_keys':len(state)-4,'skipped_keys':[]}, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--src',type=Path,required=True)
    p.add_argument('--dst',type=Path,required=True)
    a=p.parse_args()
    expand(a.src,a.dst)
