# SPDX-License-Identifier: Apache-2.0
"""Create immutable caption/id/text-disjoint artifact views without recaching."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import unicodedata


def caption_key(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation-groups', type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to replace an existing split: {args.output}')
    records = []
    manifests = {}
    for root in args.source:
        for manifest in sorted(root.glob('manifest_rank*.jsonl')):
            manifests[str(manifest.resolve())] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            records.extend((root.resolve(), json.loads(line)) for line in manifest.read_text().splitlines() if line.strip())
    if not records:
        raise ValueError('No artifact records found')
    parents = list(range(len(records)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, (root, record) in enumerate(records):
        assert record['num_latent_frames'] == 37 and record['num_audio_latents'] == 207
        for key in [('caption', caption_key(record['caption'])), ('id', record['id']), ('text', record['text_sha1'])]:
            if key in seen:
                parents[find(i)] = find(seen[key])
            seen[key] = i
        for source in (root / 'latents' / (record['id'] + '.safetensors'),
                       root / 'text' / (record['text_sha1'] + '.safetensors')):
            if not source.is_file():
                raise FileNotFoundError(source)
    groups = {}
    for i, (_, record) in enumerate(records):
        groups.setdefault(find(i), []).append(i)
    if not 0 < args.validation_groups < len(groups):
        raise ValueError('Validation group count must leave nonempty train and validation sets')
    ordered = sorted(groups, key=lambda g: hashlib.sha256(('20260906:' + min(
        caption_key(records[i][1]['caption']) for i in groups[g])).encode()).hexdigest())
    validation = set(ordered[:args.validation_groups])
    views = {'train': [], 'validation': []}
    # Preflight above completes before creating any view; never overwrite artifacts.
    for split in views:
        for kind in ('latents', 'text'):
            (args.output / split / kind).mkdir(parents=True)
    for i, (root, original) in enumerate(records):
        split = 'validation' if find(i) in validation else 'train'
        record = dict(original)
        record['source_id'] = original['id']
        record['id'] = f'{i:06d}-{original["id"]}'
        record['source_artifact_root'] = str(root)
        directory = args.output / split
        (directory / 'latents' / (record['id'] + '.safetensors')).symlink_to(root / 'latents' / (original['id'] + '.safetensors'))
        text_link = directory / 'text' / (record['text_sha1'] + '.safetensors')
        source_text = root / 'text' / text_link.name
        if text_link.exists():
            if hashlib.sha256(text_link.read_bytes()).digest() != hashlib.sha256(source_text.read_bytes()).digest():
                raise ValueError(f'Conflicting text artifact contents for {text_link.name}')
        else:
            text_link.symlink_to(source_text)
        views[split].append(record)
    receipt = {'sources': manifests, 'seed': 20260906, 'groups': len(groups),
               'validation_groups': len(validation), 'semantic_deduplication': False, 'views': {}}
    for split, selected in views.items():
        manifest = args.output / split / 'manifest_rank0.jsonl'
        manifest.write_text(''.join(json.dumps(record) + '\n' for record in selected))
        receipt['views'][split] = {'records': len(selected), 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
    for key in ('caption', 'source_id', 'text_sha1'):
        key_fn = caption_key if key == 'caption' else str
        overlap = {key_fn(r[key]) for r in views['train']} & {key_fn(r[key]) for r in views['validation']}
        assert not overlap, (key, overlap)
    receipt['normalized_caption_id_text_overlap'] = 0
    (args.output / 'split_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
