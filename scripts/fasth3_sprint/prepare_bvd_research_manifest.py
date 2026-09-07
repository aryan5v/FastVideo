#!/usr/bin/env python3
"""Validate locally available BVD clip records and reserve source-disjoint evaluation.

Input JSONL contract: id, source_video_id, video (local clip with its original
synchronized audio), caption, dataset_revision. This adapter does not fetch gated
media, infer captions, or substitute unrelated audio. Output is a preprocessing
manifest, never a ready-to-train latent dataset.
"""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    counts = {'train': 0, 'validation': 0}
    seen = set()
    streams = {split: (output / f'{split}.jsonl').open('w') for split in counts}
    try:
        for line in source.read_text().splitlines():
            row = json.loads(line)
            for key in ('id', 'source_video_id', 'caption', 'dataset_revision'):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f'Missing {key}')
            if row['id'] in seen:
                raise ValueError('Duplicate clip id')
            seen.add(row['id'])
            media = Path(row['video']).expanduser().resolve()
            if not media.is_file():
                raise FileNotFoundError(media)
            # All clips from the same source video stay in the same split.
            split = 'validation' if int(hashlib.sha256(row['source_video_id'].encode()).hexdigest()[:8], 16) % 20 == 0 else 'train'
            record = {**row, 'video': str(media), 'split': split, 'lineage': 'BVD-research',
                      'paired_audio_required': True, 'preprocessing_status': 'pending_decode_and_av_alignment'}
            streams[split].write(json.dumps(record) + '\n')
            counts[split] += 1
    finally:
        for stream in streams.values():
            stream.close()
    receipt = {'counts': counts, 'training_ready': False, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'next_stage': 'Decode video with original audio; validate duration, caption and AV alignment; encode both VAEs and Qwen text.'}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output)))
