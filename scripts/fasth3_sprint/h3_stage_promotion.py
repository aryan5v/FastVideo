#!/usr/bin/env python3
"""Fail closed before cutting a reviewed recovered checkpoint to the next depth."""
import argparse
import json
from pathlib import Path


def compose_map(parent_map, child_depth):
    if len(parent_map) != len(set(parent_map)) or parent_map != sorted(parent_map):
        raise ValueError('Parent map must be unique and ordered')
    if not 2 <= child_depth < len(parent_map):
        raise ValueError('Child depth must be smaller than the parent')
    local = [round(i * (len(parent_map) - 1) / (child_depth - 1)) for i in range(child_depth)]
    return local, [parent_map[i] for i in local]


def check_review(review, checkpoint):
    if Path(review['checkpoint']).resolve() != checkpoint.resolve():
        raise ValueError('Review belongs to a different checkpoint')
    required = ['export_parity_passed', 'video_review_passed', 'audio_review_passed',
                'motion_review_passed', 'prompt_adherence_passed', 'approve_next_cut']
    if not all(review.get(k) is True for k in required):
        raise ValueError('Next depth cut requires all quality checks, not just a loss threshold')
    if review.get('reviewed_prompt_count', 0) < 4 or not review.get('evidence_paths'):
        raise ValueError('Insufficient decoded evidence')
    if not (checkpoint / 'dcp/.metadata').is_file() or not (checkpoint / 'metadata.json').is_file():
        raise ValueError('Incomplete source checkpoint')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--parent-map', type=Path, required=True)
    p.add_argument('--child-depth', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    check_review(json.loads(a.review.read_text()), a.checkpoint)
    parent = json.loads(a.parent_map.read_text())['block_map']
    local, original = compose_map(parent, a.child_depth)
    with a.output.open('x') as f:
        json.dump({'local_extraction_map': local, 'original_base_block_map': original,
                   'source_checkpoint': str(a.checkpoint.resolve()), 'review': str(a.review.resolve())}, f, indent=2)
