# SPDX-License-Identifier: Apache-2.0
"""Read verified prompt-only cache references with native geometry, one document per batch."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from fastvideo.dataset.parquet_dataset_map_style import DP_SP_BatchSampler, passthrough
from fastvideo.distributed import get_sp_world_size, get_world_rank, get_world_size


def decode_embedding(row: dict[str, Any]) -> torch.Tensor:
    shape = tuple(row['text_embedding_shape'])
    if len(shape) != 2 or shape[0] < 1 or shape[1] != 5120:
        raise ValueError(f'Invalid H3 embedding shape {shape}')
    dtype = str(row['text_embedding_dtype'])
    if dtype in ('bfloat16', 'torch.bfloat16'):
        tensor = torch.from_numpy(np.frombuffer(row['text_embedding_bytes'], dtype=np.uint16).copy()).view(torch.bfloat16)
    elif dtype in ('float16', 'float32', 'float64'):
        tensor = torch.from_numpy(np.frombuffer(row['text_embedding_bytes'], dtype=dtype).copy())
    else:
        raise ValueError(f'Unsupported cached embedding dtype {dtype}')
    tensor = tensor.reshape(shape)
    if not torch.isfinite(tensor).all():
        raise ValueError('Nonfinite cached embedding')
    return tensor


class MiniMaxH3PromptIndexDataset(Dataset):
    def __init__(self, root: str, *, seed: int = 0):
        path = Path(root)
        receipt = json.loads((path / 'receipt.json').read_text())
        if (receipt.get('metadata_match_complete') is not True
                or receipt.get('embedding_values_validated') is not True):
            raise ValueError('Prompt cache index has not passed complete metadata and value validation')
        raw_index = (path / 'prompt_index.jsonl').read_bytes()
        if hashlib.sha256(raw_index).hexdigest() != receipt['index_sha256']:
            raise ValueError('Prompt index checksum mismatch')
        self.records = [json.loads(line) for line in raw_index.decode().splitlines()]
        if len(self.records) != receipt['requested']:
            raise ValueError('Prompt index count does not match receipt')
        self.sampler = DP_SP_BatchSampler(batch_size=1, dataset_size=len(self.records),
            num_sp_groups=get_world_size() // get_sp_world_size(), sp_world_size=get_sp_world_size(),
            global_rank=get_world_rank(), drop_last=False, seed=seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        ref = record['embedding']
        row = pq.ParquetFile(ref['parquet']).read_row_group(ref['row_group'], columns=[
            'id', 'caption', 'text_embedding_shape', 'text_embedding_dtype', 'text_embedding_bytes'
        ]).slice(ref['row'], 1).to_pylist()[0]
        if row['id'] != ref['cached_id'] or row['caption'] != record['prompt']:
            raise ValueError('Cached embedding identity changed after indexing')
        embed = decode_embedding(row)
        return {'text_embedding': embed, 'text_attention_mask': torch.ones(embed.shape[0], dtype=torch.bool),
                'prompt_geometry': record['runtime_config'], 'prompt_only': True,
                'info': {'id': record['id'], 'prompt': record['prompt']}}

    def __getitems__(self, indices):
        if len(indices) != 1:
            raise ValueError('H3 prompt-index batches require one document')
        row = self[indices[0]]
        return {**row, 'text_embedding': row['text_embedding'].unsqueeze(0),
                'text_attention_mask': row['text_attention_mask'].unsqueeze(0), 'info_list': [row['info']]}


def build_prompt_index_loader(root: str, *, seed: int = 0):
    dataset = MiniMaxH3PromptIndexDataset(root, seed=seed)
    return StatefulDataLoader(dataset, batch_sampler=dataset.sampler, collate_fn=passthrough,
                              num_workers=0, pin_memory=True)
