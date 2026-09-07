# SPDX-License-Identifier: Apache-2.0
"""Check architecture-changing parity remapping without loading model weights."""
import ast
from pathlib import Path
import re
from typing import Any


def test_parity_mapping_drops_removed_blocks_and_preserves_shared_entries():
    root = Path(__file__).resolve().parents[4]
    source = root / 'scripts/fasth3_sprint/verify_recovery_export_prediction_parity.py'
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == '_compact_named_entries')
    namespace = {'re': re, 'Any': Any}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    remap = namespace['_compact_named_entries']
    entries = {'transformer_blocks.0.attn.weight': 0, 'transformer_blocks.1.attn.weight': 1,
               'transformer_blocks.3.attn.weight': 3, 'patch_embedding.weight': 'shared'}
    assert remap(entries, (0, 3)) == {'transformer_blocks.0.attn.weight': 0,
                                    'transformer_blocks.1.attn.weight': 3,
                                    'patch_embedding.weight': 'shared'}
    assert remap({'blocks.3.attn': 'SDPA', 'blocks.2.attn': 'SDPA'}, (0, 3)) == {'blocks.1.attn': 'SDPA'}
