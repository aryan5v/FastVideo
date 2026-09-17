#!/usr/bin/env python3
"""Require completed baseline parity and exact-speech receipts before recovery."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
parity = json.loads((root / 'parity.json').read_text())
if parity.get('passed') is not True:
    raise RuntimeError('Base backend parity gate did not pass')
for p in (root / 'parent-FLASH_ATTN/wer.json', root / '6868-TORCH_SDPA-wer.json'):
    receipt = json.loads(p.read_text())
    print(p, receipt)
    # Schema is checked explicitly; unknown/missing metric never passes.
    wer = receipt.get('word_error_rate')
    if not isinstance(wer, (float, int)) or wer != 0:
        raise RuntimeError(f'Base exact speech gate failed or is missing: {p}')
