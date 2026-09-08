import importlib.util
from pathlib import Path
import pytest

p = Path(__file__).resolve().parents[4] / 'scripts/fasth3_sprint/h3_stage_promotion.py'
s = importlib.util.spec_from_file_location('promotion', p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)


def test_composed_map_keeps_base_identities():
    parent = [round(i * 49 / 41) for i in range(42)]
    local, original = m.compose_map(parent, 34)
    assert len(set(original)) == 34
    assert original == [parent[i] for i in local]
    assert original[0] == 0 and original[-1] == 49


def test_loss_only_cannot_promote(tmp_path):
    with pytest.raises(ValueError, match='all quality checks'):
        m.check_review({'checkpoint': str(tmp_path), 'loss': 0.1, 'approve_next_cut': True}, tmp_path)
