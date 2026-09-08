import importlib.util
from pathlib import Path
import json
import torch
from safetensors.torch import save_file, load_file

def test_two_cuts(tmp_path):
    path=Path(__file__).resolve().parents[4] / 'scripts/checkpoint_conversion/prune_minimax_h3_blocks.py'
    spec=importlib.util.spec_from_file_location('convert',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    src=tmp_path/'src';src.mkdir()
    tensors={f'transformer_blocks.{i}.weight':torch.tensor([float(i)]) for i in range(50)}
    save_file(tensors,src/'weights.safetensors')
    (src/'config.json').write_text(json.dumps({'num_layers':50}))
    (src/m.INDEX_NAME).write_text(json.dumps({'weight_map':{k:'weights.safetensors' for k in tensors}}))
    first=tuple(round(i*49/41) for i in range(42)); second=tuple(round(i*41/33) for i in range(34))
    kw=dict(strategy='test',source_model='base',source_revision='pinned')
    m.prune_transformer(src,tmp_path/'42',first,**kw)
    # Simulate recovery: the second extraction must preserve these changed weights.
    updated=load_file(tmp_path/'42/weights.safetensors')
    updated={k:v+100 for k,v in updated.items()};save_file(updated,tmp_path/'42/model.safetensors')
    (tmp_path/'42'/m.INDEX_NAME).unlink()
    (tmp_path/'42/weights.safetensors').unlink()
    result=m.prune_transformer(tmp_path/'42',tmp_path/'34',second,**kw)
    assert result['block_map']==[first[i] for i in second]
    assert result['source_num_layers']==50
    actual=load_file(tmp_path/'34/model.safetensors')
    for i,j in enumerate(second):
        assert actual[f'transformer_blocks.{i}.weight'].item()==first[j]+100
