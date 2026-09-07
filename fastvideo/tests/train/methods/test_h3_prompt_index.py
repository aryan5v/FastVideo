import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('indexer', ROOT / 'scripts/fasth3_sprint/index_h3_cached_prompts.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exact_caption_and_full_coverage(tmp_path):
    prompts = tmp_path / 'prompts.jsonl'
    prompts.write_text(json.dumps({'id': 'a', 'prompt': 'exact', 'runtime_config': {'num_frames': 362}}) + '\n')
    cache = tmp_path / 'cache'
    cache.mkdir()
    pq.write_table(pa.Table.from_pylist([{'id': 'source:a', 'caption': 'exact',
        'text_embedding_shape': [2, 5120], 'text_embedding_dtype': 'float32'}]), cache / 'data.parquet')
    module.build_index(prompts, [cache], tmp_path / 'index')
    record = json.loads((tmp_path / 'index/prompt_index.jsonl').read_text())
    assert record['runtime_config']['num_frames'] == 362
    receipt = json.loads((tmp_path / 'index/receipt.json').read_text())
    assert receipt['matched'] == 1 and receipt['metadata_match_complete']
    assert not receipt['training_ready']
    pq.write_table(pa.Table.from_pylist([{'id': 'source:a', 'caption': 'different',
        'text_embedding_shape': [2, 5120], 'text_embedding_dtype': 'float32'}]), cache / 'data.parquet')
    with pytest.raises(ValueError, match='missing prompts'):
        module.build_index(prompts, [cache], tmp_path / 'bad')


def test_bvd_source_group_split(tmp_path):
    spec = importlib.util.spec_from_file_location('bvd', ROOT / 'scripts/fasth3_sprint/prepare_bvd_research_manifest.py')
    bvd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bvd)
    media = tmp_path / 'clip.mp4'
    media.touch()  # Manifest adapter checks existence; decoder QA is intentionally a later stage.
    source = tmp_path / 'bvd.jsonl'
    source.write_text('\n'.join(json.dumps({'id': str(i), 'source_video_id': 'same-parent',
        'caption': 'caption', 'dataset_revision': 'revision', 'video': str(media)}) for i in range(3)))
    result = bvd.prepare(source, tmp_path / 'bvd')
    assert sorted(result['counts'].values()) == [0, 3]
    assert not result['training_ready']


def test_native_geometry_does_not_crop_to_global_defaults():
    import ast
    import torch
    from types import SimpleNamespace
    path = ROOT / 'fastvideo/train/models/minimax_h3/minimax_h3.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MiniMaxH3Model')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_resolve_clean_latents')
    module_ast = ast.Module(body=[fn], type_ignores=[])
    namespace = {'torch': torch, 'Any': object, 'Literal': __import__('typing').Literal,
                 '_VIDEO_LATENT_CHANNELS': 24, 'MINIMAX_H3_AUDIO_CHANNELS': 2, '_AUDIO_LATENT_CHANNELS': 32,
                 'video_latent_num_frames': lambda frames: frames // 17 * 5 + 2,
                 'audio_latent_num_frames': lambda frames: (frames * 32000 // 24 + 799) // 800}
    exec(compile(module_ast, str(path), 'exec'), namespace)
    model = SimpleNamespace(training_config=SimpleNamespace(data=SimpleNamespace(num_frames=124, num_latent_t=37)))
    geometry = {'height': 32, 'width': 32, 'num_frames': 362, 'fps': 24, 'generate_audio': True}
    video, audio = namespace['_resolve_clean_latents'](model, {'prompt_only': True, 'prompt_geometry': geometry},
                                                      'zeros', torch.float32, torch.device('cpu'))
    assert video.shape[2] == namespace['video_latent_num_frames'](362)
    assert audio.shape[-1] == namespace['audio_latent_num_frames'](362)
    assert model.training_config.data.num_frames == 124
    with pytest.raises(ValueError, match='zero placeholders'):
        namespace['_resolve_clean_latents'](model, {'prompt_only': True, 'prompt_geometry': geometry},
                                           'data', torch.float32, torch.device('cpu'))


def test_masked_resume_seeds_only_saved_adam_entries(monkeypatch, tmp_path):
    import torch
    from types import SimpleNamespace
    from fastvideo.train.methods.knowledge_distillation.minimax_h3_mask_recovery import MiniMaxH3MaskRecoveryMethod
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    optimizer = torch.optim.AdamW(model.parameters())
    obj = object.__new__(MiniMaxH3MaskRecoveryMethod)
    obj.student = SimpleNamespace(transformer=model)
    obj._student_optimizer = optimizer
    obj.training_config = SimpleNamespace(checkpoint=SimpleNamespace(resume_from_checkpoint=str(tmp_path)))
    keys = {'optimizers.student.state.transformer.0.weight.exp_avg': None,
            'optimizers.student.state.transformer.0.bias.exp_avg': None}
    monkeypatch.setattr(torch.distributed.checkpoint.FileSystemReader, 'read_metadata',
                        lambda self: SimpleNamespace(state_dict_metadata=keys))
    obj.seed_optimizer_state_for_resume()
    assert set(optimizer.state) == {model[0].weight, model[0].bias}
    assert all(s['exp_avg'].dtype == torch.float32 for s in optimizer.state.values())
    keys['optimizers.student.state.transformer.unknown.exp_avg'] = None
    with pytest.raises(ValueError, match='Unmatched saved Adam'):
        obj.seed_optimizer_state_for_resume()


@pytest.mark.parametrize('reset', [False, True])
def test_dataset_reset_preserves_optimizer_load(monkeypatch, tmp_path, reset):
    from types import SimpleNamespace
    from fastvideo.train.utils import checkpoint as ck
    manager = object.__new__(ck.CheckpointManager)
    manager.config = SimpleNamespace(reset_dataloader_on_resume=reset)
    manager.output_dir = str(tmp_path)
    resolved = tmp_path / 'checkpoint-200'
    states = {'dataloader': object(), 'optimizers.student': object(), 'roles.student.transformer': object()}
    manager._build_states = lambda: states.copy()
    manager._coordination_kwargs = lambda: {}
    monkeypatch.setattr(ck, '_resolve_resume_checkpoint', lambda *args, **kwargs: resolved)
    monkeypatch.setattr(ck, '_barrier', lambda: None)
    captured = []
    monkeypatch.setattr(ck.dcp, 'load', lambda data, **kwargs: captured.append(data))
    assert manager.maybe_resume(resume_from_checkpoint=str(resolved)) == 200
    assert ('dataloader' in captured[0]) is (not reset)
    assert captured[0]['optimizers.student'] is states['optimizers.student']
