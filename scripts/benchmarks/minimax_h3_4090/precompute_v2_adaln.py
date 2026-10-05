"""Exact V2 T2AV modulation tables for the checkpoint's fixed DMD ladder."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors import safe_open
from fastvideo.layers.mlp import MLP
from fastvideo.layers.visual_embedding import Timesteps
from fastvideo.models.dits.minimax_h3 import MiniMaxH3AdaLayerNormModulation
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.pipelines.basic.minimax_h3.packing import build_row_timesteps
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_denoising import MiniMaxH3DenoisingStage


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source-commit', help='FastVideo revision used to build the table')
    args = parser.parse_args()
    config = json.loads((args.model / 'transformer/config.json').read_text())
    contract = json.loads((args.model / 'fastvideo_inference.json').read_text())
    if config.get('adaln_rank') is not None or contract.get('task') != 't2av':
        raise ValueError('This tool supports full-rank V2 T2AV only')
    files = list((args.model / 'transformer').glob('*.safetensors'))
    locations = {}
    for path in files:
        with safe_open(path, framework='pt', device='cpu') as reader:
            for key in reader.keys():
                if 'adaln_proj.linear.' in key or key.startswith('time_embedder.'):
                    locations[key] = path

    def tensor(key, dtype=None):
        with safe_open(locations[key], framework='pt', device='cpu') as reader:
            return reader.get_tensor(key).to(device='cuda', dtype=dtype)

    time_proj = Timesteps(config['freq_dim'], flip_sin_to_cos=True, downscale_freq_shift=0)
    embedder = MLP(config['freq_dim'], config['time_embed_hidden_dim'], config['time_embed_dim'],
                   act_type='silu', dtype=torch.float32).cuda().eval()
    for destination, source in (('fc_in', 'linear_1'), ('fc_out', 'linear_2')):
        layer = getattr(embedder, destination)
        layer.weight.data = tensor(f'time_embedder.{source}.weight', torch.float32)
        layer.bias.data = tensor(f'time_embedder.{source}.bias', torch.float32)
    video = MiniMaxH3Scheduler(shift=contract['video_scheduler_shift'])
    audio = MiniMaxH3Scheduler(shift=contract['audio_scheduler_shift'])
    stage = MiniMaxH3DenoisingStage(None, video, audio)
    stage._set_dmd_schedule(contract['dmd_denoising_steps'], contract['num_inference_steps'], torch.device('cuda'))
    # No reference rows; text rows share the video timestep in the real packer.
    layout = SimpleNamespace(sequence_length=3, video_indices=torch.tensor([0]),
                             audio_indices=torch.tensor([1]), num_condition_video_rows=0,
                             num_condition_audio_rows=0)
    inputs, embeddings = {}, {}
    for vt, at in zip(video.timesteps, audio.timesteps, strict=True):
        unique, _ = build_row_timesteps(layout, float(vt), float(at), float(vt), 1.0)
        unique = unique.cuda()
        temb = embedder(time_proj(unique).to(embedder.fc_in.weight.dtype))
        key = repr((tuple(unique.reshape(-1).tolist()), tuple(temb.shape), str(temb.dtype)))
        embeddings[key] = temb
        inputs[key] = F.silu(temb)
    if len(inputs) != contract['transformer_forwards']:
        raise ValueError('The contract must supply one unique timestep key per transformer forward')
    tables = {}
    for block in range(config['num_layers']):
        prefix = f'transformer_blocks.{block}.adaln_proj.linear'
        weight, bias = tensor(prefix + '.weight'), tensor(prefix + '.bias')
        # Use the original release module as an independent arithmetic check.
        with torch.device('meta'):
            reference = MiniMaxH3AdaLayerNormModulation(config['time_embed_dim'], config['hidden_size'])
        reference.linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        reference.linear.bias = torch.nn.Parameter(bias, requires_grad=False)
        tables[block] = {}
        for key, x in inputs.items():
            result = F.linear(x.to(weight.dtype), weight, bias)
            expected = torch.cat(reference(embeddings[key]), dim=-1).view_as(result)
            torch.testing.assert_close(result, expected, atol=0, rtol=0)
            tables[block][key] = result.cpu()
        del reference, weight, bias
        print('BLOCK_VALIDATED', block, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tables, args.output)
    torch.save({key: value.cpu() for key, value in inputs.items()}, str(args.output) + '.inputs')
    revision_file = args.model / '.cache/huggingface/download/fastvideo_inference.json.metadata'
    metadata = {'config_sha256': hashlib.sha256((args.model / 'transformer/config.json').read_bytes()).hexdigest(),
                'contract_sha256': hashlib.sha256((args.model / 'fastvideo_inference.json').read_bytes()).hexdigest(),
                'model_revision': revision_file.read_text().splitlines()[0] if revision_file.is_file() else None,
                'source_commit': args.source_commit, 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(0),
                'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'contract': contract, 'blocks': len(tables), 'timestep_keys': list(inputs),
                'table_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                'validation': 'Every block/rung exactly equals the original release modulation module on sm89.'}
    Path(str(args.output) + '.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('TABLE_DONE', args.output, flush=True)


if __name__ == '__main__':
    main()
