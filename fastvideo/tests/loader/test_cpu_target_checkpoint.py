"""CPU-targeted inference must never stage its checkpoint on the GPU."""
from types import SimpleNamespace

import torch
from torch import nn
from safetensors.torch import save_file

from fastvideo.models.loader import fsdp_load


class TinyCPUModel(nn.Module):
    param_names_mapping = {}

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(8, 8)


def test_cpu_target_reads_cpu_checkpoint_and_preserves_weights(tmp_path, monkeypatch):
    weights = {"linear.weight": torch.arange(64, dtype=torch.float32).view(8, 8),
               "linear.bias": torch.arange(8, dtype=torch.float32)}
    path = tmp_path / "model.safetensors"
    save_file(weights, path)
    real_iterator = fsdp_load.safetensors_weights_iterator
    placements = []

    def cpu_iterator(files, *, to_cpu):
        assert to_cpu, "CPU target must not stage checkpoint tensors on the GPU"
        for name, tensor in real_iterator(files, to_cpu=to_cpu):
            placements.append(tensor.device.type)
            yield name, tensor

    monkeypatch.setattr(fsdp_load, "safetensors_weights_iterator", cpu_iterator)
    model = fsdp_load.maybe_load_fsdp_model(
        model_cls=TinyCPUModel,
        init_params={"config": SimpleNamespace(quant_config=None)},
        weight_dir_list=[str(path)], device=torch.device("cpu"),
        hsdp_replicate_dim=1, hsdp_shard_dim=1,
        default_dtype=torch.float32, param_dtype=torch.float32,
        reduce_dtype=torch.float32, training_mode=False, cpu_offload=False,
    )
    assert placements == ["cpu", "cpu"]
    for name, value in model.state_dict().items():
        assert value.device.type == "cpu"
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
    x = torch.randn(2, 8)
    torch.testing.assert_close(model.linear(x), torch.nn.functional.linear(x, weights["linear.weight"],
                                                                       weights["linear.bias"]), rtol=0, atol=0)
