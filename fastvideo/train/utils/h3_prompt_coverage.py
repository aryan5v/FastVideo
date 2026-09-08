"""Record supervision actually consumed, separately from dataset availability."""
import json
from pathlib import Path
from typing import Any

import torch.distributed as dist


def record_prompt_use(method: Any, batch: dict[str, Any], iteration: int) -> dict[str, float]:
    rank = dist.get_rank() if dist.is_initialized() else 0
    if method.student.sp_group.rank_in_group != 0:
        return {}
    root = Path(method.training_config.checkpoint.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"consumed_prompts_rank{rank}.jsonl"
    if not hasattr(method, "_consumed_prompt_ids"):
        method._consumed_prompt_ids = set()
        if path.exists():
            method._consumed_prompt_ids.update(json.loads(line)["id"] for line in path.read_text().splitlines())
    for info in batch.get("info_list", []):
        identifier = str(info["id"])
        method._consumed_prompt_ids.add(identifier)
        with path.open("a") as stream:
            stream.write(
                json.dumps({
                    "iteration": iteration,
                    "id": identifier,
                    "prompt_only": bool(batch.get("prompt_only", False))
                }) + "\n")
    return {"unique_prompts_this_dp_group": float(len(method._consumed_prompt_ids))}
