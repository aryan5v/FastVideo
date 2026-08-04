# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from fastvideo.logger import init_logger

logger = init_logger(__name__)

_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")


def _is_stateful(obj: Any) -> bool:
    return callable(getattr(obj, "state_dict", None)) and callable(getattr(obj, "load_state_dict", None))


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _release_cuda_cache(where: str = "") -> None:
    """Drop allocator caches before DCP gather/barrier spikes.

    Steady-state training can fit while checkpoint all-gathers + NCCL
    barriers OOM on the same mesh (5B job 1111 died on the step-100
    barrier with ranks reporting Cuda failure 2). Emptying the cache
    first is cheap and recovers fragmentation from the just-finished step.
    """
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - best-effort before a memory-critical path
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    # ipc_collect is not on every build; ignore if missing.
    ipc_collect = getattr(torch.cuda, "ipc_collect", None)
    if callable(ipc_collect):
        try:
            ipc_collect()
        except Exception:  # noqa: BLE001
            pass
    if where and _rank() == 0:
        try:
            free, total = torch.cuda.mem_get_info()
            logger.info(
                "CUDA cache released before %s (device free %.2f / %.2f GiB)",
                where,
                free / (1024 ** 3),
                total / (1024 ** 3),
            )
        except Exception:  # noqa: BLE001
            logger.info("CUDA cache released before %s", where)


def _parse_step_from_dir(checkpoint_dir: Path) -> int:
    match = _CHECKPOINT_DIR_RE.match(checkpoint_dir.name)
    if not match:
        raise ValueError(f"Invalid checkpoint directory name {checkpoint_dir.name!r}; "
                         "expected 'checkpoint-<step>'")
    return int(match.group(1))


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None

    candidates: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        if not _CHECKPOINT_DIR_RE.match(child.name):
            continue
        if not (child / "dcp").is_dir():
            continue
        try:
            step = _parse_step_from_dir(child)
        except Exception:
            continue
        candidates.append((step, child))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _resolve_resume_checkpoint(resume_from_checkpoint: str, *, output_dir: str) -> Path | None:
    """Resolve a user-provided resume path to a concrete checkpoint dir.

    Accepted values:
    - "latest" (auto-pick latest checkpoint-*/dcp under output_dir,
      or ``None`` if no checkpoint exists yet — starts from scratch)
    - /path/to/output_dir/checkpoint-<step>
    - /path/to/output_dir/checkpoint-<step>/dcp
    - /path/to/output_dir (auto-pick latest checkpoint-*/dcp)
    """

    if str(resume_from_checkpoint).strip().lower() == "latest":
        out = Path(os.path.expanduser(str(output_dir))).resolve()
        latest = _find_latest_checkpoint(out)
        if latest is None:
            logger.info(
                "resume_from_checkpoint='latest' but no "
                "checkpoints found under %s; starting from "
                "scratch.",
                out,
            )
        return latest

    raw = os.path.expanduser(str(resume_from_checkpoint))
    path = Path(raw).resolve()
    if not path.exists():
        raise FileNotFoundError(f"resume_from_checkpoint not found: {path}")

    if path.is_dir() and path.name == "dcp":
        path = path.parent

    if path.is_dir() and _CHECKPOINT_DIR_RE.match(path.name):
        if not (path / "dcp").is_dir():
            raise FileNotFoundError(f"Missing dcp dir under checkpoint: {path / 'dcp'}")
        return path

    # Treat as output_dir -> pick latest.
    latest = _find_latest_checkpoint(path)
    if latest is not None:
        return latest

    # Give a clearer error message.
    out = Path(os.path.expanduser(str(output_dir))).resolve()
    raise ValueError("Could not resolve resume checkpoint. Expected a checkpoint directory "
                     f"named 'checkpoint-<step>' (with 'dcp/' inside), or an output_dir "
                     f"containing such checkpoints. Got: {path} (output_dir={out}).")


class _RoleModuleContainer(torch.nn.Module):
    """Ephemeral container to expose multiple role modules as a single
    ``nn.Module``.

    Used by ``OptimizerWrapper`` which expects a single root module
    covering all parameters owned by the optimizer.
    """

    def __init__(self, modules: dict[str, torch.nn.Module]) -> None:
        super().__init__()
        for name, module in modules.items():
            self.add_module(name, module)


class _CallbackStateWrapper:
    """Wraps a CallbackDict for DCP save/load."""

    def __init__(self, callbacks: Any) -> None:
        self._callbacks = callbacks

    def state_dict(self) -> dict[str, Any]:
        return self._callbacks.state_dict()

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        self._callbacks.load_state_dict(state_dict)


@dataclass(slots=True)
class CheckpointConfig:
    save_steps: int
    keep_last: int


class CheckpointManager:
    """Role-based checkpoint manager for training runtime.

    - Checkpoint policy lives in YAML (via TrainingArgs fields).
    - Resume path is typically provided via CLI (``--resume-from-checkpoint``).
    """

    def __init__(
        self,
        *,
        method: Any,
        dataloader: Any,
        output_dir: str,
        config: CheckpointConfig,
        callbacks: Any | None = None,
        raw_config: dict[str, Any] | None = None,
    ) -> None:
        self.method = method
        self.dataloader = dataloader
        self.output_dir = str(output_dir)
        self.config = config
        self._callbacks = callbacks
        self._raw_config = raw_config
        self._last_saved_step: int | None = None

    def _build_states(self) -> dict[str, Any]:
        states: dict[str, Any] = self.method.checkpoint_state()

        # Dataloader (optional but recommended for exact resume).
        if _is_stateful(self.dataloader):
            states["dataloader"] = self.dataloader

        # Callback state (e.g. EMA shadow weights, validation RNG).
        if self._callbacks is not None and _is_stateful(self._callbacks):
            states["callbacks"] = _CallbackStateWrapper(self._callbacks, )

        return states

    def _checkpoint_dir(self, step: int) -> Path:
        return Path(self.output_dir) / f"checkpoint-{step}"

    def _dcp_dir(self, step: int) -> Path:
        return self._checkpoint_dir(step) / "dcp"

    def _checkpoint_looks_complete(self, step: int) -> bool:
        """True if a prior save left a usable DCP tree for ``step``.

        Used to skip re-saving after a crash that finished shard I/O but
        died on the trailing NCCL barrier (job 1111 / checkpoint-100).
        """
        dcp_dir = self._dcp_dir(step)
        if not dcp_dir.is_dir():
            return False
        if not (dcp_dir / ".metadata").is_file():
            return False
        shards = list(dcp_dir.glob("*.distcp"))
        if not shards:
            return False
        # Require every shard to be non-trivial (partial writes are tiny).
        return all(p.stat().st_size > 1_000_000 for p in shards)

    def maybe_save(self, step: int) -> None:
        save_steps = int(self.config.save_steps or 0)
        if save_steps <= 0:
            return
        if step % save_steps != 0:
            return
        if self._last_saved_step == step:
            return
        if self._checkpoint_looks_complete(step):
            if _rank() == 0:
                logger.info(
                    "Skipping checkpoint-%s save; complete DCP tree already on disk",
                    step,
                )
            self._last_saved_step = step
            _barrier()
            return
        self.save(step)

    def save_final(self, step: int) -> None:
        save_steps = int(self.config.save_steps or 0)
        if save_steps <= 0:
            return
        self.save(step)

    def save(self, step: int) -> None:
        checkpoint_dir = self._checkpoint_dir(step)
        dcp_dir = self._dcp_dir(step)
        os.makedirs(dcp_dir, exist_ok=True)

        # Free step fragmentation before DCP all-gather / NCCL barrier.
        _release_cuda_cache(f"checkpoint-{step} dcp.save")
        _barrier()

        states = self._build_states()
        if _rank() == 0:
            logger.info(
                "Saving checkpoint to %s",
                checkpoint_dir,
            )
            self._write_metadata(checkpoint_dir, step)
        dcp.save(states, checkpoint_id=str(dcp_dir))
        _release_cuda_cache(f"checkpoint-{step} post-dcp")
        _barrier()

        # Save RNG state AFTER dcp.save so it captures the
        # exact state the continuous run continues with.
        # dcp.save triggers FSDP all-gather ops that can
        # advance the RNG between when DCP captures it and
        # when the save completes.
        self._save_rng_snapshot(checkpoint_dir)
        _barrier()

        self._last_saved_step = step

        self._cleanup_old_checkpoints()

    def _write_metadata(
        self,
        checkpoint_dir: Path,
        step: int,
    ) -> None:
        metadata: dict[str, Any] = {"step": step}
        if self._raw_config is not None:
            metadata["config"] = self._raw_config
        meta_path = checkpoint_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def load_metadata(checkpoint_dir: str | Path, ) -> dict[str, Any]:
        """Read ``metadata.json`` from a checkpoint dir."""
        meta_path = Path(checkpoint_dir) / "metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"No metadata.json in {checkpoint_dir}")
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]

    def _save_rng_snapshot(self, checkpoint_dir: Path) -> None:
        """Save per-rank RNG state to a separate file.

        Called AFTER ``dcp.save`` so the snapshot reflects
        the exact state the continuous run continues with.
        Each rank saves its own file because CUDA RNG and
        custom generators differ across ranks.
        """
        rank = _rank()
        rng: dict[str, Any] = {
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        }
        rng["cuda_rng"] = torch.cuda.get_rng_state()
        rng["gen_cuda"] = self.method.cuda_generator.get_state()
        torch.save(
            rng,
            checkpoint_dir / f"rng_state_rank{rank}.pt",
        )

    def load_rng_snapshot(
        self,
        checkpoint_path: str,
    ) -> None:
        """Restore per-rank RNG state from the snapshot file.

        Must be called AFTER ``dcp.load`` **and** after
        ``iter(dataloader)`` so no later operation can
        clobber the restored state.
        """
        resolved = _resolve_resume_checkpoint(
            checkpoint_path,
            output_dir=self.output_dir,
        )
        if resolved is None:
            return
        rank = _rank()
        rng_path = resolved / f"rng_state_rank{rank}.pt"
        if not rng_path.is_file():
            # Fall back to legacy single-file snapshot.
            rng_path = resolved / "rng_state.pt"
        if not rng_path.is_file():
            logger.warning(
                "No rng_state in %s; skipping "
                "RNG snapshot restore.",
                resolved,
            )
            return

        rng = torch.load(
            rng_path,
            map_location="cpu",
            weights_only=False,
        )
        if "torch_rng" in rng:
            torch.set_rng_state(rng["torch_rng"])
        if "python_rng" in rng:
            random.setstate(rng["python_rng"])
        if "numpy_rng" in rng:
            np.random.set_state(rng["numpy_rng"])

        torch.cuda.set_rng_state(rng["cuda_rng"])
        self.method.cuda_generator.set_state(rng["gen_cuda"])
        logger.info(
            "Restored RNG snapshot from %s",
            rng_path,
        )

    def _mark_ema_needs_reinit(self) -> None:
        """Defer EMA shadow rebuild to the first training step.

        Cloning every student shard to CPU right after a multi-role DCP load
        spikes allocator pressure and was OOM'ing the post-load NCCL barrier
        on 5B resume (job 1122). ``EMACallback.on_training_step_end`` already
        re-inits when ``_ema_started`` is False — use that path instead.
        """
        if self._callbacks is None:
            return
        cbs = getattr(self._callbacks, "_callbacks", {}) or {}
        cb_iter = cbs.values() if isinstance(cbs, dict) else cbs
        marked = False
        for cb in cb_iter:
            if getattr(cb, "student_ema", None) is None:
                continue
            # Drop any half-applied / empty shadow from a failed EMA load.
            if getattr(cb.student_ema, "shadow", None) is not None:
                cb.student_ema.shadow = {}
            cb._ema_started = False
            marked = True
        if marked and _rank() == 0:
            logger.warning(
                "EMA load skipped; shadow will re-init from student on the "
                "first training step (decay unchanged).",
            )

    def maybe_resume(self, *, resume_from_checkpoint: str | None) -> int | None:
        if not resume_from_checkpoint:
            return None

        resolved = _resolve_resume_checkpoint(
            resume_from_checkpoint,
            output_dir=self.output_dir,
        )
        if resolved is None:
            return None
        step = _parse_step_from_dir(resolved)

        states = self._build_states()
        logger.info("Loading Phase 2 checkpoint from %s", resolved)

        # checkpoint-100 (5B job 1111) wrote empty FSDP local shards for root
        # EMA params (scale_shift_table [0,2,3072] vs live [4,2,3072]). DCP's
        # planner raises CheckpointException *before* any load_state_dict runs.
        #
        # Policy: load roles/optim/dataloader WITHOUT callback/EMA state.
        # EMA is marked for lazy re-init on the first train step (same decay).
        # empty_cache around load — DCP + 3 QAD roles is the memory cliff.
        skip_ema = os.environ.get("FASTVIDEO_RESUME_SKIP_EMA", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        states_core = dict(states)
        callback_state = states_core.pop("callbacks", None)

        _release_cuda_cache("pre-dcp-load")
        _barrier()
        dcp.load(states_core, checkpoint_id=str(resolved / "dcp"))
        _release_cuda_cache("post-dcp-load")
        _barrier()

        if callback_state is not None and not skip_ema:
            try:
                dcp.load({"callbacks": callback_state}, checkpoint_id=str(resolved / "dcp"))
                _release_cuda_cache("post-ema-dcp-load")
            except Exception as exc:  # noqa: BLE001 - any DCP/EMA planner failure
                if _rank() == 0:
                    logger.warning(
                        "EMA DCP load failed (%s: %s); will re-init on first step.",
                        type(exc).__name__,
                        str(exc).splitlines()[0][:240],
                    )
                self._mark_ema_needs_reinit()
        elif callback_state is not None and skip_ema:
            if _rank() == 0:
                logger.warning(
                    "Skipping EMA load from checkpoint (FASTVIDEO_RESUME_SKIP_EMA=1).",
                )
            self._mark_ema_needs_reinit()

        _release_cuda_cache("pre-resume-barrier")
        _barrier()
        logger.info("Checkpoint loaded; resuming from step=%s", step)
        return step

    def _cleanup_old_checkpoints(self) -> None:
        keep_last = int(self.config.keep_last or 0)
        if keep_last <= 0:
            return

        if _rank() != 0:
            _barrier()
            return

        output_dir = Path(self.output_dir)
        candidates: list[tuple[int, Path]] = []
        for child in output_dir.iterdir():
            if not child.is_dir():
                continue
            if not _CHECKPOINT_DIR_RE.match(child.name):
                continue
            try:
                step = _parse_step_from_dir(child)
            except Exception:
                continue
            candidates.append((step, child))

        candidates.sort(key=lambda x: x[0])
        to_delete = candidates[:-keep_last] if len(candidates) > keep_last else []
        for step, path in to_delete:
            logger.info("Removing old checkpoint (keep_last=%s): %s", keep_last, path)
            shutil.rmtree(path, ignore_errors=True)

        _barrier()
