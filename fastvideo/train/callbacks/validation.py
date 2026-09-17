# SPDX-License-Identifier: Apache-2.0
"""Validation callback.

All configuration is read from the YAML ``callbacks.validation``
section.  The pipeline class is resolved from
``pipeline_target``.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
import torchvision
from einops import rearrange
from torch.utils.data import DataLoader

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.dataset.validation_dataset import (
    ValidationDataset, )
from fastvideo.distributed import (
    get_sp_group,
    get_world_group,
)
from fastvideo.logger import init_logger
from fastvideo.pipelines import ForwardBatch
from fastvideo.train.callbacks.callback import Callback
from fastvideo.train.utils.instantiate import resolve_target
from fastvideo.train.utils.moduleloader import (
    make_inference_args, )
from fastvideo.train.utils.validation_media import write_validation_mp4
from fastvideo.training.trackers import DummyTracker
from fastvideo.utils import shallow_asdict

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)


@dataclass(slots=True)
class _ValidationStepResult:
    """Keep decoded media aligned with the prompt and conditions that produced it."""

    videos: list[list[np.ndarray]]
    captions: list[str]
    audio_waveforms: list[torch.Tensor | np.ndarray | None] = field(default_factory=list)
    audio_sample_rates: list[int | None] = field(default_factory=list)
    overlay_videos: list[list[np.ndarray]] = field(default_factory=list)
    overlay_captions: list[str] = field(default_factory=list)
    ref_videos: list[str | None] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any] | None] = field(default_factory=list)
    mouse_pitch_signs: list[int | None] = field(default_factory=list)


@dataclass(slots=True)
class _ValidationMetricStats:
    sums: dict[str, float] = field(default_factory=dict)
    counts: dict[str, float] = field(default_factory=dict)
    per_video: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _SavedValidationVideos:
    """Identify verified MP4 files and their positions in the generated batch."""

    filenames: list[str] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    audio_video_count: int = 0


@dataclass(slots=True)
class _ValidationMetricsConfig:
    enabled: bool = False
    names: list[str] = field(default_factory=list)
    device: str = "cuda"
    calibration_path: str | None = None
    mouse_pitch_sign: int = 1
    skip_missing_deps: bool = True
    strict: bool = False
    unload_after_validation: bool = True
    loader_threads: int = 1
    prefetch_factor: int = 2
    log_prefix: str = "metrics/validation"


DEFAULT_VALIDATION_VBENCH_METRICS = [
    "vbench.imaging_quality",
    "vbench.aesthetic_quality",
    "vbench.temporal_flickering",
    "vbench.motion_smoothness",
    "vbench.subject_consistency",
    "vbench.background_consistency",
    "vbench.dynamic_degree",
]

SYNTHETIC_OPTICAL_FLOW_METRIC = "optical_flow.synthetic_optical_flow"
SYNTHETIC_OPTICAL_FLOW_LOG_KEYS = (
    "mf_epe_mean",
    "mf_angle_err_mean",
    "mf_cosine_mean",
    "mf_mag_ratio_mean",
    "pixel_epe_mean_mean",
    "px_angle_rmse_mean",
    "fl_all_mean",
    "foe_dist_mean",
    "flow_kl_2d_mean",
)


class ValidationCallback(Callback):
    """Generic validation callback driven entirely by YAML
    config.

    Works with any pipeline that follows the
    ``PipelineCls.from_pretrained(...)`` + ``pipeline.forward()``
    contract.
    """

    def __init__(
        self,
        *,
        pipeline_target: str,
        dataset_file: str,
        every_steps: int = 100,
        run_at_start: bool = True,
        sampling_steps: list[int] | None = None,
        guidance_scale: float | None = None,
        num_frames: int | None = None,
        use_record_dimensions: bool = False,
        max_record_num_frames: int | None = None,
        num_videos_per_prompt: int = 1,
        use_validation_media_conditioning: bool = True,
        output_dir: str | None = None,
        sampling_timesteps: list[int] | None = None,
        overlay_actions: bool = False,
        keyboard_value_scale: float = 1.0,
        offload_training_state: bool = False,
        unload_pipeline_after_validation: bool = False,
        attn_qat_infer: bool = False,
        **pipeline_kwargs: Any,
    ) -> None:
        """Configure validation cadence, generation parameters, and pipeline loading.

        ``run_at_start`` controls the pre-training baseline event.
        ``use_validation_media_conditioning`` lets text-to-video recipes use
        captions from a dataset that also contains source-media paths.
        """
        self.pipeline_target = str(pipeline_target)
        self.dataset_file = str(dataset_file)
        self.every_steps = int(every_steps)
        self.run_at_start = self._coerce_bool(run_at_start)
        self.sampling_steps = ([int(s) for s in sampling_steps] if sampling_steps else [40])
        self.guidance_scale = (float(guidance_scale) if guidance_scale is not None else None)
        self.num_frames = (int(num_frames) if num_frames is not None else None)
        self.use_record_dimensions = self._coerce_bool(use_record_dimensions)
        self.max_record_num_frames = (int(max_record_num_frames) if max_record_num_frames is not None else None)
        if self.max_record_num_frames is not None and self.max_record_num_frames <= 0:
            raise ValueError("callbacks.validation.max_record_num_frames must be positive")
        self.num_videos_per_prompt = int(num_videos_per_prompt)
        if self.num_videos_per_prompt <= 0:
            raise ValueError("callbacks.validation.num_videos_per_prompt must be positive")
        self.use_validation_media_conditioning = self._coerce_bool(use_validation_media_conditioning)
        self.output_dir = (str(output_dir) if output_dir is not None else None)
        self.sampling_timesteps = ([int(s) for s in sampling_timesteps] if sampling_timesteps is not None else None)
        self.overlay_actions = self._coerce_bool(overlay_actions)
        self.keyboard_value_scale = float(keyboard_value_scale)
        metrics_config = pipeline_kwargs.pop("metrics", None)
        self.metrics_config = self._parse_metrics_config(metrics_config)
        self.offload_training_state = self._coerce_bool(offload_training_state)
        self.unload_pipeline_after_validation = self._coerce_bool(unload_pipeline_after_validation)
        self.attn_qat_infer = self._coerce_bool(attn_qat_infer)
        self.pipeline_kwargs = dict(pipeline_kwargs)

        self._pipeline: Any | None = None
        self._pipeline_key: tuple[Any, ...] | None = None
        self._sampling_param: SamplingParam | None = None
        self._metric_evaluator: Any | None = None
        self.tracker: Any = DummyTracker()
        self.validation_random_generator: (torch.Generator | None) = None
        self.seed: int = 0

    @staticmethod
    def _parse_metrics_config(config: Any) -> _ValidationMetricsConfig:
        if config is None or config is False:
            return _ValidationMetricsConfig(enabled=False)
        if config is True:
            return _ValidationMetricsConfig(
                enabled=True,
                names=list(DEFAULT_VALIDATION_VBENCH_METRICS),
            )
        if isinstance(config, str):
            return _ValidationMetricsConfig(
                enabled=True,
                names=[config],
            )
        if isinstance(config, list | tuple):
            return _ValidationMetricsConfig(
                enabled=bool(config),
                names=[str(name) for name in config],
            )
        if not isinstance(config, dict):
            raise TypeError("callbacks.validation.metrics must be a bool, string, list, or mapping")

        enabled = bool(config.get("enabled", True))
        raw_names = config.get("names", config.get("metrics", None))
        if raw_names is None:
            names = list(DEFAULT_VALIDATION_VBENCH_METRICS)
        elif isinstance(raw_names, str):
            names = [raw_names]
        else:
            names = [str(name) for name in raw_names]

        return _ValidationMetricsConfig(
            enabled=enabled and bool(names),
            names=names,
            device=str(config.get("device", "cuda")),
            calibration_path=(str(config["calibration_path"]) if config.get("calibration_path") is not None else None),
            mouse_pitch_sign=int(config.get("mouse_pitch_sign", 1)),
            skip_missing_deps=bool(config.get("skip_missing_deps", True)),
            strict=bool(config.get("strict", False)),
            unload_after_validation=bool(config.get("unload_after_validation", True)),
            loader_threads=int(config.get("loader_threads", 1)),
            prefetch_factor=int(config.get("prefetch_factor", 2)),
            log_prefix=str(config.get("log_prefix", "metrics/validation")),
        )

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)


    @staticmethod
    def _assert_attention_contract(
        inference_args: Any,
        tc: Any,
    ) -> None:
        """Fail if validation would sample off the training attention contract.

        Sparsity and tile geometry are separate knobs and both have to survive
        the training-config to inference-args hop: v8 validated a
        tile-64-trained student at the tile-256 default because only the
        sparsity was propagated, and nothing downstream noticed.
        """
        for train_attr, args_attr in (
            ("vsa_sparsity", "VSA_sparsity"),
            ("vsa_tile_size", "VSA_tile_size"),
        ):
            trained = getattr(tc, train_attr, None)
            sampled = getattr(inference_args, args_attr, None)
            if trained is None or sampled is None:
                continue
            if sampled != trained:
                raise ValueError(f"validation would sample at {args_attr}={sampled} while "
                                 f"training runs {train_attr}={trained}. The attention "
                                 "contract must match; fix the training-config to "
                                 "inference-args propagation rather than the symptom.")

    def _adopt_training_sampling_contract(
        self,
        method: TrainingMethod,
    ) -> None:
        """Keep validation sampling on the operating point training teaches.

        A few-step method trains its student on an explicit timestep ladder.
        Validation only reaches that ladder when ``sampling_timesteps`` is
        configured: ``sampling_steps`` sets ``num_inference_steps``, and the
        public scheduler turns N of those into an N-point sigma grid, i.e.
        N-1 forwards on the scheduler's own spacing. Duplicating the ladder by
        hand in the callback config is the footgun that silently validated v8
        at three forwards on the wrong grid for its whole run, so derive it
        from the method instead, and refuse to run when the two disagree.
        """
        method_config = getattr(method, "method_config", None)
        if not isinstance(method_config, dict):
            return
        raw = method_config.get("dmd_denoising_steps")
        if not isinstance(raw, list) or not raw:
            return
        trained = [int(s) for s in raw]

        if self.sampling_timesteps is None:
            self.sampling_timesteps = trained
            logger.info(
                "validation: adopting the trained denoising ladder %s "
                "(%d forwards) from the training method",
                trained,
                len(trained),
            )
            return
        if self.sampling_timesteps != trained:
            raise ValueError("callbacks.validation.sampling_timesteps "
                             f"{self.sampling_timesteps} disagrees with the trained ladder "
                             f"{trained} (method.dmd_denoising_steps). Validation would "
                             "sample off the operating point the student was distilled "
                             "for. Drop the callback override to inherit the ladder, or "
                             "align the two deliberately.")

    def on_train_start(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        self.method = method
        tc = self.training_config
        self._adopt_training_sampling_contract(method)

        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self.global_rank = self.world_group.rank
        self.rank_in_sp_group = (self.sp_group.rank_in_group)
        self.sp_world_size = self.sp_group.world_size

        seed = tc.data.seed
        if seed is None:
            raise ValueError("training.data.seed must be set "
                             "for validation")
        self.seed = int(seed)
        self.validation_random_generator = (torch.Generator(device="cpu").manual_seed(self.seed))

        tracker = getattr(method, "tracker", None)
        if tracker is not None:
            self.tracker = tracker

    def on_validation_begin(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        """Run the optional step-zero baseline and each scheduled validation event."""
        if not self.will_run_validation(iteration):
            return

        self._run_validation(method, iteration)

    def will_run_validation(self, iteration: int = 0) -> bool:
        """Return whether this callback schedules validation at ``iteration``."""
        if self.every_steps <= 0:
            return False
        if iteration == 0 and not self.run_at_start:
            return False
        return iteration % self.every_steps == 0


    def _run_validation(
        self,
        method: TrainingMethod,
        step: int,
    ) -> None:

        transformer = method.student.transformer
        try:
            with self._validation_memory_context(
                    method,
                    validation_transformer=transformer,
            ):
                ema_cb = self._find_ema_callback()
                ctx = ema_cb.ema_context(transformer) if ema_cb is not None else contextlib.nullcontext(transformer)
                with ctx as t, self._attn_qat_infer_context(t):
                    self._run_validation_inner(
                        method,
                        step,
                        t,
                    )
        finally:
            if self.unload_pipeline_after_validation:
                self._clear_pipeline_cache()

    @contextlib.contextmanager
    def _attn_qat_infer_context(self, transformer: torch.nn.Module):
        if not self.attn_qat_infer:
            yield
            return

        from fastvideo.attention.backends.attn_qat_infer import (AttnQatInferImpl, is_attn_qat_infer_available)
        from fastvideo.attention.backends.attn_qat_train import (
            AttnQatTrainImpl, )
        from fastvideo.platforms import AttentionBackendEnum

        layers = [
            module for module in transformer.modules()
            if isinstance(getattr(module, "attn_impl", None), AttnQatTrainImpl)
        ]
        if not layers:
            raise RuntimeError("attn_qat_infer validation requested, but the transformer has no ATTN_QAT_TRAIN layers")
        if not is_attn_qat_infer_available():
            from fastvideo.attention.backends.attn_qat_infer import (
                attn_qat_infer_receipt, )
            raise RuntimeError("attn_qat_infer validation requested but no ATTN_QAT_INFER kernel serves "
                               f"this device ({attn_qat_infer_receipt()}). Set "
                               "callbacks.validation.attn_qat_infer=false to validate with ATTN_QAT_TRAIN.")

        previous = [(layer, layer.attn_impl, layer.backend) for layer in layers]
        try:
            for layer, impl, _ in previous:
                layer.attn_impl = AttnQatInferImpl(
                    num_heads=layer.num_heads,
                    head_size=layer.head_size,
                    num_kv_heads=layer.num_kv_heads,
                    causal=impl.causal,
                    softmax_scale=layer.softmax_scale,
                )
                layer.backend = AttentionBackendEnum.ATTN_QAT_INFER
            logger.info("Enabled ATTN_QAT_INFER for %d validation attention layers.", len(layers))
            yield
        finally:
            for layer, impl, backend in previous:
                layer.attn_impl = impl
                layer.backend = backend

    @contextlib.contextmanager
    def _validation_memory_context(
        self,
        method: TrainingMethod,
        *,
        validation_transformer: torch.nn.Module,
    ):
        if not self.offload_training_state:
            yield
            return

        optimizer_tensor_records: list[tuple[Any, Any, torch.device]] = []
        module_records: list[tuple[str, torch.nn.Module, torch.device]] = []
        try:
            self._offload_optimizer_states_to_cpu(
                method,
                optimizer_tensor_records,
            )
            self._offload_inactive_role_modules_to_cpu(
                method,
                validation_transformer=validation_transformer,
                module_records=module_records,
            )
            self._empty_cuda_cache()
            yield
        finally:
            self._restore_inactive_role_modules(module_records)
            self._restore_optimizer_states(optimizer_tensor_records)
            self._empty_cuda_cache()
            off_cuda = [
                f"{name} on {getattr(p, '_local_tensor', p).device}"
                for name, p in validation_transformer.named_parameters()
                if getattr(p, "_local_tensor", p).device.type != "cuda"
            ]
            if off_cuda:
                logger.warning(
                    "Post-validation: %d validation-transformer params off-CUDA; first: %s",
                    len(off_cuda),
                    off_cuda[:5],
                )

    def _offload_optimizer_states_to_cpu(
        self,
        method: TrainingMethod,
        records: list[tuple[Any, Any, torch.device]],
    ) -> None:
        optimizers = getattr(method, "_optimizer_dict", {})
        if not optimizers:
            return
        moved = 0
        for optimizer in optimizers.values():
            state = getattr(optimizer, "state", None)
            if not isinstance(state, dict):
                continue
            for param_state in state.values():
                moved += self._offload_tensor_container_to_cpu(
                    param_state,
                    records,
                )
        if moved:
            logger.info(
                "Offloaded %d optimizer state tensors to CPU for validation.",
                moved,
            )

    def _offload_tensor_container_to_cpu(
        self,
        obj: Any,
        records: list[tuple[Any, Any, torch.device]],
    ) -> int:
        moved = 0
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                if torch.is_tensor(value) and value.device.type == "cuda":
                    records.append((obj, key, value.device))
                    obj[key] = value.detach().cpu()
                    moved += 1
                else:
                    moved += self._offload_tensor_container_to_cpu(value, records)
            return moved
        if isinstance(obj, list):
            for idx, value in enumerate(list(obj)):
                if torch.is_tensor(value) and value.device.type == "cuda":
                    records.append((obj, idx, value.device))
                    obj[idx] = value.detach().cpu()
                    moved += 1
                else:
                    moved += self._offload_tensor_container_to_cpu(value, records)
        return moved

    def _restore_optimizer_states(
        self,
        records: list[tuple[Any, Any, torch.device]],
    ) -> None:
        for container, key, device in reversed(records):
            value = container[key]
            if torch.is_tensor(value):
                container[key] = value.to(device=device)
        if records:
            logger.info(
                "Restored %d optimizer state tensors after validation.",
                len(records),
            )

    def _offload_inactive_role_modules_to_cpu(
        self,
        method: TrainingMethod,
        *,
        validation_transformer: torch.nn.Module,
        module_records: list[tuple[str, torch.nn.Module, torch.device]],
    ) -> None:
        role_models = getattr(method, "_role_models", {})
        if not isinstance(role_models, dict):
            return

        for role, model in role_models.items():
            module = getattr(model, "transformer", None)
            if not isinstance(module, torch.nn.Module):
                continue
            if module is validation_transformer:
                continue
            device = self._first_cuda_tensor_device(module)
            if device is None:
                continue
            if self._is_fsdp_managed(module):
                logger.info(
                    "Keeping role %r transformer on %s during validation "
                    "(FSDP-managed modules do not survive .to() round-trips).",
                    role,
                    device,
                )
                continue
            try:
                module.to("cpu")
            except Exception as exc:
                logger.warning(
                    "Could not offload role %r transformer to CPU before validation: %s",
                    role,
                    exc,
                )
                continue
            module_records.append((str(role), module, device))
            logger.info(
                "Offloaded role %r transformer from %s to CPU for validation.",
                role,
                device,
            )

    def _restore_inactive_role_modules(
        self,
        module_records: list[tuple[str, torch.nn.Module, torch.device]],
    ) -> None:
        for role, module, device in reversed(module_records):
            module.to(device)
            logger.info(
                "Restored role %r transformer to %s after validation.",
                role,
                device,
            )

    @staticmethod
    def _is_fsdp_managed(module: torch.nn.Module) -> bool:
        try:
            from torch.distributed.fsdp import FSDPModule
            from torch.distributed.tensor import DTensor
        except ImportError:
            return False
        if isinstance(module, FSDPModule):
            return True
        return any(isinstance(p, DTensor) for p in module.parameters(recurse=True))

    @staticmethod
    def _first_cuda_tensor_device(module: torch.nn.Module) -> torch.device | None:
        for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
            device = getattr(tensor, "device", None)
            if isinstance(device, torch.device) and device.type == "cuda":
                return device
        return None

    def _clear_pipeline_cache(self) -> None:
        self._pipeline = None
        self._pipeline_key = None
        self._empty_cuda_cache()

    @staticmethod
    def _empty_cuda_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _find_ema_callback(self) -> Any | None:
        """Find the EMA callback in the callback dict."""
        from fastvideo.train.callbacks.ema import (
            EMACallback, )

        cb_dict = getattr(self, "_callback_dict", None)
        if cb_dict is not None:
            for cb in cb_dict._callbacks.values():
                if isinstance(cb, EMACallback):
                    return cb
        return None

    def _run_validation_inner(
        self,
        method: TrainingMethod,
        step: int,
        transformer: torch.nn.Module,
    ) -> None:
        """Generate one event and gather sequence-parallel group outputs for logging.

        Each sequence-parallel group leader saves its local media. Global rank
        zero gathers the leaders' paths and statistics into one tracker event.
        """
        tc = self.training_config
        was_training = bool(getattr(transformer, "training", False))

        output_dir = (self.output_dir or tc.checkpoint.output_dir)

        try:
            transformer.eval()
            num_sp_groups = (self.world_group.world_size // self.sp_group.world_size)
            sp = self._get_sampling_param()

            for num_inference_steps in self.sampling_steps:
                validation_started_at = time.perf_counter()
                result = self._run_validation_for_steps(
                    num_inference_steps,
                    transformer=transformer,
                )

                if self.rank_in_sp_group != 0:
                    continue

                os.makedirs(
                    output_dir,
                    exist_ok=True,
                )
                local_videos = self._save_validation_videos(
                    result.videos,
                    output_dir=output_dir,
                    step=step,
                    num_inference_steps=num_inference_steps,
                    fps=sp.fps,
                    audio_waveforms=result.audio_waveforms,
                    audio_sample_rates=result.audio_sample_rates,
                )
                local_overlay_videos = self._save_validation_videos(
                    result.overlay_videos,
                    output_dir=output_dir,
                    step=step,
                    num_inference_steps=num_inference_steps,
                    fps=sp.fps,
                    suffix="_overlay",
                )
                local_video_filenames = local_videos.filenames
                local_overlay_video_filenames = local_overlay_videos.filenames
                local_captions = self._select_by_indices(
                    result.captions,
                    local_videos.indices,
                )
                local_overlay_captions = self._select_by_indices(
                    result.overlay_captions,
                    local_overlay_videos.indices,
                )
                local_ref_videos = self._select_by_indices(
                    result.ref_videos,
                    local_videos.indices,
                )
                local_metadata = self._select_by_indices(
                    result.metadata,
                    local_videos.indices,
                )
                local_actions = self._select_by_indices(
                    result.actions,
                    local_videos.indices,
                )
                local_mouse_pitch_signs = self._select_by_indices(
                    result.mouse_pitch_signs,
                    local_videos.indices,
                )
                local_metric_stats = self._evaluate_validation_metrics(
                    video_filenames=local_video_filenames,
                    captions=local_captions,
                    ref_videos=local_ref_videos,
                    actions=local_actions,
                    mouse_pitch_signs=local_mouse_pitch_signs,
                    fps=sp.fps,
                    output_dir=output_dir,
                    step=step,
                    num_inference_steps=num_inference_steps,
                )

                if self.global_rank == 0:
                    all_video_filenames = list(local_video_filenames)
                    all_overlay_video_filenames = list(local_overlay_video_filenames)
                    all_captions = list(local_captions)
                    all_overlay_captions = list(local_overlay_captions)
                    all_ref_videos = list(local_ref_videos)
                    all_metadata = list(local_metadata)
                    all_audio_video_count = local_videos.audio_video_count
                    all_metric_stats = local_metric_stats
                    for sp_idx in range(1, num_sp_groups):
                        src = (sp_idx * self.sp_world_size)
                        recv_v = (self.world_group.recv_object(src=src))
                        recv_c = (self.world_group.recv_object(src=src))
                        recv_ov = (self.world_group.recv_object(src=src))
                        recv_oc = (self.world_group.recv_object(src=src))
                        recv_ref = (self.world_group.recv_object(src=src))
                        recv_metadata = (self.world_group.recv_object(src=src))
                        recv_m = (self.world_group.recv_object(src=src))
                        recv_audio_video_count = (self.world_group.recv_object(src=src))
                        all_video_filenames.extend(recv_v)
                        all_overlay_video_filenames.extend(recv_ov)
                        all_captions.extend(recv_c)
                        all_overlay_captions.extend(recv_oc)
                        all_ref_videos.extend(recv_ref)
                        all_metadata.extend(recv_metadata)
                        all_audio_video_count += int(recv_audio_video_count)
                        self._merge_metric_stats(
                            all_metric_stats,
                            recv_m,
                        )

                    self._log_validation_metrics(
                        all_metric_stats,
                        step=step,
                    )
                    display_captions = [
                        self._validation_artifact_caption(caption, metadata)
                        for caption, metadata in zip(all_captions, all_metadata, strict=True)
                    ]
                    reference_filenames: list[str] = []
                    reference_captions: list[str] = []
                    for caption, ref_video, metadata in zip(
                            all_captions,
                            all_ref_videos,
                            all_metadata,
                            strict=True,
                    ):
                        if ref_video is None or not os.path.isfile(ref_video):
                            continue
                        reference_filenames.append(ref_video)
                        reference_captions.append(
                            self._validation_artifact_caption(
                                caption,
                                metadata,
                                prefix="held-out reference",
                                use_reference_num_frames=True,
                            ))
                    self._log_validation_video_artifacts(
                        all_video_filenames,
                        display_captions,
                        key=f"validation_videos_{num_inference_steps}_steps",
                        step=step,
                        fps=sp.fps,
                        reference_video_filenames=reference_filenames,
                        reference_captions=reference_captions,
                        reference_key=f"validation_references_{num_inference_steps}_steps",
                        scalar_metrics={
                            f"validation/{num_inference_steps}_steps_video_count":
                            float(len(all_video_filenames)),
                            f"validation/{num_inference_steps}_steps_duration_sec":
                            (time.perf_counter() - validation_started_at),
                            f"validation/{num_inference_steps}_steps_audio_video_count":
                            float(all_audio_video_count),
                            f"validation/{num_inference_steps}_steps_reference_video_count":
                            float(len(reference_filenames)),
                            **self._validation_metadata_scalar_metrics(
                                all_metadata,
                                num_inference_steps=num_inference_steps,
                            ),
                        },
                    )
                    if all_overlay_video_filenames:
                        self._log_validation_video_artifacts(
                            all_overlay_video_filenames,
                            all_overlay_captions,
                            key=(f"validation_videos_{num_inference_steps}"
                                 f"_steps_overlay"),
                            step=step,
                            fps=sp.fps,
                        )
                else:
                    self.world_group.send_object(
                        local_video_filenames,
                        dst=0,
                    )
                    self.world_group.send_object(
                        local_captions,
                        dst=0,
                    )
                    self.world_group.send_object(
                        local_overlay_video_filenames,
                        dst=0,
                    )
                    self.world_group.send_object(
                        local_overlay_captions,
                        dst=0,
                    )
                    self.world_group.send_object(
                        local_ref_videos,
                        dst=0,
                    )
                    self.world_group.send_object(
                        local_metadata,
                        dst=0,
                    )
                    self.world_group.send_object(
                        local_metric_stats,
                        dst=0,
                    )
                    self.world_group.send_object(
                        local_videos.audio_video_count,
                        dst=0,
                    )
        finally:
            if was_training:
                transformer.train()
            self._release_metric_evaluator()

    def _save_validation_videos(
        self,
        videos: list[list[np.ndarray]],
        *,
        output_dir: str,
        step: int,
        num_inference_steps: int,
        fps: int,
        suffix: str = "",
        audio_waveforms: list[torch.Tensor | np.ndarray | None] | None = None,
        audio_sample_rates: list[int | None] | None = None,
    ) -> _SavedValidationVideos:
        """Save aligned validation media and skip artifacts that fail encoding.

        Audio and sample-rate metadata must pass the alignment checks before
        encoding. Media writer failures are logged and omitted so validation
        artifact creation does not interrupt the training loop.
        """
        if (audio_waveforms is None) != (audio_sample_rates is None):
            raise ValueError("Validation audio waveforms and sample rates must be provided together.")
        if audio_waveforms is not None and len(audio_waveforms) != len(videos):
            raise ValueError("Validation audio waveform count must match the video count.")
        if audio_sample_rates is not None and len(audio_sample_rates) != len(videos):
            raise ValueError("Validation audio sample-rate count must match the video count.")

        saved = _SavedValidationVideos()
        for i, video in enumerate(videos):
            audio = audio_waveforms[i] if audio_waveforms is not None else None
            audio_sample_rate = audio_sample_rates[i] if audio_sample_rates is not None else None
            if (audio is None) != (audio_sample_rate is None):
                raise ValueError("Each validation waveform must have one aligned sample rate.")
            fname = os.path.join(
                output_dir,
                f"validation_step_{step}"
                f"_inference_steps_{num_inference_steps}"
                f"_rank_{self.global_rank}"
                f"_video_{i}{suffix}.mp4",
            )
            try:
                write_validation_mp4(
                    fname,
                    video,
                    fps=fps,
                    audio=audio,
                    audio_sample_rate=audio_sample_rate,
                )
            except Exception as exc:
                logger.exception(
                    "Failed to save validation media %s on rank %s; skipping artifact: %s",
                    fname,
                    self.global_rank,
                    exc,
                )
                with contextlib.suppress(OSError):
                    os.remove(fname)
                continue
            saved.filenames.append(fname)
            saved.indices.append(i)
            if audio is not None:
                saved.audio_video_count += 1
        return saved

    @staticmethod
    def _select_by_indices(
        values: list[Any],
        indices: list[int],
    ) -> list[Any]:
        return [values[i] for i in indices if i < len(values)]

    def _log_validation_video_artifacts(
        self,
        video_filenames: list[str],
        captions: list[str],
        *,
        key: str,
        step: int,
        fps: int,
        scalar_metrics: dict[str, float] | None = None,
        reference_video_filenames: list[str] | None = None,
        reference_captions: list[str] | None = None,
        reference_key: str | None = None,
    ) -> None:
        """Log validation media and its scalar verification data at one step."""
        video_logs = []
        for fname, cap in zip(
                video_filenames,
                captions,
                strict=True,
        ):
            art = self.tracker.video(
                fname,
                caption=cap,
                fps=fps,
            )
            if art is not None:
                video_logs.append(art)
        artifacts: dict[str, Any] = {}
        if video_logs:
            artifacts[key] = video_logs
        if ((reference_video_filenames is None) != (reference_captions is None)
                or (reference_video_filenames is not None) != (reference_key is not None)):
            raise ValueError("Validation reference filenames, captions, and key must be provided together.")
        if reference_video_filenames is not None:
            reference_logs = []
            for fname, cap in zip(
                    reference_video_filenames,
                    reference_captions or [],
                    strict=True,
            ):
                art = self.tracker.video(
                    fname,
                    caption=cap,
                    fps=fps,
                )
                if art is not None:
                    reference_logs.append(art)
            if reference_logs:
                assert reference_key is not None
                artifacts[reference_key] = reference_logs
        if scalar_metrics:
            artifacts.update(scalar_metrics)
        if artifacts:
            self.tracker.log_artifacts(
                artifacts,
                step,
            )

    @staticmethod
    def _validation_artifact_caption(
        caption: str,
        metadata: dict[str, Any],
        *,
        prefix: str = "generated",
        use_reference_num_frames: bool = False,
    ) -> str:
        fields = [prefix]
        source = metadata.get("source")
        sample_id = metadata.get("sample_id")
        if source:
            fields.append(f"source={source}")
        if sample_id:
            fields.append(f"id={sample_id}")
        width = metadata.get("width")
        height = metadata.get("height")
        num_frames = (metadata.get("reference_num_frames", metadata.get("num_frames"))
                      if use_reference_num_frames else metadata.get("num_frames"))
        if width and height and num_frames:
            fields.append(f"shape={width}x{height}x{num_frames}f")
        return f"[{' | '.join(fields)}] {caption}"

    @staticmethod
    def _validation_metric_segment(value: Any) -> str:
        segment = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
        return segment.strip("_") or "unknown"

    @classmethod
    def _validation_metadata_scalar_metrics(
        cls,
        metadata: list[dict[str, Any]],
        *,
        num_inference_steps: int,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        prefix = f"validation/{num_inference_steps}_steps"
        for record in metadata:
            source = cls._validation_metric_segment(record.get("source", "unknown"))
            source_key = f"{prefix}/source/{source}_count"
            metrics[source_key] = metrics.get(source_key, 0.0) + 1.0
            width = record.get("width")
            height = record.get("height")
            num_frames = record.get("num_frames")
            if width and height and num_frames:
                shape = f"{int(width)}x{int(height)}x{int(num_frames)}f"
                shape_key = f"{prefix}/shape/{shape}_count"
                metrics[shape_key] = metrics.get(shape_key, 0.0) + 1.0
        return metrics


    def _metric_device(self) -> str:
        device = self.metrics_config.device
        if device == "cuda" and torch.cuda.is_available():
            local_rank = int(getattr(self.world_group, "local_rank", 0))
            return f"cuda:{local_rank}"
        return device

    def _get_metric_evaluator(self) -> Any:
        if self._metric_evaluator is not None:
            return self._metric_evaluator
        from fastvideo.eval import Evaluator

        cfg = self.metrics_config
        self._metric_evaluator = Evaluator(
            metrics=cfg.names,
            device=self._metric_device(),
            num_gpus=1,
            loader_threads=cfg.loader_threads,
            prefetch_factor=cfg.prefetch_factor,
            pre_upload=False,
            skip_missing_deps=cfg.skip_missing_deps,
        )
        logger.info(
            "Initialized validation metrics: %s",
            ", ".join(self._metric_evaluator.metric_names),
        )
        return self._metric_evaluator

    def _release_metric_evaluator(self) -> None:
        if self._metric_evaluator is None:
            return
        if self.metrics_config.unload_after_validation:
            self._metric_evaluator.unload()
            self._metric_evaluator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _evaluate_validation_metrics(
        self,
        *,
        video_filenames: list[str],
        captions: list[str],
        ref_videos: list[str | None],
        actions: list[dict[str, Any] | None],
        mouse_pitch_signs: list[int | None],
        fps: int,
        output_dir: str,
        step: int,
        num_inference_steps: int,
    ) -> _ValidationMetricStats:
        stats = _ValidationMetricStats()
        if not self.metrics_config.enabled or not video_filenames:
            return stats

        try:
            evaluator = self._get_metric_evaluator()
            from fastvideo.eval import samples_from

            samples = samples_from(
                video=video_filenames,
                reference=self._available_paths(ref_videos),
                text_prompts=captions,
                fps=float(fps),
                extras=self._validation_metric_extras(
                    video_filenames=video_filenames,
                    actions=actions,
                    mouse_pitch_signs=mouse_pitch_signs,
                ),
            )
            results = evaluator.evaluate(samples=samples)
            for filename, metric_results in zip(
                    video_filenames,
                    results,
                    strict=True,
            ):
                row: dict[str, Any] = {"path": filename}
                self._accumulate_metric_results(
                    stats,
                    row,
                    metric_results,
                )
                stats.per_video.append(row)
            for metric_name, metric_result in results.corpus.items():
                row = {"path": "<corpus>"}
                self._accumulate_metric_results(
                    stats,
                    row,
                    {metric_name: metric_result},
                )
                stats.per_video.append(row)
        except Exception as exc:
            message = ("Validation metric evaluation failed on rank "
                       f"{self.global_rank}: {exc}")
            logger.exception(message)
            stats.errors.append(message)
            if self.metrics_config.strict:
                raise
        finally:
            if self._metric_evaluator is not None:
                self._metric_evaluator.release_cuda_memory()

        self._write_metric_summary(
            stats,
            output_dir=output_dir,
            step=step,
            num_inference_steps=num_inference_steps,
        )
        return stats

    @staticmethod
    def _accumulate_metric_results(
        stats: _ValidationMetricStats,
        row: dict[str, Any],
        metric_results: dict[str, Any],
    ) -> None:
        for metric_name, metric_result in metric_results.items():
            details = getattr(metric_result, "details", {}) or {}
            if metric_name == SYNTHETIC_OPTICAL_FLOW_METRIC:
                if not isinstance(details, dict):
                    continue
                for detail_name in SYNTHETIC_OPTICAL_FLOW_LOG_KEYS:
                    key = f"{metric_name}.{detail_name}"
                    ValidationCallback._accumulate_scalar(
                        stats,
                        row,
                        key,
                        details.get(detail_name),
                    )
                continue

            score = getattr(metric_result, "score", None)
            ValidationCallback._accumulate_scalar(
                stats,
                row,
                metric_name,
                score,
            )
            if not isinstance(details, dict):
                continue
            for detail_name, value in details.items():
                key = f"{metric_name}.{detail_name}"
                ValidationCallback._accumulate_scalar(
                    stats,
                    row,
                    key,
                    value,
                )

    @staticmethod
    def _accumulate_scalar(
        stats: _ValidationMetricStats,
        row: dict[str, Any],
        key: str,
        value: Any,
    ) -> None:
        if not isinstance(value, float | int | np.floating | np.integer):
            return
        value_float = float(value)
        if not np.isfinite(value_float):
            return
        row[key] = value_float
        stats.sums[key] = stats.sums.get(key, 0.0) + value_float
        stats.counts[key] = stats.counts.get(key, 0.0) + 1.0

    @staticmethod
    def _merge_metric_stats(
        dst: _ValidationMetricStats,
        src: _ValidationMetricStats,
    ) -> None:
        for key, value in src.sums.items():
            dst.sums[key] = dst.sums.get(key, 0.0) + float(value)
        for key, value in src.counts.items():
            dst.counts[key] = dst.counts.get(key, 0.0) + float(value)
        dst.per_video.extend(src.per_video)
        dst.errors.extend(src.errors)

    def _log_validation_metrics(
        self,
        stats: _ValidationMetricStats,
        *,
        step: int,
    ) -> None:
        if stats.errors:
            message = "Validation metric evaluation failed:\n" + "\n".join(stats.errors)
            if self.metrics_config.strict:
                raise RuntimeError(message)
            logger.warning(message)

        logs: dict[str, float] = {}
        for key, metric_sum in stats.sums.items():
            metric_count = stats.counts.get(key, 0.0)
            if metric_count <= 0:
                continue
            value = float(metric_sum / metric_count)
            if not np.isfinite(value):
                continue
            logs[self._metric_log_name(key)] = value
        if logs:
            self.tracker.log(
                logs,
                step,
            )

    def _metric_log_name(
        self,
        key: str,
    ) -> str:
        return f"{self.metrics_config.log_prefix}/{key.replace('.', '/')}"

    def _write_metric_summary(
        self,
        stats: _ValidationMetricStats,
        *,
        output_dir: str,
        step: int,
        num_inference_steps: int,
    ) -> None:
        if not self.metrics_config.enabled:
            return
        summary_dir = os.path.join(
            output_dir,
            "eval",
            f"step_{step}",
        )
        os.makedirs(
            summary_dir,
            exist_ok=True,
        )
        metrics = {
            key: float(stats.sums[key] / stats.counts[key])
            for key in sorted(stats.sums) if stats.counts.get(key, 0.0) > 0
        }
        payload = {
            "step": int(step),
            "num_inference_steps": int(num_inference_steps),
            "rank": int(self.global_rank),
            "metrics": metrics,
            "per_video": stats.per_video,
            "errors": stats.errors,
        }
        path = os.path.join(
            summary_dir,
            f"inference_steps_{num_inference_steps}_rank_{self.global_rank}.json",
        )
        with open(
                path,
                "w",
                encoding="utf-8",
        ) as f:
            json.dump(
                payload,
                f,
                indent=2,
            )

    def _validation_metric_extras(
        self,
        *,
        video_filenames: list[str],
        actions: list[dict[str, Any] | None],
        mouse_pitch_signs: list[int | None],
    ) -> list[dict[str, Any]]:
        extras: list[dict[str, Any]] = []
        for i, _filename in enumerate(video_filenames):
            extra: dict[str, Any] = {}
            action = actions[i] if i < len(actions) else None
            if action is not None:
                extra["actions"] = action
            if self.metrics_config.calibration_path is not None:
                extra["calibration"] = self.metrics_config.calibration_path
            mouse_pitch_sign = mouse_pitch_signs[i] if i < len(mouse_pitch_signs) else None
            extra["mouse_pitch_sign"] = int(mouse_pitch_sign or self.metrics_config.mouse_pitch_sign)
            extras.append(extra)
        return extras

    @staticmethod
    def _available_paths(paths: list[str | None]) -> list[str] | None:
        if not paths or any(path is None for path in paths):
            return None
        return [str(path) for path in paths]

    @staticmethod
    def _validation_actions(validation_batch: dict[str, Any]) -> dict[str, Any] | None:
        keyboard = validation_batch.get("keyboard_cond")
        mouse = validation_batch.get("mouse_cond")
        if keyboard is not None and mouse is not None:
            return {
                "keyboard": np.asarray(keyboard),
                "mouse": np.asarray(mouse),
            }

        action_path = validation_batch.get("action_path")
        if not isinstance(action_path, str) or not os.path.isfile(action_path):
            return None
        try:
            actions_obj = np.load(action_path, allow_pickle=True)
            if isinstance(actions_obj, np.ndarray) and actions_obj.dtype == object:
                actions_obj = actions_obj.item()
        except Exception as exc:
            logger.warning(
                "Failed to load validation action file %s for metrics: %s",
                action_path,
                exc,
            )
            return None
        if not isinstance(actions_obj, dict):
            return None
        keyboard = actions_obj.get("keyboard")
        mouse = actions_obj.get("mouse")
        if keyboard is None or mouse is None:
            return None
        return {
            "keyboard": np.asarray(keyboard),
            "mouse": np.asarray(mouse),
        }

    def _validation_mouse_pitch_sign(
        self,
        validation_batch: dict[str, Any],
    ) -> int:
        if validation_batch.get("mouse_pitch_sign") is not None:
            return int(validation_batch["mouse_pitch_sign"])
        flipped = validation_batch.get("mouse_pitch_flipped")
        if isinstance(flipped, str):
            flipped = flipped.lower() in {"1", "true", "yes", "y"}
        if bool(flipped):
            return -1
        return int(self.metrics_config.mouse_pitch_sign)


    def _get_sampling_param(self) -> SamplingParam:
        if self._sampling_param is None:
            self._sampling_param = (SamplingParam.from_pretrained(self._pipeline_model_path()))
        return self._sampling_param

    def _pipeline_model_path(self) -> str:
        tc = self.training_config
        pipeline_config = getattr(tc, "pipeline_config", None)
        model_path = getattr(pipeline_config, "_fastvideo_train_model_path", None)
        return str(model_path or tc.model_path)

    def _validation_pipeline_config(self, transformer: torch.nn.Module) -> Any:
        tc = self.training_config
        pipeline_config = deepcopy(tc.pipeline_config)
        self._sync_runtime_dit_arch_config(
            pipeline_config,
            transformer,
        )
        return pipeline_config

    @staticmethod
    def _keep_loaded_encoder_widths(
        validation_config: Any,
        loaded_config: Any,
    ) -> None:
        """Carry the loader-populated encoder widths onto ``validation_config``.

        Validation reaches the stages through two pipeline configs that never
        pass through ``ModelConfig.update_model_arch``: the deep copy
        ``_validation_pipeline_config`` makes of the training-side config, and
        ``tc.pipeline_config`` itself, which ``make_inference_args`` hands to
        ``pipeline.forward`` by reference. Both still hold the encoder dataclass
        defaults for whatever the checkpoint would have supplied.

        Stages read ``hidden_size`` only when they have to synthesise an
        embedding instead of measuring one: HunyuanVideo 1.5 sizes its
        zero-length ByT5 placeholder from it, so the generic ``T5ArchConfig``
        default of 512 collides with the checkpoint's real width of 1472.

        Only ``hidden_size`` is copied. Training owns the rest: model plugins
        set ``text_len`` from ``text_encoder_max_lengths`` to size the parquet
        text padding, and ``tokenizer_kwargs`` carry run-specific settings, so
        both keep their training values. A falsy width means the loader never
        populated that encoder, so there is nothing to carry over.
        """
        loaded_encoders = getattr(loaded_config, "text_encoder_configs", None)
        validation_encoders = getattr(
            validation_config,
            "text_encoder_configs",
            None,
        )
        if not loaded_encoders or not validation_encoders:
            return

        for validation_encoder, loaded_encoder in zip(
                validation_encoders,
                loaded_encoders,
                strict=False,
        ):
            hidden_size = getattr(
                getattr(loaded_encoder, "arch_config", None),
                "hidden_size",
                None,
            )
            arch_config = getattr(validation_encoder, "arch_config", None)
            if hidden_size and arch_config is not None:
                arch_config.hidden_size = hidden_size

    @staticmethod
    def _sync_runtime_dit_arch_config(
        pipeline_config: Any,
        transformer: torch.nn.Module,
    ) -> None:
        dit_config = getattr(
            pipeline_config,
            "dit_config",
            None,
        )
        arch_config = getattr(
            dit_config,
            "arch_config",
            None,
        )
        if arch_config is None:
            return

        for name in ("local_attn_size", "sink_size"):
            if not hasattr(arch_config, name) or not hasattr(transformer, name):
                continue
            setattr(
                arch_config,
                name,
                getattr(transformer, name),
            )

    def _inject_method_denoising_steps(self, validation_config: Any) -> None:
        """Sample validation at the method's trained DMD jump points.

        Distillation students only ever denoise from ``dmd_denoising_steps``;
        stages that honor ``pipeline_config.dmd_denoising_steps`` should visit
        the same points instead of the scheduler's native grid. An explicit
        value already on the pipeline config wins, and warped lists stay out
        (their raw entries are grid indices, not timesteps).
        """
        method_config = getattr(self.method, "method_config", None)
        if not isinstance(method_config, dict):
            return
        steps = method_config.get("dmd_denoising_steps")
        if not steps or bool(method_config.get("warp_denoising_step", False)):
            return
        if getattr(validation_config, "dmd_denoising_steps", "unset") is not None:
            return
        validation_config.dmd_denoising_steps = [int(step) for step in steps]

    def _get_pipeline(
        self,
        *,
        transformer: torch.nn.Module,
    ) -> Any:
        key = (id(transformer), )
        if (self._pipeline is not None and self._pipeline_key == key):
            return self._pipeline

        tc = self.training_config
        PipelineCls = resolve_target(self.pipeline_target)
        flow_shift = getattr(
            tc.pipeline_config,
            "flow_shift",
            None,
        )

        loaded_modules: dict[str, Any] = {"transformer": transformer}
        method_scheduler = getattr(self.method, "_sf_scheduler", None)
        if method_scheduler is not None:
            loaded_modules["scheduler"] = method_scheduler

        kwargs: dict[str, Any] = {
            "inference_mode": True,
            "loaded_modules": loaded_modules,
            "tp_size": tc.distributed.tp_size,
            "sp_size": tc.distributed.sp_size,
            "num_gpus": tc.distributed.num_gpus,
            "pin_cpu_memory": (tc.distributed.pin_cpu_memory),
            "dit_cpu_offload": False,
            "dit_layerwise_offload": False,
        }
        if flow_shift is not None:
            kwargs["flow_shift"] = float(flow_shift)
        kwargs.update(self.pipeline_kwargs)

        self._pipeline = PipelineCls.from_pretrained(  # type: ignore[attr-defined]
            self._pipeline_model_path(),
            **kwargs,
        )
        if tc.pipeline_config is not None:
            loaded_config = self._pipeline.fastvideo_args.pipeline_config
            validation_config = self._validation_pipeline_config(transformer)
            self._keep_loaded_encoder_widths(
                validation_config,
                loaded_config,
            )
            self._inject_method_denoising_steps(validation_config)
            self._pipeline.fastvideo_args.pipeline_config = validation_config
            arch_config = self._pipeline.fastvideo_args.pipeline_config.dit_config.arch_config
            logger.info(
                "Validation pipeline runtime config: local_attn_size=%s sink_size=%s boundary_ratio=%s",
                getattr(arch_config, "local_attn_size", None),
                getattr(arch_config, "sink_size", None),
                getattr(self._pipeline.fastvideo_args.pipeline_config.dit_config, "boundary_ratio", None),
            )

        self._pipeline_key = key
        return self._pipeline


    def _prepare_validation_batch(
        self,
        sampling_param: SamplingParam,
        validation_batch: dict[str, Any],
        num_inference_steps: int,
    ) -> ForwardBatch:
        """Build a pipeline batch with the recipe's output and conditioning policy."""
        tc = self.training_config

        sampling_param.prompt = validation_batch["prompt"]
        height, width, num_frames = self._validation_sampling_dimensions(validation_batch)
        sampling_param.height = height
        sampling_param.width = width
        sampling_param.num_inference_steps = int(num_inference_steps)
        sampling_param.data_type = "video"
        if self.guidance_scale is not None:
            sampling_param.guidance_scale = float(self.guidance_scale)
        sampling_param.seed = self.seed
        sampling_param.num_videos_per_prompt = self.num_videos_per_prompt

        sampling_param.image_path = None
        if self.use_validation_media_conditioning:
            img_path = (validation_batch.get("image_path") or validation_batch.get("video_path"))
            if img_path is not None and (img_path.startswith("http") or os.path.isfile(img_path)):
                sampling_param.image_path = img_path

        sampling_param.num_frames = num_frames

        latents_size = [
            (sampling_param.num_frames - 1) // 4 + 1,
            sampling_param.height // 8,
            sampling_param.width // 8,
        ]
        n_tokens = (latents_size[0] * latents_size[1] * latents_size[2])

        sampling_timesteps_tensor = (torch.tensor(
            [int(s) for s in self.sampling_timesteps],
            dtype=torch.long,
        ) if self.sampling_timesteps is not None else None)

        inference_args = make_inference_args(
            tc,
            model_path=tc.model_path,
        )
        self._assert_attention_contract(inference_args, tc)

        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            generator=self.validation_random_generator,
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=tc.vsa_sparsity,
            timesteps=sampling_timesteps_tensor,
        )
        batch.prompt_attention_mask = []
        batch.negative_attention_mask = []
        batch._inference_args = inference_args  # type: ignore[attr-defined]

        if ("image" in validation_batch and validation_batch["image"] is not None):
            batch.pil_image = validation_batch["image"]

        self._attach_action_conditions(
            batch,
            validation_batch,
            sampling_param.num_frames,
        )

        return batch

    def _validation_sampling_dimensions(
        self,
        validation_batch: dict[str, Any],
    ) -> tuple[int, int, int]:
        """Resolve output geometry, optionally from one validation record.

        Native-shape validation is explicit because cached ``SamplingParam``
        instances are shared across records. A complete record triplet wins;
        partial metadata fails instead of combining dimensions from unrelated
        shapes. ``max_record_num_frames`` optionally caps only the temporal
        member of a complete record triplet; fixed/default geometry is never
        changed. With the option off (the default), legacy callback/config
        behavior is unchanged.
        """
        tc = self.training_config
        temporal_compression_factor = int(
            tc.pipeline_config.vae_config.arch_config.temporal_compression_ratio  # type: ignore[union-attr]
        )
        default_num_frames = ((tc.data.num_latent_t - 1) * temporal_compression_factor + 1)
        dimensions = {
            "height": int(tc.data.num_height),
            "width": int(tc.data.num_width),
            "num_frames": (int(self.num_frames) if self.num_frames is not None else int(default_num_frames)),
        }

        if self.use_record_dimensions:
            present = {name: validation_batch.get(name) is not None for name in dimensions}
            if any(present.values()) and not all(present.values()):
                missing = [name for name, is_present in present.items() if not is_present]
                raise ValueError("Native-shape validation records must provide width, height, and num_frames together; "
                                 f"missing {missing} for prompt {validation_batch.get('prompt')!r}.")
            if all(present.values()):
                dimensions = {name: int(validation_batch[name]) for name in dimensions}
                if self.max_record_num_frames is not None:
                    dimensions["num_frames"] = min(dimensions["num_frames"], self.max_record_num_frames)

        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"Validation {name} must be positive, got {value}")
        if dimensions["height"] % 8 or dimensions["width"] % 8:
            raise ValueError("Validation width and height must be divisible by 8, got "
                             f"{dimensions['width']}x{dimensions['height']}")
        return dimensions["height"], dimensions["width"], dimensions["num_frames"]

    def _attach_action_conditions(
        self,
        batch: ForwardBatch,
        validation_batch: dict[str, Any],
        num_frames: int,
    ) -> None:
        for name in ("keyboard_cond", "mouse_cond"):
            value = validation_batch.get(name)
            if value is None:
                continue
            array = np.asarray(value)
            if array.size == 0:
                continue
            array = array[:num_frames]
            tensor = torch.as_tensor(
                array,
                dtype=torch.bfloat16,
            )
            student = getattr(self.method, "student", None)
            prepare_action = getattr(student, "prepare_validation_action_condition", None)
            if prepare_action is not None:
                tensor = prepare_action(
                    tensor,
                    name=name,
                    keyboard_value_scale=self.keyboard_value_scale,
                )
            tensor = tensor.unsqueeze(0)
            setattr(
                batch,
                name,
                tensor,
            )


    def _run_validation_for_steps(
        self,
        num_inference_steps: int,
        *,
        transformer: torch.nn.Module,
    ) -> _ValidationStepResult:
        """Generate the prompts assigned to one sequence-parallel group.

        ``ValidationDataset`` pads the prompt count to the data-parallel degree
        and assigns an equal prompt shard to each sequence-parallel group. All
        ranks in a group execute each forward pass; the group leader retains
        decoded media.
        """
        tc = self.training_config
        pipeline = self._get_pipeline(transformer=transformer, )
        sampling_param = self._get_sampling_param()

        dataset = ValidationDataset(self.dataset_file)
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
        )

        inference_args = make_inference_args(
            tc,
            model_path=tc.model_path,
        )
        self._assert_attention_contract(inference_args, tc)
        self._sync_runtime_dit_arch_config(
            inference_args.pipeline_config,
            transformer,
        )
        self._keep_loaded_encoder_widths(
            inference_args.pipeline_config,
            pipeline.fastvideo_args.pipeline_config,
        )

        if (self.sampling_timesteps is not None and inference_args.pipeline_config.dmd_denoising_steps is None):
            inference_args.pipeline_config.dmd_denoising_steps = ([int(s) for s in self.sampling_timesteps])

        videos: list[list[np.ndarray]] = []
        audio_waveforms: list[torch.Tensor | np.ndarray | None] = []
        audio_sample_rates: list[int | None] = []
        overlay_videos: list[list[np.ndarray]] = []
        captions: list[str] = []
        overlay_captions: list[str] = []
        ref_videos: list[str | None] = []
        metadata: list[dict[str, Any]] = []
        actions: list[dict[str, Any] | None] = []
        mouse_pitch_signs: list[int | None] = []

        for validation_batch in dataloader:
            batch = self._prepare_validation_batch(
                sampling_param,
                validation_batch,
                num_inference_steps,
            )

            assert (batch.prompt is not None and isinstance(batch.prompt, str))
            ref_video = (validation_batch.get("ref_video") or validation_batch.get("video_path"))
            action = self._validation_actions(validation_batch)

            with torch.no_grad():
                output_batch = pipeline.forward(
                    batch,
                    inference_args,
                )

            samples = output_batch.output.cpu()
            if self.rank_in_sp_group != 0:
                continue

            output_audio = output_batch.extra.get("audio")
            output_audio_sample_rate = output_batch.extra.get("audio_sample_rate")
            if (output_audio is None) != (output_audio_sample_rate is None):
                raise ValueError("Validation pipeline outputs must provide audio and its sample rate together.")
            if output_audio is not None and not isinstance(output_audio,
                                                           np.ndarray) and not torch.is_tensor(output_audio):
                raise TypeError("Validation pipeline audio must be a torch.Tensor or numpy.ndarray; "
                                f"got {type(output_audio).__name__}.")
            if torch.is_tensor(output_audio):
                output_audio = output_audio.detach().cpu()

            video = rearrange(
                samples,
                "b c t h w -> t b c h w",
            )
            frames: list[np.ndarray] = []
            for x in video:
                x = torchvision.utils.make_grid(
                    x,
                    nrow=6,
                )
                x = (x.transpose(0, 1).transpose(1, 2).squeeze(-1))
                frames.append((x * 255).numpy().astype(np.uint8))
            videos.append(frames)
            captions.append(batch.prompt)
            audio_waveforms.append(output_audio)
            audio_sample_rates.append(int(output_audio_sample_rate) if output_audio_sample_rate is not None else None)
            ref_videos.append(ref_video if isinstance(ref_video, str) else None)
            record_metadata: dict[str, Any] = {
                "source": validation_batch.get("source", "unknown"),
                "sample_id": validation_batch.get("sample_id", validation_batch.get("id")),
                "width": int(batch.width),
                "height": int(batch.height),
                "num_frames": int(batch.num_frames),
            }
            reference_num_frames = validation_batch.get("num_frames")
            if reference_num_frames is not None and int(reference_num_frames) != int(batch.num_frames):
                record_metadata["reference_num_frames"] = int(reference_num_frames)
            metadata.append(record_metadata)
            actions.append(action)
            mouse_pitch_signs.append(self._validation_mouse_pitch_sign(validation_batch))
            if self.overlay_actions:
                overlay_frames = self._post_process_validation_frames(
                    frames,
                    action=action,
                )
                if overlay_frames is not None:
                    overlay_videos.append(overlay_frames)
                    overlay_captions.append(batch.prompt)

        return _ValidationStepResult(
            videos=videos,
            captions=captions,
            audio_waveforms=audio_waveforms,
            audio_sample_rates=audio_sample_rates,
            overlay_videos=overlay_videos,
            overlay_captions=overlay_captions,
            ref_videos=ref_videos,
            metadata=metadata,
            actions=actions,
            mouse_pitch_signs=mouse_pitch_signs,
        )

    def _post_process_validation_frames(
        self,
        frames: list[np.ndarray],
        *,
        action: dict[str, Any] | None,
    ) -> list[np.ndarray] | None:
        student = getattr(self.method, "student", None)
        post_process = getattr(student, "post_process_validation_frames", None)
        if post_process is None:
            return None
        return post_process(
            frames,
            action=action,
        )


    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if self.validation_random_generator is not None:
            state["validation_rng"] = (self.validation_random_generator.get_state())
        return state

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        rng_state = state_dict.get("validation_rng")
        if (rng_state is not None and self.validation_random_generator is not None):
            self.validation_random_generator.set_state(rng_state)
