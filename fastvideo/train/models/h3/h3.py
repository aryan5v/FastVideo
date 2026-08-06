# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 per-role training model (dual-scheduler DMD).

Wraps ``MiniMaxH3Transformer3DModel`` (upstream #1674) for the DMD2-style
distillation stack. The H3-specific parts:

- **Joint AV latents**: video ``(B, 24, T', H', W')`` + audio
  ``(B, 2, 32, n)`` travel together; the model sees one packed sequence
  ``[text | audio | video]`` built with the upstream packing geometry.
- **Dual scheduler**: video rows denoise on the shift-12 sigma grid, audio
  rows on the shift-3 grid. The method passes a single *video* timestep
  (a step identifier); :meth:`_audio_timestep_for` maps it to the audio
  grid via nearest step index. This is the "dual-scheduler DMD" novelty.
- **Data-ward velocity**: ``x0 = x_t + (1 - t) * v`` (H3 convention, sign
  reversed vs Wan). ``predict_noise_h3`` returns the raw velocity; the
  method's flow-matching targets are ``(x0 - noise)`` per modality.
- **Guidance-distilled**: no CFG, no negative prompt; ``cfg_uncond.text``
  must be ``"keep"`` so the uncond branch reuses the conditional embeds.
- **Pre-normalized corpus**: the preprocessor stores latents already
  normalized by the VAE stats, so no VAE is needed in the training loop.

The dataloader is a lightweight safetensors reader over the corpus manifest
(``scripts/fasth3/preprocess_h3_corpus.py`` output); batch size is 1 per GPU
because the packed sequence length varies per clip (no padding contract).
"""

from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from fastvideo.logger import init_logger
from fastvideo.mlx_runtime.minimax_h3 import MINIMAX_H3_AUDIO_SHIFT, MINIMAX_H3_VIDEO_SHIFT, minimax_h3_sigmas
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.training_config import TrainingConfig

logger = init_logger(__name__)


def _sample_timestep(device: torch.device, generator: torch.Generator) -> torch.Tensor:
    """Uniform continuous timestep in [0, 1] (H3's clean-time convention)."""
    return torch.rand((1, ), device=device, dtype=torch.float32, generator=generator)


class H3CorpusDataset(torch.utils.data.Dataset):
    """Reads preprocessed H3 corpus artifacts (one clip per sample)."""

    def __init__(self, manifest_dir: str, *, max_clips: int = 0, seed: int = 1000) -> None:
        import safetensors.torch

        self._st = safetensors.torch
        manifest_dir = str(manifest_dir)
        entries: list[dict[str, str]] = []
        import pathlib

        for path in sorted(pathlib.Path(manifest_dir).glob("manifest_rank*.jsonl")):
            for line in path.read_text().splitlines():
                entries.append(json.loads(line))
        if max_clips:
            entries = entries[:max_clips]
        rng = np.random.default_rng(seed)
        rng.shuffle(entries)
        self.entries = entries
        self.latents_dir = str(pathlib.Path(manifest_dir) / "latents")
        self.text_dir = str(pathlib.Path(manifest_dir) / "text")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import pathlib

        entry = self.entries[index]
        latents = self._st.load_file(str(pathlib.Path(self.latents_dir) / f"{entry['id']}.safetensors"))
        text = self._st.load_file(str(pathlib.Path(self.text_dir) / f"{entry['text_sha1']}.safetensors"))
        video = latents["video"].unsqueeze(0)  # (1, 24, T', H', W')
        audio_rows = latents["audio"]  # (2n, 32)
        audio = audio_rows.reshape(2, -1, audio_rows.shape[-1]).transpose(1, 2).unsqueeze(0)  # (1, 2, 32, n)
        return {
            "vae_latent": video,
            "audio_latent": audio,
            "text_embedding": text["embed"].unsqueeze(0),  # (1, L, 5120)
        }


def _build_h3_dataloader(data_config: Any) -> torch.utils.data.DataLoader:
    dataset = H3CorpusDataset(
        str(data_config.data_path),
        max_clips=int(getattr(data_config, "max_clips", 0) or 0),
        seed=int(data_config.seed or 1000),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(data_config.train_batch_size),
        shuffle=False,
        num_workers=int(data_config.dataloader_num_workers or 0),
        drop_last=True,
    )


class H3Model(ModelBase):
    """MiniMax H3 per-role model (student / teacher / critic)."""

    _transformer_cls_name: str = "MiniMaxH3Transformer3DModel"

    def __init__(
        self,
        *,
        init_from: str,
        training_config: TrainingConfig,
        trainable: bool = True,
        disable_custom_init_weights: bool = False,
        video_shift: float = MINIMAX_H3_VIDEO_SHIFT,
        audio_shift: float = MINIMAX_H3_AUDIO_SHIFT,
        enable_gradient_checkpointing_type: str | None = None,
        transformer_override_safetensor: str | None = None,
        lora: Any = None,
        attention_backend: Any = None,
    ) -> None:
        super().__init__(
            trainable=trainable,
            lora=lora,
            attention_backend=attention_backend,
        )
        self._init_from = str(init_from)
        self.transformer = self._load_transformer(
            init_from=self._init_from,
            trainable=self._trainable,
            disable_custom_init_weights=bool(disable_custom_init_weights),
            enable_gradient_checkpointing_type=enable_gradient_checkpointing_type,
            training_config=training_config,
            transformer_override_safetensor=transformer_override_safetensor,
            attention_backend=self.attention_backend,
        )
        self.video_shift = float(video_shift)
        self.audio_shift = float(audio_shift)
        self._video_grid: np.ndarray | None = None
        self._audio_grid: np.ndarray | None = None

        self.vae: Any = None  # not needed: corpus is pre-normalized
        self.training_config = training_config
        self.dataloader: Any = None
        self.validator: Any = None
        self.start_step: int = 0
        self.world_group: Any = None
        self.sp_group: Any = None
        self._requires_negative_conditioning = False  # guidance-distilled

        self.patch_size = tuple(int(value) for value in self.transformer.patch_size)
        self.audio_in_channels = int(self.transformer.audio_in_channels)

    # ------------------------------------------------------------------
    # timestep mechanics (continuous t in [0, 1])
    # ------------------------------------------------------------------

    @property
    def num_train_timesteps(self) -> int:
        return 1000  # compatibility shim; H3 timesteps are continuous floats

    def shift_and_clamp_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        return timestep  # already in [0, 1]

    def set_requires_negative_conditioning(self, requires: bool) -> None:
        self._requires_negative_conditioning = bool(requires)

    def set_dmd_step_grids(self, video_steps: list[float]) -> None:
        """Store the DMD step grid; derive the audio grid (shift 3) by index."""
        self._video_grid = np.asarray(video_steps, dtype=np.float32)
        sigmas = minimax_h3_sigmas(self.audio_shift, len(self._video_grid))
        self._audio_grid = (1.0 - sigmas[:-1]).astype(np.float32)

    def _audio_timestep_for(self, video_timestep: float) -> float:
        if self._video_grid is None or self._audio_grid is None:
            raise RuntimeError("set_dmd_step_grids() must run before training.")
        index = int(np.argmin(np.abs(self._video_grid - float(video_timestep))))
        return float(self._audio_grid[index])

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def _load_transformer(
        self,
        *,
        init_from: str,
        trainable: bool,
        disable_custom_init_weights: bool,
        enable_gradient_checkpointing_type: str | None,
        training_config: TrainingConfig,
        transformer_override_safetensor: str | None = None,
        attention_backend: Any = None,
    ) -> torch.nn.Module:
        from fastvideo.train.utils.module_utils import (  # noqa: PLC0415
            apply_activation_checkpointing,
            apply_trainable,
            load_module_from_path,
        )

        transformer = load_module_from_path(
            model_path=init_from,
            module_type="transformer",
            training_config=training_config,
            disable_custom_init_weights=disable_custom_init_weights,
            override_transformer_cls_name=self._transformer_cls_name,
            transformer_override_safetensor=transformer_override_safetensor,
            attention_backend=attention_backend,
        )
        checkpoint_type = enable_gradient_checkpointing_type or getattr(
            getattr(training_config, "model", None), "enable_gradient_checkpointing_type", None)
        if trainable and checkpoint_type:
            transformer = apply_activation_checkpointing(transformer, checkpointing_type=checkpoint_type)
        return apply_trainable(transformer, trainable=trainable)

    def init_preprocessors(self, training_config: TrainingConfig) -> None:
        self.dataloader = _build_h3_dataloader(training_config.data)

    def on_train_start(self) -> None:
        return None

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("H3 decode is not part of the training loop (pre-normalized corpus).")

    def clear_caches(self, *, cache_tag: str = "pos") -> None:
        return None

    def predict_noise_streaming(self, *args: Any, **kwargs: Any) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError("streaming not applicable to H3")

    # ------------------------------------------------------------------
    # batch preparation (packed layout + per-modality noise)
    # ------------------------------------------------------------------

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> Any:
        from fastvideo.pipelines.basic.minimax_h3.packing import (  # noqa: PLC0415
            MINIMAX_H3_AUDIO_CHANNELS,
            build_packed_sequence,
            build_row_timesteps,
            patchify_video_latents,
        )

        dtype = torch.bfloat16
        device = self.device
        batch_size = 1  # packed sequence length varies per clip; B=1 per GPU

        if latents_source == "zeros":
            latents = torch.zeros(batch_size, 24, 37, 16, 16, device=device, dtype=dtype)
            audio_latents = torch.zeros(batch_size, MINIMAX_H3_AUDIO_CHANNELS, 32, 200, device=device, dtype=dtype)
        else:
            latents = raw_batch["vae_latent"][:batch_size].to(device, dtype=dtype)
            audio_latents = raw_batch["audio_latent"][:batch_size].to(device, dtype=dtype)
        text_embeds = raw_batch["text_embedding"][:batch_size].to(device, dtype=dtype)

        video_t = _sample_timestep(device, generator)
        audio_t = _sample_timestep(device, generator)

        _, channels, num_latent_frames, latent_height, latent_width = latents.shape
        num_audio_latents = audio_latents.shape[-1]
        text_len = text_embeds.shape[1]
        text_token_tags = torch.full((text_len, ), 1, dtype=torch.long)  # TEXT tag
        layout = build_packed_sequence(
            text_token_tags,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            self.patch_size,
        )
        unique, inverse = build_row_timesteps(layout, float(video_t.item()), float(audio_t.item()))

        noise = torch.randn(latents.shape, device=device, dtype=dtype, generator=generator)
        audio_noise = torch.randn(audio_latents.shape, device=device, dtype=dtype, generator=generator)

        from fastvideo.train.utils.training_config import TrainingBatch  # noqa: PLC0415

        batch = TrainingBatch()
        batch.latents = latents
        batch.audio_latents = audio_latents
        batch.encoder_hidden_states = text_embeds
        batch.timesteps = video_t
        batch.attn_metadata = None
        batch.attn_metadata_vsa = None
        batch.raw_latent_shape = tuple(latents.shape)
        batch.dmd_latent_vis_dict = {}
        batch.fake_score_latent_vis_dict = {}

        # H3-specific fields (setattr; TrainingBatch is a mutable dataclass).
        h3 = _H3BatchFields(
            audio_latents=audio_latents,
            audio_noise=audio_noise,
            noise=noise,
            video_t=video_t,
            audio_t=audio_t,
            layout=layout,
            unique_timesteps=unique,
            row_timestep_inverse=inverse,
            text_embeds=text_embeds,
        )
        batch.h3 = h3
        return batch

    # ------------------------------------------------------------------
    # noise / prediction (data-ward velocity)
    # ------------------------------------------------------------------

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.to(latents.dtype)
        return t * latents + (1.0 - t) * noise

    def add_noise_audio(self, latents: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Audio noise at the DMD-grid timestep for this (video) step (rollout path)."""
        t = torch.tensor(
            self._audio_timestep_for(float(timestep.item())),
            device=latents.device,
            dtype=latents.dtype,
        )
        return t * latents + (1.0 - t) * noise

    def add_noise_audio_continuous(self, latents: torch.Tensor, noise: torch.Tensor, audio_t: torch.Tensor
                                  ) -> torch.Tensor:
        """Audio noise at an arbitrary continuous t (score-loss paths)."""
        t = audio_t.to(latents.dtype)
        return t * latents + (1.0 - t) * noise

    def _packed_forward(self, noisy_video: torch.Tensor, noisy_audio: torch.Tensor, video_t: float, audio_t: float,
                        batch: Any):
        """Run the H3 transformer on packed rows; return (video_rows, audio_rows) velocities."""
        from fastvideo.pipelines.basic.minimax_h3.packing import (  # noqa: PLC0415
            build_row_timesteps,
            patchify_video_latents,
        )

        layout = batch.h3.layout
        video_rows = patchify_video_latents(noisy_video.float(), self.patch_size)
        audio_rows = noisy_audio.reshape(1, -1, self.audio_in_channels)
        unique, inverse = build_row_timesteps(layout, video_t, audio_t)

        text_embeds = batch.h3.text_embeds
        video_out, audio_out = self.transformer(
            hidden_states=video_rows,
            audio_hidden_states=audio_rows,
            encoder_hidden_states=text_embeds,
            timestep=torch.as_tensor(unique, dtype=torch.float32, device=noisy_video.device),
            timestep_indices=torch.as_tensor(inverse, dtype=torch.long, device=noisy_video.device),
            token_tags=torch.as_tensor(layout.token_tags, dtype=torch.long, device=noisy_video.device),
            position_ids=torch.as_tensor(layout.position_ids, dtype=torch.float32, device=noisy_video.device),
            video_indices=torch.as_tensor(layout.video_indices, dtype=torch.long, device=noisy_video.device),
            audio_indices=torch.as_tensor(layout.audio_indices, dtype=torch.long, device=noisy_video.device),
            text_indices=torch.as_tensor(layout.text_indices, dtype=torch.long, device=noisy_video.device),
        )
        return video_out, audio_out

    def predict_velocity_h3(
        self,
        noisy_video: torch.Tensor,
        noisy_audio: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
        audio_timestep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Model velocities for both modalities at a (video) timestep.

        ``audio_timestep``: continuous audio t for the score-loss paths; when
        None, the audio t is derived from the DMD grid (rollout path).
        """
        del conditional, cfg_uncond, attn_kind  # guidance-distilled: single pass
        video_t = float(timestep.item())
        audio_t = float(audio_timestep.item()) if audio_timestep is not None else self._audio_timestep_for(video_t)
        video_rows_out, audio_rows_out = self._packed_forward(noisy_video, noisy_audio, video_t, audio_t, batch)

        # rows -> latents (channel-major patch contract)
        from fastvideo.pipelines.basic.minimax_h3.packing import unpatchify_video_tokens, unpack_audio_tokens  # noqa: PLC0415

        _, channels, num_frames, height, width = noisy_video.shape
        video_vel = unpatchify_video_tokens(
            video_rows_out.float(),
            num_frames,
            height,
            width,
            channels,
            self.patch_size,
        ).to(noisy_video.device, noisy_video.dtype)
        n_audio = noisy_audio.shape[-1]
        audio_vel = unpack_audio_tokens(audio_rows_out[0].float(), n_audio).unsqueeze(0).to(
            noisy_audio.device, noisy_audio.dtype)
        return video_vel, audio_vel

    def predict_x0_h3(
        self,
        noisy_video: torch.Tensor,
        noisy_audio: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
        audio_timestep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Data-ward x0 = x_t + (1 - t) * v per modality."""
        video_vel, audio_vel = self.predict_velocity_h3(
            noisy_video, noisy_audio, timestep, batch, conditional=conditional, cfg_uncond=cfg_uncond,
            attn_kind=attn_kind, audio_timestep=audio_timestep)
        video_t = float(timestep.item())
        audio_t = float(audio_timestep.item()) if audio_timestep is not None else self._audio_timestep_for(video_t)
        video_x0 = noisy_video + (1.0 - video_t) * video_vel
        audio_x0 = noisy_audio + (1.0 - audio_t) * audio_vel
        return video_x0, audio_x0

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        raise NotImplementedError("use predict_velocity_h3 / predict_x0_h3 for H3 (dual-modality)")

    def predict_x0(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        raise NotImplementedError("use predict_x0_h3 for H3 (dual-modality)")

    def backward(
        self,
        loss: torch.Tensor,
        ctx: Any,
        *,
        grad_accum_rounds: int,
    ) -> None:
        (loss / max(1, int(grad_accum_rounds))).backward()

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device


class _H3BatchFields:
    """H3-specific batch state attached to TrainingBatch as ``batch.h3``."""

    def __init__(
        self,
        *,
        audio_latents: torch.Tensor,
        audio_noise: torch.Tensor,
        noise: torch.Tensor,
        video_t: torch.Tensor,
        audio_t: torch.Tensor,
        layout: Any,
        unique_timesteps: Any,
        row_timestep_inverse: Any,
        text_embeds: torch.Tensor,
    ) -> None:
        self.audio_latents = audio_latents
        self.audio_noise = audio_noise
        self.noise = noise
        self.video_t = video_t
        self.audio_t = audio_t
        self.layout = layout
        self.unique_timesteps = unique_timesteps
        self.row_timestep_inverse = row_timestep_inverse
        self.text_embeds = text_embeds
