# SPDX-License-Identifier: Apache-2.0
"""Dual-scheduler DMD2 distillation for MiniMax H3 (video + audio).

The one novel piece of the FastH3 training stack: H3 denoises video rows on
the shift-12 sigma grid and audio rows on the shift-3 grid inside a single
packed sequence, so every DMD step carries **two** continuous timesteps.

Design:
- The method's ``dmd_denoising_steps`` are the **video** timesteps (ascending
  t in [0, 1], ending at 1.0 = clean, from the shift-12 grid). The audio grid
  is derived per step index (shift 3) via :meth:`H3Model._audio_timestep_for`.
- Rollout, critic flow-matching, and the DMD generator loss run on
  ``(video, audio)`` pairs; every ``add_noise`` / ``predict_*`` call uses the
  per-modality timestep.
- Velocity convention is H3's data-ward form: model outputs ``v`` with
  ``x0 = x_t + (1 - t) * v``; flow-matching targets are ``(x0 - noise)``.
- Guidance-distilled teacher: ``real_score_guidance_scale`` stays 1.0 and the
  config must set ``cfg_uncond.text: keep`` so the "uncond" branch reuses the
  conditional embeddings (no negative prompt exists for H3).
"""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method


class H3DMD2Method(DMD2Method):
    """DMD2 adapted to H3's joint AV packed sequence with dual timesteps."""

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, Any],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        raw = self.method_config.get("dmd_denoising_steps", None)
        if not isinstance(raw, list) or not raw:
            raise ValueError("method_config.dmd_denoising_steps must be set for H3 DMD2")
        # Keep the raw float grid: the base class casts to int (Wan's 1000-space).
        self._video_steps: list[float] = [float(value) for value in raw]
        for role in (self.student, self.teacher, self.critic):
            setter = getattr(role, "set_dmd_step_grids", None)
            if setter is None:
                raise RuntimeError(f"role model {type(role).__name__} is not an H3Model")
            setter(self._video_steps)

    # ------------------------------------------------------------------
    # timestep sampling
    # ------------------------------------------------------------------

    def _sample_score_timestep(self, device: torch.device) -> torch.Tensor:
        """Continuous uniform timestep in [0, 1] (H3 clean-time convention)."""
        return torch.rand((1, ), device=device, dtype=torch.float32, generator=self.cuda_generator)

    def _sample_audio_score_timestep(self, device: torch.device) -> torch.Tensor:
        """Independent continuous audio timestep (critic sees all noise levels)."""
        return torch.rand((1, ), device=device, dtype=torch.float32, generator=self.cuda_generator)

    # ------------------------------------------------------------------
    # rollout
    # ------------------------------------------------------------------

    def _student_rollout(
        self,
        batch: Any,
        *,
        with_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulate-mode rollout over the DMD step grid, both modalities."""
        latents = batch.latents
        audio_latents = batch.h3.audio_latents
        device = latents.device
        dtype = latents.dtype
        step_list = self._video_steps
        target_timestep_idx = torch.randint(
            0,
            len(step_list),
            [1],
            device=device,
            dtype=torch.long,
            generator=self.cuda_generator,
        )
        target_timestep_idx_int = int(target_timestep_idx.item())
        target_timestep = step_list[target_timestep_idx_int]

        current_noise = torch.randn(latents.shape, device=device, dtype=dtype, generator=self.cuda_generator)
        current_audio_noise = torch.randn(
            audio_latents.shape, device=device, dtype=dtype, generator=self.cuda_generator)
        current_noise_copy = current_noise.clone()
        current_audio_noise_copy = current_audio_noise.clone()

        # IMPORTANT: run the FULL loop on every rank. The method's cuda_generator
        # is seeded per-rank, so target_timestep_idx_int DIFFERS across ranks;
        # the loop length must therefore be rank-invariant or the collective
        # counts diverge and FSDP allgathers deadlock. (The base DMD2 does the
        # same for this reason — the "extra" steps are not waste.)
        max_target_idx = len(step_list) - 1
        video_noise_latents: list[torch.Tensor] = []
        audio_noise_latents: list[torch.Tensor] = []
        noise_latent_index = target_timestep_idx_int - 1

        if max_target_idx > 0:
            with torch.no_grad():
                for step_idx in range(max_target_idx):
                    current_timestep = step_list[step_idx]
                    current_timestep_tensor = torch.tensor([float(current_timestep)], device=device, dtype=torch.float32)

                    pred_clean_video, pred_clean_audio = self.student.predict_x0_h3(
                        current_noise,
                        current_audio_noise,
                        current_timestep_tensor,
                        batch,
                        conditional=True,
                        cfg_uncond=self._cfg_uncond,
                        attn_kind="vsa",
                    )
                    next_timestep = step_list[step_idx + 1]
                    next_timestep_tensor = torch.tensor([float(next_timestep)], device=device, dtype=torch.float32)
                    noise = torch.randn(latents.shape, device=device, dtype=pred_clean_video.dtype,
                                        generator=self.cuda_generator)
                    audio_noise = torch.randn(audio_latents.shape, device=device, dtype=pred_clean_audio.dtype,
                                              generator=self.cuda_generator)
                    current_noise = self.student.add_noise(pred_clean_video, noise, next_timestep_tensor)
                    current_audio_noise = self.student.add_noise_audio(pred_clean_audio, audio_noise,
                                                                       next_timestep_tensor)
                    video_noise_latents.append(current_noise.clone())
                    audio_noise_latents.append(current_audio_noise.clone())

        if noise_latent_index >= 0:
            if noise_latent_index >= len(video_noise_latents):
                raise RuntimeError("noise_latent_index is out of bounds")
            noisy_input = video_noise_latents[noise_latent_index]
            noisy_audio_input = audio_noise_latents[noise_latent_index]
        else:
            noisy_input = current_noise_copy
            noisy_audio_input = current_audio_noise_copy

        target_timestep_tensor = torch.tensor([float(target_timestep)], device=device, dtype=torch.float32)
        if with_grad:
            pred_x0 = self.student.predict_x0_h3(
                noisy_input, noisy_audio_input, target_timestep_tensor, batch,
                conditional=True, cfg_uncond=self._cfg_uncond, attn_kind="vsa")
        else:
            with torch.no_grad():
                pred_x0 = self.student.predict_x0_h3(
                    noisy_input, noisy_audio_input, target_timestep_tensor, batch,
                    conditional=True, cfg_uncond=self._cfg_uncond, attn_kind="vsa")

        batch.dmd_latent_vis_dict["generator_timestep"] = torch.tensor(
            target_timestep, device=device, dtype=torch.float32)
        return pred_x0

    # ------------------------------------------------------------------
    # critic flow-matching loss (velocity target = x0 - noise)
    # ------------------------------------------------------------------

    def _critic_flow_matching_loss(
        self,
        batch: Any,
    ) -> tuple[torch.Tensor, Any, dict[str, Any]]:
        with torch.no_grad():
            generator_pred_x0 = self._student_rollout(batch, with_grad=False)
        generator_video_x0, generator_audio_x0 = generator_pred_x0

        device = generator_video_x0.device
        fake_score_timestep = self._sample_score_timestep(device)
        fake_score_audio_timestep = self._sample_audio_score_timestep(device)

        noise = torch.randn(generator_video_x0.shape, device=device, dtype=generator_video_x0.dtype,
                            generator=self.cuda_generator)
        audio_noise = torch.randn(generator_audio_x0.shape, device=device, dtype=generator_audio_x0.dtype,
                                  generator=self.cuda_generator)
        noisy_x0 = self.student.add_noise(generator_video_x0, noise, fake_score_timestep)
        noisy_audio_x0 = self.student.add_noise_audio_continuous(generator_audio_x0, audio_noise,
                                                                 fake_score_audio_timestep)

        pred_video_vel, pred_audio_vel = self.critic.predict_velocity_h3(
            noisy_x0, noisy_audio_x0, fake_score_timestep, batch,
            conditional=True, cfg_uncond=self._cfg_uncond, attn_kind="dense",
            audio_timestep=fake_score_audio_timestep)

        # velocity target: x0 - noise  (H3 predicts v = x0 - eps)
        target_video = generator_video_x0 - noise
        target_audio = generator_audio_x0 - audio_noise
        flow_matching_loss = torch.mean((pred_video_vel - target_video)**2) + torch.mean(
            (pred_audio_vel - target_audio)**2)

        batch.fake_score_latent_vis_dict = {
            "generator_pred_video": generator_video_x0,
            "generator_pred_audio": generator_audio_x0,
            "fake_score_timestep": fake_score_timestep,
        }
        outputs = {"fake_score_latent_vis_dict": batch.fake_score_latent_vis_dict}
        return flow_matching_loss, (batch.timesteps, batch.attn_metadata), outputs

    # ------------------------------------------------------------------
    # generator (DMD) loss, both modalities
    # ------------------------------------------------------------------

    def _dmd_loss(
        self,
        generator_pred_x0: tuple[torch.Tensor, torch.Tensor],
        batch: Any,
    ) -> torch.Tensor:
        # H3 is guidance-distilled (cfg_uncond.text=keep); teacher cond == uncond.
        generator_video_x0, generator_audio_x0 = generator_pred_x0
        device = generator_video_x0.device

        with torch.no_grad():
            timestep = self._sample_score_timestep(device)
            audio_timestep = self._sample_audio_score_timestep(device)

            noise = torch.randn(generator_video_x0.shape, device=device, dtype=generator_video_x0.dtype,
                                generator=self.cuda_generator)
            audio_noise = torch.randn(generator_audio_x0.shape, device=device, dtype=generator_audio_x0.dtype,
                                      generator=self.cuda_generator)
            noisy_latents = self.student.add_noise(generator_video_x0, noise, timestep)
            noisy_audio = self.student.add_noise_audio_continuous(generator_audio_x0, audio_noise, audio_timestep)

            faker_video, faker_audio = self.critic.predict_x0_h3(
                noisy_latents, noisy_audio, timestep, batch,
                conditional=True, cfg_uncond=self._cfg_uncond, attn_kind="dense",
                audio_timestep=audio_timestep)
            # Guidance-distilled: conditional == unconditional, so a single
            # teacher pass suffices (real_cfg = real_cond, scale 1.0).
            real_cfg_video, real_cfg_audio = self.teacher.predict_x0_h3(
                noisy_latents, noisy_audio, timestep, batch,
                conditional=True, cfg_uncond=self._cfg_uncond, attn_kind="dense",
                audio_timestep=audio_timestep)

            denom_video = torch.abs(generator_video_x0 - real_cfg_video).mean()
            grad_video = torch.nan_to_num((faker_video - real_cfg_video) / denom_video)
            denom_audio = torch.abs(generator_audio_x0 - real_cfg_audio).mean()
            grad_audio = torch.nan_to_num((faker_audio - real_cfg_audio) / denom_audio)

        loss = 0.5 * torch.nn.functional.mse_loss(
            generator_video_x0.float(),
            (generator_video_x0.float() - grad_video.float()).detach(),
        ) + 0.5 * torch.nn.functional.mse_loss(
            generator_audio_x0.float(),
            (generator_audio_x0.float() - grad_audio.float()).detach(),
        )
        return loss
