# SPDX-License-Identifier: Apache-2.0
"""DMD2 distillation method (algorithm layer)."""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
import torch.nn.functional as F

from fastvideo.train.methods.base import TrainingMethod, LogScalar
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.optimizer import (
    build_optimizer_and_scheduler, )
from fastvideo.train.utils.config import (
    get_optional_float,
    get_optional_int,
    parse_betas,
)


class DMD2Method(TrainingMethod):
    """DMD2 distillation algorithm (method layer).

    Owns role model instances directly:
    - ``self.student`` — trainable student :class:`ModelBase`
    - ``self.teacher`` — frozen teacher :class:`ModelBase`
    - ``self.critic`` — trainable critic :class:`ModelBase`
    """

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)

        if "student" not in role_models:
            raise ValueError("DMD2Method requires role 'student'")
        if "teacher" not in role_models:
            raise ValueError("DMD2Method requires role 'teacher'")
        if "critic" not in role_models:
            raise ValueError("DMD2Method requires role 'critic'")

        self.teacher = role_models["teacher"]
        self.critic = role_models["critic"]

        if not self.student._trainable:
            raise ValueError("DMD2Method requires student to be trainable")
        if self.teacher._trainable:
            raise ValueError("DMD2Method requires teacher to be "
                             "non-trainable")
        if not self.critic._trainable:
            raise ValueError("DMD2Method requires critic to be trainable")
        self._cfg_uncond = self._parse_cfg_uncond()
        self._rollout_mode = self._parse_rollout_mode()
        (
            self._rollout_carry,
            self._rollout_carry_slot_count,
            self._rollout_sample_type,
        ) = self._parse_rollout_carry()
        self._init_rollout_carry_state()
        self._rollout_data_forcing = self._parse_rollout_data_forcing()
        self._validate_preprocessed_data_type()
        self._configure_student_negative_conditioning()
        self._denoising_step_list: torch.Tensor | None = (None)
        (
            self._score_min_timestep,
            self._score_max_timestep,
        ) = self._parse_score_timestep_bounds()
        self._score_timestep_shift = self._parse_score_timestep_shift()
        self._fake_score_loss_space = self._parse_fake_score_loss_space()

        # Initialize preprocessors on student.
        self.student.init_preprocessors(self.training_config)

        self._init_optimizers_and_schedulers()

    @property
    def _optimizer_dict(self, ) -> dict[str, torch.optim.Optimizer]:
        return {
            "student": self._student_optimizer,
            "critic": self._critic_optimizer,
        }

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {
            "student": self._student_lr_scheduler,
            "critic": self._critic_lr_scheduler,
        }

    # TrainingMethod override: single_train_step
    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[
            dict[str, torch.Tensor],
            dict[str, Any],
            dict[str, LogScalar],
    ]:
        if self._rollout_carry:
            return self._carried_train_step(batch, iteration)

        latents_source: Literal["data", "zeros"] = "data"
        if self._rollout_mode == "simulate":
            latents_source = "zeros"

        training_batch = self.student.prepare_batch(
            batch,
            generator=self.cuda_generator,
            latents_source=latents_source,
        )

        update_student = self._should_update_student(iteration)

        generator_loss = torch.zeros(
            (),
            device=training_batch.latents.device,
            dtype=torch.float32,
        )
        student_ctx = None
        generator_metrics: dict[str, LogScalar] = {}
        fake_score_loss = torch.zeros_like(generator_loss)
        critic_ctx = None
        critic_outputs: dict[str, Any] = {}
        critic_metrics: dict[str, LogScalar] = {}
        if update_student:
            generator_pred_x0 = self._student_rollout(training_batch, with_grad=True)
            student_ctx = (
                training_batch.timesteps,
                training_batch.attn_metadata_vsa,
            )
            generator_loss, generator_metrics = self._dmd_loss(generator_pred_x0, training_batch)
            training_batch.dmd_latent_vis_dict["generator_pred_video"] = generator_pred_x0.detach()
        else:
            (
                fake_score_loss,
                critic_ctx,
                critic_outputs,
                critic_metrics,
            ) = self._critic_flow_matching_loss(training_batch)

        total_loss = generator_loss + fake_score_loss
        loss_map = {
            "total_loss": total_loss,
            "generator_loss": generator_loss,
            "fake_score_loss": fake_score_loss,
        }

        outputs: dict[str, Any] = dict(critic_outputs)
        outputs["_fv_backward"] = {
            "update_student": update_student,
            "student_ctx": student_ctx,
            "critic_ctx": critic_ctx,
        }
        metrics: dict[str, LogScalar] = {
            "update_student": float(update_student),
            **generator_metrics,
            **critic_metrics,
        }
        # Rank-local latent snapshots for LatentVisCallback.
        self.latent_vis = {
            **(training_batch.fake_score_latent_vis_dict or {}),
            **(training_batch.dmd_latent_vis_dict or {}),
            "_fv_latent_layout":
            getattr(training_batch, "minimax_h3_dmd_layout", None),
        }
        return loss_map, outputs, metrics

    # TrainingMethod override: backward
    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int = 1,
    ) -> None:
        grad_accum_rounds = max(1, int(grad_accum_rounds))
        backward_ctx = outputs.get("_fv_backward")
        if not isinstance(backward_ctx, dict):
            super().backward(
                loss_map,
                outputs,
                grad_accum_rounds=grad_accum_rounds,
            )
            return

        update_student = bool(backward_ctx.get("update_student", False))
        if update_student:
            student_ctx = backward_ctx.get("student_ctx")
            if student_ctx is None:
                raise RuntimeError("Missing student backward context")
            self.student.backward(
                loss_map["generator_loss"],
                student_ctx,
                grad_accum_rounds=grad_accum_rounds,
            )
            return

        critic_ctx = backward_ctx.get("critic_ctx")
        if critic_ctx is None:
            raise RuntimeError("Missing critic backward context")
        self.critic.backward(
            loss_map["fake_score_loss"],
            critic_ctx,
            grad_accum_rounds=grad_accum_rounds,
        )

    # TrainingMethod override: get_optimizers
    def get_optimizers(
        self,
        iteration: int,
    ) -> list[torch.optim.Optimizer]:
        if self._should_update_student(iteration):
            return [self._student_optimizer]
        return [self._critic_optimizer]

    # TrainingMethod override: get_lr_schedulers
    def get_lr_schedulers(
        self,
        iteration: int,
    ) -> list[Any]:
        if self._should_update_student(iteration):
            return [self._student_lr_scheduler]
        return [self._critic_lr_scheduler]

    # TrainingMethod override: get_grad_clip_targets
    def get_grad_clip_targets(
        self,
        iteration: int,
    ) -> dict[str, torch.nn.Module]:
        if self._should_update_student(iteration):
            return {"student": self.student.transformer}
        return {"critic": self.critic.transformer}

    def _parse_rollout_mode(self, ) -> Literal["simulate", "data_latent"]:
        """Parse how DMD2 obtains the latent point used for rollout.

        ``simulate`` starts from fresh noise and lets the student create an
        artificial latent trajectory, so it can run with text-only data.
        ``data_latent`` starts from preprocessed VAE latents and perturbs them
        at a sampled denoising timestep.
        """
        raw = self.method_config.get("rollout_mode", None)
        if raw is None:
            raise ValueError("method_config.rollout_mode must be set "
                             "for DMD2")
        if not isinstance(raw, str):
            raise ValueError("method_config.rollout_mode must be a "
                             "string, "
                             f"got {type(raw).__name__}")
        mode = raw.strip().lower()
        if mode in ("simulate", "sim"):
            return "simulate"
        if mode in ("data_latent", "data", "vae_latent"):
            return "data_latent"
        raise ValueError("method_config.rollout_mode must be one of "
                         "{simulate, data_latent}, got "
                         f"{raw!r}")

    def _parse_rollout_carry(self) -> tuple[bool, int, Literal["ode", "sde"]]:
        """Parse the carried backward-simulation knobs.

        ``rollout_carry: true`` walks the student's own sampling grid one
        rung per ``single_train_step`` call, carrying the trajectory in
        memory across calls (FastGen's backward simulation): exactly one
        generation forward per call instead of one full-grid walk. Off
        (default) keeps the existing full-rollout behavior unchanged.

        ``rollout_carry_slots`` is the number of independent trajectory
        streams per rank; it must equal
        ``training.loop.gradient_accumulation_steps`` because the trainer
        calls ``single_train_step`` once per accumulation round and the
        slots are selected round-robin over calls. Defaults to that value.

        ``rollout_sample_type`` picks how the walk re-noises onto the next
        rung: ``sde`` draws fresh noise (the existing rollout behavior),
        ``ode`` reuses the noise the current state implies per modality —
        the deterministic step the FastGen H3 recipe uses.
        """
        raw_carry = self.method_config.get("rollout_carry", None)
        if raw_carry is None:
            raw_carry = False
        if not isinstance(raw_carry, bool):
            raise ValueError("method.rollout_carry must be a bool, got "
                             f"{type(raw_carry).__name__}")
        carry = bool(raw_carry)

        raw_sample_type = self.method_config.get("rollout_sample_type", None)
        sample_type: Literal["ode", "sde"] = "sde"
        if raw_sample_type is not None:
            if not isinstance(raw_sample_type, str):
                raise ValueError("method.rollout_sample_type must be a "
                                 "string, got "
                                 f"{type(raw_sample_type).__name__}")
            normalized = raw_sample_type.strip().lower()
            if normalized not in ("ode", "sde"):
                raise ValueError("method.rollout_sample_type must be one of "
                                 f"{{ode, sde}}, got {raw_sample_type!r}")
            if not carry:
                raise ValueError("method.rollout_sample_type requires "
                                 "method.rollout_carry: true")
            sample_type = normalized  # type: ignore[assignment]

        slots_raw = get_optional_int(
            self.method_config,
            "rollout_carry_slots",
            where="method.rollout_carry_slots",
        )
        if not carry:
            if slots_raw is not None:
                raise ValueError("method.rollout_carry_slots requires "
                                 "method.rollout_carry: true")
            return False, 0, sample_type

        if self._rollout_mode != "simulate":
            raise ValueError("method.rollout_carry: true requires "
                             "method.rollout_mode: simulate")

        grad_accum = max(
            1,
            int(self.training_config.loop.gradient_accumulation_steps or 1),
        )
        slots = grad_accum if slots_raw is None else int(slots_raw)
        if slots <= 0:
            raise ValueError("method.rollout_carry_slots must be positive, "
                             f"got {slots}")
        if slots != grad_accum:
            # The trainer calls single_train_step once per accumulation round
            # without passing the round index; the round-robin slot selection
            # only matches the trainer's cadence when the counts agree.
            raise ValueError("method.rollout_carry_slots must equal "
                             "training.loop.gradient_accumulation_steps, got "
                             f"slots={slots} vs "
                             f"gradient_accumulation_steps={grad_accum}")

        if sample_type == "ode" and not callable(getattr(self.student, "extract_eps", None)):
            raise ValueError("method.rollout_sample_type: ode requires the "
                             "student model to implement "
                             "extract_eps(noisy_latents, clean_latents, "
                             "timestep)")

        _, stagger_groups = self._rollout_carry_rank_world()
        if bool(getattr(getattr(self.training_config, "data", None), "native_shape_bucketing", False)):
            stagger_groups = 1
        self._validate_rollout_carry_coverage(
            streams=stagger_groups * slots,
            grid_len=self._rollout_grid_length(),
            interval=self._generator_update_interval(),
        )
        return True, slots, sample_type

    @staticmethod
    def _validate_rollout_carry_coverage(
        *,
        streams: int,
        grid_len: int,
        interval: int,
    ) -> None:
        """Reject configurations that leave student rungs untrained.

        Student updates revisit a stream's rung modulo
        ``gcd(len(dmd_denoising_steps), generator_update_interval)``, so
        every residue class must be represented by a (rank, slot) stream;
        the consecutive stagger offsets cover all classes exactly when
        there are at least ``gcd`` streams.
        """
        phase_classes = math.gcd(grid_len, interval)
        if streams < phase_classes:
            raise ValueError("Carried backward-simulation DMD2 cannot cover every student "
                             f"rung with len(dmd_denoising_steps)={grid_len}, "
                             f"generator_update_interval={interval}, and "
                             f"{streams} trajectory stream(s) (stagger groups x slots). "
                             "Student updates preserve the rung modulo "
                             f"gcd(grid, interval)={phase_classes}, but only {streams} "
                             "stream phase(s) are present. Use at least that many "
                             "rank-slot streams or choose a coprime update interval.")

    def _rollout_carry_rank_world(self) -> tuple[int, int]:
        """Stagger rank and stream-group count for the carried rollout.

        Mirrors ``TrainingMethod.on_train_start``'s RNG grouping: ranks
        inside one sequence-parallel group shard the same document and must
        walk one shared trajectory, so they share a stagger rank (with
        ``sp_size=1`` this is exactly the global rank). Falls back to a
        single group when the distributed world is not initialized (CPU
        tests, single-process runs).
        """
        try:
            from fastvideo.distributed import get_world_group
            world_group = get_world_group()
            global_rank = int(world_group.rank)
            world_size = int(world_group.world_size)
        except (AssertionError, ImportError, RuntimeError):
            global_rank, world_size = 0, 1
        sp_size = max(
            1,
            int(getattr(self.training_config.distributed, "sp_size", 1) or 1),
        )
        return global_rank // sp_size, max(1, world_size // sp_size)

    def _parse_rollout_data_forcing(self) -> bool:
        """Parse per-batch data forcing for the carried walk.

        ``rollout_data_forcing: true`` routes latent-bearing batches (t2va
        parquet rows in a mixed ``data_path``) onto FastGen's data-driven
        student inputs — the real packed latents forward-noised at a
        uniformly drawn grid rung (``sample_from_t_list`` semantics) —
        while text-only batches keep walking the carried backward
        simulation. Off (default) keeps every batch on the walk,
        byte-identical to the carry-only behavior.
        """
        raw = self.method_config.get("rollout_data_forcing", None)
        if raw is None:
            return False
        if not isinstance(raw, bool):
            raise ValueError("method.rollout_data_forcing must be a bool, "
                             f"got {type(raw).__name__}")
        if raw and not self._rollout_carry:
            raise ValueError("method.rollout_data_forcing: true requires "
                             "method.rollout_carry: true; without the carry, "
                             "always-forced inputs are "
                             "method.rollout_mode: data_latent")
        return raw

    @staticmethod
    def _batch_has_latents(batch: dict[str, Any]) -> bool:
        """Classify a mixed-loading batch as latent-bearing or text-only.

        Under the t2va parquet schema the collate emits empty (numel-0)
        tensors for latent columns a text-only row does not carry, so
        presence means "key exists and non-empty". A row carrying exactly
        one of the pair is corrupt data, not a batch type.
        """
        video = batch.get("vae_latent")
        audio = batch.get("audio_latent")
        has_video = isinstance(video, torch.Tensor) and video.numel() > 0
        has_audio = isinstance(audio, torch.Tensor) and audio.numel() > 0
        if has_video != has_audio:
            raise ValueError("Mixed-loading batch carries exactly one of "
                             "vae_latent/audio_latent non-empty; a t2va row "
                             "must carry both and a text_only row neither "
                             f"(vae_latent={'present' if has_video else 'empty/missing'}, "
                             f"audio_latent={'present' if has_audio else 'empty/missing'})")
        return has_video

    def _rollout_grid_length(self) -> int:
        raw = self.method_config.get("dmd_denoising_steps", None)
        if not isinstance(raw, list) or not raw:
            raise ValueError("method_config.dmd_denoising_steps must "
                             "be set for DMD2 distillation")
        return len(raw)

    def _init_rollout_carry_state(self) -> None:
        # Transient in-memory trajectory state: one independent slot per
        # gradient-accumulation round, selected round-robin over calls.
        # Intentionally never checkpointed (mirrors FastGen's CarryCallback):
        # on resume every slot restarts from fresh noise — a brief warmup
        # until the walk is mid-trajectory again.
        slots = max(0, int(self._rollout_carry_slot_count))
        self._carry_call_count = 0
        self._carry_slots: list[dict[str, Any] | None] = [None] * slots
        self._carry_slot_seeded: list[bool] = [False] * slots

    def _validate_preprocessed_data_type(self) -> None:
        data_type = str(getattr(
            self.training_config.data,
            "preprocessed_data_type",
            "t2v",
        )).strip().lower()
        if data_type == "text_only" and self._rollout_mode != "simulate":
            raise ValueError("training.data.preprocessed_data_type='text_only' "
                             "requires method.rollout_mode='simulate'; "
                             "data_latent rollout requires vae_latent data.")
        if self._rollout_data_forcing and data_type != "t2va":
            raise ValueError("method.rollout_data_forcing: true requires "
                             "training.data.preprocessed_data_type='t2va': the "
                             "t2va parquet schema is the superset that reads "
                             "latent columns; text-only roots mixed into the "
                             "same data_path yield empty latent columns and "
                             "route to the carried walk.")

    def _uses_negative_prompt_conditioning(self) -> bool:
        if self._cfg_uncond is None:
            return True
        text_policy = self._cfg_uncond.get("text", None)
        if text_policy is None:
            return True
        return str(text_policy).strip().lower() == "negative_prompt"

    def _configure_student_negative_conditioning(self) -> None:
        setter = getattr(
            self.student,
            "set_requires_negative_conditioning",
            None,
        )
        if setter is not None:
            setter(self._uses_negative_prompt_conditioning())

    def _parse_cfg_uncond(self, ) -> dict[str, Any] | None:
        raw = self.method_config.get("cfg_uncond", None)
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("method_config.cfg_uncond must be a dict "
                             f"when set, got {type(raw).__name__}")

        cfg: dict[str, Any] = dict(raw)

        on_missing_raw = cfg.get("on_missing", "error")
        if on_missing_raw is None:
            on_missing_raw = "error"
        if not isinstance(on_missing_raw, str):
            raise ValueError("method_config.cfg_uncond.on_missing must "
                             "be a string, got "
                             f"{type(on_missing_raw).__name__}")
        on_missing = on_missing_raw.strip().lower()
        if on_missing not in {"error", "ignore"}:
            raise ValueError("method_config.cfg_uncond.on_missing must "
                             "be one of {error, ignore}, got "
                             f"{on_missing_raw!r}")
        cfg["on_missing"] = on_missing

        for channel, policy_raw in list(cfg.items()):
            if channel == "on_missing":
                continue
            if policy_raw is None:
                continue
            if not isinstance(policy_raw, str):
                raise ValueError("method_config.cfg_uncond values must "
                                 "be strings, got "
                                 f"{channel}="
                                 f"{type(policy_raw).__name__}")
            policy = policy_raw.strip().lower()
            allowed = {"keep", "zero", "drop"}
            if channel == "text":
                allowed = {*allowed, "negative_prompt"}
            if policy not in allowed:
                raise ValueError("method_config.cfg_uncond values must "
                                 "be one of "
                                 f"{sorted(allowed)}, got "
                                 f"{channel}={policy_raw!r}")
            cfg[channel] = policy

        return cfg

    def _init_optimizers_and_schedulers(self) -> None:
        tc = self.training_config

        # Student optimizer/scheduler.
        student_lr = float(tc.optimizer.learning_rate)
        student_betas = tc.optimizer.betas
        student_sched = str(tc.optimizer.lr_scheduler)
        student_params = [p for p in self.student.transformer.parameters() if p.requires_grad]
        (
            self._student_optimizer,
            self._student_lr_scheduler,
        ) = build_optimizer_and_scheduler(
            params=student_params,
            optimizer_config=tc.optimizer,
            loop_config=tc.loop,
            learning_rate=student_lr,
            betas=student_betas,
            scheduler_name=student_sched,
        )

        # Critic optimizer/scheduler — must be set in
        # method config.
        critic_lr_raw = get_optional_float(
            self.method_config,
            "fake_score_learning_rate",
            where="method.fake_score_learning_rate",
        )
        if critic_lr_raw is None or critic_lr_raw == 0.0:
            raise ValueError("method.fake_score_learning_rate must "
                             "be set to a positive value")
        critic_lr = float(critic_lr_raw)

        critic_betas_raw = self.method_config.get("fake_score_betas", None)
        if critic_betas_raw is None:
            raise ValueError("method.fake_score_betas must be set "
                             "(e.g. [0.0, 0.999])")
        critic_betas = parse_betas(
            critic_betas_raw,
            where="method.fake_score_betas",
        )

        critic_sched_raw = self.method_config.get("fake_score_lr_scheduler", None)
        if critic_sched_raw is None:
            raise ValueError("method.fake_score_lr_scheduler must "
                             "be set (e.g. 'constant')")
        critic_sched = str(critic_sched_raw)
        critic_params = [p for p in self.critic.transformer.parameters() if p.requires_grad]
        (
            self._critic_optimizer,
            self._critic_lr_scheduler,
        ) = build_optimizer_and_scheduler(
            params=critic_params,
            optimizer_config=tc.optimizer,
            loop_config=tc.loop,
            learning_rate=critic_lr,
            betas=critic_betas,
            scheduler_name=critic_sched,
        )

    @staticmethod
    def _add_noise_for_batch(
        model: ModelBase,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
    ) -> torch.Tensor:
        """Call the batch-aware hook while retaining lightweight test doubles."""
        hook = getattr(model, "add_noise_for_batch", None)
        if hook is not None and batch is not None:
            return hook(clean_latents, noise, timestep, batch)
        return model.add_noise(clean_latents, noise, timestep)

    @staticmethod
    def _extract_eps_for_batch(
        model: ModelBase,
        noisy_latents: torch.Tensor,
        clean_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
    ) -> torch.Tensor:
        hook = getattr(model, "extract_eps_for_batch", None)
        if hook is not None and batch is not None:
            return hook(noisy_latents, clean_latents, timestep, batch)
        extractor = getattr(model, "extract_eps", None)
        if not callable(extractor):
            raise TypeError(f"{type(model).__name__} does not implement extract_eps")
        return extractor(noisy_latents, clean_latents, timestep)

    def _modality_slices(self, batch: Any) -> tuple[tuple[str, slice], ...] | None:
        """Return slices used to normalize packed modalities independently."""
        batch_getter = getattr(self.student, "modality_slices_for_batch", None)
        if batch_getter is not None:
            slices = tuple(batch_getter(batch))
            return slices or None
        getter = getattr(self.student, "modality_slices", None)
        if getter is None:
            return None
        slices = tuple(getter())
        return slices or None

    def _modality_weight(self, name: str) -> float:
        raw = self.method_config.get("modality_loss_weights", None)
        if not isinstance(raw, dict):
            return 1.0
        value = raw.get(name, 1.0)
        return 1.0 if value is None else float(value)

    def apply_configured_lrs(self) -> None:
        """Force student/critic LRs back to the configured values (post-resume)."""
        student_lr = float(self.training_config.optimizer.learning_rate)
        critic_lr = float(self.method_config.get("fake_score_learning_rate"))
        for optimizer, scheduler, lr in (
            (self._student_optimizer, self._student_lr_scheduler, student_lr),
            (self._critic_optimizer, self._critic_lr_scheduler, critic_lr),
        ):
            for group in optimizer.param_groups:
                group["lr"] = lr
                if "initial_lr" in group:
                    group["initial_lr"] = lr
            if hasattr(scheduler, "base_lrs"):
                scheduler.base_lrs = [lr] * len(scheduler.base_lrs)

    def _generator_update_interval(self) -> int:
        interval = get_optional_int(
            self.method_config,
            "generator_update_interval",
            where="method.generator_update_interval",
        )
        if interval is None:
            interval = 5
        if interval <= 0:
            raise ValueError("method.generator_update_interval must be positive")
        return interval

    def _should_update_student(
        self,
        iteration: int,
    ) -> bool:
        return iteration % self._generator_update_interval() == 0

    def _get_denoising_step_list(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        if (self._denoising_step_list is not None and self._denoising_step_list.device == device):
            return self._denoising_step_list

        raw = self.method_config.get("dmd_denoising_steps", None)
        if not isinstance(raw, list) or not raw:
            raise ValueError("method_config.dmd_denoising_steps must "
                             "be set for DMD2 distillation")

        steps = torch.tensor(
            [int(s) for s in raw],
            dtype=torch.long,
            device=device,
        )

        warp = self.method_config.get("warp_denoising_step", None)
        if warp is None:
            warp = False
        if bool(warp):
            timesteps = torch.cat((
                self.student.noise_scheduler.timesteps.to("cpu"),
                torch.tensor([0], dtype=torch.float32),
            )).to(device)
            steps = timesteps[1000 - steps]

        self._denoising_step_list = steps
        return steps

    def _sample_rollout_timestep(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        step_list = self._get_denoising_step_list(device)
        index = torch.randint(
            0,
            len(step_list),
            [1],
            device=device,
            dtype=torch.long,
            generator=self.cuda_generator,
        )
        return step_list[index]

    def _parse_score_timestep_bounds(self) -> tuple[int, int]:
        """Resolve the score-model timestep window.

        The student rollout schedule is controlled separately by
        ``dmd_denoising_steps``. These bounds apply only to the randomly
        sampled teacher/critic score timestep.
        """
        min_ratio = get_optional_float(
            self.method_config,
            "min_timestep_ratio",
            where="method.min_timestep_ratio",
        )
        max_ratio = get_optional_float(
            self.method_config,
            "max_timestep_ratio",
            where="method.max_timestep_ratio",
        )
        min_ratio = 0.0 if min_ratio is None else float(min_ratio)
        max_ratio = 1.0 if max_ratio is None else float(max_ratio)
        if not 0.0 <= min_ratio <= max_ratio <= 1.0:
            raise ValueError("method min/max_timestep_ratio must satisfy "
                             "0 <= min <= max <= 1, got "
                             f"min={min_ratio}, max={max_ratio}")

        num_timesteps = int(self.student.num_train_timesteps)
        return (
            int(min_ratio * num_timesteps),
            int(max_ratio * num_timesteps),
        )

    def _parse_score_timestep_shift(self) -> float:
        """Resolve score-sampling density on the rectified-flow time axis.

        ``1`` samples base timesteps uniformly. A value ``s`` samples uniformly
        after the rational shift by drawing shifted sigma and inverting it.
        """
        shift = get_optional_float(
            self.method_config,
            "score_timestep_shift",
            where="method.score_timestep_shift",
        )
        shift = 1.0 if shift is None else float(shift)
        if shift <= 0.0:
            raise ValueError("method.score_timestep_shift must be > 0, "
                             f"got {shift}")
        return shift

    def _parse_fake_score_loss_space(self) -> dict[str, str]:
        """Resolve the critic regression space, globally or per modality.

        ``velocity`` is plain velocity MSE; ``x0`` multiplies each modality's
        velocity MSE by its realized sigma_m(t)^2 (the affine x0-space form).
        With one shared base timestep and unequal shifts (H3: video 12,
        audio 3), sigma_audio(t) << sigma_video(t) for most draws, so a
        global ``x0`` space suppresses the critic's audio gradient across
        the low half of audio's own noise axis — the critic goes blind
        there and audio's DMD gradients degenerate. A per-modality mapping
        such as ``{video: x0, audio: velocity}`` keeps the x0 weighting for
        video without silencing audio.
        """
        raw = self.method_config.get("fake_score_loss_space", None)
        if raw is None:
            return {"__default__": "velocity"}
        if isinstance(raw, str):
            mapping = {"__default__": raw}
        elif isinstance(raw, dict):
            mapping = {str(k).strip().lower(): str(v) for k, v in raw.items()}
            mapping.setdefault("__default__", "velocity")
        else:
            raise ValueError("method.fake_score_loss_space must be a string "
                             "or a {modality: space} mapping, got "
                             f"{type(raw).__name__}")
        normalized: dict[str, str] = {}
        for key, value in mapping.items():
            space = str(value).strip().lower()
            if space not in ("velocity", "x0"):
                raise ValueError("method.fake_score_loss_space values must be "
                                 f"one of {{velocity, x0}}, got {value!r} "
                                 f"for {key!r}")
            normalized[key] = space
        return normalized

    def _fake_score_space_for(self, modality_name: str) -> str:
        return self._fake_score_loss_space.get(
            modality_name,
            self._fake_score_loss_space["__default__"],
        )

    def _sample_score_timestep(self, device: torch.device) -> torch.Tensor:
        shift = self._score_timestep_shift
        if shift == 1.0:
            # Draw inside the bounds directly; drawing over the full range
            # and clamping piles probability atoms onto both endpoints.
            timestep = torch.randint(
                self._score_min_timestep,
                self._score_max_timestep + 1,
                [1],
                device=device,
                dtype=torch.long,
                generator=self.cuda_generator,
            )
        else:
            num_timesteps = float(self.student.num_train_timesteps)
            t_lo = self._score_min_timestep / num_timesteps
            t_hi = self._score_max_timestep / num_timesteps
            sigma_lo = shift * t_lo / (1.0 + (shift - 1.0) * t_lo)
            sigma_hi = shift * t_hi / (1.0 + (shift - 1.0) * t_hi)
            u = torch.rand(
                [1],
                device=device,
                dtype=torch.float32,
                generator=self.cuda_generator,
            ) * (sigma_hi - sigma_lo) + sigma_lo
            t = u / (shift - (shift - 1.0) * u)
            timestep = (t * num_timesteps).round().to(torch.long)
        timestep = self.student.shift_and_clamp_timestep(timestep)
        return timestep.clamp(
            self._score_min_timestep,
            self._score_max_timestep,
        )

    def _student_rollout(
        self,
        batch: Any,
        *,
        with_grad: bool,
    ) -> torch.Tensor:
        latents = batch.latents
        device = latents.device
        dtype = latents.dtype
        step_list = self._get_denoising_step_list(device)

        if self._rollout_mode != "simulate":
            timestep = self._sample_rollout_timestep(device)
            noise = torch.randn(
                latents.shape,
                device=device,
                dtype=dtype,
                generator=self.cuda_generator,
            )
            noisy_latents = self._add_noise_for_batch(self.student, latents, noise, timestep, batch)
            pred_x0 = self.student.predict_x0(
                noisy_latents,
                timestep,
                batch,
                conditional=True,
                cfg_uncond=self._cfg_uncond,
                attn_kind="vsa",
            )
            batch.dmd_latent_vis_dict["generator_timestep"] = timestep
            return pred_x0

        target_timestep_idx = torch.randint(
            0,
            len(step_list),
            [1],
            device=device,
            dtype=torch.long,
            generator=self.cuda_generator,
        )
        target_timestep_idx_int = int(target_timestep_idx.item())
        target_timestep = step_list[target_timestep_idx]

        current_noise_latents = torch.randn(
            latents.shape,
            device=device,
            dtype=dtype,
            generator=self.cuda_generator,
        )
        current_noise_latents_copy = (current_noise_latents.clone())

        max_target_idx = len(step_list) - 1
        noise_latents: list[torch.Tensor] = []
        noise_latent_index = target_timestep_idx_int - 1

        if max_target_idx > 0:
            with torch.no_grad():
                for step_idx in range(max_target_idx):
                    current_timestep = step_list[step_idx]
                    current_timestep_tensor = (current_timestep * torch.ones(
                        1,
                        device=device,
                        dtype=torch.long,
                    ))

                    pred_clean = self.student.predict_x0(
                        current_noise_latents,
                        current_timestep_tensor,
                        batch,
                        conditional=True,
                        cfg_uncond=self._cfg_uncond,
                        attn_kind="vsa",
                    )

                    next_timestep = step_list[step_idx + 1]
                    next_timestep_tensor = (next_timestep * torch.ones(
                        1,
                        device=device,
                        dtype=torch.long,
                    ))
                    noise = torch.randn(
                        latents.shape,
                        device=device,
                        dtype=pred_clean.dtype,
                        generator=self.cuda_generator,
                    )
                    current_noise_latents = self._add_noise_for_batch(
                        self.student,
                        pred_clean,
                        noise,
                        next_timestep_tensor,
                        batch,
                    )
                    noise_latents.append(current_noise_latents.clone())

        if noise_latent_index >= 0:
            if noise_latent_index >= len(noise_latents):
                raise RuntimeError("noise_latent_index is out of bounds")
            noisy_input = noise_latents[noise_latent_index]
        else:
            noisy_input = current_noise_latents_copy

        if with_grad:
            pred_x0 = self.student.predict_x0(
                noisy_input,
                target_timestep,
                batch,
                conditional=True,
                cfg_uncond=self._cfg_uncond,
                attn_kind="vsa",
            )
        else:
            with torch.no_grad():
                pred_x0 = self.student.predict_x0(
                    noisy_input,
                    target_timestep,
                    batch,
                    conditional=True,
                    cfg_uncond=self._cfg_uncond,
                    attn_kind="vsa",
                )

        batch.dmd_latent_vis_dict["generator_timestep"] = target_timestep.float().detach()
        return pred_x0

    # ------------------------------------------------------------------
    # Carried backward simulation — the student's own trajectory, walked
    # one rung per single_train_step call (port of FastGen's
    # _backward_simulation / _staggered_start / _advance_carry).
    # ------------------------------------------------------------------

    def _carried_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[
            dict[str, torch.Tensor],
            dict[str, Any],
            dict[str, LogScalar],
    ]:
        """One backward-simulation call: one generation forward, carried state.

        A multistep student is only ever correct on its own sampling
        trajectory, and walking the full grid every call costs
        ``len(dmd_denoising_steps)`` forwards. Instead the walk is spread
        over consecutive calls: each call pays for exactly one student
        forward at the carried rung, both phases (student and critic)
        consume it — the critic is fit on the same simulated states the
        student trains on — and both advance the trajectory. Each
        grad-accum round owns an independent slot, selected round-robin
        because the trainer does not pass the round index.

        An empty slot (first ever use, cleared after a finished trajectory,
        or after a resume — the carry is transient and never checkpointed)
        starts a fresh trajectory from noise and adopts the incoming loader
        batch's conditioning; mid-walk calls ignore the fresh loader batch
        and rebuild the training batch from the carried raw batch, since a
        trajectory keeps the prompt it set out with.
        """
        slot = self._carry_call_count % self._rollout_carry_slot_count
        self._carry_call_count += 1

        # Per-batch routing (off unless method.rollout_data_forcing): a batch
        # that carries real latents trains on them at a noised grid rung and
        # leaves this slot's walk untouched.
        if self._rollout_data_forcing and self._batch_has_latents(batch):
            return self._data_forced_train_step(batch, slot, iteration)

        carried = self._carry_slots[slot]
        raw_batch = (self._carry_snapshot_raw_batch(batch) if carried is None else carried["raw_batch"])

        training_batch = self.student.prepare_batch(
            raw_batch,
            generator=self.cuda_generator,
            latents_source="zeros",
        )
        latents = training_batch.latents
        device = latents.device
        step_list = self._get_denoising_step_list(device)

        if carried is None:
            rung = 0
            state = torch.randn(
                latents.shape,
                device=device,
                dtype=latents.dtype,
                generator=self.cuda_generator,
            )
            # Stagger only the first-ever fill of each slot; later fresh
            # starts begin at rung 0 with no pre-walk and stay out of phase
            # naturally.
            if not self._carry_slot_seeded[slot]:
                self._carry_slot_seeded[slot] = True
                state, rung = self._staggered_start(
                    state,
                    training_batch,
                    step_list,
                    slot,
                )
        else:
            rung = int(carried["rung"])
            state = carried["state"]

        timestep = step_list[rung] * torch.ones(
            1,
            device=device,
            dtype=torch.long,
        )

        update_student = self._should_update_student(iteration)

        generator_loss = torch.zeros((), device=device, dtype=torch.float32)
        fake_score_loss = torch.zeros_like(generator_loss)
        student_ctx = None
        critic_ctx = None
        critic_outputs: dict[str, Any] = {}
        generator_metrics: dict[str, LogScalar] = {}
        critic_metrics: dict[str, LogScalar] = {}
        if update_student:
            generator_pred_x0 = self.student.predict_x0(
                state,
                timestep,
                training_batch,
                conditional=True,
                cfg_uncond=self._cfg_uncond,
                attn_kind="vsa",
            )
            student_ctx = (
                training_batch.timesteps,
                training_batch.attn_metadata_vsa,
            )
            generator_loss, generator_metrics = self._dmd_loss(generator_pred_x0, training_batch)
            training_batch.dmd_latent_vis_dict["generator_pred_video"] = generator_pred_x0.detach()
        else:
            with torch.no_grad():
                generator_pred_x0 = self.student.predict_x0(
                    state,
                    timestep,
                    training_batch,
                    conditional=True,
                    cfg_uncond=self._cfg_uncond,
                    attn_kind="vsa",
                )
            (
                fake_score_loss,
                critic_ctx,
                critic_outputs,
                critic_metrics,
            ) = self._critic_flow_matching_loss(
                training_batch,
                generator_pred_x0=generator_pred_x0,
            )
        training_batch.dmd_latent_vis_dict["generator_timestep"] = timestep.float().detach()

        # Advance after the loss path: both phases generated, so both hand
        # the trajectory on.
        self._advance_carry(
            slot,
            state,
            generator_pred_x0,
            timestep,
            rung,
            step_list,
            training_batch,
            raw_batch,
        )

        total_loss = generator_loss + fake_score_loss
        loss_map = {
            "total_loss": total_loss,
            "generator_loss": generator_loss,
            "fake_score_loss": fake_score_loss,
        }
        outputs: dict[str, Any] = dict(critic_outputs)
        outputs["_fv_backward"] = {
            "update_student": update_student,
            "student_ctx": student_ctx,
            "critic_ctx": critic_ctx,
        }
        metrics: dict[str, LogScalar] = {
            "update_student": float(update_student),
            "rollout_step": float(rung),
            **generator_metrics,
            **critic_metrics,
        }
        if self._rollout_data_forcing:
            # The running mean of this metric is the realized latent-row
            # fraction of the mix; emitted only when routing is enabled so
            # carry-only runs keep their exact metric set.
            metrics["data_forced"] = 0.0
        # Rank-local latent snapshots for LatentVisCallback.
        self.latent_vis = {
            **(training_batch.fake_score_latent_vis_dict or {}),
            **(training_batch.dmd_latent_vis_dict or {}),
            "_fv_latent_layout":
            getattr(training_batch, "minimax_h3_dmd_layout", None),
        }
        return loss_map, outputs, metrics

    def _data_forced_train_step(
        self,
        batch: dict[str, Any],
        slot: int,
        iteration: int,
    ) -> tuple[
            dict[str, torch.Tensor],
            dict[str, Any],
            dict[str, LogScalar],
    ]:
        """One data-forced call: train on real latents noised at a grid rung.

        FastGen's data-driven multistep student inputs (its
        ``backward_simulation: false`` regime): ``t_student`` is drawn
        uniformly over the student grid's rungs — ``sample_from_t_list``
        semantics, never t=0 — and the real packed latents are
        forward-noised to that rung under each modality's shift, exactly
        the uncarried ``rollout_mode: data_latent`` math. The slot's
        carried walk pauses untouched and resumes on this stream's next
        text-only batch: FastGen picks one regime per config, so pausing
        is the minimal per-batch composition of its two modes. Both
        phases consume the same forced generation, mirroring the carried
        step's critic passthrough.

        The slot's one-time stagger pre-walk still runs on its first-ever
        call even when that call is data-forced: the pre-walk's FSDP
        collective count must stay uniform across ranks, and ranks whose
        first batch is text-only run theirs on this same call. The seeded
        walk adopts this batch's conditioning and waits at its stagger
        rung.
        """
        training_batch = self.student.prepare_batch(
            batch,
            generator=self.cuda_generator,
            latents_source="data",
        )
        latents = training_batch.latents
        device = latents.device
        if not self._carry_slot_seeded[slot]:
            self._carry_slot_seeded[slot] = True
            step_list = self._get_denoising_step_list(device)
            state = torch.randn(
                latents.shape,
                device=device,
                dtype=latents.dtype,
                generator=self.cuda_generator,
            )
            state, rung = self._staggered_start(
                state,
                training_batch,
                step_list,
                slot,
            )
            self._carry_slots[slot] = {
                "state": state.detach(),
                "rung": rung,
                "raw_batch": self._carry_snapshot_raw_batch(batch),
            }

        forced_timestep = self._sample_rollout_timestep(device)
        noise = torch.randn(
            latents.shape,
            device=device,
            dtype=latents.dtype,
            generator=self.cuda_generator,
        )
        noisy_latents = self._add_noise_for_batch(self.student, latents, noise, forced_timestep, training_batch)

        update_student = self._should_update_student(iteration)

        generator_loss = torch.zeros((), device=device, dtype=torch.float32)
        fake_score_loss = torch.zeros_like(generator_loss)
        student_ctx = None
        critic_ctx = None
        critic_outputs: dict[str, Any] = {}
        generator_metrics: dict[str, LogScalar] = {}
        critic_metrics: dict[str, LogScalar] = {}
        if update_student:
            generator_pred_x0 = self.student.predict_x0(
                noisy_latents,
                forced_timestep,
                training_batch,
                conditional=True,
                cfg_uncond=self._cfg_uncond,
                attn_kind="vsa",
            )
            student_ctx = (
                training_batch.timesteps,
                training_batch.attn_metadata_vsa,
            )
            generator_loss, generator_metrics = self._dmd_loss(generator_pred_x0, training_batch)
            training_batch.dmd_latent_vis_dict["generator_pred_video"] = generator_pred_x0.detach()
        else:
            with torch.no_grad():
                generator_pred_x0 = self.student.predict_x0(
                    noisy_latents,
                    forced_timestep,
                    training_batch,
                    conditional=True,
                    cfg_uncond=self._cfg_uncond,
                    attn_kind="vsa",
                )
            (
                fake_score_loss,
                critic_ctx,
                critic_outputs,
                critic_metrics,
            ) = self._critic_flow_matching_loss(
                training_batch,
                generator_pred_x0=generator_pred_x0,
            )
        training_batch.dmd_latent_vis_dict["generator_timestep"] = forced_timestep.float().detach()

        total_loss = generator_loss + fake_score_loss
        loss_map = {
            "total_loss": total_loss,
            "generator_loss": generator_loss,
            "fake_score_loss": fake_score_loss,
        }
        outputs: dict[str, Any] = dict(critic_outputs)
        outputs["_fv_backward"] = {
            "update_student": update_student,
            "student_ctx": student_ctx,
            "critic_ctx": critic_ctx,
        }
        metrics: dict[str, LogScalar] = {
            "update_student": float(update_student),
            "data_forced": 1.0,
            **generator_metrics,
            **critic_metrics,
        }
        # Rank-local latent snapshots for LatentVisCallback.
        self.latent_vis = {
            **(training_batch.fake_score_latent_vis_dict or {}),
            **(training_batch.dmd_latent_vis_dict or {}),
            "_fv_latent_layout":
            getattr(training_batch, "minimax_h3_dmd_layout", None),
        }
        return loss_map, outputs, metrics

    def _carry_snapshot_raw_batch(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        """Adopt the incoming loader batch as a trajectory's conditioning.

        A trajectory keeps the prompt it set out with for its whole walk,
        so the raw dict is snapshotted (tensors detached and kept on the
        student device) and mid-walk calls rebuild the training batch from
        it via ``prepare_batch``; H3's ``prepare_batch`` reads the dict
        without mutating it and rebuilds the packed layout and VSA
        attention metadata deterministically on every call.
        """
        device = self.student.device
        snapshot: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                snapshot[key] = value.detach().to(device)
            else:
                snapshot[key] = value
        return snapshot

    def _staggered_start(
        self,
        state: torch.Tensor,
        training_batch: Any,
        step_list: torch.Tensor,
        slot: int,
    ) -> tuple[torch.Tensor, int]:
        """Pre-walk a fresh trajectory and snapshot it at this stream's rung.

        First-ever fill of a slot only. Each (rank, slot) stream starts at
        ``(stagger_rank * slots + slot) % len(grid)``, spreading the
        streams evenly across the grid so student updates (which revisit
        rungs modulo ``gcd(len(grid), generator_update_interval)``) see
        every rung. Every rank walks the whole grid under ``no_grad``
        regardless of its offset so the FSDP forwards issue a uniform
        collective count — a rank-dependent count would desynchronize the
        all-gathers and hang; only the kept snapshot differs per rank.
        """
        rank, _ = self._rollout_carry_rank_world()
        grid_len = len(step_list)
        # Exact-shape batches must remain shape-synchronous across every rank.
        # Rank-staggered clears would let one rank adopt the next loader bucket
        # while its peers were still carrying the previous geometry. Keep the
        # slot staggering, but make it rank-independent for native-shape data.
        data_config = getattr(self.training_config, "data", None)
        stagger_rank = 0 if bool(getattr(data_config, "native_shape_bucketing", False)) else rank
        offset = (stagger_rank * self._rollout_carry_slot_count + slot) % grid_len
        device = state.device
        snapshot = state
        with torch.no_grad():
            for rung in range(grid_len - 1):
                timestep = step_list[rung] * torch.ones(
                    1,
                    device=device,
                    dtype=torch.long,
                )
                pred_x0 = self.student.predict_x0(
                    state,
                    timestep,
                    training_batch,
                    conditional=True,
                    cfg_uncond=self._cfg_uncond,
                    attn_kind="vsa",
                )
                state = self._renoise(
                    state,
                    pred_x0.detach(),
                    timestep,
                    rung + 1,
                    step_list,
                    training_batch,
                )
                if rung + 1 == offset:
                    snapshot = state
        return snapshot, offset

    def _renoise(
        self,
        state: torch.Tensor,
        pred_x0: torch.Tensor,
        timestep: torch.Tensor,
        next_rung: int,
        step_list: torch.Tensor,
        batch: Any | None = None,
    ) -> torch.Tensor:
        """Re-noise an x0 prediction made at ``timestep`` onto the next rung.

        ``sde`` draws fresh noise — the existing full-rollout hop.
        ``ode`` reuses the noise the current state implies per modality
        (``eps_m = (x_t - alpha_m(t) x0) / sigma_m(t)`` with each
        modality's shifted sigma, via the adapter's ``extract_eps``), the
        deterministic step the FastGen H3 recipe uses.
        """
        device = state.device
        next_timestep = step_list[next_rung] * torch.ones(
            1,
            device=device,
            dtype=torch.long,
        )
        if self._rollout_sample_type == "ode":
            eps = self._extract_eps_for_batch(self.student, state, pred_x0, timestep, batch)
        else:
            eps = torch.randn(
                state.shape,
                device=device,
                dtype=pred_x0.dtype,
                generator=self.cuda_generator,
            )
        return self._add_noise_for_batch(self.student, pred_x0, eps, next_timestep, batch)

    def _advance_carry(
        self,
        slot: int,
        state: torch.Tensor,
        generator_pred_x0: torch.Tensor,
        timestep: torch.Tensor,
        rung: int,
        step_list: torch.Tensor,
        training_batch: Any,
        raw_batch: dict[str, Any],
    ) -> None:
        """Hand the one paid-for step to the slot, or clear a finished walk.

        The advanced state is detached and produced under ``no_grad``: it
        feeds a later call, not a gradient path. Walking past the last rung
        ends the trajectory (the terminal clean sample is never trained
        on), so the slot empties and the next call starts fresh at rung 0.
        """
        if rung + 1 >= len(step_list):
            self._carry_slots[slot] = None
            return
        with torch.no_grad():
            next_state = self._renoise(
                state,
                generator_pred_x0.detach(),
                timestep,
                rung + 1,
                step_list,
                training_batch,
            )
        self._carry_slots[slot] = {
            "state": next_state.detach(),
            "rung": rung + 1,
            "raw_batch": raw_batch,
        }

    def _critic_flow_matching_loss(
        self,
        batch: Any,
        *,
        generator_pred_x0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any, dict[str, Any], dict[str, LogScalar]]:
        if generator_pred_x0 is None:
            with torch.no_grad():
                generator_pred_x0 = self._student_rollout(batch, with_grad=False)

        device = generator_pred_x0.device
        fake_score_timestep = self._sample_score_timestep(device)

        noise = torch.randn(
            generator_pred_x0.shape,
            device=device,
            dtype=generator_pred_x0.dtype,
            generator=self.cuda_generator,
        )
        noisy_x0 = self._add_noise_for_batch(
            self.student,
            generator_pred_x0,
            noise,
            fake_score_timestep,
            batch,
        )

        pred_noise = self.critic.predict_noise(
            noisy_x0,
            fake_score_timestep,
            batch,
            conditional=True,
            cfg_uncond=self._cfg_uncond,
            attn_kind="dense",
        )
        if not isinstance(pred_noise, torch.Tensor):
            raise TypeError("DMD2 critic predict_noise must return one packed tensor")
        target = noise - generator_pred_x0
        slices = self._modality_slices(batch)
        emit_modality_metrics = slices is not None
        if slices is None:
            slices = (("packed", slice(None)), )
        flow_matching_loss = torch.zeros((), device=device, dtype=torch.float32)
        metrics: dict[str, LogScalar] = {}
        for name, modality in slices:
            loss_m = torch.mean((pred_noise[:, modality].float() - target[:, modality].float())**2)
            if self._fake_score_space_for(name) == "x0":
                # For affine rectified flow, x0 MSE is sigma_m(t)^2 times
                # velocity MSE. Estimate sigma_m^2 from the realized tensors.
                with torch.no_grad():
                    num = torch.mean((noisy_x0[:, modality].float() - generator_pred_x0[:, modality].float())**2)
                    den = torch.mean(target[:, modality].float()**2)
                    sigma_sq = num / den
                loss_m = sigma_sq * loss_m
            flow_matching_loss = flow_matching_loss + self._modality_weight(name) * loss_m
            if emit_modality_metrics:
                metrics[f"fake_score_loss_{name}"] = loss_m.detach()

        batch.fake_score_latent_vis_dict = {
            "generator_pred_video": generator_pred_x0,
            "fake_score_timestep": fake_score_timestep,
        }
        outputs = {"fake_score_latent_vis_dict": (batch.fake_score_latent_vis_dict)}
        return (
            flow_matching_loss,
            (batch.timesteps, batch.attn_metadata),
            outputs,
            metrics,
        )

    def _dmd_loss(
        self,
        generator_pred_x0: torch.Tensor,
        batch: Any,
    ) -> tuple[torch.Tensor, dict[str, LogScalar]]:
        guidance_scale = get_optional_float(
            self.method_config,
            "real_score_guidance_scale",
            where="method.real_score_guidance_scale",
        )
        if guidance_scale is None:
            guidance_scale = 1.0
        device = generator_pred_x0.device

        with torch.no_grad():
            timestep = self._sample_score_timestep(device)

            noise = torch.randn(
                generator_pred_x0.shape,
                device=device,
                dtype=generator_pred_x0.dtype,
                generator=self.cuda_generator,
            )
            noisy_latents = self._add_noise_for_batch(
                self.student,
                generator_pred_x0,
                noise,
                timestep,
                batch,
            )

            faker_x0 = self.critic.predict_x0(
                noisy_latents,
                timestep,
                batch,
                conditional=True,
                cfg_uncond=self._cfg_uncond,
                attn_kind="dense",
            )
            real_cond_x0 = self.teacher.predict_x0(
                noisy_latents,
                timestep,
                batch,
                conditional=True,
                cfg_uncond=self._cfg_uncond,
                attn_kind="dense",
            )
            if float(guidance_scale) == 1.0:
                # Scale 1 is the conditional prediction and needs no
                # unconditional forward.
                real_cfg_x0 = real_cond_x0
            else:
                real_uncond_x0 = self.teacher.predict_x0(
                    noisy_latents,
                    timestep,
                    batch,
                    conditional=False,
                    cfg_uncond=self._cfg_uncond,
                    attn_kind="dense",
                )
                real_cfg_x0 = real_uncond_x0 + (real_cond_x0 - real_uncond_x0) * guidance_scale

            # LatentVisCallback decodes these estimates on rank 0.
            batch.dmd_latent_vis_dict.update({
                "real_score_pred_video": real_cfg_x0.detach(),
                "faker_score_pred_video": faker_x0.detach(),
                "dmd_timestep": timestep.detach(),
            })

        slices = self._modality_slices(batch)
        emit_modality_metrics = slices is not None
        if slices is None:
            slices = (("packed", slice(None)), )
        loss = torch.zeros((), device=device, dtype=torch.float32)
        metrics: dict[str, LogScalar] = {}
        for name, modality in slices:
            gen_m = generator_pred_x0[:, modality].float()
            with torch.no_grad():
                # Keep the VSD weight stable for low-precision or degenerate
                # teacher residuals.
                real_m = real_cfg_x0[:, modality].float()
                denom = (gen_m - real_m).abs().mean() + 1e-6
                grad = torch.nan_to_num((faker_x0[:, modality].float() - real_m) / denom)
            loss_m = 0.5 * F.mse_loss(gen_m, (gen_m - grad).detach())
            loss = loss + self._modality_weight(name) * loss_m
            if emit_modality_metrics:
                metrics[f"generator_loss_{name}"] = loss_m.detach()
        return loss, metrics
