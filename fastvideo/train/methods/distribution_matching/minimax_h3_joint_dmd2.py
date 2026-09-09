# SPDX-License-Identifier: Apache-2.0
"""Joint H3 carried DMD2 for the compact recovery wrapper.

Math follows the FastVideo dense V12 reference at67f019ee8fd6: continuous
inverse-2.4 score warp on0.999, modality shifts12/3, direct-x0 critic loss,
FP64 scheduler arithmetic, four critic phases per generator phase. This
adapter uses matched SDPA, compact42 student and full Base critic, and one carry slot;
it is an adaptation, not a bitwise reproduction of the32-GPU release run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method
from fastvideo.train.methods.knowledge_distillation.minimax_h3_recovery import _local_parameter_tensor
from fastvideo.train.models.minimax_h3 import MiniMaxH3Model
from fastvideo.train.utils.h3_prompt_coverage import record_prompt_use


def warp_sigma(base: torch.Tensor, shift: float, maximum: float) -> torch.Tensor:
    return base.double() * shift * maximum / (base.double() * (shift - 1) + maximum)


def clean_estimate(state: torch.Tensor, flow: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return (state.double() - sigma.double() * flow.double()).to(state.dtype)


def mix_state(clean: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return ((1 - sigma.double()) * clean.double() + sigma.double() * noise.double()).to(clean.dtype)


def advance_ode(state: torch.Tensor, clean: torch.Tensor, sigma: torch.Tensor,
                next_sigma: torch.Tensor) -> torch.Tensor:
    eps = (state.double() - (1 - sigma.double()) * clean.double()) / sigma.double()
    return ((1 - next_sigma.double()) * clean.double() + next_sigma.double() * eps).to(state.dtype)


def distribution_loss(generated: torch.Tensor,
                      fake: torch.Tensor,
                      real: torch.Tensor,
                      denom_floor_ratio: float = 0.05,
                      grad_cap: float = 100.0) -> torch.Tensor:
    """DMD direction with a scale-relative denominator floor.

    The audit (2026-09-08) traced jobs 6951/6953 to the absolute ``+ 1e-6``
    floor: when the critic is scored on near-clean states at warped low
    sigmas, ``mean|generated - real|`` can fall below it and the direction
    explodes into nonfinite critic gradients (first reported on
    ``context_embedder.weight``). Floor at a fraction of the generated scale
    and cap the direction instead.
    """
    with torch.no_grad():
        scale = generated.float().abs().mean().clamp_min(1e-4)
        denom = (generated.float() - real.float()).abs().mean().clamp_min(float(denom_floor_ratio) * scale)
        grad = ((fake.float() - real.float()) / denom).clamp(-float(grad_cap), float(grad_cap))
        # Fail on broken numerics instead of hiding NaNs with nan_to_num.
        if not torch.isfinite(grad).all():
            raise RuntimeError("Nonfinite joint DMD direction")
    return 0.5 * F.mse_loss(generated.float(), (generated.float() - grad).detach())


class MiniMaxH3JointDMD2Method(DMD2Method):

    def __init__(self, *, cfg: Any, role_models: dict[str, Any]) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        if any(not isinstance(m, MiniMaxH3Model) for m in role_models.values()):
            raise TypeError("Joint DMD requires the audited MiniMaxH3Model wrapper")
        if any(m.attention_backend_name != "TORCH_SDPA" for m in role_models.values()):
            raise ValueError("This compact DMD experiment requires matched SDPA roles")
        if self.method_config.get("dmd_denoising_steps") != [999, 749, 500, 250]:
            raise ValueError("This experiment requires the dense release four-call grid")
        if float(self.method_config.get("real_score_guidance_scale", 1)) != 1:
            raise ValueError("Joint DMD pilot requires guidance1")
        if int(self.training_config.loop.gradient_accumulation_steps) != 1:
            raise ValueError("This pilot supports one carry slot and accumulation1")
        if int(self.method_config.get("generator_update_interval", 5)) != 5:
            raise ValueError("Require four critic phases per generator phase")
        self._carry: Any = None
        self._verified_roles: set[str] = set()
        self._denom_floor_ratio = float(self.method_config.get("dmd_denom_floor_ratio", 0.05))
        self._sanitize_nonfinite_grads = bool(self.method_config.get("sanitize_nonfinite_grads", False))

    def _parse_score_timestep_bounds(self) -> tuple[int, int]:
        return (1, 999)

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        return [self._student_optimizer if self._should_update_student(iteration) else self._critic_optimizer]

    def get_lr_schedulers(self, iteration: int) -> list[Any]:
        return [self._student_lr_scheduler if self._should_update_student(iteration) else self._critic_lr_scheduler]

    def get_grad_clip_targets(self, iteration: int) -> dict[str, torch.nn.Module]:
        name = "student" if self._should_update_student(iteration) else "critic"
        return {name: self._role_models[name].transformer}

    def on_checkpoint_loaded(self, iteration: int) -> None:
        self._carry = None  # Release carry is transient; do not promise exact replay after resume.

    def _sigmas(self, base: torch.Tensor, *, score: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        maximum = .999 if score else 1.0
        return warp_sigma(base, 12., maximum), warp_sigma(base, 3., maximum)

    def _observe(self, label: str, tensors: Any) -> None:
        if not self.method_config.get("diagnostic_boundaries", False):
            return
        stats = []
        for x in tensors:
            value = x.detach().float()
            finite = torch.isfinite(value)
            stats.append({
                "shape": list(value.shape),
                "dtype": str(x.dtype),
                "finite": bool(finite.all()),
                "absmax": float(value.abs().max()),
                "rms": float(value.double().square().mean().sqrt())
            })
        rank = dist.get_rank() if dist.is_initialized() else 0
        root = Path(self.training_config.checkpoint.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        with (root / f"dmd_boundaries_rank{rank}.jsonl").open("a") as stream:
            stream.write(
                json.dumps({
                    "iteration": getattr(self, "_diagnostic_iteration", 0),
                    "boundary": label,
                    "tensors": stats
                }) + "\n")
        if not all(t["finite"] for t in stats):
            raise RuntimeError(f"First nonfinite DMD boundary: {label}")

    def _predict(self, model: Any, states: Any, sigmas: Any, tb: Any) -> Any:
        flows = model.predict_joint_noise(*states,
                                          1 - sigmas[0],
                                          1 - sigmas[1],
                                          tb,
                                          conditional=True,
                                          attn_kind="dense")
        role = next(name for name, candidate in self._role_models.items() if candidate is model)
        self._observe(f"{role}_flow", flows)
        estimates = tuple(clean_estimate(x, flow, s) for x, flow, s in zip(states, flows, sigmas, strict=True))
        self._observe(f"{role}_clean", estimates)
        return estimates

    def single_train_step(self, batch: dict[str, Any], iteration: int) -> Any:
        if not batch.get("prompt_only", False):
            raise ValueError("This DMD pilot requires audited prompt-only batches")
        self._diagnostic_iteration = iteration
        fresh = self._carry is None
        raw = batch if fresh else self._carry["raw"]
        tb = self.student.prepare_batch(raw, generator=self.cuda_generator, latents_source="zeros")
        device = self.student.device
        if fresh:
            rung = 0
            states = (tb.noise.permute(0, 2, 1, 3, 4), tb.audio_noise)
        else:
            rung, states = self._carry["rung"], self._carry["states"]
        grid = torch.tensor([.999, .749, .500, .250, 0.], device=device, dtype=torch.float64)
        sigmas = self._sigmas(grid[rung:rung + 1])
        self._observe("carried_state", states)
        update_student = self._should_update_student(iteration)
        with torch.set_grad_enabled(update_student):
            generated = self._predict(self.student, states, sigmas, tb)
        # Score clock: U~Uniform(.001,.999), inverse2.4 then shifts12/3.
        u = torch.rand((1, ), device=device, dtype=torch.float64, generator=self.cuda_generator) * .998 + .001
        base_score = warp_sigma(u, 1 / 2.4, .999)
        score_sigmas = self._sigmas(base_score, score=True)
        noises = tuple(
            torch.randn(x.shape, device=device, dtype=x.dtype, generator=self.cuda_generator) for x in generated)
        with torch.no_grad():
            noised = tuple(mix_state(x.detach(), n, s) for x, n, s in zip(generated, noises, score_sigmas, strict=True))
        self._observe("score_noised", noised)
        if update_student:
            with torch.no_grad():
                fake = self._predict(self.critic, noised, score_sigmas, tb)
                real = self._predict(self.teacher, noised, score_sigmas, tb)
            video_loss, audio_loss = (distribution_loss(g, f, r, self._denom_floor_ratio)
                                      for g, f, r in zip(generated, fake, real, strict=True))
            context = (1 - sigmas[0], tb.attn_metadata)
        else:
            predicted = self._predict(self.critic, noised, score_sigmas, tb)
            video_loss, audio_loss = (F.mse_loss(p.float(),
                                                 g.detach().float()) for p, g in zip(predicted, generated, strict=True))
            context = (1 - score_sigmas[0], tb.attn_metadata)
        total = video_loss + audio_loss
        self._observe("loss", (video_loss, audio_loss))
        if not torch.isfinite(total):
            raise RuntimeError("Nonfinite joint DMD loss")
        with torch.no_grad():
            if rung == 3:
                self._carry = None
            else:
                next_sigmas = self._sigmas(grid[rung + 1:rung + 2])
                next_states = tuple(
                    advance_ode(x, g.detach(), s, ns)
                    for x, g, s, ns in zip(states, generated, sigmas, next_sigmas, strict=True))
                self._carry = {"raw": raw, "rung": rung + 1, "states": next_states}
        coverage = record_prompt_use(self, raw, iteration)
        phase = "generator" if update_student else "critic"
        return {
            "total_loss": total,
            f"{phase}_video": video_loss,
            f"{phase}_audio": audio_loss
        }, {
            "role": "student" if update_student else "critic",
            "context": context
        }, {
            "generator_phase": int(update_student),
            "rollout_rung": rung,
            "new_prompt_adopted": int(fresh),
            **coverage
        }

    def backward(self,
                 loss_map: dict[str, torch.Tensor],
                 outputs: dict[str, Any],
                 *,
                 grad_accum_rounds: int = 1) -> None:
        model = self._role_models[outputs["role"]]
        model.backward(loss_map["total_loss"], outputs["context"], grad_accum_rounds=grad_accum_rounds)
        if self._sanitize_nonfinite_grads:
            sanitized = 0
            for p in model.transformer.parameters():
                if p.grad is not None and not torch.isfinite(_local_parameter_tensor(p.grad)).all():
                    p.grad.detach().nan_to_num_(0.0, posinf=0.0, neginf=0.0)
                    sanitized += 1
            if sanitized:
                rank = dist.get_rank() if dist.is_initialized() else 0
                path = Path(self.training_config.checkpoint.output_dir) / f"dmd_sanitized_grad_rank{rank}.jsonl"
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"iteration": self._diagnostic_iteration, "role": outputs["role"],
                                             "sanitized": sanitized}) + "\n")
        if self.method_config.get("diagnostic_boundaries", False):
            bad = [
                name for name, p in model.transformer.named_parameters()
                if p.grad is not None and not torch.isfinite(_local_parameter_tensor(p.grad)).all()
            ]
            if bad:
                rank = dist.get_rank() if dist.is_initialized() else 0
                path = Path(self.training_config.checkpoint.output_dir) / f"dmd_bad_gradient_rank{rank}.json"
                path.write_text(
                    json.dumps({
                        "iteration": self._diagnostic_iteration,
                        "role": outputs["role"],
                        "parameters": bad
                    }))
                raise RuntimeError(f"Nonfinite {outputs['role']} gradients before clipping/Adam: {bad[:3]}")

    def optimizers_schedulers_step(self, iteration: int) -> None:
        name = "student" if self._should_update_student(iteration) else "critic"
        model = self._role_models[name]
        optimizer = self.get_optimizers(iteration)[0]
        first = name not in self._verified_roles
        probes: list[tuple[torch.Tensor, torch.Tensor]] = []
        if first:
            for p in model.transformer.parameters():
                if p.requires_grad:
                    local = _local_parameter_tensor(p).detach().reshape(-1)
                    if local.dtype != torch.float32:
                        raise RuntimeError("DMD master weights must be FP32")
                    if local.numel() and len(probes) < 16:
                        probes.append((local, local[:4096].clone()))
        if self.method_config.get("diagnostic_boundaries", False):
            gradients = [
                _local_parameter_tensor(p.grad).detach() for p in model.transformer.parameters() if p.grad is not None
            ]
            # Inspect reduced/clipped gradients before Adam can contaminate weights.
            self._observe(f"{name}_gradients", gradients)
        super().optimizers_schedulers_step(iteration)
        if first:
            changed = sum(int(torch.count_nonzero(x[:b.numel()] != b)) for x, b in probes)
            moments = [v for st in optimizer.state.values() for k, v in st.items() if k in {"exp_avg", "exp_avg_sq"}]
            if (not changed or not moments or any(v.dtype != torch.float32 for v in moments)
                    or any(not torch.isfinite(x[:b.numel()]).all() for x, b in probes)):
                raise RuntimeError(f"No finite FP32 Adam update for {name}")
            rank = dist.get_rank() if dist.is_initialized() else 0
            root = Path(self.training_config.checkpoint.output_dir)
            root.mkdir(parents=True, exist_ok=True)
            (root / f"dmd_update_{name}_rank{rank}.json").write_text(
                json.dumps({
                    "role": name,
                    "iteration": iteration,
                    "changed_probe_elements": changed,
                    "passed": True
                }))
            self._verified_roles.add(name)
