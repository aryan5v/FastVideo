# World Model Program — Source of Truth

Status: living document. Owner: willlin. Started 2026-08-17.
Compute envelope: 32–64 GB200 (8–16 trays, NVL72 rack(s), autoscaled Slinky
Slurm + k8s/Kueue fallback).
Companion docs: [`HANDOFF-h3-dmd2-vsa.md`](HANDOFF-h3-dmd2-vsa.md) (distillation
line ops), [`h3_dmd.md`](examples/train/configs/distribution_matching/minimax_h3/h3_dmd.md)
(DMD2 recipe + open issues),
[`.agents/lessons/2026-08-14_h3-dmd2-training-preflight-rules.md`](.agents/lessons/2026-08-14_h3-dmd2-training-preflight-rules.md)
(training preflight rules).
Update discipline: decisions and results land here (append to changelog);
recipe mechanics stay in the runbooks.

---

## 1. North star and thesis

**Goal: a physically grounded world model that reasons through its state and
relies on grounded visual characteristics to do physical reasoning.**
Visual fidelity is subordinate to consistency and physical accuracy.

**Thesis: reasoning happens *by simulating* in the visual-geometric substrate,
not by verbalizing about it.** The model answers "what happens if…" by
imagining it under physics-honest dynamics; a coupled reasoner decides *what*
to simulate, reads outcomes, and selects among them.

**State has a precise referent**: the jointly-denoised multimodal block
(RGB + audio + geometry + actions) plus the KV history of past blocks.
Reasoning through state = roll it, branch it, score it, articulate it.

External validation of the thesis (2025–26): fine-tuned video models beat
strong VLMs on long-horizon spatial planning (VR-Bench); video diffusion
models commit to motion plans in the first few denoising steps (ChEaP);
RL-with-verifiable-rewards works directly on video diffusion (VideoRLVR).
See §9 literature.

## 2. Model-class framing

| | VLM | Video diffusion (Wan/LTX/H3) | World/action model | This program |
|---|---|---|---|---|
| Models | p(text \| pixels) | p(video \| text) | p(next state \| state, action) | p(AV+geometry \| text, actions, history) |
| Direction | pixels→semantics (lossy compression) | semantics→pixels (expansion) | state×action→state | all three composed |
| Time | encodes clips | bidirectional fixed clip | causal, interventional | block-causal after conversion |
| Knows | *about* the world (verbal) | *how it looks/moves* (implicit) | *what happens if* (causal) | all three, queryable |

VLMs reason but can't simulate; video priors simulate but can't be queried or
steered; a world model = simulator + causal clock + action interface +
reasoner at the boundary. One probe that separates them: "will the tower
fall?" — the VLM answers from verbal priors; the world model watches itself
knock it over, and the geometry stream lets us check the fall obeyed physics.

## 3. Architecture: H3 as the substrate

**Why H3 (unified packed stream) over dual-stream (Wan/LTX)**: 50
modality-agnostic blocks; modality identity enters only at five touchpoints —
(1) I/O projections (`proj_in/out`, `audio_proj_in/out`; the fp32 FSDP
compute-group list is the adapter inventory), (2) rope choice, (3)
per-modality timestep shift (video 12 / audio 3 from one shared base-t),
(4) attention mask policy (VSA exempt/compete), (5) packing layout +
`modality_slices()`. Adding a modality = one projection pair + rope + shift +
slice. Audio (270:1 element imbalance vs video) is the existence proof.

**Unified stream ⇒ one joint score ⇒ conditioning by masking**: clamp any
subset clean, denoise the rest — p(depth|video), p(audio|video),
p(future|past, actions) from one model. Per-modality σ schedules are a
*control surface* (train with per-modality/per-token noise ⇒ staggered
inference schedules: geometry-as-anchor; plan-then-render).

**Block-causal packing (target layout)**: time-interleaved blocks
`[block_i: video+audio+depth+action]`, block-lower-triangular attention,
KV cache over past blocks. Bidirectional attention is order-invariant, so the
teacher trains on the interleaved layout at zero cost — full layout
compatibility with the causal student.

**Numbers (H3 real shapes)**
- 768×1344×124f window: 37 latent frames × 1008 tokens = 37,296 video tokens,
  plus ~414 audio tokens ≈ 38k total (~5.2 s).
- Block = 3 latent frames ≈ 3.1k tokens (768p) / 1.2k (480p); ~12 blocks per
  window at 768p, **~32 blocks ≈ 16 s at 480p** — resolution is officially
  subordinated to horizon.
- KV cache: ~1.4 MB/token if full MHA (50 layers, 56×128 attn) → ~54 GB per
  768p context, ~17 GB at 480p. **OPEN CHECK: GQA/MQA in H3 attention** —
  decides search width (N≈4 vs N≈32 particles/tray).

## 4. Empirical findings so far (distillation line, v1–v6)

The DMD2 few-step line is the enabling layer (cheap steps ⇒ affordable
search), and its findings are program-level lessons:

1. **fp32 master weights are mandatory** — bf16 masters bit-froze 205/211
   norm gains for 1000 steps; fixed + hard-checked in both stacks; verified
   210/210 moving at v5-ckpt500. (Lessons file, rules 1–4.)
2. **Every timestep-space constant must be recomputed through the shift map**
   — Wan's ladder under shift 12 made the middle step a no-op; v5/v6 ladder
   `[1000,667,333]`; supervision *density* must be checked in σ-space.
3. **Per-modality supervision asymmetries are the recurring trap** — packed
   mean muted audio (fix: per-modality losses); global x0-space critic
   silenced audio's low-σ axis via σ_a² weighting (fix: per-modality loss
   space `{video: x0, audio: velocity}`). The causal student's critic
   inherits the same trap *inside every block*.
4. **Deterministic Euler carry beats stochastic x0-renoise on immature
   students** (A/B at v5-ckpt500 and v2-ckpt1000: stochastic collapses to
   noise after ~1 s; interframe-corr 0.29 vs 0.77, gradient energy 6×).
   Teacher control shows x0 overshoot at σ≈1 (std 2.33) is the root; revisit
   stochastic on mature students.
5. **DMD scalars are game state, not quality** (generator loss band 0.2–1.0;
   critic loss = tracking meter; grad-norm-vs-clip = delivered LR).
6. Economics: teacher sample ≈ 4–5 GPU-min; 3-step student ≈ 15 s on 4 GPUs
   ⇒ **best-of-16 student search ≈ one teacher sample**. fp32 DCP
   checkpoints: 741 GiB measured (A/V trio).

Current artifact: v6 (`dmd2_sp1_vidprom_v6_fp32_compute`) parked at
checkpoint-2700/4000 with the audio fix live since 2500; ~1,300 steps remain.

## 5. The program (phases and gates)

**Sequencing rule: distribution content goes into the bidirectional teacher;
control interfaces enter at causalization.** (Teacher can't distill what it
doesn't know; causalization is distillation; cross-modal consistency is
learned most easily bidirectionally; one variable at a time.)

### Phase 0 — Foundations (login node + ≤1 tray + v6 tail)
- Finish v6 (~1 day) → audio-fix verdict + final sampler A/B.
- **Probe suite v1** (~20 physical-reasoning tasks): permanence, collision,
  support, trajectories, containment, counterfactual pairs, ego-motion,
  shadows/contact, **A/V causality** (impact-sound timing — unimodal-impossible,
  ours uniquely). Scoring: VLM 2AFC + reprojection + A/V sync offset.
  **Kubric generates the gold subset** (exact physics, programmable
  counterfactual pairs). Add the **2D-puzzle track** (maze/Sokoban videos,
  trajectory-extraction scoring; VR-Bench-compatible).
- Data acquisition: MOVi, TartanAir2 slice, ScanNet++/ARKitScenes,
  Stereo4D pipeline assessment; licenses.
- Code checks: GQA/KV layout; interleaved block-packer spec.

### Phase 1 — Geometry go/no-go (the 32-GPU week)
Twin-arm controlled experiment, 16 GPUs/arm, identical except depth:
- Arm A: A/V SFT side-tune (control). Arm B: + co-denoised depth
  (`depth_proj_in/out` cloned from video; one new slice).
- Data mix: ~20k teacher clips (in-distribution, audio, pseudo-depth) +
  Stereo4D (real dynamic) + TartanAir2/Kubric (perfect GT) + ScanNet++
  (indoor real). 480p.
- **Gate**: Arm B > Arm A on RGB-only probe scores (depth head discarded)
  with meaningful effect size. Fail ⇒ geometry demotes to verifier-only.
- Byproducts survive either outcome: teacher-clip pool (ODE-init + GAN branch
  - perception adapter), labeled geometry corpus, third-modality adapter.

### Phase 2 — Distill + measure
- Tri-modal DMD2 (per-modality machinery as-is; add x0-scale and per-σ-bin
  telemetry).
- Test-time-search baselines: best-of-N with verifier stack, x0-preview tree
  pruning (evidence-backed by plan-commitment), student-proposes /
  teacher-refines cascade → first N-vs-quality curves.
- Reasoner L0/L1: VLM judge in harness; physics-explicit conditioning probe
  (reasoner-rewritten prompts through the existing frozen Qwen3-VL encoder —
  H3 already reads VLM hidden states).

### Phase 3 — Causalization
- Interleaved packer into teacher training (free for bidirectional).
- ODE-init causal student from teacher-clip trajectories
  (`ode_causal_pipeline` pattern; block = 3 latent frames) → **self-forcing
  DMD** ported to `fastvideo/train/` (in-repo templates:
  `self_forcing_distillation_pipeline`, `matrixgame2_*`; carry the audio
  loss-space fix into the per-block critic).
- Action modality: camera trajectories first (Stereo4D/RealEstate10K poses),
  timed text events second — converges with the reasoner interface.
  Embodied controls later (Matrix-Game-2 pattern: actionless teacher as
  prior, actions learned at causal stage).
- 480p default; context-noise/rolling-KV for horizon.

### Phase 4 — The reasoning world model
- Reasoner at block boundaries (coupling ladder: L0 verifier/planner outside
  the loop → L1 reasoned conditioning through the existing encoder → **L2
  frozen-giant adapters** (control in: near-native Qwen-family; perception
  out: decode→VLM first, latent adapter later) → L3 unified single model =
  explicitly out of scope (pretraining-scale, unnecessary until L2
  saturates; same verdict for text diffusion as a modality).
- Verifier-guided particle search over futures at block boundaries (width set
  by GQA answer; per-block pruning over wide beams).
- **RLVR bridge**: SDE-GRPO with verifiable rewards (puzzle + Kubric
  verifiers) as post-training on the causal student.
- Kubric-calibration finetune; report on the probe benchmark
  (simulate-then-read accuracy, not FVD) + VR-Bench (first A/V entrant) +
  functional evals (WorldSimBench-style inverse-dynamics utility).

## 6. Test-time scaling (bidirectional works today)

Axes, by practicality: (A) restart/churn sampling; (B) best-of-N + verifiers
— internal: A/V coherence, teacher-as-energy (renoise + teacher residual),
later geometry reprojection; external: VLM judge (hackable at large N — keep
N moderate until grounded verifiers exist); (C) path search: branch at high σ,
score x0 previews, prune — plans are legible early (plan-commitment);
principled form = Feynman-Kac/twisted SMC; (D) token-selective renoising
repair (per-token σ mask; modality-conditional resampling: clamp video,
resample audio = p(audio|video) natively); (E) window composition (keyframe →
infill; overlapping windows) for horizon without causalization;
(F) economics: distill for cheap steps, spend savings on search.
Samples-vs-answers: consistency voting across N rollouts (diffusion analog of
self-consistency CoT) + verifier-weighted selection.

## 7. Data plan

| Tier | Source | Modalities | Role |
|---|---|---|---|
| Real dynamic video | **Stereo4D** (100k+ VR180 seqs, CVPR'25; StereoWorld-11M to assess) | metric-ish depth, poses, 3D tracks | real-geometry training slice; breaks teacher-circularity |
| Real indoor scans | ScanNet++ (Faro+iPhone), ARKitScenes (450k frames) | mm depth (+normals from scans) | gold static eval + train slice |
| Real driving | KITTI-360/Waymo/nuScenes | LiDAR depth video | outdoor dynamics slice |
| Real normals | DIODE (stills) | dense depth+normals | normals eval reference |
| Synthetic perfect | **Kubric/MOVi** (generator: depth, normals, flow, seg, **collision events, physics params**) | everything | probe gold + calibration finetune + counterfactual pairs |
| Synthetic video | TartanAir2, Spring, Dynamic Replica, PointOdyssey, Hypersim | depth/normals/flow | diverse GT train/eval |
| Pseudo-labels | Video-Depth-Anything / DepthCrafter / GeometryCrafter | depth on any corpus | teacher-clip labeling |
| Prompts | VidProM 248k (encoded, verified) | text conditioning | distillation + clip generation |
| Puzzles | VR-Bench + procedural maze/Sokoban renders | exact verifiers | reasoning track |

Teacher-clip pool (~20k × 50-step rollouts ≈ 1.9 days on 32 GPUs) serves four
consumers: geometry SFT, ODE-init, GAN branch real-side, perception-adapter
pairs.

## 8. Risks and kill criteria

| Risk | Signal | Response |
|---|---|---|
| Geometry doesn't transfer to RGB consistency | Phase-1 gate fails | Demote to verifier-only (estimator-based reprojection); reroute via actions/causality |
| Plausible ≠ correct dynamics | probe suite vs Kubric gold divergence | calibration finetune on sim gold; weight grounded verifiers |
| Causal drift beyond ~10 blocks | rollout probes | context-noise/rolling-anchor ablations before scale |
| Verifier hacking under search | reward/probe divergence at large N | cap N; Kubric-calibrated verifier weighting |
| A/V continuity across block boundaries | audio artifacts at block seams | first novel-research territory; per-block per-modality schedules |
| Per-modality supervision asymmetry (recurring) | one modality's game goes slack | per-modality loss spaces/shifts/weights (machinery exists) |
| Compute contention / maintenance | Slurm drained | k8s path validated: `vllm` namespace on OKE, create-pods yes, Kueue installed |

## 9. Related work and literature

**World models & interactive generation**: DIAMOND (diffusion WM, Atari-100k
HNS 1.46, CS:GO engine) · GameNGen (DOOM) · Oasis (Minecraft) · Genie 1–3 ·
Muse/WHAM · Matrix-Game-2 (in-repo pipelines) · UniSim · Diffusion Forcing
(per-token noise) · CausVid / Self-Forcing (causal students; in-repo) ·
"A Definition and Roadmap for World Models" (position survey).

**Video/diffusion models as reasoners (2D puzzles)**: VR-Bench / Reasoning
via Video (5 puzzle types; video models > VLMs on spatial planning) ·
ChEaP / plan-commitment (plans fix in early denoising; 7%→67% long mazes) ·
HDR (hierarchical denoising; maze/Hanoi/Sokoban/sliding) · VideoRLVR &
Wan-R1 (SDE-GRPO, verifiable rewards on video diffusion) · DiffThinker ·
discrete-diffusion planning (diffusion-of-thought; beyond-AR Sokoban) ·
Veo-class zero-shot maze results.

**VLM puzzle benchmarks**: VGRP-Bench (20 grid puzzles) · iVISPAR (sliding
tile, interactive) · lmgame-Bench (Sokoban/Tetris harness) · point-and-click
study · VAGEN (RL-trained internal world model in VLM agents — the mirror of
our coupling).

**World-model benchmarks**: EWMBench (perceptual) · WorldSimBench
(inverse-dynamics functional utility) · iWorld-Bench / WBench / WorldArena
(multi-turn interactive control) · WorldPrediction / WorldReasonBench
(persistent state, procedural planning).

**Methods backbone**: DMD (Yin'23) / DMD2 (Yin'24; TTUR, backward simulation,
GAN branch) · rectified flow / SD3 timestep shift · consistency models ·
EDM · restart sampling · FKC/twisted SMC steering · Transfusion (AR text +
diffusion in one transformer; the L3 reference) · LLaDA/MDLM (dLLMs) ·
Marigold/GeoWizard/DepthCrafter (diffusion geometry) · Video-Depth-Anything.

**Datasets**: Stereo4D · StereoWorld-11M · ScanNet++ · ARKitScenes · DIODE ·
Kubric/MOVi · TartanAir(2) · Spring · Dynamic Replica · PointOdyssey ·
Hypersim · KITTI-360/Waymo/nuScenes · RealEstate10K/DL3DV (poses) ·
VidProM (prompts).

Links (key): [VR-Bench](https://github.com/FoundationAgents/VR-Bench) ·
[Reasoning via Video](https://arxiv.org/html/2511.15065v1) ·
[ChEaP](https://arxiv.org/pdf/2603.30043) ·
[VideoRLVR](https://arxiv.org/html/2605.15458v1) ·
[Wan-R1](https://arxiv.org/html/2603.27866) ·
[HDR](https://arxiv.org/html/2607.15278) ·
[VGRP-Bench](https://arxiv.org/abs/2503.23064) ·
[lmgame-Bench](https://www.emergentmind.com/papers/2505.15146) ·
[VAGEN](https://vagen-ai.github.io/vagen_paper.pdf) ·
[DIAMOND](https://arxiv.org/abs/2405.12399) ·
[EWMBench](https://arxiv.org/pdf/2505.09694) ·
[WBench](https://arxiv.org/html/2605.25874v1) ·
[iWorld-Bench](https://arxiv.org/html/2605.03941v2) ·
[WorldArena 2.0](https://arxiv.org/html/2605.17912v1) ·
[World-model roadmap](https://arxiv.org/pdf/2607.06401) ·
[Stereo4D](https://stereo4d.github.io/) ·
[Kubric](https://github.com/google-research/kubric) ·
[MOVi](https://github.com/google-research/kubric/blob/main/challenges/movi/README.md) ·
[ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation) ·
[Hypersim](https://arxiv.org/pdf/2011.02523) ·
[PointOdyssey](https://openaccess.thecvf.com/content/ICCV2023/papers/Zheng_PointOdyssey_A_Large-Scale_Synthetic_Dataset_for_Long-Term_Point_Tracking_ICCV_2023_paper.pdf) ·
[TartanAir](https://www.emergentmind.com/topics/tartanair-dataset) ·
[StereoWorld-11M](https://www.emergentmind.com/topics/stereoworld-11m-dataset)

## 10. Open questions

1. Does H3 use GQA/MQA? (KV budget ⇒ search width.) — code check pending.
2. Audio (and geometry) continuity across causal block boundaries under KV
   cache — beyond published work; likely first novel result.
3. Stochastic-renoise viability on mature students (re-A/B at v6 end).
4. Optimal per-modality σ schedules for staggered plan-then-render inference.
5. State persistence beyond the window: rolling KV vs explicit scene memory
   (accumulated depth+pose is a map — the geometry stream may be the memory).
6. How much physics does the H3 prior already contain? (Probe the teacher
   before/alongside Phase 1 — calibrates every downstream claim.)

## 11. Changelog

- 2026-08-17: v1. Synthesis of the distillation-line findings (v1–v6), the
  modality-extension and reasoner-coupling brainstorm, block-causal
  commitment, dataset survey (real depth/normals), and the 2D-puzzle
  reasoning survey. Phases 0–4 defined with gates.
