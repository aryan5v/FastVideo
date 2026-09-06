# FastH3 consumer model: decision and next experiment

Decision date: September 6, 2026. This supersedes the target and stage ordering
in the earlier pruning/release plans. The latest user direction prioritizes
consumer memory, speed and joint audiovisual quality, with freedom to choose
the method. Neither 14B nor pruning is an unconditional release requirement
under that direction. A week or longer is available when results justify it.

## Decision

Make released FastH3 four-call to two-call distillation the next main experiment.
Keep the healthy backbone and its trained attention backend for this test.
Use quantization to address weight memory after the high-precision two-call
model passes quality checks. Preserve approximately 20B/four-call structural
compression as the alternative if the larger model misses memory or latency
targets. Do not automatically extend the current 24-block recovery pilots.

This is a choice of the next experiment, not evidence that two-call H3 works.
It isolates call reduction from capacity reduction. An already distilled
teacher can still lose motion, diversity or speech when distilled again.

## Evidence checked on the cluster

The researcher account is `vlm-wlsaidhi`. Read-only inspection covered the
versioned recipes, PDD implementation/port record, logs, and resolved configs
inside the latest completed training checkpoints. No other user's run or file
was changed.

| Run | GPUs | Recipe and saved checkpoint observed |
|---|---:|---|
| 6440, V23 | 64 | Grid32, maximum trained block width 8, validation at 4/8 calls; complete training checkpoint 1400, inference-directory entry 1450 |
| 6432, V24 | 32 | Grid32, maximum trained block width 6, validation at 6/8 calls; complete training checkpoint 700 |

Both are active, full-backbone PDD runs initialized from Base MiniMax-H3, with
a frozen dense Base H3 teacher and a 90%-sparse tile-64 VSA student. They use
prompt-only carried trajectories, FP32 trainable parameters/decoding, BF16
model boundaries, AdamW at 1e-5, global batch 64 and video/audio shifts 12/3.
The configured horizon is 4,000 updates. Their three named prompt sources
contain roughly 119K records before any overlap audit; this is not a verified
unique-caption count.

Two calls on Grid32 require two width-16 blocks. That exceeds both recipes'
trained widths. The PDD method explicitly rejects validation partitions wider
than `block_size_max`; the standalone pipeline accepting a partition does not
make it trained. Existing two-call inference would be an extrapolation control,
not a ready two-call candidate. Do not bypass this training validation guard.

The latest V23 inference checkpoint directory was not readable by our account.
Training metadata and media filenames were readable. Checkpoint/media existence
does not establish quality; neither run's audiovisual quality has been reviewed
in this investigation. Reusing its inference weights remains dependent on a
readable published/exported artifact. Do not change permissions or use another
identity to obtain it.

Read-only provenance:

- Research root: `/mnt/lustre/vlm-wlsaidhi/fastvideo`.
- V23 checkout: `FastVideo-h3-v23`, HEAD `bf16199660d9fdf7319dc2b8466b41644e5480bc`.
- V24 checkout: `FastVideo-h3-v24`, HEAD `2c9ccc5e8982c48b4c6ab1dcae2ce96be2365aac`.
- Recipes: `examples/train/configs/trajectory_matching/minimax_h3/`, files
  `pdd_sp4_fsdp64_v23_datafree_prompt_only_vsa90_grid32_4step_shift12_3.yaml`
  and `pdd_sp4_fsdp32_v24_datafree_prompt_only_vsa90_grid32_6step_shift12_3.yaml`.
- Port contract: `examples/train/configs/distribution_matching/minimax_h3/h3_pdd.md`;
  partition guard: `fastvideo/train/methods/trajectory_matching/pdd.py`.
- [V23 W&B](https://wandb.ai/wlsaidhi/h3-dmd2-vsa/runs/meyg9crv) and
  [V24 W&B](https://wandb.ai/wlsaidhi/h3-dmd2-vsa/runs/ynaikgpq).

These HEADs describe the inspected checkouts, not proof that every live file
was clean at launch. Saved checkpoint configs independently confirm the stated
model lineage, method and modality shifts.

## What the block-skipping experiment currently says

Both arms use the same fixed final 24-block mask and two held-out canaries.
The figures below are relative reductions in normalized closed-loop latent
endpoint error from the untrained masked model. They are not percentages of
visual quality recovered, accuracy, or distance to release readiness.

| Policy | Update | Video error reduction | Audio error reduction |
|---|---:|---:|---:|
| Hard removal | 175 | 1.60% | 10.95% |
| Gradual block skipping | 175 | 0.84% | 10.09% |
| Hard removal | 200, complete | 2.01% | 11.90% |

At the matched 175-update point hard removal leads both canary metrics.
Gradual removal reached its final mask at update 101, so this comparison also
reflects different exposure to the final architecture. It does not prove that
all gradual or learned pruning methods are inferior. Hard job 6533 completed;
annealed job 6535 was still running when these figures were recorded.

Finish the existing bounded 200-update run, preserve its checkpoints, and
compare the physically extracted models after prediction parity. Recovered
decoded media has not yet been evaluated. The small video gains do not support
scaling either policy on their own. The prepared export/parity procedure remains
necessary; numerical learning success is not a media quality gate.

## Original FastH3 and the two distinct distillation options

The released four-call FastH3 used DMD2: a frozen Base H3 teacher and learned
critic provide a distribution-matching signal. Prompt-only training uses
student-generated trajectories. The VSA student sparsifies eligible video
attention while retaining dense text/audio attention. This is distinct from
the newer PDD runs. See the [original release recipe](https://haoailab.com/blogs/fasth3-preview/).

[Parallel Decoding Distillation](https://research.nvidia.com/labs/genair/pdd/)
adds interval-specific output projections and fuses them for each inference
block. Its public results cover 4–8 calls on other image/video/audio models;
they do not establish two-call H3 quality.

The proposed direct four-to-two experiment instead follows the idea of
[progressive distillation](https://arxiv.org/abs/2202.00512): teach one student
update to reproduce two teacher updates. This is an H3 adaptation to validate,
not a reproduction of that paper or a completed PDD implementation.

## Concrete execution sequence

1. Close the current pruning pilot and retain decoded comparisons as evidence.
   Build a fixed release-evaluation set of 32 held-out prompts and two seeds,
   covering appearance, fast motion, camera movement, exact speech, multiple
   speakers, sound effects, music and synchronization. Use the same prompts,
   seeds, dimensions and full VAE for every candidate. Review eight diverse
   prompts first as a collapse screen; advance promising models to all 64 clips.
   Existing two-canary metrics remain debugging signals only.
2. Implement one branch-matched four-to-two bootstrap from the released healthy
   V1 model. Start with the dense branch, for which the current audited recovery
   path and portable consumer reference already exist. Keep every block and
   both modality interfaces. CUDA VSA follows as a separate branch-matched
   experiment after the mechanism passes; never infer VSA results from dense
   training or run VSA-trained weights through dense attention.
3. Preserve the released teacher's exact five-node schedule, guidance 1 and
   video/audio shifts. Student boundaries are nodes 0, 2 and 4. From the same
   joint starting state, run teacher updates 0→1→2 and teach the first student
   update to reach that endpoint; repeat 2→3→4 for the second interval.
   Normalize joint modality losses separately. Derive each target using that
   modality's actual sigma change, not a shared unshifted time difference.
   Do not query the four-call teacher as an arbitrary continuous Base-H3 score.
4. Validate teacher-state pair targets first, then expose training to the
   student's own intermediate states with teacher correction from those states.
   Keep the schedule, state precision and endpoints identical in training and
   sampling. FP32 parameters/Adam and integration, BF16 forwards. Unit checks
   must establish both-modality endpoint algebra, target detachment and exactly
   two deployment forwards; a short GPU gate must prove finite real updates,
   checkpoint reload and train/inference parity.
5. After the numerical gate, run a bounded 200-update pilot with diagnostics
   at 25/100/200 and decoded media at 100/200. This budget tests the mechanism,
   not convergence. Use our corrected held-out split for the gate; before a
   longer run, build and audit a larger prompt-only training manifest from
   existing authorized corpora and exclude evaluation prompts. Reuse cached
   embeddings where compatible. Record examples, tokens and GPU-hours, not
   just update counts. Save full optimizer state less frequently than cheap
   metrics; the pruning pilots spent most wall time on large checkpoints.
6. Advance only if decoded two-call samples retain coherent motion and audio
   and improve over the untrained two-call control across the fixed screen.
   If they fail, inspect which interval/modality fails; do not blindly extend.
   If they pass, increase data and recovery budget, then compare all 64 clips
   against released four-call V1. Report paired wins/ties/losses and failure
   counts, speech WER on applicable prompts, audiovisual sync and motion review.
   No loss threshold alone can declare quality equivalence.
7. Keep PDD as a concrete alternative, not a compulsory serial stage. Evaluate
   readable existing PDD artifacts at their trained 4/6/8-call settings first.
   If that lineage has better joint media quality, test a separate two-call
   continuation with width-16 coverage, suitable two-call rollout states and
   matching train/export/runtime grids. Preserve the continuous 0.999 PDD clock;
   it is different from the released DMD discrete ladder. Do not silently mix
   the two. A FastH3 initialization with a Base H3 PDD teacher would be a new
   treatment, not what the current V23/V24 runs already do.
8. Apply target quantization to the accepted high-precision candidate. Compare
   post-training quantization first, then use QAT and quantized-generator
   QAD/DMD refinement where measured degradation requires recovery. Keep Base
   H3 as the frozen score teacher for DMD refinement and use a learned critic;
   the distilled four-call teacher is a trajectory anchor, not a replacement
   score oracle. Gate video and audio again after each change. PDD, QAT and QAD
   remain available tools; running every stage regardless of results adds risk.

## Consumer outcome and fallback

Two calls reduce transformer evaluations, not stored backbone weights. Rough
weight-only sizes at four bits are 16.5 GB for 33B and 10 GB for 20B, before
scales, unquantized tensors, encoder, activations and decoders. Calls do not
scale end-to-end latency exactly because encoding and decoding remain.

The lab already reports a full-backbone INT4 M4 Max run at 14.8 GiB peak under
its particular settings, showing that depth pruning is not the only possible
memory route. That measurement is not a guarantee for our candidate, longer
clips or other hardware. See the [local runtime report](https://haoailab.com/blogs/fasth3-local/).

If the full-backbone two-call model misses the chosen consumer memory or
latency target, test approximately 20B at four calls before combining depth
and call compression again. The existing audited maps suggest dense30 or
VSA28 as approximately 20B candidates; they still require recovery and decoded
quality evidence. A smaller four-call model may offer the better product.
Keep 14B as a research option, not a forced release criterion.

Consumer validation follows model quality work: measure peak memory, cold/warm
end-to-end time and output quality on the available RTX and 24/36 GB Macs.
Use hardware-supported quantization and attention paths. Do not translate a
GB200 speedup into an RTX or Apple claim. Keep team dogfood instructions and
full-quality decode in the release gate; draft interpolation/preview decoding
must be separately labeled.

## Budget and current execution status

Allow 1–2 days for implementation, numerical checks and a decoded four-to-two
pilot, excluding queue/access delays. Allow several more days for successful
continuation and target quantization; one to two weeks is a planning allowance,
not a promised release. Measure throughput before giving a training ETA.

No new four-to-two, PDD, QAT or QAD job was launched during this investigation.
The current work is completing the existing pruning experiment and preparing
the next distillation implementation. This document specifies the next work;
it must not be reported as completed training or a ready consumer model.
