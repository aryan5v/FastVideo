# MiniMax-H3 SFT overfit configs

Related H3 DMD2 handoff:
[`HANDOFF-h3-dmd2-vsa.md`](../../../../../HANDOFF-h3-dmd2-vsa.md).

Single- and two-sample SFT overfit runs used to validate the VSA-H3 backend,
sparsity sweep (0 / 0.8 / 0.9 / 0.95 / 0.97), the FA4 dense control, and
effective-batch-2 via gradient accumulation (`sft_fa4_bs2_overfit.yaml`).

The H3 DMD2 recipe and open parity issues are tracked separately in
[`../../distribution_matching/minimax_h3/h3_dmd.md`](../../distribution_matching/minimax_h3/h3_dmd.md).

Note: `sft_vsa95_overfit_2gpu.yaml` / `sft_vsa97_overfit_2gpu.yaml` are kept
as negative results — 2x GB200 cannot fit this model/sequence without CPU
offload (backward-pass working set alone OOMs 184 GiB cards at sp=2).
