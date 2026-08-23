# MiniMax-H3 native T2VA preprocessing

This harness freezes producer outputs and encodes H3 video, audio, and text
latents without resizing or truncating their native aligned shape. It is the
v10 data-only DMD input contract.

Do not execute Python on the login node. Compute nodes do not mount `/home`,
so use the supplied Slurm jobs, the Lustre execution clone, and a
Lustre-backed `HOME`. Any standalone Python command shown below is intended
to run only inside an allocated compute job. The login node is used only to
commit/sync code and submit or inspect Slurm jobs.

## Freeze

```bash
python scripts/preprocess/minimax_h3_native_t2va/freeze_sources.py --dry-run
python scripts/preprocess/minimax_h3_native_t2va/freeze_sources.py
python scripts/preprocess/minimax_h3_native_t2va/freeze_sources.py --verify-existing
python scripts/preprocess/minimax_h3_native_t2va/validate_heldout_media.py
```

Eligibility is the intersection of completed producer status, a direct child
of the canonical `videos/` directory, and a valid prompt. Recursive video
discovery is forbidden because it would admit `server_tmp` and `_archive`
artifacts. The freeze is immutable; reruns verify it instead of updating it.

The validation split uses seed `20260822`, contains 64 unique conditioning
IDs, and is balanced 32 NuVA / 32 VidProM. Its backend input is a JSON object
with the records under the `data` field, at exactly:

```text
/mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v1/validation/heldout64.json
```

Every selected conditioning ID is removed from every training source, so a
low/high-resolution NuVA duplicate cannot leak through another root.

## Extend the frozen v1 dataset

The v2 flow preserves v1 byte-for-byte, retains its exact held-out 64, and
extends only `h3_t2av_video_nuva_50k_720_mixed_len` from a single captured
producer-status snapshot. It writes the combined prompts and training media
manifests into the same five source-specific folders under:

```text
/mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v2
```

First submit the compute-only freeze and seed job. It verifies v1, freezes
the newly completed video/prompt intersection, and hardlinks an old parquet
only when its native shape, ordered conditioning IDs, and frozen rows match
exactly. All unmatched chunks remain for GPU workers.

```bash
PREP_JOB=$(sbatch --parsable --partition=hpc-rack-3 \
  scripts/preprocess/minimax_h3_native_t2va/prepare_extension_1tray.sbatch)
```

After that job succeeds, fan the one NuVA-50k worklist across both full
18-tray racks (up to 36 trays / 144 GPUs). Chunk claims make the workers safe
to run concurrently, and independent one-node requests let Slinky scale each
rack without requiring a contiguous multi-node allocation.

```bash
ROOT=/mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v2
WORKLIST=$ROOT/h3_t2av_video_nuva_50k_720_mixed_len/work/worklist.json
ENCODE_JOBS=()
for partition in hpc-rack-2 hpc-rack-3; do
  for worker in $(seq 1 18); do
    ENCODE_JOBS+=("$(sbatch --parsable --partition="$partition" \
      --dependency=afterok:$PREP_JOB --export=NIL \
      scripts/preprocess/minimax_h3_native_t2va/encode_native_t2va_1tray.sbatch \
      "$WORKLIST" 120)")
  done
done
```

Publish `READY.json` only after all workers exit successfully:

```bash
DEPENDENCY=$(IFS=:; echo "${ENCODE_JOBS[*]}")
sbatch --parsable --partition=hpc-rack-3 \
  --dependency=afterok:$DEPENDENCY \
  scripts/preprocess/minimax_h3_native_t2va/finalize_extension_1tray.sbatch
```

The finalizer rejects missing, duplicate, failed, stray, mis-bucketed, or
wrong-shaped rows and verifies that each encoded caption equals its frozen
prompt. Do not point training at v2 until its aggregate `READY.json` exists.

## Encode

Run the CPU schema check, then a real one-GPU probe. A probe writes under
`work/probe/`; it never creates production completion or READY markers.

```bash
python scripts/preprocess/minimax_h3_native_t2va/schema_dry_run.py

WORKLIST=/mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v1/h3_t2av_video_nuva_10k_mixed_res_len/work/worklist.json
sbatch --export=NIL scripts/preprocess/minimax_h3_native_t2va/probe_native_t2va.sbatch "$WORKLIST"
```

After inspecting the probe log and parquet, submit one or more tray workers
per source. Each output is under the exact sampler contract
`data/bucket=<width>x<height>-<num_frames>f`, uses the T2VA PyArrow schema,
Zstandard compression, and one row per row group.

```bash
WORKLIST=/absolute/source/work/worklist.json
sbatch --export=NIL scripts/preprocess/minimax_h3_native_t2va/encode_native_t2va_1tray.sbatch "$WORKLIST"
```

Only after all five worklists finish successfully:

```bash
python scripts/preprocess/minimax_h3_native_t2va/finalize_dataset.py
python scripts/preprocess/minimax_h3_native_t2va/finalize_dataset.py --verify-only
```

`finalize_dataset.py` rejects missing, duplicate, failed, stray,
mis-bucketed, or wrong-shaped rows before atomically publishing each source's
`READY.json` and the aggregate root `READY.json`. Each source also gets an
exact `data/map_style_cache/file_info.pkl` only after the full parquet set
passes inspection.
