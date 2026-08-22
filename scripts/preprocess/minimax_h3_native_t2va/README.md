# MiniMax-H3 native T2VA preprocessing

This harness freezes producer outputs and encodes H3 video, audio, and text
latents without resizing or truncating their native aligned shape. It is the
v10 data-only DMD input contract.

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
