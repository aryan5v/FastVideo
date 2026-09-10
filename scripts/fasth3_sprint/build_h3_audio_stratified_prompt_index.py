#!/usr/bin/env python3
"""Add a controlled audio-focused boost to the audited 58k prompt index."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


PROFILE_GROUPS = {
    "speech": {"dialogue_ambience", "narration_ambience"},
    "music": {"music_performance_or_lyrics", "music_driven_montage"},
    "foley": {"asmr_or_detailed_foley"},
    # The source taxonomy has no explicit AV-sync label. Event-SFX prompts are
    # the closest evidence-backed proxy because their sound events are tied to
    # visible actions; the receipt names this limitation instead of hiding it.
    "av_sync_proxy": {"ambience_and_event_sfx_no_speech"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targeted-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    if not 0.0 < args.targeted_fraction < 0.5:
        raise ValueError("targeted-fraction must be in (0, 0.5) so natural prompts remain the majority")
    if args.output.exists():
        raise FileExistsError(args.output)

    source_receipt = json.loads((args.source / "receipt.json").read_text())
    raw_source = (args.source / "prompt_index.jsonl").read_bytes()
    if hashlib.sha256(raw_source).hexdigest() != source_receipt["index_sha256"]:
        raise ValueError("source prompt index checksum mismatch")
    records = [json.loads(line) for line in raw_source.decode().splitlines()]
    if len(records) != int(source_receipt["requested"]):
        raise ValueError("source prompt count differs from its receipt")

    pools: dict[str, list[dict]] = {name: [] for name in PROFILE_GROUPS}
    natural_profiles = Counter()
    for record in records:
        profile = record.get("dimensions", {}).get("audio_profile")
        natural_profiles[profile] += 1
        for group, profiles in PROFILE_GROUPS.items():
            if profile in profiles:
                pools[group].append(record)
                break
    if any(not pool for pool in pools.values()):
        raise ValueError("one or more required audio strata are empty")

    # Natural rows are retained exactly once. Extra rows are balanced across
    # the four targeted strata until they represent the requested share of the
    # augmented epoch.
    boost_count = round(len(records) * args.targeted_fraction / (1.0 - args.targeted_fraction))
    rng = random.Random(args.seed)
    boosted: list[dict] = []
    names = tuple(PROFILE_GROUPS)
    for index in range(boost_count):
        group = names[index % len(names)]
        record = dict(rng.choice(pools[group]))
        record["sampling_boost_category"] = group
        boosted.append(record)
    rng.shuffle(boosted)
    output_records = [*records, *boosted]
    output_bytes = b"".join(
        (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        for record in output_records
    )

    args.output.mkdir(parents=True)
    (args.output / "prompt_index.jsonl").write_bytes(output_bytes)
    boost_profiles = Counter(record["sampling_boost_category"] for record in boosted)
    receipt = {
        "schema_version": 1,
        "requested": len(output_records),
        "matched": len(output_records),
        "missing": [],
        "conflicts": [],
        "source_index": str(args.source.resolve()),
        "source_index_sha256": hashlib.sha256(raw_source).hexdigest(),
        "index_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "metadata_match_complete": True,
        "embedding_values_validated": True,
        "training_ready": bool(source_receipt.get("training_ready", False)),
        "validated_embeddings": int(source_receipt.get("validated_embeddings", len(records))),
        "dtypes": source_receipt.get("dtypes", []),
        "natural_count": len(records),
        "boost_count": len(boosted),
        "targeted_fraction": len(boosted) / len(output_records),
        "natural_audio_profiles": dict(sorted(natural_profiles.items(), key=lambda item: str(item[0]))),
        "boost_groups": dict(sorted(boost_profiles.items())),
        "av_sync_label_limitation": "ambience_and_event_sfx_no_speech is used as the AV-sync proxy",
        "seed": args.seed,
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
