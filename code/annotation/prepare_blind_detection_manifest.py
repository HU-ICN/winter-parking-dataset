#!/usr/bin/env python3
"""Build the frozen randomized INDEPENDENT-BLIND-331 manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--expected-count", type=int, default=331)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_stems = [
        line.strip() for line in args.test_split.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with args.provenance.open(newline="", encoding="utf-8") as handle:
        provenance = {row["image"]: row["source"] for row in csv.DictReader(handle)}
    test_names = [stem if stem.endswith(".png") else f"{stem}.png" for stem in test_stems]
    selected = [name for name in test_names if provenance.get(name) == "manual"]
    missing_provenance = [name for name in test_names if name not in provenance]
    if missing_provenance:
        raise ValueError(f"test images missing provenance: {missing_provenance[:5]}")
    if len(test_names) != 723:
        raise ValueError(f"expected 723 test frames, found {len(test_names)}")
    if len(selected) != args.expected_count or len(set(selected)) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} unique manual test frames, found {len(selected)}"
        )
    missing_images = [name for name in selected if not (args.image_dir / name).is_file()]
    if missing_images:
        raise FileNotFoundError(f"images missing: {missing_images[:5]}")

    rng = random.Random(args.seed)
    rng.shuffle(selected)
    fields = [
        "sequence",
        "anonymous_id",
        "image_name",
        "image_path",
        "image_sha256",
        "label_set_id",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=args.output.parent,
        delete=False, suffix=".tmp"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, name in enumerate(selected, start=1):
            image_path = (args.image_dir / name).resolve()
            writer.writerow({
                "sequence": index,
                "anonymous_id": f"B{index:04d}",
                "image_name": name,
                "image_path": str(image_path),
                "image_sha256": sha256_file(image_path),
                "label_set_id": "INDEPENDENT-BLIND-331",
            })
        temporary = Path(handle.name)
    os.replace(temporary, args.output)

    metadata = {
        "label_set_id": "INDEPENDENT-BLIND-331",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "random_seed": args.seed,
        "frame_count": len(selected),
        "source_test_frame_count": len(test_names),
        "selection_rule": "test split intersect provenance source=manual",
        "test_split_sha256": sha256_file(args.test_split),
        "provenance_sha256": sha256_file(args.provenance),
        "manifest_sha256": sha256_file(args.output),
    }
    atomic_text(
        args.output.with_suffix(".meta.json"),
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

