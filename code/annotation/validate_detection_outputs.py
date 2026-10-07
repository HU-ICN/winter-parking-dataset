#!/usr/bin/env python3
"""Validate blind detection outputs without opening historical labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from blind_detection_tool import AnnotationStore, load_manifest, validate_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--addendum", type=Path)
    parser.add_argument("--instrumentation-addendum", type=Path)
    args = parser.parse_args()

    rows, manifest_hash = load_manifest(args.manifest)
    store = AnnotationStore(
        args.output_dir,
        args.protocol,
        manifest_hash,
        addendum_path=args.addendum,
        instrumentation_addendum_path=args.instrumentation_addendum,
    )
    valid = 0
    complete = 0
    boxes = 0
    for row in rows:
        path = store.output_path(row)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        validate_payload(
            payload,
            row,
            store.protocol_hash,
            store.manifest_hash,
            store.addendum_hash,
            store.instrumentation_addendum_hash,
        )
        valid += 1
        complete += int(bool(payload["revision_annotation"].get("complete")))
        boxes += len(payload.get("shapes", []))
    print(json.dumps({
        "manifest_frames": len(rows),
        "valid_outputs": valid,
        "complete_outputs": complete,
        "boxes": boxes,
        "protocol_sha256": store.protocol_hash,
        "addendum_sha256": store.addendum_hash,
        "instrumentation_addendum_sha256": store.instrumentation_addendum_hash,
        "manifest_sha256": store.manifest_hash,
    }, indent=2))


if __name__ == "__main__":
    main()
