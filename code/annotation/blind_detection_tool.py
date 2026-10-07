#!/usr/bin/env python3
"""Blind axis-aligned vehicle annotation for the revision study.

The tool reads only image paths from a frozen randomized manifest and its own
output directory. It never loads sibling annotation files, historical labels,
prelabels, predictions, provenance, or snow-level metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


BASE_TOOL_VERSION = "1.0.14"
ADDENDUM_TOOL_VERSION = "1.1.12"
IDLE_THRESHOLD_SECONDS = 60.0
TRIM_FRACTION = 0.10
ACTIVE_WALL_RATIO_WARNING_THRESHOLD = 0.25
RENDER_VARIANCE_MINIMUM = 25.0
WINDOW_SCREEN_FRACTION = 0.90
WINDOW_BORDER_MARGIN_X = 80
WINDOW_BORDER_MARGIN_Y = 120
WINDOW_MIN_WIDTH = 960
WINDOW_MIN_HEIGHT = 640
FIT_MARGIN_PIXELS = 8
VIEWPORT_TOLERANCE_PIXELS = 0
AUTOSAVE_INTERVAL_MS = 30_000
RENDER_VALIDATION_AUTOSAVE_INTERVAL_MS = 250
RESIZE_DEBOUNCE_MS = 80
REQUIRED_MANIFEST_COLUMNS = {
    "sequence",
    "anonymous_id",
    "image_name",
    "image_path",
    "image_sha256",
    "label_set_id",
}


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def trimmed_mean(values: list[float], fraction: float = TRIM_FRACTION) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    trim = math.floor(len(ordered) * fraction)
    retained = ordered[trim : len(ordered) - trim] if trim else ordered
    return statistics.fmean(retained)


def clock_increments(
    last_clock: float,
    now: float,
    last_input_at: float | None,
    idle_threshold: float = IDLE_THRESHOLD_SECONDS,
) -> tuple[float, float]:
    """Return active and wall increments for one monotonic-clock interval."""
    wall_delta = max(0.0, now - last_clock)
    if last_input_at is None or wall_delta == 0.0:
        return 0.0, wall_delta
    active_until = last_input_at + idle_threshold
    active_delta = max(0.0, min(now, active_until) - last_clock)
    return min(active_delta, wall_delta), wall_delta


def clamped(value: int, lower: int, upper: int) -> int:
    if upper < lower:
        lower = upper
    return max(lower, min(value, upper))


def adaptive_window_geometry(screen_width: int, screen_height: int) -> dict[str, int]:
    if screen_width <= 0 or screen_height <= 0:
        raise ValueError("screen dimensions must be positive")
    available_width = max(320, screen_width - WINDOW_BORDER_MARGIN_X)
    available_height = max(320, screen_height - WINDOW_BORDER_MARGIN_Y)
    minimum_width = min(WINDOW_MIN_WIDTH, available_width)
    minimum_height = min(WINDOW_MIN_HEIGHT, available_height)
    width = clamped(
        round(screen_width * WINDOW_SCREEN_FRACTION),
        minimum_width,
        available_width,
    )
    height = clamped(
        round(screen_height * WINDOW_SCREEN_FRACTION),
        minimum_height,
        available_height,
    )
    x = max(0, (screen_width - width) // 2)
    y = max(24, (screen_height - height) // 2)
    if y + height > screen_height - 24:
        y = max(0, screen_height - height - 24)
    return {
        "screen_width": screen_width,
        "screen_height": screen_height,
        "available_width": available_width,
        "available_height": available_height,
        "window_width": width,
        "window_height": height,
        "window_x": x,
        "window_y": y,
    }


def viewport_containment_report(
    canvas_width: int,
    canvas_height: int,
    image_bbox: tuple[int, int, int, int],
    tolerance: int = VIEWPORT_TOLERANCE_PIXELS,
) -> dict[str, Any]:
    left, top, right, bottom = image_bbox
    corners = {
        "top_left": [left, top],
        "top_right": [right, top],
        "bottom_left": [left, bottom],
        "bottom_right": [right, bottom],
    }
    visible = {
        name: (
            -tolerance <= point[0] <= canvas_width + tolerance
            and -tolerance <= point[1] <= canvas_height + tolerance
        )
        for name, point in corners.items()
    }
    image_width = max(0, right - left)
    image_height = max(0, bottom - top)
    intersection_width = max(0, min(right, canvas_width) - max(left, 0))
    intersection_height = max(0, min(bottom, canvas_height) - max(top, 0))
    image_area = image_width * image_height
    intersection_area = intersection_width * intersection_height
    intersection_ratio = intersection_area / image_area if image_area else 0.0
    width_fits = image_width <= canvas_width
    height_fits = image_height <= canvas_height
    fully_visible = (
        all(visible.values())
        and width_fits
        and height_fits
        and abs(intersection_ratio - 1.0) <= 1e-12
    )
    return {
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "image_bbox": [left, top, right, bottom],
        "image_corners": corners,
        "corner_visibility": visible,
        "tolerance_pixels": tolerance,
        "image_width": image_width,
        "image_height": image_height,
        "image_width_fits_canvas": width_fits,
        "image_height_fits_canvas": height_fits,
        "image_canvas_intersection_area": intersection_area,
        "image_area": image_area,
        "image_canvas_intersection_ratio": round(intersection_ratio, 12),
        "image_fully_visible": fully_visible,
    }


def legacy_render_regression_report(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    photo_width = int(payload["tk_photo_width"])
    photo_height = int(payload["tk_photo_height"])
    canvas_width = int(payload["tk_canvas_width"])
    canvas_height = int(payload["tk_canvas_height"])
    width_fits = photo_width <= canvas_width
    height_fits = photo_height <= canvas_height
    corner_visibility = payload.get("corner_visibility")
    corner_evidence_complete = isinstance(corner_visibility, dict) and all(
        corner_visibility.get(name) is True
        for name in ("top_left", "top_right", "bottom_left", "bottom_right")
    )
    accepted = width_fits and height_fits and corner_evidence_complete
    reasons = []
    if not width_fits:
        reasons.append("photo_width_exceeds_canvas")
    if not height_fits:
        reasons.append("photo_height_exceeds_canvas")
    if not corner_evidence_complete:
        reasons.append("four_corner_visibility_evidence_missing_or_failed")
    return {
        "source": str(path),
        "source_sha256": sha256_file(path),
        "old_render_assertion_passed": payload.get("render_assertion_passed"),
        "tk_photo_width": photo_width,
        "tk_photo_height": photo_height,
        "tk_canvas_width": canvas_width,
        "tk_canvas_height": canvas_height,
        "photo_width_fits_canvas": width_fits,
        "photo_height_fits_canvas": height_fits,
        "four_corner_evidence_complete": corner_evidence_complete,
        "new_visibility_predicate_passed": accepted,
        "failure_reasons": reasons,
    }


@dataclass(frozen=True)
class ManifestRow:
    sequence: int
    anonymous_id: str
    image_name: str
    image_path: Path
    image_sha256: str
    label_set_id: str
    eligible_from: str | None = None  # optional ISO-8601 datetime with offset (tools 1.0.12, test-retest)


def parse_eligible_from(value: str | None) -> datetime | None:
    if value is None or not str(value).strip():
        return None
    parsed = datetime.fromisoformat(str(value).strip())
    if parsed.tzinfo is None:
        raise ValueError("eligible_from must carry a UTC offset")
    return parsed


def row_eligible(row: ManifestRow, now: datetime | None = None) -> bool:
    """A row is eligible when it has no eligible_from or that instant has passed (14-day test-retest rule)."""
    threshold = parse_eligible_from(row.eligible_from)
    if threshold is None:
        return True
    return (now or datetime.now(timezone.utc)) >= threshold


def eligibility_summary(rows: list[ManifestRow], completed: set[str], now: datetime | None = None) -> dict[str, Any]:
    """Counts used by the launcher and the startup gate; never lists frames."""
    now = now or datetime.now(timezone.utc)
    pending = [r for r in rows if r.anonymous_id not in completed]
    eligible = [r for r in pending if row_eligible(r, now)]
    blocked = [r for r in pending if not row_eligible(r, now)]
    next_at = min((parse_eligible_from(r.eligible_from) for r in blocked), default=None)
    return dict(checked_at=now.isoformat(timespec="seconds"), completed=len(rows) - len(pending), pending=len(pending),
                eligible_pending=len(eligible), blocked_pending=len(blocked),
                next_eligible_at=next_at.isoformat(timespec="seconds") if next_at else None)


def load_manifest(path: Path, verify_images: bool = True) -> tuple[list[ManifestRow], str]:
    manifest_hash = sha256_file(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_MANIFEST_COLUMNS - fields
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        raw_rows = list(reader)

    rows: list[ManifestRow] = []
    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    for raw in raw_rows:
        eligible_from = (raw.get("eligible_from") or "").strip() or None
        parse_eligible_from(eligible_from)  # validates the format; raises on a naive datetime
        row = ManifestRow(
            sequence=int(raw["sequence"]),
            anonymous_id=raw["anonymous_id"],
            image_name=raw["image_name"],
            image_path=Path(raw["image_path"]),
            image_sha256=raw["image_sha256"],
            label_set_id=raw["label_set_id"],
            eligible_from=eligible_from,
        )
        if row.anonymous_id in seen_ids:
            raise ValueError(f"duplicate anonymous_id: {row.anonymous_id}")
        if row.image_name in seen_images:
            raise ValueError(f"duplicate image: {row.image_name}")
        if row.image_path.name != row.image_name:
            raise ValueError(f"image name/path mismatch for {row.anonymous_id}")
        if verify_images and not row.image_path.is_file():
            raise FileNotFoundError(row.image_path)
        seen_ids.add(row.anonymous_id)
        seen_images.add(row.image_name)
        rows.append(row)

    rows.sort(key=lambda item: item.sequence)
    if [row.sequence for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("manifest sequence must be contiguous and one-based")
    if len({row.label_set_id for row in rows}) != 1:
        raise ValueError("manifest must contain exactly one label_set_id")
    return rows, manifest_hash


class AnnotationStore:
    def __init__(
        self,
        output_dir: Path,
        protocol_path: Path,
        manifest_hash: str,
        addendum_path: Path | None = None,
        instrumentation_addendum_path: Path | None = None,
        sealed: dict[str, str] | None = None,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # file name -> sealed SHA256; sealed frames are never written (tools 1.0.10+)
        self.sealed: dict[str, str] = dict(sealed or {})
        self.protocol_path = protocol_path
        self.protocol_hash = sha256_file(protocol_path)
        self.manifest_hash = manifest_hash
        self.addendum_hash = sha256_file(addendum_path) if addendum_path else None
        self.instrumentation_addendum_hash = (
            sha256_file(instrumentation_addendum_path)
            if instrumentation_addendum_path
            else None
        )
        self.tool_version = (
            ADDENDUM_TOOL_VERSION if addendum_path else BASE_TOOL_VERSION
        )

    def output_path(self, row: ManifestRow) -> Path:
        return self.output_dir / f"{row.anonymous_id}.json"

    def blank(self, row: ManifestRow, width: int, height: int) -> dict[str, Any]:
        now = iso_now()
        return {
            "version": "3.2.6",
            "flags": {},
            "shapes": [],
            "imagePath": row.image_name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
            "description": "",
            "revision_annotation": {
                "tool": "blind_detection_tool",
                "tool_version": self.tool_version,
                "label_set_id": row.label_set_id,
                "anonymous_id": row.anonymous_id,
                "sequence": row.sequence,
                "protocol_sha256": self.protocol_hash,
                "addendum_sha256": self.addendum_hash,
                "instrumentation_addendum_sha256": self.instrumentation_addendum_hash,
                "manifest_sha256": self.manifest_hash,
                "image_sha256": row.image_sha256,
                "started_at": now,
                "last_saved_at": now,
                "completed_at": None,
                "active_seconds": 0.0,
                "wall_seconds": 0.0,
                "complete": False,
                "frame_note": "",
                **({"eligible_from": row.eligible_from} if row.eligible_from else {}),
            },
        }

    def load(self, row: ManifestRow, width: int, height: int) -> dict[str, Any]:
        path = self.output_path(row)
        if not path.exists():
            return self.blank(row, width, height)
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        validate_payload(
            payload,
            row,
            self.protocol_hash,
            self.manifest_hash,
            self.addendum_hash,
            self.instrumentation_addendum_hash,
        )
        return payload

    def is_sealed(self, row: ManifestRow) -> bool:
        return self.output_path(row).name in self.sealed

    def verify_sealed(self) -> list[str]:
        """Names of sealed files that are missing or whose digest differs from the seal list."""
        bad = []
        for name, digest in self.sealed.items():
            path = self.output_dir / name
            if not path.exists() or sha256_file(path) != digest:
                bad.append(name)
        return bad

    def save(self, row: ManifestRow, payload: dict[str, Any]) -> bool:
        if self.is_sealed(row):
            return False  # sealed frame: no write of any kind, not even timing metadata
        validate_payload(
            payload,
            row,
            self.protocol_hash,
            self.manifest_hash,
            self.addendum_hash,
            self.instrumentation_addendum_hash,
        )
        atomic_json_write(self.output_path(row), payload)
        return True


def load_sealed_lists(paths: list[Path]) -> dict[str, str]:
    sealed: dict[str, str] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                sealed[parts[1]] = parts[0]
    return sealed


def discover_seal_lists(seals_dir: Path, rows: list[ManifestRow]) -> list[Path]:
    found: list[Path] = []
    for label_set_id in sorted({row.label_set_id for row in rows}):
        found += sorted(seals_dir.glob(f"{label_set_id}.sha256")) + sorted(seals_dir.glob(f"{label_set_id}.*.sha256"))
    return found


def make_shape(box: tuple[float, float, float, float], partially_buried: bool) -> dict[str, Any]:
    x1, y1, x2, y2 = box
    return {
        "kie_linking": [],
        "label": "car",
        "score": None,
        "points": [[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y2, 2)]],
        "group_id": None,
        "description": "",
        "difficult": False,
        "shape_type": "rectangle",
        "flags": {"partially_buried": bool(partially_buried)},
        "attributes": {"partially_buried": bool(partially_buried)},
    }


def validate_payload(
    payload: dict[str, Any],
    row: ManifestRow,
    protocol_hash: str,
    manifest_hash: str,
    addendum_hash: str | None = None,
    instrumentation_addendum_hash: str | None = None,
) -> None:
    if payload.get("imagePath") != row.image_name:
        raise ValueError(f"output imagePath mismatch for {row.anonymous_id}")
    width = int(payload.get("imageWidth", 0))
    height = int(payload.get("imageHeight", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid dimensions for {row.anonymous_id}")
    metadata = payload.get("revision_annotation") or {}
    expected = {
        "label_set_id": row.label_set_id,
        "anonymous_id": row.anonymous_id,
        "protocol_sha256": protocol_hash,
        "manifest_sha256": manifest_hash,
        "image_sha256": row.image_sha256,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"metadata {key} mismatch for {row.anonymous_id}")
    if addendum_hash is not None and metadata.get("addendum_sha256") != addendum_hash:
        raise ValueError(f"metadata addendum_sha256 mismatch for {row.anonymous_id}")
    if (
        instrumentation_addendum_hash is not None
        and metadata.get("instrumentation_addendum_sha256")
        != instrumentation_addendum_hash
    ):
        raise ValueError(
            f"metadata instrumentation_addendum_sha256 mismatch for {row.anonymous_id}"
        )
    if not metadata.get("started_at") or not metadata.get("last_saved_at"):
        raise ValueError(f"timestamps missing for {row.anonymous_id}")

    boxes: set[tuple[float, float, float, float]] = set()
    for shape in payload.get("shapes", []):
        if shape.get("label") != "car" or shape.get("shape_type") != "rectangle":
            raise ValueError(f"unsupported shape in {row.anonymous_id}")
        points = shape.get("points") or []
        if len(points) != 2 or any(len(point) != 2 for point in points):
            raise ValueError(f"invalid rectangle points in {row.anonymous_id}")
        x1, y1 = map(float, points[0])
        x2, y2 = map(float, points[1])
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"out-of-bounds rectangle in {row.anonymous_id}")
        key = tuple(round(value, 2) for value in (x1, y1, x2, y2))
        if key in boxes:
            raise ValueError(f"duplicate rectangle in {row.anonymous_id}")
        boxes.add(key)


def progress_report(
    output_dir: Path, rows: list[ManifestRow], daily_hours: float
) -> dict[str, Any]:
    durations: list[float] = []
    wall_durations: list[float] = []
    completed_at: list[datetime] = []
    for row in rows:
        path = output_dir / f"{row.anonymous_id}.json"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            metadata = (json.load(handle).get("revision_annotation") or {})
        if metadata.get("complete"):
            active_seconds = float(metadata.get("active_seconds", 0.0))
            wall_seconds = float(metadata.get("wall_seconds", active_seconds))
            durations.append(active_seconds)
            wall_durations.append(wall_seconds)
            if metadata.get("completed_at"):
                completed_at.append(datetime.fromisoformat(metadata["completed_at"]))
    if not durations:
        raise ValueError("no completed frames for progress report")
    mean_seconds = statistics.fmean(durations)
    median_seconds = statistics.median(durations)
    q75_seconds = quantile(durations, 0.75)
    trimmed_mean_seconds = trimmed_mean(durations)
    total_frames = len(rows)
    remaining = max(0, total_frames - len(durations))
    remaining_hours = remaining * median_seconds / 3600.0
    projected_date = datetime.now(timezone.utc).astimezone() + timedelta(
        days=remaining_hours / max(daily_hours, 0.1)
    )
    deadline = datetime.fromisoformat("2026-08-23T23:59:59+09:00")
    total_active = sum(durations)
    total_wall = sum(wall_durations)
    active_wall_ratio = total_active / total_wall if total_wall > 0 else 0.0
    ratio_warning = active_wall_ratio < ACTIVE_WALL_RATIO_WARNING_THRESHOLD
    report = {
        "generated_at": iso_now(),
        "completed_frames": len(durations),
        "total_frames": total_frames,
        "mean_seconds_per_frame": round(mean_seconds, 2),
        "median_seconds_per_frame": round(median_seconds, 2),
        "q75_seconds_per_frame": round(q75_seconds, 2),
        "trimmed_mean_seconds_per_frame": round(trimmed_mean_seconds, 2),
        "trim_fraction_each_tail": TRIM_FRACTION,
        "projection_estimator": "median_seconds_per_frame",
        "projected_total_annotation_hours": round(
            total_frames * median_seconds / 3600.0, 2
        ),
        "remaining_annotation_hours": round(remaining_hours, 2),
        "total_active_seconds": round(total_active, 2),
        "total_wall_seconds": round(total_wall, 2),
        "active_wall_ratio": round(active_wall_ratio, 4),
        "active_wall_ratio_warning_threshold": ACTIVE_WALL_RATIO_WARNING_THRESHOLD,
        "active_wall_ratio_warning": ratio_warning,
        "active_wall_ratio_warning_message": (
            "Active time is below 25% of wall time; review idle periods before using the projection."
            if ratio_warning
            else None
        ),
        "idle_threshold_seconds": IDLE_THRESHOLD_SECONDS,
        "assumed_daily_annotation_hours": daily_hours,
        "projected_completion_date": projected_date.date().isoformat(),
        "schedule_deadline": "2026-08-23",
        "schedule_warning": projected_date > deadline,
        "uses_label_comparison": False,
    }
    atomic_json_write(output_dir / "progress_gate_30.json", report)
    return report


def run_self_test(
    manifest_path: Path,
    output_dir: Path,
    protocol_path: Path,
    sample_count: int,
    addendum_path: Path | None = None,
    instrumentation_addendum_path: Path | None = None,
) -> None:
    from PIL import Image

    rows, manifest_hash = load_manifest(manifest_path)
    store = AnnotationStore(
        output_dir,
        protocol_path,
        manifest_hash,
        addendum_path=addendum_path,
        instrumentation_addendum_path=instrumentation_addendum_path,
    )
    for row in rows[:sample_count]:
        with Image.open(row.image_path) as image:
            width, height = image.size
        payload = store.blank(row, width, height)
        payload["shapes"].append(
            make_shape((width * 0.1, height * 0.1, width * 0.2, height * 0.2), False)
        )
        metadata = payload["revision_annotation"]
        metadata["active_seconds"] = 1.0
        metadata["wall_seconds"] = 1.0
        metadata["last_saved_at"] = iso_now()
        metadata["completed_at"] = iso_now()
        metadata["complete"] = True
        store.save(row, payload)
        reloaded = store.load(row, width, height)
        if len(reloaded["shapes"]) != 1 or not reloaded["revision_annotation"]["complete"]:
            raise AssertionError(f"round-trip failure for {row.anonymous_id}")
    print(f"self-test passed for {sample_count} manifest images in {output_dir}")


def run_eligibility_self_test() -> None:
    """Synthetic rows only: past / future / absent eligible_from, naive datetime rejected, summary counts."""
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone(timedelta(hours=9)))
    mk = lambda i, e: ManifestRow(i, f"T{i:04d}", f"t{i}.png", Path(f"/nonexistent/t{i}.png"), "0" * 64, "SELFTEST", e)
    rows = [mk(1, "2026-09-01T00:00:00+09:00"), mk(2, "2026-09-20T00:00:00+09:00"), mk(3, None), mk(4, "2026-09-10T12:00:00+09:00")]
    assert row_eligible(rows[0], now) and not row_eligible(rows[1], now) and row_eligible(rows[2], now) and row_eligible(rows[3], now)
    s = eligibility_summary(rows, {"T0001"}, now)
    assert (s["completed"], s["pending"], s["eligible_pending"], s["blocked_pending"]) == (1, 3, 2, 1), s
    assert s["next_eligible_at"] == "2026-09-20T00:00:00+09:00", s
    s2 = eligibility_summary([rows[1]], set(), now)
    assert s2["eligible_pending"] == 0 and s2["blocked_pending"] == 1
    try:
        parse_eligible_from("2026-09-20T00:00:00"); raise AssertionError("naive datetime accepted")
    except ValueError:
        pass
    print("eligibility self-test passed (4 rows; past/future/absent/boundary; naive datetime rejected)")


def run_timing_self_test(output_path: Path) -> None:
    idle_active, idle_wall = clock_increments(100.0, 220.0, None)
    recent_active, recent_wall = clock_increments(100.0, 220.0, 100.0)
    delayed_active, delayed_wall = clock_increments(100.0, 220.0, 130.0)
    if abs(idle_active) > 1e-9 or abs(idle_wall - 120.0) > 1e-9:
        raise AssertionError("idle interval must add wall time but no active time")
    if abs(recent_active - 60.0) > 1e-9 or abs(recent_wall - 120.0) > 1e-9:
        raise AssertionError("one input must cap active time at the 60-second threshold")
    if abs(delayed_active - 90.0) > 1e-9 or abs(delayed_wall - 120.0) > 1e-9:
        raise AssertionError("input at 30 seconds must count 30 prior plus 60 post-input seconds")

    synthetic = [20.0, 25.0, 30.0, 35.0, 3600.0]
    if statistics.median(synthetic) != 30.0:
        raise AssertionError("median projection fixture is invalid")
    with tempfile.TemporaryDirectory(prefix="blind-timing-test-") as temporary_name:
        temporary = Path(temporary_name)
        rows = [
            ManifestRow(
                sequence=index,
                anonymous_id=f"T{index:04d}",
                image_name=f"fixture_{index}.png",
                image_path=temporary / f"fixture_{index}.png",
                image_sha256="0" * 64,
                label_set_id="TIMING-SELF-TEST",
            )
            for index in range(1, len(synthetic) + 1)
        ]
        for row, duration in zip(rows, synthetic):
            atomic_json_write(
                temporary / f"{row.anonymous_id}.json",
                {
                    "revision_annotation": {
                        "complete": True,
                        "completed_at": iso_now(),
                        "active_seconds": duration,
                        "wall_seconds": 5000.0,
                    }
                },
            )
        projection = progress_report(temporary, rows, daily_hours=2.0)
    if projection["projection_estimator"] != "median_seconds_per_frame":
        raise AssertionError("progress report must declare the median estimator")
    if projection["projected_total_annotation_hours"] != round(5 * 30 / 3600, 2):
        raise AssertionError("progress report did not use the median projection")
    if not projection["active_wall_ratio_warning"]:
        raise AssertionError("low active/wall ratio must produce a warning")
    geometry = adaptive_window_geometry(1440, 900)
    if geometry["window_width"] > geometry["available_width"]:
        raise AssertionError("adaptive window exceeds available screen width")
    if geometry["window_height"] > geometry["available_height"]:
        raise AssertionError("adaptive window exceeds available screen height")
    visible_fixture = viewport_containment_report(1000, 800, (10, 10, 990, 790))
    hidden_fixture = viewport_containment_report(1000, 800, (-4, 10, 990, 790))
    if not visible_fixture["image_fully_visible"]:
        raise AssertionError("valid viewport fixture was rejected")
    if hidden_fixture["image_fully_visible"]:
        raise AssertionError("out-of-bounds viewport fixture was accepted")
    legacy_evidence_root = Path(__file__).resolve().parent / "evidence" / "tool_1_0_2"
    legacy_regressions = [
        legacy_render_regression_report(legacy_evidence_root / cohort / "render_evidence.json")
        for cohort in ("render_b", "render_x")
    ]
    for regression in legacy_regressions:
        if regression["old_render_assertion_passed"] is not True:
            raise AssertionError("legacy regression fixture did not pass the old predicate")
        if regression["new_visibility_predicate_passed"]:
            raise AssertionError("new visibility predicate accepted clipped legacy evidence")
    payload = {
        "generated_at": iso_now(),
        "tool_version": BASE_TOOL_VERSION,
        "idle_threshold_seconds": IDLE_THRESHOLD_SECONDS,
        "simulated_idle_seconds": 120.0,
        "idle_active_increment": idle_active,
        "idle_wall_increment": idle_wall,
        "single_input_active_increment": recent_active,
        "single_input_wall_increment": recent_wall,
        "input_at_30s_active_increment": delayed_active,
        "input_at_30s_wall_increment": delayed_wall,
        "projection_fixture_seconds": synthetic,
        "fixture_mean_seconds": statistics.fmean(synthetic),
        "fixture_median_seconds": statistics.median(synthetic),
        "projection_estimator": "median_seconds_per_frame",
        "progress_report_projection_hours": projection[
            "projected_total_annotation_hours"
        ],
        "progress_report_mean_seconds": projection["mean_seconds_per_frame"],
        "progress_report_median_seconds": projection["median_seconds_per_frame"],
        "progress_report_q75_seconds": projection["q75_seconds_per_frame"],
        "progress_report_trimmed_mean_seconds": projection[
            "trimmed_mean_seconds_per_frame"
        ],
        "progress_report_active_wall_ratio": projection["active_wall_ratio"],
        "progress_report_ratio_warning": projection["active_wall_ratio_warning"],
        "adaptive_window_geometry_fixture": geometry,
        "visible_viewport_fixture_passed": visible_fixture["image_fully_visible"],
        "out_of_bounds_viewport_fixture_rejected": not hidden_fixture[
            "image_fully_visible"
        ],
        "legacy_clipped_render_regressions": legacy_regressions,
        "assertions_passed": True,
    }
    atomic_json_write(output_path, payload)
    print(json.dumps(payload, indent=2))


class BlindDetectionApp:
    def __init__(
        self,
        rows: list[ManifestRow],
        store: AnnotationStore,
        protocol_path: Path,
        progress_gate: int,
        acknowledge_progress_gate: bool,
        daily_hours: float,
        autosave_interval_ms: int = AUTOSAVE_INTERVAL_MS,
    ):
        import tkinter as tk
        from tkinter import messagebox, simpledialog
        from PIL import Image, ImageTk

        self.tk = tk
        self.messagebox = messagebox
        self.simpledialog = simpledialog
        self.Image = Image
        self.ImageTk = ImageTk
        self.rows = rows
        self.store = store
        self.protocol_path = protocol_path
        self.progress_gate = progress_gate
        self.acknowledge_progress_gate = acknowledge_progress_gate
        self.daily_hours = daily_hours
        self.autosave_interval_ms = autosave_interval_ms
        self.index = 0
        self.payload: dict[str, Any] = {}
        self.image = None
        self.photo = None
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.selected: int | None = None
        self.draw_start: tuple[float, float] | None = None
        self.preview_id: int | None = None
        self.pan_start: tuple[float, float] | None = None
        self.undo_stack: list[list[dict[str, Any]]] = []
        self.last_clock = time.monotonic()
        self.last_input_at: float | None = None
        self.is_fullscreen = False
        self._closing = False
        self._autosave_after_id: str | None = None
        self._resize_after_id: str | None = None
        self._last_canvas_size: tuple[int, int] | None = None
        self.autosave_count = 0
        self.image_item_id: int | None = None
        self.startup_viewport_report: dict[str, Any] | None = None
        self.last_viewport_report: dict[str, Any] | None = None

        self.root = tk.Tk()
        self.root.title("Independent blind vehicle annotation")
        self.initial_geometry = adaptive_window_geometry(
            self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        )
        self.root.geometry(
            "{window_width}x{window_height}+{window_x}+{window_y}".format(
                **self.initial_geometry
            )
        )
        self.root.minsize(
            min(800, self.initial_geometry["window_width"]),
            min(560, self.initial_geometry["window_height"]),
        )
        self.root.configure(bg="#151515")
        self.status = tk.Label(
            self.root, bg="#151515", fg="#f0f0f0", anchor="w", font=("Menlo", 13)
        )
        self.status.pack(fill="x")
        self.help = tk.Label(
            self.root,
            bg="#151515",
            fg="#aaaaaa",
            anchor="w",
            font=("Menlo", 10),
            text=(
                "drag: new box | right-click: select | Delete: remove | b: buried | "
                "u: undo | n: frame note | wheel: zoom | middle-drag: pan | Enter: complete+next | "
                "arrows: save draft+navigate | g: next incomplete | s: save | "
                "f: fullscreen | 0/r: fit | Esc: leave fullscreen | q: save+quit"
            ),
        )
        self.help.pack(fill="x")
        self.canvas = tk.Canvas(self.root, bg="#090909", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Button-2>", self.on_pan_press)
        self.canvas.bind("<B2-Motion>", self.on_pan_motion)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_release)
        self.canvas.bind("<Button-3>", self.on_select)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", lambda event: self.zoom_at(event.x, event.y, 1.15))
        self.canvas.bind("<Button-5>", lambda event: self.zoom_at(event.x, event.y, 1 / 1.15))
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.root.bind("<Key>", self.on_key)
        self.root.bind("<KeyRelease-Return>", self.on_return_release)
        self.root.bind("<KeyRelease-KP_Enter>", self.on_return_release)
        self._return_armed = True
        self.root.protocol("WM_DELETE_WINDOW", self.request_quit)
        self.root.update_idletasks()
        self.goto_first_incomplete()
        self.startup_viewport_report = self.assert_image_fully_visible("startup")
        self.schedule_autosave()

    def completed_count(self) -> int:
        count = 0
        for row in self.rows:
            path = self.store.output_path(row)
            if not path.exists():
                continue
            try:
                with path.open(encoding="utf-8") as handle:
                    if (json.load(handle).get("revision_annotation") or {}).get("complete"):
                        count += 1
            except (OSError, json.JSONDecodeError):
                continue
        return count

    def _row_complete(self, row: ManifestRow) -> bool:
        path = self.store.output_path(row)
        if not path.exists():
            return False
        try:
            with path.open(encoding="utf-8") as handle:
                return bool((json.load(handle).get("revision_annotation") or {}).get("complete"))
        except (OSError, json.JSONDecodeError):
            return False

    def goto_first_incomplete(self) -> None:
        # First incomplete frame that is eligible (tools 1.0.12); blocked frames are skipped in manifest order
        # and become reachable once their eligible_from instant has passed.
        blocked = 0
        for index, row in enumerate(self.rows):
            if self._row_complete(row):
                continue
            if not row_eligible(row):
                blocked += 1
                continue
            self.load_index(index)
            if blocked:
                self._hint(f"{blocked} incomplete frame(s) before this one are not yet eligible and were skipped")
            return
        if blocked:
            summary = eligibility_summary(self.rows, {r.anonymous_id for r in self.rows if self._row_complete(r)})
            msg = (f"none of the {summary['pending']} remaining frame(s) is eligible yet (14-day test-retest rule); "
                   f"next frame becomes eligible at {summary['next_eligible_at']}")
            if self.payload:
                self._hint(msg); return
            sys.stderr.write("ELIGIBILITY GATE: " + msg + "\n")
            try:
                self.root.destroy()
            except Exception:
                pass
            raise SystemExit(7)
        self.load_index(0)

    def load_index(self, index: int) -> None:
        target = max(0, min(len(self.rows) - 1, index))
        target_row = self.rows[target]
        if not row_eligible(target_row) and not self._row_complete(target_row):
            if self.payload:
                self._hint(f"frame not yet eligible (from {target_row.eligible_from}); staying on the current frame")
                return
            raise SystemExit(7)
        if self.payload:
            self.save_current()
        self.index = target
        row = self.rows[self.index]
        self.image = self.Image.open(row.image_path).convert("RGB")
        width, height = self.image.size
        self.payload = self.store.load(row, width, height)
        self.current_sealed = self.store.is_sealed(row)
        self.selected = None
        self.undo_stack = []
        self.last_clock = time.monotonic()
        self._frame_loaded_at = self.last_clock
        self.last_input_at = None
        self.fit_to_window("frame_load")
        if self.current_sealed:
            self._hint("SEALED frame: read-only, nothing is saved (view only)")

    def _sealed_block(self) -> bool:
        if getattr(self, "current_sealed", False):
            self._hint("sealed frame: editing disabled")
            return True
        return False

    def realized_canvas_size(self) -> tuple[int, int]:
        self.root.update_idletasks()
        canvas_width = int(self.canvas.winfo_width())
        canvas_height = int(self.canvas.winfo_height())
        if canvas_width <= 1 or canvas_height <= 1:
            raise RuntimeError(
                f"canvas was not realized: {canvas_width}x{canvas_height}"
            )
        return canvas_width, canvas_height

    def fit_to_window(self, reason: str = "manual") -> None:
        if self.image is None:
            return
        canvas_width, canvas_height = self.realized_canvas_size()
        usable_width = canvas_width - 2 * FIT_MARGIN_PIXELS
        usable_height = canvas_height - 2 * FIT_MARGIN_PIXELS
        if usable_width <= 0 or usable_height <= 0:
            raise RuntimeError(
                f"canvas is too small for the fit margin: {canvas_width}x{canvas_height}"
            )
        self.scale = min(
            usable_width / self.image.width,
            usable_height / self.image.height,
            1.0,
        )
        rendered_width = max(1, round(self.image.width * self.scale))
        rendered_height = max(1, round(self.image.height * self.scale))
        self.offset_x = (canvas_width - rendered_width) / 2
        self.offset_y = (canvas_height - rendered_height) / 2
        self.redraw()
        self.root.update_idletasks()
        self.last_viewport_report = self.assert_image_fully_visible(reason)

    def on_canvas_configure(self, event: Any) -> None:
        if self._closing or self.image is None:
            return
        new_size = (int(event.width), int(event.height))
        if new_size == self._last_canvas_size:
            return
        self._last_canvas_size = new_size
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(
            RESIZE_DEBOUNCE_MS, self.apply_configure_fit
        )

    def apply_configure_fit(self) -> None:
        self._resize_after_id = None
        if self._closing or self.image is None:
            return
        self.fit_to_window("configure")

    def image_point(self, canvas_x: float, canvas_y: float) -> tuple[float, float]:
        width, height = self.image.size
        x = min(max((canvas_x - self.offset_x) / self.scale, 0.0), float(width))
        y = min(max((canvas_y - self.offset_y) / self.scale, 0.0), float(height))
        return x, y

    def canvas_box(self, shape: dict[str, Any]) -> tuple[float, float, float, float]:
        (x1, y1), (x2, y2) = shape["points"]
        return (
            self.offset_x + x1 * self.scale,
            self.offset_y + y1 * self.scale,
            self.offset_x + x2 * self.scale,
            self.offset_y + y2 * self.scale,
        )

    def redraw(self) -> None:
        self.canvas.delete("all")
        width = max(1, round(self.image.width * self.scale))
        height = max(1, round(self.image.height * self.scale))
        resampling = getattr(self.Image, "Resampling", self.Image).LANCZOS
        rendered = self.image.resize((width, height), resampling)
        self.photo = self.ImageTk.PhotoImage(rendered)
        self.image_item_id = self.canvas.create_image(
            self.offset_x, self.offset_y, image=self.photo, anchor="nw"
        )
        for index, shape in enumerate(self.payload.get("shapes", [])):
            buried = bool((shape.get("flags") or {}).get("partially_buried"))
            color = "#ff9f1c" if buried else "#00d4ff"
            line_width = 4 if index == self.selected else 2
            self.canvas.create_rectangle(*self.canvas_box(shape), outline=color, width=line_width)
        metadata = self.payload["revision_annotation"]
        state = "COMPLETE" if metadata.get("complete") else "DRAFT"
        selected = "none" if self.selected is None else str(self.selected + 1)
        self.status.config(
            text=(
                f"[{self.index + 1}/{len(self.rows)}] {self.rows[self.index].anonymous_id} | "
                f"boxes={len(self.payload.get('shapes', []))} | selected={selected} | "
                f"{state} | completed={self.completed_count()}/{len(self.rows)}"
            )
        )

    def viewport_report(self, reason: str) -> dict[str, Any]:
        canvas_width, canvas_height = self.realized_canvas_size()
        if self.image_item_id is None:
            raise RuntimeError("rendered image item is missing from the canvas")
        bbox = self.canvas.bbox(self.image_item_id)
        if bbox is None:
            raise RuntimeError("rendered image has no canvas bounding box")
        report = viewport_containment_report(
            canvas_width,
            canvas_height,
            tuple(int(value) for value in bbox),
        )
        report.update(
            {
                "reason": reason,
                "fit_margin_pixels": FIT_MARGIN_PIXELS,
                "scale": round(self.scale, 8),
                "offset_x": round(self.offset_x, 3),
                "offset_y": round(self.offset_y, 3),
            }
        )
        return report

    def assert_image_fully_visible(self, reason: str) -> dict[str, Any]:
        report = self.viewport_report(reason)
        if not report["image_fully_visible"]:
            raise RuntimeError(
                "viewport self-check failed; the rendered image is not fully visible: "
                + json.dumps(report, sort_keys=True)
            )
        return report

    def snapshot_undo(self) -> None:
        self.undo_stack.append(deepcopy(self.payload.get("shapes", [])))
        self.undo_stack = self.undo_stack[-30:]

    def on_left_press(self, event: Any) -> None:
        if self._sealed_block():
            return
        self.mark_activity()
        self.draw_start = self.image_point(event.x, event.y)
        self.preview_id = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="#ffffff", width=2, dash=(4, 2)
        )

    def on_left_motion(self, event: Any) -> None:
        if self._sealed_block():
            return
        self.mark_activity()
        if self.preview_id is not None and self.draw_start is not None:
            x1 = self.offset_x + self.draw_start[0] * self.scale
            y1 = self.offset_y + self.draw_start[1] * self.scale
            self.canvas.coords(self.preview_id, x1, y1, event.x, event.y)

    def on_left_release(self, event: Any) -> None:
        if self._sealed_block():
            return
        self.mark_activity()
        if self.draw_start is None:
            return
        end = self.image_point(event.x, event.y)
        x1, x2 = sorted((self.draw_start[0], end[0]))
        y1, y2 = sorted((self.draw_start[1], end[1]))
        self.draw_start = None
        self.preview_id = None
        if x2 - x1 >= 3 and y2 - y1 >= 3:
            self.snapshot_undo()
            self.payload["shapes"].append(make_shape((x1, y1, x2, y2), False))
            self.payload["revision_annotation"]["complete"] = False
            self.payload["revision_annotation"]["completed_at"] = None
            self.selected = len(self.payload["shapes"]) - 1
        self.redraw()

    def on_select(self, event: Any) -> None:
        self.mark_activity()
        x, y = self.image_point(event.x, event.y)
        candidates: list[tuple[float, int]] = []
        for index, shape in enumerate(self.payload.get("shapes", [])):
            (x1, y1), (x2, y2) = shape["points"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                candidates.append(((x2 - x1) * (y2 - y1), index))
        self.selected = min(candidates)[1] if candidates else None
        self.redraw()

    def on_pan_press(self, event: Any) -> None:
        self.mark_activity()
        self.pan_start = (event.x, event.y)

    def on_pan_motion(self, event: Any) -> None:
        self.mark_activity()
        if self.pan_start is None:
            return
        dx = event.x - self.pan_start[0]
        dy = event.y - self.pan_start[1]
        self.offset_x += dx
        self.offset_y += dy
        self.pan_start = (event.x, event.y)
        self.redraw()

    def on_pan_release(self, _event: Any) -> None:
        self.mark_activity()
        self.pan_start = None

    def on_wheel(self, event: Any) -> None:
        self.mark_activity()
        self.zoom_at(event.x, event.y, 1.15 if event.delta > 0 else 1 / 1.15)

    def zoom_at(self, canvas_x: float, canvas_y: float, factor: float) -> None:
        before = self.image_point(canvas_x, canvas_y)
        new_scale = min(max(self.scale * factor, 0.15), 4.0)
        self.scale = new_scale
        self.offset_x = canvas_x - before[0] * self.scale
        self.offset_y = canvas_y - before[1] * self.scale
        self.redraw()

    def toggle_fullscreen(self) -> None:
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)
        self.root.after(RESIZE_DEBOUNCE_MS, lambda: self.fit_to_window("fullscreen"))

    def leave_fullscreen(self) -> None:
        if not self.is_fullscreen:
            return
        self.is_fullscreen = False
        self.root.attributes("-fullscreen", False)
        self.root.after(RESIZE_DEBOUNCE_MS, lambda: self.fit_to_window("leave_fullscreen"))

    def update_clock(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if getattr(self, "current_sealed", False):
            self.last_clock = now
            return  # sealed frame: timing metadata is frozen with the seal
        metadata = self.payload["revision_annotation"]
        active_delta, wall_delta = clock_increments(
            self.last_clock, now, self.last_input_at
        )
        metadata["active_seconds"] = round(
            float(metadata.get("active_seconds", 0.0)) + active_delta, 2
        )
        metadata["wall_seconds"] = round(
            float(metadata.get("wall_seconds", 0.0)) + wall_delta, 2
        )
        metadata["last_saved_at"] = iso_now()
        self.last_clock = now

    def mark_activity(self) -> None:
        now = time.monotonic()
        self.update_clock(now)
        self.last_input_at = now

    def save_current(self) -> None:
        if getattr(self, "current_sealed", False):
            return
        self.update_clock()
        self.store.save(self.rows[self.index], self.payload)

    def schedule_autosave(self) -> None:
        if self._closing:
            return
        self._autosave_after_id = self.root.after(
            self.autosave_interval_ms, self.periodic_autosave
        )

    def periodic_autosave(self) -> None:
        self._autosave_after_id = None
        if self._closing:
            return
        if self.payload:
            self.save_current()
            self.autosave_count += 1
        self.schedule_autosave()

    MIN_DWELL_SECONDS = 2.0        # a frame cannot be completed sooner than this after loading
    COMPLETE_DEBOUNCE_SECONDS = 1.0  # repeated complete requests within this window are ignored

    def _hint(self, text: str) -> None:
        try: self.status.config(text=self.status.cget("text") + "  |  " + text)
        except Exception: pass

    def on_return_release(self, _event: Any) -> None:
        # A completion is armed only by a physical key release; OS auto-repeat delivers
        # repeated KeyPress events without a KeyRelease, so a held key cannot complete twice.
        self._return_armed = True

    def complete_and_next(self) -> None:
        if getattr(self, "current_sealed", False):
            self._hint("sealed frame: cannot be completed again; moving on"); self.load_index(min(self.index + 1, len(self.rows) - 1)); return
        now = time.monotonic()
        if not getattr(self, "_return_armed", True):
            self._hint("Return ignored: release the key before completing another frame"); return
        if now - getattr(self, "_frame_loaded_at", 0.0) < self.MIN_DWELL_SECONDS:
            self._hint("Return ignored: frame loaded less than 2 s ago"); return
        if now - getattr(self, "_last_complete_at", 0.0) < self.COMPLETE_DEBOUNCE_SECONDS:
            self._hint("Return ignored: debounce"); return
        self._return_armed = False
        if not self.payload.get("shapes"):
            if not self.messagebox.askyesno(
                "Empty frame",
                "This frame has NO boxes. Mark it complete as an empty frame?\n\n"
                "Only confirm if you inspected the whole image and found no vehicle.",
                default="no", icon="warning",
            ):
                return
        self._last_complete_at = now
        self.payload["revision_annotation"]["complete"] = True
        self.payload["revision_annotation"]["completed_at"] = iso_now()
        self.save_current()
        count = self.completed_count()
        if count == self.progress_gate and not self.acknowledge_progress_gate:
            report = progress_report(self.store.output_dir, self.rows, self.daily_hours)
            self.messagebox.showinfo(
                "Progress gate reached",
                "Thirty-frame progress gate reached. The workload report was saved. "
                "Stop and report it before resuming with --acknowledge-progress-gate.\n\n"
                + json.dumps(report, indent=2),
            )
            self.destroy_app(save=False)
            return
        self.load_index(min(self.index + 1, len(self.rows) - 1))

    def delete_selected(self) -> None:
        if self._sealed_block():
            return
        if self.selected is None:
            return
        self.snapshot_undo()
        del self.payload["shapes"][self.selected]
        self.selected = None
        self.payload["revision_annotation"]["complete"] = False
        self.payload["revision_annotation"]["completed_at"] = None
        self.redraw()

    def toggle_buried(self) -> None:
        if self._sealed_block():
            return
        if self.selected is None:
            return
        self.snapshot_undo()
        shape = self.payload["shapes"][self.selected]
        value = not bool((shape.get("flags") or {}).get("partially_buried"))
        shape.setdefault("flags", {})["partially_buried"] = value
        shape.setdefault("attributes", {})["partially_buried"] = value
        self.redraw()

    def undo(self) -> None:
        if self._sealed_block():
            return
        if not self.undo_stack:
            return
        self.payload["shapes"] = self.undo_stack.pop()
        self.selected = None
        self.payload["revision_annotation"]["complete"] = False
        self.payload["revision_annotation"]["completed_at"] = None
        self.redraw()

    def edit_frame_note(self) -> None:
        if self._sealed_block():
            return
        current = str(self.payload["revision_annotation"].get("frame_note", ""))
        note = self.simpledialog.askstring(
            "Frame note",
            "Record an ambiguity or protocol issue. Do not consult old labels.",
            initialvalue=current,
            parent=self.root,
        )
        if note is not None:
            self.payload["revision_annotation"]["frame_note"] = note.strip()
            self.save_current()
            self.redraw()

    def on_key(self, event: Any) -> None:
        self.mark_activity()
        key = event.keysym.lower()
        if key in ("delete", "backspace"):
            self.delete_selected()
        elif key == "b":
            self.toggle_buried()
        elif key == "u":
            self.undo()
        elif key == "n":
            self.edit_frame_note()
        elif key in ("return", "kp_enter"):
            self.complete_and_next()
        elif key == "right":
            self.load_index(self.index + 1)
        elif key == "left":
            self.load_index(self.index - 1)
        elif key == "g":
            self.goto_first_incomplete()
        elif key == "s":
            self.save_current()
            self.redraw()
        elif key == "f":
            self.toggle_fullscreen()
        elif key in ("0", "r"):
            self.fit_to_window("keyboard_fit")
        elif key == "escape":
            self.leave_fullscreen()
        elif key == "q":
            self.request_quit()

    def request_quit(self) -> None:
        if self._closing:
            return
        confirmed = self.messagebox.askyesno(
            "Save and quit",
            "Save the current draft and quit the annotation tool?",
            parent=self.root,
        )
        if confirmed:
            self.destroy_app(save=True)

    def destroy_app(self, save: bool = True) -> None:
        if self._closing:
            return
        self._closing = True
        if save and self.payload:
            self.save_current()
        for after_id in (self._autosave_after_id, self._resize_after_id):
            if after_id is not None:
                try:
                    self.root.after_cancel(after_id)
                except self.tk.TclError:
                    pass
        self.root.destroy()

    def validation_close(self) -> None:
        self.destroy_app(save=True)

    def run(self) -> None:
        self.root.mainloop()

    def wait_for_tk_events(self, delay_ms: int) -> None:
        completed = self.tk.BooleanVar(value=False)
        self.root.after(delay_ms, lambda: completed.set(True))
        self.root.wait_variable(completed)

    def validate_configure_resize(self) -> dict[str, Any]:
        original_width = int(self.root.winfo_width())
        original_height = int(self.root.winfo_height())
        original_x = int(self.root.winfo_x())
        original_y = int(self.root.winfo_y())
        target_width = max(
            int(self.root.minsize()[0]), round(original_width * 0.80)
        )
        target_height = max(
            int(self.root.minsize()[1]), round(original_height * 0.80)
        )
        self.root.geometry(
            f"{target_width}x{target_height}+{original_x}+{original_y}"
        )
        self.root.update_idletasks()
        self.wait_for_tk_events(RESIZE_DEBOUNCE_MS + 80)
        resized_report = self.assert_image_fully_visible(
            "configure_resize_validation"
        )

        self.root.geometry(
            f"{original_width}x{original_height}+{original_x}+{original_y}"
        )
        self.root.update_idletasks()
        self.wait_for_tk_events(RESIZE_DEBOUNCE_MS + 80)
        restored_report = self.assert_image_fully_visible(
            "configure_restore_validation"
        )
        return {
            "requested_window_size": [target_width, target_height],
            "resized_viewport": resized_report,
            "restored_window_size": [original_width, original_height],
            "restored_viewport": restored_report,
            "configure_refit_passed": (
                resized_report["image_fully_visible"]
                and restored_report["image_fully_visible"]
            ),
        }

    def capture_render_evidence(self, artifact_dir: Path) -> None:
        from PIL import ImageStat

        artifact_dir.mkdir(parents=True, exist_ok=True)
        self.root.update_idletasks()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.update()
        configure_resize_report = self.validate_configure_resize()
        self.fit_to_window("render_validation")
        capture_viewport_report = self.assert_image_fully_visible(
            "render_validation_capture"
        )

        root_w = self.root.winfo_width()
        root_h = self.root.winfo_height()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        image_items = [
            item for item in self.canvas.find_all() if self.canvas.type(item) == "image"
        ]
        if len(image_items) != 1 or self.photo is None:
            raise AssertionError("Tk canvas does not contain exactly one rendered image")
        photo_width_fits = self.photo.width() <= canvas_w
        photo_height_fits = self.photo.height() <= canvas_h
        bbox_matches_photo = (
            capture_viewport_report["image_width"] == self.photo.width()
            and capture_viewport_report["image_height"] == self.photo.height()
        )
        intersection_is_complete = (
            capture_viewport_report["image_canvas_intersection_ratio"] == 1.0
        )
        if not (
            photo_width_fits
            and photo_height_fits
            and bbox_matches_photo
            and intersection_is_complete
            and capture_viewport_report["image_fully_visible"]
        ):
            raise AssertionError(
                "rendered image does not satisfy the complete-visibility predicate"
            )
        if self.autosave_count < 1:
            raise AssertionError("periodic autosave did not run before render capture")
        autosave_output_path = self.store.output_path(self.rows[self.index])
        if not autosave_output_path.is_file():
            raise AssertionError("periodic autosave did not create a draft output")
        with autosave_output_path.open(encoding="utf-8") as handle:
            autosaved_payload = json.load(handle)
        autosave_probe = (autosaved_payload.get("revision_annotation") or {}).get(
            "frame_note"
        )
        if autosave_probe != "viewport_autosave_validation_probe":
            raise AssertionError("periodic autosave did not persist the validation probe")
        wm_delete_handler = self.root.protocol("WM_DELETE_WINDOW")
        if not wm_delete_handler:
            raise AssertionError("WM_DELETE_WINDOW has no save-confirmation handler")

        postscript_path = artifact_dir / "canvas_snapshot.ps"
        tk_photo_path = artifact_dir / "tk_photo_buffer.png"
        self.photo._PhotoImage__photo.write(str(tk_photo_path), format="png")
        screenshot_path = artifact_dir / "window_screenshot.png"
        self.canvas.postscript(
            file=str(postscript_path),
            colormode="color",
            x=0,
            y=0,
            width=canvas_w,
            height=canvas_h,
            pagewidth=f"{canvas_w}p",
            pageheight=f"{canvas_h}p",
            rotate=False,
        )
        capture = subprocess.run(
            [
                "/usr/local/bin/gs",
                "-dSAFER",
                "-dBATCH",
                "-dNOPAUSE",
                "-sDEVICE=png16m",
                "-r144",
                "-dEPSCrop",
                f"-sOutputFile={screenshot_path}",
                str(postscript_path),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if capture.returncode != 0 or not screenshot_path.is_file():
            raise RuntimeError(
                "Tk canvas raster capture failed; install Ghostscript and replay the launcher command: "
                + capture.stderr.strip()
            )

        with self.Image.open(screenshot_path) as screenshot:
            screenshot = screenshot.convert("RGB")
            canvas_path = artifact_dir / "canvas_crop.png"
            screenshot.save(canvas_path)
            channel_variances = ImageStat.Stat(screenshot).var
            pixel_variance = statistics.fmean(channel_variances)

        if pixel_variance <= RENDER_VARIANCE_MINIMUM:
            raise AssertionError(
                f"canvas pixel variance {pixel_variance:.3f} does not prove image rendering"
            )
        render_assertion_passed = (
            len(image_items) == 1
            and photo_width_fits
            and photo_height_fits
            and bbox_matches_photo
            and intersection_is_complete
            and capture_viewport_report["image_fully_visible"]
            and configure_resize_report["configure_refit_passed"]
            and self.autosave_count >= 1
            and pixel_variance > RENDER_VARIANCE_MINIMUM
        )
        if not render_assertion_passed:
            raise AssertionError("combined render assertion failed")

        import tkinter
        import PIL

        evidence = {
            "generated_at": iso_now(),
            "launcher": os.environ.get("SNOW_REVIEW_LAUNCHER_PATH"),
            "launcher_selected_python": os.environ.get(
                "SNOW_REVIEW_SELECTED_PYTHON"
            ),
            "sys_executable": sys.executable,
            "python_version": platform.python_version(),
            "tk_version": str(tkinter.TkVersion),
            "pillow_version": PIL.__version__,
            "tool_version": self.store.tool_version,
            "label_set_id": self.rows[self.index].label_set_id,
            "capture_method": "Tk PhotoImage buffer plus Canvas.postscript after root.update, rasterized by Ghostscript",
            "tk_canvas_image_item_count": len(image_items),
            "tk_photo_width": self.photo.width(),
            "tk_photo_height": self.photo.height(),
            "tk_root_width": root_w,
            "tk_root_height": root_h,
            "tk_canvas_width": canvas_w,
            "tk_canvas_height": canvas_h,
            "photo_width_fits_canvas": photo_width_fits,
            "photo_height_fits_canvas": photo_height_fits,
            "image_bbox_matches_photo_dimensions": bbox_matches_photo,
            "image_canvas_intersection_ratio": capture_viewport_report[
                "image_canvas_intersection_ratio"
            ],
            "initial_adaptive_geometry": self.initial_geometry,
            "startup_viewport_check": self.startup_viewport_report,
            "configure_resize_check": configure_resize_report,
            "capture_viewport_check": capture_viewport_report,
            "image_fully_visible": capture_viewport_report["image_fully_visible"],
            "image_corners": capture_viewport_report["image_corners"],
            "corner_visibility": capture_viewport_report["corner_visibility"],
            "autosave_interval_ms_under_validation": self.autosave_interval_ms,
            "autosave_count_before_capture": self.autosave_count,
            "autosave_output_exists": True,
            "autosave_probe_persisted": True,
            "formal_autosave_interval_ms": AUTOSAVE_INTERVAL_MS,
            "wm_delete_window_handler_bound": True,
            "quit_key": "q_with_save_confirmation",
            "escape_key": "leave_fullscreen_or_noop",
            "fullscreen_key": "f",
            "fit_keys": ["0", "r"],
            "window_screenshot": str(screenshot_path),
            "tk_photo_buffer": str(tk_photo_path),
            "canvas_crop": str(canvas_path),
            "canvas_channel_variances": [round(value, 3) for value in channel_variances],
            "canvas_pixel_variance": round(pixel_variance, 3),
            "required_pixel_variance": RENDER_VARIANCE_MINIMUM,
            "render_assertion_passed": render_assertion_passed,
        }
        atomic_json_write(artifact_dir / "render_evidence.json", evidence)
        print(json.dumps(evidence, indent=2))
        self.validation_close()


def run_sealed_guard_test(args: argparse.Namespace, rows: list[ManifestRow], manifest_hash: str) -> None:
    """Isolated proof that sealed frames cannot be changed by the GUI (tools 1.0.10). Requires an EMPTY
    scratch --output-dir: builds 3 completed frames there, seals the first two, then drives the real app
    through every edit/save path on the sealed frames and one normal save on the unsealed frame."""
    # 1.0.11: the fixture is built in an internal temporary directory; the requested --output-dir is never
    # written (its listing is asserted unchanged), so this entry point cannot pollute a formal directory.
    import tempfile
    requested = args.output_dir
    req_before = sorted(p.name for p in requested.glob("*")) if requested.exists() else None
    out = Path(tempfile.mkdtemp(prefix="snow_review_sealed_guard_"))
    from PIL import Image
    fixture = AnnotationStore(out, args.protocol, manifest_hash, addendum_path=args.addendum, instrumentation_addendum_path=args.instrumentation_addendum)
    sizes = {}
    for row in rows[:3]:
        with Image.open(row.image_path) as image:
            sizes[row.anonymous_id] = image.size
        payload = fixture.blank(row, *sizes[row.anonymous_id]); payload["shapes"].append(make_shape((100, 100, 200, 200), False))
        meta = payload["revision_annotation"]; meta["active_seconds"] = 5.0; meta["wall_seconds"] = 5.0; meta["last_saved_at"] = iso_now(); meta["completed_at"] = iso_now(); meta["complete"] = True
        fixture.save(row, payload)
    sealed_rows = rows[:2]; seal_list = out / "guard_test.sha256"
    seal_list.write_text("".join(f"{sha256_file(fixture.output_path(r))}  {fixture.output_path(r).name}\n" for r in sealed_rows))
    sealed = load_sealed_lists([seal_list]); before = {n: sha256_file(out / n) for n in sealed}
    store = AnnotationStore(out, args.protocol, manifest_hash, addendum_path=args.addendum, instrumentation_addendum_path=args.instrumentation_addendum, sealed=sealed)
    app = BlindDetectionApp(rows[:3], store, args.protocol, args.progress_gate, True, args.daily_hours)
    result = {"sealed_unchanged": None, "unsealed_saved": None, "store_refused_writes": 0}

    class Ev:
        def __init__(self, x, y): self.x = x; self.y = y

    def probe():
        for i in range(2):
            app.load_index(i); app._frame_loaded_at = 0.0; app._return_armed = True
            app.on_left_press(Ev(50, 50)); app.on_left_motion(Ev(120, 120)); app.on_left_release(Ev(120, 120))
            app.selected = 0 if app.payload.get("shapes") else None; app.toggle_buried(); app.delete_selected(); app.undo()
            app.mark_activity(); app.periodic_autosave(); app.save_current(); app.complete_and_next()
            result["store_refused_writes"] += int(not store.save(rows[i], app.payload) if store.is_sealed(rows[i]) else 0)
        app.load_index(2); app.payload["shapes"].append(make_shape((300, 300, 400, 400), True)); app.save_current()
        app.destroy_app(save=True)

    app.root.after(800, probe); app.run()
    after = {n: sha256_file(out / n) for n in sealed}; result["sealed_unchanged"] = after == before
    result["unsealed_saved"] = len(json.loads(store.output_path(rows[2]).read_text())["shapes"]) == 2
    req_after = sorted(p.name for p in requested.glob("*")) if requested.exists() else None
    result["requested_dir_untouched"] = req_before == req_after
    ok = result["sealed_unchanged"] and result["unsealed_saved"] and result["store_refused_writes"] == 2 and not store.verify_sealed() and result["requested_dir_untouched"]
    print(("SEALED GUARD TEST PASSED " if ok else "SEALED GUARD TEST FAILED ") + json.dumps(result) + f"; fixture in {out}; requested output directory untouched")
    raise SystemExit(0 if ok else 7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)  # tools 1.0.14: no option-prefix abbreviations (launcher whitelists cannot be bypassed)
    parser.add_argument("--sealed-list", type=Path, nargs="*", default=[], help="seal lists (sha256  name) whose frames are read-only; lists matching the manifest's label set in --seals-dir are added automatically")
    parser.add_argument("--seals-dir", type=Path, default=Path(__file__).resolve().parent / "seals")
    parser.add_argument("--sealed-guard-test", action="store_true", help="validation mode on an EMPTY scratch --output-dir: prove sealed frames cannot be changed")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--addendum", type=Path)
    parser.add_argument("--instrumentation-addendum", type=Path)
    parser.add_argument("--verify-image-hashes", action="store_true")
    parser.add_argument("--progress-gate", type=int, default=30)
    parser.add_argument("--acknowledge-progress-gate", action="store_true")
    parser.add_argument("--daily-hours", type=float, default=2.0)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--allow-foreign-label-sets", action="store_true", help="permit an output directory that already holds outputs of another label set (shared audit directory only)")
    parser.add_argument("--eligibility-self-test", action="store_true", help="validation mode: exercise the eligible_from logic on synthetic rows; opens nothing, writes nothing")
    parser.add_argument("--self-test-count", type=int, default=3)
    parser.add_argument("--ui-smoke-test", action="store_true")
    parser.add_argument("--timing-self-test", action="store_true")
    parser.add_argument("--timing-test-output", type=Path)
    parser.add_argument("--render-validation-dir", type=Path)
    parser.add_argument("--render-validation-delay-ms", type=int, default=1800)
    return parser.parse_args()



from output_lock import acquire_output_lock, release_output_lock  # flock-based single-instance lock (tools 1.0.8+)


def main() -> None:
    args = parse_args()
    rows, manifest_hash = load_manifest(args.manifest)
    if args.timing_self_test:
        if not args.timing_test_output:
            raise ValueError("--timing-self-test requires --timing-test-output")
        run_timing_self_test(args.timing_test_output)
        return
    if args.verify_image_hashes:
        for row in rows:
            actual = sha256_file(row.image_path)
            if actual != row.image_sha256:
                raise ValueError(f"image hash mismatch: {row.anonymous_id}")
    if args.eligibility_self_test:
        run_eligibility_self_test()
        return
    if args.self_test:
        # Self-test never touches the requested output directory (tools 1.0.9): it writes synthetic,
        # completed frames, so it runs in a fresh temporary directory and proves the formal one untouched.
        import tempfile
        before = sorted(p.name for p in args.output_dir.glob("*")) if args.output_dir.exists() else None
        selftest_dir = Path(tempfile.mkdtemp(prefix="snow_review_selftest_"))
        run_self_test(
            args.manifest,
            selftest_dir,
            args.protocol,
            args.self_test_count,
            addendum_path=args.addendum,
            instrumentation_addendum_path=args.instrumentation_addendum,
        )
        after = sorted(p.name for p in args.output_dir.glob("*")) if args.output_dir.exists() else None
        if before != after:
            raise AssertionError(f"self-test changed the requested output directory {args.output_dir}")
        print(f"self-test outputs isolated in {selftest_dir}; requested output directory untouched")
        return
    if args.sealed_guard_test:
        run_sealed_guard_test(args, rows, manifest_hash)
        return
    # Validation modes must never write into the formal output directory:
    # the autosave probe is persisted through the store, so redirect it.
    store_output_dir = args.output_dir
    if args.render_validation_dir:
        store_output_dir = args.render_validation_dir / "probe_outputs"
    elif args.ui_smoke_test:
        import tempfile
        store_output_dir = Path(tempfile.mkdtemp(prefix="snow_review_uismoke_"))
    lock = acquire_output_lock(store_output_dir, "blind_detection_tool")
    try:
        _main_locked(args, rows, manifest_hash, store_output_dir)
    finally:
        release_output_lock(lock)


def foreign_label_sets(output_dir: Path, label_set_id: str) -> dict[str, int]:
    """Label sets other than label_set_id found in existing outputs of output_dir (tools 1.0.13)."""
    found: dict[str, int] = {}
    if not output_dir.exists():
        return found
    for path in sorted(output_dir.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                other = (json.load(handle).get("revision_annotation") or {}).get("label_set_id")
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
        if other and other != label_set_id:
            found[other] = found.get(other, 0) + 1
    return found


def _main_locked(args, rows, manifest_hash, store_output_dir):
    formal = not (args.render_validation_dir or args.ui_smoke_test)
    if formal and not args.allow_foreign_label_sets:
        foreign = foreign_label_sets(store_output_dir, rows[0].label_set_id)
        if foreign:
            sys.stderr.write(f"FOREIGN LABEL SET: {store_output_dir} already holds outputs of {sorted(foreign)}; refusing to open manifest label set {rows[0].label_set_id} there\n")
            raise SystemExit(8)
    seal_paths = list(args.sealed_list) + (discover_seal_lists(args.seals_dir, rows) if formal else [])
    sealed = load_sealed_lists(sorted(set(seal_paths))) if formal else {}
    store = AnnotationStore(
        store_output_dir,
        args.protocol,
        manifest_hash,
        addendum_path=args.addendum,
        instrumentation_addendum_path=args.instrumentation_addendum,
        sealed=sealed,
    )
    if sealed:
        bad = store.verify_sealed()
        if bad:
            sys.stderr.write(f"SEAL MISMATCH at startup: {len(bad)} sealed files differ from the seal lists: {bad[:5]}\n"); raise SystemExit(5)
        print(f"sealed frames loaded read-only: {len(sealed)} from {len(set(seal_paths))} seal list(s)")
    app = BlindDetectionApp(
        rows,
        store,
        args.protocol,
        args.progress_gate,
        args.acknowledge_progress_gate,
        args.daily_hours,
        autosave_interval_ms=(
            RENDER_VALIDATION_AUTOSAVE_INTERVAL_MS
            if args.render_validation_dir
            else AUTOSAVE_INTERVAL_MS
        ),
    )
    if args.ui_smoke_test and args.render_validation_dir:
        raise ValueError("choose either --ui-smoke-test or --render-validation-dir")
    if not (args.ui_smoke_test or args.render_validation_dir):
        done = app.completed_count()
        summary = eligibility_summary(rows, {r.anonymous_id for r in rows if app._row_complete(r)})
        if summary["pending"] and not summary["eligible_pending"]:
            msg = (f"none of the {summary['pending']} remaining frame(s) is eligible yet (14-day test-retest rule); "
                   f"next frame becomes eligible at {summary['next_eligible_at']}")
            if os.environ.get("SNOW_REVIEW_NO_DIALOG"): sys.stderr.write("ELIGIBILITY GATE: " + msg + "\n")
            else: app.messagebox.showinfo("Not yet eligible", msg)
            app.root.destroy(); raise SystemExit(7)
        if summary["blocked_pending"]:
            print(f"eligibility: {summary['eligible_pending']} eligible, {summary['blocked_pending']} blocked until {summary['next_eligible_at']} (earliest)")
        if done >= args.progress_gate and not args.acknowledge_progress_gate:
            msg = f"{done} frames are complete; the {args.progress_gate}-frame gate requires the report to be reviewed before continuing. Restart with --acknowledge-progress-gate."
            if os.environ.get("SNOW_REVIEW_NO_DIALOG"): sys.stderr.write("PROGRESS GATE: " + msg + "\n")
            else: app.messagebox.showerror("Progress gate", msg)
            app.root.destroy(); raise SystemExit(3)
    if args.ui_smoke_test:
        app.root.after(1200, app.validation_close)
    if args.render_validation_dir:
        app.payload["revision_annotation"][
            "frame_note"
        ] = "viewport_autosave_validation_probe"
        app.root.after(
            args.render_validation_delay_ms,
            lambda: app.capture_render_evidence(args.render_validation_dir),
        )
    app.run()
    if sealed:
        bad = store.verify_sealed()
        if bad:
            sys.stderr.write(f"SEAL MISMATCH at exit: {bad[:5]}\n"); raise SystemExit(6)
        print(f"sealed frames verified unchanged at exit: {len(sealed)}")
    if args.ui_smoke_test:
        print("UI smoke test passed")


if __name__ == "__main__":
    main()
