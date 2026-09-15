from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .models import Segment

MIN_SEGMENT_DURATION = 0.5


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: str | Path) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def to_relative(base_dir: str | Path, target: str | Path | None) -> str | None:
    if target is None:
        return None
    base = Path(base_dir).resolve()
    path = Path(target).resolve()
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return str(path)


def resolve_project_path(base_dir: str | Path, path: str | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((Path(base_dir) / candidate).resolve())


def normalize_segments(segments: list[Segment]) -> list[Segment]:
    normalized: list[Segment] = []
    for idx, segment in enumerate(sorted(segments, key=lambda item: item.start_time), start=1):
        start = float(segment.start_time)
        end = float(segment.end_time)
        if end <= start + MIN_SEGMENT_DURATION:
            end = start + MIN_SEGMENT_DURATION
        normalized.append(
            replace(
                segment,
                id=idx,
                start_time=start,
                end_time=end,
                duration=end - start,
                confidence=float(segment.confidence),
            )
        )
    return normalized


def parse_segment_selector(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    selected: set[int] = set()
    for token in [item.strip() for item in raw.split(",") if item.strip()]:
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            step = 1 if start <= end else -1
            for value in range(start, end + step, step):
                selected.add(value)
        else:
            selected.add(int(token))
    return sorted(selected)


def filter_segments(segments: list[Segment], selected_ids: list[int] | None) -> list[Segment]:
    if selected_ids is None:
        return list(segments)
    allowed = set(selected_ids)
    filtered = [segment for segment in segments if segment.id in allowed]
    missing = [segment_id for segment_id in selected_ids if segment_id not in {segment.id for segment in filtered}]
    if missing:
        raise ValueError(f"Segments not found: {missing}")
    return filtered


def apply_segment_adjustments(segments: list[Segment], adjustments: list[dict[str, float | int | str]]) -> list[Segment]:
    working = [replace(segment) for segment in segments]
    index = {segment.id: idx for idx, segment in enumerate(working)}
    for adjustment in adjustments:
        segment_id = int(adjustment["segment"])
        if segment_id not in index:
            raise ValueError(f"Segment {segment_id} not found")
        segment = working[index[segment_id]]
        kind = adjustment["type"]
        if kind == "range":
            start = float(adjustment["start"])
            end = float(adjustment["end"])
        elif kind == "start":
            start = float(adjustment["value"])
            end = float(segment.end_time)
        elif kind == "end":
            start = float(segment.start_time)
            end = float(adjustment["value"])
        else:
            raise ValueError(f"Unknown adjustment type: {kind}")
        working[index[segment_id]] = replace(segment, start_time=start, end_time=end, duration=end - start)
    return normalize_segments(working)


def format_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
