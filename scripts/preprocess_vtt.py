from __future__ import annotations

"""Align VTT subtitles from Underline transcripts with pipeline segment boundaries.

Usage:
    python scripts/preprocess_vtt.py \
        --segmentation-manifest path/to/manifests/segmentation.v1.json \
        --vtt-file path/to/transcript.vtt \
        --output path/to/ground_truth_transcripts.json
"""

import argparse
import json
import re
import sys
from pathlib import Path


VTT_TIMESTAMP_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)


def parse_timestamp(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(vtt_path: str) -> list[tuple[float, float, str]]:
    """Return list of (start_sec, end_sec, text) from a WebVTT file."""
    cues: list[tuple[float, float, str]] = []
    current_start: float | None = None
    current_end: float | None = None
    current_lines: list[str] = []

    def flush():
        nonlocal current_start, current_end, current_lines
        if current_start is not None and current_lines:
            text = " ".join(" ".join(current_lines).split())
            text = re.sub(r"<[^>]+>", "", text).strip()
            if text:
                cues.append((current_start, current_end, text))
        current_start = None
        current_end = None
        current_lines = []

    for line in Path(vtt_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
            flush()
            continue
        match = VTT_TIMESTAMP_RE.match(line)
        if match:
            flush()
            current_start = parse_timestamp(*match.group(1, 2, 3, 4))
            current_end = parse_timestamp(*match.group(5, 6, 7, 8))
            continue
        if current_start is not None:
            current_lines.append(line)
    flush()
    return cues


def align_to_segments(
    cues: list[tuple[float, float, str]],
    segments: list[dict],
) -> dict[str, str]:
    """For each segment, concatenate all VTT cues that overlap with it."""
    result: dict[str, str] = {}
    for segment in segments:
        seg_id = str(segment["id"])
        seg_start = float(segment["start_time"])
        seg_end = float(segment["end_time"])
        overlapping = []
        for cue_start, cue_end, text in cues:
            if cue_start < seg_end and cue_end > seg_start:
                overlapping.append(text)
        result[seg_id] = " ".join(overlapping).strip()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Align VTT subtitles to pipeline segment boundaries.")
    parser.add_argument("--segmentation-manifest", required=True)
    parser.add_argument("--vtt-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    seg_manifest = json.loads(Path(args.segmentation_manifest).read_text(encoding="utf-8"))
    segments = seg_manifest.get("segments", [])
    if not segments:
        print("No segments found in manifest.", file=sys.stderr)
        return 1

    cues = parse_vtt(args.vtt_file)
    aligned = align_to_segments(cues, segments)

    Path(args.output).write_text(json.dumps(aligned, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Aligned {len(aligned)} segments -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
