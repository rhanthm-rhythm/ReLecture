from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..media import check_ffmpeg_available
from ..models import ASSEMBLY_SCHEMA_VERSION, ArtifactRef, AssemblyManifest, StageStatus
from ..storage import (
    ensure_pipeline_layout,
    ensure_project_manifest,
    load_stage_manifest,
    project_paths,
    save_stage_manifest,
)
from ..utils import filter_segments, format_srt_time, now_iso, parse_segment_selector, resolve_project_path, to_relative


def _probe_duration(video_path: str) -> float:
    cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def run_assembly(
    project_file: str,
    *,
    segment_ids: str | None = None,
    output_filename: str = "lecture_final.mp4",
    generate_subtitles: bool = True,
    ffmpeg_available_checker=None,
    progress_cb=None,  # optional callable(fraction: float, desc: str)
) -> AssemblyManifest:
    if ffmpeg_available_checker is None:
        ffmpeg_available_checker = check_ffmpeg_available

    project = ensure_project_manifest(project_file)
    paths = project_paths(project_file)
    ensure_pipeline_layout(paths)
    synthesis = load_stage_manifest(project, project_file, "synthesis")
    if synthesis.mode == "preview":
        raise RuntimeError("Cannot assemble from preview-only synthesis output. Run synthesis with --mode full.")
    if not ffmpeg_available_checker():
        raise RuntimeError("FFmpeg is not available.")

    selected_segments = filter_segments(synthesis.segments, parse_segment_selector(segment_ids))
    selected_ids = {segment.id for segment in selected_segments}
    selected_results = [result for result in synthesis.results if result.segment_id in selected_ids]
    if not selected_results:
        raise RuntimeError("No synthesized outputs matched the requested segments.")
    if any(result.video is None for result in selected_results):
        missing = [result.segment_id for result in selected_results if result.video is None]
        raise RuntimeError(f"Synthesized videos missing for segments: {missing}")

    output_path = os.path.join(paths.final_dir, output_filename if output_filename.endswith(".mp4") else f"{output_filename}.mp4")
    concat_list = os.path.join(paths.final_dir, "concat_list.txt")
    started_at = now_iso()

    with open(concat_list, "w", encoding="utf-8") as handle:
        for result in sorted(selected_results, key=lambda item: item.segment_id):
            video_path = resolve_project_path(paths.project_dir, result.video.path)
            normalized_video_path = video_path.replace("\\", "/")
            handle.write(f"file '{normalized_video_path}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_list,
        "-c",
        "copy",
        output_path,
    ]
    if progress_cb:
        progress_cb(0.3, "Concatenating segments…")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")

    subtitle_ref = None
    if generate_subtitles:
        subtitle_path = os.path.join(paths.final_dir, Path(output_path).with_suffix(".srt").name)
        lines: list[str] = []
        cursor = 0.0
        synthesis_segments = {segment.id: segment for segment in synthesis.segments}
        for index, synth_result in enumerate(sorted(selected_results, key=lambda item: item.segment_id), start=1):
            segment = synthesis_segments[synth_result.segment_id]
            lecture_text = segment.transformed_transcript or segment.clean_transcript or segment.raw_transcript or ""
            narration_text = (segment.visual_narration or "").strip()
            text = f"[Slide description] {narration_text}\n{lecture_text}" if narration_text else lecture_text
            start = format_srt_time(cursor)
            end = format_srt_time(cursor + synth_result.duration_seconds)
            lines.extend([str(index), f"{start} --> {end}", text, ""])
            cursor += synth_result.duration_seconds
        with open(subtitle_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        subtitle_ref = ArtifactRef(kind="subtitle", path=to_relative(paths.project_dir, subtitle_path))

    duration = _probe_duration(output_path)
    manifest = AssemblyManifest(
        schema_version=ASSEMBLY_SCHEMA_VERSION,
        status=StageStatus(
            name="assembly",
            state="completed",
            started_at=started_at,
            completed_at=now_iso(),
        ),
        source_video=project.source_video,
        output_video=ArtifactRef(kind="video", path=to_relative(paths.project_dir, output_path)),
        subtitles=subtitle_ref,
        included_segments=sorted(selected_ids),
        duration_seconds=duration,
    )
    save_stage_manifest(project, project_file, "assembly", manifest)
    if progress_cb:
        progress_cb(1.0, "Assembly complete")
    return manifest
