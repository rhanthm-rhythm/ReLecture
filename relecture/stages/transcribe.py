from __future__ import annotations

import logging
from dataclasses import replace

from ..config import load_config
from ..media import process_video_segments
from ..models import ArtifactRef, Segment, StageStatus, TRANSCRIPTION_SCHEMA_VERSION, TranscriptionManifest
from ..services import TranscriptionService, TransformationService
from ..storage import (
    ensure_pipeline_layout,
    ensure_project_manifest,
    load_stage_manifest,
    project_paths,
    save_stage_manifest,
)
from ..utils import normalize_segments, now_iso, resolve_project_path, to_relative

logger = logging.getLogger(__name__)


def _extract_audio_assets(source_video: str, segments: list[Segment], audio_dir: str) -> list[Segment]:
    return process_video_segments(
        source_video,
        segments,
        output_dir=audio_dir,
        extract_video=False,
        extract_audio=True,
        extract_images=False,
    )


def run_transcription(
    project_file: str,
    *,
    language: str = "en",
    transcription_service: TranscriptionService | None = None,
    transformation_service: TransformationService | None = None,
    progress_cb=None,  # optional callable(fraction: float, desc: str)
) -> TranscriptionManifest:
    project = ensure_project_manifest(project_file)
    paths = project_paths(project_file)
    ensure_pipeline_layout(paths)
    segmentation = load_stage_manifest(project, project_file, "segmentation")
    source_video = resolve_project_path(paths.project_dir, project.source_video)
    if source_video is None:
        raise FileNotFoundError("Project source video is missing.")

    started_at = now_iso()
    segments = _extract_audio_assets(source_video, segmentation.segments, paths.audio_dir)

    language = language or "en"

    if transcription_service is None:
        transcription_service = TranscriptionService(project.models.whisper_endpoint or "http://localhost:5001/transcribe")
    if transformation_service is None:
        transformation_service = TransformationService.from_config()

    total = max(len(segments), 1)
    transcribed_pairs: list[tuple[Segment, str, str]] = []
    for index, segment in enumerate(segments, start=1):
        if progress_cb:
            progress_cb((index - 1) / total, f"Transcribing segment {index}/{total}")
        if not segment.audio:
            raise RuntimeError(f"Missing extracted audio for segment {segment.id}")
        audio_path = resolve_project_path(paths.project_dir, segment.audio.path)
        raw = transcription_service.transcribe(audio_path, language=language).strip()
        preview_text = raw[:50] + "..." if len(raw) > 50 else (raw or "<silent>")
        logger.info(f"Segment {segment.id}/{len(segments)} transcribed: '{preview_text}'")
        transcribed_pairs.append((segment, audio_path, raw))
        if progress_cb:
            progress_cb(index / total, f"Transcribed segment {index}/{total}")

    # Two-layer math sanitization:
    # Layer 1: Batched LLM assessment over all segments in a single prompt to identify which IDs require sanitization
    items_to_assess = [(seg.id, raw) for seg, _, raw in transcribed_pairs if raw]
    flagged_ids: set[int] = set()
    if items_to_assess:
        if hasattr(transformation_service, "identify_segments_needing_sanitization"):
            res = transformation_service.identify_segments_needing_sanitization(items_to_assess)
            if isinstance(res, (set, list, tuple)):
                flagged_ids = set(res)
            else:
                # If mock returns a non-collection in tests, fallback to all non-empty
                flagged_ids = {seg_id for seg_id, _ in items_to_assess}
        else:
            flagged_ids = {seg_id for seg_id, _ in items_to_assess}

    logger.info(
        "Math sanitization assessment: %d/%d segments flagged for speech sanitization: %s",
        len(flagged_ids),
        len(transcribed_pairs),
        sorted(flagged_ids),
    )

    # Layer 2: Targeted sanitization for only the flagged segments
    updated: list[Segment] = []
    for segment, audio_path, raw in transcribed_pairs:
        if not raw:
            clean = ""
        elif segment.id in flagged_ids:
            logger.info("Sanitizing math for segment %d...", segment.id)
            clean = transformation_service.sanitize_for_speech(raw).strip()
        else:
            clean = raw

        audio_ref = ArtifactRef(kind="audio", path=to_relative(paths.project_dir, audio_path))
        updated.append(
            replace(
                segment,
                audio=audio_ref,
                raw_transcript=raw or None,
                clean_transcript=clean or raw or None,
            )
        )

    logger.info(f"Transcription completed: {len(updated)} segments transcribed.")

    manifest = TranscriptionManifest(
        schema_version=TRANSCRIPTION_SCHEMA_VERSION,
        status=StageStatus(
            name="transcription",
            state="completed",
            started_at=started_at,
            completed_at=now_iso(),
        ),
        source_video=project.source_video,
        segments=normalize_segments(updated),
    )
    save_stage_manifest(project, project_file, "transcription", manifest)
    return manifest
