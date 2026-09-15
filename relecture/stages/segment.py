from __future__ import annotations

from pathlib import Path

from ..logger import get_run_logger
from ..models import SEGMENTATION_SCHEMA_VERSION, SegmentationManifest, Segment, StageStatus
from ..storage import ensure_pipeline_layout, ensure_project_manifest, project_paths, save_stage_manifest
from ..utils import normalize_segments, now_iso, resolve_project_path

logger = get_run_logger("relecture.segment")


def run_segmentation(
    project_file: str,
    *,
    source_video: str | None = None,
    project_name: str | None = None,
    detector=None,
    progress_cb=None,  # optional callable(fraction: float, desc: str)
) -> SegmentationManifest:
    if detector is None:
        from ..media import detect_slide_boundaries

        detector = detect_slide_boundaries
    project = ensure_project_manifest(project_file, source_video=source_video, project_name=project_name)
    paths = project_paths(project_file)
    ensure_pipeline_layout(paths)

    source_path = resolve_project_path(paths.project_dir, project.source_video)
    if source_path is None or not Path(source_path).exists():
        raise FileNotFoundError(f"Source video not found: {source_path}")

    logger.info(f"Starting segmentation for video: {source_path}")
    started_at = now_iso()
    if progress_cb:
        progress_cb(0.1, "Detecting slide boundaries…")
    raw_segments = detector(source_path)
    segments = normalize_segments(
        [
            Segment(
                id=int(item.get("id", index)),
                start_time=float(item.get("start", 0.0)),
                end_time=float(item.get("end", 0.0)),
                duration=float(item.get("end", 0.0)) - float(item.get("start", 0.0)),
                confidence=float(item.get("confidence", 0.0)),
            )
            for index, item in enumerate(raw_segments, start=1)
        ]
    )

    manifest = SegmentationManifest(
        schema_version=SEGMENTATION_SCHEMA_VERSION,
        status=StageStatus(
            name="segmentation",
            state="completed",
            started_at=started_at,
            completed_at=now_iso(),
        ),
        source_video=project.source_video,
        segments=segments,
    )
    save_stage_manifest(project, project_file, "segmentation", manifest)
    logger.info(f"Segmentation completed: {len(segments)} segments detected.")
    if progress_cb:
        progress_cb(1.0, f"{len(segments)} segments detected")
    return manifest
