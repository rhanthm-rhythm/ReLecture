from __future__ import annotations

from pathlib import Path
from dataclasses import replace

from ..config import load_config
from ..media import process_video_segments
from ..models import (
    ArtifactRef,
    Segment,
    StageStatus,
    TRANSFORMATION_SCHEMA_VERSION,
    VISION_SCHEMA_VERSION,
    TransformationManifest,
    VisionManifest,
    VisionResult,
)
from ..logger import get_run_logger
from ..services import TransformationService, VisionService
from ..storage import (
    ensure_pipeline_layout,
    ensure_project_manifest,
    load_vision_manifest,
    load_stage_manifest,
    project_paths,
    save_stage_manifest,
    save_json,
    transform_variant_manifest_path,
    vision_manifest_path,
)
from ..utils import filter_segments, normalize_segments, now_iso, parse_segment_selector, resolve_project_path, to_relative

logger = get_run_logger("relecture.transform")

TRANSFORMATION_STRATEGIES = ("independent", "full_context", "two_pass", "sliding_window")


def _build_clients() -> tuple[TransformationService, VisionService | None]:
    return TransformationService.from_config(), VisionService.from_config()


def _ensure_image_assets(source_video: str, segments: list[Segment], image_dir: str) -> list[Segment]:
    return process_video_segments(
        source_video,
        segments,
        output_dir=image_dir,
        extract_video=False,
        extract_audio=False,
        extract_images=True,
    )


def _collect_vision_contexts(
    *,
    project_file: str,
    project_source_video: str,
    paths,
    segments: list[Segment],
    accessibility_profile: str,
    vision_service: VisionService | None,
) -> dict[int, str | None]:
    if accessibility_profile == "none" or vision_service is None:
        return {}

    vision_path = vision_manifest_path(project_file, vision_service.model_name, accessibility_profile)
    if Path(vision_path).exists():
        cached = load_vision_manifest(vision_path)
        return {result.segment_id: result.visual_context for result in cached.results}

    started_at = now_iso()
    results: list[VisionResult] = []
    contexts: dict[int, str | None] = {}
    for segment in segments:
        image_ref = segment.image
        visual_context = None
        if image_ref:
            image_path = resolve_project_path(paths.project_dir, image_ref.path)
            visual_context = vision_service.extract_visual_context(image_path)
        results.append(
            VisionResult(
                segment_id=segment.id,
                image=image_ref,
                model_name=vision_service.model_name,
                visual_context=visual_context,
            )
        )
        contexts[segment.id] = visual_context

    manifest = VisionManifest(
        schema_version=VISION_SCHEMA_VERSION,
        status=StageStatus(
            name="vision",
            state="completed",
            started_at=started_at,
            completed_at=now_iso(),
        ),
        source_video=project_source_video,
        model_name=vision_service.model_name,
        accessibility_profile=accessibility_profile,
        results=results,
    )
    save_json(vision_path, manifest.to_dict())
    return contexts


def _all_transcripts_text(segments: list[Segment]) -> str:
    lines = []
    for seg in segments:
        t = seg.clean_transcript or seg.raw_transcript or ""
        lines.append(f"[Segment {seg.id}]: {t}")
    return "\n\n".join(lines)


def _generate_adaptation_plan(
    transformation_service: TransformationService,
    segments: list[Segment],
    target_audience: str,
    describe_visuals: bool = False,
    background_profile: str = "none",
    accessibility_profile: str = "none",
) -> str:
    full_text = _all_transcripts_text(segments)
    desc_str = " This adaptation must include verbal visual descriptions for diagrams and figures." if describe_visuals else ""
    prompt = (
        f"You are planning an adaptation of a lecture for {target_audience}.{desc_str}\n\n"
        f"Here is the full lecture transcript:\n{full_text}\n\n"
        "Generate a concise adaptation plan (bullet points) describing how each section should be "
        "rewritten: what technical terms to explain, what visual elements to verbalize, "
        "what tone to adopt. This plan will guide per-segment rewriting."
    )
    try:
        response = transformation_service.client.chat.completions.create(
            model=transformation_service.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to generate adaptation plan for two_pass strategy: {exc}") from exc
    content = response.choices[0].message.content if response and response.choices else ""
    plan = (content or "").strip()
    if not plan:
        raise RuntimeError("Adaptation plan generation returned empty content for two_pass strategy.")
    return plan


def run_transformation(
    project_file: str,
    *,
    segment_ids: str | None = None,
    target_audience: str = "students without a CS background",
    describe_visuals: bool = False,
    background_profile: str = "none",
    accessibility_profile: str = "none",
    custom_instructions: str | None = None,
    strategy: str = "independent",
    transformation_service: TransformationService | None = None,
    vision_service: VisionService | None = None,
    progress_cb=None,  # optional callable(fraction: float, desc: str)
) -> TransformationManifest:
    if strategy not in TRANSFORMATION_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {TRANSFORMATION_STRATEGIES}")

    if accessibility_profile == "visual_impairment":
        describe_visuals = True
    if background_profile == "cs_background" and target_audience == "general audience":
        target_audience = "students without a CS background"

    project = ensure_project_manifest(project_file)
    paths = project_paths(project_file)
    ensure_pipeline_layout(paths)
    transcription = load_stage_manifest(project, project_file, "transcription")
    source_video = resolve_project_path(paths.project_dir, project.source_video)
    selected_ids = parse_segment_selector(segment_ids)
    target_segments = filter_segments(transcription.segments, selected_ids)
    started_at = now_iso()

    if transformation_service is None:
        transformation_service, default_vision = _build_clients()
        if vision_service is None:
            vision_service = default_vision

    segments = list(transcription.segments)
    if describe_visuals:
        segments = _ensure_image_assets(source_video, segments, paths.images_dir)
    segments = normalize_segments(segments)

    vision_contexts = (
        _collect_vision_contexts(
            project_file=project_file,
            project_source_video=project.source_video,
            paths=paths,
            segments=segments,
            accessibility_profile="visual_impairment" if describe_visuals else "none",
            vision_service=vision_service,
        )
        if describe_visuals
        else {}
    )

    # Strategy: two_pass — generate a plan first
    adaptation_plan: str | None = None
    if strategy == "two_pass":
        adaptation_plan = _generate_adaptation_plan(
            transformation_service,
            segments,
            target_audience=target_audience,
            describe_visuals=describe_visuals,
            background_profile=background_profile,
            accessibility_profile=accessibility_profile,
        )

    full_context_text = _all_transcripts_text(segments) if strategy == "full_context" else None

    segment_map = {segment.id: segment for segment in segments}
    target_segment_ids = {item.id for item in target_segments}
    transformed: list[Segment] = []
    sliding_window_context: list[str] = []

    total = max(len(segments), 1)
    for index, segment in enumerate(segments, start=1):
        if progress_cb:
            progress_cb((index - 1) / total, f"Transforming segment {index}/{total}")
        if segment.id not in target_segment_ids:
            transformed.append(segment)
            continue

        transcript = (segment.clean_transcript or segment.raw_transcript or "").strip()
        image_ref = segment.image
        if image_ref:
            image_ref = ArtifactRef(
                kind=image_ref.kind,
                path=to_relative(paths.project_dir, resolve_project_path(paths.project_dir, image_ref.path)),
            )
        if not transcript:
            logger.info(f"Segment {segment.id} has no transcript (silent audio). Preserving as silent segment.")
            transformed.append(
                replace(
                    segment_map[segment.id],
                    image=image_ref,
                    transformed_transcript="",
                    visual_narration=None,
                )
            )
            continue

        visual_context = vision_contexts.get(segment.id)

        strategy_ctx: str | None = None
        if strategy == "full_context" and full_context_text:
            strategy_ctx = (
                f"Full lecture context for reference:\n{full_context_text}\n\n"
                f"Transform ONLY segment {segment.id}. Do not rewrite other segments."
            )
        elif strategy == "two_pass" and adaptation_plan:
            strategy_ctx = f"Adaptation plan:\n{adaptation_plan}"
        elif strategy == "sliding_window" and sliding_window_context:
            strategy_ctx = (
                "Previously adapted segments (for continuity):\n"
                + "\n".join(f"- {t}" for t in sliding_window_context[-2:])
            )

        effective_instructions = custom_instructions
        if strategy_ctx:
            effective_instructions = (
                f"{custom_instructions}\n\n{strategy_ctx}" if custom_instructions else strategy_ctx
            )

        transformed_text, visual_narration = transformation_service.transform_content(
            transcript=transcript,
            visual_context=visual_context,
            target_audience=target_audience,
            describe_visuals=describe_visuals,
            custom_instructions=effective_instructions,
            background_profile=background_profile,
            accessibility_profile=accessibility_profile,
        )
        if transformed_text is None:
            raise RuntimeError(f"Transformation failed for segment {segment.id}")

        if strategy == "sliding_window":
            sliding_window_context.append(transformed_text.strip())

        transformed.append(
            replace(
                segment_map[segment.id],
                image=image_ref,
                transformed_transcript=transformed_text.strip(),
                visual_narration=visual_narration.strip() if visual_narration else None,
            )
        )

    manifest = TransformationManifest(
        schema_version=TRANSFORMATION_SCHEMA_VERSION,
        status=StageStatus(
            name="transformation",
            state="completed",
            started_at=started_at,
            completed_at=now_iso(),
        ),
        source_video=project.source_video,
        target_audience=target_audience,
        describe_visuals=describe_visuals,
        background_profile=background_profile,
        accessibility_profile="visual_impairment" if describe_visuals else accessibility_profile,
        custom_instructions=custom_instructions,
        strategy=strategy,
        adaptation_plan=adaptation_plan,
        segments=normalize_segments(transformed),
    )
    save_stage_manifest(project, project_file, "transformation", manifest)
    variant_path = transform_variant_manifest_path(
        project_file,
        model_name=transformation_service.model_name,
        target_audience=target_audience,
        describe_visuals=describe_visuals,
        background_profile=background_profile,
        accessibility_profile=accessibility_profile,
    )
    save_json(variant_path, manifest.to_dict())
    return manifest
