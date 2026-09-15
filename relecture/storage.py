from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .models import (
    ASSEMBLY_SCHEMA_VERSION,
    PROJECT_SCHEMA_VERSION,
    SEGMENTATION_SCHEMA_VERSION,
    SYNTHESIS_SCHEMA_VERSION,
    TRANSCRIPTION_SCHEMA_VERSION,
    TRANSFORMATION_SCHEMA_VERSION,
    VISION_SCHEMA_VERSION,
    AssemblyManifest,
    ModelConfig,
    PipelinePaths,
    ProjectManifest,
    SegmentationManifest,
    SynthesisManifest,
    TranscriptionManifest,
    TransformationManifest,
    VisionManifest,
)
from .utils import ensure_dir, now_iso, resolve_project_path


class ManifestError(RuntimeError):
    """Raised when a manifest is missing or malformed."""


STAGE_MANIFEST_FILES = {
    "segmentation": "manifests/segmentation.v1.json",
    "transcription": "manifests/transcription.v1.json",
    "transformation": "manifests/transformation.v1.json",
    "synthesis": "manifests/synthesis.v1.json",
    "assembly": "manifests/assembly.v1.json",
}


def derive_project_file(source_video: str, project_file: str | None = None) -> str:
    source_path = Path(source_video).resolve()
    if project_file:
        return str(Path(project_file).resolve())

    base_parent = source_path.parent
    preferred_dir = base_parent / source_path.stem
    candidate = preferred_dir / "project.json"
    if not candidate.exists():
        return str(candidate)

    try:
        payload = load_json(str(candidate))
        existing = ProjectManifest.from_dict(payload)
        existing_source = resolve_project_path(candidate.parent, existing.source_video)
        if existing_source and Path(existing_source).resolve() == source_path:
            return str(candidate)
    except Exception:
        pass

    suffix = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:8]
    return str(base_parent / f"{source_path.stem}-{suffix}" / "project.json")


def _default_models() -> ModelConfig:
    try:
        from .config import load_config
    except Exception:
        config = None
    else:
        config = load_config()
    return ModelConfig(
        llm_model=config.llm.model_name if config else None,
        vision_model=config.vision.model_name if config else None,
        tts_model="qwen3",
        whisper_endpoint=config.whisper_endpoint if config else "http://localhost:5001/transcribe",
    )


def project_paths(project_file: str) -> PipelinePaths:
    project_path = Path(project_file).resolve()
    project_dir = project_path.parent
    manifest_dir = project_dir / "manifests"
    artifacts_dir = project_dir / "artifacts"
    return PipelinePaths(
        project_file=str(project_path),
        project_dir=str(project_dir),
        output_root=str(project_dir),
        manifests_dir=str(manifest_dir),
        artifacts_dir=str(artifacts_dir),
        segments_dir=str(artifacts_dir / "segments"),
        images_dir=str(artifacts_dir / "images"),
        audio_dir=str(artifacts_dir / "audio"),
        synthesis_dir=str(artifacts_dir / "synthesis"),
        final_dir=str(artifacts_dir / "final"),
    )


def ensure_pipeline_layout(paths: PipelinePaths) -> None:
    ensure_dir(paths.manifests_dir)
    ensure_dir(Path(paths.manifests_dir) / "vision")
    ensure_dir(Path(paths.manifests_dir) / "transforms")
    ensure_dir(Path(paths.manifests_dir) / "synthesis")
    ensure_dir(paths.artifacts_dir)
    ensure_dir(paths.segments_dir)
    ensure_dir(paths.images_dir)
    ensure_dir(paths.audio_dir)
    ensure_dir(paths.synthesis_dir)
    ensure_dir(paths.final_dir)


def save_json(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def create_project_manifest(project_file: str, source_video: str, project_name: str | None = None) -> ProjectManifest:
    project_path = Path(project_file).resolve()
    project_dir = project_path.parent
    created_at = now_iso()
    paths = project_paths(str(project_path))
    # Resolve source_video relative to CWD first, then check if it's inside project_dir
    source_absolute = Path(source_video).resolve()
    if source_absolute.is_relative_to(project_dir):
        source_stored = str(source_absolute.relative_to(project_dir))
    else:
        source_stored = str(source_absolute)
    if not source_absolute.exists():
        raise ManifestError(f"Source video not found: {source_absolute}")
    project = ProjectManifest(
        schema_version=PROJECT_SCHEMA_VERSION,
        project_name=project_name or project_path.parent.name,
        created_at=created_at,
        updated_at=created_at,
        source_video=source_stored,
        output_root=".",
        models=_default_models(),
        manifests=dict(STAGE_MANIFEST_FILES),
    )
    save_project_manifest(str(project_path), project)
    ensure_pipeline_layout(paths)
    return project


def load_project_manifest(project_file: str) -> ProjectManifest:
    path = Path(project_file).resolve()
    if not path.exists():
        raise ManifestError(f"Project manifest not found: {path}")
    payload = load_json(str(path))
    manifest = ProjectManifest.from_dict(payload)
    if manifest.schema_version != PROJECT_SCHEMA_VERSION:
        raise ManifestError(
            f"Unsupported project manifest version '{manifest.schema_version}'. Expected '{PROJECT_SCHEMA_VERSION}'."
        )
    return manifest


def save_project_manifest(project_file: str, project: ProjectManifest) -> None:
    save_json(project_file, project.to_dict())


def ensure_project_manifest(project_file: str, source_video: str | None = None, project_name: str | None = None) -> ProjectManifest:
    path = Path(project_file).resolve()
    if path.exists():
        return load_project_manifest(str(path))
    if not source_video:
        raise ManifestError(f"Project manifest does not exist: {path}. Provide --source-video to create it.")
    return create_project_manifest(str(path), source_video=source_video, project_name=project_name)


def stage_manifest_path(project: ProjectManifest, project_file: str, stage_name: str) -> str:
    relative = project.manifests.get(stage_name)
    if not relative:
        raise ManifestError(f"Unknown stage '{stage_name}'.")
    return resolve_project_path(Path(project_file).resolve().parent, relative)


def slugify_model_name(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "default"


def vision_manifest_path(project_file: str, model_name: str, accessibility_profile: str) -> str:
    project_dir = Path(project_file).resolve().parent
    model_slug = slugify_model_name(model_name)
    profile_slug = slugify_model_name(accessibility_profile)
    return str(project_dir / "manifests" / "vision" / f"vision.{profile_slug}.{model_slug}.v1.json")


def transform_variant_manifest_path(
    project_file: str,
    *,
    model_name: str,
    target_audience: str = "general audience",
    describe_visuals: bool = False,
    background_profile: str = "none",
    accessibility_profile: str = "none",
) -> str:
    project_dir = Path(project_file).resolve().parent
    model_slug = slugify_model_name(model_name)
    if background_profile != "none" or accessibility_profile != "none":
        background_slug = slugify_model_name(background_profile)
        accessibility_slug = slugify_model_name(accessibility_profile)
        return str(
            project_dir
            / "manifests"
            / "transforms"
            / f"transform.{background_slug}.{accessibility_slug}.{model_slug}.v1.json"
        )
    audience_slug = slugify_model_name(target_audience)
    visual_slug = "visual" if describe_visuals else "text"
    return str(
        project_dir
        / "manifests"
        / "transforms"
        / f"transform.{audience_slug}.{visual_slug}.{model_slug}.v1.json"
    )


def synthesis_variant_manifest_path(
    project_file: str,
    *,
    backend: str,
    language: str = "en",
) -> str:
    project_dir = Path(project_file).resolve().parent
    backend_slug = slugify_model_name(backend)
    lang_slug = slugify_model_name(language)
    return str(
        project_dir
        / "manifests"
        / "synthesis"
        / f"synthesis.{lang_slug}.{backend_slug}.v1.json"
    )


def save_stage_manifest(project: ProjectManifest, project_file: str, stage_name: str, manifest) -> str:
    path = stage_manifest_path(project, project_file, stage_name)
    save_json(path, manifest.to_dict())
    project.updated_at = now_iso()
    save_project_manifest(project_file, project)
    return path


def load_stage_manifest(project: ProjectManifest, project_file: str, stage_name: str):
    path = stage_manifest_path(project, project_file, stage_name)
    if not Path(path).exists():
        raise ManifestError(f"Required {stage_name} manifest not found: {path}")
    payload = load_json(path)
    if stage_name == "segmentation":
        manifest = SegmentationManifest.from_dict(payload)
        expected = SEGMENTATION_SCHEMA_VERSION
    elif stage_name == "transcription":
        manifest = TranscriptionManifest.from_dict(payload)
        expected = TRANSCRIPTION_SCHEMA_VERSION
    elif stage_name == "transformation":
        manifest = TransformationManifest.from_dict(payload)
        expected = TRANSFORMATION_SCHEMA_VERSION
    elif stage_name == "synthesis":
        manifest = SynthesisManifest.from_dict(payload)
        expected = SYNTHESIS_SCHEMA_VERSION
    elif stage_name == "assembly":
        manifest = AssemblyManifest.from_dict(payload)
        expected = ASSEMBLY_SCHEMA_VERSION
    else:
        raise ManifestError(f"Unsupported stage '{stage_name}'.")
    if manifest.schema_version != expected:
        raise ManifestError(
            f"Unsupported {stage_name} manifest version '{manifest.schema_version}'. Expected '{expected}'."
        )
    return manifest


def load_vision_manifest(path: str) -> VisionManifest:
    payload = load_json(path)
    manifest = VisionManifest.from_dict(payload)
    if manifest.schema_version != VISION_SCHEMA_VERSION:
        raise ManifestError(
            f"Unsupported vision manifest version '{manifest.schema_version}'. Expected '{VISION_SCHEMA_VERSION}'."
        )
    return manifest
