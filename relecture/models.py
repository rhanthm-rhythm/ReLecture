from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


PROJECT_SCHEMA_VERSION = "project-manifest/v1"
SEGMENTATION_SCHEMA_VERSION = "segmentation-manifest/v1"
TRANSCRIPTION_SCHEMA_VERSION = "transcription-manifest/v1"
TRANSFORMATION_SCHEMA_VERSION = "transformation-manifest/v1"
VISION_SCHEMA_VERSION = "vision-manifest/v1"
SYNTHESIS_SCHEMA_VERSION = "synthesis-manifest/v1"
ASSEMBLY_SCHEMA_VERSION = "assembly-manifest/v1"


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_none(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value


@dataclass
class ArtifactRef:
    kind: str
    path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ArtifactRef | None":
        if not data:
            return None
        return cls(kind=data["kind"], path=data["path"])


@dataclass
class Segment:
    id: int
    start_time: float
    end_time: float
    duration: float
    confidence: float = 0.0
    audio: ArtifactRef | None = None
    image: ArtifactRef | None = None
    video: ArtifactRef | None = None
    raw_transcript: str | None = None
    clean_transcript: str | None = None
    transformed_transcript: str | None = None
    visual_narration: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Segment":
        return cls(
            id=int(data["id"]),
            start_time=float(data["start_time"]),
            end_time=float(data["end_time"]),
            duration=float(data["duration"]),
            confidence=float(data.get("confidence", 0.0)),
            audio=ArtifactRef.from_dict(data.get("audio")),
            image=ArtifactRef.from_dict(data.get("image")),
            video=ArtifactRef.from_dict(data.get("video")),
            raw_transcript=data.get("raw_transcript"),
            clean_transcript=data.get("clean_transcript"),
            transformed_transcript=data.get("transformed_transcript"),
            visual_narration=data.get("visual_narration"),
        )


@dataclass
class ModelConfig:
    llm_model: str | None = None
    vision_model: str | None = None
    tts_model: str | None = "qwen3"
    whisper_endpoint: str | None = "http://localhost:5001/transcribe"

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ModelConfig":
        if not data:
            return cls()
        return cls(
            llm_model=data.get("llm_model"),
            vision_model=data.get("vision_model"),
            tts_model=data.get("tts_model", "qwen3"),
            whisper_endpoint=data.get("whisper_endpoint", "http://localhost:5001/transcribe"),
        )


@dataclass
class StageStatus:
    name: str
    state: str
    started_at: str
    completed_at: str | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageStatus":
        return cls(
            name=data["name"],
            state=data["state"],
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
            error=data.get("error"),
            warnings=list(data.get("warnings", [])),
        )


@dataclass
class PipelinePaths:
    project_file: str
    project_dir: str
    output_root: str
    manifests_dir: str
    artifacts_dir: str
    segments_dir: str
    images_dir: str
    audio_dir: str
    synthesis_dir: str
    final_dir: str


@dataclass
class ProjectManifest:
    schema_version: str
    project_name: str
    created_at: str
    updated_at: str
    source_video: str
    output_root: str
    models: ModelConfig = field(default_factory=ModelConfig)
    manifests: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_name": self.project_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_video": self.source_video,
            "output_root": self.output_root,
            "models": self.models.to_dict(),
            "manifests": dict(self.manifests),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectManifest":
        return cls(
            schema_version=data["schema_version"],
            project_name=data["project_name"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            source_video=data["source_video"],
            output_root=data["output_root"],
            models=ModelConfig.from_dict(data.get("models")),
            manifests=dict(data.get("manifests", {})),
        )


@dataclass
class SegmentationManifest:
    schema_version: str
    status: StageStatus
    source_video: str
    segments: list[Segment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.to_dict(),
            "source_video": self.source_video,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SegmentationManifest":
        return cls(
            schema_version=data["schema_version"],
            status=StageStatus.from_dict(data["status"]),
            source_video=data["source_video"],
            segments=[Segment.from_dict(item) for item in data.get("segments", [])],
        )


@dataclass
class TranscriptionManifest:
    schema_version: str
    status: StageStatus
    source_video: str
    segments: list[Segment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.to_dict(),
            "source_video": self.source_video,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptionManifest":
        return cls(
            schema_version=data["schema_version"],
            status=StageStatus.from_dict(data["status"]),
            source_video=data["source_video"],
            segments=[Segment.from_dict(item) for item in data.get("segments", [])],
        )


@dataclass
class TransformationManifest:
    schema_version: str
    status: StageStatus
    source_video: str
    target_audience: str
    describe_visuals: bool = False
    strategy: str = "independent"
    custom_instructions: str | None = None
    adaptation_plan: str | None = None
    background_profile: str = "none"
    accessibility_profile: str = "none"
    segments: list[Segment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "schema_version": self.schema_version,
                "status": self.status.to_dict(),
                "source_video": self.source_video,
                "target_audience": self.target_audience,
                "describe_visuals": self.describe_visuals,
                "strategy": self.strategy,
                "custom_instructions": self.custom_instructions,
                "adaptation_plan": self.adaptation_plan,
                "background_profile": self.background_profile,
                "accessibility_profile": self.accessibility_profile,
                "segments": [segment.to_dict() for segment in self.segments],
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransformationManifest":
        desc_vis = data.get("describe_visuals")
        if desc_vis is None:
            desc_vis = data.get("accessibility_profile") == "visual_impairment"
        return cls(
            schema_version=data["schema_version"],
            status=StageStatus.from_dict(data["status"]),
            source_video=data["source_video"],
            target_audience=data.get("target_audience", "general audience"),
            describe_visuals=bool(desc_vis),
            strategy=data.get("strategy", "independent"),
            custom_instructions=data.get("custom_instructions"),
            adaptation_plan=data.get("adaptation_plan"),
            background_profile=data.get("background_profile", "none"),
            accessibility_profile="visual_impairment" if desc_vis else data.get("accessibility_profile", "none"),
            segments=[Segment.from_dict(item) for item in data.get("segments", [])],
        )


@dataclass
class VisionResult:
    segment_id: int
    image: ArtifactRef | None
    model_name: str
    visual_context: str | None

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "segment_id": self.segment_id,
                "image": self.image.to_dict() if self.image else None,
                "model_name": self.model_name,
                "visual_context": self.visual_context,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisionResult":
        return cls(
            segment_id=int(data["segment_id"]),
            image=ArtifactRef.from_dict(data.get("image")),
            model_name=data["model_name"],
            visual_context=data.get("visual_context"),
        )


@dataclass
class VisionManifest:
    schema_version: str
    status: StageStatus
    source_video: str
    model_name: str
    accessibility_profile: str
    results: list[VisionResult]

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "schema_version": self.schema_version,
                "status": self.status.to_dict(),
                "source_video": self.source_video,
                "model_name": self.model_name,
                "accessibility_profile": self.accessibility_profile,
                "results": [result.to_dict() for result in self.results],
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisionManifest":
        return cls(
            schema_version=data["schema_version"],
            status=StageStatus.from_dict(data["status"]),
            source_video=data["source_video"],
            model_name=data["model_name"],
            accessibility_profile=data.get("accessibility_profile", "none"),
            results=[VisionResult.from_dict(item) for item in data.get("results", [])],
        )


@dataclass
class SynthesisChunk:
    index: int
    text: str
    audio: ArtifactRef
    kind: str = "lecture"  # "lecture" (main voice) or "narration" (visual description, narrator voice)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "audio": self.audio.to_dict(),
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SynthesisChunk":
        return cls(
            index=int(data["index"]),
            text=data["text"],
            audio=ArtifactRef.from_dict(data["audio"]),
            kind=data.get("kind", "lecture"),
        )


@dataclass
class SynthesisResult:
    segment_id: int
    mode: str
    text_used: str
    audio: ArtifactRef
    duration_seconds: float
    chunks: list[SynthesisChunk] = field(default_factory=list)
    video: ArtifactRef | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "segment_id": self.segment_id,
                "mode": self.mode,
                "text_used": self.text_used,
                "audio": self.audio.to_dict(),
                "video": self.video.to_dict() if self.video else None,
                "duration_seconds": self.duration_seconds,
                "chunks": [chunk.to_dict() for chunk in self.chunks],
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SynthesisResult":
        return cls(
            segment_id=int(data["segment_id"]),
            mode=data["mode"],
            text_used=data["text_used"],
            audio=ArtifactRef.from_dict(data["audio"]),
            video=ArtifactRef.from_dict(data.get("video")),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            chunks=[SynthesisChunk.from_dict(item) for item in data.get("chunks", [])],
        )


@dataclass
class SynthesisManifest:
    schema_version: str
    status: StageStatus
    source_video: str
    mode: str
    model: str
    backend: str
    language: str
    output_suffix: str | None
    resolved_voice_reference: ArtifactRef | None
    resolved_voice_reference_transcript: str | None
    segments: list[Segment]
    results: list[SynthesisResult]

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "schema_version": self.schema_version,
                "status": self.status.to_dict(),
                "source_video": self.source_video,
                "mode": self.mode,
                "model": self.model,
                "backend": self.backend,
                "language": self.language,
                "output_suffix": self.output_suffix,
                "resolved_voice_reference": self.resolved_voice_reference.to_dict() if self.resolved_voice_reference else None,
                "resolved_voice_reference_transcript": self.resolved_voice_reference_transcript,
                "segments": [segment.to_dict() for segment in self.segments],
                "results": [result.to_dict() for result in self.results],
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SynthesisManifest":
        return cls(
            schema_version=data["schema_version"],
            status=StageStatus.from_dict(data["status"]),
            source_video=data["source_video"],
            mode=data["mode"],
            model=data["model"],
            backend=data.get("backend", "qwen3"),
            language=data["language"],
            output_suffix=data.get("output_suffix"),
            resolved_voice_reference=ArtifactRef.from_dict(data.get("resolved_voice_reference")),
            resolved_voice_reference_transcript=data.get("resolved_voice_reference_transcript"),
            segments=[Segment.from_dict(item) for item in data.get("segments", [])],
            results=[SynthesisResult.from_dict(item) for item in data.get("results", [])],
        )


@dataclass
class AssemblyManifest:
    schema_version: str
    status: StageStatus
    source_video: str
    output_video: ArtifactRef
    subtitles: ArtifactRef | None
    included_segments: list[int]
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "schema_version": self.schema_version,
                "status": self.status.to_dict(),
                "source_video": self.source_video,
                "output_video": self.output_video.to_dict(),
                "subtitles": self.subtitles.to_dict() if self.subtitles else None,
                "included_segments": list(self.included_segments),
                "duration_seconds": self.duration_seconds,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssemblyManifest":
        return cls(
            schema_version=data["schema_version"],
            status=StageStatus.from_dict(data["status"]),
            source_video=data["source_video"],
            output_video=ArtifactRef.from_dict(data["output_video"]),
            subtitles=ArtifactRef.from_dict(data.get("subtitles")),
            included_segments=[int(item) for item in data.get("included_segments", [])],
            duration_seconds=float(data.get("duration_seconds", 0.0)),
        )
