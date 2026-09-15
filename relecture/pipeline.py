from __future__ import annotations

from pathlib import Path

from .stages.assemble import run_assembly
from .stages.segment import run_segmentation
from .stages.synthesize import run_synthesis
from .stages.transform import run_transformation
from .stages.transcribe import run_transcription


def _manifest_exists(project_file: str, name: str) -> bool:
    return (Path(project_file).parent / "manifests" / name).exists()


def run_pipeline(
    project_file: str,
    *,
    source_video: str | None = None,
    project_name: str | None = None,
    background_profile: str = "none",
    accessibility_profile: str = "none",
    target_audience: str = "general audience",
    describe_visuals: bool = False,
    custom_instructions: str | None = None,
    strategy: str = "independent",
    synthesis_mode: str = "full",
    voice_sample_path: str | None = None,
    language: str = "en",
    style: str = "neutral",
    model: str = "qwen3",
    backend: str = "qwen3",
    output_suffix: str | None = None,
    speed: float = 1.0,
    temperature: float = 0.9,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    output_filename: str = "lecture_final.mp4",
    generate_subtitles: bool = True,
) -> dict[str, object | None]:
    if _manifest_exists(project_file, "segmentation.v1.json"):
        print("[pipeline] segmentation already done — skipping")
        segmentation = None
    else:
        segmentation = run_segmentation(project_file, source_video=source_video, project_name=project_name)

    if _manifest_exists(project_file, "transcription.v1.json"):
        print("[pipeline] transcription already done — skipping")
        transcription = None
    else:
        transcription = run_transcription(project_file, language=language)

    if _manifest_exists(project_file, "transformation.v1.json"):
        print("[pipeline] transformation already done — skipping")
        transformation = None
    else:
        transformation = run_transformation(
            project_file,
            target_audience=target_audience,
            describe_visuals=describe_visuals,
            background_profile=background_profile,
            accessibility_profile=accessibility_profile,
            custom_instructions=custom_instructions,
            strategy=strategy,
        )

    if _manifest_exists(project_file, "synthesis.v1.json"):
        print("[pipeline] synthesis already done — skipping")
        synthesis = None
    else:
        synthesis = run_synthesis(
            project_file,
            mode=synthesis_mode,
            voice_sample_path=voice_sample_path,
            language=language,
            style=style,
            model=model,
            backend=backend,
            output_suffix=output_suffix,
            speed=speed,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    assembly = None
    if synthesis_mode == "full":
        if _manifest_exists(project_file, "assembly.v1.json"):
            print("[pipeline] assembly already done — skipping")
        else:
            assembly = run_assembly(
                project_file,
                output_filename=output_filename,
                generate_subtitles=generate_subtitles,
            )

    return {
        "segmentation": segmentation,
        "transcription": transcription,
        "transformation": transformation,
        "synthesis": synthesis,
        "assembly": assembly,
    }
