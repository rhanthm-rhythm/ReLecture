from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

from ..media import check_ffmpeg_available, process_video_segments
from ..models import (
    ArtifactRef,
    Segment,
    StageStatus,
    SynthesisChunk,
    SynthesisManifest,
    SynthesisResult,
    SYNTHESIS_SCHEMA_VERSION,
)
from ..services import AudioProcessingService, TextChunker, TranscriptionService
from ..storage import (
    ensure_pipeline_layout,
    ensure_project_manifest,
    load_stage_manifest,
    project_paths,
    save_stage_manifest,
    save_json,
    synthesis_variant_manifest_path,
)
from ..utils import filter_segments, normalize_segments, now_iso, parse_segment_selector, resolve_project_path, to_relative

logger = logging.getLogger(__name__)

MAX_VOICE_REFERENCE_DURATION = 10.0

# Bundled narrator reference sample used to voice slide/visual descriptions for
# the visually-impaired accessibility track. Kept distinct from the lecturer's
# cloned voice so listeners can tell the two apart. See assets/narrator_voice.txt
# for the exact spoken text (used as the ref_text for voice-clone backends).
_NARRATOR_VOICE_PATH = Path(__file__).resolve().parents[2] / "assets" / "narrator_voice.wav"
_NARRATOR_VOICE_TEXT_PATH = _NARRATOR_VOICE_PATH.with_suffix(".txt")


def _resolve_narrator_reference() -> tuple[str | None, str | None]:
    """Resolve the bundled narrator voice reference (path, transcript).

    Returns (None, None) if the bundled sample is missing, in which case
    callers should fall back to the lecturer's own voice reference for
    narration rather than dropping the narration content.
    """
    if not _NARRATOR_VOICE_PATH.exists():
        logger.warning(
            "Narrator voice reference not found at %s; visual narration will "
            "fall back to the lecturer's voice.",
            _NARRATOR_VOICE_PATH,
        )
        return None, None
    narrator_text = None
    if _NARRATOR_VOICE_TEXT_PATH.exists():
        narrator_text = _NARRATOR_VOICE_TEXT_PATH.read_text(encoding="utf-8").strip() or None
    return str(_NARRATOR_VOICE_PATH), narrator_text


def build_tts_backend(backend_name: str):
    from ..config import load_config
    cfg = load_config()
    if backend_name == "qwen3":
        from ..backends import Qwen3Backend
        endpoint = cfg.tts.endpoint_url if cfg.tts.backend == "qwen3" else os.getenv("QWEN3_TTS_ENDPOINT", "http://localhost:5000")
        return Qwen3Backend(endpoint=endpoint)
    if backend_name == "chatterbox":
        from ..backends import ChatterboxBackend
        return ChatterboxBackend(endpoint=os.getenv("CHATTERBOX_ENDPOINT", "http://localhost:5002"))
    if backend_name == "cosyvoice":
        from ..backends import CosyVoiceBackend
        return CosyVoiceBackend(endpoint=os.getenv("COSYVOICE_ENDPOINT", "http://localhost:5003"))
    raise ValueError(f"Unknown TTS backend: {backend_name!r}. Choose from: qwen3, chatterbox, cosyvoice")


def select_segments_for_mode(segments: list[Segment], mode: str) -> list[Segment]:
    if not segments:
        return []
    if mode == "preview":
        return [segments[0]]
    return list(segments)


def _ensure_image_assets(
    source_video: str,
    segments: list[Segment],
    image_dir: str,
    progress_cb=None,
) -> list[Segment]:
    if all(segment.image for segment in segments):
        return segments
    return process_video_segments(
        source_video,
        segments,
        output_dir=image_dir,
        extract_video=False,
        extract_audio=False,
        extract_images=True,
        progress_cb=progress_cb,
    )


def _truncate_audio_for_voice_reference(audio_path: str, max_duration: float = MAX_VOICE_REFERENCE_DURATION) -> str:
    info = AudioProcessingService.get_audio_info(audio_path)
    duration = float(info.get("duration_seconds", 0.0))
    if duration <= max_duration:
        return audio_path
    truncated = audio_path.replace(".wav", f"_voice_ref_{int(max_duration)}s.wav")
    if os.path.exists(truncated):
        return truncated
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-t",
        str(max_duration),
        "-c:a",
        "copy",
        truncated,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to truncate voice reference audio: {result.stderr}")
    return truncated


def _render_video_from_image(image_path: str, audio_path: str, output_path: str) -> str:
    if not check_ffmpeg_available():
        raise RuntimeError("FFmpeg is not available.")
    AudioProcessingService.ensure_directory_exists(output_path)
    duration = AudioProcessingService.get_audio_info(audio_path).get("duration_seconds", 0.0)
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        image_path,
        "-i",
        audio_path,
        "-c:v",
        "libx264",
        "-tune",
        "stillimage",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        str(duration),
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg rendering failed: {result.stderr}")
    return output_path


def _resolve_voice_reference(
    paths,
    segments: list[Segment],
    voice_sample_path: str | None,
    transcription_service,
) -> tuple[str, str | None]:
    if voice_sample_path:
        voice_path = resolve_project_path(paths.project_dir, voice_sample_path)
        try:
            voice_text = transcription_service.transcribe(voice_path).strip() or None
        except Exception as exc:
            logger.warning("Failed to transcribe custom voice reference '%s': %s", voice_path, exc)
            voice_text = None
        return voice_path, voice_text
    for segment in segments:
        if segment.audio:
            audio_path = resolve_project_path(paths.project_dir, segment.audio.path)
            voice_path = _truncate_audio_for_voice_reference(audio_path)
            # CRITICAL: Always transcribe the exact truncated voice reference clip!
            # Do NOT use clean_transcript (which may have been rewritten) or the full
            # segment raw_transcript (which may be 40s while voice_path is 10s).
            # ref_text must match what is *actually spoken* in voice_path -- a
            # mismatched ref_audio/ref_text confuses the voice-clone model, causing
            # it to "continue" past the truncation point and echo/hallucinate
            # extra content instead of speaking the requested text.
            try:
                voice_text = transcription_service.transcribe(voice_path).strip() or None
            except Exception as exc:
                logger.warning("Failed to transcribe voice reference '%s': %s", voice_path, exc)
                voice_text = segment.raw_transcript or None
            return voice_path, voice_text
    raise RuntimeError("Could not resolve a voice reference. Provide --voice-sample-path or run transcription first.")


def run_synthesis(
    project_file: str,
    *,
    segment_ids: str | None = None,
    mode: str = "preview",
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
    transcription_service=None,
    speech_generator=None,  # kept for backward compat in tests
    progress_cb=None,  # optional callable(fraction: float, desc: str)
) -> SynthesisManifest:
    project = ensure_project_manifest(project_file)
    paths = project_paths(project_file)
    ensure_pipeline_layout(paths)
    try:
        previous = load_stage_manifest(project, project_file, "transformation")
    except Exception:
        previous = load_stage_manifest(project, project_file, "transcription")

    source_video = resolve_project_path(paths.project_dir, project.source_video)
    if source_video is None:
        raise FileNotFoundError("Project source video is missing.")

    if transcription_service is None:
        transcription_service = TranscriptionService(project.models.whisper_endpoint or "http://localhost:5001/transcribe")

    tts_backend = build_tts_backend(backend) if speech_generator is None else None

    selected = filter_segments(previous.segments, parse_segment_selector(segment_ids))
    selected = select_segments_for_mode(selected, mode)

    # Reserve the first slice of the progress bar for one-time prep work
    # (slide image extraction, voice reference resolution) so slow steps on
    # long videos are visible instead of looking like a hang on "Synthesis".
    PREP_FRACTION = 0.15

    def _prep_cb(frac: float, desc: str) -> None:
        if progress_cb:
            progress_cb(frac * PREP_FRACTION, desc)

    logger.info("Synthesis: preparing slide images…")
    if progress_cb:
        progress_cb(0.0, "Preparing slide images…")
    segments = _ensure_image_assets(source_video, list(previous.segments), paths.images_dir, progress_cb=_prep_cb)
    segment_map = {segment.id: segment for segment in segments}
    started_at = now_iso()

    logger.info("Synthesis: resolving voice reference audio…")
    if progress_cb:
        progress_cb(PREP_FRACTION, "Resolving voice reference audio…")
    voice_ref_path, voice_ref_text = _resolve_voice_reference(paths, segments, voice_sample_path, transcription_service)
    logger.info("Synthesis: voice reference ready (%s). Starting TTS calls to backend=%s…", voice_ref_path, backend)
    if progress_cb:
        progress_cb(PREP_FRACTION, "Voice reference ready. Starting synthesis…")

    has_visual_narration = any((segment.visual_narration or "").strip() for segment in segments)
    narrator_ref_path, narrator_ref_text = (
        _resolve_narrator_reference() if has_visual_narration else (None, None)
    )
    if has_visual_narration and narrator_ref_path is None:
        # No bundled narrator sample available -- narration will still be
        # spoken, just cloned from the lecturer's voice instead of a distinct one.
        narrator_ref_path, narrator_ref_text = voice_ref_path, voice_ref_text

    total_selected = max(len(selected), 1)
    results: list[SynthesisResult] = []
    for seg_idx, selected_segment in enumerate(selected):
        segment = segment_map[selected_segment.id]
        raw_text = segment.transformed_transcript or segment.clean_transcript or segment.raw_transcript or ""
        lecture_text = raw_text.strip()
        narration_text = (segment.visual_narration or "").strip()
        filename_base = f"segment_{segment.id:03d}" + (f"_{output_suffix}" if output_suffix else "")
        audio_output = os.path.join(paths.synthesis_dir, f"{filename_base}_synth.wav")
        video_output = os.path.join(paths.synthesis_dir, f"{filename_base}_synth.mp4")
        chunk_dir = os.path.join(paths.synthesis_dir, "chunks")
        os.makedirs(chunk_dir, exist_ok=True)

        if os.path.exists(audio_output) and os.path.getsize(audio_output) > 0:
            logger.info("Segment %d already synthesized — skipping", segment.id)
            video_ref = ArtifactRef(kind="video", path=to_relative(paths.project_dir, video_output)) if os.path.exists(video_output) else None
            duration = float(AudioProcessingService.get_audio_info(audio_output).get("duration_seconds", 0.0))
            results.append(
                SynthesisResult(
                    segment_id=segment.id,
                    mode=mode,
                    text_used=lecture_text,
                    audio=ArtifactRef(kind="audio", path=to_relative(paths.project_dir, audio_output)),
                    video=video_ref,
                    duration_seconds=duration,
                    chunks=[],
                )
            )
            continue

        if not lecture_text and not narration_text:
            logger.info(f"Segment {segment.id} has no speech to synthesize. Generating silent audio ({segment.duration:.2f}s).")
            import wave
            with wave.open(audio_output, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                num_frames = int(24000 * max(segment.duration, 0.5))
                wf.writeframes(b"\x00" * num_frames * 2)

            video_ref = None
            if segment.image:
                image_path = resolve_project_path(paths.project_dir, segment.image.path)
                _render_video_from_image(image_path, audio_output, video_output)
                video_ref = ArtifactRef(kind="video", path=to_relative(paths.project_dir, video_output))

            results.append(
                SynthesisResult(
                    segment_id=segment.id,
                    mode=mode,
                    text_used="",
                    audio=ArtifactRef(kind="audio", path=to_relative(paths.project_dir, audio_output)),
                    video=video_ref,
                    duration_seconds=float(segment.duration),
                    chunks=[],
                )
            )
            continue

        # Build the render plan: narration chunks (distinct narrator voice)
        # are spoken first, followed by the lecture chunks (lecturer's cloned
        # voice), so the audio and subtitles both surface the slide
        # description before the lecturer's own words.
        narrator_chunks_text = TextChunker().split_text(narration_text, max_chars=350) if narration_text else []
        lecture_chunks_text = TextChunker().split_text(lecture_text, max_chars=350) if lecture_text else []
        if mode == "preview":
            narrator_chunks_text = narrator_chunks_text[:1]
            lecture_chunks_text = lecture_chunks_text[:1]
        render_plan = [
            (chunk_text, narrator_ref_path, narrator_ref_text, "narration") for chunk_text in narrator_chunks_text
        ] + [
            (chunk_text, voice_ref_path, voice_ref_text, "lecture") for chunk_text in lecture_chunks_text
        ]
        total_chunks = max(len(render_plan), 1)
        chunk_results: list[SynthesisChunk] = []
        chunk_paths: list[str] = []
        for index, (chunk_text, ref_path, ref_text, chunk_kind) in enumerate(render_plan, start=1):
            step_desc = f"Calling {backend} — segment {seg_idx+1}/{total_selected} · chunk {index}/{total_chunks}"
            logger.info(step_desc)
            if progress_cb:
                raw_frac = (seg_idx + (index - 1) / total_chunks) / total_selected
                frac = PREP_FRACTION + raw_frac * (1.0 - PREP_FRACTION)
                progress_cb(frac, step_desc)
            chunk_path = os.path.join(chunk_dir, f"{filename_base}_chunk_{index}.wav")
            if speech_generator is not None:
                result = speech_generator(
                    text=chunk_text,
                    output_path=chunk_path,
                    style=style,
                    language=language,
                    reference_audio_path=ref_path,
                    ref_text=ref_text,
                    model=model,
                    speed=speed,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                if isinstance(result, str) and result.startswith("Error"):
                    raise RuntimeError(result)
            else:
                audio_bytes = tts_backend.synthesize(
                    text=chunk_text,
                    reference_audio_path=ref_path,
                    reference_text=ref_text,
                    language=language,
                    style=style,
                    speed=speed,
                    extra={"temperature": temperature, "top_p": top_p, "repetition_penalty": repetition_penalty},
                )
                with open(chunk_path, "wb") as fh:
                    fh.write(audio_bytes)
            logger.info("✅ Chunk %d/%d for segment %d written to %s", index, total_chunks, segment.id, chunk_path)

            chunk_paths.append(chunk_path)
            chunk_results.append(
                SynthesisChunk(
                    index=index,
                    text=chunk_text,
                    audio=ArtifactRef(kind="audio", path=to_relative(paths.project_dir, chunk_path)),
                    kind=chunk_kind,
                )
            )

        if len(chunk_paths) == 1:
            shutil.copy(chunk_paths[0], audio_output)
        else:
            AudioProcessingService.merge_wav_files(chunk_paths, audio_output)

        video_ref = None
        if segment.image:
            image_path = resolve_project_path(paths.project_dir, segment.image.path)
            _render_video_from_image(image_path, audio_output, video_output)
            video_ref = ArtifactRef(kind="video", path=to_relative(paths.project_dir, video_output))

        duration = float(AudioProcessingService.get_audio_info(audio_output).get("duration_seconds", 0.0))
        text_used = f"{narration_text}\n\n{lecture_text}" if narration_text else lecture_text
        results.append(
            SynthesisResult(
                segment_id=segment.id,
                mode=mode,
                text_used=text_used,
                audio=ArtifactRef(kind="audio", path=to_relative(paths.project_dir, audio_output)),
                video=video_ref,
                duration_seconds=duration,
                chunks=chunk_results,
            )
        )
        logger.info("✅ Segment %d/%d synthesized (%.1fs audio)", seg_idx + 1, total_selected, duration)
        if progress_cb:
            frac = PREP_FRACTION + ((seg_idx + 1) / total_selected) * (1.0 - PREP_FRACTION)
            progress_cb(frac, f"✅ Segment {seg_idx+1}/{total_selected} synthesized")

    persisted_segments = normalize_segments(
        [
            replace(
                segment,
                audio=segment.audio and ArtifactRef(kind=segment.audio.kind, path=to_relative(paths.project_dir, resolve_project_path(paths.project_dir, segment.audio.path))),
                image=segment.image and ArtifactRef(kind=segment.image.kind, path=to_relative(paths.project_dir, resolve_project_path(paths.project_dir, segment.image.path))),
                video=segment.video and ArtifactRef(kind=segment.video.kind, path=to_relative(paths.project_dir, resolve_project_path(paths.project_dir, segment.video.path))),
            )
            for segment in segments
        ]
    )

    manifest = SynthesisManifest(
        schema_version=SYNTHESIS_SCHEMA_VERSION,
        status=StageStatus(
            name="synthesis",
            state="completed",
            started_at=started_at,
            completed_at=now_iso(),
        ),
        source_video=project.source_video,
        mode=mode,
        model=model,
        backend=backend,
        language=language,
        output_suffix=output_suffix,
        resolved_voice_reference=ArtifactRef(kind="audio", path=to_relative(paths.project_dir, voice_ref_path)),
        resolved_voice_reference_transcript=voice_ref_text,
        segments=persisted_segments,
        results=results,
    )
    save_stage_manifest(project, project_file, "synthesis", manifest)
    variant_path = synthesis_variant_manifest_path(project_file, backend=backend, language=language)
    save_json(variant_path, manifest.to_dict())
    return manifest
