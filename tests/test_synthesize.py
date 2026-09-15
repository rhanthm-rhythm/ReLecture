from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from relecture.models import (
    ArtifactRef,
    Segment,
    StageStatus,
    TransformationManifest,
    TRANSFORMATION_SCHEMA_VERSION,
)
from relecture.services import TransformationService
from relecture.stages.synthesize import (
    _resolve_voice_reference,
    run_synthesis,
    select_segments_for_mode,
)
from relecture.storage import ensure_project_manifest, save_stage_manifest
from relecture.utils import now_iso


class ResolveVoiceReferenceTests(unittest.TestCase):
    def test_transcribes_truncated_reference_audio(self) -> None:
        mock_paths = MagicMock()
        mock_paths.project_dir = "/tmp/fake_project"

        segment = Segment(
            id=1,
            start_time=0.0,
            end_time=5.0,
            duration=5.0,
            audio=ArtifactRef(kind="audio", path="audio/seg1.wav"),
            raw_transcript="Raw original speech",
            clean_transcript="Clean speech",
        )

        mock_transcription_service = MagicMock()
        mock_transcription_service.transcribe.return_value = "Exact audio words"

        with patch("relecture.stages.synthesize._truncate_audio_for_voice_reference", return_value="/tmp/fake_project/audio/seg1.wav"):
            voice_path, voice_text = _resolve_voice_reference(
                mock_paths,
                [segment],
                voice_sample_path=None,
                transcription_service=mock_transcription_service,
            )

        self.assertIn("seg1.wav", voice_path)
        self.assertEqual(voice_text, "Exact audio words")
        mock_transcription_service.transcribe.assert_called_once_with(voice_path)

    def test_custom_voice_sample_is_transcribed(self) -> None:
        mock_paths = MagicMock()
        mock_paths.project_dir = "/tmp/fake_project"

        mock_transcription_service = MagicMock()
        mock_transcription_service.transcribe.return_value = "Custom sample speech"

        voice_path, voice_text = _resolve_voice_reference(
            mock_paths,
            [],
            voice_sample_path="custom_voice.wav",
            transcription_service=mock_transcription_service,
        )

        self.assertEqual(voice_text, "Custom sample speech")
        mock_transcription_service.transcribe.assert_called_once()

    def test_transcription_failure_falls_back_gracefully(self) -> None:
        mock_paths = MagicMock()
        mock_paths.project_dir = "/tmp/fake_project"

        segment = Segment(
            id=1,
            start_time=0.0,
            end_time=5.0,
            duration=5.0,
            audio=ArtifactRef(kind="audio", path="audio/seg1.wav"),
            raw_transcript="Fallback raw",
        )

        mock_transcription_service = MagicMock()
        mock_transcription_service.transcribe.side_effect = RuntimeError("Whisper down")

        with patch("relecture.stages.synthesize._truncate_audio_for_voice_reference", return_value="/tmp/fake_project/audio/seg1.wav"):
            voice_path, voice_text = _resolve_voice_reference(
                mock_paths,
                [segment],
                voice_sample_path=None,
                transcription_service=mock_transcription_service,
            )

        self.assertEqual(voice_text, "Fallback raw")


class SelectSegmentsForModeTests(unittest.TestCase):
    def test_preview_selects_only_first_segment(self) -> None:
        s1 = Segment(id=1, start_time=0.0, end_time=5.0, duration=5.0)
        s2 = Segment(id=2, start_time=5.0, end_time=10.0, duration=5.0)
        result = select_segments_for_mode([s1, s2], "preview")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 1)

    def test_full_selects_all_segments(self) -> None:
        s1 = Segment(id=1, start_time=0.0, end_time=5.0, duration=5.0)
        s2 = Segment(id=2, start_time=5.0, end_time=10.0, duration=5.0)
        result = select_segments_for_mode([s1, s2], "full")
        self.assertEqual(len(result), 2)


class MathSanitizationSafetyTests(unittest.TestCase):
    def test_empty_transcript_returns_verbatim_without_llm(self) -> None:
        mock_client = MagicMock()
        service = TransformationService(mock_client, model_name="test-model")

        self.assertEqual(service.sanitize_for_speech(""), "")
        self.assertEqual(service.sanitize_for_speech("   "), "   ")
        mock_client.chat.completions.create.assert_not_called()

    def test_calls_llm_with_strict_prompt_and_zero_temperature(self) -> None:
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "x squared equals y"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        service = TransformationService(mock_client, model_name="test-model")

        text = "Let d/dx f(x) = partial I partial x + x^2"
        result = service.sanitize_for_speech(text)

        self.assertEqual(result, "x squared equals y")
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["temperature"], 0.0)
        system_msg = call_kwargs["messages"][0]["content"]
        self.assertIn("Do NOT reply conversationally", system_msg)
        self.assertIn("Output ONLY the normalized transcript", system_msg)


class RunSynthesisSilentSegmentTests(unittest.TestCase):
    def test_silent_segment_produces_silence_without_calling_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_file = os.path.join(tmp_dir, "project.json")
            video_path = os.path.join(tmp_dir, "inputs", "video.mp4")
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            Path(video_path).touch()

            project = ensure_project_manifest(project_file, source_video=video_path)

            # Create dummy audio file for reference
            audio_path = os.path.join(tmp_dir, "artifacts", "audio", "seg_001.wav")
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            import wave
            with wave.open(audio_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(b"\x00" * 48000)

            # Create transformation manifest with a silent segment
            silent_seg = Segment(
                id=1,
                start_time=0.0,
                end_time=2.0,
                duration=2.0,
                audio=ArtifactRef(kind="audio", path="artifacts/audio/seg_001.wav"),
                transformed_transcript="",  # Silent
            )
            manifest = TransformationManifest(
                schema_version=TRANSFORMATION_SCHEMA_VERSION,
                status=StageStatus(name="transformation", state="completed", started_at=now_iso(), completed_at=now_iso()),
                source_video=project.source_video,
                segments=[silent_seg],
                strategy="independent",
                target_audience="students",
            )
            save_stage_manifest(project, project_file, "transformation", manifest)

            mock_backend = MagicMock()
            mock_transcription_service = MagicMock()
            mock_transcription_service.transcribe.return_value = "Some ref speech"

            with patch("relecture.stages.synthesize.build_tts_backend", return_value=mock_backend):
                synth_manifest = run_synthesis(
                    project_file,
                    mode="full",
                    transcription_service=mock_transcription_service,
                )

            # Backend synthesize should NOT be called for a silent segment
            mock_backend.synthesize.assert_not_called()
            self.assertEqual(len(synth_manifest.results), 1)
            self.assertEqual(synth_manifest.results[0].text_used, "")
            self.assertEqual(synth_manifest.results[0].duration_seconds, 2.0)
