from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from relecture.models import (
    SEGMENTATION_SCHEMA_VERSION,
    TRANSCRIPTION_SCHEMA_VERSION,
    ArtifactRef,
    Segment,
    SegmentationManifest,
    StageStatus,
)
from relecture.services import TranscriptionService, TransformationService
from relecture.stages.transcribe import run_transcription
from relecture.storage import ensure_project_manifest, save_stage_manifest
from relecture.utils import now_iso


class TranscriptionServiceTests(unittest.TestCase):
    def test_transcribe_returns_stripped_text_on_success(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"dummy audio data")
            tmp_path = tmp.name

        try:
            service = TranscriptionService("http://localhost:5001/transcribe")
            mock_response = MagicMock()
            mock_response.json.return_value = {"transcription": "  hello world  "}
            mock_response.raise_for_status.return_value = None

            with patch("requests.post", return_value=mock_response) as mock_post:
                result = service.transcribe(tmp_path, language="en")

            self.assertEqual(result, "hello world")
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs
            self.assertEqual(call_kwargs["data"], {"language": "en"})
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_transcribe_returns_empty_string_for_silent_audio(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"dummy audio")
            tmp_path = tmp.name

        try:
            service = TranscriptionService("http://localhost:5001/transcribe")
            mock_response = MagicMock()
            mock_response.json.return_value = {"transcription": ""}
            mock_response.raise_for_status.return_value = None

            with patch("requests.post", return_value=mock_response):
                result = service.transcribe(tmp_path)

            self.assertEqual(result, "")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_transcribe_raises_file_not_found_for_missing_file(self) -> None:
        service = TranscriptionService("http://localhost:5001/transcribe")
        with self.assertRaises(FileNotFoundError):
            service.transcribe("non_existent_audio_path_12345.wav")

    def test_transcribe_raises_runtime_error_on_network_failure(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"dummy audio")
            tmp_path = tmp.name

        try:
            service = TranscriptionService("http://localhost:5001/transcribe")
            with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Refused")):
                with self.assertRaises(RuntimeError) as ctx:
                    service.transcribe(tmp_path)
            self.assertIn("Whisper transcription failed", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_transcribe_raises_runtime_error_on_http_error(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"dummy audio")
            tmp_path = tmp.name

        try:
            service = TranscriptionService("http://localhost:5001/transcribe")
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
            with patch("requests.post", return_value=mock_response):
                with self.assertRaises(RuntimeError) as ctx:
                    service.transcribe(tmp_path)
            self.assertIn("Whisper transcription failed", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_health_reports_true_on_200(self) -> None:
        service = TranscriptionService("http://localhost:5001/transcribe")
        mock_response = MagicMock(status_code=200)
        with patch("requests.get", return_value=mock_response) as mock_get:
            self.assertTrue(service.health())
        mock_get.assert_called_once_with("http://localhost:5001/health", timeout=5)

    def test_health_reports_false_on_connection_failure(self) -> None:
        service = TranscriptionService("http://localhost:5001/transcribe")
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Down")):
            self.assertFalse(service.health())


class RunTranscriptionStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.video = self.root / "input.mp4"
        self.video.write_bytes(b"video")
        self.project_path = self.root / "project.json"
        self.project = ensure_project_manifest(str(self.project_path), source_video=str(self.video))

        # Create segmentation manifest
        seg_manifest = SegmentationManifest(
            schema_version=SEGMENTATION_SCHEMA_VERSION,
            status=StageStatus(
                name="segmentation",
                state="completed",
                started_at=now_iso(),
                completed_at=now_iso(),
            ),
            source_video="input.mp4",
            segments=[
                Segment(
                    id=1,
                    start_time=0.0,
                    end_time=5.0,
                    duration=5.0,
                    audio=ArtifactRef(kind="audio", path="artifacts/audio/segment_001_audio.wav"),
                ),
                Segment(
                    id=2,
                    start_time=5.0,
                    end_time=10.0,
                    duration=5.0,
                    audio=ArtifactRef(kind="audio", path="artifacts/audio/segment_002_audio.wav"),
                ),
            ],
        )
        save_stage_manifest(self.project, str(self.project_path), "segmentation", seg_manifest)
        self.addCleanup(self._tmp.cleanup)

    @patch("relecture.stages.transcribe._extract_audio_assets")
    def test_run_transcription_persists_valid_manifest(self, mock_extract) -> None:
        # Mock extracted segments
        def fake_extract(source_video, segments, audio_dir):
            audio1 = Path(audio_dir) / "segment_001_audio.wav"
            audio2 = Path(audio_dir) / "segment_002_audio.wav"
            audio1.write_bytes(b"audio1")
            audio2.write_bytes(b"audio2")
            return [
                Segment(id=1, start_time=0.0, end_time=5.0, duration=5.0, audio=ArtifactRef(kind="audio", path=str(audio1))),
                Segment(id=2, start_time=5.0, end_time=10.0, duration=5.0, audio=ArtifactRef(kind="audio", path=str(audio2))),
            ]

        mock_extract.side_effect = fake_extract

        mock_transcription_service = MagicMock()
        mock_transcription_service.transcribe.side_effect = [
            "Introduction to algorithms",
            "Next slide is $x^2 + y^2$",
        ]

        mock_transformation_service = MagicMock()
        mock_transformation_service.sanitize_for_speech.side_effect = [
            "Introduction to algorithms",
            "Next slide is x squared plus y squared",
        ]

        progress_calls = []

        manifest = run_transcription(
            str(self.project_path),
            language="en",
            transcription_service=mock_transcription_service,
            transformation_service=mock_transformation_service,
            progress_cb=lambda frac, desc: progress_calls.append((frac, desc)),
        )

        self.assertEqual(manifest.schema_version, TRANSCRIPTION_SCHEMA_VERSION)
        self.assertEqual(manifest.status.state, "completed")
        self.assertEqual(len(manifest.segments), 2)
        self.assertEqual(manifest.segments[0].raw_transcript, "Introduction to algorithms")
        self.assertEqual(manifest.segments[0].clean_transcript, "Introduction to algorithms")
        self.assertEqual(manifest.segments[1].raw_transcript, "Next slide is $x^2 + y^2$")
        self.assertEqual(manifest.segments[1].clean_transcript, "Next slide is x squared plus y squared")

        # Verify progress callback was invoked
        self.assertGreater(len(progress_calls), 0)

        # Verify language was forwarded
        mock_transcription_service.transcribe.assert_any_call(unittest.mock.ANY, language="en")

    @patch("relecture.stages.transcribe._extract_audio_assets")
    def test_run_transcription_raises_when_transcribe_fails(self, mock_extract) -> None:
        def fake_extract(source_video, segments, audio_dir):
            audio1 = Path(audio_dir) / "segment_001_audio.wav"
            audio1.write_bytes(b"audio1")
            return [
                Segment(id=1, start_time=0.0, end_time=5.0, duration=5.0, audio=ArtifactRef(kind="audio", path=str(audio1))),
            ]

        mock_extract.side_effect = fake_extract

        mock_transcription_service = MagicMock()
        mock_transcription_service.transcribe.side_effect = RuntimeError("Whisper server connection refused")

        mock_transformation_service = MagicMock()

        with self.assertRaises(RuntimeError) as ctx:
            run_transcription(
                str(self.project_path),
                transcription_service=mock_transcription_service,
                transformation_service=mock_transformation_service,
            )
        self.assertIn("Whisper server connection refused", str(ctx.exception))


class IdentifySegmentsNeedingSanitizationTests(unittest.TestCase):
    def test_identify_segments_returns_flagged_ids(self) -> None:
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"sanitize_segment_ids": [2]}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        service = TransformationService(mock_client, model_name="test-model")
        items = [
            (1, "Hi, this is a presentation."),
            (2, "Let d/dx f(x) = partial I partial x + x^2"),
            (3, "Thank you very much."),
        ]
        flagged = service.identify_segments_needing_sanitization(items)
        self.assertEqual(flagged, {2})
        mock_client.chat.completions.create.assert_called_once()

    def test_identify_segments_empty_input_returns_empty_set(self) -> None:
        mock_client = MagicMock()
        service = TransformationService(mock_client, model_name="test-model")
        self.assertEqual(service.identify_segments_needing_sanitization([]), set())
        self.assertEqual(service.identify_segments_needing_sanitization([(1, ""), (2, "   ")]), set())
        mock_client.chat.completions.create.assert_not_called()

    def test_identify_segments_falls_back_on_api_failure(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")
        service = TransformationService(mock_client, model_name="test-model")
        items = [(1, "Speech 1"), (2, "Speech 2")]
        flagged = service.identify_segments_needing_sanitization(items)
        self.assertEqual(flagged, {1, 2})


if __name__ == "__main__":
    unittest.main()
