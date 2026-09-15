from __future__ import annotations

import logging
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from relecture.models import ArtifactRef, Segment
from relecture.ui import (
    STAGES,
    _comparison_rows,
    build_ui,
    collect_readiness,
    initialize_run,
    run_demo_workflow,
)


def _manifest(segments=None, results=None, backend="qwen3"):
    return SimpleNamespace(
        segments=segments or [],
        results=results or [],
        backend=backend,
    )


class UIDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        # initialize_run() (called for real by these tests, not mocked) writes
        # into RELECTURE_OUTPUT_ROOT if set, so point it at an isolated temp dir
        # instead of the repo's real output/ folder.
        self._output_root = tempfile.TemporaryDirectory()
        self.addCleanup(self._output_root.cleanup)
        # setup_run_logger() opens a FileHandler on run.log that stays open
        # until the next run starts; close it first so Windows lets us
        # delete the temp directory above (LIFO: registered after, runs first).
        self.addCleanup(self._close_run_log_handlers)
        self._env_patch = patch.dict(os.environ, {"RELECTURE_OUTPUT_ROOT": self._output_root.name})
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

    @staticmethod
    def _close_run_log_handlers() -> None:
        logger = logging.getLogger("relecture")
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)

    def test_initialize_run_copies_inputs_and_returns_exact_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "lecture.mp4"
            voice = root / "voice.wav"
            video.write_bytes(b"video")
            voice.write_bytes(b"voice")

            state = initialize_run(str(video), str(voice))

        project_file = Path(state["project_file"])
        self.assertEqual(project_file.name, "project.json")
        self.assertEqual(Path(state["source_video"]).read_bytes(), b"video")
        self.assertEqual(Path(state["voice_sample"]).read_bytes(), b"voice")
        self.assertEqual(Path(state["source_video"]).parent.name, "inputs")

    def test_comparison_rows_keep_complete_transcripts(self) -> None:
        original_text = "Original " * 40
        adapted_text = "Adapted " * 40
        transcription = _manifest([
            Segment(id=1, start_time=0.0, end_time=2.0, duration=2.0, clean_transcript=original_text),
        ])
        transformation = _manifest([
            Segment(
                id=1,
                start_time=0.0,
                end_time=2.0,
                duration=2.0,
                clean_transcript=original_text,
                transformed_transcript=adapted_text,
            ),
        ])

        rows = _comparison_rows(transcription, transformation)

        self.assertEqual(rows[0][2], original_text)
        self.assertEqual(rows[0][3], adapted_text)

    @patch("relecture.ui._endpoint_health", return_value=True)
    @patch("relecture.ui.build_tts_backend")
    @patch("relecture.ui.TranscriptionService.health", return_value=True)
    @patch("relecture.ui.check_ffmpeg_available", return_value=True)
    @patch("relecture.ui.load_config")
    def test_readiness_only_requires_vision_for_visual_profile(
        self,
        load_config,
        check_ffmpeg,
        whisper_health,
        build_backend,
        endpoint_health,
    ) -> None:
        load_config.return_value = SimpleNamespace(
            whisper_endpoint="http://localhost:5001/transcribe",
            llm=SimpleNamespace(base_url="http://localhost:9000/v1", model_name="llm", api_key=None),
            vision=SimpleNamespace(base_url="http://localhost:8005/v1", model_name="vision"),
        )
        build_backend.return_value.health.return_value = True

        without_vision = collect_readiness("qwen3", "none")
        with_vision = collect_readiness("qwen3", "visual_impairment")

        self.assertIn(["Vision model", "Not required", "Enable visual-impairment adaptation to use it"], without_vision)
        self.assertIn(["Vision model", "Ready", "http://localhost:8005/v1"], with_vision)
        endpoint_health.assert_called_once_with("http://localhost:8005/v1")
        build_backend.assert_called_with("qwen3")

    def _run_workflow(self, mode: str):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        video = root / "video.mp4"
        video.write_bytes(b"video")
        segment = Segment(
            id=1,
            start_time=0.0,
            end_time=2.0,
            duration=2.0,
            clean_transcript="Original text",
            transformed_transcript="Adapted text",
        )
        audio = root / "audio.wav"
        preview = root / "preview.mp4"
        final = root / "final.mp4"
        subtitle = root / "final.srt"
        for path in [audio, preview, final, subtitle]:
            path.write_bytes(b"output")

        synthesis_result = SimpleNamespace(
            audio=ArtifactRef(kind="audio", path=str(audio)),
            video=ArtifactRef(kind="video", path=str(preview)),
        )
        assembly = SimpleNamespace(
            output_video=ArtifactRef(kind="video", path=str(final)),
            subtitles=ArtifactRef(kind="subtitle", path=str(subtitle)),
            included_segments=[1],
            duration_seconds=2.0,
        )
        patches = [
            patch("relecture.ui.collect_readiness", return_value=[["All", "Ready", ""]]),
            patch("relecture.ui.run_segmentation", return_value=_manifest([segment])),
            patch("relecture.ui.run_transcription", return_value=_manifest([segment])),
            patch("relecture.ui.run_transformation", return_value=_manifest([segment])),
            patch("relecture.ui.run_synthesis", return_value=_manifest([segment], [synthesis_result])),
            patch("relecture.ui.run_assembly", return_value=assembly),
        ]
        mocks = [item.start() for item in patches]
        self.addCleanup(lambda: [item.stop() for item in patches])
        outputs = list(run_demo_workflow(
            str(video),
            None,
            mode,
            "custom",
            "none",
            "students",
            "Use short explanations",
            "two_pass",
            "chatterbox",
            1.1,
            0.7,
            0.8,
            1.2,
        ))
        return outputs, mocks

    def test_preview_workflow_reports_stages_and_skips_assembly(self) -> None:
        outputs, mocks = self._run_workflow("preview")
        run_synthesis = mocks[4]
        run_assembly = mocks[5]

        run_synthesis.assert_called_once()
        self.assertEqual(run_synthesis.call_args.kwargs["mode"], "preview")
        self.assertEqual(run_synthesis.call_args.kwargs["backend"], "chatterbox")
        run_assembly.assert_not_called()
        final = outputs[-1]
        process_rows = final[1]
        self.assertEqual([row[0] for row in process_rows], list(STAGES))
        self.assertEqual(process_rows[-1][1], "Skipped")
        self.assertEqual(final[2][0][3], "Adapted text")
        self.assertIsNone(final[5])

    def test_full_workflow_assembles_final_video(self) -> None:
        outputs, mocks = self._run_workflow("full")
        run_synthesis = mocks[4]
        run_assembly = mocks[5]

        self.assertEqual(run_synthesis.call_args.kwargs["mode"], "full")
        run_assembly.assert_called_once()
        final = outputs[-1]
        self.assertEqual(final[1][-1][1], "Completed")
        self.assertTrue(final[5].endswith("final.mp4"))
        self.assertTrue(any(path.endswith("final.srt") for path in final[6]))

    def test_failed_stage_marks_failure_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "video.mp4"
            video.write_bytes(b"video")
            with (
                patch("relecture.ui.collect_readiness", return_value=[["All", "Ready", ""]]),
                patch("relecture.ui.run_segmentation", side_effect=RuntimeError("bad video")),
                patch("relecture.ui.run_transcription") as transcribe,
            ):
                outputs = list(run_demo_workflow(
                    str(video), None, "preview", "none", "none", "students", "",
                    "independent", "qwen3", 1.0, 0.9, 1.0, 1.05,
                ))

        final = outputs[-1]
        self.assertIn("Segmentation failed", final[0])
        self.assertEqual(final[1][1][1], "Failed")
        transcribe.assert_not_called()

    def test_build_ui_smoke(self) -> None:
        demo = build_ui()
        self.assertEqual(demo.__class__.__name__, "Blocks")


if __name__ == "__main__":
    unittest.main()
