from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from relecture.models import (
    TRANSCRIPTION_SCHEMA_VERSION,
    TRANSFORMATION_SCHEMA_VERSION,
    ArtifactRef,
    Segment,
    StageStatus,
    TranscriptionManifest,
)
from relecture.services import TransformationService
from relecture.stages.transform import _generate_adaptation_plan, run_transformation
from relecture.storage import ensure_project_manifest, load_stage_manifest, save_stage_manifest
from relecture.utils import now_iso


class TransformationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_client = MagicMock()
        self.service = TransformationService(self.mock_client, model_name="test-model")

    def test_build_transformation_prompt_basic(self) -> None:
        prompt = self.service.build_transformation_prompt(target_audience="high schoolers")
        self.assertIn("high schoolers", prompt)
        self.assertIn("plain spoken prose only", prompt)
        self.assertNotIn("[Narrator]:", prompt)

    def test_build_transformation_prompt_with_visuals(self) -> None:
        prompt = self.service.build_transformation_prompt(target_audience="students", describe_visuals=True)
        self.assertIn("[Narrator]:", prompt)
        self.assertIn("[Lecturer]:", prompt)
        self.assertIn("as you can see on the slide", prompt)  # banned phrase mentioned in rule

    def test_transform_content_parses_narrator_and_lecturer(self) -> None:
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "[Narrator]: A binary tree with three nodes.\n"
            "[Lecturer]: In this section we explore tree traversals."
        )
        mock_response.choices = [mock_choice]
        self.mock_client.chat.completions.create.return_value = mock_response

        lecturer, narrator = self.service.transform_content(
            transcript="Raw transcript",
            visual_context="Slide with tree",
            describe_visuals=True,
        )

        self.assertEqual(lecturer, "In this section we explore tree traversals.")
        self.assertEqual(narrator, "A binary tree with three nodes.")

    def test_transform_content_without_narrator_tags(self) -> None:
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Just plain adapted lecture text."
        mock_response.choices = [mock_choice]
        self.mock_client.chat.completions.create.return_value = mock_response

        lecturer, narrator = self.service.transform_content(
            transcript="Raw transcript",
            describe_visuals=False,
        )

        self.assertEqual(lecturer, "Just plain adapted lecture text.")
        self.assertIsNone(narrator)

    def test_transform_content_returns_none_tuple_on_failure(self) -> None:
        self.mock_client.chat.completions.create.side_effect = RuntimeError("API error")
        lecturer, narrator = self.service.transform_content("Hello")
        self.assertIsNone(lecturer)
        self.assertIsNone(narrator)


class TransformationStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.video = self.root / "input.mp4"
        self.video.write_bytes(b"video")
        self.project_path = self.root / "project.json"
        self.project = ensure_project_manifest(str(self.project_path), source_video=str(self.video))

        trans_manifest = TranscriptionManifest(
            schema_version=TRANSCRIPTION_SCHEMA_VERSION,
            status=StageStatus(
                name="transcription",
                state="completed",
                started_at=now_iso(),
                completed_at=now_iso(),
            ),
            source_video="input.mp4",
            segments=[
                Segment(id=1, start_time=0.0, end_time=5.0, duration=5.0, raw_transcript="Welcome to class", clean_transcript="Welcome to class"),
                Segment(id=2, start_time=5.0, end_time=10.0, duration=5.0, raw_transcript="Here is a graph", clean_transcript="Here is a graph"),
            ],
        )
        save_stage_manifest(self.project, str(self.project_path), "transcription", trans_manifest)
        self.addCleanup(self._tmp.cleanup)

    def test_two_pass_plan_generation_raises_on_failure(self) -> None:
        mock_service = MagicMock()
        mock_service.client.chat.completions.create.side_effect = RuntimeError("LLM rate limited")
        segments = [Segment(id=1, start_time=0.0, end_time=5.0, duration=5.0, clean_transcript="Test")]

        with self.assertRaises(RuntimeError) as ctx:
            _generate_adaptation_plan(mock_service, segments, target_audience="students")
        self.assertIn("Failed to generate adaptation plan", str(ctx.exception))

    def test_two_pass_plan_generation_raises_on_empty_content(self) -> None:
        mock_service = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="  "))]
        mock_service.client.chat.completions.create.return_value = mock_response
        segments = [Segment(id=1, start_time=0.0, end_time=5.0, duration=5.0, clean_transcript="Test")]

        with self.assertRaises(RuntimeError) as ctx:
            _generate_adaptation_plan(mock_service, segments, target_audience="students")
        self.assertIn("empty content", str(ctx.exception))

    def test_run_transformation_independent_strategy(self) -> None:
        mock_service = MagicMock()
        mock_service.model_name = "test-model"
        mock_service.transform_content.side_effect = [
            ("Adapted welcome", None),
            ("Adapted graph explanation", None),
        ]

        manifest = run_transformation(
            str(self.project_path),
            target_audience="high school students",
            describe_visuals=False,
            strategy="independent",
            transformation_service=mock_service,
        )

        self.assertEqual(manifest.schema_version, TRANSFORMATION_SCHEMA_VERSION)
        self.assertEqual(manifest.status.state, "completed")
        self.assertEqual(manifest.target_audience, "high school students")
        self.assertFalse(manifest.describe_visuals)
        self.assertEqual(len(manifest.segments), 2)
        self.assertEqual(manifest.segments[0].transformed_transcript, "Adapted welcome")
        self.assertIsNone(manifest.segments[0].visual_narration)
        self.assertEqual(manifest.segments[1].transformed_transcript, "Adapted graph explanation")

        # Verify saved manifest on disk
        loaded = load_stage_manifest(self.project, str(self.project_path), "transformation")
        self.assertEqual(loaded.segments[0].transformed_transcript, "Adapted welcome")

    @patch("relecture.stages.transform._ensure_image_assets")
    def test_run_transformation_with_describe_visuals_and_narrator(self, mock_ensure_images) -> None:
        mock_ensure_images.side_effect = lambda vid, segs, out_dir: segs

        mock_service = MagicMock()
        mock_service.model_name = "test-model"
        mock_service.transform_content.side_effect = [
            ("Adapted welcome", None),
            ("Adapted graph explanation", "A line plot with an upward trajectory"),
        ]

        manifest = run_transformation(
            str(self.project_path),
            target_audience="visually impaired learners",
            describe_visuals=True,
            strategy="independent",
            transformation_service=mock_service,
        )

        self.assertTrue(manifest.describe_visuals)
        self.assertEqual(manifest.segments[1].visual_narration, "A line plot with an upward trajectory")

    def test_run_transformation_sliding_window_passes_context(self) -> None:
        mock_service = MagicMock()
        mock_service.model_name = "test-model"
        mock_service.transform_content.side_effect = [
            ("Segment 1 adapted", None),
            ("Segment 2 adapted", None),
        ]

        run_transformation(
            str(self.project_path),
            strategy="sliding_window",
            transformation_service=mock_service,
        )

        self.assertEqual(mock_service.transform_content.call_count, 2)
        second_call = mock_service.transform_content.call_args_list[1]
        custom_inst = second_call.kwargs.get("custom_instructions") or ""
        self.assertIn("Previously adapted segments", custom_inst)
        self.assertIn("Segment 1 adapted", custom_inst)


if __name__ == "__main__":
    unittest.main()
