from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from relecture.cli import main
from relecture.pipeline import run_pipeline
from relecture.models import (
    PROJECT_SCHEMA_VERSION,
    ArtifactRef,
    SYNTHESIS_SCHEMA_VERSION,
    VISION_SCHEMA_VERSION,
    Segment,
    StageStatus,
    SynthesisManifest,
    SynthesisResult,
    VisionManifest,
    VisionResult,
)
from relecture.stages.assemble import run_assembly
from relecture.stages.synthesize import select_segments_for_mode
from relecture.storage import (
    ensure_project_manifest,
    load_vision_manifest,
    save_json,
    save_stage_manifest,
    slugify_model_name,
    transform_variant_manifest_path,
    vision_manifest_path,
)
from relecture.utils import apply_segment_adjustments, now_iso


class PipelineContractTests(unittest.TestCase):
    def test_project_manifest_round_trip_and_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "input.mp4"
            video.write_bytes(b"video")
            project_path = root / "project.json"

            project = ensure_project_manifest(str(project_path), source_video=str(video))

            self.assertEqual(project.schema_version, PROJECT_SCHEMA_VERSION)
            self.assertTrue(project_path.exists())
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], PROJECT_SCHEMA_VERSION)
            self.assertEqual(payload["source_video"], "input.mp4")

    def test_apply_segment_adjustments_normalizes_segments(self) -> None:
        segments = [
            Segment(id=1, start_time=0.0, end_time=10.0, duration=10.0),
            Segment(id=2, start_time=10.0, end_time=20.0, duration=10.0),
        ]
        adjusted = apply_segment_adjustments(
            segments,
            [
                {"type": "range", "segment": 2, "start": 8.0, "end": 19.0},
                {"type": "end", "segment": 1, "value": 8.0},
            ],
        )
        self.assertEqual(adjusted[0].end_time, 8.0)
        self.assertEqual(adjusted[1].start_time, 8.0)
        self.assertEqual(adjusted[1].duration, 11.0)

    def test_cli_segment_wires_new_project_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project.json"
            video = root / "input.mp4"
            video.write_bytes(b"video")
            with patch("relecture.cli.run_segmentation") as run_segmentation:
                exit_code = main(["segment", "--project", str(project), "--source-video", str(video)])
            self.assertEqual(exit_code, 0)
            run_segmentation.assert_called_once()

    def test_cli_segment_derives_project_path_from_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "lecture_a.mp4"
            video.write_bytes(b"video")
            with patch("relecture.cli.run_segmentation") as run_segmentation:
                exit_code = main(["segment", "--source-video", str(video)])
            self.assertEqual(exit_code, 0)
            derived_project = root / "lecture_a" / "project.json"
            run_segmentation.assert_called_once_with(str(derived_project.resolve()), source_video=str(video), project_name=None)

    def test_cli_transcribe_forwards_project_and_default_language(self) -> None:
        with patch("relecture.cli.run_transcription") as run_transcription:
            exit_code = main(["transcribe", "--project", "project.json"])
        self.assertEqual(exit_code, 0)
        run_transcription.assert_called_once_with("project.json", language="en")

    def test_cli_transform_uses_discrete_background_and_accessibility_args(self) -> None:
        with patch("relecture.cli.run_transformation") as run_transformation:
            exit_code = main([
                "transform",
                "--project", "project.json",
                "--background", "cs_background",
                "--accessibility", "visual_impairment",
                "--target-audience", "students",
            ])
        self.assertEqual(exit_code, 0)
        run_transformation.assert_called_once_with(
            "project.json",
            segment_ids=None,
            background_profile="cs_background",
            accessibility_profile="visual_impairment",
            target_audience="students",
            describe_visuals=True,
            custom_instructions=None,
            strategy="independent",
        )

    @patch("relecture.pipeline.run_assembly")
    @patch("relecture.pipeline.run_synthesis")
    @patch("relecture.pipeline.run_transformation")
    @patch("relecture.pipeline.run_transcription")
    @patch("relecture.pipeline.run_segmentation")
    def test_pipeline_preview_forwards_options_and_skips_assembly(
        self,
        run_segmentation,
        run_transcription,
        run_transformation,
        run_synthesis,
        run_assembly,
    ) -> None:
        result = run_pipeline(
            "project.json",
            strategy="two_pass",
            synthesis_mode="preview",
            backend="chatterbox",
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.2,
        )

        run_transformation.assert_called_once_with(
            "project.json",
            target_audience="general audience",
            describe_visuals=False,
            background_profile="none",
            accessibility_profile="none",
            custom_instructions=None,
            strategy="two_pass",
        )
        run_synthesis.assert_called_once_with(
            "project.json",
            mode="preview",
            voice_sample_path=None,
            language="en",
            style="neutral",
            model="qwen3",
            backend="chatterbox",
            output_suffix=None,
            speed=1.0,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.2,
        )
        run_assembly.assert_not_called()
        self.assertIsNone(result["assembly"])

    @patch("relecture.pipeline.run_assembly")
    @patch("relecture.pipeline.run_synthesis")
    @patch("relecture.pipeline.run_transformation")
    @patch("relecture.pipeline.run_transcription")
    @patch("relecture.pipeline.run_segmentation")
    def test_pipeline_full_assembles_output(
        self,
        run_segmentation,
        run_transcription,
        run_transformation,
        run_synthesis,
        run_assembly,
    ) -> None:
        result = run_pipeline("project.json", synthesis_mode="full")

        run_assembly.assert_called_once_with(
            "project.json",
            output_filename="lecture_final.mp4",
            generate_subtitles=True,
        )
        self.assertIs(result["assembly"], run_assembly.return_value)

    def test_cli_run_forwards_non_default_options(self) -> None:
        with patch("relecture.cli.run_pipeline") as pipeline:
            exit_code = main([
                "run",
                "--project", "project.json",
                "--mode", "preview",
                "--strategy", "sliding_window",
                "--backend", "cosyvoice",
                "--temperature", "0.6",
                "--top-p", "0.75",
                "--repetition-penalty", "1.1",
            ])

        self.assertEqual(exit_code, 0)
        forwarded = pipeline.call_args.kwargs
        self.assertEqual(forwarded["synthesis_mode"], "preview")
        self.assertEqual(forwarded["strategy"], "sliding_window")
        self.assertEqual(forwarded["backend"], "cosyvoice")
        self.assertEqual(forwarded["temperature"], 0.6)
        self.assertEqual(forwarded["top_p"], 0.75)
        self.assertEqual(forwarded["repetition_penalty"], 1.1)

    def test_preview_mode_selects_only_first_segment(self) -> None:
        segments = [
            Segment(id=1, start_time=0.0, end_time=5.0, duration=5.0),
            Segment(id=2, start_time=5.0, end_time=10.0, duration=5.0),
        ]
        preview = select_segments_for_mode(segments, "preview")
        full = select_segments_for_mode(segments, "full")
        self.assertEqual([segment.id for segment in preview], [1])
        self.assertEqual([segment.id for segment in full], [1, 2])

    def test_assembly_rejects_missing_video_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "input.mp4"
            video.write_bytes(b"video")
            project_path = root / "project.json"
            project = ensure_project_manifest(str(project_path), source_video=str(video))

            synthesis_manifest = SynthesisManifest(
                schema_version=SYNTHESIS_SCHEMA_VERSION,
                status=StageStatus(
                    name="synthesis",
                    state="completed",
                    started_at=now_iso(),
                    completed_at=now_iso(),
                ),
                source_video=project.source_video,
                mode="full",
                model="qwen3",
                backend="qwen3",
                language="en",
                output_suffix=None,
                resolved_voice_reference=None,
                resolved_voice_reference_transcript=None,
                segments=[
                    Segment(
                        id=1,
                        start_time=0.0,
                        end_time=5.0,
                        duration=5.0,
                        clean_transcript="Hello",
                    )
                ],
                results=[
                    SynthesisResult(
                        segment_id=1,
                        mode="full",
                        text_used="Hello",
                        audio=ArtifactRef(kind="audio", path="artifacts/synthesis/segment_001_synth.wav"),
                        duration_seconds=5.0,
                    )
                ],
            )
            save_stage_manifest(project, str(project_path), "synthesis", synthesis_manifest)
            with self.assertRaises(RuntimeError):
                run_assembly(str(project_path), ffmpeg_available_checker=lambda: True)

    def test_stage_manifest_version_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "input.mp4"
            video.write_bytes(b"video")
            project_path = root / "project.json"
            project = ensure_project_manifest(str(project_path), source_video=str(video))
            path = root / "manifests" / "segmentation.v1.json"
            save_json(
                str(path),
                {
                    "schema_version": "segmentation-manifest/v0",
                    "status": {
                        "name": "segmentation",
                        "state": "completed",
                        "started_at": now_iso(),
                        "completed_at": now_iso(),
                    },
                    "source_video": project.source_video,
                    "segments": [],
                },
            )
            from relecture.storage import load_stage_manifest, ManifestError

            with self.assertRaises(ManifestError):
                load_stage_manifest(project, str(project_path), "segmentation")

    def test_variant_manifest_paths_are_model_specific(self) -> None:
        path = transform_variant_manifest_path(
            "C:\\repo\\project.json",
            model_name="openai/gpt-oss-20b",
            background_profile="cs_background",
            accessibility_profile="visual_impairment",
        )
        self.assertIn("transforms", path)
        self.assertIn("openai-gpt-oss-20b", path)
        self.assertIn("cs-background", path)
        self.assertIn("visual-impairment", path)

    def test_vision_manifest_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vision.v1.json"
            manifest = VisionManifest(
                schema_version=VISION_SCHEMA_VERSION,
                status=StageStatus(
                    name="vision",
                    state="completed",
                    started_at=now_iso(),
                    completed_at=now_iso(),
                ),
                source_video="input.mp4",
                model_name="Qwen/Qwen3-VL-8B-Instruct",
                accessibility_profile="visual_impairment",
                results=[
                    VisionResult(
                        segment_id=1,
                        image=ArtifactRef(kind="image", path="artifacts/images/slide1.png"),
                        model_name="Qwen/Qwen3-VL-8B-Instruct",
                        visual_context="A title and a figure.",
                    )
                ],
            )
            save_json(str(path), manifest.to_dict())
            loaded = load_vision_manifest(str(path))
            self.assertEqual(loaded.model_name, "Qwen/Qwen3-VL-8B-Instruct")
            self.assertEqual(loaded.results[0].visual_context, "A title and a figure.")


if __name__ == "__main__":
    unittest.main()
