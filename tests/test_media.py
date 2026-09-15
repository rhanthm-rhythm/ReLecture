from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from relecture.media import process_video_segments
from relecture.models import ArtifactRef, Segment


def _segment(segment_id: int, **overrides) -> Segment:
    base = dict(id=segment_id, start_time=0.0, end_time=10.0, duration=10.0)
    base.update(overrides)
    return Segment(**base)


class ProcessVideoSegmentsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.output_dir = str(Path(self._tmp.name) / "artifacts")
        self.addCleanup(self._tmp.cleanup)

    def test_extracted_artifacts_are_named_by_segment_id(self) -> None:
        segments = [_segment(3), _segment(7)]
        with patch("relecture.media.extract_audio_segment", return_value=True):
            processed = process_video_segments(
                "lecture.mp4",
                segments,
                output_dir=self.output_dir,
                extract_video=False,
                extract_audio=True,
                extract_images=False,
            )

        names = [Path(segment.audio.path).name for segment in processed]
        self.assertEqual(names, ["segment_003_audio.wav", "segment_007_audio.wav"])
        self.assertEqual([segment.id for segment in processed], [3, 7])

    def test_failed_extraction_preserves_existing_artifact(self) -> None:
        existing = ArtifactRef(kind="image", path="artifacts/images/segment_001_slide.png")
        segments = [_segment(1, image=existing)]
        with patch("relecture.media.extract_slide_image", return_value=False):
            processed = process_video_segments(
                "lecture.mp4",
                segments,
                output_dir=self.output_dir,
                extract_video=False,
                extract_audio=False,
                extract_images=True,
            )

        self.assertEqual(processed[0].image, existing)

    def test_disabled_extraction_leaves_other_artifacts_untouched(self) -> None:
        audio = ArtifactRef(kind="audio", path="artifacts/audio/segment_001_audio.wav")
        segments = [_segment(1, audio=audio)]
        with patch("relecture.media.extract_slide_image", return_value=True):
            processed = process_video_segments(
                "lecture.mp4",
                segments,
                output_dir=self.output_dir,
                extract_video=False,
                extract_audio=False,
                extract_images=True,
            )

        self.assertEqual(processed[0].audio, audio)
        self.assertIsNotNone(processed[0].image)
        self.assertIsNone(processed[0].video)

    def test_inputs_are_not_mutated(self) -> None:
        segments = [_segment(1)]
        with patch("relecture.media.extract_audio_segment", return_value=True):
            processed = process_video_segments(
                "lecture.mp4",
                segments,
                output_dir=self.output_dir,
                extract_video=False,
                extract_audio=True,
                extract_images=False,
            )

        self.assertIsNone(segments[0].audio)
        self.assertIsNotNone(processed[0].audio)

    def test_transcripts_survive_the_round_trip(self) -> None:
        segments = [_segment(1, raw_transcript="raw text", clean_transcript="clean text")]
        with patch("relecture.media.extract_audio_segment", return_value=True):
            processed = process_video_segments(
                "lecture.mp4",
                segments,
                output_dir=self.output_dir,
                extract_video=False,
                extract_audio=True,
                extract_images=False,
            )

        self.assertEqual(processed[0].raw_transcript, "raw text")
        self.assertEqual(processed[0].clean_transcript, "clean text")



def _write_slide_video(path: str, slides: int = 3, seconds_per_slide: float = 4.0, fps: float = 10.0) -> float:
    """Write a synthetic slide deck: `slides` distinct static frames in sequence."""
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    if not writer.isOpened():
        raise unittest.SkipTest("cv2.VideoWriter unavailable in this environment")
    frames_per_slide = int(fps * seconds_per_slide)
    for slide in range(slides):
        image = np.full((120, 160, 3), slide * 90, np.uint8)
        cv2.putText(image, f"S{slide}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255 - slide * 80,) * 3, 5)
        for _ in range(frames_per_slide):
            writer.write(image)
    writer.release()
    return slides * frames_per_slide / fps


class DetectSlideBoundariesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.video = str(Path(self._tmp.name) / "deck.mp4")
        self.duration = _write_slide_video(self.video)

    def test_finds_a_segment_per_slide(self) -> None:
        from relecture.media import detect_slide_boundaries

        segments = detect_slide_boundaries(self.video)
        self.assertEqual(len(segments), 3)

    def test_segments_cover_the_whole_video_contiguously(self) -> None:
        from relecture.media import detect_slide_boundaries

        segments = detect_slide_boundaries(self.video)
        self.assertEqual(segments[0]["start"], 0.0)
        for previous, current in zip(segments, segments[1:]):
            self.assertAlmostEqual(previous["end"], current["start"], places=6)
        # The tail must reach the real duration, not the last sampled frame.
        self.assertAlmostEqual(segments[-1]["end"], self.duration, places=2)

    def test_confidence_is_per_boundary_not_a_constant(self) -> None:
        from relecture.media import detect_slide_boundaries

        segments = detect_slide_boundaries(self.video)
        confidences = [segment["confidence"] for segment in segments]
        self.assertEqual(confidences[0], 1.0)  # video start is a certain boundary
        for value in confidences:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        # Detected boundaries carry the dissimilarity that triggered them, so they
        # are not all the same number the way the old constant threshold was.
        self.assertGreater(len(set(confidences)), 1)

    def test_ids_are_sequential_from_one(self) -> None:
        from relecture.media import detect_slide_boundaries

        segments = detect_slide_boundaries(self.video)
        self.assertEqual([segment["id"] for segment in segments], list(range(1, len(segments) + 1)))
if __name__ == "__main__":
    unittest.main()
