from __future__ import annotations

import os
import subprocess
from dataclasses import replace
from typing import Callable

from .models import ArtifactRef, Segment


def check_ffmpeg_available() -> bool:
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    except FileNotFoundError:
        return False
    return result.returncode == 0


def compute_frame_diffs(frames: list) -> list[float]:
    """Structural dissimilarity (1 - SSIM) between consecutive sampled frames.

    SSIM is what the detector's threshold (``mean + max(0.15, std * 0.8)``) is
    calibrated against: a slide change scores near 1.0, so the 0.15 floor sits
    below real peaks. A plain mean-absolute pixel difference peaks around 0.1 on
    the same content, which puts every genuine slide change under that floor and
    collapses the lecture into a single segment.

    Falls back to a histogram distance when SSIM cannot run on a frame pair.
    """
    import cv2

    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise RuntimeError(
            "scikit-image is required for slide-change detection (pip install scikit-image)."
        ) from exc

    def histogram_distance(first, second) -> float:
        first_hist = cv2.calcHist([first], [0], None, [64], [0, 256])
        second_hist = cv2.calcHist([second], [0], None, [64], [0, 256])
        cv2.normalize(first_hist, first_hist)
        cv2.normalize(second_hist, second_hist)
        return float(cv2.compareHist(first_hist, second_hist, cv2.HISTCMP_BHATTACHARYYA))

    diffs: list[float] = []
    previous_gray = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_gray is None:
            diffs.append(0.0)
        else:
            try:
                diffs.append(float(1.0 - structural_similarity(previous_gray, gray)))
            except Exception:
                diffs.append(histogram_distance(previous_gray, gray))
        previous_gray = gray
    return diffs


def detect_slide_boundaries(
    video_path: str,
    sample_rate: float = 1.0,
    min_segment_sec: float = 2.0,
    max_suggestions: int = 50,
    progress_callback: Callable | None = None,
) -> list[dict]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("opencv-python and numpy are required for segmentation.") from exc

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    interval_frames = max(1, int(round(fps * sample_rate)))
    frames: list[dict] = []
    index = 0
    while True:
        # Only the sampled frames are decoded; grab() advances past the rest
        # without paying for decoding, which dominates runtime at 1 frame/sec.
        if index % interval_frames == 0:
            ok, frame = cap.read()
            if not ok:
                break
            ts = index / fps
            small = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
            frames.append({"ts": ts, "frame": small})
        elif not cap.grab():
            break
        index += 1
    cap.release()
    if progress_callback:
        progress_callback(100, "Frame extraction complete", len(frames), frame_count)
    if not frames:
        return []

    diffs = compute_frame_diffs([entry["frame"] for entry in frames])
    arr = np.array(diffs)
    threshold = float(arr.mean() + max(0.15, arr.std() * 0.8))
    change_times = sorted({frames[i]["ts"] for i, value in enumerate(diffs) if value >= threshold})

    # A segment's confidence is the dissimilarity of the slide change that opened
    # it, so it says how sure the detector is about that specific boundary. The
    # first segment starts at the video start, which is a certain boundary rather
    # than a detected one.
    dissimilarity_at = {entry["ts"]: float(value) for entry, value in zip(frames, diffs)}

    def confidence_at(timestamp: float) -> float:
        if timestamp <= 0.0:
            return 1.0
        return max(0.0, min(1.0, dissimilarity_at.get(timestamp, 0.0)))

    segments: list[dict] = []
    start = 0.0
    for timestamp in change_times:
        if timestamp - start >= min_segment_sec:
            segments.append({"start": start, "end": timestamp, "confidence": confidence_at(start)})
            start = timestamp

    # The last sampled frame is up to sample_rate seconds short of the true end,
    # and anything past it would never be transcribed.
    sampled_end = frames[-1]["ts"]
    total_duration = frame_count / fps if frame_count > 0 else 0.0
    if total_duration < sampled_end:
        total_duration = sampled_end
    if total_duration - start >= 0.5:
        segments.append({"start": start, "end": total_duration, "confidence": confidence_at(start)})
    elif segments:
        # Remainder too short to stand alone; extend the last segment over it so
        # the segments always cover the whole video.
        segments[-1]["end"] = total_duration
    while len(segments) > max_suggestions:
        durations = [segment["end"] - segment["start"] for segment in segments]
        smallest = int(np.argmin(durations))
        if smallest == 0:
            smallest = 1
        segments[smallest - 1]["end"] = segments[smallest]["end"]
        segments.pop(smallest)
    for idx, segment in enumerate(segments, start=1):
        segment["id"] = idx
    return segments


def extract_audio_segment(video_path: str, start_time: float, end_time: float, output_path: str) -> bool:
    if not check_ffmpeg_available():
        raise RuntimeError("FFmpeg is not available.")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-ss",
        str(start_time),
        "-t",
        str(end_time - start_time),
        "-vn",
        "-acodec",
        "pcm_s16le",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def extract_video_segment(video_path: str, start_time: float, end_time: float, output_path: str) -> bool:
    if not check_ffmpeg_available():
        raise RuntimeError("FFmpeg is not available.")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-ss",
        str(start_time),
        "-t",
        str(end_time - start_time),
        "-c",
        "copy",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def extract_slide_image(video_path: str, timestamp: float, output_path: str) -> bool:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for slide extraction.") from exc

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_number = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    return bool(cv2.imwrite(output_path, frame))


def process_video_segments(
    video_path: str,
    segments: list[Segment],
    output_dir: str,
    extract_video: bool = True,
    extract_audio: bool = True,
    extract_images: bool = True,
    progress_cb=None,
) -> list[Segment]:
    """Extract media assets for each segment.

    Returns copies of the input segments with ``audio``/``image``/``video``
    set to the newly written artifacts. An artifact the caller already had is
    kept when its extraction is skipped or fails, so a failed extraction never
    silently discards a previous one.

    ``progress_cb``, if given, is called as ``progress_cb(frac, desc)`` after
    each segment is processed so slow extractions on long videos are visible
    to the caller instead of appearing to hang.
    """
    os.makedirs(output_dir, exist_ok=True)
    processed: list[Segment] = []
    total = max(len(segments), 1)
    for index, segment in enumerate(segments, start=1):
        base = f"segment_{segment.id:03d}"
        audio = segment.audio
        image = segment.image
        video = segment.video

        if extract_video:
            output_path = os.path.join(output_dir, f"{base}_video.mp4")
            if extract_video_segment(video_path, segment.start_time, segment.end_time, output_path):
                video = ArtifactRef(kind="video", path=output_path)
        if extract_audio:
            output_path = os.path.join(output_dir, f"{base}_audio.wav")
            if extract_audio_segment(video_path, segment.start_time, segment.end_time, output_path):
                audio = ArtifactRef(kind="audio", path=output_path)
        if extract_images:
            output_path = os.path.join(output_dir, f"{base}_slide.png")
            midpoint = (segment.start_time + segment.end_time) / 2.0
            if extract_slide_image(video_path, midpoint, output_path):
                image = ArtifactRef(kind="image", path=output_path)

        processed.append(replace(segment, audio=audio, image=image, video=video))
        if progress_cb:
            progress_cb(index / total, f"Extracted media for segment {segment.id} ({index}/{total})")
    return processed
