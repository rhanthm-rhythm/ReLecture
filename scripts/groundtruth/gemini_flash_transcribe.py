"""
Ground-truth transcription via Gemini 3.7 Flash (audio pipeline).

Videos are split into 5-min audio chunks with ffmpeg before upload so the
model doesn't hallucinate past the ~5-minute mark.  Each chunk is transcribed
independently, timestamps are offset, then everything is merged.

Output goes to:  eval_results/groundtruth/transcription/gemini-flash/<stem>.json
                 eval_results/groundtruth/transcription/gemini-flash/<stem>.txt

Usage (single):
    python -m groundtruth.gemini_flash_transcribe --video data/videos/72694.mp4 --api-key KEY

Usage (batch):
    python -m groundtruth.gemini_flash_transcribe --videos data/videos/*.mp4 --api-key KEY

Usage (all, skip existing):
    python -m groundtruth.gemini_flash_transcribe --all --api-key KEY
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path


PROMPT = (
    "Give me the timestamped transcription of this video. "
    "Format every line as:  HH:MM:SS --> HH:MM:SS  <text>"
)

MODEL = "gemini-3-flash-preview"
CHUNK_SECONDS = 300
OUTPUT_ROOT = Path("eval_results/groundtruth/transcription/gemini-flash")
RPM_LIMIT = 3
MIN_INTERVAL = 60.0 / RPM_LIMIT  # 20s between generate_content calls

_last_api_call: float = 0.0


# ── Timestamp helpers ─────────────────────────────────────────────────────────

def _ts_to_secs(ts: str) -> float:
    parts = ts.replace(",", ".").split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def _secs_to_ts(s: float) -> str:
    s = max(0.0, s)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:05.2f}"


# ── Video / audio helpers ─────────────────────────────────────────────────────

def _video_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _split_audio(path: Path, chunk_secs: int, tmp_dir: str) -> list[tuple[Path, float]]:
    duration = _video_duration(path)
    n_chunks = int(duration // chunk_secs) + (1 if duration % chunk_secs else 0)
    print(f"    [ffmpeg] extracting & splitting into {n_chunks} audio chunks ...", flush=True)
    chunks = []
    start = 0.0
    idx = 0
    while start < duration:
        out = Path(tmp_dir) / f"chunk_{idx:03d}.mp3"
        t0 = time.time()
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-i", str(path),
             "-t", str(chunk_secs), "-vn", "-acodec", "mp3", "-q:a", "2", str(out)],
            capture_output=True, check=True,
        )
        print(f"    [ffmpeg] chunk {idx+1}/{n_chunks} done ({time.time()-t0:.1f}s)  {out.stat().st_size/1e6:.1f} MB", flush=True)
        chunks.append((out, start))
        start += chunk_secs
        idx += 1
    return chunks


def _extract_audio(path: Path, tmp_dir: str) -> Path:
    out = Path(tmp_dir) / f"{path.stem}.mp3"
    t0 = time.time()
    print(f"    [ffmpeg] extracting audio ...", flush=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-vn", "-acodec", "mp3", "-q:a", "2", str(out)],
        capture_output=True, check=True,
    )
    print(f"    [ffmpeg] done ({time.time()-t0:.1f}s)  {out.stat().st_size/1e6:.1f} MB", flush=True)
    return out


# ── Gemini helpers ────────────────────────────────────────────────────────────

def _wait_for_file(client, file_obj, poll_interval: float = 5.0, timeout: float = 900.0):
    deadline = time.time() + timeout
    while True:
        f = client.files.get(name=file_obj.name)
        state = getattr(f, "state", None)
        state_str = state.name if hasattr(state, "name") else str(state)
        if state_str == "ACTIVE":
            return f
        if state_str == "FAILED":
            raise RuntimeError(f"File processing failed: {f}")
        if time.time() > deadline:
            raise TimeoutError("Timed out waiting for file to become ACTIVE")
        time.sleep(poll_interval)


def _generate_with_retry(client, genai_types, file_uri: str, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=[
                    genai_types.Content(parts=[
                        genai_types.Part(file_data=genai_types.FileData(file_uri=file_uri)),
                        genai_types.Part(text=PROMPT),
                    ])
                ],
            )
        except Exception as e:
            err = str(e)
            if attempt < max_retries - 1 and ("429" in err or "503" in err):
                if "429" in err:
                    m = re.search(r"retryDelay.*?(\d+)s", err)
                    wait = int(m.group(1)) + 5 if m else 65
                    print(f"    [429] quota hit, waiting {wait}s before retry {attempt+2}/{max_retries} ...", flush=True)
                else:
                    wait = 30 * (attempt + 1)
                    print(f"    [503] model overloaded, waiting {wait}s before retry {attempt+2}/{max_retries} ...", flush=True)
                time.sleep(wait)
            else:
                raise


def _transcribe_chunk(client, genai_types, chunk_path: Path, offset_secs: float) -> tuple[list[dict], str]:
    t0 = time.time()
    print(f"    [upload] {chunk_path.name} ...", flush=True)
    with open(chunk_path, "rb") as fh:
        uploaded = client.files.upload(
            file=fh,
            config=genai_types.UploadFileConfig(
                display_name=chunk_path.name,
                mime_type="audio/mpeg",
            ),
        )
    print(f"    [upload] done ({time.time()-t0:.1f}s) — waiting for Gemini to process ...", flush=True)
    t1 = time.time()
    ready = _wait_for_file(client, uploaded)
    print(f"    [ready]  ({time.time()-t1:.1f}s) — requesting transcription ...", flush=True)
    t2 = time.time()

    response = _generate_with_retry(client, genai_types, file_uri=ready.uri)
    raw = response.text or ""
    print(f"    [transcribe] done ({time.time()-t2:.1f}s)  {len(raw)} chars", flush=True)

    segments = _parse_segments(raw)
    for seg in segments:
        seg["start"] = _secs_to_ts(_ts_to_secs(seg["start"]) + offset_secs)
        seg["end"]   = _secs_to_ts(_ts_to_secs(seg["end"])   + offset_secs)

    try:
        client.files.delete(name=ready.name)
    except Exception:
        pass

    return segments, raw


# ── Parse ─────────────────────────────────────────────────────────────────────

def _parse_segments(text: str) -> list[dict]:
    ts = r"\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?"
    splitter = re.compile(rf"(?=(?:\d{{1,2}}:\d{{2}}(?::\d{{2}}(?:[.,]\d+)?)?)\s*-->)")
    entry_pattern = re.compile(rf"^({ts})\s*-->\s*({ts})\s+(.*)", re.DOTALL)
    segments = []
    for chunk in splitter.split(text):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = entry_pattern.match(chunk)
        if m:
            segments.append({
                "start": m.group(1),
                "end":   m.group(2),
                "text":  m.group(3).strip(),
            })
    return segments


# ── Main entry point ──────────────────────────────────────────────────────────

def transcribe_video(
    video_path: str | Path,
    api_key: str,
    output_dir: str | Path | None = None,
) -> dict:
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError as exc:
        raise ImportError("google-genai is required: uv pip install google-genai") from exc

    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    out_dir = Path(output_dir) if output_dir else OUTPUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=api_key)

    duration = _video_duration(video_path)
    print(f"{video_path.name}  duration={duration:.1f}s  ({duration/60:.1f} min)", flush=True)

    all_segments: list[dict] = []
    raw_parts: list[str] = []

    if duration <= CHUNK_SECONDS:
        print("  Single chunk — extracting audio ...", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            audio = _extract_audio(video_path, tmp)
            segs, raw = _transcribe_chunk(client, genai_types, audio, 0.0)
        all_segments = segs
        raw_parts = [raw]
    else:
        with tempfile.TemporaryDirectory() as tmp:
            chunks = _split_audio(video_path, CHUNK_SECONDS, tmp)
            print(f"  Split into {len(chunks)} audio chunks of ~{CHUNK_SECONDS}s", flush=True)
            for i, (chunk_path, offset) in enumerate(chunks):
                print(f"  Chunk {i+1}/{len(chunks)}  offset={offset:.0f}s ...", flush=True)
                segs, raw = _transcribe_chunk(client, genai_types, chunk_path, offset)
                all_segments.extend(segs)
                raw_parts.append(raw)
                print(f"    -> {len(segs)} segments", flush=True)

    result = {
        "video":    str(video_path),
        "model":    MODEL,
        "prompt":   PROMPT,
        "duration": duration,
        "chunks":   len(raw_parts),
        "raw_text": "\n\n".join(raw_parts),
        "segments": all_segments,
    }

    stem = video_path.stem
    json_out = out_dir / f"{stem}.json"
    txt_out  = out_dir / f"{stem}.txt"

    json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_out.write_text(
        "\n".join(f"{s['start']} --> {s['end']}  {s['text']}" for s in all_segments),
        encoding="utf-8",
    )

    print(f"  Saved {len(all_segments)} segments -> {json_out}", flush=True)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Gemini 3.7 Flash ground-truth transcription")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video",  help="Single .mp4 file")
    group.add_argument("--videos", nargs="+", help="Multiple .mp4 files")
    group.add_argument("--all",    action="store_true",
                       help="Transcribe every .mp4 in data/videos/, skipping already-done ones")
    parser.add_argument("--api-key",    default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--check-dir",  default=None,
                        help="Check this dir for existing .json files instead of --output-dir (useful to fill gaps from another model's run)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        parser.error("Provide --api-key or set GEMINI_API_KEY.")

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT
    skip_dir = Path(args.check_dir) if args.check_dir else out_dir

    if args.all:
        videos_dir = Path("data/videos")
        all_videos = sorted(videos_dir.glob("*.mp4"))
        pending = [v for v in all_videos if not (skip_dir / f"{v.stem}.json").exists()]
        skipped = len(all_videos) - len(pending)
        print(f"Found {len(all_videos)} videos. {skipped} already done, {len(pending)} to process.")
        videos = [str(v) for v in pending]
    elif args.video:
        videos = [args.video]
    else:
        videos = args.videos

    if not videos:
        print("Nothing to do.")
        return

    failed = []
    for i, v in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {v}")
        try:
            transcribe_video(v, api_key, out_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append((v, str(e)))

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for v, err in failed:
            print(f"  {v}: {err}")
    else:
        print(f"\nAll {len(videos)} videos done.")


if __name__ == "__main__":
    main()
