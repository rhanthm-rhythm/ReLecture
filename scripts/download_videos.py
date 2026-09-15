"""Bulletproof bulk downloader for the Underline HLS videos listed in data/video_links.txt.

Resumability model
------------------
* Completed videos are kept as ``<video_id>.mp4`` and are never re-downloaded.
* In-progress downloads go to ``<video_id>.mp4.part`` and are only renamed on
  success, so a partial file is never mistaken for a finished one.
* ffmpeg cannot resume an interrupted HLS stream mid-file, so a leftover
  ``.part`` file is deleted and that video restarts from the beginning on the
  next run. All *other* videos are untouched.
* Per-video state (done/failed + error) is persisted to ``_download_state.json``
  after every item, so you can stop with Ctrl-C at any time and re-run the
  exact same command to continue where you left off.
* A link that errors (dead source, 404, network drop) is retried with
  exponential backoff, then logged as failed and the script continues with the
  rest of the database. Re-running picks failed items up again.
* A live progress bar (percent, downloaded/total duration, speed) is shown
  while each video downloads. Total duration is probed from the stream first;
  if probing fails the bar degrades to an elapsed-time display. When stdout is
  not a terminal (e.g. nohup), progress is printed as periodic lines instead.

Usage
-----
    python scripts/download_videos.py                      # download everything
    python scripts/download_videos.py --limit 2            # smoke test
    python scripts/download_videos.py --retries 10         # more patient retries

Requires: ffmpeg on PATH (ffprobe optional, used to verify completed files).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LINKS_FILE = REPO_ROOT / "data" / "video_links.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "videos"
STATE_FILENAME = "_download_state.json"
FAILED_LOG_FILENAME = "failed_downloads.log"

VIDEO_ID_RE = re.compile(r"/video/(?P<video_id>\d+)/file/abr/")
RW_TIMEOUT_US = 30_000_000  # abort the connection after 30s of network silence


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download all videos listed in a links file (resumable).")
    parser.add_argument("--links-file", default=str(DEFAULT_LINKS_FILE),
                        help="Text file with one URL per line (default: data/video_links.txt).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Where videos are written (default: <repo>/data/videos).")
    parser.add_argument("--retries", type=int, default=5,
                        help="Attempts per video before marking it failed (default: 5).")
    parser.add_argument("--limit", type=int, help="Only process the first N links.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download videos even if the final .mp4 already exists.")
    return parser.parse_args(argv)


def read_links(path: Path, limit: int | None = None) -> list[tuple[str, str]]:
    """Return a de-duplicated list of (video_id, url) preserving file order."""
    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        url = raw.strip()
        if not url or url.startswith("#"):
            continue
        match = VIDEO_ID_RE.search(url)
        if not match:
            print(f"[warn] line {line_no}: cannot extract video id, skipping: {url}")
            continue
        video_id = match.group("video_id")
        if video_id in seen:
            continue
        seen.add(video_id)
        items.append((video_id, url))
        if limit is not None and len(items) >= limit:
            break
    return items


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[warn] corrupt state file {state_path}, starting fresh")
    return {}


def save_state(state_path: Path, state: dict) -> None:
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(state_path)  # atomic on both Windows and Linux


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fmt_hms(seconds: float) -> str:
    total = int(max(seconds, 0))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def parse_ffmpeg_time(value: str) -> float | None:
    """Parse ffmpeg's out_time format (HH:MM:SS.ffffff) into seconds."""
    try:
        hours, minutes, rest = value.strip().split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(rest)
    except ValueError:
        return None


def probe_duration(url: str) -> float | None:
    """Ask ffprobe for the stream's total duration (seconds), or None."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error",
             "-reconnect", "1", "-rw_timeout", str(RW_TIMEOUT_US),
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", url],
            capture_output=True, text=True, timeout=120,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


def render_progress(out_s: float, duration_s: float | None, speed: str,
                    is_tty: bool, state: dict) -> None:
    if duration_s:
        pct = min(out_s / duration_s * 100, 100.0)
        filled = int(28 * pct / 100)
        bar = "#" * filled + "-" * (28 - filled)
        text = f"  [{bar}] {pct:5.1f}%  {fmt_hms(out_s)}/{fmt_hms(duration_s)}  speed {speed}"
    else:
        pct = None
        text = f"  downloaded {fmt_hms(out_s)}  speed {speed}"
    if is_tty:
        sys.stdout.write("\r" + text + "\x1b[K")
        sys.stdout.flush()
    else:
        # Log-friendly mode: one line per 10% (or per 5 minutes without duration).
        bucket = int(pct // 10) if pct is not None else int(out_s // 300)
        if bucket > state.get("bucket", -1):
            state["bucket"] = bucket
            print(text, flush=True)


def build_ffmpeg_cmd(url: str, destination: Path) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-nostats",
        "-progress", "pipe:1",
        # Reconnect automatically when the HTTP connection drops mid-stream.
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_at_eof", "1",
        "-reconnect_delay_max", "30",
        # Abort a completely stalled socket instead of hanging forever.
        "-rw_timeout", str(RW_TIMEOUT_US),
        "-y",
        "-i", url,
        # Output goes to a .part file, so the mp4 muxer must be forced —
        # ffmpeg cannot guess the format from that extension.
        "-f", "mp4",
        "-c", "copy",
        "-bsf:a", "aac_adtstoasc",
        str(destination),
    ]


def verify_output(path: Path) -> bool:
    """Sanity-check a finished download: non-empty, and probeable if ffprobe exists."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return True
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False
    try:
        return float(result.stdout.strip()) > 0
    except ValueError:
        return False


def download_one(video_id: str, url: str, output_dir: Path, retries: int,
                 duration_s: float | None) -> tuple[bool, str | None]:
    final_path = output_dir / f"{video_id}.mp4"
    part_path = output_dir / f"{video_id}.mp4.part"
    last_error: str | None = None
    is_tty = sys.stdout.isatty()

    for attempt in range(1, retries + 1):
        if part_path.exists():
            part_path.unlink()  # stale partial from an interrupted run
        print(f"  attempt {attempt}/{retries} -> {part_path.name}", flush=True)
        try:
            proc = subprocess.Popen(
                build_ffmpeg_cmd(url, part_path),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )
        except Exception as exc:  # e.g. ffmpeg vanished mid-run
            last_error = f"failed to launch ffmpeg: {exc}"
        else:
            out_s = 0.0
            speed = "?"
            progress_state: dict = {}
            try:
                assert proc.stdout is not None
                for line in proc.stdout:  # -progress pipe:1 emits key=value lines
                    key, _, value = line.partition("=")
                    if key == "out_time":
                        parsed = parse_ffmpeg_time(value)
                        if parsed is not None:
                            out_s = parsed
                            render_progress(out_s, duration_s, speed, is_tty, progress_state)
                    elif key == "speed":
                        speed = value.strip()
                proc.wait()
            except KeyboardInterrupt:
                proc.kill()
                proc.wait()
                if is_tty:
                    sys.stdout.write("\n")
                raise
            if is_tty and out_s > 0:
                sys.stdout.write("\n")
            stderr = proc.stderr.read() if proc.stderr else ""

            if proc.returncode == 0 and verify_output(part_path):
                part_path.replace(final_path)
                return True, None
            lines = (stderr or "").strip().splitlines()
            last_error = lines[-1] if lines else f"ffmpeg exited with code {proc.returncode}"

        wait = min(2 ** attempt, 120)
        print(f"  failed: {last_error} (retrying in {wait}s)", flush=True)
        time.sleep(wait)

    return False, last_error


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not found on PATH. Install it first (e.g. 'sudo apt install ffmpeg' "
              "or 'winget install ffmpeg').", file=sys.stderr)
        return 2

    links_file = Path(args.links_file)
    if not links_file.exists():
        print(f"ERROR: links file not found: {links_file}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / STATE_FILENAME
    failed_log = output_dir / FAILED_LOG_FILENAME
    state = load_state(state_path)

    items = read_links(links_file, limit=args.limit)
    if not items:
        print("No valid links found.", file=sys.stderr)
        return 2

    todo: list[tuple[str, str]] = []
    for video_id, url in items:
        final_path = output_dir / f"{video_id}.mp4"
        if state.get(video_id, {}).get("status") == "done" and final_path.exists() and not args.overwrite:
            continue
        if final_path.exists() and not args.overwrite:
            # File exists but state is missing (e.g. copied in manually): trust it.
            state[video_id] = {"status": "done", "url": url, "finished_at": utc_now()}
            save_state(state_path, state)
            continue
        todo.append((video_id, url))

    print(f"{len(items)} links total, {len(items) - len(todo)} already done, {len(todo)} to download.")
    print(f"Output: {output_dir}\nState:  {state_path}\n")

    for index, (video_id, url) in enumerate(todo, start=1):
        print(f"[{index}/{len(todo)}] video {video_id}", flush=True)
        duration_s = probe_duration(url)
        if duration_s:
            print(f"  total duration: {fmt_hms(duration_s)}", flush=True)
        ok, error = download_one(video_id, url, output_dir, retries=args.retries,
                                 duration_s=duration_s)
        state[video_id] = {
            "status": "done" if ok else "failed",
            "url": url,
            "finished_at": utc_now() if ok else None,
            "error": None if ok else error,
        }
        save_state(state_path, state)
        if not ok:
            with failed_log.open("a", encoding="utf-8") as fh:
                fh.write(f"{utc_now()}\t{video_id}\t{url}\t{error}\n")
            print(f"  GIVING UP on {video_id} (logged to {failed_log.name}); continuing.\n", flush=True)

    done = sum(1 for v in state.values() if v.get("status") == "done")
    failed = len(state) - done
    print(f"\nFinished. done={done} failed={failed}")
    if failed:
        print(f"Failed items are recorded in {failed_log} — just re-run the same command to retry them.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
