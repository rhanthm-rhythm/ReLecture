#!/usr/bin/env python3
"""Resumable evaluation runner for ReLecture.

Each invocation targets one (backend, strategy, audience) config and processes
all 87 videos.  Progress is saved after every video so runs survive any crash.
Re-running the same command skips completed videos automatically.

Usage:
    uv run python scripts/eval_runner.py \
        --backend qwen3 --strategy independent \
        --audience-slug cs_students \
        --target-audience "students with Computer Science background"

    # Visual impairment (needs vision server on localhost:8005):
    uv run python scripts/eval_runner.py \
        --backend qwen3 --strategy independent \
        --audience-slug visual_impairment \
        --target-audience "visually impaired students" \
        --accessibility visual_impairment --describe-visuals

Flags:
    --status         Show progress and exit
    --dry-run        List what would run without executing
    --retry-failed   Retry videos that failed in a previous run
    --video ID       Run a single video only (for debugging)
    --timeout SECS   Per-video timeout (default: 86400 = 24 hours)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


class _FlushHandler(logging.StreamHandler):
    """StreamHandler that flushes stdout after every record."""
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "videos"
EVAL_ROOT = PROJECT_ROOT / "eval_result"

DEFAULT_TIMEOUT = 86400  # 24 hours


def discover_videos() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob("*.mp4"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- state -----------------------------------------------------------------

def _state_path(eval_dir: Path) -> Path:
    return eval_dir / ".eval_state.json"


def load_state(eval_dir: Path) -> dict:
    p = _state_path(eval_dir)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"completed": [], "failed": {}}


def save_state(eval_dir: Path, st: dict) -> None:
    p = _state_path(eval_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2)
        f.write("\n")
    tmp.replace(p)


def is_completed_on_disk(run_dir: Path) -> bool:
    return (run_dir / "manifests" / "assembly.v1.json").exists()


# -- logging ---------------------------------------------------------------

def setup_logging(eval_dir: Path) -> logging.Logger:
    eval_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("eval")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(eval_dir / "eval_run.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    ch = _FlushHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    return log


# -- helpers ---------------------------------------------------------------

_STAGES = [
    ("assembly",       "assembly.v1.json"),
    ("synthesis",      "synthesis.v1.json"),
    ("transformation", "transformation.v1.json"),
    ("transcription",  "transcription.v1.json"),
    ("segmentation",   "segmentation.v1.json"),
]

def _current_stage(run_dir: Path) -> str:
    """Return the most recently written stage name, or 'starting'."""
    manifests = run_dir / "manifests"
    for stage_name, filename in _STAGES:
        if (manifests / filename).exists():
            return stage_name
    return "starting"


# -- single run ------------------------------------------------------------

def run_single(
    video_id: str,
    args: argparse.Namespace,
    eval_dir: Path,
    log: logging.Logger,
) -> bool:
    run_dir = eval_dir / video_id
    run_dir.mkdir(parents=True, exist_ok=True)

    project_file = run_dir / "project.json"
    video_path = DATA_DIR / f"{video_id}.mp4"
    run_log = run_dir / "run.log"

    if not video_path.exists():
        log.error("  Video not found: %s", video_path)
        return False

    cmd = [
        "uv", "run", "python", "-m", "relecture", "run",
        "--source-video", str(video_path),
        "--project", str(project_file),
        "--background", "none",
        "--accessibility", args.accessibility,
        "--target-audience", args.target_audience,
        "--strategy", args.strategy,
        "--mode", "full",
        "--backend", args.backend,
        "--output-filename", "lecture_final.mp4",
    ]
    if args.describe_visuals:
        cmd.append("--describe-visuals")

    t0 = time.monotonic()
    returncode = None
    try:
        with open(run_log, "a", encoding="utf-8") as lf:
            lf.write(f"\n{'=' * 60}\n")
            lf.write(f"START: {now_iso()}\n")
            lf.write(f"CMD:   {' '.join(cmd)}\n")
            lf.write(f"{'=' * 60}\n\n")
            lf.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=lf,
                stderr=subprocess.STDOUT,
            )
            deadline = t0 + args.timeout
            while proc.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    proc.kill()
                    proc.wait()
                    log.error("  TIMEOUT after %ds", args.timeout)
                    return False
                try:
                    proc.wait(timeout=min(60, remaining))
                except subprocess.TimeoutExpired:
                    elapsed_min = (time.monotonic() - t0) / 60
                    stage = _current_stage(run_dir)
                    log.info("  [%s] stage=%-14s  %.0f min elapsed", video_id, stage, elapsed_min)
            returncode = proc.returncode
    except Exception as exc:
        log.error("  EXCEPTION: %s", exc)
        return False

    elapsed = time.monotonic() - t0

    if returncode != 0:
        log.error("  FAILED (exit %d) in %.0fs — see %s", returncode, elapsed, run_log)
        return False

    if not is_completed_on_disk(run_dir):
        log.warning("  Exit 0 but no assembly manifest — marking failed")
        return False

    log.info("  OK in %.0fs", elapsed)
    return True


# -- status ----------------------------------------------------------------

def show_status(args: argparse.Namespace) -> None:
    eval_dir = EVAL_ROOT / args.backend / args.strategy / args.audience_slug
    st = load_state(eval_dir)
    videos = discover_videos()
    n_done = len(st["completed"])
    n_fail = len(st.get("failed", {}))

    print(f"Config: {args.backend} / {args.strategy} / {args.audience_slug}")
    print(f"Videos:    {len(videos)}")
    print(f"Completed: {n_done}")
    print(f"Failed:    {n_fail}")
    print(f"Remaining: {len(videos) - n_done}")
    if st.get("failed"):
        print("\nFailed videos:")
        for vid, info in st["failed"].items():
            print(f"  {vid}  (attempts: {info.get('attempts', '?')}, last: {info.get('timestamp', '?')})")


# -- main ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ReLecture evaluation runner")
    parser.add_argument("--backend", required=True, choices=["qwen3", "chatterbox", "cosyvoice"])
    parser.add_argument("--strategy", required=True, choices=["independent", "two_pass"])
    parser.add_argument("--audience-slug", required=True,
                        help="Directory name for this audience config")
    parser.add_argument("--target-audience", required=True,
                        help="Value passed to --target-audience in the pipeline")
    parser.add_argument("--accessibility", default="none",
                        choices=["none", "visual_impairment"])
    parser.add_argument("--describe-visuals", action="store_true")
    parser.add_argument("--video", help="Run a single video ID only")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry previously failed videos")
    parser.add_argument("--status", action="store_true",
                        help="Show progress and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would run without executing")
    args = parser.parse_args()

    if args.status:
        show_status(args)
        return

    eval_dir = EVAL_ROOT / args.backend / args.strategy / args.audience_slug
    log = setup_logging(eval_dir)
    st = load_state(eval_dir)

    videos = [args.video] if args.video else discover_videos()

    # Auto-detect runs completed on disk but not yet in state
    for vid in videos:
        if vid not in st["completed"] and is_completed_on_disk(eval_dir / vid):
            st["completed"].append(vid)
            log.info("Detected existing completed run: %s", vid)
    save_state(eval_dir, st)

    # Build work list
    work: list[str] = []
    for vid in videos:
        if vid in st["completed"]:
            continue
        if vid in st.get("failed", {}) and not args.retry_failed:
            continue
        work.append(vid)

    total = len(videos)
    config_label = f"{args.backend} / {args.strategy} / {args.audience_slug}"

    log.info("=" * 60)
    log.info("ReLecture Eval: %s", config_label)
    log.info("=" * 60)
    log.info("Target audience: %s", args.target_audience)
    log.info("Accessibility: %s  Describe visuals: %s", args.accessibility, args.describe_visuals)
    log.info("Videos: %d  |  Completed: %d  |  To run: %d", total, len(st["completed"]), len(work))

    if args.dry_run:
        log.info("DRY RUN — would execute:")
        for vid in work:
            log.info("  %s", vid)
        return

    if not work:
        log.info("Nothing to do.")
        return

    log.info("")

    ok = 0
    fail = 0

    for i, vid in enumerate(work, 1):
        log.info("[%d/%d] %s", i, len(work), vid)

        if run_single(vid, args, eval_dir, log):
            st["completed"].append(vid)
            st.get("failed", {}).pop(vid, None)
            ok += 1
        else:
            st.setdefault("failed", {})[vid] = {
                "error": "see run.log",
                "timestamp": now_iso(),
                "attempts": st.get("failed", {}).get(vid, {}).get("attempts", 0) + 1,
            }
            fail += 1

        save_state(eval_dir, st)

    log.info("")
    log.info("=" * 60)
    log.info("Session done: %d ok, %d failed  |  Overall: %d/%d", ok, fail, len(st["completed"]), total)
    log.info("=" * 60)

    st["last_run"] = now_iso()
    save_state(eval_dir, st)
    sys.exit(1 if fail > 0 else 0)


if __name__ == "__main__":
    main()
