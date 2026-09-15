#!/usr/bin/env python3
"""Run a single case-study configuration with real-time progress output.

Edit the command below to match the video and configuration you want to run.
"""
import subprocess
import sys

cmd = [
    "python", "-m", "relecture", "run",
    "--source-video", "data/videos/43590.mp4",
    "--project-name", "case_study_43590_qwen3_independent_cs_students",
    "--background", "none",
    "--accessibility", "none",
    "--target-audience", "students with Computer Science background",
    "--strategy", "independent",
    "--mode", "full",
    "--backend", "qwen3",
    "--output-filename", "lecture_final.mp4"
]

print("=" * 60)
print("STARTING CASE STUDY RUN")
print("=" * 60)
print(f"Command: {' '.join(cmd)}")
print()

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

for line in process.stdout:
    print(line, end="")
    sys.stdout.flush()

process.wait()
print()
print("=" * 60)
print(f"EXIT CODE: {process.returncode}")
print("=" * 60)
