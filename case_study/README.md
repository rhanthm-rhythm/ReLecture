# ReLecture Case Study

This directory contains configuration and data for the case-study evaluation described in the paper.

## Structure

```
case_study/
├── configs/
│   ├── qwen3.json
│   ├── chatterbox.json
│   └── cosyvoice.json
├── videos/              # Place case-study videos here
└── README.md
```

Backend-specific TTS parameters (speed, temperature, top_p, repetition_penalty) are stored in `configs/`. The evaluation runner and case-study scripts in `scripts/` read these configs automatically.

## Videos

Three videos are used in the case study: **43590**, **65781**, **68145**. Place the corresponding `.mp4` files in `videos/` before running. See `data/video_links.txt` for download URLs.

## Evaluation Matrix

Each video is evaluated across:

- **Strategies:** `independent`, `two_pass`
- **Target audiences:** CS students, linguistics students, visually impaired students
- **TTS backends:** Qwen3, Chatterbox, CosyVoice

Results are written by `scripts/eval_runner.py` into the `eval_result/` directory at the project root, organized as `eval_result/<backend>/<strategy>/<audience>/<video_id>/`.

## Collected Metrics

From each run the pipeline preserves:

1. Versioned JSON manifests from each stage
2. Per-stage latency (from run log timestamps)
3. Segment counts and audio durations
4. Before/after transcript pairs (for qualitative analysis)
5. Final assembled video and SRT subtitles

## Notes

- Voice reference audio is automatically extracted from each source video (10 s max).
- The narrator voice in `assets/narrator_voice.wav` is used for visual-impairment scene descriptions.
- WER evaluation requires ground-truth transcriptions; see `eval_results/groundtruth/` for Gemini-produced references.
