# Scripts

Helper scripts for dataset preparation, evaluation, and ground-truth transcription. All scripts are run from the repository root.

## Dataset

### `download_videos.py`

Bulk downloader for the Underline HLS lecture videos. Resumable — completed videos are skipped on re-run, partial downloads restart cleanly. Requires `ffmpeg` on PATH.

```bash
python scripts/download_videos.py                  # download all 87 videos
python scripts/download_videos.py --limit 2         # smoke-test with 2
```

Videos are saved to `data/videos/`. The source URLs are listed in `data/video_links.txt`.

### `create_dataset.py`

Fetches metadata and transcripts from the Underline API for each video link, builds a `dataset_index.json` catalog with lecture titles, speaker info, and downloaded VTT subtitle files.

```bash
python scripts/create_dataset.py --links-file data/video_links.txt --output-dir data/dataset
```

### `preprocess_vtt.py`

Aligns VTT subtitle cues from Underline transcripts with ReLecture segment boundaries to produce per-segment ground-truth text for WER evaluation.

```bash
python scripts/preprocess_vtt.py \
    --segmentation-manifest path/to/manifests/segmentation.v1.json \
    --vtt-file path/to/transcript.vtt \
    --output path/to/ground_truth.json
```

## Evaluation

### `eval_runner.py`

Resumable evaluation runner. Each invocation targets one (backend, strategy, audience) configuration and processes all videos. Progress is persisted after every video, so runs survive crashes and re-running skips completed items.

```bash
# Standard run
uv run python scripts/eval_runner.py \
    --backend qwen3 --strategy independent \
    --audience-slug cs_students \
    --target-audience "students with Computer Science background"

# Visual impairment (requires vision server on localhost:8005)
uv run python scripts/eval_runner.py \
    --backend qwen3 --strategy independent \
    --audience-slug visual_impairment \
    --target-audience "visually impaired students" \
    --accessibility visual_impairment --describe-visuals

# Check progress
uv run python scripts/eval_runner.py --backend qwen3 --strategy independent \
    --audience-slug cs_students --target-audience "" --status

# Retry failed videos
uv run python scripts/eval_runner.py ... --retry-failed
```

### `batch_run.py`

Runs the pipeline over a dataset index with a JSON plan file that specifies which stages, strategies, backends, and audience profiles to execute.

```bash
uv run python scripts/batch_run.py \
    --dataset data/dataset/dataset_index.json \
    --plan plan.json \
    --output-root data/processed/
```

## Case Study

### `run_case_study.sh`

Shell script that runs the full pipeline across a matrix of videos, backends, strategies, and audience profiles. Backend-specific synthesis parameters are loaded from `case_study/configs/`.

```bash
bash scripts/run_case_study.sh
```

### `run_case_study.py`

Minimal Python wrapper that runs a single case-study configuration with real-time stdout streaming. Edit the command inside the script to match your target video and config.

```bash
python scripts/run_case_study.py
```

## Ground Truth

### `groundtruth/gemini_transcribe.py`

Generates ground-truth transcriptions using Gemini. Videos longer than 5 minutes are split into chunks before upload to avoid hallucination. Output is written to `eval_results/groundtruth/transcription/gemini/`.

```bash
python -m scripts.groundtruth.gemini_transcribe \
    --video data/videos/72694.mp4 --api-key YOUR_KEY
```

### `groundtruth/gemini_flash_transcribe.py`

Same as above but uses Gemini Flash with an audio-only pipeline. Output goes to `eval_results/groundtruth/transcription/gemini-flash/`.

```bash
python -m scripts.groundtruth.gemini_flash_transcribe \
    --video data/videos/72694.mp4 --api-key YOUR_KEY

# All videos, skip existing
python -m scripts.groundtruth.gemini_flash_transcribe --all --api-key YOUR_KEY
```
