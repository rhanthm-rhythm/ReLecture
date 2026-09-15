# ReLecture

**ReLecture** is a manifest-driven pipeline that re-narrates recorded lectures for different audiences while preserving the original speaker's voice. Given a lecture video, it segments the recording by slide transitions, transcribes the audio, adapts the transcript for a target audience or accessibility need, synthesizes new speech with voice cloning, and assembles a final video with subtitles.

The system is designed for reproducibility: every stage writes a versioned JSON manifest, making the pipeline resumable, auditable, and decoupled from any particular UI.

## Pipeline

```
 lecture.mp4
      |
 [1. Segment]      Detect slide transitions, split into segments
      |
 [2. Transcribe]   Whisper ASR on each segment
      |
 [3. Transform]    LLM rewrites transcript for target audience
      |
 [4. Synthesize]   Voice-cloning TTS re-narrates each segment
      |
 [5. Assemble]     Stitch segments into final MP4 + SRT subtitles
      |
 adapted_lecture.mp4
```

## Quick Start

```bash
# Install (requires Python 3.10; uv downloads it automatically if not present)
uv sync

# Start the required services (see servers/ for setup guides)
# At minimum: Whisper + one TTS backend

# Run the full pipeline
uv run python -m relecture run \
  --source-video data/videos/lecture.mp4 \
  --target-audience "students without a CS background" \
  --strategy two_pass \
  --backend qwen3 \
  --mode full

# Or launch the Gradio demo UI
uv run python -m relecture ui
```

## Services

Start the services needed by your configuration before running the pipeline.

| Service | Default endpoint | When required |
|---|---|---|
| Whisper | `localhost:5001/transcribe` | Always |
| Transformation LLM | `api.studio.nebius.com/v1/` | Always |
| Qwen3 TTS | `localhost:5000` | `--backend qwen3` |
| Chatterbox | `localhost:5002` | `--backend chatterbox` |
| CosyVoice | `localhost:5003` | `--backend cosyvoice` |
| Vision model | `localhost:8005/v1` | `--accessibility visual_impairment` |
| FFmpeg / FFprobe | On `PATH` | Always |

Each TTS backend has its own setup guide and server script under [`servers/`](servers/).

## Configuration

Copy `.env.example` to `.env` and fill in your API keys. All settings are loaded from environment variables by `relecture/config.py`.

| Variable | Default | Description |
|---|---|---|
| `NEBIUS_API_KEY` | *required* | API key for the transformation LLM |
| `LLM_BASE_URL` | `https://api.studio.nebius.com/v1/` | LLM endpoint |
| `LLM_MODEL_NAME` | `openai/gpt-oss-120b` | LLM model |
| `WHISPER_ENDPOINT` | `http://localhost:5001/transcribe` | Whisper server |
| `QWEN3_TTS_ENDPOINT` | `http://localhost:5000` | Qwen3 TTS server |
| `CHATTERBOX_ENDPOINT` | `http://localhost:5002` | Chatterbox TTS server |
| `COSYVOICE_ENDPOINT` | `http://localhost:5003` | CosyVoice TTS server |
| `VISION_BASE_URL` | `http://localhost:8005/v1` | Vision model endpoint |
| `VISION_MODEL_NAME` | `Qwen/Qwen3-VL-8B-Instruct` | Vision model |
| `VISION_API_KEY` | `EMPTY` | Vision endpoint API key |

## CLI Reference

```bash
uv run python -m relecture --help
```

The pipeline can be run end-to-end or one stage at a time:

```bash
# End-to-end
uv run python -m relecture run --source-video lecture.mp4 --mode full

# Stage-by-stage
uv run python -m relecture segment    --source-video lecture.mp4
uv run python -m relecture transcribe --project project.json
uv run python -m relecture transform  --project project.json --strategy two_pass
uv run python -m relecture synthesize --project project.json --backend qwen3 --mode full
uv run python -m relecture assemble   --project project.json
```

### Transformation options

**Audience background:** `none`, `cs_background`, `custom` (with `--custom-instructions`)

**Accessibility:** `none`, `visual_impairment` (generates slide descriptions via a vision model)

**Strategy:** `independent`, `full_context`, `two_pass`, `sliding_window`

**TTS backend:** `qwen3`, `chatterbox`, `cosyvoice`

## Gradio Demo

```bash
uv run python -m relecture ui
```

Open `http://localhost:7860`. The UI provides upload, configuration, service readiness checks, stage-by-stage progress, before/after transcript comparison, audio preview, and the final assembled video. Two modes are available: **preview** (first segment only, no assembly) and **full** (all segments + final MP4).

## Project Layout

Each pipeline run creates a self-contained project directory:

```
project_dir/
  project.json                  # Project manifest
  manifests/
    segmentation.v1.json        # Per-stage versioned manifests
    transcription.v1.json
    transformation.v1.json
    synthesis.v1.json
    assembly.v1.json
  artifacts/
    audio/                      # Extracted audio segments
    images/                     # Slide images
    synthesis/                  # Synthesized audio
    final/                      # Assembled video + subtitles
```

## Repository Structure

```
relecture/              Main Python package
  stages/               Pipeline stage implementations
  backends/             TTS backend clients (Qwen3, Chatterbox, CosyVoice)
  eval/                 Evaluation metrics (WER, speaker similarity, LLM-as-judge)
servers/                Standalone model server scripts and setup guides
  whisper/              Whisper ASR server
  qwen3/                Qwen3-TTS voice cloning server
  chatterbox/           Chatterbox TTS server
  cosyvoice/            CosyVoice TTS server
scripts/                Dataset, evaluation, and case study helper scripts
tests/                  Offline unit tests and optional live server tests
case_study/             Case study configurations
data/                   Dataset video links and download tooling
assets/                 Narrator voice sample for visual-accessibility mode
eval_results/           Ground-truth transcriptions for evaluation
```

## Testing

Offline tests do not require GPU model servers:

```bash
uv run python -m unittest discover tests -v
```

Live server tests under `tests/servers/` require the corresponding service to be running and should be run intentionally.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Citation

```bibtex
@inproceedings{do2027relecture,
  title     = {{ReLecture}: A Manifest-Driven Pipeline for Audience-Adaptive
               Lecture Re-Narration with Voice Cloning},
  author    = {Do, Nha and Hanafi, Abdelrahman and Vu, Thang},
  booktitle = {Proceedings of the 18th Conference of the European Chapter of
               the Association for Computational Linguistics: System Demonstrations},
  year      = {2027},
  publisher = {Association for Computational Linguistics},
}
```
