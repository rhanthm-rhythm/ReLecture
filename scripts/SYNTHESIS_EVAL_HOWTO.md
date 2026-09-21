# Synthesis Evaluation — How to Run

Covers four metrics: **WER** (intelligibility), **Speaker Similarity** (voice cloning quality), **UTMOS** (naturalness/MOS score), and **Duration Expansion Ratio** (listening time impact).

All scripts are fail-proof and resumable — if they crash, rerun the same command with `--skip-existing` and they pick up where they stopped.

---

## 1. WER (Word Error Rate)

**What it does:** Transcribes every chunk WAV with the running Whisper service and measures WER against the text that was fed to the TTS.

**Runs on:** Windows (same machine as the pipeline). Uses `http://localhost:5001/transcribe`.

**Output:**
```
eval_result/groundtruth/synthesis/transcription/whisper/
  <videoID>/wer.chatterbox.json    ← per-chunk transcriptions + WER scores
  _aggregate.json                  ← corpus-level WER across all videos
```

### Prerequisites

Make sure the Whisper service is running (it's already running if the pipeline is active):
```bash
curl http://localhost:5001/health
```

### Run

```bash
uv run python scripts/run_wer_batch.py \
  --dataset eval_result/chatterbox/independent/visual_impairment \
  --skip-existing
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--dataset` | required | Root dir of the synthesized projects |
| `--limit` | 50 | Max videos to process (alphabetical) |
| `--skip-existing` | off | Skip videos that already have a valid result file |
| `--whisper-endpoint` | `http://localhost:5001/transcribe` | Whisper service URL |
| `--output-root` | `eval_result/groundtruth/synthesis/transcription/whisper` | Where to write results |

### Resume after crash

Same command with `--skip-existing` — it validates each existing file before trusting it. Corrupt or partial files are automatically reprocessed.

### What the output looks like

Per-video JSON (`wer.chatterbox.json`):
```json
{
  "aggregates": {
    "wer": {
      "lecture":   { "wer": 0.11, "edits": 340, "ref_words": 3100, ... },
      "narration": { "wer": 0.13, "edits": 80,  "ref_words": 600, ...  },
      "combined":  { "wer": 0.12, "edits": 420, "ref_words": 3700, ... }
    }
  },
  "per_segment": [
    {
      "segment_id": 1,
      "wer": { "lecture": {...}, "narration": {...}, "combined": {...} },
      "chunks": [
        { "chunk_index": 1, "kind": "lecture", "reference": "...", "hypothesis": "...", "wer": 0.08 }
      ]
    }
  ]
}
```

---

## 2. Speaker Similarity

**What it does:** Embeds each synthesized segment WAV and the original voice reference with resemblyzer d-vectors, then reports cosine similarity per segment.

**Runs on:** WSL / Linux (resemblyzer requires `webrtcvad` which does not compile on Windows without MSVC build tools).

**Output:**
```
eval_result/synthesis/speaker_similarity/
  <videoID>/speaker_similarity.chatterbox.json    ← per-segment cosine similarity
  _aggregate.json                                 ← mean similarity across all videos
```

### Step-by-step

**1. Open WSL**

Press `Win + R`, type `wsl`, hit Enter. Or open Windows Terminal and pick Ubuntu.

**2. Go to the project**

```bash
cd /mnt/c/Users/NHADO10/Desktop/Code2/ReLecture
```

**3. Install resemblyzer (one-time)**

```bash
pip install resemblyzer numpy
```

webrtcvad compiles cleanly on Linux so no special build tools needed.

**4. Run**

```bash
python3 scripts/run_speaker_similarity_batch.py \
  --dataset eval_result/chatterbox/independent/visual_impairment \
  --skip-existing
```

If Python can't find the `relecture` package, prefix with:
```bash
PYTHONPATH=/mnt/c/Users/NHADO10/Desktop/Code2/ReLecture python3 scripts/run_speaker_similarity_batch.py \
  --dataset eval_result/chatterbox/independent/visual_impairment \
  --skip-existing
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--dataset` | required | Root dir of the synthesized projects |
| `--limit` | 50 | Max videos to process (alphabetical) |
| `--skip-existing` | off | Skip videos that already have a valid result file |
| `--output-root` | `eval_result/synthesis/speaker_similarity` | Where to write results |

### Resume after crash

Same command with `--skip-existing`. Validates existing files before trusting them — corrupt files are reprocessed automatically.

### Resources and estimated time

| Environment | Per video | 50 videos |
|---|---|---|
| CPU only | ~1–3 min | ~50–150 min |
| GPU (WSL2 + CUDA) | ~15–40 sec | ~12–33 min |

resemblyzer auto-detects CUDA — no flag needed. If CUDA is available in your WSL2 environment it uses GPU automatically.

### Two similarity scores

The script computes two separate cosine similarity scores per segment:

- **`speaker_similarity`** — lecture chunks (`kind=lecture`) vs the cloned lecturer voice reference (`artifacts/audio/segment_001_audio_voice_ref_10s.wav`). This is the core voice-cloning quality metric.
- **`narration_similarity`** — narration chunks (`kind=narration`) vs the bundled narrator voice (`assets/narrator_voice.wav`). This measures whether the narrator track matches the expected narrator voice.

Each chunk is embedded individually; the per-segment score is the mean across all chunks of that kind in the segment.

If a segment has no chunk-level audio logged (e.g. synthesized before chunk tracking was added), it falls back to the full merged segment WAV and marks it `lecturer_similarity_source: full_segment_wav_fallback`.

### What the log looks like while running

```
[ 1/50] 37343
  lecturer ref: segment_001_audio_voice_ref_10s.wav
  narrator ref: narrator_voice.wav
  11 segments — 89 lecture chunks, 22 narration chunks
  seg 1/11  id=1  lec=0.8734  nar=0.9201
  seg 2/11  id=2  lec=0.8901  nar=0.9388
  ...
  seg 11/11  id=11  lec=0.9147  nar=0.9512
  DONE  lec=0.9012  std=0.0234  min=0.8734  max=0.9427  n=11  narr=0.9341  [42.3s elapsed, ETA 34.5 min]

[ 2/50] 43588
  ...
```

### What the output looks like

Per-video JSON (`speaker_similarity.chatterbox.json`):
```json
{
  "aggregates": {
    "speaker_similarity":  { "mean": 0.90, "std": 0.02, "min": 0.87, "max": 0.94, "n": 11 },
    "narration_similarity": { "mean": 0.93, "std": 0.01, "min": 0.91, "max": 0.95, "n": 11 }
  },
  "per_segment": [
    {
      "segment_id": 1,
      "duration_seconds": 81.6,
      "n_lecture_chunks": 7,
      "n_narration_chunks": 2,
      "speaker_similarity": 0.8734,
      "lecture_chunk_similarities": [0.87, 0.89, 0.86, 0.88, 0.87, 0.86, 0.88],
      "narration_similarity": 0.9201,
      "narration_chunk_similarities": [0.92, 0.92]
    }
  ]
}
```

---

## 3. UTMOS (Naturalness / MOS Score)

**What it does:** Predicts a Mean Opinion Score (MOS) for each synthesized segment WAV using the UTMOS22 model. Higher = more natural-sounding speech.

**Runs on:** Windows (PowerShell or cmd). Uses its own isolated venv — do NOT run from the main project venv (PyTorch CUDA kernel mismatch on Blackwell GPUs).

**GPU:** Auto-detected. Will print `[UTMOS] Device: cuda` on startup if GPU is available.

**Output:**
```
eval_result/synthesis/UTMOS/chatterbox/
  <videoID>/utmos.chatterbox.json    ← per-segment UTMOS scores
  _aggregate.json                    ← mean UTMOS across all videos
```

### Setup (one-time) — Isolated venv

```cmd
python -m venv .venv-utmos
.venv-utmos\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install soundfile
pip install git+https://github.com/tarepan/SpeechMOS@v1.0.0
```

### Run

```cmd
.venv-utmos\Scripts\activate

uv run python scripts/run_utmos_batch.py ^
    --dataset eval_result\chatterbox\independent\visual_impairment ^
    --output-root eval_result\synthesis\UTMOS\chatterbox ^
    --limit 50 --skip-existing
```

### Confirm GPU is being used

On first startup you should see:
```
[UTMOS] Device: cuda
[UTMOS] GPU: NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU
UTMOS predictor ready.
```

If it says `cpu` instead, the torch install doesn't have CUDA — reinstall with the `cu128` index URL above.

### Options

| Flag | Default | Description |
|---|---|---|
| `--dataset` | required | Root dir of the synthesized projects |
| `--output-root` | required | Where to write results |
| `--limit` | 50 | Max videos to process (alphabetical) |
| `--skip-existing` | off | Skip videos that already have a valid result file |

### Resume after crash

Same command with `--skip-existing`. Validates existing files before trusting them.

### What the log looks like

```
[ 1/50] 37343
  11 segments, 842s total audio
  seg 1/11  id=1  scoring 76.2s audio ...
  seg 1/11  id=1  UTMOS=3.8234  [1.2s]  running_mean=3.8234  ETA 12s
  seg 2/11  id=2  scoring 91.4s audio ...
  seg 2/11  id=2  UTMOS=4.1023  [1.4s]  running_mean=3.9629  ETA 10s
  ...
  DONE  mean=3.9412  std=0.2134  min=3.4021  max=4.3201  n=11  [14.3s elapsed, ETA 11.6 min]
```

### What the output looks like

Per-video JSON (`utmos.chatterbox.json`):
```json
{
  "aggregates": {
    "utmos": { "mean": 3.82, "std": 0.21, "min": 3.41, "max": 4.15, "n": 11 }
  },
  "per_segment": [
    { "segment_id": 1, "duration_seconds": 81.6, "utmos": 3.91 },
    { "segment_id": 2, "duration_seconds": 74.2, "utmos": 3.77 }
  ]
}
```

---

## 4. Duration Expansion Ratio

**What it does:** Computes the ratio of synthesized segment duration to original segment duration. No audio processing — reads numbers from the synthesis manifest only.

**Runs on:** Anywhere (CPU only, instant). Uses `uv run` from the main project.

**Output:**
```
eval_result/synthesis/duration_expansion_ratio/chatterbox/
  <videoID>/duration_expansion_ratio.chatterbox.json    ← per-segment ratios
  _aggregate.json                                       ← mean ratio across all videos
```

### Run

```bash
uv run python scripts/run_duration_expansion_ratio_batch.py \
    --dataset eval_result/chatterbox/independent/visual_impairment \
    --output-root eval_result/synthesis/duration_expansion_ratio/chatterbox \
    --limit 50 \
    --skip-existing
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--dataset` | required | Root dir of the synthesized projects |
| `--output-root` | `eval_result/synthesis/duration_expansion_ratio` | Where to write results |
| `--limit` | 50 | Max videos to process (alphabetical) |
| `--skip-existing` | off | Skip videos that already have a valid result file |

### What the output looks like

Per-video JSON (`duration_expansion_ratio.chatterbox.json`):
```json
{
  "aggregates": {
    "duration_expansion_ratio": { "mean": 5.63, "std": 4.60, "min": 1.21, "max": 18.73, "n": 12 }
  },
  "per_segment": [
    { "segment_id": 1, "synth_duration_seconds": 57.04, "original_duration_seconds": 31.03, "ratio": 1.838 },
    { "segment_id": 2, "synth_duration_seconds": 94.12, "original_duration_seconds": 12.50, "ratio": 7.530 }
  ]
}
```

### Resources

< 1 second per video, < 30 seconds total for 50 videos. Zero CPU/GPU load — just JSON reads and division.

---

## Appendix: Whisper GPU Setup

By default the Whisper server may load a CPU-only torch. To move it to GPU:

**1. Reinstall torch with CUDA support in the Whisper venv**

```bash
/root/venvs/whisper/bin/pip uninstall torch torchvision torchaudio -y
/root/venvs/whisper/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**2. Verify CUDA is available**

```bash
/root/venvs/whisper/bin/python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

Expected: `CUDA: True`

**3. Start Whisper server**

```bash
cd /mnt/c/Users/NHADO10/Desktop/Code2/ReLecture
/root/venvs/whisper/bin/python servers/whisper/server.py --model large-v3 --port 5001
```

Expected first lines:
```
[Whisper] PyTorch device: cuda
[Whisper] GPU: NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU
[Whisper] VRAM: 24.5 GB
```

**4. Monitor GPU**

In a second terminal while WER eval is running:
```bash
nvidia-smi -l 1
```

GPU-Util should spike to 80–100% during transcription.
