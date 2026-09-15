# How to Serve Qwen3-TTS Locally (WSL2, GPU)

This server wraps **Qwen3-TTS-12Hz-1.7B-Base** (voice cloning) via the
`qwen-tts` Python package as a standalone Flask app in its own venv,
inside **WSL2 Ubuntu-22.04** (see the CosyVoice guide for WSL setup).
Serving natively on Windows is not supported.

GPU note: RTX PRO 5000 Blackwell requires **PyTorch cu128+**.

## 1. Create the venv (inside WSL)

```bash
python3 -m venv /root/venvs/qwen3
/root/venvs/qwen3/bin/pip install --upgrade pip
```

## 2. Install dependencies

Install torch/torchaudio STRICTLY from the cu128 index — do NOT mix in
`--extra-index-url https://pypi.org/simple`, otherwise pip resolves
torch from PyPI (CUDA 13) and torchaudio then fails with
"PyTorch and TorchAudio were compiled with different CUDA versions":

```bash
/root/venvs/qwen3/bin/pip install torch torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
/root/venvs/qwen3/bin/pip install flask qwen-tts soundfile
/root/venvs/qwen3/bin/pip install nvidia-cuda-runtime-cu12   # torchaudio needs libcudart.so.12
```

(`qwen-tts` 0.1.1 works fine on Python 3.10 despite the README
suggesting 3.12. flash-attn is optional — the server uses sdpa.)

## 3. Download the model (~4 GB)

```bash
/root/venvs/qwen3/bin/huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

Or let the server download it on first request. Override with
`QWEN3_MODEL` (e.g. the 0.6B variant).

## 4. Start the server

```bash
cd /path/to/relecture
/root/venvs/qwen3/bin/python servers/qwen3/server.py
```

WSL quirk: keep it in the foreground of a long-lived shell —
`nohup ... &` from a one-shot `wsl -e` call gets killed.

The server listens on 0.0.0.0:5000 (reachable from Windows at
http://localhost:5000) and exposes:

- GET  /health
- POST /generate  (multipart: text, reference_audio, ref_text, style,
  language, speed_ref, temperature)

Notes on the implementation (servers/qwen3/server.py):

- Wraps `Qwen3TTSModel.generate_voice_clone` behind a
  `synthesize(...) -> wav bytes` adapter; loads with
  `device_map="cuda:0"`, `dtype=bfloat16`, `attn_implementation="sdpa"`.
- `language` accepts short codes (`en`, `zh`, ...) which are mapped to
  Qwen3-TTS names (`English`, `Chinese`, ...).
- If `ref_text` (transcript of the reference audio) is missing, it
  falls back to `x_vector_only_mode=True` (speaker embedding only,
  lower quality — same gibberish caveat as CosyVoice).
- `style` is currently ignored (Base model has no instruct support;
  use the CustomVoice/VoiceDesign models for that).

## 5. Smoke test

```powershell
curl http://localhost:5000/health
```

```json
{"status": "ok", "model": "qwen3:Qwen/Qwen3-TTS-12Hz-1.7B-Base"}
```

Live tests (unit tests + real GPU voice clone of voice.wav, output
saved to `qwen3_test.wav`):

```powershell
.venv-cosyvoice\Scripts\python.exe -m unittest tests.servers.test_qwen3_server -v
```

The live tests skip automatically if the server is not running.
