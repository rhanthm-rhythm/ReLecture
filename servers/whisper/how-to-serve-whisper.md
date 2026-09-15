# How to Serve Whisper Locally (WSL2, GPU)

This server runs OpenAI Whisper as a standalone Flask app in its own
Python virtual environment. On this machine it runs inside the
**WSL2 `Ubuntu-22.04` distro** (see the CosyVoice guide for the WSL
setup). Serving natively on Windows is not supported.

GPU note: RTX PRO 5000 Blackwell requires **PyTorch cu128+** — the
cu121 wheels do NOT work on this GPU.

## 1. Create the venv (inside WSL)

```bash
python3 -m venv /root/venvs/whisper
/root/venvs/whisper/bin/pip install --upgrade pip
```

## 2. Install PyTorch and Whisper

```bash
/root/venvs/whisper/bin/pip install 'setuptools<70' wheel flask
/root/venvs/whisper/bin/pip install torch \
  --index-url https://download.pytorch.org/whl/cu128
# --no-build-isolation avoids the 'pkg_resources' build failure with
# modern setuptools:
/root/venvs/whisper/bin/pip install --no-build-isolation openai-whisper
```

Use a recent openai-whisper (20250625+): it no longer drags in an old
torch. (20231117 forces torch 2.3.1, which breaks Blackwell — see the
CosyVoice guide.)

`server.py` monkeypatches `torch.load(..., weights_only=False)` around
`whisper.load_model`, because torch >= 2.6 defaults to
`weights_only=True`, which breaks whisper's checkpoint loading.

## 3. Start the server

From WSL. The model downloads to /root/.cache/whisper on first use
(~3 GB for large-v3 — the first /transcribe call can take several
minutes):

```bash
cd /path/to/relecture
/root/venvs/whisper/bin/python servers/whisper/server.py --model large-v3 --port 5001
```

WSL quirk: keep the server in the foreground of a long-lived shell —
`nohup ... &` from a one-shot `wsl -e` call gets killed.

The server listens on 0.0.0.0:5001 (reachable from Windows at
http://localhost:5001) and exposes:

- GET  /health
- POST /transcribe   (multipart form, field name: file)

## 4. Smoke test

```powershell
curl http://localhost:5001/health
```

```json
{"status": "ok", "model": "whisper:large-v3"}
```

Live tests (unit tests + real GPU transcription of voice.wav):

```powershell
.venv-cosyvoice\Scripts\python.exe -m unittest tests.servers.test_whisper_server -v
```

The live tests skip automatically if the server is not running.
