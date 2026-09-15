# How to Serve Chatterbox Locally (WSL2, GPU)

This server wraps chatterbox-tts in a standalone Flask app in its own
venv, inside **WSL2 Ubuntu-22.04** (see the CosyVoice guide for the WSL
setup). Serving natively on Windows is not supported.

GPU note: RTX PRO 5000 Blackwell requires **PyTorch cu128+** (sm_120).

## 1. Create the venv (inside WSL)

```bash
python3 -m venv /root/venvs/chatterbox
/root/venvs/chatterbox/bin/pip install --upgrade pip
```

## 2. Install dependencies

```bash
/root/venvs/chatterbox/bin/pip install torch torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
/root/venvs/chatterbox/bin/pip install flask soundfile chatterbox-tts
```

CRITICAL: `chatterbox-tts` pins `torch==2.6.0` and pip will happily
DOWNGRADE the cu128 torch — that build has no sm_120 kernels and fails
at model load with `RuntimeError: CUDA error: no kernel image is
available for execution on the device`. After installing
chatterbox-tts you MUST restore the cu128 build:

```bash
/root/venvs/chatterbox/bin/pip install --force-reinstall torch torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

(then restart the server — a running process keeps the old torch in
memory.)

`server.py` writes output WAV with `soundfile` (imported lazily inside
`synthesize`) because `torchaudio.save` routes through TorchCodec,
which cannot infer the format from a `BytesIO` buffer.

## 3. Start the server

```bash
cd /path/to/relecture
/root/venvs/chatterbox/bin/python servers/chatterbox/server.py
```

WSL quirk: keep it in the foreground of a long-lived shell —
`nohup ... &` from a one-shot `wsl -e` call gets killed.

The model auto-downloads on first /synthesize. The server listens on
0.0.0.0:5002 (reachable from Windows at http://localhost:5002) and
exposes:

- GET  /health
- POST /synthesize  (multipart: text, audio_prompt [optional clone
  reference], exaggeration, cfg_weight, temperature)

## 4. Smoke test

```powershell
curl http://localhost:5002/health
```

```json
{"status": "ok", "model": "chatterbox"}
```

Live tests (unit tests + real GPU voice clone of voice.wav, output
saved to `chatterbox_test.wav`):

```powershell
.venv-cosyvoice\Scripts\python.exe -m unittest tests.servers.test_chatterbox_server -v
```

The live tests skip automatically if the server is not running.
