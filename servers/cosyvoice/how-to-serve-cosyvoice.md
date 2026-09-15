# How to Serve CosyVoice Locally (WSL2, GPU)

This server runs CosyVoice2 as a standalone Flask app in its own
Python virtual environment. On this machine it runs inside the
**WSL2 `Ubuntu-22.04` distro** (Python 3.10, NVIDIA GPU passthrough).
Serving natively on Windows is not supported — always run inside WSL.

GPU: NVIDIA RTX PRO 5000 Blackwell (24 GB). Blackwell (sm_120)
requires **PyTorch built for CUDA 12.8+** — the older cu121 wheels
from CosyVoice's own requirements DO NOT work on this GPU.

## 0. Install the WSL distro (one-time, from PowerShell)

```powershell
wsl --install -d Ubuntu-22.04 --no-launch
```

All following steps run inside WSL. From PowerShell prefix them with
`wsl -d Ubuntu-22.04 -u root -e bash -c "..."`, or open a shell with
`wsl -d Ubuntu-22.04`.

## 1. System packages

```bash
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3-venv python3-pip ffmpeg git git-lfs wget build-essential sox libsox-dev
```

## 2. Create the venv

Keep the venv on the Linux filesystem (fast I/O), NOT under /mnt/c:

```bash
mkdir -p /root/venvs
python3 -m venv /root/venvs/cosyvoice
/root/venvs/cosyvoice/bin/pip install --upgrade pip
```

## 3. Install PyTorch (Blackwell: CUDA 12.8)

```bash
/root/venvs/cosyvoice/bin/pip install torch torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

Do NOT let anything downgrade torch afterwards (see step 5 note).

## 4. Install CosyVoice from source

CosyVoice has no setup.py — it is used via PYTHONPATH (together with
its Matcha-TTS submodule), not `pip install -e .`:

```bash
cd /root
git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive   # third_party/Matcha-TTS
```

## 5. Install CosyVoice Python deps

CosyVoice's requirements.txt pins `torch==2.3.1` / cu121 / deepspeed.
**Skip those pins** (they break Blackwell) and filter them out:

```bash
grep -vE '^(torch|torchaudio|deepspeed)|extra-index-url' \
  /root/CosyVoice/requirements.txt > /root/cv_reqs.txt
```

`openai-whisper==20231117` fails to build with modern setuptools
(`pkg_resources` was removed). Workaround:

```bash
/root/venvs/cosyvoice/bin/pip install 'setuptools<70' wheel
/root/venvs/cosyvoice/bin/pip install --no-build-isolation openai-whisper==20231117
```

WARNING: openai-whisper drags in `torch 2.3.1` as a dependency — you
MUST reinstall the cu128 torch after it:

```bash
/root/venvs/cosyvoice/bin/pip install --force-reinstall torch torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

Use `--force-reinstall` WITHOUT `--no-deps` so the matching
`nvidia-cudnn-cu12` etc. are restored; otherwise you get
`ImportError: libcudnn.so.9: cannot open shared object file`. The
`triton<3` version warning from openai-whisper is harmless.

Then install the rest and the server dep:

```bash
/root/venvs/cosyvoice/bin/pip install -r /root/cv_reqs.txt
/root/venvs/cosyvoice/bin/pip install flask
```

torchaudio 2.11 routes ALL `load`/`save` through TorchCodec, so:

```bash
# Must come from the cu128 index — the PyPI build (0.16.0) needs CUDA 13
# and fails with "libnvrtc.so.13: cannot open shared object file".
/root/venvs/cosyvoice/bin/pip install torchcodec \
  --index-url https://download.pytorch.org/whl/cu128
# torchcodec also needs NVIDIA NPP (libnppicc.so.12) at runtime:
/root/venvs/cosyvoice/bin/pip install nvidia-npp-cu12
```

The NPP libs are NOT on the default loader path. Either register them
system-wide (recommended, one-time, as root):

```bash
echo '/root/venvs/cosyvoice/lib/python3.10/site-packages/nvidia/npp/lib' > /etc/ld.so.conf.d/cosyvoice-npp.conf
echo '/root/venvs/cosyvoice/lib/python3.10/site-packages/torch/lib' >> /etc/ld.so.conf.d/cosyvoice-npp.conf
ldconfig
```

…or start the server with `LD_LIBRARY_PATH` set (see step 8). Without
one of these you get
`OSError: libnppicc.so.12: cannot open shared object file` on the first
`/synthesize` call.

## 6. Verify import

```bash
export PYTHONPATH=/root/CosyVoice:/root/CosyVoice/third_party/Matcha-TTS
/root/venvs/cosyvoice/bin/python -c \
  "from cosyvoice.cli.cosyvoice import CosyVoice2; print('cosyvoice OK')"
```

## 7. Download the model checkpoint

```bash
/root/venvs/cosyvoice/bin/huggingface-cli download \
  FunAudioLLM/CosyVoice2-0.5B --local-dir /root/models/CosyVoice2-0.5B
```

## 8. Start the server

```bash
# LD_LIBRARY_PATH only needed if you skipped the ldconfig step above:
export LD_LIBRARY_PATH=/root/venvs/cosyvoice/lib/python3.10/site-packages/nvidia/npp/lib:/root/venvs/cosyvoice/lib/python3.10/site-packages/torch/lib
export PYTHONPATH=/root/CosyVoice:/root/CosyVoice/third_party/Matcha-TTS
export COSYVOICE_MODEL=/root/models/CosyVoice2-0.5B
cd /path/to/relecture
/root/venvs/cosyvoice/bin/python servers/cosyvoice/server.py
```

IMPORTANT (WSL quirk): processes backgrounded with `nohup ... &` from a
one-shot `wsl -e bash -c` call are KILLED when the WSL session ends.
Keep the server in the foreground of a long-lived shell (e.g. a
detached `wsl -d Ubuntu-22.04 -u root -e bash -c "... exec python
servers/cosyvoice/server.py"` from PowerShell).

The server listens on `0.0.0.0:5003` (reachable from Windows at
`http://localhost:5003`) and exposes:

- `GET  /health`
- `POST /synthesize`

Note: `server.py` writes the output WAV with `soundfile` (not
`torchaudio.save`) because TorchCodec cannot infer the format from a
`BytesIO` buffer.

## 9. Smoke test

From PowerShell on Windows:

```powershell
curl http://localhost:5003/health
```

Expected:

```json
{"status": "ok", "model": "cosyvoice:/root/models/CosyVoice2-0.5B"}
```

To exercise `/synthesize` directly you can send a small prompt clip
as base64. The ReLecture backend does this for you in normal use.

IMPORTANT (zero-shot quality): `prompt_text` must be the TRANSCRIPT of
the prompt audio. With an empty `prompt_text` the cloned voice timbre
still matches, but the generated speech is gibberish. There is a
unittest in `tests/servers/test_cosyvoice_server.py`
(`CosyVoiceLiveServerTests`) that synthesizes a real clip via the
running server and saves the result to `synth_test.wav`.
