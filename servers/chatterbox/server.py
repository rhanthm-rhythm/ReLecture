"""Chatterbox TTS server — runs in its own venv.

Install: pip install -r servers/chatterbox/requirements.txt
Start:   python servers/chatterbox/server.py

Exposes:
  POST /synthesize   multipart: text, audio_prompt, exaggeration, cfg_weight, temperature
  GET  /health
"""
from __future__ import annotations

import io
import os
import tempfile
import threading

from flask import Flask, jsonify, request, send_file

app = Flask(__name__)
_model = None
_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                import torch
                from chatterbox.tts import ChatterboxTTS
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _model = ChatterboxTTS.from_pretrained(device=device)
    return _model


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": "chatterbox"})


@app.post("/synthesize")
def synthesize():
    text = request.form.get("text", "")
    exaggeration = float(request.form.get("exaggeration", 0.5))
    cfg_weight = float(request.form.get("cfg_weight", 0.5))
    temperature = float(request.form.get("temperature", 0.8))
    audio_prompt = request.files.get("audio_prompt")

    model = get_model()
    with _lock:
        ref_path = None
        if audio_prompt:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_prompt.read())
                ref_path = tmp.name
        try:
            wav = model.generate(
                text,
                audio_prompt_path=ref_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )
        finally:
            if ref_path and os.path.exists(ref_path):
                os.unlink(ref_path)

    import soundfile as sf
    buf = io.BytesIO()
    # torchaudio.save routes through TorchCodec, which cannot infer the
    # format from a BytesIO — write with soundfile instead.
    sf.write(buf, wav.squeeze(0).cpu().numpy(), model.sr, format="WAV")
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
