"""Whisper transcription server — runs in its own venv.

Install: pip install -r servers/whisper/requirements.txt
Start:   python servers/whisper/server.py [--model large-v3] [--port 5001]

Exposes:
  POST /transcribe   multipart: file (audio)
  GET  /health
"""
from __future__ import annotations

import argparse
import os
import tempfile
import threading

from flask import Flask, jsonify, request
import torch

app = Flask(__name__)
_model = None
_lock = threading.Lock()
_model_name = os.getenv("WHISPER_MODEL", "large-v3")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Whisper] PyTorch device: {DEVICE}", flush=True)
if DEVICE == "cuda":
    print(f"[Whisper] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[Whisper] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)


def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                # torch>=2.6 defaults weights_only=True, which breaks
                # whisper's checkpoint loading — force the old behavior.
                _orig_load = torch.load
                torch.load = lambda *a, **k: _orig_load(
                    *a, **{**k, "weights_only": False}
                )
                import whisper
                _model = whisper.load_model(_model_name, device=DEVICE)
                torch.load = _orig_load
    return _model


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": f"whisper:{_model_name}"})


@app.post("/transcribe")
def transcribe():
    audio_file = request.files.get("file")
    if not audio_file:
        return jsonify({"error": "No file provided"}), 400

    language = request.form.get("language") or os.getenv("LANGUAGE", "en")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        model = get_model()
        with _lock:
            result = model.transcribe(tmp_path, language=language)
        return jsonify({"transcription": result["text"].strip()})
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=_model_name)
    parser.add_argument("--port", type=int, default=5001)
    parsed = parser.parse_args()
    _model_name = parsed.model
    app.run(host="0.0.0.0", port=parsed.port)
