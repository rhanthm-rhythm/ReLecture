"""CosyVoice zero-shot TTS server — runs in its own venv.

Install: pip install -r servers/cosyvoice/requirements.txt
         (plus CosyVoice from source: https://github.com/FunAudioLLM/CosyVoice)
Start:   python servers/cosyvoice/server.py

Exposes:
  POST /synthesize   JSON: {tts_text, prompt_audio_b64, prompt_text, speed}
  GET  /health
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
import threading

from flask import Flask, jsonify, request

app = Flask(__name__)
_model = None
_lock = threading.Lock()
COSYVOICE_MODEL = os.getenv("COSYVOICE_MODEL", "CosyVoice2-0.5B")


def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from cosyvoice.cli.cosyvoice import CosyVoice2
                _model = CosyVoice2(COSYVOICE_MODEL, load_jit=False, load_trt=False)
    return _model


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": f"cosyvoice:{COSYVOICE_MODEL}"})


@app.post("/synthesize")
def synthesize():
    import soundfile as sf
    data = request.get_json(force=True)
    tts_text = data.get("tts_text", "")
    prompt_text = data.get("prompt_text", "")
    audio_b64 = data.get("prompt_audio_b64", "")
    speed = float(data.get("speed", 1.0))

    audio_bytes = base64.b64decode(audio_b64) if audio_b64 else None

    model = get_model()
    with _lock:
        ref_path = None
        if audio_bytes:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                ref_path = tmp.name
        try:
            results = list(model.inference_zero_shot(tts_text, prompt_text, ref_path, speed=speed))
        finally:
            if ref_path and os.path.exists(ref_path):
                os.unlink(ref_path)

    if not results:
        return jsonify({"error": "no output"}), 500

    wav = results[0]["tts_speech"]
    buf = io.BytesIO()
    # torchaudio.save routes through TorchCodec, which cannot infer the
    # format from a BytesIO — write with soundfile instead.
    sf.write(buf, wav.cpu().numpy().T, model.sample_rate, format="WAV")
    buf.seek(0)
    out_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return jsonify({"audio_b64": out_b64})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)
